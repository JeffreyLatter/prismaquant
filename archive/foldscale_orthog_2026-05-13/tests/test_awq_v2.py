from __future__ import annotations

import torch
import torch.nn as nn


def test_activation_collector_respects_resident_device():
    from prismaquant.production_weight_cache import _LinearActivationCollector

    class TinyCollectorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 32, bias=False)

        def forward(self, x):
            return self.proj(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCollectorModel().to(device)
    collector = _LinearActivationCollector(
        model,
        qnames={"proj"},
        max_rows=3,
        store_qnames={"proj"},
        store_device=device,
    )
    collector.install()
    try:
        with torch.no_grad():
            model(torch.randn(2, 32, device=device))
    finally:
        collector.remove()

    activations = collector.collected()
    assert activations["proj"].device.type == device.type
    assert activations["proj"].shape == (2, 32)
    assert collector.max_abs["proj"] > 0.0


def test_awq_search_uses_rendered_output_error():
    from prismaquant.awq import AwqSearchTarget, search_awq_scale
    from prismaquant.export_native_compressed import _rtn_dequant_nvfp4

    torch.manual_seed(123)
    W = torch.zeros(16, 16)
    W[:, 0] = 0.075
    W[:, 1] = 1.0
    W[:, 2:] = torch.randn(16, 14) * 0.01

    X = torch.randn(512, 16) * 0.01
    X[:, 0] *= 300.0
    X[:, 1] *= 0.01

    target = AwqSearchTarget(
        name="demo.q_proj",
        fmt="NVFP4",
        weight=W,
        activations=X,
        group_size=16,
    )

    def render_scaled(_idx, weight_scaled, _activations_scaled, _scale):
        return _rtn_dequant_nvfp4(weight_scaled, group_size=16)

    result = search_awq_scale(
        [target],
        render_scaled,
        n_grid=20,
        clamp_ratio=10.0,
    )

    assert result.best_score <= result.baseline_score
    assert result.selected_label != "identity"
    assert result.relative_gain > 0.0
    assert result.scale.shape == (16,)


def test_production_cache_records_awq_scales(monkeypatch):
    import prismaquant.production_weight_cache as pwc
    from prismaquant.awq import AwqSearchResult

    class TinyAwqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4, 48)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            layer = self.model.layers[0]
            layer.input_layernorm = nn.RMSNorm(48)
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(48, 48, bias=False)

        def forward(self, input_ids, use_cache=False):
            x = self.embed(input_ids)
            layer = self.model.layers[0]
            h = layer.input_layernorm(x)
            return layer.self_attn.q_proj(h)

    forced_scale = torch.linspace(0.5, 1.5, 48)
    seen_kwargs = {}
    render_levers = []

    def fake_search(targets, render_scaled, **kwargs):
        seen_kwargs.update(kwargs)
        render_scaled(
            0,
            targets[0].weight,
            targets[0].activations,
            torch.ones_like(forced_scale),
        )
        return AwqSearchResult(
            scale=forced_scale,
            selected_label="duo_ratio_0.5000",
            selected_ratio=0.5,
            baseline_score=10.0,
            best_score=5.0,
            relative_gain=0.5,
            n_candidates=2,
            trace=[],
        )

    def fake_render_awq_scaled_for_cache(
        *,
        weight_scaled,
        levers,
        **_kwargs,
    ):
        render_levers.append(dict(levers))
        return weight_scaled

    monkeypatch.setattr(pwc, "search_awq_scale", fake_search)
    monkeypatch.setattr(
        pwc,
        "_render_awq_scaled_for_cache",
        fake_render_awq_scaled_for_cache,
    )

    qname = "model.layers.0.self_attn.q_proj"
    cache = pwc.fill_production_weight_cache(
        TinyAwqModel(),
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        qnames=[qname],
        formats=["NVFP4"],
        render_assignment={qname: "NVFP4"},
        levers={"awq": True, "gptq": True, "scale_sweep": True},
        max_act_rows=16,
        progress=False,
    )

    assert cache.levers["awq"] is True
    assert cache.awq_scales is not None
    assert torch.allclose(cache.awq_scales[qname], forced_scale)
    assert cache.metadata["awq"]["status"] == "applied"
    assert cache.metadata["awq"]["n_scaled_linears"] == 1
    assert cache.metadata["awq"]["min_gain"] == 0.03
    assert cache.metadata["awq"]["groups"]["model.layers.0.input_layernorm"][
        "gate_reason"
    ] == "improved"
    assert cache.metadata["awq"]["search_levers"]["gptq"] is False
    assert seen_kwargs["min_gain"] == 0.03
    assert render_levers[0]["gptq"] is False
    assert cache.metadata["render_gates"]["entries"] == 1
    mechanisms = cache.metadata["render_gates"]["mechanisms"]
    assert "gptq" in mechanisms
    assert "scale_sweep" in mechanisms


def test_precomputed_awq_fold_preserves_mixed_reader_outputs():
    from prismaquant.export_native_compressed import (
        _awq_fold_layer_precomputed_scales,
    )

    class IdentityProfile:
        def live_to_recipe_name(self, name):
            return name

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = nn.RMSNorm(16)
            self.gate_proj = nn.Linear(16, 16, bias=False)
            self.gate = nn.Linear(16, 4, bias=False)

        def forward(self, x):
            h = self.post_attention_layernorm(x)
            return self.gate_proj(h), self.gate(h)

    torch.manual_seed(7)
    layer = Layer().eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        ref_a, ref_b = layer(x)
        bf16_before = layer.gate.weight.detach().clone()

    scale = torch.linspace(0.5, 1.5, 16)
    returned = _awq_fold_layer_precomputed_scales(
        layer,
        "",
        {"gate_proj": "NVFP4", "gate": "BF16"},
        IdentityProfile(),
        {"gate_proj": scale},
        torch.device("cpu"),
    )

    assert "gate_proj" in returned
    assert not torch.equal(layer.gate.weight.detach(), bf16_before)
    with torch.no_grad():
        after_a, after_b = layer(x)
    assert torch.allclose(after_a, ref_a, rtol=2e-3, atol=2e-3)
    assert torch.allclose(after_b, ref_b, rtol=2e-3, atol=2e-3)


def test_fold_scale_methods_are_mutually_exclusive():
    import pytest
    import prismaquant.production_weight_cache as pwc

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)

        def forward(self, input_ids, use_cache=False):
            x = torch.nn.functional.one_hot(input_ids % 8, num_classes=8)
            return self.proj(x.to(self.proj.weight.dtype))

    with pytest.raises(ValueError, match="mutually exclusive"):
        pwc.fill_production_weight_cache(
            TinyModel(),
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            qnames=["proj"],
            formats=["NVFP4"],
            levers={"awq": True, "smoothquant": True},
            progress=False,
        )


def test_smoothquant_alpha_limit_defaults_to_requested_hi(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    monkeypatch.delenv("PRISMAQUANT_SMOOTHQUANT_NVFP4_ALPHA_MAX", raising=False)
    monkeypatch.delenv("PRISMAQUANT_SMOOTHQUANT_MXFP8_ALPHA_MAX", raising=False)
    monkeypatch.delenv("PRISMAQUANT_SMOOTHQUANT_FP8_E4M3_ALPHA_MAX", raising=False)
    monkeypatch.delenv(
        "PRISMAQUANT_SMOOTHQUANT_PROMOTED_ALPHA_MAX",
        raising=False,
    )

    assert pwc._smoothquant_alpha_hi_for_formats(
        ["NVFP4"],
        requested_hi=1.0,
    ) == 1.0
    assert pwc._smoothquant_alpha_hi_for_formats(
        ["MXFP8_E4M3"],
        requested_hi=1.0,
    ) == 1.0
    assert pwc._smoothquant_alpha_hi_for_formats(
        ["FP8_E4M3"],
        requested_hi=1.0,
    ) == 1.0
    assert pwc._smoothquant_alpha_hi_for_formats(
        ["NVFP4", "MXFP8_E4M3"],
        requested_hi=1.0,
    ) == 1.0
    assert pwc._smoothquant_alpha_hi_for_formats(
        ["NVFP4"],
        requested_hi=0.7,
    ) == 0.7

    monkeypatch.setenv("PRISMAQUANT_SMOOTHQUANT_NVFP4_ALPHA_MAX", "0.375")
    assert pwc._smoothquant_alpha_hi_for_formats(
        ["NVFP4"],
        requested_hi=1.0,
    ) == 0.375
    monkeypatch.setenv("PRISMAQUANT_SMOOTHQUANT_MXFP8_ALPHA_MAX", "0.625")
    assert pwc._smoothquant_alpha_hi_for_formats(
        ["MXFP8_E4M3"],
        requested_hi=1.0,
    ) == 0.625
    monkeypatch.setenv("PRISMAQUANT_SMOOTHQUANT_PROMOTED_ALPHA_MAX", "0.75")
    monkeypatch.delenv("PRISMAQUANT_SMOOTHQUANT_MXFP8_ALPHA_MAX", raising=False)
    assert pwc._smoothquant_alpha_hi_for_formats(
        ["MXFP8_E4M3"],
        requested_hi=1.0,
    ) == 0.75


def test_smoothquant_solver_uses_requested_alpha_hi_by_default(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    qname = "model.layers.0.self_attn.q_proj"
    mod = nn.Linear(16, 16, bias=False)
    acts = torch.randn(8, 16)
    seen_hi: list[float] = []

    def fake_render_awq_scaled_for_cache(*, weight_scaled, **_kwargs):
        return weight_scaled

    def fake_golden_section_search(f, a, b, *, tol, max_iter):
        seen_hi.append(float(b))
        x = 0.5 * (float(a) + float(b))
        return x, float(f(x))

    monkeypatch.setattr(
        pwc,
        "_render_awq_scaled_for_cache",
        fake_render_awq_scaled_for_cache,
    )
    monkeypatch.setattr(
        pwc,
        "_golden_section_search",
        fake_golden_section_search,
    )
    monkeypatch.delenv("PRISMAQUANT_SMOOTHQUANT_NVFP4_ALPHA_MAX", raising=False)
    monkeypatch.delenv("PRISMAQUANT_SMOOTHQUANT_MXFP8_ALPHA_MAX", raising=False)
    monkeypatch.delenv("PRISMAQUANT_SMOOTHQUANT_FP8_E4M3_ALPHA_MAX", raising=False)
    monkeypatch.delenv(
        "PRISMAQUANT_SMOOTHQUANT_PROMOTED_ALPHA_MAX",
        raising=False,
    )

    _scales, meta = pwc._solve_smoothquant_scales(
        qname_to_module={qname: mod},
        activations={qname: acts},
        render_formats_by_qname={qname: ("NVFP4",)},
        levers={"smoothquant": True},
        progress=False,
    )

    assert seen_hi == [1.0]
    group = meta["groups"]["model.layers.0.input_layernorm"]
    assert group["effective_alpha_hi"] == 1.0
    assert group["format_alpha_caps"]["NVFP4"] == 1.0

    seen_hi.clear()
    _scales, meta = pwc._solve_smoothquant_scales(
        qname_to_module={qname: mod},
        activations={qname: acts},
        render_formats_by_qname={qname: ("MXFP8_E4M3",)},
        levers={"smoothquant": True},
        progress=False,
    )

    assert seen_hi == [1.0]
    group = meta["groups"]["model.layers.0.input_layernorm"]
    assert group["effective_alpha_hi"] == 1.0
    assert group["format_alpha_caps"]["MXFP8_E4M3"] == 1.0

    seen_hi.clear()
    _scales, meta = pwc._solve_smoothquant_scales(
        qname_to_module={qname: mod},
        activations={qname: acts},
        render_formats_by_qname={qname: ("FP8_E4M3",)},
        levers={"smoothquant": True},
        progress=False,
    )

    assert seen_hi == [1.0]
    group = meta["groups"]["model.layers.0.input_layernorm"]
    assert group["effective_alpha_hi"] == 1.0
    assert group["format_alpha_caps"]["FP8_E4M3"] == 1.0


def test_smoothquant_solver_honors_explicit_alpha_cap(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    qname = "model.layers.0.self_attn.q_proj"
    mod = nn.Linear(16, 16, bias=False)
    acts = torch.randn(8, 16)
    seen_hi: list[float] = []

    def fake_render_awq_scaled_for_cache(*, weight_scaled, **_kwargs):
        return weight_scaled

    def fake_golden_section_search(f, a, b, *, tol, max_iter):
        seen_hi.append(float(b))
        x = 0.5 * (float(a) + float(b))
        return x, float(f(x))

    monkeypatch.setattr(
        pwc,
        "_render_awq_scaled_for_cache",
        fake_render_awq_scaled_for_cache,
    )
    monkeypatch.setattr(
        pwc,
        "_golden_section_search",
        fake_golden_section_search,
    )
    monkeypatch.setenv("PRISMAQUANT_SMOOTHQUANT_FP8_E4M3_ALPHA_MAX", "0.0")

    _scales, meta = pwc._solve_smoothquant_scales(
        qname_to_module={qname: mod},
        activations={qname: acts},
        render_formats_by_qname={qname: ("FP8_E4M3",)},
        levers={"smoothquant": True},
        progress=False,
    )

    assert seen_hi == []
    group = meta["groups"]["model.layers.0.input_layernorm"]
    assert group["effective_alpha_hi"] == 0.0
    assert group["format_alpha_caps"]["FP8_E4M3"] == 0.0
    assert group["selected"] == "identity"


def test_block_rotation_is_mutually_exclusive_with_awq():
    import pytest
    import prismaquant.production_weight_cache as pwc

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)

        def forward(self, input_ids, use_cache=False):
            x = torch.nn.functional.one_hot(input_ids % 8, num_classes=8)
            return self.proj(x.to(self.proj.weight.dtype))

    with pytest.raises(ValueError, match="mutually exclusive"):
        pwc.fill_production_weight_cache(
            TinyModel(),
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            qnames=["proj"],
            formats=["NVFP4"],
            levers={"awq": True, "block_rotation": True},
            progress=False,
        )
