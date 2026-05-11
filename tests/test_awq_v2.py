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
    assert cache.metadata["awq"]["search_levers"]["gptq"] is False
    assert seen_kwargs["min_gain"] == 0.03
    assert render_levers[0]["gptq"] is False
    assert render_levers[-1]["gptq"] is True


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


def test_awq_and_smoothquant_are_mutually_exclusive():
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
