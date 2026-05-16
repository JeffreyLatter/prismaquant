from __future__ import annotations

import torch
import torch.nn as nn


def test_duquant_format_traits_follow_microscale_registry():
    from prismaquant.duquant import duquant_format_traits, supports_duquant_fold

    assert duquant_format_traits("NVFP4").group_size == 16
    assert duquant_format_traits("MXFP4").group_size == 32
    assert duquant_format_traits("MXFP8_E4M3").group_size == 32
    assert duquant_format_traits("MXFP8_E5M2").group_size == 32
    assert duquant_format_traits("MXFP6_E3M2").group_size == 32
    assert supports_duquant_fold("FP8_E4M3") is False
    assert supports_duquant_fold("BF16") is False


def test_duquant_block_scale_increases_outlier_activation_channel():
    from prismaquant.duquant import duquant_block_scale_from_stats

    weight = torch.ones(8, 16) * 0.1
    activations = torch.ones(32, 16) * 0.01
    activations[:, 3] = 10.0

    scale = duquant_block_scale_from_stats(
        weight,
        activations,
        group_size=16,
        alpha=0.5,
        clamp_ratio=10.0,
    )

    assert scale.shape == (16,)
    assert scale[3] > scale.median()
    assert torch.isfinite(scale).all()


def test_fold_scaled_nvfp4_scores_final_packed_weight():
    from prismaquant.production_weight_cache import _render_fold_scaled_for_cache
    from prismaquant.render_score import score_render_error

    torch.manual_seed(7)
    weight = torch.randn(16, 16, dtype=torch.float32) * 0.13
    activations = torch.randn(64, 16, dtype=torch.float32)

    rendered = _render_fold_scaled_for_cache(
        qname="model.layers.0.self_attn.q_proj",
        fmt="NVFP4",
        weight_scaled=weight,
        activations_scaled=activations,
        levers={"gptq": False, "scale_sweep": False},
        joint_global_real=None,
    )

    assert rendered.shape == weight.shape
    assert not torch.allclose(rendered, weight)
    assert score_render_error(weight, rendered, activations) > 0.0


def test_production_cache_records_duquant_fold_scales_for_mxfp4(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    class TinyDuQuantModel(nn.Module):
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

    model = TinyDuQuantModel()
    qname = "model.layers.0.self_attn.q_proj"
    base_weight = model.model.layers[0].self_attn.q_proj.weight.detach().to(torch.float32)
    forced_scale = torch.linspace(0.5, 1.5, 48)

    def fake_duquant_scale_from_targets(_targets, **_kwargs):
        return forced_scale

    def fake_render_fold_scaled_for_cache(*, weight_scaled, **_kwargs):
        expected = base_weight.to(weight_scaled.device) * forced_scale.to(
            weight_scaled.device
        ).unsqueeze(0)
        if torch.allclose(weight_scaled.to(torch.float32), expected, atol=1e-6):
            return weight_scaled.to(torch.float32)
        return weight_scaled.to(torch.float32) + 2.0

    monkeypatch.setattr(pwc, "duquant_scale_from_targets", fake_duquant_scale_from_targets)
    monkeypatch.setattr(pwc, "_render_fold_scaled_for_cache", fake_render_fold_scaled_for_cache)

    cache = pwc.fill_production_weight_cache(
        model,
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        qnames=[qname],
        formats=["MXFP4"],
        render_assignment={qname: "MXFP4"},
        levers={"duquant": True, "gptq": False, "scale_sweep": False},
        max_act_rows=16,
        progress=False,
    )

    assert cache.levers["duquant"] is True
    assert cache.awq_scales is not None
    assert torch.allclose(cache.awq_scales[qname], forced_scale)
    assert cache.metadata["duquant"]["status"] == "applied"
    assert cache.metadata["duquant"]["n_scaled_linears"] == 1
    assert cache.metadata["duquant"]["runtime_support_required"] == "none"
    assert cache.metadata["duquant"]["paper_faithful_duquantpp"] is False
