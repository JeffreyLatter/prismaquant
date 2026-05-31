"""Regression for issue #6 — Gemma4 multi-layer-type rotary init.

Gemma4's text rotary registers one `<layer_type>_inv_freq` buffer per entry in
`config.layer_types`, with *mixed* rope types (sliding_attention=default,
full_attention=proportional). The generic single-rope fallback in
`_init_rotary_inplace` calls `compute_default_rope_parameters(cfg, device)` with
no `layer_type` → `KeyError: None` on `config.rope_parameters[layer_type]`.
`Gemma4Profile.init_rotaries` must re-init the rotary on the real device,
rebuilding every per-type buffer (using the rotary's own per-type rope init, so
the proportional layer is computed correctly).

Uses a fake rotary that mimics Gemma4TextRotaryEmbedding's shape, so the test
runs without the (newer) transformers Gemma4 modeling installed.
"""
import torch
import torch.nn as nn

from prismaquant.model_profiles.gemma4 import Gemma4Profile


class _Cfg:
    rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
        "full_attention": {"rope_type": "proportional", "rope_theta": 1e6},
    }
    layer_types = ["sliding_attention", "full_attention"]


class _MultiLayerRotary(nn.Module):
    """Mimics Gemma4TextRotaryEmbedding: config-only __init__, per-layer-type
    `<lt>_inv_freq` buffers. Indexing rope_parameters[lt] would KeyError if
    called with layer_type=None (the generic single-rope path)."""

    def __init__(self, config, device=None, layer_type=None):
        super().__init__()
        self.config = config
        self.layer_types = set(config.layer_types)
        for lt in self.layer_types:
            _ = config.rope_parameters[lt]["rope_theta"]  # per-type, needs lt
            self.register_buffer(f"{lt}_inv_freq",
                                 torch.ones(4, device=device), persistent=False)
            setattr(self, f"{lt}_attention_scaling", 1.0)


def test_gemma4_init_rotaries_reinits_multilayer_rotary():
    cfg = _Cfg()
    rot = _MultiLayerRotary(cfg, device=torch.device("meta"))
    assert rot.full_attention_inv_freq.device.type == "meta"

    ok = Gemma4Profile().init_rotaries(rot, cfg, torch.device("cpu"), torch.float32)
    assert ok is True
    # every layer-type buffer rebuilt on the real device
    assert rot.full_attention_inv_freq.device.type == "cpu"
    assert rot.sliding_attention_inv_freq.device.type == "cpu"


def test_gemma4_init_rotaries_defers_when_not_multilayer():
    """Returns False (→ generic path) for a plain single-rope rotary."""
    class _Plain(nn.Module):
        pass
    assert Gemma4Profile().init_rotaries(
        _Plain(), _Cfg(), torch.device("cpu"), torch.float32) is False
