"""Multi-layer-type rope in PrismaQuant's streaming forward.

Single-rope models (Qwen/MiniMax/LFM2.5) return one `(cos, sin)` and feed it to
every layer. Multi-layer-type-rope models (Gemma3/Gemma4) need a different rope
per attention type. `_compute_position_embeddings` returns a
`{layer_type: (cos, sin)}` dict for those, and `_call_layer` selects each
layer's entry by its attention `layer_type`. These tests pin both behaviors and
the single-rope non-regression — without any transformers modeling dependency.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
)


# --- fakes -----------------------------------------------------------------
class _SingleRotary(nn.Module):
    def forward(self, hidden, position_ids):
        return (torch.zeros(2), torch.ones(2))  # (cos, sin)


class _MultiRotary(nn.Module):
    """Per-type rope keyed by layer_type (mimics Gemma4TextRotaryEmbedding)."""
    layer_types = {"sliding_attention", "full_attention"}

    def forward(self, hidden, position_ids, layer_type=None):
        if layer_type is None:
            raise KeyError(None)  # the bug: generic call has no layer_type
        scale = 1.0 if layer_type == "sliding_attention" else 2.0
        return (torch.full((2,), scale), torch.full((2,), scale))


class _MultiRotaryNoLayerKwarg(nn.Module):
    """Multi-type model whose forward doesn't accept layer_type (DSv4-style):
    one rope used for all layers."""
    layer_types = {"a", "b"}

    def forward(self, hidden, position_ids):
        return (torch.full((2,), 7.0), torch.full((2,), 7.0))


class _Base(nn.Module):
    def __init__(self, rotary, config=None):
        super().__init__()
        self.rotary_emb = rotary
        self.config = config


class _Layer(nn.Module):
    """Records the position_embeddings it actually received."""
    def __init__(self, layer_type=None):
        super().__init__()
        if layer_type is not None:
            self.self_attn = nn.Module()
            self.self_attn.layer_type = layer_type
        self.received = None
        self.received_mask = None

    def forward(self, *, hidden_states, position_embeddings, **kw):
        self.received = position_embeddings
        self.received_mask = kw.get("attention_mask")
        return hidden_states


# --- _compute_position_embeddings ------------------------------------------
def test_single_rope_returns_tuple():
    pe = _compute_position_embeddings(_Base(_SingleRotary()), torch.zeros(1), torch.zeros(1))
    assert isinstance(pe, tuple) and len(pe) == 2


def test_multilayer_returns_per_type_dict():
    pe = _compute_position_embeddings(_Base(_MultiRotary()), torch.zeros(1), torch.zeros(1))
    assert isinstance(pe, dict)
    assert set(pe) == {"sliding_attention", "full_attention"}
    assert float(pe["sliding_attention"][0][0]) == 1.0
    assert float(pe["full_attention"][0][0]) == 2.0


def test_multilayer_without_layer_kwarg_falls_back():
    # DSv4-style: forward has no layer_type → same rope per type, no crash
    pe = _compute_position_embeddings(_Base(_MultiRotaryNoLayerKwarg()), torch.zeros(1), torch.zeros(1))
    assert isinstance(pe, dict)
    assert all(float(v[0][0]) == 7.0 for v in pe.values())


# --- _compute_attention_mask -----------------------------------------------
def test_multilayer_sliding_attention_builds_per_type_masks():
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["sliding_attention", "full_attention"]
    cfg.sliding_window = 2
    cfg._attn_implementation = "eager"
    base = _Base(_SingleRotary(), config=cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)

    masks = _compute_attention_mask(base, hidden, position_ids)

    assert set(masks) == {"sliding_attention", "full_attention"}
    assert masks["full_attention"].shape == (1, 1, 4, 4)
    assert masks["sliding_attention"].shape == (1, 1, 4, 4)
    assert float(masks["full_attention"][0, 0, 3, 0]) == 0.0
    assert float(masks["sliding_attention"][0, 0, 3, 0]) < -1e20


# --- _call_layer selection -------------------------------------------------
def test_call_layer_selects_by_layer_type():
    pe = {"sliding_attention": ("s", "s"), "full_attention": ("f", "f")}
    for lt, expect in (("sliding_attention", "s"), ("full_attention", "f")):
        layer = _Layer(layer_type=lt)
        _call_layer(layer, torch.zeros(1), position_embeddings=pe,
                    attention_mask=None, position_ids=None)
        assert layer.received[0] == expect


def test_call_layer_passes_tuple_unchanged_for_single_rope():
    pe = ("cos", "sin")
    layer = _Layer(layer_type=None)
    _call_layer(layer, torch.zeros(1), position_embeddings=pe,
                attention_mask=None, position_ids=None)
    assert layer.received is pe  # untouched for single-rope models


def test_call_layer_single_entry_dict_unknown_type_falls_back():
    pe = {"only": ("x", "x")}
    layer = _Layer(layer_type="missing")  # not in dict
    _call_layer(layer, torch.zeros(1), position_embeddings=pe,
                attention_mask=None, position_ids=None)
    assert layer.received == ("x", "x")  # falls back to an entry, no crash


def test_call_layer_rejects_unknown_multi_rope_type():
    pe = {"sliding_attention": ("s", "s"), "full_attention": ("f", "f")}
    layer = _Layer(layer_type="missing")
    try:
        _call_layer(layer, torch.zeros(1), position_embeddings=pe,
                    attention_mask=None, position_ids=None)
    except RuntimeError as exc:
        assert "per-layer position_embeddings" in str(exc)
    else:  # pragma: no cover - assert path keeps compatibility without pytest.raises
        raise AssertionError("unknown layer_type accepted for multi-rope dict")


def test_call_layer_selects_attention_mask_by_layer_type():
    masks = {"sliding_attention": "sliding", "full_attention": "full"}
    for lt, expect in (("sliding_attention", "sliding"), ("full_attention", "full")):
        layer = _Layer(layer_type=lt)
        _call_layer(layer, torch.zeros(1), position_embeddings=None,
                    attention_mask=masks, position_ids=None)
        assert layer.received_mask == expect


def test_call_layer_rejects_unknown_attention_mask_type():
    layer = _Layer(layer_type="missing")
    try:
        _call_layer(layer, torch.zeros(1), position_embeddings=None,
                    attention_mask={"full_attention": "mask"},
                    position_ids=None)
    except RuntimeError as exc:
        assert "per-layer attention mask" in str(exc)
    else:  # pragma: no cover - assert path keeps compatibility without pytest.raises
        raise AssertionError("unknown layer_type accepted for attention mask")
