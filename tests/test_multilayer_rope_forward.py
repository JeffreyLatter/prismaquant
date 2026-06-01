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

from prismaquant.layer_streaming import (
    _call_layer,
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
    def __init__(self, rotary):
        super().__init__()
        self.rotary_emb = rotary


class _Layer(nn.Module):
    """Records the position_embeddings it actually received."""
    def __init__(self, layer_type=None):
        super().__init__()
        if layer_type is not None:
            self.self_attn = nn.Module()
            self.self_attn.layer_type = layer_type
        self.received = None

    def forward(self, *, hidden_states, position_embeddings, **kw):
        self.received = position_embeddings
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


def test_call_layer_dict_unknown_type_falls_back():
    pe = {"only": ("x", "x")}
    layer = _Layer(layer_type="missing")  # not in dict
    _call_layer(layer, torch.zeros(1), position_embeddings=pe,
                attention_mask=None, position_ids=None)
    assert layer.received == ("x", "x")  # falls back to an entry, no crash
