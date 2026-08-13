"""Linear-attention (recurrent) hybrid masking and layer-type resolution.

Pins the PR #80 review contract. Two halves:

- `linear_attention` layers are routed through transformers'
  recurrent-mask contract (`masking_utils.create_recurrent_attention_mask`
  when the installed transformers ships it, the contract-identical
  `_recurrent_padding_mask` fallback on the locked transformers 5.8):
  2D padding mask trimmed to the local sequence, `None` whenever masking
  would be a no-op — never the dense additive causal mask, never the raw
  un-trimmed growing mask.
- layer-type resolution covers the transformers>=5.15 `.block_type` rename
  and the Qwen3.5/3.6 `linear_attn` child module through the existing
  attribute/index/config lookups, and an otherwise unknown layer still
  fails closed instead of being guessed structurally.

No transformers modeling dependency — fakes only, same as
test_multilayer_rope_forward.py.
"""
import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _layer_attention_type,
    _linear_attention_mask,
)


# --- fakes -----------------------------------------------------------------
class _Base(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config


class _RecorderLayer(nn.Module):
    """Records the attention_mask it actually received."""
    def __init__(self, *, block_type=None, layer_type=None):
        super().__init__()
        if block_type is not None:
            self.block_type = block_type
        if layer_type is not None:
            self.layer_type = layer_type
        self.received_mask = "UNSET"

    def forward(self, *, hidden_states, **kw):
        self.received_mask = kw.get("attention_mask")
        return hidden_states


def _hybrid_cfg():
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["full_attention", "linear_attention"]
    cfg._attn_implementation = "eager"
    return cfg


def _mask_kwargs(cfg, hidden, pad, position_ids=None):
    return {
        "config": cfg,
        "inputs_embeds": hidden,
        "attention_mask": pad,
        "past_key_values": None,
        "position_ids": position_ids,
    }


# --- layer-type resolution -------------------------------------------------
@pytest.mark.parametrize("lt", ["linear_attention", "full_attention"])
def test_block_type_resolves(lt):
    # transformers>=5.15: Qwen3_5DecoderLayer stores `.block_type`
    layer = nn.Module()
    layer.block_type = lt
    assert _layer_attention_type(layer) == lt


@pytest.mark.parametrize("lt", ["linear_attention", "sliding_attention"])
def test_legacy_layer_type_still_resolves(lt):
    # transformers<5.15 name — must keep working unchanged
    layer = nn.Module()
    layer.layer_type = lt
    assert _layer_attention_type(layer) == lt


def test_linear_attn_child_layer_type_resolves():
    # Qwen3_5GatedDeltaNet carries its own `layer_type`; the outer layer
    # exposes neither layer_type/block_type nor self_attn/attention.
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.linear_attn.layer_type = "linear_attention"
    assert _layer_attention_type(layer) == "linear_attention"


def test_linear_attn_child_layer_idx_config_resolves():
    # ...and its `layer_idx` feeds the generic config.layer_types fallback.
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.linear_attn.layer_idx = 1
    layer.config = _hybrid_cfg()
    assert _layer_attention_type(layer) == "linear_attention"


def test_unknown_self_attn_layer_fails_closed():
    # A layer whose type cannot be resolved must stay unresolved (None) and
    # make _call_layer raise — never silently assume full_attention.
    layer = _RecorderLayer()
    layer.self_attn = nn.Module()
    assert _layer_attention_type(layer) is None
    with pytest.raises(RuntimeError, match="known layer_type"):
        _call_layer(layer, torch.zeros(1, 2, 4),
                    position_embeddings=None,
                    attention_mask={"full_attention": None},
                    position_ids=None)


# --- recurrent-mask contract (direct) --------------------------------------
def test_continuation_mask_trims_to_local_sequence():
    cfg = _hybrid_cfg()
    hidden = torch.zeros(2, 4, 8)
    # growing cache-continuation mask: 6 total positions, local seq is 4
    pad = torch.tensor([[1, 1, 0, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1]])
    out = _linear_attention_mask(_mask_kwargs(cfg, hidden, pad), hidden, pad)
    assert out.shape == (2, 4)
    assert torch.equal(out, pad[:, -4:])
    assert out.is_contiguous()


def test_single_token_decode_returns_none():
    cfg = _hybrid_cfg()
    hidden = torch.zeros(1, 1, 8)
    # decode step continuing a cached, left-padded prompt: not all-ones,
    # longer than the local sequence — None purely because local seq == 1
    pad = torch.tensor([[0, 1, 1, 1, 1, 1]])
    assert _linear_attention_mask(
        _mask_kwargs(cfg, hidden, pad), hidden, pad) is None


def test_non_2d_mask_returns_none():
    cfg = _hybrid_cfg()
    hidden = torch.zeros(1, 4, 8)
    pad4d = torch.zeros(1, 1, 4, 4)
    assert _linear_attention_mask(
        _mask_kwargs(cfg, hidden, pad4d), hidden, pad4d) is None


def test_missing_mask_returns_none():
    cfg = _hybrid_cfg()
    hidden = torch.zeros(1, 4, 8)
    assert _linear_attention_mask(
        _mask_kwargs(cfg, hidden, None), hidden, None) is None


# --- routing through _compute_attention_mask -------------------------------
def test_left_padded_mask_routes_2d_to_linear():
    cfg = _hybrid_cfg()
    base = _Base(cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)
    pad = torch.tensor([[0, 1, 1, 1]])

    masks = _compute_attention_mask(base, hidden, position_ids,
                                    attention_mask=pad)

    lin = masks["linear_attention"]
    assert lin is not None and lin.ndim == 2  # never the dense 4D mask
    assert torch.equal(lin, pad)
    assert lin.is_contiguous()
    assert masks["full_attention"].shape == (1, 1, 4, 4)


def test_all_ones_mask_routes_none_to_linear():
    cfg = _hybrid_cfg()
    base = _Base(cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)

    masks = _compute_attention_mask(base, hidden, position_ids,
                                    attention_mask=torch.ones(1, 4))

    assert masks["linear_attention"] is None
    assert masks["full_attention"].shape == (1, 1, 4, 4)


def test_sliding_full_mapping_non_regression():
    # Gemma-style hybrid must be untouched by the linear-attention branch.
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["sliding_attention", "full_attention"]
    cfg.sliding_window = 2
    cfg._attn_implementation = "eager"
    base = _Base(cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)

    masks = _compute_attention_mask(base, hidden, position_ids)

    assert set(masks) == {"sliding_attention", "full_attention"}
    assert masks["full_attention"].shape == (1, 1, 4, 4)
    assert masks["sliding_attention"].shape == (1, 1, 4, 4)
    assert float(masks["full_attention"][0, 0, 3, 0]) == 0.0
    assert float(masks["sliding_attention"][0, 0, 3, 0]) < -1e20


# --- end-to-end selection through _call_layer ------------------------------
def test_call_layer_delivers_recurrent_mask_to_linear_layer():
    pad = torch.tensor([[0, 1, 1, 1]])
    dense = torch.zeros(1, 1, 4, 4)
    masks = {"full_attention": dense, "linear_attention": pad}

    linear = _RecorderLayer(block_type="linear_attention")
    _call_layer(linear, torch.zeros(1, 4, 8), position_embeddings=None,
                attention_mask=masks, position_ids=None)
    assert linear.received_mask is pad

    full = _RecorderLayer(block_type="full_attention")
    _call_layer(full, torch.zeros(1, 4, 8), position_embeddings=None,
                attention_mask=masks, position_ids=None)
    assert full.received_mask is dense
