"""Integration: streaming forward through REAL Qwen3.5 hybrid layers.

The fakes in test_linear_attention_mask_routing.py pin the contract; this
file pins it against transformers' actual Qwen3_5 modules (tiny random-weight
hybrid, CPU) — the family whose transformers>=5.15 refactor (`.block_type`
rename, `apply_mask_to_padding_states` without a shape guard) motivated the
fix. The forward-path tests run on every transformers version that ships
qwen3_5 (the locked 5.8 environment included — same contract, pre-refactor
attribute names); the two assertions specific to the 5.15 refactor
(`.block_type`/child `layer_type`, guardless ``apply_mask_to_padding_states``)
are version-gated.
"""
import pytest
import torch

pytest.importorskip(
    "transformers.models.qwen3_5",
    reason="qwen3_5 modeling not available in this transformers version",
)

import transformers
from packaging.version import parse as _parse_version

_POST_REFACTOR = _parse_version(
    transformers.__version__) >= _parse_version("5.15.0")

from transformers.models.qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _layer_attention_type,
)


def _tiny_hybrid():
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        vocab_size=128, max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    layers = [Qwen3_5DecoderLayer(cfg, i) for i in range(2)]
    rotary = Qwen3_5TextRotaryEmbedding(cfg)

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg

    return cfg, _Base(), layers, rotary


def _stream(base, layers, rotary, hidden, attention_mask):
    position_ids = torch.arange(hidden.size(1)).unsqueeze(0).expand(
        hidden.size(0), -1)
    masks = _compute_attention_mask(base, hidden, position_ids,
                                    attention_mask=attention_mask)
    pe = rotary(hidden, position_ids)
    out = hidden
    for layer in layers:
        out = _call_layer(layer, out, position_embeddings=pe,
                          attention_mask=masks, position_ids=position_ids)
    return masks, out


def test_real_layers_resolve():
    # Passes pre- and post-refactor: `.layer_type` on 5.8, `.block_type`
    # (plus the recurrent child's own attributes) from 5.15 on.
    _, _, layers, _ = _tiny_hybrid()
    assert _layer_attention_type(layers[0]) == "linear_attention"
    assert _layer_attention_type(layers[1]) == "full_attention"


@pytest.mark.skipif(not _POST_REFACTOR,
                    reason="`.block_type` rename landed in transformers 5.15")
def test_post_refactor_child_carries_fallback_attributes():
    _, _, layers, _ = _tiny_hybrid()
    assert layers[0].block_type == "linear_attention"
    assert layers[0].linear_attn.layer_type == "linear_attention"
    assert layers[0].linear_attn.layer_idx == 0


def test_streaming_forward_left_padded_batch():
    _, base, layers, rotary = _tiny_hybrid()
    hidden = torch.randn(2, 6, 64)
    pad = torch.tensor([[0, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1]])

    masks, out = _stream(base, layers, rotary, hidden, pad)

    lin = masks["linear_attention"]
    assert lin is not None and lin.ndim == 2 and torch.equal(lin, pad)
    assert out.shape == hidden.shape
    assert bool(torch.isfinite(out).all())


def test_streaming_forward_unpadded_batch():
    _, base, layers, rotary = _tiny_hybrid()
    hidden = torch.randn(1, 6, 64)

    # all-ones → linear_attention mask is None; forward must still work
    masks, out = _stream(base, layers, rotary, hidden, torch.ones(1, 6))

    assert masks["linear_attention"] is None
    assert out.shape == hidden.shape
    assert bool(torch.isfinite(out).all())


@pytest.mark.skipif(
    not _POST_REFACTOR,
    reason="pre-5.15 apply_mask_to_padding_states still shape-guards "
           "non-2D masks, so the dense mask is silently ignored there")
def test_dense_mask_crashes_real_linear_layer():
    # Documents the pre-fix failure mode this branch removes: a dense
    # [1, 1, T, T] additive mask fed to a real GatedDeltaNet layer
    # broadcasts against hidden_states and raises a trailing-dim mismatch.
    _, _, layers, rotary = _tiny_hybrid()
    hidden = torch.randn(1, 6, 64)
    position_ids = torch.arange(6).unsqueeze(0)
    pe = rotary(hidden, position_ids)
    dense = torch.zeros(1, 1, 6, 6)
    with pytest.raises(RuntimeError, match="must match the size"):
        layers[0](hidden_states=hidden, attention_mask=dense,
                  position_ids=position_ids, past_key_values=None,
                  use_cache=False, position_embeddings=pe)
