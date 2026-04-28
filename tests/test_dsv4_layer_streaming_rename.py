"""Unit tests for the DSv4 ckpt→live rename in layer_streaming.

Forward-pair to test_deepseek_v4_profile.py: the profile owns
`source_tensor_name` (live → ckpt); this file pins the inverse
direction (`_rename_dsv4_text_only(ckpt_key) → live_qname`) used by
`_build_weight_map` to populate the streaming loader's
{model_key: ckpt_key} dict.
"""
from __future__ import annotations

import pytest

from prismaquant.layer_streaming import _rename_dsv4_text_only as _rename


def test_top_level_rename():
    assert _rename("embed.weight") == "model.embed_tokens.weight"
    assert _rename("head.weight") == "lm_head.weight"
    assert _rename("norm.weight") == "model.norm.weight"
    assert _rename("hc_head_fn") == "model.hc_head.hc_fn"
    assert _rename("hc_head_base") == "model.hc_head.hc_base"
    assert _rename("hc_head_scale") == "model.hc_head.hc_scale"


def test_top_level_drops():
    """MTP, scale-only siblings, and weight_scale_inv all drop. (hc_head_*
    are now mapped, not dropped — they live under model.hc_head and are
    needed for the multi-stream → single-stream collapse at end-of-body.)"""
    assert _rename("mtp.0.norm.weight") is None
    assert _rename("mtp.0.hc_attn_base") is None
    assert _rename("head.scale") is None
    assert _rename("embed.scale") is None
    assert _rename("layers.0.attn.wkv.weight_scale_inv") is None


def test_attn_rename():
    cases = [
        ("layers.0.attn.wkv.weight",
         "model.layers.0.self_attn.wkv.weight"),
        ("layers.5.attn.wq_a.weight",
         "model.layers.5.self_attn.wq_a.weight"),
        ("layers.42.attn.wq_b.weight",
         "model.layers.42.self_attn.wq_b.weight"),
        ("layers.0.attn.q_norm.weight",
         "model.layers.0.self_attn.q_norm.weight"),
        # PR #45643's `DeepseekV4Attention` exposes the per-head bias
        # buffer as `self.sinks`; checkpoint stores it as `attn.attn_sink`.
        ("layers.0.attn.attn_sink",
         "model.layers.0.self_attn.sinks"),
        # PRISMAQUANT probe mode drops compressor + indexer keys
        # entirely (see comment in `_rename_dsv4_text_only`). PR
        # #45643's compressor wkv shape doesn't match the checkpoint
        # (K and V are concatenated in the source), and the modeling
        # patch makes attention skip the compressor branch always.
        # Test that the drop is consistent for every variant we know
        # about.
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_ffn_rename():
    """`ffn` infix → `mlp`, including for routing gate and shared experts."""
    cases = [
        ("layers.0.ffn.gate.weight", "model.layers.0.mlp.gate.weight"),
        ("layers.0.ffn.gate.bias",   "model.layers.0.mlp.gate.bias"),
        # Shared expert leafs renamed: w1→gate_proj, w2→down_proj, w3→up_proj.
        ("layers.0.ffn.shared_experts.w1.weight",
         "model.layers.0.mlp.shared_experts.gate_proj.weight"),
        ("layers.0.ffn.shared_experts.w2.weight",
         "model.layers.0.mlp.shared_experts.down_proj.weight"),
        ("layers.0.ffn.shared_experts.w3.weight",
         "model.layers.0.mlp.shared_experts.up_proj.weight"),
        ("layers.0.ffn.shared_experts.w1.scale",
         "model.layers.0.mlp.shared_experts.gate_proj.scale"),
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_compressor_and_indexer_drop():
    """Probe-mode drops compressor + indexer keys (modeling patch
    skips the long-range branch; weights pass through at export)."""
    drops = [
        "layers.0.attn.compressor.wkv.weight",
        "layers.5.attn.compressor.ape",
        "layers.5.attn.compressor.norm.weight",
        "layers.5.attn.compressor.wgate.weight",
        "layers.2.attn.indexer.compressor.wkv.weight",
        "layers.2.attn.indexer.compressor.norm.weight",
        "layers.2.attn.indexer.compressor.ape",
        "layers.2.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.weights_proj.weight",
    ]
    for k in drops:
        assert _rename(k) is None, f"{k} should drop, got {_rename(k)}"


def test_routed_experts_per_expert_rename():
    """Routed experts map to per-expert ModuleList live names (set up
    by `enable_per_expert_experts()`). w1→gate_proj, w2→down_proj,
    w3→up_proj per expert. Suffixes (.weight, .scale) preserved."""
    cases = [
        ("layers.0.ffn.experts.0.w1.weight",
         "model.layers.0.mlp.experts.0.gate_proj.weight"),
        ("layers.0.ffn.experts.0.w2.weight",
         "model.layers.0.mlp.experts.0.down_proj.weight"),
        ("layers.0.ffn.experts.0.w3.weight",
         "model.layers.0.mlp.experts.0.up_proj.weight"),
        ("layers.0.ffn.experts.255.w3.weight",
         "model.layers.0.mlp.experts.255.up_proj.weight"),
        ("layers.42.ffn.experts.128.w2.weight",
         "model.layers.42.mlp.experts.128.down_proj.weight"),
        # FP8 block scale siblings preserve suffix
        ("layers.0.ffn.experts.0.w1.scale",
         "model.layers.0.mlp.experts.0.gate_proj.scale"),
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_hyper_connection_rename():
    """`hc_attn_X` → `attn_hc.X`, `hc_ffn_X` → `ffn_hc.X` (inverse of
    profile's compaction)."""
    cases = [
        ("layers.0.hc_attn_base", "model.layers.0.attn_hc.base"),
        ("layers.0.hc_attn_fn",   "model.layers.0.attn_hc.fn"),
        ("layers.0.hc_attn_scale", "model.layers.0.attn_hc.scale"),
        ("layers.5.hc_ffn_base",  "model.layers.5.ffn_hc.base"),
        ("layers.5.hc_ffn_scale", "model.layers.5.ffn_hc.scale"),
    ]
    for ck, live in cases:
        assert _rename(ck) == live


def test_attn_norm_ffn_norm_to_layernorms():
    """The DSv4 checkpoint stores `attn_norm` (pre-attention RMSNorm)
    and `ffn_norm` (pre-MLP RMSNorm) at the layer scope. The
    transformers `DecoderLayer` exposes them as `input_layernorm`
    and `post_attention_layernorm`."""
    cases = [
        ("layers.0.attn_norm.weight",
         "model.layers.0.input_layernorm.weight"),
        ("layers.5.attn_norm.weight",
         "model.layers.5.input_layernorm.weight"),
        ("layers.0.ffn_norm.weight",
         "model.layers.0.post_attention_layernorm.weight"),
        ("layers.42.ffn_norm.weight",
         "model.layers.42.post_attention_layernorm.weight"),
    ]
    for ck, live in cases:
        assert _rename(ck) == live, f"{ck} ↦ {_rename(ck)}, expected {live}"


def test_layernorm_passthrough_already_standard():
    """If a checkpoint already uses the standard names, pass through with
    just the `model.` prefix add (defensive)."""
    assert (_rename("layers.0.input_layernorm.weight")
            == "model.layers.0.input_layernorm.weight")
    assert (_rename("layers.0.post_attention_layernorm.weight")
            == "model.layers.0.post_attention_layernorm.weight")


def test_unrecognized_drops():
    """Unrecognized top-level keys drop rather than risk shadowing real
    body tensors. The probe will report which keys went missing if the
    DSv4 checkpoint introduces new top-level structure."""
    # No real DSv4 keys collide here, but verify the conservative drop.
    assert _rename("some_unknown_key") is None
    assert _rename("debug_only_thing") is None
