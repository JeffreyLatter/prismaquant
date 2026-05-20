"""Unit tests for the DeepseekV4Profile naming bridge.

DSv4 has the most complex live↔checkpoint name remapping of any profile
(self_attn↔attn, mlp↔ffn, gate_proj/up_proj/down_proj↔w1/w3/w2 for
shared experts, hyper-connection compaction). This file pins the
expected behavior so renames stay correct as the rest of the pipeline
evolves.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile


@pytest.fixture
def profile() -> DeepseekV4Profile:
    return DeepseekV4Profile()


def test_match_by_model_type(profile):
    assert DeepseekV4Profile.matches("deepseek_v4", [])
    assert DeepseekV4Profile.matches("deepseek-v4", [])
    assert not DeepseekV4Profile.matches("deepseek_v3", [])
    assert not DeepseekV4Profile.matches("minimax_m2", [])


def test_match_by_architecture(profile):
    assert DeepseekV4Profile.matches("", ["DeepseekV4ForCausalLM"])
    assert DeepseekV4Profile.matches("", ["DeepSeek-V4-Custom"])
    assert not DeepseekV4Profile.matches("", ["DeepseekV3ForCausalLM"])


def test_layer_prefixes(profile):
    assert profile.body_layer_prefix() == "layers"
    assert profile.lm_head_name() == "head"
    assert profile.mtp_layer_prefix() == "mtp"


def test_packed_expert_param_names(profile):
    assert profile.packed_expert_param_names() == frozenset({"gate_up_proj", "down_proj"})


def test_source_passthrough_prefixes_are_spec_backed(profile):
    assert profile.source_passthrough_prefixes() == (
        "attn_sink",
        "hc_",
        "compressor.ape",
        "tid2eid",
        "kv_norm",
        "q_norm",
        "norm.",
        "mtp.",
    )


def test_probe_skip_linear_class_names_are_spec_backed(profile):
    DeepseekV4GroupedLinear = type(
        "DeepseekV4GroupedLinear",
        (nn.Linear,),
        {},
    )

    assert profile.structure_spec().probe_skip_module_class_names == (
        "DeepseekV4GroupedLinear",
    )
    assert profile.should_probe_linear("model.layers.0.self_attn.wq_a", nn.Linear(4, 4))
    assert not profile.should_probe_linear(
        "model.layers.0.self_attn.wo_a",
        DeepseekV4GroupedLinear(4, 4),
    )


def test_split_packed_for_export(profile):
    # Mirror Qwen3.5: always split per-expert for export safety.
    assert profile.split_packed_experts_for_format("nvfp4") is True
    assert profile.split_packed_experts_for_format("fp8_source") is True
    assert profile.split_packed_experts_for_format("bf16") is True


def test_top_level_name_bridge(profile):
    """`lm_head` / `embed_tokens` / `norm` rename to DSv4's flat keys."""
    assert profile.source_tensor_name("lm_head.weight") == "head.weight"
    assert profile.source_tensor_name("model.embed_tokens.weight") == "embed.weight"
    assert profile.source_tensor_name("model.norm.weight") == "norm.weight"


def test_decoder_layer_attn_bridge(profile):
    """`self_attn` → `attn`, including for nested compressor + indexer."""
    cases = [
        ("model.layers.0.self_attn.wkv.weight",
         "layers.0.attn.wkv.weight"),
        ("model.layers.5.self_attn.wq_a.weight",
         "layers.5.attn.wq_a.weight"),
        ("model.layers.42.self_attn.wq_b.weight",
         "layers.42.attn.wq_b.weight"),
        ("model.layers.0.self_attn.q_norm.weight",
         "layers.0.attn.q_norm.weight"),
        ("model.layers.0.self_attn.compressor.wkv.weight",
         "layers.0.attn.compressor.wkv.weight"),
        ("model.layers.0.self_attn.compressor.wgate.weight",
         "layers.0.attn.compressor.wgate.weight"),
        ("model.layers.0.self_attn.compressor.indexer.weights_proj.weight",
         "layers.0.attn.compressor.indexer.weights_proj.weight"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source, (
            f"{live} ↦ {profile.source_tensor_name(live)}, expected {source}")


def test_decoder_layer_mlp_bridge(profile):
    """`mlp` → `ffn`, including routed/shared experts."""
    cases = [
        ("model.layers.0.mlp.gate.weight", "layers.0.ffn.gate.weight"),
        ("model.layers.0.mlp.experts.gate_up_proj",
         "layers.0.ffn.experts.gate_up_proj"),
        ("model.layers.0.mlp.experts.down_proj",
         "layers.0.ffn.experts.down_proj"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source


def test_shared_expert_leaf_rename(profile):
    """Shared experts: gate_proj/up_proj/down_proj → w1/w3/w2 in source."""
    cases = [
        ("model.layers.0.mlp.shared_experts.gate_proj.weight",
         "layers.0.ffn.shared_experts.w1.weight"),
        ("model.layers.0.mlp.shared_experts.up_proj.weight",
         "layers.0.ffn.shared_experts.w3.weight"),
        ("model.layers.0.mlp.shared_experts.down_proj.weight",
         "layers.0.ffn.shared_experts.w2.weight"),
        # Same with .scale suffix (FP8 block scales)
        ("model.layers.5.mlp.shared_experts.gate_proj.scale",
         "layers.5.ffn.shared_experts.w1.scale"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source


def test_hyper_connection_compaction(profile):
    """`attn_hc.X` / `ffn_hc.X` → `hc_attn_X` / `hc_ffn_X` (flat keys)."""
    cases = [
        ("model.layers.0.attn_hc.base", "layers.0.hc_attn_base"),
        ("model.layers.0.attn_hc.fn",   "layers.0.hc_attn_fn"),
        ("model.layers.0.attn_hc.scale", "layers.0.hc_attn_scale"),
        ("model.layers.5.ffn_hc.base", "layers.5.hc_ffn_base"),
        ("model.layers.5.ffn_hc.scale", "layers.5.hc_ffn_scale"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source


def test_layernorm_passthrough(profile):
    """input_layernorm / post_attention_layernorm pass through unchanged
    (they're at the layer level, no rename needed beyond model-prefix strip)."""
    assert (profile.source_tensor_name("model.layers.0.input_layernorm.weight")
            == "layers.0.input_layernorm.weight")
    assert (profile.source_tensor_name("model.layers.0.post_attention_layernorm.weight")
            == "layers.0.post_attention_layernorm.weight")


def test_mtp_layer_count(profile):
    assert profile.mtp_layer_count({"num_nextn_predict_layers": 1}) == 1
    assert profile.mtp_layer_count({"num_nextn_predict_layers": 2}) == 2
    # Backward-compat key
    assert profile.mtp_layer_count({"num_mtp_layers": 1}) == 1
    # Missing → 0
    assert profile.mtp_layer_count({}) == 0


def test_no_fused_siblings(profile):
    """DSv4-Flash has no Q/K/V or gate/up fused siblings on the live side."""
    assert profile.fused_sibling_group("model.layers.0.self_attn.wq_a") is None
    assert profile.fused_sibling_group("model.layers.0.self_attn.wkv") is None
    assert profile.fused_sibling_group("model.layers.0.mlp.shared_experts.gate_proj") is None
