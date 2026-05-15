from __future__ import annotations

from prismaquant.serving_profiles import (
    check_serving_format,
    check_serving_shape,
    load_serving_profile,
    serving_profile_names,
)


VLLM_PROFILE = "vllm_qwen3_5_packed_moe"


def test_serving_profile_names_are_config_discovered():
    assert "research" in serving_profile_names()
    assert VLLM_PROFILE in serving_profile_names()


def test_vllm_profile_extends_runtime_shape_rules():
    profile = load_serving_profile(VLLM_PROFILE)

    assert profile.extends == ("research",)
    assert any(rule.id == "mxfp8_cutlass_shape" for rule in profile.shape_rules)
    assert any(rule.id == "packed_moe_expert_formats" for rule in profile.format_rules)


def test_serving_profile_format_rules_are_config_backed():
    expert = "model.layers.0.mlp.experts.gate_up_proj"
    gemma_expert = "model.layers.0.experts.gate_up_proj"
    dense = "model.layers.0.self_attn.q_proj"

    assert check_serving_format(VLLM_PROFILE, expert, "MXFP8_E4M3").legal
    assert check_serving_format(VLLM_PROFILE, gemma_expert, "MXFP4").legal
    expert_fp8 = check_serving_format(VLLM_PROFILE, expert, "FP8_E4M3")
    assert not expert_fp8.legal
    assert expert_fp8.rule == "packed_moe_expert_formats"
    gemma_fp8 = check_serving_format(VLLM_PROFILE, gemma_expert, "FP8_E4M3")
    assert not gemma_fp8.legal
    assert gemma_fp8.rule == "packed_moe_expert_formats"

    dense_mxfp4 = check_serving_format(VLLM_PROFILE, dense, "MXFP4")
    assert not dense_mxfp4.legal
    assert dense_mxfp4.rule == "dense_formats_without_vllm_fast_path"


def test_serving_profile_shape_rules_are_config_backed():
    small_n = check_serving_shape(
        "research",
        "MXFP8_E4M3",
        in_features=5120,
        out_features=48,
    )
    standard = check_serving_shape(
        VLLM_PROFILE,
        "MXFP8_E4M3",
        in_features=5120,
        out_features=10240,
    )
    nvfp4_bad_k = check_serving_shape(
        "research",
        "NVFP4",
        in_features=17,
        out_features=128,
    )

    assert not small_n.legal
    assert small_n.reason == "kernel_shape"
    assert "out_features=48" in small_n.detail
    assert standard.legal
    assert not nvfp4_bad_k.legal
