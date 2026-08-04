"""Qwen3.6-35B-A3B producer-profile contract.

The census fixture below is transcribed from the official
``Qwen/Qwen3.6-35B-A3B`` config and safetensors index.  It pins the real
checkpoint names that drive source/live/vLLM rewrites, fused siblings, packed
expert handling, and the serving-contract producer id without checking model
weights into the repository.
"""
from __future__ import annotations

import re

import pytest

from prismaquant.model_profiles.registry import profile_from_config
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.model_profiles.qwen3_5_dense import Qwen3_5DenseProfile


@pytest.fixture
def qwen36_35b_census():
    return {
        "repo_id": "Qwen/Qwen3.6-35B-A3B",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_model_type": "qwen3_5_moe_text",
        "layers": 40,
        "linear_attention_layers": 30,
        "full_attention_layers": 10,
        "experts": 256,
        "top_k": 8,
        "hidden": 2048,
        "expert_intermediate": 512,
        "shared_expert_intermediate": 512,
        "mtp_layers": 1,
        "real_checkpoint_names": (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.39.mlp.experts.down_proj",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj",
            "model.language_model.layers.39.mlp.shared_expert.down_proj",
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "model.language_model.layers.38.linear_attn.out_proj",
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.39.self_attn.o_proj",
        ),
    }


def test_qwen36_moe_profile_keeps_packed_expert_units():
    profile = profile_from_config({
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_6MoeForCausalLM"],
    })

    assert isinstance(profile, Qwen3_5Profile)
    assert profile.packed_expert_param_names() == frozenset({
        "gate_up_proj",
        "down_proj",
    })
    assert profile.to_vllm_internal_name(
        "model.layers.0.mlp.experts.gate_up_proj"
    ) == "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert profile.per_expert_moe_regex() is not None


def test_qwen36_35b_official_config_resolves_to_gridbook_producer_id(
    qwen36_35b_census,
):
    profile = profile_from_config({
        "model_type": qwen36_35b_census["model_type"],
        "architectures": qwen36_35b_census["architectures"],
    })

    assert isinstance(profile, Qwen3_5Profile)
    assert profile.name == "qwen3_5"
    assert profile.vllm_architecture_class() == (
        "Qwen3_5MoeForConditionalGeneration"
    )
    assert profile.supported_export_lanes() == (
        "compressed-tensors",
        "nvfp4_cb",
    )


def test_qwen36_35b_real_census_names_round_trip(qwen36_35b_census):
    profile = Qwen3_5Profile()
    for checkpoint_name in qwen36_35b_census["real_checkpoint_names"]:
        live_name = profile.checkpoint_to_live_name(checkpoint_name)
        assert live_name == checkpoint_name.replace(
            "model.language_model.", "model.", 1
        )
        assert profile.source_tensor_name(live_name) == checkpoint_name
        assert profile.export_tensor_name(live_name) == checkpoint_name
        assert profile.to_vllm_internal_name(live_name) == (
            checkpoint_name.replace(
                "model.language_model.", "language_model.model.", 1
            )
        )


def test_qwen36_35b_census_pins_packed_and_shared_expert_shapes(
    qwen36_35b_census,
):
    c = qwen36_35b_census
    assert (c["experts"], 2 * c["expert_intermediate"], c["hidden"]) == (
        256,
        1024,
        2048,
    )
    assert (c["experts"], c["hidden"], c["expert_intermediate"]) == (
        256,
        2048,
        512,
    )
    assert (c["shared_expert_intermediate"], c["hidden"]) == (512, 2048)
    assert (c["hidden"], c["shared_expert_intermediate"]) == (2048, 512)

    profile = Qwen3_5Profile()
    assert profile.packed_expert_param_names() == frozenset({
        "gate_up_proj",
        "down_proj",
    })
    assert profile.packed_expert_projection_names("gate_up_proj") == (
        "gate_proj",
        "up_proj",
    )
    assert profile.packed_expert_projection_names("down_proj") == (
        "down_proj",
    )


def test_qwen36_35b_per_expert_vllm_names_cover_the_full_census(
    qwen36_35b_census,
):
    c = qwen36_35b_census
    names = [
        "language_model.model.layers."
        f"{layer}.mlp.experts.{expert}.{projection}_proj"
        for layer in range(c["layers"])
        for expert in range(c["experts"])
        for projection in ("gate", "up", "down")
    ]
    regex = re.compile(
        Qwen3_5Profile().per_expert_moe_regex().removeprefix("re:")
    )
    assert len(names) == 40 * 256 * 3 == 30_720
    assert all(regex.fullmatch(name) for name in names)


def test_qwen36_35b_fused_groups_match_real_layout():
    profile = Qwen3_5Profile()
    assert profile.fused_sibling_group(
        "model.layers.0.mlp.shared_expert.gate_proj"
    ) == "model.layers.0.mlp.shared_expert.gate_up_proj"
    assert profile.fused_sibling_group(
        "model.layers.0.mlp.shared_expert.up_proj"
    ) == "model.layers.0.mlp.shared_expert.gate_up_proj"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_qkv"
    ) == "model.layers.0.linear_attn.in_proj_qkvz"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_z"
    ) == "model.layers.0.linear_attn.in_proj_qkvz"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_b"
    ) == "model.layers.0.linear_attn.in_proj_ba"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_a"
    ) == "model.layers.0.linear_attn.in_proj_ba"


def test_qwen36_dense_profile_still_wins_for_non_moe_arch():
    profile = profile_from_config({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_6ForCausalLM"],
    })

    assert isinstance(profile, Qwen3_5DenseProfile)
    assert profile.packed_expert_param_names() == frozenset()
