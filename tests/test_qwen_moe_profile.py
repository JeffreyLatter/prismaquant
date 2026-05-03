from prismaquant.model_profiles.registry import profile_from_config
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.model_profiles.qwen3_5_dense import Qwen3_5DenseProfile


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


def test_qwen36_dense_profile_still_wins_for_non_moe_arch():
    profile = profile_from_config({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_6ForCausalLM"],
    })

    assert isinstance(profile, Qwen3_5DenseProfile)
    assert profile.packed_expert_param_names() == frozenset()
