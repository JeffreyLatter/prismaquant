from __future__ import annotations

from prismaquant.budget_swaps import build_budget_neutral_swaps


def _stats(shape=(64, 64)):
    out_features, in_features = shape
    return {
        "out_features": out_features,
        "in_features": in_features,
        "n_params": out_features * in_features,
        "h_trace": 10.0,
    }


class _FakeFusedProfile:
    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        if name.endswith((".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj")):
            return name.rsplit(".", 1)[0] + ".gate_up_proj"
        return None


def test_budget_swap_builder_pairs_sensitive_promotion_with_low_risk_demotion():
    assignment = {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.1.mlp.shared_expert.down_proj": "BF16",
    }
    stats = {name: _stats() for name in assignment}
    costs = {
        "model.layers.0.linear_attn.in_proj_qkv": {
            "NVFP4": {"output_mse": 0.50, "predicted_dloss": 0.20},
            "FP8_E4M3": {"output_mse": 0.10, "predicted_dloss": 0.05},
            "BF16": {"output_mse": 0.0, "predicted_dloss": 0.0},
        },
        "model.layers.1.mlp.shared_expert.down_proj": {
            "NVFP4": {"output_mse": 0.20, "predicted_dloss": 0.08},
            "FP8_E4M3": {"output_mse": 0.01, "predicted_dloss": 0.01},
            "BF16": {"output_mse": 0.0, "predicted_dloss": 0.0},
        },
    }
    report = {
        "rows": [{
            "key": "tensor:model.layers.0.linear_attn.in_proj_qkv",
            "members": ["model.layers.0.linear_attn.in_proj_qkv"],
            "propagated_kl": 0.2,
            "propagated_kl_per_added_bit": 1.0,
        }]
    }

    payload = build_budget_neutral_swaps(
        assignment,
        costs=costs,
        stats=stats,
        propagated_report=report,
        formats=["NVFP4", "FP8_E4M3", "BF16"],
        categories=["linear_attn", "shared_expert"],
        max_promotions=4,
        max_swaps=4,
    )

    assert payload["swap_count"] == 1
    swap = payload["swaps"][0]
    assert swap["net_bits_delta"] <= 0.0
    assert swap["override"]["model.layers.0.linear_attn.in_proj_qkv"] == "FP8_E4M3"
    assert swap["override"]["model.layers.1.mlp.shared_expert.down_proj"] == "FP8_E4M3"


def test_budget_swap_builder_respects_fused_sibling_units():
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "NVFP4",
        "model.layers.0.self_attn.v_proj": "NVFP4",
        "model.layers.1.mlp.shared_expert.gate_proj": "BF16",
        "model.layers.1.mlp.shared_expert.up_proj": "BF16",
    }
    stats = {name: _stats() for name in assignment}
    costs = {
        name: {
            "NVFP4": {"output_mse": 0.2, "predicted_dloss": 0.1},
            "FP8_E4M3": {"output_mse": 0.05, "predicted_dloss": 0.03},
            "BF16": {"output_mse": 0.0, "predicted_dloss": 0.0},
        }
        for name in assignment
    }
    report = {
        "rows": [{
            "key": "fused:model.layers.0.self_attn.qkv_proj",
            "members": [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ],
            "propagated_kl": 0.3,
            "propagated_kl_per_added_bit": 2.0,
        }]
    }

    payload = build_budget_neutral_swaps(
        assignment,
        costs=costs,
        stats=stats,
        propagated_report=report,
        formats=["NVFP4", "FP8_E4M3", "BF16"],
        categories=["self_attn", "shared_expert"],
        profile=_FakeFusedProfile(),
        max_promotions=2,
        max_demotions_per_swap=1,
        max_swaps=2,
    )

    assert payload["swap_count"] == 1
    swap = payload["swaps"][0]
    assert set(swap["promotion_unit"]["members"]) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert set(swap["demotion_units"][0]["members"]) == {
        "model.layers.1.mlp.shared_expert.gate_proj",
        "model.layers.1.mlp.shared_expert.up_proj",
    }
    assert all(
        swap["override"][name] == "FP8_E4M3"
        for name in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        )
    )
    assert all(
        swap["override"][name] == "FP8_E4M3"
        for name in (
            "model.layers.1.mlp.shared_expert.gate_proj",
            "model.layers.1.mlp.shared_expert.up_proj",
        )
    )


def test_budget_swap_builder_skips_unfunded_promotions():
    assignment = {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.1.mlp.shared_expert.down_proj": "NVFP4",
    }
    stats = {name: _stats() for name in assignment}
    costs = {
        name: {
            "NVFP4": {"output_mse": 0.2, "predicted_dloss": 0.1},
            "BF16": {"output_mse": 0.0, "predicted_dloss": 0.0},
        }
        for name in assignment
    }

    payload = build_budget_neutral_swaps(
        assignment,
        costs=costs,
        stats=stats,
        propagated_report={"rows": []},
        formats=["NVFP4", "BF16"],
        categories=["linear_attn", "shared_expert"],
    )

    assert payload["promotion_candidate_count"] == 2
    assert payload["demotion_candidate_count"] == 0
    assert payload["swap_count"] == 0
