from __future__ import annotations

import pytest

from prismaquant.budget_swaps import (
    build_budget_neutral_swaps,
    select_measured_budget_swaps,
)


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


def test_select_measured_budget_swaps_keeps_improving_disjoint_rows():
    assignment = {
        "a": "NVFP4",
        "b": "MXFP8_E4M3",
        "c": "NVFP4",
        "d": "MXFP8_E4M3",
    }
    rows = [
        {
            "key": "good-1",
            "measured_rank": 1,
            "swap_delta_kl_vs_bf16": -0.02,
            "swap_kl_vs_bf16": 0.10,
            "swap_kl_vs_base_assignment": 0.01,
            "net_bits_delta": -8.0,
            "override": {"a": "MXFP8_E4M3", "b": "NVFP4"},
        },
        {
            "key": "conflicts-with-good-1",
            "measured_rank": 2,
            "swap_delta_kl_vs_bf16": -0.01,
            "swap_kl_vs_bf16": 0.11,
            "swap_kl_vs_base_assignment": 0.02,
            "net_bits_delta": -4.0,
            "override": {"a": "BF16", "c": "NVFP4"},
        },
        {
            "key": "worse",
            "measured_rank": 3,
            "swap_delta_kl_vs_bf16": 0.01,
            "swap_kl_vs_bf16": 0.13,
            "swap_kl_vs_base_assignment": 0.01,
            "net_bits_delta": -4.0,
            "override": {"d": "NVFP4"},
        },
        {
            "key": "good-2",
            "measured_rank": 4,
            "swap_delta_kl_vs_bf16": -0.005,
            "swap_kl_vs_bf16": 0.12,
            "swap_kl_vs_base_assignment": 0.01,
            "net_bits_delta": 4.0,
            "override": {"c": "MXFP8_E4M3"},
        },
    ]

    result = select_measured_budget_swaps(assignment, rows)

    assert result["selected_count"] == 2
    assert result["selected_net_bits_delta"] == -4.0
    assert result["selected_delta_kl_vs_bf16_sum"] == pytest.approx(-0.025)
    assert result["assignment"]["a"] == "MXFP8_E4M3"
    assert result["assignment"]["b"] == "NVFP4"
    assert result["assignment"]["c"] == "MXFP8_E4M3"
    assert [row["key"] for row in result["selected"]] == ["good-1", "good-2"]
    skipped = {row["key"]: row["skip_reason"] for row in result["skipped"]}
    assert skipped["conflicts-with-good-1"] == "conflict"
    assert skipped["worse"] == "below_min_kl_improvement"
