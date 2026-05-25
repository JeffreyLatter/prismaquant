from __future__ import annotations

import pytest

from prismaquant.mse_promotion import (
    build_mse_promotion_assignment,
    build_promotion_candidate_report,
    layer_config_from_assignment,
)
from prismaquant.propagated_sensitivity_costs import (
    apply_propagated_sensitivity_penalty,
)


def _stats(shape):
    out_features, in_features = shape
    return {
        "out_features": out_features,
        "in_features": in_features,
        "n_params": out_features * in_features,
    }


class _FakeFusedProfile:
    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        if name.endswith((".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z")):
            return name.rsplit(".", 1)[0] + ".in_proj_qkvz"
        if name.endswith((".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj")):
            return name.rsplit(".", 1)[0] + ".gate_up_proj"
        return None


def test_mse_promotion_selects_highest_output_mse_per_bit_group():
    assignment = {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.0.linear_attn.in_proj_z": "NVFP4",
        "model.layers.1.self_attn.q_proj": "NVFP4",
        "model.layers.1.mlp.shared_expert.down_proj": "NVFP4",
    }
    stats = {
        name: _stats((64, 64))
        for name in assignment
    }
    costs = {
        "model.layers.0.linear_attn.in_proj_qkv": {
            "NVFP4": {"output_mse": 0.40, "weight_mse": 0.01}
        },
        "model.layers.0.linear_attn.in_proj_z": {
            "NVFP4": {"output_mse": 0.20, "weight_mse": 0.01}
        },
        "model.layers.1.self_attn.q_proj": {
            "NVFP4": {"output_mse": 0.05, "weight_mse": 0.01}
        },
        "model.layers.1.mlp.shared_expert.down_proj": {
            "NVFP4": {"output_mse": 10.0, "weight_mse": 0.01}
        },
    }

    result = build_mse_promotion_assignment(
        assignment,
        costs=costs,
        stats=stats,
        categories=["linear_attn", "self_attn"],
        target_format="BF16",
        max_bpp_delta=20.0,
        group_by="layer_category",
    )

    promoted = result["assignment"]
    report = result["report"]
    assert promoted["model.layers.0.linear_attn.in_proj_qkv"] == "BF16"
    assert promoted["model.layers.0.linear_attn.in_proj_z"] == "BF16"
    assert promoted["model.layers.1.self_attn.q_proj"] == "BF16"
    assert promoted["model.layers.1.mlp.shared_expert.down_proj"] == "NVFP4"
    assert report["selected_group_count"] == 2
    assert report["selected_output_mse_removed"] == pytest.approx(0.65)
    assert report["selected"][0]["key"] == "linear_attn.layer_0"


def test_mse_promotion_respects_bpp_budget():
    assignment = {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.0.linear_attn.in_proj_z": "NVFP4",
        "model.layers.1.self_attn.q_proj": "NVFP4",
    }
    stats = {
        name: _stats((64, 64))
        for name in assignment
    }
    costs = {
        "model.layers.0.linear_attn.in_proj_qkv": {
            "NVFP4": {"output_mse": 0.40, "weight_mse": 0.01}
        },
        "model.layers.0.linear_attn.in_proj_z": {
            "NVFP4": {"output_mse": 0.20, "weight_mse": 0.01}
        },
        "model.layers.1.self_attn.q_proj": {
            "NVFP4": {"output_mse": 0.05, "weight_mse": 0.01}
        },
    }

    result = build_mse_promotion_assignment(
        assignment,
        costs=costs,
        stats=stats,
        categories=["linear_attn", "self_attn"],
        target_format="BF16",
        max_bpp_delta=8.0,
        group_by="layer_category",
    )

    promoted = result["assignment"]
    report = result["report"]
    assert promoted["model.layers.0.linear_attn.in_proj_qkv"] == "BF16"
    assert promoted["model.layers.0.linear_attn.in_proj_z"] == "BF16"
    assert promoted["model.layers.1.self_attn.q_proj"] == "NVFP4"
    assert report["selected_group_count"] == 1
    assert report["budget_skipped_count"] == 1


def test_promotion_candidate_report_emits_current_format_overrides():
    assignment = {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.0.linear_attn.in_proj_z": "MXFP8_E4M3",
        "model.layers.1.self_attn.q_proj": "BF16",
        "model.layers.1.mlp.shared_expert.down_proj": "NVFP4",
    }
    stats = {
        name: _stats((64, 64))
        for name in assignment
    }
    costs = {
        "model.layers.0.linear_attn.in_proj_qkv": {
            "NVFP4": {"output_mse": 0.40, "weight_mse": 0.01}
        },
        "model.layers.0.linear_attn.in_proj_z": {
            "MXFP8_E4M3": {"output_mse": 0.20, "weight_mse": 0.01}
        },
        "model.layers.1.mlp.shared_expert.down_proj": {
            "NVFP4": {"output_mse": 10.0, "weight_mse": 0.01}
        },
    }

    payload = build_promotion_candidate_report(
        assignment,
        costs=costs,
        stats=stats,
        categories=["linear_attn", "self_attn"],
        target_format="BF16",
        group_by="layer_category",
    )

    candidates = payload["candidates"]
    overrides = payload["current_format_overrides"]
    assert [candidate.key for candidate in candidates] == ["linear_attn.layer_0"]
    assert overrides["linear_attn.layer_0"] == {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.0.linear_attn.in_proj_z": "MXFP8_E4M3",
    }
    assert payload["base_bpp"] > 0.0


def test_serving_unit_grouping_keeps_fused_siblings_atomic_not_layer_wide():
    assignment = {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.0.linear_attn.in_proj_z": "MXFP8_E4M3",
        "model.layers.0.linear_attn.out_proj": "NVFP4",
        "model.layers.1.self_attn.q_proj": "NVFP4",
        "model.layers.1.self_attn.k_proj": "NVFP4",
        "model.layers.1.self_attn.v_proj": "MXFP8_E4M3",
        "model.layers.1.self_attn.o_proj": "NVFP4",
        "model.layers.2.mlp.shared_expert.gate_proj": "NVFP4",
        "model.layers.2.mlp.shared_expert.up_proj": "NVFP4",
        "model.layers.2.mlp.shared_expert.down_proj": "MXFP8_E4M3",
    }
    stats = {name: _stats((64, 64)) for name in assignment}
    costs = {
        name: {
            fmt: {"output_mse": float(idx + 1), "weight_mse": 0.01}
        }
        for idx, (name, fmt) in enumerate(assignment.items())
    }

    payload = build_promotion_candidate_report(
        assignment,
        costs=costs,
        stats=stats,
        categories=["linear_attn", "self_attn", "shared_expert"],
        target_format="BF16",
        group_by="serving_unit",
        profile=_FakeFusedProfile(),
    )

    by_key = {candidate.key: tuple(candidate.members)
              for candidate in payload["candidates"]}
    assert by_key["fused:model.layers.0.linear_attn.in_proj_qkvz"] == (
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.in_proj_z",
    )
    assert by_key["tensor:model.layers.0.linear_attn.out_proj"] == (
        "model.layers.0.linear_attn.out_proj",
    )
    assert by_key["fused:model.layers.1.self_attn.qkv_proj"] == (
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.v_proj",
    )
    assert by_key["tensor:model.layers.1.self_attn.o_proj"] == (
        "model.layers.1.self_attn.o_proj",
    )
    assert by_key["fused:model.layers.2.mlp.shared_expert.gate_up_proj"] == (
        "model.layers.2.mlp.shared_expert.gate_proj",
        "model.layers.2.mlp.shared_expert.up_proj",
    )
    assert by_key["tensor:model.layers.2.mlp.shared_expert.down_proj"] == (
        "model.layers.2.mlp.shared_expert.down_proj",
    )

    overrides = payload["current_format_overrides"]
    assert overrides["fused:model.layers.0.linear_attn.in_proj_qkvz"] == {
        "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
        "model.layers.0.linear_attn.in_proj_z": "MXFP8_E4M3",
    }
    assert "linear_attn.layer_0" not in by_key


def test_layer_config_from_assignment_writes_autoround_entries():
    layer_config = layer_config_from_assignment({
        "model.layers.0.linear_attn.in_proj_qkv": "BF16",
        "model.layers.1.self_attn.q_proj": "NVFP4",
    })

    assert layer_config["model.layers.0.linear_attn.in_proj_qkv"]["bits"] == 16
    assert layer_config["model.layers.1.self_attn.q_proj"]["data_type"] == "nv_fp"


def test_propagated_sensitivity_penalty_counts_fused_unit_once():
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "NVFP4",
    }
    stats = {
        name: {**_stats((64, 64)), "h_trace": 10.0}
        for name in assignment
    }
    costs = {
        name: {
            "NVFP4": {
                "output_mse": 0.2,
                "output_mse_measured": True,
                "weight_mse": 0.1,
            },
            "MXFP8_E4M3": {
                "output_mse": 0.05,
                "output_mse_measured": True,
                "weight_mse": 0.01,
            },
            "BF16": {
                "output_mse": 0.0,
                "output_mse_measured": False,
                "weight_mse": 0.0,
                "predicted_dloss": 0.0,
            },
        }
        for name in assignment
    }
    report = build_promotion_candidate_report(
        assignment,
        costs=costs,
        stats=stats,
        categories=["self_attn"],
        target_format="BF16",
        group_by="serving_unit",
        profile=_FakeFusedProfile(),
    )
    row = report["candidates"][0].to_json()
    row["candidate_lane_override"] = report["current_format_overrides"][row["key"]]
    row["propagated_kl"] = 0.6
    sensitivity_report = {
        "target_format": "BF16",
        "rows": [row],
    }

    adjusted, summary = apply_propagated_sensitivity_penalty(
        costs,
        stats=stats,
        report=sensitivity_report,
        scale=1.0,
    )

    assert summary["adjusted_entries"] == 4
    assert summary["total_scaled_member_penalty"] == pytest.approx(0.75)
    assert summary["total_scaled_current_format_penalty"] == pytest.approx(0.6)
    assert summary["max_current_format_penalty_abs_error"] == pytest.approx(0.0)
    current_format_penalty = 0.0
    for name in assignment:
        nvfp4 = adjusted[name]["NVFP4"]
        mxfp8 = adjusted[name]["MXFP8_E4M3"]
        bf16 = adjusted[name]["BF16"]
        assert nvfp4["propagated_kl_penalty"] == pytest.approx(0.3)
        assert nvfp4["propagated_serving_unit_uses_output_mse"] is True
        assert nvfp4["output_mse"] == pytest.approx(0.26)
        assert mxfp8["propagated_kl_penalty"] == pytest.approx(0.075)
        assert mxfp8["propagated_serving_unit_uses_output_mse"] is True
        assert mxfp8["output_mse"] == pytest.approx(0.065)
        assert bf16["output_mse"] == 0.0
        current_format_penalty += nvfp4["propagated_kl_penalty"]
    assert current_format_penalty == pytest.approx(0.6)


def test_propagated_sensitivity_penalty_preserves_unmeasured_output_mse_path():
    stats = {"expert.pack": {**_stats((64, 64)), "h_trace": 10.0}}
    costs = {
        "expert.pack": {
            "NVFP4": {
                "output_mse": 0.0,
                "output_mse_measured": False,
                "weight_mse": 0.1,
                "predicted_dloss": 0.5,
            },
            "BF16": {
                "output_mse": 0.0,
                "output_mse_measured": False,
                "weight_mse": 0.0,
                "predicted_dloss": 0.0,
            },
        }
    }
    report = {
        "target_format": "BF16",
        "rows": [{
            "key": "tensor:expert.pack",
            "members": ["expert.pack"],
            "candidate_lane_override": {"expert.pack": "NVFP4"},
            "propagated_kl": 0.25,
            "bits_delta": 4096.0,
        }],
    }

    adjusted, summary = apply_propagated_sensitivity_penalty(
        costs,
        stats=stats,
        report=report,
        scale=2.0,
        format_extrapolation="bits_interp",
    )

    entry = adjusted["expert.pack"]["NVFP4"]
    assert summary["total_scaled_current_format_penalty"] == pytest.approx(0.5)
    assert entry["propagated_serving_unit_uses_output_mse"] is False
    assert entry["output_mse_measured"] is False
    assert entry["output_mse"] == 0.0
    assert entry["predicted_dloss"] == pytest.approx(1.0)


def test_propagated_sensitivity_current_only_extrapolation_prices_current_format():
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "NVFP4",
    }
    stats = {
        name: {**_stats((64, 64)), "h_trace": 10.0}
        for name in assignment
    }
    costs = {
        name: {
            "NVFP4": {
                "output_mse": 0.2,
                "output_mse_measured": True,
                "weight_mse": 0.1,
            },
            "MXFP8_E4M3": {
                "output_mse": 0.05,
                "output_mse_measured": True,
                "weight_mse": 0.01,
            },
            "BF16": {"predicted_dloss": 0.0, "weight_mse": 0.0},
        }
        for name in assignment
    }
    report = build_promotion_candidate_report(
        assignment,
        costs=costs,
        stats=stats,
        categories=["self_attn"],
        target_format="BF16",
        group_by="serving_unit",
        profile=_FakeFusedProfile(),
    )
    row = report["candidates"][0].to_json()
    row["candidate_lane_override"] = report["current_format_overrides"][row["key"]]
    row["propagated_kl"] = 0.6

    adjusted, summary = apply_propagated_sensitivity_penalty(
        costs,
        stats=stats,
        report={"target_format": "BF16", "rows": [row]},
        scale=1.0,
        format_extrapolation="current_only",
    )

    assert summary["adjusted_entries"] == 2
    for name in assignment:
        assert adjusted[name]["NVFP4"]["propagated_kl_penalty"] == pytest.approx(0.3)
        assert "propagated_kl_penalty" not in adjusted[name]["MXFP8_E4M3"]


def test_mse_promotion_skips_packed_expert_stats():
    assignment = {
        "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
        "model.layers.0.mlp.shared_expert.down_proj": "NVFP4",
    }
    stats = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            "out_features": 64,
            "in_features": 64,
            "n_params": 8 * 64 * 64,
            "num_experts": 8,
            "_packed_param": True,
        },
        "model.layers.0.mlp.shared_expert.down_proj": _stats((64, 64)),
    }
    costs = {
        name: {"NVFP4": {"output_mse": 1.0, "weight_mse": 0.01}}
        for name in assignment
    }

    payload = build_promotion_candidate_report(
        assignment,
        costs=costs,
        stats=stats,
        categories=["routed_experts", "shared_expert"],
        target_format="BF16",
        group_by="name",
    )

    assert [candidate.key for candidate in payload["candidates"]] == [
        "model.layers.0.mlp.shared_expert.down_proj"
    ]
