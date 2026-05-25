from __future__ import annotations

import pytest

from prismaquant.mse_promotion import (
    build_mse_promotion_assignment,
    layer_config_from_assignment,
)


def _stats(shape):
    out_features, in_features = shape
    return {
        "out_features": out_features,
        "in_features": in_features,
        "n_params": out_features * in_features,
    }


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


def test_layer_config_from_assignment_writes_autoround_entries():
    layer_config = layer_config_from_assignment({
        "model.layers.0.linear_attn.in_proj_qkv": "BF16",
        "model.layers.1.self_attn.q_proj": "NVFP4",
    })

    assert layer_config["model.layers.0.linear_attn.in_proj_qkv"]["bits"] == 16
    assert layer_config["model.layers.1.self_attn.q_proj"]["data_type"] == "nv_fp"
