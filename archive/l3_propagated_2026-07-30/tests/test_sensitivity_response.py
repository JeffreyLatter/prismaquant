from __future__ import annotations

import pytest

from prismaquant.sensitivity_response import build_response_report


def test_response_report_quantifies_unit_and_category_sensitivity():
    payload = {
        "schema": "prismaquant.kl_sensitivity_probe.v2",
        "git_commit": "abc123",
        "floor": {"kl": 0.10},
        "rows": [
            {
                "qname": "model.layers.0.self_attn.q_proj",
                "decision_unit": "model.layers.0.self_attn.qkv_proj",
                "format": "NVFP4",
                "shape": [4, 4],
                "bits_delta": 0.0,
                "candidate_kl": 0.10,
                "sensitivity": 0.0,
            },
            {
                "qname": "model.layers.0.self_attn.k_proj",
                "decision_unit": "model.layers.0.self_attn.qkv_proj",
                "format": "NVFP4",
                "shape": [4, 4],
                "bits_delta": 0.0,
                "candidate_kl": 0.10,
                "sensitivity": 0.0,
            },
            {
                "qname": "model.layers.0.self_attn.q_proj",
                "decision_unit": "model.layers.0.self_attn.qkv_proj",
                "format": "BF16",
                "shape": [4, 4],
                "bits_delta": 192.0,
                "candidate_kl": 0.07,
                "sensitivity": 0.015,
            },
            {
                "qname": "model.layers.0.self_attn.k_proj",
                "decision_unit": "model.layers.0.self_attn.qkv_proj",
                "format": "BF16",
                "shape": [4, 4],
                "bits_delta": 192.0,
                "candidate_kl": 0.07,
                "sensitivity": 0.015,
            },
            {
                "qname": "model.layers.1.mlp.shared_expert.down_proj",
                "decision_unit": "model.layers.1.mlp.shared_expert.down_proj",
                "format": "MXFP8_E4M3",
                "shape": [4, 8],
                "bits_delta": 0.0,
                "candidate_kl": 0.10,
                "sensitivity": 0.0,
            },
            {
                "qname": "model.layers.1.mlp.shared_expert.down_proj",
                "decision_unit": "model.layers.1.mlp.shared_expert.down_proj",
                "format": "BF16",
                "shape": [4, 8],
                "bits_delta": 256.0,
                "candidate_kl": 0.11,
                "sensitivity": -0.01,
            },
        ],
    }
    costs = {
        "model.layers.0.self_attn.q_proj": {
            "NVFP4": {"output_mse": 0.20, "predicted_dloss": 0.004}
        },
        "model.layers.0.self_attn.k_proj": {
            "NVFP4": {"output_mse": 0.10, "predicted_dloss": 0.002}
        },
        "model.layers.1.mlp.shared_expert.down_proj": {
            "MXFP8_E4M3": {"output_mse": 0.05, "predicted_dloss": 0.001}
        },
    }

    report = build_response_report(payload, costs=costs, target_format="BF16")

    assert report["unit_count"] == 2
    self_attn = report["units"][0]
    assert self_attn["decision_unit"] == "model.layers.0.self_attn.qkv_proj"
    assert self_attn["category"] == "self_attn"
    assert self_attn["member_count"] == 2
    assert self_attn["kl_gain"] == pytest.approx(0.03)
    assert self_attn["row_sensitivity_sum"] == pytest.approx(0.03)
    assert self_attn["output_mse_sum"] == pytest.approx(0.30)
    assert self_attn["predicted_dloss_sum"] == pytest.approx(0.006)
    assert self_attn["kl_gain_per_output_mse"] == pytest.approx(0.10)
    assert self_attn["kl_gain_per_predicted_dloss"] == pytest.approx(5.0)

    categories = {row["category"]: row for row in report["categories"]}
    assert categories["self_attn"]["positive_kl_gain_sum"] == pytest.approx(0.03)
    assert categories["shared_expert"]["negative_kl_gain_sum"] == pytest.approx(-0.01)
    assert categories["self_attn"]["positive_gain_enrichment_vs_output_mse"] > 1.0
