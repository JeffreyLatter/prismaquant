from __future__ import annotations

import json
import pathlib
import pickle

import pytest

# These tests exercise repo-root `tools/` scripts, which are not part of the
# installed package, so they can only run from a checkout. The release
# pipeline runs the suite against a non-editable install from outside the
# checkout (docs/RELEASING.md); skip there instead of failing collection.
if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from tools.build_sensitivity_ablation_assignments import main


def test_builds_baseline_category_and_group_overlays(tmp_path):
    base = tmp_path / "layer_config.json"
    probe = tmp_path / "probe.pkl"
    costs = tmp_path / "cost.pkl"
    output = tmp_path / "ablation"

    base.write_text(json.dumps({
        "model.layers.0.mlp.shared_expert.gate_proj": "NVFP4",
        "model.layers.0.mlp.shared_expert.up_proj": "MXFP8_E4M3",
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.1.linear_attn.out_proj": "BF16",
    }))
    stats = {
        "model.layers.0.mlp.shared_expert.gate_proj": {
            "h_trace": 2.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        },
        "model.layers.0.mlp.shared_expert.up_proj": {
            "h_trace": 4.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        },
        "model.layers.0.self_attn.q_proj": {
            "h_trace": 3.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        },
        "model.layers.1.linear_attn.out_proj": {
            "h_trace": 1.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        },
    }
    with probe.open("wb") as fh:
        pickle.dump({"stats": stats}, fh)
    with costs.open("wb") as fh:
        pickle.dump({
            "formats": ["NVFP4", "MXFP8_E4M3", "BF16"],
            "costs": {
                "model.layers.0.mlp.shared_expert.gate_proj": {
                    "NVFP4": {"predicted_dloss": 0.10},
                },
                "model.layers.0.mlp.shared_expert.up_proj": {
                    "MXFP8_E4M3": {"predicted_dloss": 0.05},
                },
                "model.layers.0.self_attn.q_proj": {
                    "NVFP4": {"predicted_dloss": 0.20},
                },
                "model.layers.1.linear_attn.out_proj": {
                    "BF16": {"weight_mse": 0.0},
                },
            },
        }, fh)

    rc = main([
        "--base-assignment", str(base),
        "--probe", str(probe),
        "--costs", str(costs),
        "--output-dir", str(output),
        "--categories", "shared_expert,self_attn,linear_attn",
        "--top-groups-per-category", "1",
    ])

    assert rc == 0
    manifest = json.loads((output / "manifest.json").read_text())
    labels = {row["label"] for row in manifest["variants"]}
    assert "baseline" in labels
    assert "promote_all_shared_expert_to_bf16" in labels
    assert "promote_shared_expert_layer_0_to_bf16" in labels

    shared_overlay = json.loads(
        (output / "assignments" / "promote_all_shared_expert_to_bf16.json")
        .read_text()
    )
    assert shared_overlay == {
        "model.layers.0.mlp.shared_expert.gate_proj": "BF16",
        "model.layers.0.mlp.shared_expert.up_proj": "BF16",
    }
    shared = next(
        row for row in manifest["variants"]
        if row["label"] == "promote_all_shared_expert_to_bf16"
    )
    assert shared["non_bf16_count"] == 2
    assert shared["predicted_dloss_saved"] == pytest.approx(0.15)
