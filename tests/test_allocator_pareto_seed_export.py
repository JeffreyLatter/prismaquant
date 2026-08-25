from __future__ import annotations

import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path


def test_allocator_exports_expanded_pareto_seed_assignments(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    }))

    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "mtp.layers.0.mlp.down_proj",
        "lm_head",
    ]
    expected_assignment_names = set(names) - {"lm_head"}
    stats = {}
    costs = {}
    for idx, name in enumerate(names):
        stats[name] = {
            "h_trace": float(idx + 1),
            "n_params": 128 * 128,
            "in_features": 128,
            "out_features": 128,
        }
        costs[name] = {
            "NVFP4": {"predicted_dloss": 10.0 + idx},
            "MXFP8_E4M3": {"predicted_dloss": 1.0 + 0.1 * idx},
            "BF16": {"predicted_dloss": 0.0},
        }

    probe_path = tmp_path / "probe.pkl"
    cost_path = tmp_path / "cost.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats, "meta": {"model": str(model_dir)}}, f)
    with open(cost_path, "wb") as f:
        pickle.dump({
            "costs": costs,
            "formats": ["NVFP4", "MXFP8_E4M3", "BF16"],
        }, f)

    out_dir = tmp_path / "pareto_seeds"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.allocator",
            "--probe",
            str(probe_path),
            "--costs",
            str(cost_path),
            "--model-override",
            str(model_dir),
            "--formats",
            "NVFP4,MXFP8_E4M3,BF16",
            "--target-bits",
            "8.0",
            "--pareto-targets",
            "4.6,8.0,16.0",
            "--bit-precision",
            "0.1",
            "--layer-config",
            str(tmp_path / "layer_config.json"),
            "--pareto-csv",
            str(tmp_path / "pareto.csv"),
            "--pareto-output-dir",
            str(out_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema"] == "prismaquant.allocator.pareto_manifest.v1"
    assert len(manifest["candidates"]) >= 2

    saw_mxfp8 = False
    for row in manifest["candidates"]:
        payload = json.loads(Path(row["path"]).read_text())
        assert payload["schema"] == "prismaquant.allocator.pareto_assignment.v2"
        assignment = payload["assignment"]
        assert set(assignment) == expected_assignment_names
        assert all(".__siblings__." not in name for name in assignment)
        assert len(payload["label"]) > len("allocator_target_")
        assert assignment["mtp.layers.0.mlp.down_proj"] == "BF16"
        assert "lm_head" not in assignment
        saw_mxfp8 = saw_mxfp8 or "MXFP8_E4M3" in assignment.values()

        qkv_formats = {
            assignment[name]
            for name in names
            if name.endswith((".q_proj", ".k_proj", ".v_proj"))
        }
        gate_up_formats = {
            assignment[name]
            for name in names
            if name.endswith((".gate_proj", ".up_proj"))
        }
        assert len(qkv_formats) == 1
        assert len(gate_up_formats) == 1

    assert saw_mxfp8


def test_allocator_excludes_fixed_quantized_mtp_from_body_budget(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    }))

    names = [
        "model.layers.0.mlp.down_proj",
        "mtp.layers.0.mlp.down_proj",
    ]
    stats = {
        name: {
            "h_trace": 1.0,
            "n_params": 128 * 128,
            "in_features": 128,
            "out_features": 128,
        }
        for name in names
    }
    costs = {
        "model.layers.0.mlp.down_proj": {
            "NVFP4": {"predicted_dloss": 100.0},
            "BF16": {"predicted_dloss": 0.0},
        },
        "mtp.layers.0.mlp.down_proj": {
            "NVFP4": {"predicted_dloss": 7.0},
            "BF16": {"predicted_dloss": 0.0},
        },
    }

    probe_path = tmp_path / "probe.pkl"
    cost_path = tmp_path / "cost.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats, "meta": {"model": str(model_dir)}}, f)
    with open(cost_path, "wb") as f:
        pickle.dump({"costs": costs, "formats": ["NVFP4", "BF16"]}, f)

    pareto_path = tmp_path / "pareto.csv"
    layer_config_path = tmp_path / "layer_config.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.allocator",
            "--probe",
            str(probe_path),
            "--costs",
            str(cost_path),
            "--model-override",
            str(model_dir),
            "--target-profile",
            "research",
            "--formats",
            "NVFP4,BF16",
            "--mtp-format",
            "NVFP4",
            "--target-bits",
            "10.5",
            "--pareto-targets",
            "10.5",
            "--bit-precision",
            "0.1",
            "--layer-config",
            str(layer_config_path),
            "--pareto-csv",
            str(pareto_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    with open(pareto_path, newline="") as f:
        row = next(csv.DictReader(f))

    assert row["feasible"] == "True"
    # Exact assignment payload accounting includes the two fp32 global-scale
    # scalars emitted for each NVFP4 Linear (8 B / 16,384 params here).
    assert float(row["achieved_bits"]) == 4.50390625
    assert float(row["predicted_dloss"]) == 100.0
    assert float(row["aux_fixed_predicted_dloss"]) == 7.0
    assert float(row["total_predicted_dloss_with_aux"]) == 107.0
    assert int(row["layers_NVFP4"]) == 1
    assert int(row.get("layers_BF16") or 0) == 0

    layer_config = json.loads(layer_config_path.read_text())
    assert layer_config["mtp.layers.0.mlp.down_proj"]["data_type"] == "nv_fp"
