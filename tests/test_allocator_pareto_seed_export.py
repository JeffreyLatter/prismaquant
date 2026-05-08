from __future__ import annotations

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
    ]
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
        assert payload["schema"] == "prismaquant.allocator.pareto_assignment.v1"
        assignment = payload["assignment"]
        assert set(assignment) == set(names)
        assert all(".__siblings__." not in name for name in assignment)
        assert len(payload["label"]) > len("allocator_target_")
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
