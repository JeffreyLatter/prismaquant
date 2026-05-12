from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from prismaquant.select_validated_frontier import (
    measured_frontier,
    select_frontier_point,
)


def test_measured_frontier_drops_dominated_points():
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.20},
        {"label": "b", "path": "b.json", "bpp": 4.6, "last_token_kl": 0.30},
        {"label": "c", "path": "c.json", "bpp": 5.0, "last_token_kl": 0.10},
        {"label": "d", "path": "d.json", "bpp": 5.5, "last_token_kl": 0.09},
    ]

    frontier = measured_frontier(results)

    assert [row["label"] for row in frontier] == ["a", "c", "d"]


def test_select_frontier_best_kl():
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.20},
        {"label": "b", "path": "b.json", "bpp": 5.0, "last_token_kl": 0.10},
        {"label": "c", "path": "c.json", "bpp": 5.5, "last_token_kl": 0.11},
    ]

    selected, frontier = select_frontier_point(results, mode="best-kl")

    assert selected["label"] == "b"
    assert [row["label"] for row in frontier] == ["a", "b"]


def test_select_validated_frontier_cli_writes_layer_config(tmp_path):
    assignment_path = tmp_path / "candidate.json"
    assignment_path.write_text(json.dumps({
        "schema": "prismaquant.allocator.pareto_assignment.v1",
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.mlp.down_proj": "MXFP8_E4M3",
            "model.layers.1.mlp.down_proj": "BF16",
        },
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate",
            "path": str(assignment_path),
            "bpp": 5.0,
            "last_token_kl": 0.01,
            "format_counts": {"NVFP4": 1, "MXFP8_E4M3": 1, "BF16": 1},
        }],
    }))
    layer_config = tmp_path / "layer_config.json"
    assignment_out = tmp_path / "selected_assignment.json"
    summary = tmp_path / "selection.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--output-layer-config",
            str(layer_config),
            "--output-assignment",
            str(assignment_out),
            "--output-summary",
            str(summary),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    payload = json.loads(layer_config.read_text())
    assert set(payload) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    }
    assert payload["model.layers.0.self_attn.q_proj"]["data_type"] == "nv_fp"
    assert payload["model.layers.0.mlp.down_proj"]["data_type"] == "mx_fp"
    assert payload["model.layers.1.mlp.down_proj"]["data_type"] == "float"

    selected = json.loads(summary.read_text())["selected"]
    assert selected["label"] == "candidate"
