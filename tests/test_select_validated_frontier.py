from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from prismaquant.select_validated_frontier import (
    _saturation_pick,
    leave_one_out_kneedle_diagnostic,
    measured_frontier,
    practical_knee,
    select_frontier_point,
    spearman_rank_correlation,
    worst_rank_inversion,
)


def _sat_results(stderr):
    # flat tail (6.0..8.0 within noise of asymptote), decreasing before it
    rows = [(4.5, 0.10), (5.0, 0.06), (6.0, 0.030), (7.0, 0.029), (8.0, 0.028)]
    out = []
    for bpp, kl in rows:
        r = {"label": f"a{bpp}", "path": f"/x/a{bpp}.json", "bpp": bpp,
             "last_token_kl": kl, "format_counts": {}}
        if stderr is not None:
            r["kl_stderr"] = stderr
        out.append(r)
    return out


def test_saturation_mode_picks_bstar_with_real_stderr():
    sel, frontier = select_frontier_point(
        _sat_results(3e-3), mode="saturation", sat_z=2.0)
    assert sel["bpp"] == 6.0   # 6/7/8 indistinguishable within the band -> B*=6
    idx, sat = _saturation_pick(frontier, 2.0)
    assert sat["no_noise_floor"] is False
    assert frontier[idx]["bpp"] == 6.0


def test_saturation_mode_zero_stderr_flags_no_noise_floor():
    sel, frontier = select_frontier_point(
        _sat_results(0.0), mode="saturation", sat_z=2.0)
    assert sel["bpp"] == 8.0   # band collapses -> densest asymptote (most bits)
    _idx, sat = _saturation_pick(frontier, 2.0)
    assert sat["no_noise_floor"] is True


def test_saturation_mode_missing_stderr_key_is_no_noise_floor():
    # rows entirely lacking kl_stderr must not KeyError; treated as 0 stderr.
    sel, frontier = select_frontier_point(
        _sat_results(None), mode="saturation", sat_z=2.0)
    _idx, sat = _saturation_pick(frontier, 2.0)
    assert sat["no_noise_floor"] is True
    assert sel["bpp"] == 8.0


def test_saturation_single_point_frontier_does_not_crash():
    res = [{"label": "only", "path": "/x/only.json", "bpp": 6.0,
            "last_token_kl": 0.03, "kl_stderr": 1e-3}]
    sel, frontier = select_frontier_point(res, mode="saturation", sat_z=2.0)
    assert sel["bpp"] == 6.0 and len(frontier) == 1


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


def test_measured_frontier_can_use_ucb_metric():
    results = [
        {
            "label": "a",
            "path": "a.json",
            "bpp": 4.5,
            "last_token_kl": 0.10,
            "kl_ucb": 0.30,
        },
        {
            "label": "b",
            "path": "b.json",
            "bpp": 5.0,
            "last_token_kl": 0.12,
            "kl_ucb": 0.20,
        },
    ]

    frontier = measured_frontier(results, metric="ucb")

    assert [row["label"] for row in frontier] == ["a", "b"]
    assert frontier[0]["kl"] == 0.30
    assert frontier[1]["kl"] == 0.20


def test_practical_knee_picks_lowest_bpp_within_tolerance():
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 5.0, "kl": 0.101},
        {"label": "b", "path": "b.json", "bpp": 5.5, "kl": 0.100},
        {"label": "c", "path": "c.json", "bpp": 6.0, "kl": 0.090},
    ]

    selected = practical_knee(frontier, rel_eps=0.02)

    assert selected["label"] == "c"
    selected = practical_knee(frontier, rel_eps=0.13)
    assert selected["label"] == "a"


def test_select_frontier_reports_rank_and_leave_one_out_helpers():
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30, "surrogate_loss": 3.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20, "surrogate_loss": 2.0},
        {"label": "c", "path": "c.json", "bpp": 5.5, "kl": 0.10, "surrogate_loss": 1.0},
        {"label": "d", "path": "d.json", "bpp": 6.0, "kl": 0.09, "surrogate_loss": 0.5},
    ]

    assert spearman_rank_correlation(frontier) > 0.9
    diagnostic = leave_one_out_kneedle_diagnostic(
        frontier,
        frontier[1],
        tolerance_bpp=10.0,
        kl_noise_floor=10.0,
    )
    assert diagnostic["enabled"]
    assert diagnostic["stable"]


def test_measured_frontier_extracts_surrogate_from_nested_mse():
    # Real validate_assignments_kl rows carry the surrogate nested as
    # mse.predicted_dloss_sum, NOT a top-level surrogate_loss. This is the data
    # path that previously left surrogate_spearman silently None on every run.
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.30,
         "mse": {"predicted_dloss_sum": 3.0}},
        {"label": "b", "path": "b.json", "bpp": 5.0, "last_token_kl": 0.20,
         "mse": {"predicted_dloss_sum": 2.0}},
        {"label": "c", "path": "c.json", "bpp": 5.5, "last_token_kl": 0.10,
         "mse": {"predicted_dloss_sum": 1.0}},
        {"label": "d", "path": "d.json", "bpp": 6.0, "last_token_kl": 0.09,
         "mse": {"predicted_dloss_sum": 0.5}},
    ]
    frontier = measured_frontier(results)
    for row in frontier:
        assert row["surrogate_loss"] is not None
    corr = spearman_rank_correlation(frontier)
    assert corr is not None
    assert corr > 0.9


def test_measured_frontier_top_level_surrogate_loss_takes_precedence():
    # Backward compat: an explicit top-level surrogate_loss still wins.
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.30,
         "surrogate_loss": 9.0, "mse": {"predicted_dloss_sum": 3.0}},
    ]
    frontier = measured_frontier(results)
    assert frontier[0]["surrogate_loss"] == 9.0


def test_worst_rank_inversion_detects_mispredicted_pair():
    # 'a' is predicted best (lowest surrogate) but measured worst (highest KL).
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30, "surrogate_loss": 1.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20, "surrogate_loss": 2.0},
        {"label": "c", "path": "c.json", "bpp": 5.5, "kl": 0.10, "surrogate_loss": 3.0},
    ]
    inv = worst_rank_inversion(frontier)
    assert inv is not None
    # 'a' (lowest surrogate) is the predicted-best of the worst inverted pair.
    assert inv["predicted_best_label"] == "a"
    assert inv["predicted_worse_label"] == "c"
    assert inv["rank_gap"] > 0.0
    assert "measured KL was worse" in inv["verdict"]


def test_worst_rank_inversion_none_when_concordant():
    # Perfectly concordant surrogate/KL ordering -> no inversion.
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30, "surrogate_loss": 3.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20, "surrogate_loss": 2.0},
        {"label": "c", "path": "c.json", "bpp": 5.5, "kl": 0.10, "surrogate_loss": 1.0},
    ]
    assert worst_rank_inversion(frontier) is None


def test_worst_rank_inversion_none_when_too_few_pairs():
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30, "surrogate_loss": 1.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20, "surrogate_loss": 2.0},
    ]
    assert worst_rank_inversion(frontier) is None


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
            "--mode",
            "practical-knee",
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
