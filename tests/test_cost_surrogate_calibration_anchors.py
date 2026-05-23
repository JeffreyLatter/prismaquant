from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


ANCHOR_PATH = Path(__file__).parent / "fixtures" / "cost_surrogate_calibration_anchors.json"


def _load_anchors() -> dict:
    with ANCHOR_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rel_err(observed: float, expected: float) -> float:
    return abs(float(observed) - float(expected)) / max(abs(float(expected)), 1e-12)


def test_cost_surrogate_anchor_fixture_is_actionable():
    payload = _load_anchors()
    assert payload["schema"] == "prismaquant.cost_surrogate_calibration_anchors.v1"
    anchors = payload["anchors"]
    assert anchors
    assert any(
        "predicted_dloss" in anchor["expected"]
        and "ppl_wikitext" in anchor["expected"]
        for anchor in anchors
    )
    assert any("end_kl" in anchor["expected"] for anchor in anchors)

    names = set()
    for anchor in anchors:
        assert anchor["name"] not in names
        names.add(anchor["name"])
        assert anchor["model"]
        assert Path(anchor["source_doc"]).exists()
        assert anchor["target_bpp"] > 0
        assert anchor["achieved_bpp"] > 0
        assert anchor["formats"]
        assert anchor["expected"]
        assert anchor["tolerance"]
        assert anchor["logs"]
        for metric, expected in anchor["expected"].items():
            assert isinstance(expected, int | float)
            tolerance_key = f"{metric}_rel"
            assert 0 < anchor["tolerance"][tolerance_key] < 0.1


def test_cost_surrogate_anchor_opt_in_metrics_match_expected():
    current_path = os.environ.get("PRISMAQUANT_COST_ANCHOR_RESULTS")
    if not current_path:
        pytest.skip("set PRISMAQUANT_COST_ANCHOR_RESULTS to compare a fresh run")

    anchors = {a["name"]: a for a in _load_anchors()["anchors"]}
    with Path(current_path).open("r", encoding="utf-8") as f:
        current = json.load(f)

    for name, metrics in current.items():
        anchor = anchors[name]
        for metric, expected in anchor["expected"].items():
            if metric not in metrics:
                continue
            rel = _rel_err(metrics[metric], expected)
            assert rel <= anchor["tolerance"][f"{metric}_rel"], (
                f"{name}.{metric} drifted by {rel:.4%}: "
                f"observed={metrics[metric]} expected={expected}"
            )
