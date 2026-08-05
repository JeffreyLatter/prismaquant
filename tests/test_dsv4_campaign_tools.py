from __future__ import annotations

import math

from tools.dsv4_ldlq_burn import (
    ANCHORS,
    EXPERT_COUNT,
    HOLDOUTS,
    MEASURED_RUNGS,
    _fit_slices,
)


def test_dual_holdout_fit_accepts_exact_production_law():
    errors = {
        k: [3.0 * 2.0 ** (-k / 4.0) for _ in range(EXPERT_COUNT)]
        for k in MEASURED_RUNGS
    }
    accepted, laws, report = _fit_slices(errors)
    assert len(accepted) == EXPERT_COUNT
    assert len(laws) == EXPERT_COUNT
    assert report["rejected"] == 0
    assert set(report["holdouts"]) == {f"K{k}" for k in HOLDOUTS}


def test_dual_holdout_fit_rejects_a_bad_second_holdout():
    errors = {
        k: [2.0 * 2.0 ** (-k / 4.0) for _ in range(EXPERT_COUNT)]
        for k in MEASURED_RUNGS
    }
    errors[HOLDOUTS[1]][17] *= 1.5
    accepted, laws, report = _fit_slices(errors)
    assert 17 not in accepted
    assert 17 not in laws
    assert report["rejected"] == 1
    assert math.isfinite(report["holdouts"][f"K{HOLDOUTS[1]}"]["p95"])
    assert tuple(sorted(ANCHORS)) == ANCHORS
