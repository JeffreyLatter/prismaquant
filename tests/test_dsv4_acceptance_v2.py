from __future__ import annotations

from tools.dsv4_afast_burn import (
    ANCHORS,
    BACKSTOP_TOLERANCE,
    EXPERT_COUNT,
    _acceptance,
    _audit_rung,
    _audit_stats,
)


def _linear_anchor_errors() -> dict[int, list[float]]:
    return {
        rung: [float(100 - rung)] * EXPERT_COUNT
        for rung in ANCHORS
    }


def test_v2_audit_draw_is_layer_deterministic() -> None:
    assert _audit_rung(0) == 32
    assert _audit_rung(14) == 29
    assert _audit_rung(21) == 46
    assert _audit_rung(14) == _audit_rung(14)


def test_v2_backstop_only_rejects_gross_four_anchor_cv_outlier() -> None:
    errors = _linear_anchor_errors()
    errors[33][0] = 1.0

    accepted, rejected, fit = _acceptance(errors)

    assert rejected == [0]
    assert len(accepted) == EXPERT_COUNT - 1
    assert fit["backstop_failed"] == 1
    assert fit["backstop_tolerance"] == BACKSTOP_TOLERANCE
    assert fit["per_slice"][0]["K33_relative_error"] > BACKSTOP_TOLERANCE


def test_v2_audit_gate_scores_five_anchor_prediction() -> None:
    errors = _linear_anchor_errors()
    audit = _audit_stats(errors, [66.0] * EXPERT_COUNT, 34)

    assert audit["pass"] is True
    assert audit["median"] == 0.0
    assert audit["p95"] == 0.0
