from __future__ import annotations

import pytest

from prismaquant.allocator import (
    _pareto_knee_summary,
    kneedle,
    kneedle_log_error,
    kneedle_raw_linear,
)


def test_allocator_default_kneedle_uses_log_error_on_ordered_curve():
    achieved = [4.501, 4.550, 4.600, 4.650, 4.700, 4.751, 4.851, 5.001]
    dloss = [487.33, 84.30, 28.74, 12.45, 5.24, 1.73, 0.0977, 0.0614]

    raw_idx = kneedle_raw_linear(achieved, dloss)
    log_idx = kneedle_log_error(achieved, dloss)

    assert achieved[raw_idx] == pytest.approx(4.600)
    assert achieved[log_idx] == pytest.approx(4.851)
    assert kneedle(achieved, dloss) == log_idx


def test_allocator_pareto_knee_summary_reports_both_modes():
    curve = [
        {"target_bits": x, "achieved_bits": x, "predicted_dloss": y, "feasible": True}
        for x, y in [
            (4.501, 487.33),
            (4.550, 84.30),
            (4.600, 28.74),
            (4.650, 12.45),
            (4.700, 5.24),
            (4.751, 1.73),
            (4.851, 0.0977),
            (5.001, 0.0614),
        ]
    ]

    summary = _pareto_knee_summary(curve)

    assert summary["primary"] == "log_error"
    assert summary["log_error"]["achieved_bits"] == pytest.approx(4.851)
    assert summary["raw_linear"]["achieved_bits"] == pytest.approx(4.600)

