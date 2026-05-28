from __future__ import annotations

import pytest

from prismaquant.allocator import (
    _pareto_knee_summary,
    kneedle,
    kneedle_log_error_global,
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
    assert achieved[kneedle_log_error_global(achieved, dloss)] == pytest.approx(4.851)
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
    assert summary["global_log_error"]["achieved_bits"] == pytest.approx(4.851)
    assert summary["raw_linear"]["achieved_bits"] == pytest.approx(4.600)
    assert summary["log_error"]["kneedle_error_source"] == "predicted_dloss"


def test_allocator_pareto_knee_summary_uses_body_loss_with_auxiliary_costs():
    achieved = [
        4.501,
        4.550,
        4.600,
        4.650,
        4.700,
        4.713,
        4.800,
        4.826,
        4.906,
        4.976,
        5.126,
        5.213,
        5.211,
        5.502,
    ]
    variable_dloss = [
        492.7,
        87.5,
        33.1,
        17.2,
        10.24,
        9.23,
        5.87,
        4.44,
        2.31,
        1.12,
        0.87,
        0.59,
        0.59,
        0.23,
    ]
    fixed_dloss = 858.0
    curve = [
        {
            "target_bits": x,
            "achieved_bits": x,
            "predicted_dloss": y,
            "variable_predicted_dloss": y,
            "aux_fixed_predicted_dloss": fixed_dloss,
            "total_predicted_dloss_with_aux": y + fixed_dloss,
            "feasible": True,
        }
        for x, y in zip(achieved, variable_dloss)
    ]

    summary = _pareto_knee_summary(curve)

    assert summary["log_error"]["achieved_bits"] == pytest.approx(4.976)
    assert summary["log_error"]["kneedle_dloss"] == pytest.approx(1.12)
    assert summary["log_error"]["predicted_dloss"] == pytest.approx(1.12)
    assert summary["log_error"]["total_predicted_dloss_with_aux"] == pytest.approx(859.12)
    assert summary["log_error"]["kneedle_error_source"] == "predicted_dloss"


def test_allocator_log_knee_ignores_catastrophic_prefix():
    curve = [
        {"target_bits": x, "achieved_bits": b, "predicted_dloss": y, "feasible": True}
        for x, b, y in [
            (4.50, 4.501442602523453, 490.95972004462783),
            (4.55, 4.551172022807636, 67.35287259481342),
            (4.60, 4.601404057010391, 18.683437952014543),
            (4.65, 4.6518201202141585, 10.703462196719535),
            (4.70, 4.665439625439821, 9.547101648912008),
            (4.75, 4.750897175358707, 6.219027463373),
            (4.80, 4.777108667901863, 5.0935395090179005),
            (4.85, 4.857494411172436, 2.7732721621546785),
            (4.90, 4.901644246117178, 1.6603677761045836),
            (4.95, 4.951453040800616, 1.160422153623775),
            (5.00, 4.965169589384272, 1.0826194275764642),
            (5.05, 5.020449916928711, 0.9037446622433383),
            (5.10, 5.043613378174432, 0.7936928507197837),
            (5.15, 5.1170068348969, 0.6833838506105339),
            (5.20, 5.200734593755301, 0.5815067413494683),
            (5.25, 5.2094799128403855, 0.5682548239957342),
            (5.31, 5.213559811328141, 0.5650757980483958),
            (5.40, 5.342056906007663, 0.41237890772033614),
            (5.50, 5.507399895834692, 0.22296933781706152),
        ]
    ]

    summary = _pareto_knee_summary(curve)

    assert summary["global_log_error"]["target_bits"] == pytest.approx(4.70)
    assert summary["log_error"]["target_bits"] == pytest.approx(5.00)
    assert summary["log_error"]["achieved_bits"] == pytest.approx(4.965169589384272)
