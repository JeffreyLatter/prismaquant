from __future__ import annotations

import pytest

from prismaquant.allocator import (
    _pareto_knee_summary,
    _rd_curve_diagnostic,
    kneedle,
    kneedle_log_error_global,
    kneedle_log_error,
    kneedle_raw_linear,
    refine_knee_golden,
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
    assert summary["diagnostic_thresholds"] == {
        "tail_min_log_span_decades": 1.0,
        "tail_midpoint_fraction": 0.5,
    }


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


# --------------------------------------------------------------------------
# RD-curve log-linearity diagnostic (kneedle demotion)
# --------------------------------------------------------------------------
def test_rd_curve_diagnostic_log_linear_has_no_intrinsic_knee():
    # dloss = 10^(-0.3*bpp): exactly straight in (bpp, log10 dloss).
    feasible = [
        {"achieved_bits": b, "predicted_dloss": 10.0 ** (-0.3 * b), "feasible": True}
        for b in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
    ]
    rd = _rd_curve_diagnostic(feasible)
    assert rd["available"] is True
    assert rd["r2"] == pytest.approx(1.0, abs=1e-9)
    assert rd["log_linear"] is True
    assert rd["intrinsic_knee"] is False
    assert rd["slope_decades_per_bit"] == pytest.approx(-0.3, abs=1e-9)
    assert rd["diagnostic_thresholds"]["log_linear_r2"] == pytest.approx(0.99)


def test_rd_curve_diagnostic_flags_curvature():
    # Start log-linear, then inject a fat outlier on the middle point so the
    # least-squares R^2 drops below 0.99 -> a curvature knee may be meaningful.
    feasible = [
        {"achieved_bits": b, "predicted_dloss": 10.0 ** (-0.3 * b), "feasible": True}
        for b in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
    ]
    feasible[3]["predicted_dloss"] *= 5.0  # ~0.7-decade kink at 6.0 bpp
    rd = _rd_curve_diagnostic(feasible)
    assert rd["available"] is True
    assert rd["r2"] < 0.99
    assert rd["log_linear"] is False
    assert rd["intrinsic_knee"] is True


def test_rd_curve_diagnostic_too_few_points():
    rd = _rd_curve_diagnostic([
        {"achieved_bits": 5.0, "predicted_dloss": 0.04, "feasible": True},
        {"achieved_bits": 6.0, "predicted_dloss": 0.02, "feasible": True},
    ])
    assert rd["available"] is False


# --------------------------------------------------------------------------
# Golden-section knee refinement (reuses the sub-second DP between grid rungs)
# --------------------------------------------------------------------------
def test_refine_knee_golden_pins_knee_inside_bracket():
    # A genuine convex knee around 5.0 bpp (steep below, flat above) on a coarse
    # 0.25-bpp grid; the refiner re-solves the DP at interior budgets.
    def f(bpp):
        return 100.0 * (10.0 ** (-1.5 * bpp)) + 0.05 * (10.0 ** (-0.1 * bpp))

    grid_targets = [4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0]
    curve = [
        {"target_bits": t, "achieved_bits": t, "predicted_dloss": f(t),
         "feasible": True}
        for t in grid_targets
    ]
    summary = _pareto_knee_summary(curve)
    assert summary["enabled"]

    def solve_fn(target):
        # mimic _solve_for_target's 4-tuple: (assign, achieved, dloss, mutable)
        return ({"x": "NVFP4"}, float(target), f(float(target)), f(float(target)))

    refined, extra = refine_knee_golden(solve_fn, summary, curve, tol=0.01)
    assert refined is not None
    lo, hi = refined["bracket_target_bits"]
    assert lo <= refined["target_bits"] <= hi          # stays inside the bracket
    assert refined["evals"] > 0
    assert refined["mode"] == "log_error_golden_refined"
    # every refinement sample re-solved a real interior target inside the bracket
    assert all(lo <= pt["target_bits"] <= hi for pt in extra)
    assert refined["predicted_dloss"] > 0.0

    # Convergence: the golden section must land on the actual max-dip target, not
    # just somewhere in the bracket. Brute-force the perpendicular dip below the
    # bracket chord in (bpp, log10 dloss) space over a fine grid and compare.
    import math
    xL, yL = lo, math.log10(f(lo))
    xH, yH = hi, math.log10(f(hi))
    def _dip(t):
        x, y = t, math.log10(f(t))
        ychord = yL + (yH - yL) * (x - xL) / (xH - xL)
        return ychord - y
    brute = max((lo + (hi - lo) * i / 2000.0 for i in range(2001)), key=_dip)
    assert refined["target_bits"] == pytest.approx(brute, abs=0.02)  # ~2*tol


def test_refine_knee_golden_noop_when_disabled():
    curve = [
        {"target_bits": t, "achieved_bits": t, "predicted_dloss": 1.0, "feasible": True}
        for t in [4.5, 5.0, 5.5]
    ]
    refined, extra = refine_knee_golden(
        lambda t: (None, float("nan"), float("inf"), float("inf")),
        {"enabled": False}, curve,
    )
    assert refined is None and extra == []
