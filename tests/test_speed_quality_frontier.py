"""The speed/quality lever: a constraint on a frontier, never a weighted sum.

The load-bearing property is `test_frontier_keeps_a_point_no_weight_can_select`:
a scalarizing weight cannot reach points in a non-convex pocket, which is the
same reason Lagrangian lambda-bisection was rejected as a selector. If that test
ever fails, someone has quietly turned the lever back into a weight.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismaquant.format_cost_protocol import CostComponents, CostModel, RenderBasis
from prismaquant.speed_quality_frontier import (
    AllocationPoint,
    aggregate_speed_index,
    allocation_point,
    frontier_table,
    pareto_frontier,
    select_by_quality_ceiling,
    select_by_speed_floor,
)


def _pt(label: str, dloss: float, speed: float | None,
        *, params: int = 1000, mem: int = 500,
        unpriced: int = 0) -> AllocationPoint:
    return AllocationPoint(label=label, predicted_dloss=dloss, speed_index=speed,
                           memory_bytes=mem, n_params=params, n_units=1,
                           unpriced_activation_units=unpriced)


def _cc(name: str, *, dloss: float, speed: float | None, params: int,
        bits: float = 4.0, act: float | None = None,
        model: CostModel = CostModel.AQUA,
        quantizes_activations: bool = True) -> CostComponents:
    return CostComponents(
        unit_name=name, format_name="F", model=model,
        render_basis=RenderBasis.RTN, weight_mse=1e-4, weight_dloss=dloss,
        act_dloss=act, bits_per_param=bits,
        memory_bytes=int(params * bits / 8), n_params=params,
        speed_index=speed, quantizes_activations=quantizes_activations)


# ------------------------------------------------------------ the speed axis


def test_aggregate_speed_is_a_params_weighted_harmonic_mean():
    """Throughput composes as total work over total time, not arithmetically.

    A big slow unit next to a small fast one must land near the slow one; an
    arithmetic mean would hide it.
    """
    comps = [_cc("slow", dloss=0.0, speed=1.0, params=9000),
             _cc("fast", dloss=0.0, speed=100.0, params=1000)]
    got = aggregate_speed_index(comps)
    want = 10000 / (9000 / 1.0 + 1000 / 100.0)
    assert got == pytest.approx(want)
    # The arithmetic mean would be ~10.9x higher -- the fixture is not vacuous.
    arithmetic = (9000 * 1.0 + 1000 * 100.0) / 10000
    assert arithmetic > 9.0 * got


def test_unknown_speed_never_reads_as_fast():
    """One unit without a hint makes the whole aggregate unknown, not optimistic."""
    comps = [_cc("a", dloss=0.0, speed=10.0, params=100),
             _cc("b", dloss=0.0, speed=None, params=100)]
    assert aggregate_speed_index(comps) is None
    pt = allocation_point("mixed", comps)
    assert pt.speed_index is None
    # And such a point cannot silently win a selection.
    assert select_by_speed_floor([pt], 0.0) is None


def test_allocation_point_counts_unpriced_activation_units():
    """An unmeasured A-side must be visible, not absorbed into the total."""
    comps = [_cc("priced", dloss=1.0, speed=2.0, params=100, act=0.5),
             _cc("unpriced", dloss=1.0, speed=2.0, params=100, act=None)]
    pt = allocation_point("half", comps)
    assert pt.unpriced_activation_units == 1
    assert pt.fully_priced is False
    # The total is a LOWER BOUND: it omits the missing term rather than guessing.
    assert pt.predicted_dloss == pytest.approx(2.5)


def test_a_weight_only_format_is_not_counted_as_unpriced():
    """W4A16 leaves activations alone, so act_dloss=None is CORRECT there.

    This is the case that caught an inference bug: deriving "does this format
    quantize activations" from the cost model marked every NVFP4A16 candidate
    as having a measurement hole, which would have made a weight-only format
    look untrustworthy next to a W4A4 one on the frontier. The predicate is
    carried as data on CostComponents for exactly this reason.
    """
    weight_only = [_cc("a16", dloss=1.0, speed=2.0, params=100, act=None,
                       quantizes_activations=False)]
    assert allocation_point("w4a16", weight_only).unpriced_activation_units == 0
    assert allocation_point("w4a16", weight_only).fully_priced

    # ...while a format that DOES quantize activations and came back unpriced
    # is still flagged. Same act_dloss=None, opposite meaning.
    w4a4 = [_cc("a4", dloss=1.0, speed=2.0, params=100, act=None,
                quantizes_activations=True)]
    assert allocation_point("w4a4", w4a4).unpriced_activation_units == 1


# --------------------------------------------------------------- the frontier


def test_pareto_frontier_drops_dominated_points():
    pts = [_pt("good", 1.0, 10.0),
           _pt("dominated", 2.0, 5.0),      # worse on both
           _pt("fast", 3.0, 20.0)]
    got = {p.label for p in pareto_frontier(pts)}
    assert got == {"good", "fast"}


def test_points_without_speed_are_excluded_from_the_frontier():
    pts = [_pt("known", 1.0, 10.0), _pt("unknown", 0.5, None)]
    assert [p.label for p in pareto_frontier(pts)] == ["known"]


def test_frontier_keeps_a_point_no_weight_can_select():
    """THE property: a constraint reaches points a scalarizing weight cannot.

    Three non-dominated points where the middle one lies ABOVE the line joining
    its neighbours in (dloss, -speed) space, i.e. in a non-convex pocket. No
    alpha in `alpha*dloss - (1-alpha)*speed` ever minimizes at the middle point,
    but a speed floor selects it directly. This is the same geometry that
    retired Lagrangian lambda-bisection as a selector.
    """
    a, b, c = _pt("a", 1.0, 1.0), _pt("b", 2.0, 3.0), _pt("c", 3.0, 10.0)
    pts = [a, b, c]
    assert len(pareto_frontier(pts)) == 3          # all three non-dominated

    # No scalarizing weight picks b.
    picked = set()
    for alpha in np.linspace(0.0, 1.0, 2001):
        scores = [alpha * p.predicted_dloss - (1.0 - alpha) * (p.speed_index or 0.0)
                  for p in pts]
        picked.add(pts[int(np.argmin(scores))].label)
    assert "b" not in picked, f"expected a non-convex pocket, weights reached {picked}"

    # The constraint form reaches it without difficulty.
    assert select_by_speed_floor(pts, 3.0).label == "b"
    assert select_by_quality_ceiling(pts, 2.0).label == "b"


# ------------------------------------------------------------------ selection


def test_speed_floor_picks_best_quality_that_clears_it():
    pts = [_pt("slow_best", 1.0, 1.0), _pt("mid", 2.0, 5.0), _pt("fast", 4.0, 9.0)]
    assert select_by_speed_floor(pts, 5.0).label == "mid"
    assert select_by_speed_floor(pts, 0.0).label == "slow_best"


def test_quality_ceiling_picks_the_fastest_that_fits():
    pts = [_pt("slow_best", 1.0, 1.0), _pt("mid", 2.0, 5.0), _pt("fast", 4.0, 9.0)]
    assert select_by_quality_ceiling(pts, 2.5).label == "mid"
    assert select_by_quality_ceiling(pts, 99.0).label == "fast"


def test_an_unsatisfiable_constraint_returns_none_not_a_relaxation():
    """The lever must never quietly hand back something outside the budget."""
    pts = [_pt("a", 1.0, 1.0), _pt("b", 2.0, 3.0)]
    assert select_by_speed_floor(pts, 1e6) is None
    assert select_by_quality_ceiling(pts, 1e-9) is None


def test_bits_per_param_is_derived_not_assumed():
    pt = _pt("x", 1.0, 1.0, params=1024, mem=512)
    assert pt.bits_per_param == pytest.approx(4.0)
    assert _pt("empty", 1.0, 1.0, params=0, mem=0).bits_per_param == 0.0


def test_frontier_table_surfaces_unpriced_units():
    """A frontier flattered by missing measurement must say so in the artifact."""
    pts = [_pt("full", 1.0, 5.0), _pt("partial", 0.5, 6.0, unpriced=3)]
    text = frontier_table(pts)
    assert "partial" in text and "<-- 3" in text
    assert frontier_table([_pt("n", 1.0, None)]).startswith("(no candidate")
