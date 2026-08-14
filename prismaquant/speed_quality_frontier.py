"""The speed/quality axis: selecting between W4A4 and W4A8 without a weight.

WHY THIS IS A SEPARATE MODULE
-----------------------------
``format_cost_protocol`` deliberately refuses to invent a speed/quality
scalarization constant -- it returns the two axes separately and says the choice
between them is a frontier selection, not a weighted sum. This module is that
selection. Keeping it out of the coster is the point: the coster must stay a
statement about *what things cost*, and the lever is a statement about *what we
are willing to trade*, which is a policy the operator sets.

WHAT THE LEVER IS, AND WHAT IT IS NOT
-------------------------------------
It is NOT ``alpha * quality + (1 - alpha) * speed``. A scalarizing weight is the
same object as the Lagrangian ``lambda`` the project already rejected as a
SELECTOR: the discrete frontier has non-convex pockets that no weight can ever
select, so a weighted sum cannot even express some of the points an operator
would want. It also silently changes which point it picks when the *units* of
either axis change, which makes it unreproducible across models.

The lever is a CONSTRAINT, and it comes in the two directions an operator
actually thinks in:

* "I need at least this much throughput -- give me the best quality that clears
  it."  -> :func:`select_by_speed_floor`
* "I will accept at most this much predicted loss -- give me the fastest thing
  inside it."  -> :func:`select_by_quality_ceiling`

Both return a point that is ON the empirical Pareto frontier, so neither can
pick something another candidate dominates outright.

WHERE THIS SITS RELATIVE TO THE BYTE BUDGET
-------------------------------------------
The existing allocator already selects under a byte budget ("fit the card").
This frontier is the axis that opens up *at a fixed byte budget*: W4A4 and W4A8
can cost nearly the same bytes -- the weights are 4-bit either way -- while
differing sharply in both throughput and loss. That is exactly why the A-side
had to be priced before this module could mean anything: with the activation
term missing or mis-priced, every W4Ax candidate at one weight width collapses
to the same predicted_dloss and the frontier is a single degenerate point.

STATUS -- READ THIS BEFORE QUOTING A FRONTIER
---------------------------------------------
``speed_index`` is a DECLARED per-format hint, and the aggregate below composes
those hints. It is a **proxy, not a serve measurement**: it knows nothing about
kernel dispatch, memory-bandwidth limits, batch shape, or the fact that a mixed
assignment may fall off a fused kernel entirely. A frontier built from it is a
research artifact for *ranking candidates*, and the ranking still has to survive
a served A/B before any point on it ships. The machinery is honest; the numbers
flowing through it are only as good as the hints, and today's hints are
declarations rather than measurements.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from .format_cost_protocol import CostComponents


@dataclasses.dataclass(frozen=True)
class AllocationPoint:
    """One whole-model candidate assignment, on both axes at once."""

    label: str
    #: Summed predicted loss delta. Lower is better.
    predicted_dloss: float
    #: Params-weighted aggregate throughput proxy. Higher is better. ``None``
    #: when any unit in the assignment declared no speed hint -- an unknown
    #: speed must not read as a fast one.
    speed_index: float | None
    memory_bytes: int
    n_params: int
    n_units: int
    #: Units that quantize activations but whose activation cost came back
    #: ``None``. Their A-side is UNPRICED, so ``predicted_dloss`` is a lower
    #: bound rather than an estimate, and comparing such a point against a fully
    #: priced one flatters it. Kept explicit so a consumer can refuse.
    unpriced_activation_units: int = 0

    @property
    def bits_per_param(self) -> float:
        if self.n_params <= 0:
            return 0.0
        return 8.0 * self.memory_bytes / self.n_params

    @property
    def fully_priced(self) -> bool:
        return self.unpriced_activation_units == 0


def aggregate_speed_index(components: Sequence[CostComponents]) -> float | None:
    """Compose per-unit throughput hints into one whole-model proxy.

    Throughput does NOT average arithmetically. If a unit's ``speed_index`` is
    a relative rate, the time it takes is proportional to ``n_params /
    speed_index``, and the aggregate rate is total work over total time::

        speed = sum(n_params) / sum(n_params / speed_index)

    -- the params-weighted HARMONIC mean. Using the arithmetic mean instead
    would let a handful of fast units hide a slow one, which is backwards: in a
    serial stack the slow unit is exactly the one that sets the pace.

    Returns ``None`` if any unit declares no hint, because an assignment whose
    speed is partly unknown has no defensible aggregate.
    """
    total_params = 0
    total_time = 0.0
    for c in components:
        if c.speed_index is None or c.speed_index <= 0.0:
            return None
        params = int(c.n_params)
        if params <= 0:
            continue
        total_params += params
        total_time += params / float(c.speed_index)
    if total_params <= 0 or total_time <= 0.0:
        return None
    return total_params / total_time


def allocation_point(label: str,
                     components: Sequence[CostComponents]) -> AllocationPoint:
    """Collapse a whole assignment's per-unit costs onto the two axes."""
    unpriced = sum(1 for c in components
                   if c.quantizes_activations and c.act_dloss is None)
    return AllocationPoint(
        label=label,
        predicted_dloss=float(sum(c.to_predicted_dloss() for c in components)),
        speed_index=aggregate_speed_index(components),
        memory_bytes=int(sum(c.memory_bytes for c in components)),
        n_params=int(sum(c.n_params for c in components)),
        n_units=len(components),
        unpriced_activation_units=unpriced,
    )


def pareto_frontier(points: Iterable[AllocationPoint],
                    ) -> list[AllocationPoint]:
    """The non-dominated subset: minimize dloss, maximize speed.

    A point is dominated when another is at least as good on BOTH axes and
    strictly better on one. Points with an unknown aggregate speed are dropped:
    they cannot be placed on the frontier, and defaulting them to fast or slow
    would be a fabricated number either way.

    Candidates equal on BOTH axes are all kept, not deduplicated: neither
    dominates the other (``strictly_better`` is false), so both survive. That is
    intentional -- two assignments can price identically and still differ in
    ways this module cannot see -- but it means the frontier may contain
    duplicates on the two axes, and a consumer that needs a unique answer must
    say which one it wants. The selection functions here do not: ``min``/``max``
    break such a tie arbitrarily.
    """
    known = [p for p in points if p.speed_index is not None]
    frontier: list[AllocationPoint] = []
    for cand in sorted(known, key=lambda p: (p.predicted_dloss,
                                             -(p.speed_index or 0.0))):
        dominated = False
        for keep in frontier:
            assert keep.speed_index is not None and cand.speed_index is not None
            at_least_as_good = (keep.predicted_dloss <= cand.predicted_dloss
                                and keep.speed_index >= cand.speed_index)
            strictly_better = (keep.predicted_dloss < cand.predicted_dloss
                               or keep.speed_index > cand.speed_index)
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(cand)
    return frontier


def select_by_speed_floor(points: Iterable[AllocationPoint],
                          min_speed: float) -> AllocationPoint | None:
    """Best predicted quality among candidates that clear a throughput floor.

    This is the "I need N tokens/s, spend everything else on quality" lever.
    Returns ``None`` when nothing clears the floor -- an empty answer, never a
    silent relaxation of the constraint the caller set.
    """
    eligible = [p for p in pareto_frontier(points)
                if p.speed_index is not None and p.speed_index >= min_speed]
    if not eligible:
        return None
    return min(eligible, key=lambda p: p.predicted_dloss)


def select_by_quality_ceiling(points: Iterable[AllocationPoint],
                              max_dloss: float) -> AllocationPoint | None:
    """Fastest candidate whose predicted loss stays inside a budget.

    This is the "I will tolerate this much degradation, buy me speed with it"
    lever. Returns ``None`` when nothing fits.
    """
    eligible = [p for p in pareto_frontier(points)
                if p.predicted_dloss <= max_dloss]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p.speed_index or 0.0)


def frontier_table(points: Iterable[AllocationPoint]) -> str:
    """A human-readable frontier dump, for putting in a report.

    Deliberately prints the unpriced-unit count: a frontier that looks good
    because part of its cost was never measured is the failure mode this
    project keeps re-learning, and it should be visible in the artifact rather
    than only in the code.
    """
    rows = pareto_frontier(points)
    if not rows:
        return "(no candidate carried a usable speed index)"
    width = max(len(p.label) for p in rows)
    out = [f"{'candidate'.ljust(width)}  {'dloss':>12}  {'speed':>9}  "
           f"{'bpp':>6}  {'GiB':>7}  unpriced"]
    for p in sorted(rows, key=lambda q: q.predicted_dloss):
        gib = p.memory_bytes / float(1 << 30)
        flag = "" if p.fully_priced else f" <-- {p.unpriced_activation_units}"
        out.append(
            f"{p.label.ljust(width)}  {p.predicted_dloss:12.6g}  "
            f"{p.speed_index or 0.0:9.3f}  {p.bits_per_param:6.3f}  "
            f"{gib:7.2f}  {p.unpriced_activation_units:>8}{flag}")
    return "\n".join(out)
