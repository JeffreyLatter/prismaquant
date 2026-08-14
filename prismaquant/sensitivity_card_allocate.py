"""Bridge a Sensitivity Card + a format menu into the existing allocator DP.

This is the last piece of "probe once, price any menu". The optimizer is not
modified: :func:`candidates_from_card` produces exactly the
``dict[name, list[Candidate]]`` that ``allocator_solver.solve_allocation``
already consumes, so an arbitrary format menu enters as an arbitrary list of
plugins.

The weights are supplied by a caller-provided lookup rather than loaded here.
A consumer quantizing a model already has its weights; the card supplies the
sensitivity, which is the part they cannot compute. Keeping the load out of this
module is what lets the same code price a menu from a shared card without ever
touching our calibration.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .allocator_solver import Candidate, solve_allocation
from .format_cost_protocol import CostComponents, CostModel, FormatCostPlugin, price
from .sensitivity_card import RenderBasis, SensitivityCard, SensitivityUnit

WeightLookup = Callable[[str], np.ndarray]


def price_unit_menu(unit: SensitivityUnit, weight: np.ndarray,
                    plugins: Sequence[FormatCostPlugin], *,
                    render_basis: RenderBasis,
                    model: CostModel = CostModel.MARGINAL,
                    ) -> list[CostComponents]:
    """Price one unit against every format in the menu.

    Illegal formats (passthrough on a mismatched source dtype) are dropped, not
    penalized. A format is never dropped for looking risky.
    """
    out = []
    for plugin in plugins:
        c = price(unit, weight, plugin, render_basis=render_basis, model=model)
        if c is not None:
            out.append(c)
    return out


def candidates_from_card(
    card: SensitivityCard,
    weights: WeightLookup,
    plugins: Sequence[FormatCostPlugin],
    *,
    model: CostModel = CostModel.MARGINAL,
    names: Iterable[str] | None = None,
) -> tuple[dict[str, dict], dict[str, list[Candidate]]]:
    """Build the ``(stats, candidates)`` pair ``solve_allocation`` expects.

    Returns ``stats`` carrying only what the solver reads (``n_params``), and
    ``candidates`` keyed by unit name.

    A unit with no legal format is omitted from both, so the solver never sees
    an empty option list -- which it would otherwise treat as a zero-cost free
    choice.
    """
    stats: dict[str, dict] = {}
    candidates: dict[str, list[Candidate]] = {}

    for name in (names if names is not None else card.names()):
        unit = card[name]
        priced = price_unit_menu(unit, weights(name), plugins,
                                 render_basis=card.provenance.render_basis,
                                 model=model)
        if not priced:
            continue
        stats[name] = {"n_params": unit.n_params,
                       "in_features": unit.in_features,
                       "out_features": unit.out_features}
        candidates[name] = [
            Candidate(fmt=c.format_name,
                      bits_per_param=c.bits_per_param,
                      memory_bytes=c.memory_bytes,
                      predicted_dloss=c.to_predicted_dloss())
            for c in priced
        ]
    return stats, candidates


def allocate_from_card(
    card: SensitivityCard,
    weights: WeightLookup,
    plugins: Sequence[FormatCostPlugin],
    *,
    target_bits: float,
    model: CostModel = CostModel.MARGINAL,
    names: Iterable[str] | None = None,
):
    """Card + menu + budget -> per-unit format assignment.

    Returns ``None`` when the budget is below what the cheapest legal menu can
    reach, which is ``solve_allocation``'s own contract for an infeasible target.
    """
    stats, candidates = candidates_from_card(
        card, weights, plugins, model=model, names=names)
    if not candidates:
        return None
    return solve_allocation(stats, candidates, target_bits)


def assignment_summary(assignment: Mapping[str, str]) -> dict[str, int]:
    """Format histogram -- the first thing to look at after an allocation."""
    counts: dict[str, int] = {}
    for fmt in assignment.values():
        counts[fmt] = counts.get(fmt, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
