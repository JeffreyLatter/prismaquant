"""Knapsack bin discretization: strictly-positive bit deltas are never free.

Audit 2026-07-02 §3.3: ``int(round(d_avg_bits / bit_precision))`` charged 0
bins for any unit whose average-bit delta was below half a bin, handing tiny
units free one-directional format upgrades that the tightening loop could
never correct (the returned assignment violated the solver's own overshoot
tolerance). The charge is now conservative — minimum 1 bin for any strictly
positive delta (max over-charge: one bin = ``bit_precision`` avg bits per
unit) — and the DP backtrack mirrors the forward charge exactly.
"""
from __future__ import annotations

import itertools
import random

import pytest

from prismaquant.allocator_solver import (
    Candidate,
    _charged_bins,
    solve_allocation,
    solve_with_promotion,
)


def test_charged_bins_minimum_one_for_positive_delta():
    assert _charged_bins(0.0, 0.001) == 0
    assert _charged_bins(0.00049, 0.001) == 1   # rounded to 0 before the fix
    assert _charged_bins(0.0006, 0.001) == 1
    assert _charged_bins(0.0026, 0.001) == 3


def test_dp_matches_bruteforce_on_toy_instances():
    """8 units x 3 formats, deltas exactly on the bin lattice: the DP must
    reproduce the exhaustive-search optimum (min total dloss s.t. avg bits
    <= target)."""
    rng = random.Random(1234)
    n_units = 8
    n_params = 1000
    bits_menu = (4.0, 8.0, 16.0)
    for _trial in range(20):
        stats = {}
        candidates = {}
        for i in range(n_units):
            name = f"u{i}"
            stats[name] = {"n_params": n_params}
            dl4 = rng.uniform(1.0, 10.0)
            dl8 = dl4 * rng.uniform(0.05, 0.5)
            candidates[name] = [
                Candidate("NVFP4", 4.0, n_params // 2, dl4),
                Candidate("FP8_E4M3", 8.0, n_params, dl8),
                Candidate("BF16", 16.0, 2 * n_params, 0.0),
            ]
        # Targets on the achievable 0.5-avg-bit lattice so the +2-bin
        # capacity slack cannot admit anything brute force would exclude.
        target = rng.choice([4.0 + 0.5 * k for k in range(2, 17)])

        result = solve_allocation(stats, candidates, target,
                                  bit_precision=0.001)
        assert result is not None
        _assign, chosen = result
        dp_bits = sum(
            chosen[n].bits_per_param for n in chosen) / n_units
        dp_dloss = sum(chosen[n].predicted_dloss for n in chosen)
        assert dp_bits <= target + 1e-9

        best = None
        for combo in itertools.product(range(3), repeat=n_units):
            avg_bits = sum(bits_menu[c] for c in combo) / n_units
            if avg_bits > target + 1e-9:
                continue
            total = sum(
                candidates[f"u{i}"][c].predicted_dloss
                for i, c in enumerate(combo)
            )
            if best is None or total < best:
                best = total
        assert best is not None
        assert dp_dloss == pytest.approx(best, rel=1e-12, abs=1e-12)


def test_sub_bin_upgrades_are_charged_not_free():
    """Audit repro: 500 tiny units whose 4->16 bpp delta rounds to 0 bins
    must not be upgraded for free; the returned assignment must respect the
    solver's own overshoot tolerance."""
    stats = {}
    candidates = {}
    for i in range(2):
        name = f"big{i}"
        stats[name] = {"n_params": 1_000_000}
        candidates[name] = [
            Candidate("NVFP4", 4.0, 500_000, 1.0),
            Candidate("BF16", 16.0, 2_000_000, 0.0),
        ]
    # Mid units are properly charged and attractive, so the DP fills the
    # budget to ~target with them; free tiny upgrades then push PAST it,
    # and tightening cannot remove a 0-bin charge.
    for i in range(40):
        name = f"mid{i}"
        stats[name] = {"n_params": 20_000}
        candidates[name] = [
            Candidate("NVFP4", 4.0, 10_000, 10.0),
            Candidate("BF16", 16.0, 40_000, 0.0),
        ]
    for i in range(500):
        name = f"tiny{i}"
        stats[name] = {"n_params": 80}
        candidates[name] = [
            # A very attractive upgrade: free-bin bugs always take it.
            Candidate("NVFP4", 4.0, 40, 100.0),
            Candidate("BF16", 16.0, 160, 0.0),
        ]
    total_params = 2 * 1_000_000 + 40 * 20_000 + 500 * 80
    d_avg_tiny = 12.0 * 80 / total_params
    # The scenario's premise: the tiny delta rounds to 0 bins ...
    assert int(round(d_avg_tiny / 0.001)) == 0
    # ... and the conservative charge prices it at 1 bin.
    assert _charged_bins(d_avg_tiny, 0.001) == 1

    overshoot_tolerance = 0.01
    format_rank = {"NVFP4": 0, "BF16": 1}

    # Un-correctable-before-the-fix regime: the free upgrades alone
    # (500 * d_avg_tiny ~ 0.169 avg bits) exceed the whole budget excess
    # (0.1), so no amount of tightening could pull the old solver back
    # under its own tolerance (free upgrades are invisible to the budget).
    target = 4.1
    assign, achieved = solve_with_promotion(
        stats, candidates, target, {}, format_rank, 0.001,
        overshoot_tolerance=overshoot_tolerance,
    )
    assert 500 * d_avg_tiny > (target - 4.0) + overshoot_tolerance
    assert assign is not None
    assert achieved <= target + overshoot_tolerance

    # Loose-budget regime: charging the tiny upgrades must not forbid them
    # either — they are the best gain-per-bin in the menu and all fit.
    target = 4.5
    assign, achieved = solve_with_promotion(
        stats, candidates, target, {}, format_rank, 0.001,
        overshoot_tolerance=overshoot_tolerance,
    )
    assert assign is not None
    assert achieved <= target + overshoot_tolerance
    upgraded = sum(1 for n, f in assign.items()
                   if n.startswith("tiny") and f == "BF16")
    assert upgraded == 500
