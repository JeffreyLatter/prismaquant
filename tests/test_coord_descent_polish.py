"""Tests for prismaquant.coord_descent_polish.

Coord descent is wrapped around the model + measure_assignment_kl primitive,
so we test it with a stub-KL function that doesn't need a real model.
"""
from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import patch

import pytest

from prismaquant import block_clado as bc
from prismaquant import coord_descent_polish as cdp


def _mk_unit(name: str, members: tuple[str, ...], options):
    opts = []
    for fmt, omega_ii, bits_per_param, memory_bytes in options:
        opts.append(bc.FormatCost(
            fmt=fmt,
            omega_ii=float(omega_ii),
            bits_per_param=float(bits_per_param),
            memory_bytes=int(memory_bytes),
        ))
    opts.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
    return bc.DecisionUnit(
        name=name,
        block_id=name,
        member_qnames=members,
        options=tuple(opts),
    )


def _stub_run(units, starting, kl_table):
    """Run coord descent against a stubbed measure_assignment_kl."""
    def fake_measure(model, assignment, calib_ids, ref_log_probs, **kwargs):
        # Build a frozen tuple over fused-group choices for lookup.
        key_parts = []
        for unit in units:
            fmt = None
            for member in unit.member_qnames:
                if member in assignment:
                    fmt = assignment[member]
                    break
            key_parts.append((unit.name, fmt))
        return float(kl_table[tuple(key_parts)])

    with patch("prismaquant.coord_descent_polish.measure_assignment_kl",
               side_effect=fake_measure):
        return cdp.coord_descent_polish(
            model=object(),
            calib_ids=object(),
            ref_log_probs=object(),
            units=units,
            starting_assignment=starting,
            work_root="/tmp",
            noise_floor=0.0,
            max_passes=8,
        )


def test_polish_accepts_strict_improvement():
    unit_a = _mk_unit("a", ("a.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    unit_b = _mk_unit("b", ("b.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.2, 4, 256)])
    units = [unit_a, unit_b]

    # KL surface: accepting a→LO improves; b stays at BF16 (no improvement).
    kl_table = {
        (("a", "BF16"), ("b", "BF16")): 1.0,
        (("a", "LO"),   ("b", "BF16")): 0.4,
        (("a", "BF16"), ("b", "LO")):   0.9,
        (("a", "LO"),   ("b", "LO")):   0.5,
    }
    starting = {"a.weight": "BF16", "b.weight": "BF16"}
    result = _stub_run(units, starting, kl_table)

    assert result.initial_kl == pytest.approx(1.0)
    assert result.final_kl == pytest.approx(0.4)
    assert result.final_assignment == {"a.weight": "LO", "b.weight": "BF16"}
    assert len(result.steps) == 1
    assert result.steps[0].unit == "a"
    assert result.steps[0].from_fmt == "BF16"
    assert result.steps[0].to_fmt == "LO"


def test_polish_picks_largest_improvement_per_pass():
    unit_a = _mk_unit("a", ("a.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    unit_b = _mk_unit("b", ("b.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.2, 4, 256)])
    units = [unit_a, unit_b]

    # Both flips help, but b helps more (1.0 → 0.3 vs 1.0 → 0.5).
    # Greedy-best should pick b first.
    kl_table = {
        (("a", "BF16"), ("b", "BF16")): 1.0,
        (("a", "LO"),   ("b", "BF16")): 0.5,
        (("a", "BF16"), ("b", "LO")):   0.3,
        (("a", "LO"),   ("b", "LO")):   0.2,
    }
    starting = {"a.weight": "BF16", "b.weight": "BF16"}
    result = _stub_run(units, starting, kl_table)

    assert result.final_kl == pytest.approx(0.2)
    accepted_units = [s.unit for s in result.steps]
    # b accepted first (largest single-step improvement), then a.
    assert accepted_units == ["b", "a"]


def test_polish_terminates_when_no_improvement_exists():
    unit = _mk_unit("a", ("a.weight",),
                    [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    # Starting state is the global minimum.
    kl_table = {
        (("a", "BF16"),): 0.5,
        (("a", "LO"),):   1.0,
    }
    starting = {"a.weight": "BF16"}
    result = _stub_run([unit], starting, kl_table)

    assert result.initial_kl == pytest.approx(0.5)
    assert result.final_kl == pytest.approx(0.5)
    assert result.final_assignment == {"a.weight": "BF16"}
    assert result.steps == []


def test_polish_respects_noise_floor():
    """A move that improves by less than the noise floor should be rejected."""
    unit = _mk_unit("a", ("a.weight",),
                    [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    # Tiny improvement (1e-6) — below default noise floor (1e-5).
    kl_table = {
        (("a", "BF16"),): 0.500001,
        (("a", "LO"),):   0.500000,
    }
    starting = {"a.weight": "BF16"}
    units = [unit]

    def fake_measure(model, assignment, calib_ids, ref_log_probs, **kwargs):
        for member in unit.member_qnames:
            if member in assignment:
                return float(kl_table[(("a", assignment[member]),)])
        raise KeyError(unit.member_qnames)

    with patch("prismaquant.coord_descent_polish.measure_assignment_kl",
               side_effect=fake_measure):
        result = cdp.coord_descent_polish(
            model=object(), calib_ids=object(), ref_log_probs=object(),
            units=units, starting_assignment=starting,
            work_root="/tmp",
            noise_floor=1e-5, max_passes=4,
        )
    # Improvement is 1e-6 < 1e-5 noise floor → reject.
    assert result.steps == []
    assert result.final_kl == pytest.approx(0.500001)


def test_polish_fused_sibling_flips_all_members():
    """All members of a fused group must move together."""
    unit = _mk_unit("group", ("a.weight", "b.weight", "c.weight"),
                    [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    kl_table = {
        (("group", "BF16"),): 1.0,
        (("group", "LO"),):   0.3,
    }
    starting = {"a.weight": "BF16", "b.weight": "BF16", "c.weight": "BF16"}
    result = _stub_run([unit], starting, kl_table)
    assert result.final_assignment == {
        "a.weight": "LO", "b.weight": "LO", "c.weight": "LO",
    }


def test_polish_records_full_trace():
    unit_a = _mk_unit("a", ("a.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    units = [unit_a]
    kl_table = {
        (("a", "BF16"),): 1.0,
        (("a", "LO"),):   0.5,
    }
    starting = {"a.weight": "BF16"}
    result = _stub_run(units, starting, kl_table)

    # 1 starting + 1 trial (pass 0) + 1 confirmation trial (pass 1, finds
    # nothing new to flip) = 3 total.
    assert result.n_kl_measurements == 3
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.kl_before == pytest.approx(1.0)
    assert step.kl_after == pytest.approx(0.5)
    assert step.candidates_evaluated == 1
