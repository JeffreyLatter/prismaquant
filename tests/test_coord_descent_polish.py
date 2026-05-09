"""Tests for prismaquant.coord_descent_polish.

Coord descent is wrapped around the model + measure_assignment_kl primitive,
so we test it with a stub-KL function that doesn't need a real model.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from prismaquant import block_clado as bc
from prismaquant import coord_descent_polish as cdp
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.weight_session import WeightSession


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


def _stub_run(units, starting, kl_table, **kwargs):
    """Run coord descent against a stubbed measure_assignment_kl."""
    def fake_measure(model, assignment, calib_ids, ref_log_probs, **_):
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
            **kwargs,
        )


class _ToyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32, bias=False)


def test_weight_session_stages_reverts_and_commits(tmp_path):
    model = _ToyLinearModel()
    original = model.linear.weight.detach().clone()
    rendered = torch.full_like(original, 0.125)
    cache = ProductionWeightCache(
        weights={("linear", "NVFP4"): rendered.clone()},
        levers={},
    )
    unit = _mk_unit(
        "linear",
        ("linear",),
        [("BF16", 0.0, 16, original.numel() * 2),
         ("NVFP4", 0.1, 4.5, original.numel() // 2)],
    )

    session = WeightSession(
        model,
        production_weight_cache=cache,
        snapshot_dir=str(tmp_path),
    )
    session.initialize({"linear": "BF16"}, [unit])

    assert session.n_bf16_snapshots == 0
    assert not list(tmp_path.glob("*__bf16src.pt"))

    assert session.stage_format("linear", "NVFP4") is not None
    torch.testing.assert_close(model.linear.weight, rendered)
    assert session.n_bf16_snapshots == 1
    session.revert_last()
    torch.testing.assert_close(model.linear.weight, original)

    assert session.stage_format("linear", "NVFP4") is not None
    session.commit_last()
    torch.testing.assert_close(model.linear.weight, rendered)
    assert session.current_assignment()["linear"] == "NVFP4"


def test_delta_quantize_polish_restores_external_weight_env(monkeypatch, tmp_path):
    model = _ToyLinearModel()
    rendered = torch.full_like(model.linear.weight.detach(), 0.125)
    cache = ProductionWeightCache(
        weights={("linear", "NVFP4"): rendered},
        levers={},
    )
    unit = _mk_unit(
        "linear",
        ("linear",),
        [("BF16", 0.0, 16, 2048), ("NVFP4", 0.1, 4.5, 512)],
    )
    kl_table = {
        (("linear", "BF16"),): 1.0,
        (("linear", "NVFP4"),): 0.5,
    }
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "sentinel")

    def fake_measure(model, assignment, calib_ids, ref_log_probs, **_):
        return float(kl_table[(("linear", assignment["linear"]),)])

    with patch(
        "prismaquant.coord_descent_polish.measure_assignment_kl",
        side_effect=fake_measure,
    ):
        result = cdp.coord_descent_polish(
            model=model,
            calib_ids=object(),
            ref_log_probs=object(),
            units=[unit],
            starting_assignment={"linear": "BF16"},
            work_root=str(tmp_path),
            noise_floor=0.0,
            max_passes=2,
            production_weight_cache=cache,
            delta_quantize=True,
        )

    assert result.final_assignment == {"linear": "NVFP4"}
    torch.testing.assert_close(model.linear.weight, rendered)
    assert os.environ.get("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT") == "sentinel"


def test_delta_quantize_polish_can_restore_bf16_on_exit(tmp_path):
    model = _ToyLinearModel()
    original = model.linear.weight.detach().clone()
    rendered = torch.full_like(original, 0.125)
    cache = ProductionWeightCache(
        weights={("linear", "NVFP4"): rendered},
        levers={},
    )
    unit = _mk_unit(
        "linear",
        ("linear",),
        [("BF16", 0.0, 16, 2048), ("NVFP4", 0.1, 4.5, 512)],
    )

    def fake_measure(model, assignment, calib_ids, ref_log_probs, **_):
        return 0.5 if assignment["linear"] == "NVFP4" else 1.0

    with patch(
        "prismaquant.coord_descent_polish.measure_assignment_kl",
        side_effect=fake_measure,
    ):
        result = cdp.coord_descent_polish(
            model=model,
            calib_ids=object(),
            ref_log_probs=object(),
            units=[unit],
            starting_assignment={"linear": "BF16"},
            work_root=str(tmp_path),
            noise_floor=0.0,
            max_passes=2,
            production_weight_cache=cache,
            delta_quantize=True,
            weight_session_snapshot_dir=tmp_path / "snapshots",
            restore_bf16_on_exit=True,
        )

    assert result.final_assignment == {"linear": "NVFP4"}
    torch.testing.assert_close(model.linear.weight, original)


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


def test_polish_respects_bits_budget():
    """A KL-improving move that busts the budget must be rejected."""
    # 16-bit upgrade is the only KL-improver but exceeds the 4-bit budget.
    unit = _mk_unit("a", ("a.weight",),
                    [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    units = [unit]
    kl_table = {
        (("a", "BF16"),): 0.0,   # zero KL but 16 bits
        (("a", "LO"),):   0.1,   # higher KL but 4 bits
    }
    starting = {"a.weight": "LO"}

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
            noise_floor=0.0, max_passes=4,
            bits_budget=2048.0,  # = 4 bits × 512 = LO budget; BF16 (16384) busts it
        )
    # BF16 would lower KL (0.1 → 0.0) but exceeds budget — rejected.
    assert result.steps == []
    assert result.final_assignment == {"a.weight": "LO"}


def test_polish_unconstrained_when_no_budget():
    """When no budget, polish accepts even precision-upgrading moves."""
    unit = _mk_unit("a", ("a.weight",),
                    [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    units = [unit]
    kl_table = {
        (("a", "BF16"),): 0.0,
        (("a", "LO"),):   0.1,
    }
    starting = {"a.weight": "LO"}

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
            noise_floor=0.0, max_passes=4,
            bits_budget=None,  # explicitly unconstrained
        )
    # Without a budget, polish accepts the LO → BF16 upgrade.
    assert len(result.steps) == 1
    assert result.steps[0].to_fmt == "BF16"


def test_steepest_first_uses_surrogate_priority():
    """With steepest_first + pairs_by_block, accept the first surrogate-
    ranked move that improves real KL — don't sweep all candidates."""
    unit_a = _mk_unit("a", ("a.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    unit_b = _mk_unit("b", ("b.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.2, 4, 256)])
    units = [unit_a, unit_b]

    # Surrogate predicts b's flip is bigger improvement than a's:
    #   ΔΩ(a→LO) = 0.1 - 0.0 = 0.1
    #   ΔΩ(b→LO) = 0.2 - 0.0 = 0.2
    # but for steepest-first we want the MOST-NEGATIVE delta first, so
    # negate by using formats where LO is BETTER (omega_ii smaller).
    unit_a = _mk_unit("a", ("a.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", -0.5, 4, 256)])
    unit_b = _mk_unit("b", ("b.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", -0.1, 4, 256)])
    units = [unit_a, unit_b]
    # No pair info → ΔΩ(a→LO) = -0.5, ΔΩ(b→LO) = -0.1; a gets ranked first.
    pairs_by_block: dict[str, list] = {"a": [], "b": []}

    # Both flips actually improve real KL by the same amount; only the
    # FIRST surrogate-ranked move should be evaluated since it improves.
    kl_table = {
        (("a", "BF16"), ("b", "BF16")): 1.0,
        (("a", "LO"),   ("b", "BF16")): 0.5,
        (("a", "BF16"), ("b", "LO")):   0.5,
        (("a", "LO"),   ("b", "LO")):   0.0,
    }
    starting = {"a.weight": "BF16", "b.weight": "BF16"}

    def fake_measure(model, assignment, calib_ids, ref_log_probs, **kwargs):
        key_parts = []
        for unit in units:
            for member in unit.member_qnames:
                if member in assignment:
                    key_parts.append((unit.name, assignment[member]))
                    break
        return float(kl_table[tuple(key_parts)])

    with patch("prismaquant.coord_descent_polish.measure_assignment_kl",
               side_effect=fake_measure):
        result = cdp.coord_descent_polish(
            model=object(), calib_ids=object(), ref_log_probs=object(),
            units=units, starting_assignment=starting,
            work_root="/tmp", noise_floor=0.0, max_passes=4,
            pairs_by_block=pairs_by_block,
            steepest_first=True,
        )
    # Pass 0: a→LO ranked first (most negative ΔΩ), evaluated, accepts.
    # Pass 1: with current = (LO, BF16), now b→LO is the only remaining
    # change (a→BF16 worsens KL), evaluated, accepts.
    # Pass 2: no improving moves left, terminates.
    accepted = [s.unit for s in result.steps]
    assert accepted == ["a", "b"]
    assert result.final_assignment == {"a.weight": "LO", "b.weight": "LO"}


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


def test_top_down_decreases_bits_subject_to_kl_budget():
    """Top-down: starts at high-bpp/low-KL, drops bits while KL ≤ kl_budget."""
    unit_a = _mk_unit("a", ("a.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    unit_b = _mk_unit("b", ("b.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.2, 4, 256)])
    units = [unit_a, unit_b]

    # KL surface: starting at all-BF16 = 0.0 (lossless).  Each downgrade
    # raises KL.  With kl_budget=0.5 polish accepts both downgrades; with
    # kl_budget=0.4 it accepts only one.
    kl_table = {
        (("a", "BF16"), ("b", "BF16")): 0.0,
        (("a", "LO"),   ("b", "BF16")): 0.3,
        (("a", "BF16"), ("b", "LO")):   0.4,
        (("a", "LO"),   ("b", "LO")):   0.5,
    }
    starting = {"a.weight": "BF16", "b.weight": "BF16"}

    # kl_budget=0.5 → accept both: ends at all-LO with KL=0.5.
    result = _stub_run(
        units, starting, kl_table,
        direction="top_down", kl_budget=0.5,
    )
    assert result.final_kl == pytest.approx(0.5)
    assert result.final_assignment == {"a.weight": "LO", "b.weight": "LO"}
    assert len(result.steps) == 2

    # kl_budget=0.35 → only the cheaper downgrade is accepted.
    # Of {a→LO (KL 0.3), b→LO (KL 0.4)}, only a→LO is feasible.
    result = _stub_run(
        units, starting, kl_table,
        direction="top_down", kl_budget=0.35,
    )
    assert result.final_assignment["a.weight"] == "LO"
    assert result.final_assignment["b.weight"] == "BF16"
    assert result.final_kl == pytest.approx(0.3)


def test_top_down_picks_minimum_kl_among_bits_decreasing_moves():
    """Top-down should prefer the move with smallest trial_kl (= smallest
    quality cost) among bits-decreasing budget-respecting moves."""
    unit_a = _mk_unit("a", ("a.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    unit_b = _mk_unit("b", ("b.weight",),
                      [("BF16", 0.0, 16, 1024), ("LO", 0.2, 4, 256)])
    units = [unit_a, unit_b]

    # Both downgrades save the SAME bits (1024 - 256 = 768 each).  But
    # a→LO costs 0.1 KL and b→LO costs 0.4 KL.  Top-down should accept
    # a first (smaller KL increase).
    kl_table = {
        (("a", "BF16"), ("b", "BF16")): 0.0,
        (("a", "LO"),   ("b", "BF16")): 0.1,
        (("a", "BF16"), ("b", "LO")):   0.4,
        (("a", "LO"),   ("b", "LO")):   0.5,
    }
    starting = {"a.weight": "BF16", "b.weight": "BF16"}
    # _stub_run sets max_passes=8 by default; override in kwargs.
    def fake_measure(model, assignment, calib_ids, ref_log_probs, **_):
        key_parts = []
        for u in units:
            fmt = None
            for member in u.member_qnames:
                if member in assignment:
                    fmt = assignment[member]
                    break
            key_parts.append((u.name, fmt))
        return float(kl_table[tuple(key_parts)])
    with patch("prismaquant.coord_descent_polish.measure_assignment_kl",
               side_effect=fake_measure):
        result = cdp.coord_descent_polish(
            model=object(), calib_ids=object(), ref_log_probs=object(),
            units=units, starting_assignment=starting,
            work_root="/tmp", noise_floor=0.0, max_passes=1,
            direction="top_down", kl_budget=1.0,
        )
    assert result.steps[0].unit == "a"
    assert result.steps[0].to_fmt == "LO"


def test_top_down_requires_kl_budget_ish():
    """Without a kl_budget, top-down has no termination condition on KL — it
    accepts any bits-decreasing move regardless of quality.  We don't enforce
    'kl_budget required' inside coord_descent_polish itself (the CLI does)
    so this just verifies it runs."""
    unit = _mk_unit("a", ("a.weight",),
                    [("BF16", 0.0, 16, 1024), ("LO", 0.1, 4, 256)])
    kl_table = {
        (("a", "BF16"),): 0.0,
        (("a", "LO"),):   100.0,  # huge KL increase
    }
    starting = {"a.weight": "BF16"}
    result = _stub_run(
        [unit], starting, kl_table,
        direction="top_down", kl_budget=None,
    )
    # With no kl_budget, the move is accepted because it decreases bits.
    assert result.final_assignment == {"a.weight": "LO"}
