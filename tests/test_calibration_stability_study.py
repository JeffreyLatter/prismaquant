"""Tests for the self-contained CPU calibration stability analysis."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "calib-study" / "calibration_stability.py"
SPEC = importlib.util.spec_from_file_location("calibration_stability", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_subsample_bootstrap_known_constant_and_full_fraction() -> None:
    rng = np.random.default_rng(7)
    constant = study.subsample_means([3.5, 3.5, 3.5, 3.5], 0.25, 50, rng)
    assert np.array_equal(constant, np.full(50, 3.5))

    full = study.subsample_means([1.0, 2.0, 7.0], 1.0, 30, rng)
    assert np.array_equal(full, np.full(30, 10.0 / 3.0))


def test_subsample_bootstrap_two_point_known_distribution() -> None:
    rng = np.random.default_rng(123)
    draws = study.subsample_means([0.0, 2.0], 0.5, 10_000, rng)
    assert set(draws) == {0.0, 2.0}
    assert draws.mean() == pytest.approx(1.0, abs=0.03)
    assert draws.std(ddof=0) == pytest.approx(1.0, abs=0.01)


def test_jaccard() -> None:
    assert study.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert study.jaccard(set(), set()) == 1.0
    assert study.jaccard({1}, set()) == 0.0


def test_allocation_churn() -> None:
    assert study.allocation_churn(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == 0.5
    assert study.allocation_churn([], []) == 0.0
    with pytest.raises(ValueError):
        study.allocation_churn(["a"], ["a", "b"])
