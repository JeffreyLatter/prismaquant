"""Pinning tests for the paired pairwise-interaction stderr.

Audit 2026-07-02 §3.11: the pairs study's significance test used an unpaired
stderr √(σ_ab²+σ_a²+σ_b²) although all three KL means are computed over the
SAME calibration windows — the shared window-difficulty variance is
common-mode and cancels in the paired per-window interaction
I_w = KL_ab,w − KL_a,w − KL_b,w. The unpaired form overstated the stderr,
biasing the test toward the additivity null (it fed the "3/1180 pairs
significant" result).
"""
from __future__ import annotations

import math

import pytest

from prismaquant.cross_layer_residual import pair_interaction_stats


def _stats(vals: list[float]) -> dict:
    """Mimic measure_subset's summary of a per-window KL vector."""
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    return {"kl_mean": mean, "kl_stderr": math.sqrt(var / n),
            "kl_windows": list(vals)}


def test_paired_stderr_flags_interaction_the_unpaired_test_misses():
    # Strong common-mode window difficulty + a small but REAL interaction
    # (constant across windows, up to tiny per-window noise).
    diff = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 1.5, 6.0]
    noise = [1e-4 * s for s in (1, -1, 1, -1, -1, 1, -1, 1)]
    delta = 0.01  # the true interaction
    ka = _stats([0.30 * d for d in diff])
    kb = _stats([0.20 * d for d in diff])
    kab = _stats([0.50 * d + delta + e for d, e in zip(diff, noise)])

    out = pair_interaction_stats(kab, ka, kb)

    # The interaction estimate itself is unchanged (difference of means).
    assert out["interaction"] == pytest.approx(delta, rel=1e-6)
    assert out["kl_joint"] == kab["kl_mean"]

    # Paired stderr: sample-std(I_w)/sqrt(n), computed by hand.
    i_w = [ab - a - b for ab, a, b in zip(
        kab["kl_windows"], ka["kl_windows"], kb["kl_windows"])]
    n = len(i_w)
    m = sum(i_w) / n
    var = sum((v - m) ** 2 for v in i_w) / (n - 1)
    assert out["interaction_stderr"] == pytest.approx(
        math.sqrt(var / n), rel=1e-9)

    # The old unpaired value is preserved under the renamed key.
    exp_unpaired = math.sqrt(kab["kl_stderr"] ** 2 + ka["kl_stderr"] ** 2
                             + kb["kl_stderr"] ** 2)
    assert out["interaction_stderr_unpaired"] == pytest.approx(
        exp_unpaired, rel=1e-9)

    # THE point: the paired test flags the true interaction; the unpaired
    # test, inflated by common-mode window difficulty, would call it null.
    assert out["significant"] is True
    assert abs(out["interaction"]) < 2 * out["interaction_stderr_unpaired"]


def test_no_interaction_stays_insignificant_under_paired_test():
    # Pure additivity + common mode + noise: paired must NOT manufacture a
    # significant interaction out of the common mode it removed.
    diff = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 1.5, 6.0]
    noise = [2e-3 * s for s in (1, -1, 1, -1, -1, 1, -1, 1)]
    ka = _stats([0.30 * d for d in diff])
    kb = _stats([0.20 * d for d in diff])
    kab = _stats([0.50 * d + e for d, e in zip(diff, noise)])

    out = pair_interaction_stats(kab, ka, kb)
    assert out["significant"] is False


def test_pair_interaction_requires_aligned_windows():
    ka = _stats([1.0, 2.0, 3.0])
    kb = _stats([1.0, 2.0, 3.0])
    kab = {"kl_mean": 1.5, "kl_stderr": 0.1, "kl_windows": [1.0, 2.0]}
    with pytest.raises(ValueError, match="aligned"):
        pair_interaction_stats(kab, ka, kb)
