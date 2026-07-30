"""R9: the per-sequence tail and the free rung-2 NLL term.

Both fall out of tensors every KL site already holds, so these tests pin the
*math* (against a torch reference) and the *shape contract*, not any GPU path.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from prismaquant.kl_measurement import (
    _quantile,
    sequence_token_nll,
    summarize_per_sequence_kl,
)


def test_sequence_token_nll_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 11)
    ids = torch.randint(0, 11, (1, 6))

    expected = F.cross_entropy(
        logits[0, :-1, :].float(), ids[0, 1:], reduction="mean"
    ).item()

    assert sequence_token_nll(logits, ids) == pytest.approx(expected, rel=1e-6)


def test_sequence_token_nll_chunking_is_exact():
    torch.manual_seed(1)
    logits = torch.randn(1, 17, 23)
    ids = torch.randint(0, 23, (1, 17))
    whole = sequence_token_nll(logits, ids, chunk=1024)
    for chunk in (1, 2, 5, 16):
        assert sequence_token_nll(logits, ids, chunk=chunk) == pytest.approx(
            whole, rel=1e-6)


def test_sequence_token_nll_is_none_without_a_next_token_label():
    # The last-token KL scope emits only position T-1, whose label lies outside
    # the calibration window: there is nothing to score, so report None.
    logits = torch.randn(1, 1, 7)
    ids = torch.randint(0, 7, (1, 8))
    assert sequence_token_nll(logits, ids) is None
    # Batched student logits are not the per-sequence contract.
    assert sequence_token_nll(torch.randn(2, 4, 7), torch.zeros(2, 4).long()) is None


def test_quantile_matches_torch():
    values = [0.4, 0.1, 0.9, 0.2, 0.7]
    tensor = torch.tensor(values, dtype=torch.float64)
    for q in (0.5, 0.95, 0.99, 1.0, 0.0):
        assert _quantile(values, q) == pytest.approx(
            float(tensor.quantile(q)), rel=1e-12)
    assert _quantile([0.3], 0.99) == 0.3


def test_summarize_per_sequence_kl_uses_gold_lane_key_names():
    summary = summarize_per_sequence_kl([0.1, 0.2, 0.3, 1.0])
    assert set(summary) >= {
        "kl_per_sample", "kl_p95", "kl_p99", "kl_max", "kl_tail_domain"}
    assert summary["kl_max"] == 1.0
    # Honest about the sample unit: these are per-SEQUENCE means, not positions.
    assert summary["kl_tail_domain"] == "sequence"
    assert "nll_mean" not in summary


def test_summarize_per_sequence_kl_drops_non_finite_nll():
    summary = summarize_per_sequence_kl(
        [0.1, 0.2], nll_values=[2.0, float("nan"), 4.0, None])
    assert summary["nll_per_sample"] == [2.0, 4.0]
    assert summary["nll_mean"] == pytest.approx(3.0)
    assert math.isfinite(summary["nll_p99"])


def test_summarize_per_sequence_kl_refuses_empty():
    with pytest.raises(ValueError):
        summarize_per_sequence_kl([])
