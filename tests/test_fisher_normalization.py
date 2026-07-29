"""Fisher h_trace must be normalized by the GLOBAL calib token count.

Synthetic reproducer: two identical Linears with identical gradient
statistics over the same calibration run. The "dense" one sees every token
(most carrying zero gradient); the "expert" one only sees the tokens routed
to it — the very tokens that carry the gradient. Tokens never routed to an
expert contribute zero gradient, so both rows have the SAME empirical
Fisher. The old per-row `h_trace_raw / n_tokens_seen` divided the expert by
its routed count only, inflating it by (global/routed).

Also pins:
  - scalar/detail denominator agreement: the h-detail blob (which feeds
    `predicted_dloss` via measure_quant_cost) and the scalar `h_trace`
    must share the one global-token denominator — sum(h_diag) == h_trace;
  - the sensitivity backend (`FisherAccumulator.finalize`) applies the
    same convention end-to-end, scalar and blob;
  - the allocator's load-time renormalization: recompute from raw
    accumulators when meta carries the token count, hard-fail when it
    does not (escape hatch: --allow-legacy-fisher-norm).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from prismaquant.incremental_probe import finalize_fisher_stats
from prismaquant.measure_quant_cost import HDetailIndex
from prismaquant.sensitivity_probe import FisherAccumulator, h_detail_blob


def _h_trace_raw(x: torch.Tensor, gy: torch.Tensor) -> float:
    """The probe's per-token trace accumulator: sum_t ||gy_t||^2 * ||x_t||^2."""
    return float((gy.pow(2).sum(dim=1) * x.pow(2).sum(dim=1)).sum().item())


def test_h_trace_equal_for_equal_gradient_mass_after_global_normalization():
    torch.manual_seed(0)
    global_tokens, routed_tokens = 32, 4  # the expert is routed 1/8 of tokens
    dense = nn.Linear(16, 8, bias=False)
    expert = nn.Linear(16, 8, bias=False)
    expert.load_state_dict(dense.state_dict())

    x = torch.randn(global_tokens, 16)
    # Identical per-token gradients on the routed tokens; zero elsewhere
    # (a token never routed to the expert contributes zero gradient).
    gy = torch.zeros(global_tokens, 8)
    gy[:routed_tokens] = torch.randn(routed_tokens, 8)

    y_dense = dense(x)                      # dense row: all 32 tokens
    y_dense.backward(gy)
    y_expert = expert(x[:routed_tokens])    # expert row: its 4 routed tokens
    y_expert.backward(gy[:routed_tokens])

    stats = {
        "dense": {"h_trace_raw": _h_trace_raw(x, gy),
                  "h_w2_sum_raw": 0.0, "n_tokens_seen": global_tokens},
        "expert": {"h_trace_raw": _h_trace_raw(x[:routed_tokens],
                                               gy[:routed_tokens]),
                   "h_w2_sum_raw": 0.0, "n_tokens_seen": routed_tokens},
    }
    # Same gradient mass -> same raw accumulator (and same weight grads).
    assert stats["expert"]["h_trace_raw"] == pytest.approx(
        stats["dense"]["h_trace_raw"])
    assert torch.allclose(dense.weight.grad, expert.weight.grad)

    # OLD normalization (per-row n_tokens_seen), computed inline for
    # contrast: the expert looks (global/routed) = 8x more sensitive.
    old = {n: s["h_trace_raw"] / s["n_tokens_seen"] for n, s in stats.items()}
    assert old["expert"] == pytest.approx(
        old["dense"] * global_tokens / routed_tokens)

    finalize_fisher_stats(stats, global_tokens)
    assert stats["expert"]["h_trace"] == pytest.approx(stats["dense"]["h_trace"])
    assert stats["dense"]["h_trace"] == pytest.approx(
        stats["dense"]["h_trace_raw"] / global_tokens)
    # n_tokens_seen stays raw (routed count) for the h_detail consumers.
    assert stats["expert"]["n_tokens_seen"] == routed_tokens
    assert stats["expert"]["h_trace_norm_tokens"] == global_tokens


def test_rows_without_token_counter_use_global_count_not_one():
    """Stat rows filled by the batched MoE-block flush path never increment
    n_tokens_seen; the old finalize divided them by max(0, 1) == 1."""
    stats = {"expert": {"h_trace_raw": 64.0, "h_w2_sum_raw": 0.0,
                        "n_tokens_seen": 0}}
    finalize_fisher_stats(stats, 32)
    assert stats["expert"]["h_trace"] == pytest.approx(2.0)


def test_scalar_and_h_detail_share_the_global_denominator():
    """The h-detail blob (predicted_dloss fallback input) and the scalar
    h_trace must be divided by the SAME global token count. Identity:
    h_diag[i,j] = sum_t gy_ti^2 x_tj^2 / N  =>  sum(h_diag) == h_trace.
    Dividing the expert blob by its routed count instead (the pre-fix
    writer behavior) leaves the detail (global/routed)x hotter than the
    scalar — the mixed-units error, relocated into the knapsack."""
    torch.manual_seed(1)
    global_tokens, routed_tokens = 32, 4
    x_r = torch.randn(routed_tokens, 16)
    gy_r = torch.randn(routed_tokens, 8)

    # Expert row: raw accumulators over its routed tokens only.
    h_full_raw = gy_r.pow(2).t() @ x_r.pow(2)       # token-summed [out, in]
    stats = {"expert": {"h_trace_raw": _h_trace_raw(x_r, gy_r),
                        "h_w2_sum_raw": 0.0, "n_tokens_seen": routed_tokens}}
    finalize_fisher_stats(stats, global_tokens)

    blob = h_detail_blob(h_full_raw, global_tokens, "expert", kind="linear")
    h_diag = HDetailIndex.h_diag_from_blob(blob)
    assert blob["norm_tokens"] == global_tokens
    assert float(h_diag.sum()) == pytest.approx(
        stats["expert"]["h_trace"], rel=1e-5)

    # Contrast: a per-routed-token blob disagrees with the scalar by
    # exactly (global/routed).
    stale = HDetailIndex.h_diag_from_blob(
        h_detail_blob(h_full_raw, routed_tokens, "expert", kind="linear"))
    assert float(stale.sum()) == pytest.approx(
        stats["expert"]["h_trace"] * global_tokens / routed_tokens, rel=1e-5)


def test_sensitivity_backend_scalar_and_blob_agree_end_to_end():
    """FisherAccumulator.finalize: an unpacked expert Linear routed a
    fraction of the calib tokens lands at the same h_trace as a dense
    twin with identical gradient mass, and its h-detail blob sums to the
    scalar (shared global denominator), while n_tokens_seen stays raw."""
    torch.manual_seed(2)
    global_tokens, routed_tokens = 24, 3
    model = nn.Module()
    model.dense = nn.Linear(6, 5, bias=False)
    model.expert = nn.Linear(6, 5, bias=False)
    model.expert.load_state_dict(model.dense.state_dict())
    for p in model.parameters():
        p.requires_grad_(False)

    x = torch.randn(global_tokens, 6)
    v = torch.zeros(global_tokens, 5)
    v[:routed_tokens] = torch.randn(routed_tokens, 5)

    with tempfile.TemporaryDirectory() as td:
        h_dir = Path(td) / "h"
        acc = FisherAccumulator(model, ["dense", "expert"], {},
                                h_detail_dir=h_dir)
        xg = x.detach().requires_grad_(True)
        xr = x[:routed_tokens].detach().requires_grad_(True)
        loss = (model.dense(xg) * v).sum() + \
            (model.expert(xr) * v[:routed_tokens]).sum()
        loss.backward()
        acc.finalize(None, global_tokens=global_tokens)
        acc.remove_hooks()

        d, e = acc.stats["dense"], acc.stats["expert"]
        assert d["n_tokens_seen"] == global_tokens
        assert e["n_tokens_seen"] == routed_tokens
        assert e["h_trace"] == pytest.approx(d["h_trace"], rel=1e-4)
        assert e["h_trace_norm_tokens"] == global_tokens

        index = HDetailIndex(h_dir, ["dense", "expert"])
        for name, s in (("dense", d), ("expert", e)):
            blob = index.load_blob(name)
            assert blob["h_detail_version"] == 4
            assert blob["norm_tokens"] == global_tokens
            assert float(blob["h_diag"].sum()) == pytest.approx(
                s["h_trace"], rel=1e-4)
