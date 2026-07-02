"""Teacher/student pairing guards in kl_measurement (audit M7 / M10).

M7: measure_assignment_kl must hard-fail when the teacher references were
built at a different KL scope than the one the measurement resolves to,
instead of silently broadcasting a [1,1,V] teacher against [1,T,V] student
log-probs.

M10: the lane-replay KL must pair each (possibly microbatch-regrouped)
teacher entry against exactly its own student rows; teacher group i must
never broadcast against student row i.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from prismaquant.build_rtn_cache import kl_divergence
from prismaquant.kl_measurement import (
    _replay_lane_kl_totals,
    measure_assignment_kl,
)


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _KnownLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits, requires_grad=False)

    def forward(self, input_ids):
        return _Output(self.logits[: input_ids.size(0)])


def test_measure_assignment_kl_rejects_scope_mismatched_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "0")
    student_logits = torch.randn(1, 3, 5)
    teacher_logits = torch.randn(1, 3, 5)
    # Last-token references, as _end_kl-style callers build them...
    last_token_refs = [F.log_softmax(teacher_logits[:, -1:, :].float(), dim=-1)]
    calib_ids = torch.ones(1, 3, dtype=torch.long)
    model = _KnownLogits(student_logits)

    # ... meeting a full-sequence student must fail loudly, not broadcast
    # into mean_t KL(p_last || q_t) (audit M7 repro: 0.0043 -> 4.27 silent).
    with pytest.raises(RuntimeError, match="shape mismatch"):
        measure_assignment_kl(
            model,
            {},
            calib_ids,
            last_token_refs,
            work_root=tmp_path,
            kl_scope="full_sequence",
        )

    # The matched scope still measures fine on the exact same refs.
    value = measure_assignment_kl(
        model,
        {},
        calib_ids,
        last_token_refs,
        work_root=tmp_path,
        kl_scope="last_token",
    )
    expected = kl_divergence(
        student_logits[:, -1:, :], last_token_refs[0],
    ).item()
    assert value == pytest.approx(expected)


def _reference_lane_totals(stacked, teacher_rows):
    """Per-lane KL totals computed one (lane, row) pair at a time."""
    lanes, n_rows = stacked.size(0), stacked.size(1)
    totals = torch.zeros(lanes, dtype=torch.float64)
    for lane in range(lanes):
        for row in range(n_rows):
            student_lp = F.log_softmax(stacked[lane, row].double(), dim=-1)
            teacher_lp = teacher_rows[row].double()
            kl_per_pos = (
                teacher_lp.exp() * (teacher_lp - student_lp)
            ).sum(dim=-1)
            totals[lane] += kl_per_pos.mean()
    return totals


def test_replay_lane_kl_totals_microbatched_refs_match_mb1():
    torch.manual_seed(0)
    lanes, n_rows, seq, vocab = 2, 4, 3, 5
    stacked = torch.randn(lanes, n_rows, seq, vocab)
    teacher_rows = [
        F.log_softmax(torch.randn(seq, vocab), dim=-1) for _ in range(n_rows)
    ]
    refs_mb1 = [row.unsqueeze(0) for row in teacher_rows]           # 4 x [1,L,V]
    refs_mb2 = [
        torch.cat(refs_mb1[i : i + 2], dim=0) for i in range(0, n_rows, 2)
    ]                                                               # 2 x [2,L,V]

    totals_mb1 = _replay_lane_kl_totals(
        stacked, refs_mb1, full_sequence_kl=True,
    )
    totals_mb2 = _replay_lane_kl_totals(
        stacked, refs_mb2, full_sequence_kl=True,
    )
    reference = _reference_lane_totals(stacked, teacher_rows)

    assert torch.allclose(totals_mb1, totals_mb2, atol=1e-5)
    assert torch.allclose(totals_mb1, reference.float(), atol=1e-5)


def test_replay_lane_kl_totals_last_token_scope_slices_teacher():
    torch.manual_seed(1)
    lanes, n_rows, vocab = 2, 4, 5
    # Replay logits already sliced to the last position by the caller.
    stacked = torch.randn(lanes, n_rows, 1, vocab)
    full_teachers = [
        F.log_softmax(torch.randn(1, 3, vocab), dim=-1) for _ in range(n_rows)
    ]
    refs_mb2 = [
        torch.cat(full_teachers[i : i + 2], dim=0) for i in range(0, n_rows, 2)
    ]
    totals = _replay_lane_kl_totals(stacked, refs_mb2, full_sequence_kl=False)
    reference = _reference_lane_totals(
        stacked, [t[0, -1:, :] for t in full_teachers],
    )
    assert torch.allclose(totals, reference.float(), atol=1e-5)


def test_replay_lane_kl_totals_row_mismatch_raises():
    torch.manual_seed(2)
    stacked = torch.randn(2, 4, 3, 5)
    refs_short = [F.log_softmax(torch.randn(3, 3, 5), dim=-1)]       # 3 of 4 rows
    refs_long = [F.log_softmax(torch.randn(5, 3, 5), dim=-1)]        # 5 of 4 rows

    with pytest.raises(RuntimeError, match="does not cover"):
        _replay_lane_kl_totals(stacked, refs_short, full_sequence_kl=True)
    with pytest.raises(RuntimeError, match="row mismatch"):
        _replay_lane_kl_totals(stacked, refs_long, full_sequence_kl=True)
