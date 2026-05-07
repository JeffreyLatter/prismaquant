from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import check_format_applicability
from prismaquant.build_rtn_cache import kl_divergence
from prismaquant.kl_sensitivity_probe import (
    UnitOption,
    choose_kneedle_point,
    solve_multi_choice_frontier,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.propagated_cost import resolve_kl_scope


def test_full_sequence_kl_equals_average_of_position_kls():
    teacher_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.25, 1.25, -0.5], [-1.0, 0.0, 2.0]]],
        dtype=torch.float32,
    )
    student_logits = torch.tensor(
        [[[1.5, -0.25, 0.0], [1.0, 0.25, -0.75], [-0.5, 1.5, 0.25]]],
        dtype=torch.float32,
    )
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    full_sequence = kl_divergence(student_logits, teacher_log_probs)
    per_position = torch.stack([
        kl_divergence(
            student_logits[:, idx:idx + 1, :],
            teacher_log_probs[:, idx:idx + 1, :],
        )
        for idx in range(student_logits.size(1))
    ]).mean()

    assert full_sequence.item() == pytest.approx(per_position.item(), abs=1e-12)


def test_explicit_kl_scope_wins_over_legacy_env(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_FULL_SEQUENCE_KL", "1")

    assert resolve_kl_scope(None) == "full_sequence"
    assert resolve_kl_scope("last_token") == "last_token"
    assert resolve_kl_scope("full_sequence") == "full_sequence"


def test_measure_assignment_kl_scope_controls_reduction(tmp_path, monkeypatch):
    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class _KnownLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.nn.Parameter(logits, requires_grad=False)

        def forward(self, input_ids):
            return _Output(self.logits[: input_ids.size(0)])

    monkeypatch.setenv("PRISMAQUANT_FULL_SEQUENCE_KL", "1")
    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "0")
    student_logits = torch.tensor(
        [[[1.5, -0.25, 0.0], [1.0, 0.25, -0.75], [-0.5, 1.5, 0.25]]],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.25, 1.25, -0.5], [-1.0, 0.0, 2.0]]],
        dtype=torch.float32,
    )
    ref_log_probs = [F.log_softmax(teacher_logits, dim=-1)]
    calib_ids = torch.ones(1, 3, dtype=torch.long)
    model = _KnownLogits(student_logits)

    last = measure_assignment_kl(
        model,
        {},
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
        kl_scope="last_token",
    )
    full = measure_assignment_kl(
        model,
        {},
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
        kl_scope="full_sequence",
    )
    legacy_env = measure_assignment_kl(
        model,
        {},
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
        kl_scope=None,
    )

    expected_last = kl_divergence(
        student_logits[:, -1:, :],
        ref_log_probs[0][:, -1:, :],
    ).item()
    expected_full = kl_divergence(student_logits, ref_log_probs[0]).item()
    assert last == pytest.approx(expected_last)
    assert full == pytest.approx(expected_full)
    assert legacy_env == pytest.approx(expected_full)
    assert last != pytest.approx(full)


def test_format_applicability_positive_and_negative_cases():
    assert check_format_applicability(
        (128, 128),
        fr.get_format("NVFP4"),
        qname="model.layers.0.mlp.down_proj",
        source_kind="bf16",
        target_profile="research",
    ).legal

    group_bad = check_format_applicability(
        (128, 17),
        fr.get_format("NVFP4"),
        qname="model.layers.0.mlp.down_proj",
        source_kind="bf16",
        target_profile="research",
    )
    assert not group_bad.legal
    assert group_bad.reason == "group_divisibility"

    profile_bad = check_format_applicability(
        (128, 128),
        fr.get_format("MXFP4"),
        qname="model.layers.0.self_attn.q_proj",
        source_kind="bf16",
        target_profile="vllm_qwen3_5_packed_moe",
    )
    assert not profile_bad.legal
    assert profile_bad.reason == "profile_mismatch"

    source_bad = check_format_applicability(
        (256, 256),
        fr.get_format("FP8_SOURCE"),
        qname="model.layers.0.mlp.down_proj",
        source_kind="bf16",
        target_profile="research",
    )
    assert not source_bad.legal
    assert source_bad.reason == "source_dtype_mismatch"


def test_multi_choice_frontier_finds_non_greedy_knapsack_optimum():
    floor_assignment = {"a": "NVFP4", "b": "NVFP4", "c": "NVFP4"}
    options = {
        "a": [
            UnitOption("a", "NVFP4", ("a",), 100.0, 0.0, 0.0),
            UnitOption("a", "FP8_E4M3", ("a",), 110.0, 10.0, 60.0),
        ],
        "b": [
            UnitOption("b", "NVFP4", ("b",), 100.0, 0.0, 0.0),
            UnitOption("b", "FP8_E4M3", ("b",), 120.0, 20.0, 100.0),
        ],
        "c": [
            UnitOption("c", "NVFP4", ("c",), 100.0, 0.0, 0.0),
            UnitOption("c", "FP8_E4M3", ("c",), 130.0, 30.0, 120.0),
        ],
    }

    frontier = solve_multi_choice_frontier(
        options,
        floor_assignment=floor_assignment,
        floor_kl=1.0,
        budget_points=7,
        bit_precision_bits=1.0,
    )
    at_350 = max(
        (point for point in frontier if point.budget_bits <= 350.0),
        key=lambda point: point.gain,
    )

    assert at_350.gain == pytest.approx(220.0)
    assert at_350.assignment == {
        "a": "NVFP4",
        "b": "FP8_E4M3",
        "c": "FP8_E4M3",
    }
    assert choose_kneedle_point(frontier) >= 0


def test_kl_sensitivity_probe_help_parses():
    result = subprocess.run(
        [sys.executable, "-m", "prismaquant.kl_sensitivity_probe", "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "--kl-scope" in result.stdout


@pytest.mark.skipif(
    os.environ.get("PRISMAQUANT_RUN_QWEN_SMOKE") != "1",
    reason="set PRISMAQUANT_RUN_QWEN_SMOKE=1 for local model smoke",
)
def test_kl_sensitivity_probe_qwen_smoke(tmp_path):
    model = Path("/home/rob/.cache/huggingface/Qwen3-0.6B")
    if not model.exists():
        pytest.skip(f"{model} not present")
    output = tmp_path / "probe.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.kl_sensitivity_probe",
            "--model",
            str(model),
            "--output",
            str(output),
            "--work-root",
            str(tmp_path),
            "--floor-format",
            "NVFP4",
            "--formats",
            "registry",
            "--pin",
            "lm_head",
            "--calib-split",
            "train",
            "--n-calib-samples",
            "1",
            "--calib-seqlen",
            "32",
            "--calib-seed",
            "42",
            "--kl-scope",
            "last_token",
            "--max-lanes-per-batch",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == "prismaquant.kl_sensitivity_probe.v1"
    assert payload["rows"]
    assert payload["floor"]["kl"] >= 0.0
