from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import check_format_applicability
from prismaquant.build_rtn_cache import kl_divergence
import prismaquant.kl_sensitivity_probe as ksp
from prismaquant.kl_sensitivity_probe import (
    FrontierPoint,
    LinearTarget,
    ProbeRow,
    UnitOption,
    choose_kneedle_point,
    _build_unit_options,
    _fused_assignment_violations,
    measure_frontier_points,
    _replay_cache_window_size,
    solve_multi_choice_frontier,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.model_profiles import Qwen3Profile
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

    source_unknown = check_format_applicability(
        (256, 256),
        fr.get_format("FP8_SOURCE"),
        qname="model.layers.0.mlp.down_proj",
        source_kind=None,
        target_profile="research",
    )
    assert not source_unknown.legal
    assert source_unknown.reason == "source_dtype_mismatch"


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


def test_kneedle_can_select_from_measured_frontier_gain():
    points = solve_multi_choice_frontier(
        {
            "a": [
                UnitOption("a", "NVFP4", ("a",), 100.0, 0.0, 0.0),
                UnitOption("a", "FP8_E4M3", ("a",), 120.0, 20.0, 8.0),
            ],
            "b": [
                UnitOption("b", "NVFP4", ("b",), 100.0, 0.0, 0.0),
                UnitOption("b", "FP8_E4M3", ("b",), 140.0, 40.0, 100.0),
            ],
        },
        floor_assignment={"a": "NVFP4", "b": "NVFP4"},
        floor_kl=1.0,
        budget_points=4,
        bit_precision_bits=1.0,
    )
    assert len(points) > 1
    measured = [
        replace(point, measured_kl=1.0 - float(idx), measured_gain=float(idx))
        for idx, point in enumerate(points)
    ]

    assert choose_kneedle_point(measured, use_measured=True) >= 0
    with pytest.raises(ValueError):
        choose_kneedle_point(points, use_measured=True)


def test_measure_frontier_points_reuses_floor_kl(monkeypatch, tmp_path):
    floor_assignment = {"a": "NVFP4", "b": "NVFP4"}
    promoted_assignment = {"a": "BF16", "b": "NVFP4"}
    frontier = [
        FrontierPoint(
            budget_bits=100.0,
            bits_total=100.0,
            bits_delta=0.0,
            gain=0.0,
            predicted_kl=0.75,
            unit_assignment=dict(floor_assignment),
            assignment=dict(floor_assignment),
            promotion_count=0,
        ),
        FrontierPoint(
            budget_bits=120.0,
            bits_total=120.0,
            bits_delta=20.0,
            gain=0.25,
            predicted_kl=0.5,
            unit_assignment=dict(promoted_assignment),
            assignment=dict(promoted_assignment),
            promotion_count=1,
        ),
    ]
    calls = []

    def _measure(_model, assignment, *_args, **_kwargs):
        calls.append(dict(assignment))
        return 0.4

    monkeypatch.setattr(ksp, "measure_assignment_kl", _measure)

    measured = measure_frontier_points(
        torch.nn.Linear(1, 1),
        frontier,
        torch.ones(1, 1, dtype=torch.long),
        [torch.zeros(1, 1, 1)],
        floor_kl=0.75,
        floor_assignment=floor_assignment,
        work_root=tmp_path,
        profile=None,
        kl_scope="last_token",
    )

    assert calls == [promoted_assignment]
    assert measured[0].measured_kl == pytest.approx(0.75)
    assert measured[0].measured_gain == pytest.approx(0.0)
    assert measured[1].measured_kl == pytest.approx(0.4)
    assert measured[1].measured_gain == pytest.approx(0.35)


def test_qwen3_profile_has_vllm_packed_module_fallback_without_vllm():
    profile = Qwen3Profile()
    profile._vllm_cls = None
    profile._vllm_cls_loaded = True
    profile._fused_matcher = None

    assert (
        profile.fused_sibling_group("model.layers.0.self_attn.q_proj")
        == "model.layers.0.self_attn.qkv_proj"
    )
    assert (
        profile.fused_sibling_group("model.layers.0.self_attn.k_proj")
        == "model.layers.0.self_attn.qkv_proj"
    )
    assert (
        profile.fused_sibling_group("model.layers.0.mlp.gate_proj")
        == "model.layers.0.mlp.gate_up_proj"
    )
    assert profile.fused_sibling_group("model.layers.0.mlp.down_proj") is None


def test_probe_frontier_groups_vllm_packed_modules_into_coherent_assignment():
    class _Profile:
        def fused_sibling_group(self, qname):
            if qname.endswith(("gate_proj", "up_proj")):
                return "model.layers.0.mlp.gate_up_proj"
            return None

    targets = [
        LinearTarget("model.layers.0.mlp.gate_proj", (4, 4), 16),
        LinearTarget("model.layers.0.mlp.up_proj", (4, 4), 16),
        LinearTarget("model.layers.0.mlp.down_proj", (4, 4), 16),
    ]
    floor_assignment = {target.qname: "NVFP4" for target in targets}
    rows = [
        ProbeRow(
            qname=target.qname,
            format=fmt,
            shape=target.shape,
            bits_baseline=100.0,
            bits_format=bits,
            bits_delta=bits - 100.0,
            candidate_kl=1.0 - gain,
            sensitivity=gain,
        )
        for target in targets
        for fmt, bits, gain in [
            ("NVFP4", 100.0, 0.0),
            ("MXFP8_E4M3", 140.0, 1.0),
            ("BF16", 200.0, 2.0),
        ]
    ]

    options, unit_for_qname, missing = _build_unit_options(
        rows,
        targets,
        floor_format="NVFP4",
        floor_assignment=floor_assignment,
        profile=_Profile(),
    )

    assert missing == {}
    assert unit_for_qname["model.layers.0.mlp.gate_proj"] == (
        "model.layers.0.mlp.gate_up_proj"
    )
    assert options["model.layers.0.mlp.gate_up_proj"][1].members == (
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    )

    frontier = solve_multi_choice_frontier(
        options,
        floor_assignment=floor_assignment,
        floor_kl=10.0,
        budget_points=4,
        bit_precision_bits=1.0,
    )
    assert frontier
    for point in frontier:
        assert _fused_assignment_violations(
            point.assignment, targets, _Profile()
        ) == {}


def test_replay_cache_auto_window_caps_effective_lane_batch():
    class _Config:
        hidden_size = 16

    class _Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Config()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(4)]
            )

    window = _replay_cache_window_size(
        _Toy(),
        torch.ones(64, 32, dtype=torch.long),
        dtype=torch.float32,
        window_arg="auto",
        max_cache_gb=128.0,
        max_lanes_per_batch=4,
        max_effective_batch=16,
    )

    assert window == 4


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
