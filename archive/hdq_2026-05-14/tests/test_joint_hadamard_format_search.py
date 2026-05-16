"""Unit tests for prismaquant.joint_hadamard_format_search."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as load_safetensors

from prismaquant.hadamard_duquant import (
    ClusterRotationTarget,
    HadamardDuQuantSpec,
    InsertionPointKind,
    NVFP4_GROUP_SIZE,
    MXFP8_GROUP_SIZE,
    sylvester_hadamard,
)
from prismaquant.joint_hadamard_format_search import (
    CandidateScore,
    ClusterInputs,
    ClusterJointResult,
    DEFAULT_BPP,
    DEFAULT_FORMAT_MENU,
    DEFAULT_FORMATS_WITH_ROTATION,
    default_renderer,
    search_cluster,
    run_joint_search,
    _score_candidate,
    _rotation_gate_status,
)


# ---------------------------------------------------------------------------
# Default renderer
# ---------------------------------------------------------------------------


def test_default_renderer_nvfp4_returns_same_shape():
    W = torch.randn(8, 32, dtype=torch.float32)
    out = default_renderer(W, W, "NVFP4", NVFP4_GROUP_SIZE)
    assert out.shape == W.shape
    assert torch.isfinite(out).all()


def test_default_renderer_mxfp8_returns_same_shape():
    W = torch.randn(4, 64, dtype=torch.float32)
    out = default_renderer(W, W, "MXFP8_E4M3", MXFP8_GROUP_SIZE)
    assert out.shape == W.shape
    assert torch.isfinite(out).all()


def test_default_renderer_fp8_per_tensor_returns_same_shape():
    W = torch.randn(8, 48, dtype=torch.float32)
    out = default_renderer(W, W, "FP8_E4M3", None)
    assert out.shape == W.shape
    assert torch.isfinite(out).all()


def test_default_renderer_bf16_is_identity():
    W = torch.randn(4, 16, dtype=torch.float32)
    out = default_renderer(W, W, "BF16", None)
    torch.testing.assert_close(out, W)


def test_default_renderer_rejects_unknown_format():
    W = torch.randn(4, 16, dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown format_label"):
        default_renderer(W, W, "INT4", 16)


# ---------------------------------------------------------------------------
# Per-cluster search
# ---------------------------------------------------------------------------


def _make_inputs(
    cluster_key: str = "model.layers.0.attn.residual",
    n_targets: int = 2,
    out: int = 8,
    in_features: int = 32,
    *,
    seed: int = 0,
) -> ClusterInputs:
    torch.manual_seed(seed)
    return ClusterInputs(
        cluster_key=cluster_key,
        targets=[
            ClusterRotationTarget(
                qname=f"target_{i}",
                weight=torch.randn(out, in_features),
                activations=torch.randn(64, in_features),
            )
            for i in range(n_targets)
        ],
    )


def test_rotation_gate_requires_train_and_validation_margin():
    accepted, reason, train_gain, validation_gain = _rotation_gate_status(
        train_baseline=10.0,
        train_rotated=9.0,
        validation_baseline=10.0,
        validation_rotated=10.5,
        min_train_relative_gain=0.0,
        min_validation_relative_gain=0.0,
    )
    assert accepted is False
    assert reason == "validation_gain_below_margin"
    assert train_gain == pytest.approx(0.1)
    assert validation_gain == pytest.approx(-0.05)


def test_rotation_gate_accepts_when_both_splits_improve():
    accepted, reason, train_gain, validation_gain = _rotation_gate_status(
        train_baseline=10.0,
        train_rotated=9.0,
        validation_baseline=20.0,
        validation_rotated=19.0,
        min_train_relative_gain=0.01,
        min_validation_relative_gain=0.01,
    )
    assert accepted is True
    assert reason is None
    assert train_gain == pytest.approx(0.1)
    assert validation_gain == pytest.approx(0.05)


def _nvfp4_spec(cluster_key: str = "model.layers.0.attn.residual",
                input_dim: int = 32) -> HadamardDuQuantSpec:
    return HadamardDuQuantSpec(
        cluster_key=cluster_key,
        kind=InsertionPointKind.RESIDUAL,
        input_dim=input_dim,
        group_size=NVFP4_GROUP_SIZE,
        consumer_qnames=("a", "b"),
        producer_qnames=(),
        online=True,
    )


def _mxfp8_spec(cluster_key: str = "model.layers.0.attn.residual",
                input_dim: int = 64) -> HadamardDuQuantSpec:
    return HadamardDuQuantSpec(
        cluster_key=cluster_key,
        kind=InsertionPointKind.RESIDUAL,
        input_dim=input_dim,
        group_size=MXFP8_GROUP_SIZE,
        consumer_qnames=("a", "b"),
        producer_qnames=(),
        online=True,
    )


def test_search_cluster_emits_all_default_candidates_for_nvfp4_cluster():
    """NVFP4-sized cluster gets no_rot for all formats + rot for NVFP4 only.

    MXFP8 with G=32 won't fit a cluster sized for G=16 (in_features=32 is
    too small for MXFP8 microscale block alignment), so no rot+MXFP8.
    """
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)
    result = search_cluster(
        spec, inputs,
        solver_n_iters=2, solver_lr=1e-2,
    )
    labels = set(result.candidates.keys())
    # no_rot for every format
    assert "no_rot+NVFP4" in labels
    assert "no_rot+MXFP8_E4M3" in labels
    assert "no_rot+FP8_E4M3" in labels
    assert "no_rot+BF16" in labels
    # rot only for NVFP4 (spec.group_size == 16, only NVFP4 matches)
    assert "rot+NVFP4" in labels
    assert "rot+MXFP8_E4M3" not in labels


def test_search_cluster_rot_for_mxfp8_when_cluster_sized_for_mxfp8():
    """MXFP8-sized cluster (G=32) gets rot+MXFP8 but not rot+NVFP4."""
    spec = _mxfp8_spec(input_dim=64)
    inputs = _make_inputs(in_features=64)
    result = search_cluster(
        spec, inputs,
        solver_n_iters=2, solver_lr=1e-2,
    )
    labels = set(result.candidates.keys())
    assert "rot+MXFP8_E4M3" in labels
    assert "rot+NVFP4" not in labels


def test_search_cluster_rotation_key_only_on_rot_candidates():
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)
    result = search_cluster(spec, inputs, solver_n_iters=2)
    for label, cs in result.candidates.items():
        if label.startswith("rot+"):
            assert cs.rotation_key is not None
            assert spec.cluster_key in cs.rotation_key
        else:
            assert cs.rotation_key is None


def test_search_cluster_online_default_uses_hadamard_runtime_transform():
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)
    result = search_cluster(spec, inputs, solver_n_iters=2)
    cand = result.candidates["rot+NVFP4"]
    assert cand.runtime_transform_type == "hadamard"
    assert cand.runtime_head_dim == NVFP4_GROUP_SIZE
    rr = result.rotations["NVFP4"]
    assert rr.init_strategy == "hadamard"
    assert rr.n_iters == 0
    assert rr.composed_matrix.shape == (
        NVFP4_GROUP_SIZE,
        NVFP4_GROUP_SIZE,
    )


def test_search_cluster_online_learned_mode_marks_random_matrix():
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)
    result = search_cluster(
        spec,
        inputs,
        solver_n_iters=2,
        online_rotation_mode="learned",
    )
    assert result.candidates["rot+NVFP4"].runtime_transform_type == "random-matrix"


def test_search_cluster_default_bpp_attached_to_each_candidate():
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)
    result = search_cluster(spec, inputs, solver_n_iters=2)
    assert result.candidates["no_rot+NVFP4"].bpp == DEFAULT_BPP["NVFP4"]
    assert result.candidates["no_rot+MXFP8_E4M3"].bpp == DEFAULT_BPP["MXFP8_E4M3"]
    assert result.candidates["no_rot+FP8_E4M3"].bpp == DEFAULT_BPP["FP8_E4M3"]
    assert result.candidates["no_rot+BF16"].bpp == DEFAULT_BPP["BF16"]
    assert result.candidates["rot+NVFP4"].bpp == DEFAULT_BPP["NVFP4"]


def test_search_cluster_custom_bpp_override():
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)
    result = search_cluster(
        spec, inputs,
        bpp_per_format={"NVFP4": 4.0625},  # override only NVFP4
        solver_n_iters=2,
    )
    assert result.candidates["no_rot+NVFP4"].bpp == 4.0625
    # Untouched formats use the default
    assert result.candidates["no_rot+MXFP8_E4M3"].bpp == DEFAULT_BPP["MXFP8_E4M3"]


def test_search_cluster_format_menu_filtering():
    """Custom format menu drops formats not in the menu."""
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)
    result = search_cluster(
        spec, inputs,
        format_menu=("NVFP4", "BF16"),
        solver_n_iters=2,
    )
    labels = set(result.candidates.keys())
    assert "no_rot+NVFP4" in labels
    assert "no_rot+BF16" in labels
    assert "no_rot+MXFP8_E4M3" not in labels
    assert "no_rot+FP8_E4M3" not in labels


def test_search_cluster_rotations_dict_keyed_by_format():
    """Per-format rotations are stored under the format label."""
    spec = _mxfp8_spec(input_dim=64)
    inputs = _make_inputs(in_features=64)
    result = search_cluster(spec, inputs, solver_n_iters=2)
    assert set(result.rotations.keys()) == {"MXFP8_E4M3"}
    rr = result.rotations["MXFP8_E4M3"]
    assert rr.composed_matrix.shape == (32, 32)
    assert rr.orthogonality_err < 1e-3


def test_search_cluster_rejects_empty_targets():
    spec = _nvfp4_spec()
    empty_inputs = ClusterInputs(cluster_key=spec.cluster_key, targets=[])
    with pytest.raises(ValueError, match="no targets"):
        search_cluster(spec, empty_inputs)


def test_search_cluster_custom_renderer_is_honored():
    """A constant-output renderer should produce predictable scores."""
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32)

    def zero_renderer(weight, activations, format_label, group_size):
        return torch.zeros_like(weight)

    result = search_cluster(
        spec, inputs,
        renderer=zero_renderer,
        # Rotation solver still uses its internal STE — we can't override
        # the solver's quantizer through search_cluster, so rot+NVFP4
        # still measures non-zero. We just check no_rot+BF16 = ||W||² mean.
        solver_n_iters=2,
    )
    # The zero-renderer makes every no_rot candidate have score equal to
    # the activations-weighted squared norm of W. They should all match
    # each other for no_rot (since the renderer ignores format).
    no_rot_scores = [
        cs.fisher_mse for label, cs in result.candidates.items()
        if label.startswith("no_rot+")
    ]
    # All no_rot scores should be approximately equal — the renderer is
    # format-blind, so they all collapse to the same value.
    for s in no_rot_scores[1:]:
        assert math.isclose(s, no_rot_scores[0], rel_tol=1e-6)


def test_search_cluster_w4a4_score_includes_activation_quantization():
    spec = _nvfp4_spec(input_dim=32)
    inputs = _make_inputs(in_features=32, seed=123)

    def identity_renderer(weight, activations, format_label, group_size):
        return weight

    w_only = search_cluster(
        spec,
        inputs,
        format_menu=("NVFP4",),
        formats_with_rotation=(),
        renderer=identity_renderer,
        solver_n_iters=1,
        score_loss="w_only",
    )
    w4a4 = search_cluster(
        spec,
        inputs,
        format_menu=("NVFP4",),
        formats_with_rotation=(),
        renderer=identity_renderer,
        solver_n_iters=1,
        score_loss="w4a4",
    )
    assert w_only.candidates["no_rot+NVFP4"].fisher_mse == pytest.approx(0.0)
    assert w4a4.candidates["no_rot+NVFP4"].fisher_mse > 0.0


def test_rotated_w4a4_score_uses_original_reference_activations():
    torch.manual_seed(123)
    target = ClusterRotationTarget(
        qname="target",
        weight=torch.randn(8, 32),
        activations=torch.randn(13, 32),
    )
    M = sylvester_hadamard(16)

    def identity_renderer(weight, activations, format_label, group_size):
        return weight

    score = _score_candidate(
        [target],
        "BF16",
        16,
        identity_renderer,
        composed_matrix=M,
        score_loss="w4a4",
        row_chunk=5,
    )

    assert score == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Whole-model driver
# ---------------------------------------------------------------------------


def _build_specs_and_inputs(
    *, n_clusters: int = 2, seed: int = 0,
) -> tuple[list[HadamardDuQuantSpec], dict[str, ClusterInputs]]:
    """Build a small synthetic cluster set for end-to-end tests."""
    specs: list[HadamardDuQuantSpec] = []
    inputs: dict[str, ClusterInputs] = {}
    for i in range(n_clusters):
        key = f"model.layers.{i}.attn.residual"
        specs.append(_nvfp4_spec(cluster_key=key, input_dim=32))
        inputs[key] = _make_inputs(
            cluster_key=key, n_targets=2, in_features=32, seed=seed + i,
        )
    return specs, inputs


def test_run_joint_search_writes_sidecar_with_expected_structure(tmp_path: Path):
    specs, inputs = _build_specs_and_inputs(n_clusters=2)
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    results = run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        solver_n_iters=2,
    )
    payload = json.loads(sidecar.read_text())
    assert payload["version"] == "1"
    assert set(payload["clusters"].keys()) == {
        "model.layers.0.attn.residual",
        "model.layers.1.attn.residual",
    }
    for cluster_key, entry in payload["clusters"].items():
        assert entry["insertion_kind"] == "residual"
        assert "no_rot+NVFP4" in entry["candidates"]
        assert "rot+NVFP4" in entry["candidates"]
        # rot+ candidate carries rotation_key; no_rot+ does not
        assert "rotation_key" in entry["candidates"]["rot+NVFP4"]
        assert (
            entry["candidates"]["rot+NVFP4"]["runtime_transform_type"]
            == "hadamard"
        )
        assert (
            entry["candidates"]["rot+NVFP4"]["runtime_head_dim"]
            == NVFP4_GROUP_SIZE
        )
        assert "rotation_key" not in entry["candidates"]["no_rot+BF16"]

    # Returned in-memory result agrees
    assert len(results) == 2


def test_run_joint_search_writes_rotation_safetensors(tmp_path: Path):
    specs, inputs = _build_specs_and_inputs(n_clusters=2)
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        solver_n_iters=2,
    )
    assert rotations.exists()
    tensors = load_safetensors(str(rotations))
    expected_keys = {
        "model.layers.0.attn.residual/NVFP4/composed_matrix",
        "model.layers.1.attn.residual/NVFP4/composed_matrix",
    }
    assert set(tensors.keys()) == expected_keys
    for key, t in tensors.items():
        assert t.shape == (NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
        # Composed matrix should be orthogonal
        identity = torch.eye(NVFP4_GROUP_SIZE, dtype=t.dtype)
        err = (t @ t.t() - identity).norm().item()
        assert err < 1e-2, f"{key} not orthogonal: err={err}"


def test_run_joint_search_emits_decision_log(tmp_path: Path):
    specs, inputs = _build_specs_and_inputs(n_clusters=2)
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    log = tmp_path / "decisions.jsonl"
    run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        decision_log_path=log,
        solver_n_iters=2,
    )
    assert log.exists()
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    objs = [json.loads(ln) for ln in lines]
    cluster_keys = {o["cluster_key"] for o in objs}
    assert cluster_keys == {
        "model.layers.0.attn.residual",
        "model.layers.1.attn.residual",
    }
    for obj in objs:
        assert obj["rotation"]["applied"] is True
        assert obj["rotation"]["G"] == NVFP4_GROUP_SIZE
        assert "per_format" in obj["rotation"]
        assert "NVFP4" in obj["rotation"]["per_format"]


def test_run_joint_search_decision_log_truncates_on_rerun(tmp_path: Path):
    """Re-running joint search should truncate the log, not append."""
    specs, inputs = _build_specs_and_inputs(n_clusters=1)
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    log = tmp_path / "decisions.jsonl"
    run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        decision_log_path=log,
        solver_n_iters=2,
    )
    first_lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert len(first_lines) == 1
    # Re-run should produce 1 line, not 2
    run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        decision_log_path=log,
        solver_n_iters=2,
    )
    second_lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert len(second_lines) == 1


def test_run_joint_search_skips_specs_without_inputs(tmp_path: Path):
    """Specs whose cluster_key isn't in cluster_inputs are silently skipped."""
    specs, inputs = _build_specs_and_inputs(n_clusters=2)
    # Drop one cluster's data
    del inputs["model.layers.1.attn.residual"]
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    results = run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        solver_n_iters=2,
    )
    assert set(results.keys()) == {"model.layers.0.attn.residual"}
    payload = json.loads(sidecar.read_text())
    assert set(payload["clusters"].keys()) == {"model.layers.0.attn.residual"}


def test_run_joint_search_no_rotations_safetensors_when_no_rot_solved(
    tmp_path: Path,
):
    """If no candidate triggers rotation, no safetensors file is written."""
    specs, inputs = _build_specs_and_inputs(n_clusters=1)
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        formats_with_rotation=(),  # No rotation attempted for any format
        solver_n_iters=2,
    )
    # Sidecar still written, but rotations file should NOT be written
    assert sidecar.exists()
    assert not rotations.exists()


def test_run_joint_search_validation_scores_feed_sidecar(tmp_path: Path):
    specs, train_inputs = _build_specs_and_inputs(n_clusters=1, seed=1)
    _, validation_inputs = _build_specs_and_inputs(n_clusters=1, seed=2)
    spec = specs[0]
    expected = _score_candidate(
        validation_inputs[spec.cluster_key].targets,
        "NVFP4",
        NVFP4_GROUP_SIZE,
        default_renderer,
        composed_matrix=None,
        score_loss="w4a4",
        row_chunk=256,
    )
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    run_joint_search(
        specs,
        train_inputs,
        validation_cluster_inputs=validation_inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        formats_with_rotation=(),
        solver_n_iters=2,
        solver_loss="w4a4",
    )
    payload = json.loads(sidecar.read_text())
    candidate = payload["clusters"][spec.cluster_key]["candidates"]["no_rot+NVFP4"]
    assert candidate["fisher_mse"] == pytest.approx(expected)
    assert candidate["validation_fisher_mse"] == pytest.approx(expected)
    assert "train_fisher_mse" in candidate


def test_run_joint_search_decision_log_no_rotation_summary(tmp_path: Path):
    """When no rotation is solved, the decision-log entry reflects that."""
    specs, inputs = _build_specs_and_inputs(n_clusters=1)
    sidecar = tmp_path / "sidecar.json"
    rotations = tmp_path / "rotations.safetensors"
    log = tmp_path / "decisions.jsonl"
    run_joint_search(
        specs, inputs,
        sidecar_path=sidecar,
        rotation_safetensors_path=rotations,
        decision_log_path=log,
        formats_with_rotation=(),
        solver_n_iters=2,
    )
    obj = json.loads(log.read_text().strip())
    assert obj["rotation"]["applied"] is False
    assert "per_format" not in obj["rotation"]


def test_run_joint_search_sidecar_is_stable_under_rerun(tmp_path: Path):
    """Same inputs + same seed should produce byte-identical sidecar.

    The solver is deterministic given the same input tensors and parameters,
    so two runs with identical inputs should produce identical sidecar JSON.
    """
    specs, inputs = _build_specs_and_inputs(n_clusters=1, seed=42)
    sidecar1 = tmp_path / "sidecar1.json"
    sidecar2 = tmp_path / "sidecar2.json"
    rotations1 = tmp_path / "rotations1.safetensors"
    rotations2 = tmp_path / "rotations2.safetensors"
    run_joint_search(
        specs, inputs,
        sidecar_path=sidecar1,
        rotation_safetensors_path=rotations1,
        solver_n_iters=3, solver_lr=1e-2,
    )
    # Rebuild inputs with same seed so tensors are identical
    specs2, inputs2 = _build_specs_and_inputs(n_clusters=1, seed=42)
    run_joint_search(
        specs2, inputs2,
        sidecar_path=sidecar2,
        rotation_safetensors_path=rotations2,
        solver_n_iters=3, solver_lr=1e-2,
    )
    assert sidecar1.read_text() == sidecar2.read_text()
