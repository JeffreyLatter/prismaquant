"""Unit tests for prismaquant.hadamard_duquant_cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors

from prismaquant.hadamard_duquant import (
    HadamardDuQuantSpec,
    InsertionPointKind,
    MXFP8_GROUP_SIZE,
    NVFP4_GROUP_SIZE,
    apply_block_rotation_input,
    sylvester_hadamard,
)
from prismaquant.hadamard_duquant_cache import (
    CachedRotation,
    HadamardDuQuantCacheState,
    load_cache_state,
    parse_pick_label,
    rotate_consumer_activations,
    rotate_consumer_weight,
    rotate_producer_weight,
)


# ---------------------------------------------------------------------------
# parse_pick_label
# ---------------------------------------------------------------------------


def test_parse_pick_label_rot_format():
    assert parse_pick_label("rot+NVFP4") == (True, "NVFP4")
    assert parse_pick_label("rot+MXFP8_E4M3") == (True, "MXFP8_E4M3")


def test_parse_pick_label_no_rot_format():
    assert parse_pick_label("no_rot+BF16") == (False, "BF16")
    assert parse_pick_label("no_rot+FP8_E4M3") == (False, "FP8_E4M3")


def test_parse_pick_label_invalid_returns_none():
    assert parse_pick_label("") is None
    assert parse_pick_label("garbage") is None
    assert parse_pick_label("maybe+NVFP4") is None


# ---------------------------------------------------------------------------
# CachedRotation validation
# ---------------------------------------------------------------------------


def test_cached_rotation_rejects_non_square_matrix():
    with pytest.raises(ValueError, match="square 2D tensor"):
        CachedRotation(
            cluster_key="x",
            format_label="NVFP4",
            composed_matrix=torch.zeros(16, 17),
            group_size=16,
            insertion_kind="residual",
            online=False,
        )


def test_cached_rotation_rejects_size_mismatch():
    with pytest.raises(ValueError, match="does not match group_size"):
        CachedRotation(
            cluster_key="x",
            format_label="NVFP4",
            composed_matrix=torch.eye(16),
            group_size=32,
            insertion_kind="residual",
            online=False,
        )


# ---------------------------------------------------------------------------
# HadamardDuQuantCacheState lookups
# ---------------------------------------------------------------------------


def _make_state_with_one_cluster() -> HadamardDuQuantCacheState:
    rot = CachedRotation(
        cluster_key="model.layers.0.attn.residual",
        format_label="NVFP4",
        composed_matrix=sylvester_hadamard(NVFP4_GROUP_SIZE),
        group_size=NVFP4_GROUP_SIZE,
        insertion_kind="residual",
        online=False,
    )
    return HadamardDuQuantCacheState(
        rotations_by_cluster={"model.layers.0.attn.residual": rot},
        consumer_to_cluster={
            "model.layers.0.self_attn.q_proj": "model.layers.0.attn.residual",
            "model.layers.0.self_attn.k_proj": "model.layers.0.attn.residual",
            "model.layers.0.self_attn.v_proj": "model.layers.0.attn.residual",
        },
        producer_to_cluster={},
    )


def test_state_rotation_for_consumer_hits():
    state = _make_state_with_one_cluster()
    rot = state.rotation_for_consumer("model.layers.0.self_attn.q_proj")
    assert rot is not None
    assert rot.format_label == "NVFP4"


def test_state_metadata_field_carries_recache_routing():
    state = _make_state_with_one_cluster()
    meta = state.as_metadata_field()

    assert meta["enabled"] is True
    assert (
        meta["consumer_to_cluster"]["model.layers.0.self_attn.q_proj"]
        == "model.layers.0.attn.residual"
    )
    cluster = meta["clusters"]["model.layers.0.attn.residual"]
    assert cluster["format_label"] == "NVFP4"
    assert cluster["group_size"] == NVFP4_GROUP_SIZE
    assert cluster["runtime_transform_type"] == "random-matrix"


def test_state_rotation_for_consumer_misses():
    state = _make_state_with_one_cluster()
    assert state.rotation_for_consumer("model.layers.0.mlp.gate_proj") is None
    assert state.rotation_for_consumer("not_in_index") is None


def test_state_is_empty():
    empty = HadamardDuQuantCacheState()
    assert empty.is_empty()
    state = _make_state_with_one_cluster()
    assert not state.is_empty()


def test_state_as_block_rotations_field():
    state = _make_state_with_one_cluster()
    field = state.as_block_rotations_field()
    assert set(field.keys()) == {"model.layers.0.attn.residual"}
    tensor = field["model.layers.0.attn.residual"]
    assert tensor.shape == (NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    assert tensor.device.type == "cpu"


# ---------------------------------------------------------------------------
# load_cache_state — end-to-end
# ---------------------------------------------------------------------------


def _make_test_artifacts(
    tmp_path: Path,
    *,
    include_rotation_for: tuple[str, ...] = ("NVFP4",),
    runtime_transform_type: str | None = None,
    runtime_head_dim: int | None = None,
) -> tuple[Path, Path, list[HadamardDuQuantSpec]]:
    """Create sidecar JSON + rotations safetensors + insertion specs.

    Returns (sidecar_path, rotations_path, specs).
    """
    cluster_key = "model.layers.0.attn.residual"
    specs = [
        HadamardDuQuantSpec(
            cluster_key=cluster_key,
            kind=InsertionPointKind.RESIDUAL,
            input_dim=NVFP4_GROUP_SIZE * 2,
            group_size=NVFP4_GROUP_SIZE,
            consumer_qnames=(
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
            ),
            producer_qnames=(),
            online=True,
        ),
    ]

    sidecar_path = tmp_path / "sidecar.json"
    sidecar_payload = {
        "version": "1",
        "clusters": {
            cluster_key: {
                "insertion_kind": "residual",
                "candidates": {
                    "no_rot+NVFP4": {"fisher_mse": 0.02, "bpp": 4.5},
                    "rot+NVFP4": {
                        "fisher_mse": 0.01,
                        "bpp": 4.5,
                        "rotation_key": f"{cluster_key}/NVFP4/composed_matrix",
                        **(
                            {"runtime_transform_type": runtime_transform_type}
                            if runtime_transform_type is not None else {}
                        ),
                        **(
                            {"runtime_head_dim": runtime_head_dim}
                            if runtime_head_dim is not None else {}
                        ),
                    },
                },
            },
        },
    }
    sidecar_path.write_text(json.dumps(sidecar_payload, indent=2))

    rotations_path = tmp_path / "rotations.safetensors"
    rotation_tensors: dict[str, torch.Tensor] = {}
    for fmt in include_rotation_for:
        rotation_tensors[
            f"{cluster_key}/{fmt}/composed_matrix"
        ] = sylvester_hadamard(
            runtime_head_dim or NVFP4_GROUP_SIZE
        ).contiguous()
    if rotation_tensors:
        save_safetensors(rotation_tensors, str(rotations_path))

    return sidecar_path, rotations_path, specs


def test_load_cache_state_with_rot_pick(tmp_path: Path):
    sidecar, rotations, specs = _make_test_artifacts(tmp_path)
    picks = {"model.layers.0.attn.residual": "rot+NVFP4"}
    state = load_cache_state(sidecar, rotations, picks, specs)
    assert not state.is_empty()
    rot = state.rotation_for_consumer("model.layers.0.self_attn.q_proj")
    assert rot is not None
    assert rot.format_label == "NVFP4"
    assert rot.group_size == NVFP4_GROUP_SIZE
    assert rot.insertion_kind == "residual"
    assert rot.online is True
    assert rot.runtime_transform_type == "random-matrix"


def test_load_cache_state_preserves_runtime_transform_type(tmp_path: Path):
    sidecar, rotations, specs = _make_test_artifacts(
        tmp_path,
        runtime_transform_type="hadamard",
        runtime_head_dim=NVFP4_GROUP_SIZE,
    )
    picks = {"model.layers.0.attn.residual": "rot+NVFP4"}
    state = load_cache_state(sidecar, rotations, picks, specs)
    rot = state.rotation_for_consumer("model.layers.0.self_attn.q_proj")
    assert rot is not None
    assert rot.runtime_transform_type == "hadamard"
    assert rot.group_size == NVFP4_GROUP_SIZE


def test_load_cache_state_with_no_rot_pick_yields_empty(tmp_path: Path):
    sidecar, rotations, specs = _make_test_artifacts(tmp_path)
    picks = {"model.layers.0.attn.residual": "no_rot+NVFP4"}
    state = load_cache_state(sidecar, rotations, picks, specs)
    assert state.is_empty()


def test_load_cache_state_raises_missing_rotation_by_default(tmp_path: Path):
    """If allocator picked rot+FMT but safetensors lacks that key, fail fast."""
    sidecar, rotations, specs = _make_test_artifacts(
        tmp_path, include_rotation_for=()
    )
    # rotations file is non-existent (no tensors written)
    picks = {"model.layers.0.attn.residual": "rot+NVFP4"}
    with pytest.raises(FileNotFoundError):
        load_cache_state(sidecar, rotations, picks, specs)


def test_load_cache_state_can_skip_missing_rotation_when_not_strict(tmp_path: Path):
    sidecar, rotations, specs = _make_test_artifacts(
        tmp_path, include_rotation_for=()
    )
    picks = {"model.layers.0.attn.residual": "rot+NVFP4"}
    state = load_cache_state(sidecar, rotations, picks, specs, strict=False)
    assert state.is_empty()


def test_load_cache_state_raises_if_sidecar_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_cache_state(
            tmp_path / "does_not_exist.json",
            tmp_path / "rotations.safetensors",
            {},
            [],
        )


def test_load_cache_state_tolerates_missing_safetensors(tmp_path: Path):
    """If no clusters were picked with rotation, safetensors may not exist."""
    sidecar, _rotations, specs = _make_test_artifacts(
        tmp_path, include_rotation_for=()
    )
    picks = {"model.layers.0.attn.residual": "no_rot+BF16"}
    state = load_cache_state(
        sidecar, tmp_path / "does_not_exist.safetensors", picks, specs
    )
    assert state.is_empty()


def test_load_cache_state_indexes_all_consumers(tmp_path: Path):
    sidecar, rotations, specs = _make_test_artifacts(tmp_path)
    picks = {"model.layers.0.attn.residual": "rot+NVFP4"}
    state = load_cache_state(sidecar, rotations, picks, specs)
    # Both q_proj and k_proj should resolve to the same cluster.
    rot_q = state.rotation_for_consumer("model.layers.0.self_attn.q_proj")
    rot_k = state.rotation_for_consumer("model.layers.0.self_attn.k_proj")
    assert rot_q is not None and rot_k is not None
    assert rot_q.cluster_key == rot_k.cluster_key


# ---------------------------------------------------------------------------
# Weight / activation rotation helpers
# ---------------------------------------------------------------------------


def _make_rot(g: int = NVFP4_GROUP_SIZE) -> CachedRotation:
    return CachedRotation(
        cluster_key="cluster",
        format_label="NVFP4",
        composed_matrix=sylvester_hadamard(g),
        group_size=g,
        insertion_kind="residual",
        online=False,
    )


def test_rotate_consumer_weight_matches_block_apply():
    rot = _make_rot()
    W = torch.randn(8, NVFP4_GROUP_SIZE * 3)
    result = rotate_consumer_weight(W, rot)
    expected = apply_block_rotation_input(W, rot.composed_matrix.t())
    torch.testing.assert_close(result, expected)


def test_rotate_consumer_activations_matches_block_apply():
    rot = _make_rot()
    x = torch.randn(32, NVFP4_GROUP_SIZE * 4)
    result = rotate_consumer_activations(x, rot)
    expected = apply_block_rotation_input(x, rot.composed_matrix.t())
    torch.testing.assert_close(result, expected)


def test_rotate_consumer_weight_preserves_shape_and_device():
    rot = _make_rot()
    W = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    result = rotate_consumer_weight(W, rot)
    assert result.shape == W.shape
    assert result.dtype == W.dtype
    assert result.device == W.device


def test_rotate_producer_weight_block_rotates_output_axis():
    """Output-axis rotation: M @ W per output G-block."""
    g = 4
    out_features = 12
    in_features = 7
    M = sylvester_hadamard(g, dtype=torch.float64)
    rot = CachedRotation(
        cluster_key="cluster",
        format_label="NVFP4",
        composed_matrix=M,
        group_size=g,
        insertion_kind="residual",
        online=False,
    )
    W = torch.randn(out_features, in_features, dtype=torch.float64)
    new_W, new_b = rotate_producer_weight(W, rot)
    assert new_b is None
    # Each output G-block should be left-multiplied by M.
    for blk in range(out_features // g):
        expected = M @ W[blk * g:(blk + 1) * g]
        torch.testing.assert_close(
            new_W[blk * g:(blk + 1) * g], expected, atol=1e-10, rtol=0
        )


def test_rotate_producer_weight_rotates_bias_too():
    g = 4
    out_features = 8
    M = sylvester_hadamard(g, dtype=torch.float64)
    rot = CachedRotation(
        cluster_key="cluster",
        format_label="NVFP4",
        composed_matrix=M,
        group_size=g,
        insertion_kind="residual",
        online=False,
    )
    W = torch.randn(out_features, 3, dtype=torch.float64)
    b = torch.randn(out_features, dtype=torch.float64)
    new_W, new_b = rotate_producer_weight(W, rot, bias=b)
    assert new_b is not None
    for blk in range(out_features // g):
        expected_b = M @ b[blk * g:(blk + 1) * g]
        torch.testing.assert_close(
            new_b[blk * g:(blk + 1) * g], expected_b, atol=1e-10, rtol=0
        )


def test_rotate_producer_weight_rejects_indivisible_out_features():
    rot = _make_rot()
    W = torch.randn(13, 4)  # 13 not divisible by 16
    with pytest.raises(ValueError, match="not divisible"):
        rotate_producer_weight(W, rot)


def test_rotate_producer_weight_rejects_non_2d_weight():
    rot = _make_rot()
    W = torch.randn(NVFP4_GROUP_SIZE)  # 1D
    with pytest.raises(ValueError, match="2D weight"):
        rotate_producer_weight(W, rot)


def test_rotate_producer_weight_rejects_bias_mismatch():
    rot = _make_rot()
    W = torch.randn(NVFP4_GROUP_SIZE, 4)
    bad_bias = torch.randn(NVFP4_GROUP_SIZE + 1)
    with pytest.raises(ValueError, match="bias shape"):
        rotate_producer_weight(W, rot, bias=bad_bias)


# ---------------------------------------------------------------------------
# Round-trip: rotate consumer weight + activations, then matmul matches
# ---------------------------------------------------------------------------


def test_consumer_rotation_round_trip_preserves_matmul():
    """For orthogonal M, ``(x M^T) @ (W M^T)^T = x @ W^T`` per block.

    Verifies that rotating both the weight (W → W M^T) and the activations
    (x → x M^T) leaves the output unchanged in floating-point — only
    quantization error after the rotation should affect downstream cache
    quality.
    """
    W = torch.randn(8, NVFP4_GROUP_SIZE * 3, dtype=torch.float64)
    x = torch.randn(32, NVFP4_GROUP_SIZE * 3, dtype=torch.float64)
    # Need to recreate rotation with float64 matrix for tight tolerance
    from prismaquant.hadamard_duquant import random_orthogonal
    M = random_orthogonal(
        NVFP4_GROUP_SIZE,
        generator=torch.Generator().manual_seed(123),
        dtype=torch.float64,
    )
    rot64 = CachedRotation(
        cluster_key="cluster",
        format_label="NVFP4",
        composed_matrix=M,
        group_size=NVFP4_GROUP_SIZE,
        insertion_kind="residual",
        online=False,
    )
    W_rot = rotate_consumer_weight(W, rot64)
    x_rot = rotate_consumer_activations(x, rot64)
    y = x @ W.t()
    y_rot = x_rot @ W_rot.t()
    torch.testing.assert_close(y, y_rot, atol=1e-10, rtol=0)
