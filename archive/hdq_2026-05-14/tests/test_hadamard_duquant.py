"""Unit tests for prismaquant.hadamard_duquant."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from prismaquant.activation_sampling import update_priority_reservoir
from prismaquant.hadamard_duquant import (
    # Constants
    NVFP4_GROUP_SIZE,
    MXFP8_GROUP_SIZE,
    SUPPORTED_FORMATS,
    # Primitives
    sylvester_hadamard,
    random_orthogonal,
    cayley_orthogonal,
    apply_block_rotation_input,
    rotate_weight_for_storage,
    effective_weight_for_scoring,
    # STE
    ste_rtn_nvfp4_per_group,
    ste_rtn_mxfp8_per_group,
    # Permutation
    calibrate_zigzag_permutation,
    compose_rotation_with_permutation,
    # Insertion
    InsertionPointKind,
    HadamardDuQuantSpec,
    insertion_specs_for_layer,
    default_insertion_specs,
    # Solver
    ClusterRotationTarget,
    ClusterRotationResult,
    solve_cluster_rotation,
    # Logging
    ClusterDecisionRecord,
    ShipSummaryRecord,
    emit_cluster_decision,
    emit_ship_summary,
)
from prismaquant.render_score import (
    registered_render_mechanisms,
    resolve_render_mechanism_order,
)


# ---------------------------------------------------------------------------
# Algebra primitives
# ---------------------------------------------------------------------------


def test_sylvester_is_involution_after_normalization():
    """Normalized Sylvester Hadamard satisfies H @ H == I (its own inverse)."""
    for g in (2, 4, 8, 16, 32, 64):
        H = sylvester_hadamard(g, dtype=torch.float64)
        identity = torch.eye(g, dtype=torch.float64)
        torch.testing.assert_close(H @ H, identity, atol=1e-10, rtol=0)
        torch.testing.assert_close(H @ H.t(), identity, atol=1e-10, rtol=0)


def test_sylvester_requires_power_of_two():
    with pytest.raises(ValueError):
        sylvester_hadamard(3)
    with pytest.raises(ValueError):
        sylvester_hadamard(0)


def test_random_orthogonal_is_orthogonal():
    gen = torch.Generator()
    gen.manual_seed(42)
    for g in (3, 8, 16, 32):
        R = random_orthogonal(g, generator=gen, dtype=torch.float64)
        identity = torch.eye(g, dtype=torch.float64)
        torch.testing.assert_close(R @ R.t(), identity, atol=1e-10, rtol=0)


def test_cayley_produces_orthogonal_matrix():
    """Cayley(A) is orthogonal for any A (it operates on the skew-symmetric part)."""
    torch.manual_seed(0)
    for g in (4, 8, 16):
        A = torch.randn(g, g, dtype=torch.float64)
        R = cayley_orthogonal(A)
        identity = torch.eye(g, dtype=torch.float64)
        torch.testing.assert_close(R @ R.t(), identity, atol=1e-9, rtol=0)


def test_cayley_at_zero_is_identity():
    """Cayley(0) = (I - 0)^-1 (I + 0) = I."""
    for g in (4, 16):
        A = torch.zeros(g, g, dtype=torch.float64)
        R = cayley_orthogonal(A)
        identity = torch.eye(g, dtype=torch.float64)
        torch.testing.assert_close(R, identity, atol=1e-12, rtol=0)


def test_apply_block_rotation_input_block_diagonal():
    """Per-G-block matmul: each block transforms independently."""
    g = 4
    in_features = 12  # 3 blocks of 4
    R = random_orthogonal(
        g, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    W = torch.randn(7, in_features, dtype=torch.float64)
    out = apply_block_rotation_input(W, R)
    for blk in range(in_features // g):
        expected = W[:, blk * g:(blk + 1) * g] @ R
        torch.testing.assert_close(
            out[:, blk * g:(blk + 1) * g], expected, atol=1e-12, rtol=0
        )


def test_apply_block_rotation_input_rejects_indivisible():
    R = sylvester_hadamard(4)
    W = torch.randn(3, 13)
    with pytest.raises(ValueError, match="not divisible"):
        apply_block_rotation_input(W, R)


def test_storage_scoring_round_trip_no_quantization():
    """W_eff = effective_weight_for_scoring(rotate_weight_for_storage(W, R), R) == W
    when no quantization is applied. Validates the storage/scoring algebra."""
    g = 8
    in_features = 32
    R = random_orthogonal(
        g, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    W = torch.randn(5, in_features, dtype=torch.float64)
    W_stored = rotate_weight_for_storage(W, R)
    W_eff = effective_weight_for_scoring(W_stored, R)
    torch.testing.assert_close(W_eff, W, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# STE quantizers
# ---------------------------------------------------------------------------


def test_ste_rtn_nvfp4_shape_preservation_and_finite():
    W = torch.randn(8, NVFP4_GROUP_SIZE * 3, dtype=torch.float32)
    Wq = ste_rtn_nvfp4_per_group(W, group_size=NVFP4_GROUP_SIZE)
    assert Wq.shape == W.shape
    assert torch.isfinite(Wq).all()


def test_ste_rtn_nvfp4_passes_gradient():
    """STE bypass keeps the operation differentiable end-to-end."""
    W = torch.randn(2, 16, dtype=torch.float32, requires_grad=True)
    Wq = ste_rtn_nvfp4_per_group(W, group_size=16)
    loss = Wq.pow(2).sum()
    loss.backward()
    assert W.grad is not None
    assert W.grad.abs().sum() > 0


def test_ste_rtn_nvfp4_codebook_membership():
    """Each output per block is on the E2M1 codebook (after dividing by scale)."""
    W = torch.randn(4, 16, dtype=torch.float32)
    Wq = ste_rtn_nvfp4_per_group(W, group_size=16)
    # Re-derive the FP8-rounded scale per block and check that output / scale
    # is in the codebook.
    grouped_W = W.reshape(4, 1, 16)
    max_abs = grouped_W.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = max_abs / 6.0
    global_real = (scale.float().amax() / 448.0).clamp_min(1e-12)
    scale = (
        (scale.float() / global_real)
        .clamp(0, 448.0)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        * global_real
    )
    grouped_Wq = Wq.reshape(4, 1, 16)
    normalized = grouped_Wq / scale
    codebook = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    distances = (normalized.unsqueeze(-1) - codebook).abs().amin(dim=-1)
    assert (distances < 1e-4).all(), "outputs are not on the E2M1 codebook"


def test_ste_rtn_mxfp8_shape_preservation_and_finite():
    W = torch.randn(4, MXFP8_GROUP_SIZE * 2, dtype=torch.float32)
    Wq = ste_rtn_mxfp8_per_group(W, group_size=MXFP8_GROUP_SIZE)
    assert Wq.shape == W.shape
    assert torch.isfinite(Wq).all()


def test_ste_rtn_mxfp8_passes_gradient():
    W = torch.randn(2, 32, dtype=torch.float32, requires_grad=True)
    Wq = ste_rtn_mxfp8_per_group(W, group_size=32)
    loss = Wq.pow(2).sum()
    loss.backward()
    assert W.grad is not None
    assert W.grad.abs().sum() > 0


def test_priority_reservoir_does_not_keep_first_rows_only():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    rows = None
    priorities = None
    rows, priorities = update_priority_reservoir(
        rows,
        priorities,
        torch.arange(0, 10, dtype=torch.float32).reshape(10, 1),
        max_rows=5,
        generator=gen,
    )
    rows, priorities = update_priority_reservoir(
        rows,
        priorities,
        torch.arange(100, 110, dtype=torch.float32).reshape(10, 1),
        max_rows=5,
        generator=gen,
    )
    assert rows is not None
    assert priorities is not None
    assert rows.shape == (5, 1)
    assert (rows >= 100).any()
    assert set(rows.flatten().tolist()) != {0.0, 1.0, 2.0, 3.0, 4.0}


# ---------------------------------------------------------------------------
# Zigzag permutation
# ---------------------------------------------------------------------------


def test_zigzag_permutation_is_deterministic():
    torch.manual_seed(0)
    mags = torch.rand(64)
    perm1 = calibrate_zigzag_permutation(mags, 8)
    perm2 = calibrate_zigzag_permutation(mags, 8)
    torch.testing.assert_close(perm1, perm2)


def test_zigzag_permutation_is_valid_bijection():
    mags = torch.tensor([float(i) for i in range(32)])
    perm = calibrate_zigzag_permutation(mags, 8)
    assert perm.shape == (8,)
    assert sorted(perm.tolist()) == list(range(8))


def test_zigzag_snake_order_for_known_magnitudes():
    """For mags = [0, 1, ..., 31] grouped into 4 blocks of 8, per-position
    averages are monotonically increasing in position. Snake interleave
    picks [smallest, largest, 2nd-smallest, 2nd-largest, ...] from the
    sorted-by-magnitude ordering, which for already-sorted means:
    [0, 7, 1, 6, 2, 5, 3, 4]."""
    mags = torch.arange(32, dtype=torch.float32)
    perm = calibrate_zigzag_permutation(mags, 8)
    assert perm.tolist() == [0, 7, 1, 6, 2, 5, 3, 4]


def test_zigzag_rejects_indivisible_input():
    with pytest.raises(ValueError, match="not divisible"):
        calibrate_zigzag_permutation(torch.rand(13), 8)


def test_zigzag_handles_g_equals_1():
    perm = calibrate_zigzag_permutation(torch.rand(4), 1)
    assert perm.tolist() == [0]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_compose_preserves_orthogonality():
    g = 8
    R = random_orthogonal(
        g, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    perm = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
    M = compose_rotation_with_permutation(R, perm)
    identity = torch.eye(g, dtype=torch.float64)
    torch.testing.assert_close(M @ M.t(), identity, atol=1e-10, rtol=0)


def test_compose_identity_perm_returns_R():
    g = 4
    R = random_orthogonal(g, generator=torch.Generator().manual_seed(0))
    M = compose_rotation_with_permutation(R, torch.arange(g))
    torch.testing.assert_close(M, R)


def test_compose_matches_gather_indexing():
    """The composed matrix equals R[perm][:, perm] by definition."""
    g = 6
    R = torch.arange(g * g, dtype=torch.float32).reshape(g, g)
    perm = torch.tensor([3, 1, 4, 0, 2, 5])
    M = compose_rotation_with_permutation(R, perm)
    expected = R[perm][:, perm]
    torch.testing.assert_close(M, expected)


def test_compose_rejects_size_mismatch():
    R = torch.eye(4)
    perm_too_short = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="must be"):
        compose_rotation_with_permutation(R, perm_too_short)


# ---------------------------------------------------------------------------
# Insertion-point identification
# ---------------------------------------------------------------------------


class _MockAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)


class _MockMlp(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


class _MockLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64, intermediate_dim: int = 128):
        super().__init__()
        self.self_attn = _MockAttention(hidden_dim)
        self.mlp = _MockMlp(hidden_dim, intermediate_dim)


def test_insertion_specs_for_dense_layer_finds_residual_v_o_down_proj():
    layer = _MockLayer(hidden_dim=64, intermediate_dim=128)
    specs = insertion_specs_for_layer(
        layer, "model.layers.0", group_size=16, hidden_dim=64,
        include_online=True,
        include_residual_online=True,
    )
    kinds = [s.kind for s in specs]
    assert kinds.count(InsertionPointKind.RESIDUAL) == 2  # attn + mlp
    assert kinds.count(InsertionPointKind.V_O) == 1
    assert kinds.count(InsertionPointKind.DOWN_PROJ) == 1
    # ATTN_OUT is currently subsumed by V_O in the dense path.
    assert kinds.count(InsertionPointKind.ATTN_OUT) == 0


def test_insertion_specs_down_proj_is_online_no_producers():
    layer = _MockLayer(hidden_dim=64, intermediate_dim=128)
    specs = insertion_specs_for_layer(
        layer, "model.layers.0", group_size=16, hidden_dim=64,
        include_online=True,
    )
    down = [s for s in specs if s.kind == InsertionPointKind.DOWN_PROJ]
    assert len(down) == 1
    assert down[0].online is True
    assert down[0].producer_qnames == ()
    assert down[0].consumer_qnames == ("model.layers.0.mlp.down_proj",)


def test_insertion_specs_residual_consumer_qkv():
    layer = _MockLayer(hidden_dim=64, intermediate_dim=128)
    specs = insertion_specs_for_layer(
        layer, "model.layers.0", group_size=16, hidden_dim=64,
        include_residual_online=True,
    )
    attn_res = [
        s for s in specs
        if s.kind == InsertionPointKind.RESIDUAL and ".attn." in s.cluster_key
    ]
    assert len(attn_res) == 1
    assert set(attn_res[0].consumer_qnames) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert attn_res[0].producer_qnames == ()
    assert attn_res[0].online is True


def test_insertion_specs_v_o_uses_v_out_dim():
    layer = _MockLayer(hidden_dim=64, intermediate_dim=128)
    specs = insertion_specs_for_layer(
        layer, "model.layers.0", group_size=16, hidden_dim=64
    )
    v_o = [s for s in specs if s.kind == InsertionPointKind.V_O]
    assert len(v_o) == 1
    assert v_o[0].input_dim == 64  # V_proj.out_features
    assert v_o[0].consumer_qnames == ("model.layers.0.self_attn.o_proj",)
    assert v_o[0].producer_qnames == ("model.layers.0.self_attn.v_proj",)


def test_insertion_specs_skip_when_dim_not_divisible():
    layer = _MockLayer(hidden_dim=48, intermediate_dim=128)
    specs = insertion_specs_for_layer(
        layer, "model.layers.0", group_size=32, hidden_dim=48
    )
    # 48 % 32 != 0, so no RESIDUAL specs (which use hidden_dim).
    assert all(s.kind != InsertionPointKind.RESIDUAL for s in specs)


def test_default_insertion_specs_walks_two_layer_model():
    class _MockInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                _MockLayer(hidden_dim=64, intermediate_dim=128),
                _MockLayer(hidden_dim=64, intermediate_dim=128),
            ])

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MockInner()

    model = _MockModel()
    specs = default_insertion_specs(
        model,
        group_size=16,
        include_online=True,
        include_residual_online=True,
    )
    # 4 specs per layer (2 RESIDUAL + 1 V_O + 1 DOWN_PROJ) * 2 layers
    assert len(specs) == 8
    keys = [s.cluster_key for s in specs]
    assert "model.layers.0.attn.residual" in keys
    assert "model.layers.1.attn.v_o" in keys
    assert "model.layers.1.mlp.down" in keys


def test_default_insertion_specs_production_default_is_folded_only():
    class _MockInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                _MockLayer(hidden_dim=64, intermediate_dim=128),
                _MockLayer(hidden_dim=64, intermediate_dim=128),
            ])

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MockInner()

    model = _MockModel()
    specs = default_insertion_specs(model, group_size=16)
    assert len(specs) == 2
    assert {s.kind for s in specs} == {InsertionPointKind.V_O}
    assert all(not s.online for s in specs)
    assert all(s.producer_qnames for s in specs)


def test_hadamard_duquant_spec_validation_input_dim_divisible():
    with pytest.raises(ValueError, match="not divisible"):
        HadamardDuQuantSpec(
            cluster_key="x",
            kind=InsertionPointKind.RESIDUAL,
            input_dim=48,
            group_size=32,
            consumer_qnames=("a",),
            producer_qnames=(),
            online=False,
        )


def test_hadamard_duquant_spec_validation_online_no_producers():
    with pytest.raises(ValueError, match="online insertion points"):
        HadamardDuQuantSpec(
            cluster_key="x",
            kind=InsertionPointKind.DOWN_PROJ,
            input_dim=32,
            group_size=32,
            consumer_qnames=("a",),
            producer_qnames=("b",),
            online=True,
        )


def test_hadamard_duquant_spec_rejects_offline_without_producer():
    with pytest.raises(ValueError, match="offline rotations require producer"):
        HadamardDuQuantSpec(
            cluster_key="x",
            kind=InsertionPointKind.RESIDUAL,
            input_dim=32,
            group_size=16,
            consumer_qnames=("a",),
            producer_qnames=(),
            online=False,
        )


# ---------------------------------------------------------------------------
# Cluster rotation solver
# ---------------------------------------------------------------------------


def _make_targets(n: int = 1, out: int = 8, in_features: int = 32, *,
                  seed: int = 0) -> list[ClusterRotationTarget]:
    torch.manual_seed(seed)
    return [
        ClusterRotationTarget(
            qname=f"target_{i}",
            weight=torch.randn(out, in_features),
            activations=torch.randn(64, in_features),
        )
        for i in range(n)
    ]


def test_solve_cluster_rotation_returns_orthogonal_R():
    targets = _make_targets(n=1, in_features=32)
    result = solve_cluster_rotation(
        targets, group_size=16, format_label="NVFP4",
        init_strategy="sylvester", n_iters=5, lr=1e-2,
    )
    assert result.orthogonality_err < 1e-3
    assert result.R.shape == (16, 16)
    assert result.composed_matrix.shape == (16, 16)
    assert result.permutation.shape == (16,)


def test_solve_cluster_rotation_multiple_siblings():
    targets = _make_targets(n=3, in_features=64)
    result = solve_cluster_rotation(
        targets, group_size=16, format_label="NVFP4", n_iters=3, lr=1e-2,
    )
    assert result.orthogonality_err < 1e-3
    assert result.composed_matrix.shape == (16, 16)


def test_solve_cluster_rotation_mxfp8():
    targets = _make_targets(n=1, out=4, in_features=64)
    result = solve_cluster_rotation(
        targets, group_size=32, format_label="MXFP8_E4M3",
        init_strategy="sylvester", n_iters=3, lr=1e-2,
    )
    assert result.orthogonality_err < 1e-3
    assert result.R.shape == (32, 32)


def test_solve_cluster_rotation_finite_scores_at_zero_iters():
    """With n_iters=0, both baseline_score and rotated_score must be finite —
    the solver computes both from final M, not from accumulated loop state."""
    targets = _make_targets(n=1, in_features=32)
    result = solve_cluster_rotation(
        targets, group_size=16, format_label="NVFP4",
        init_strategy="identity", n_iters=0,
    )
    assert math.isfinite(result.baseline_score)
    assert math.isfinite(result.rotated_score)
    # With identity init and zero iters, R is identity but M = P_perm. Scores
    # may differ from baseline depending on the permutation's effect.


def test_solve_cluster_rotation_rejects_unsupported_format():
    targets = _make_targets(n=1, in_features=16)
    with pytest.raises(ValueError, match="unsupported format_label"):
        solve_cluster_rotation(
            targets, group_size=16, format_label="BF16", n_iters=1,
        )


def test_solve_cluster_rotation_rejects_dim_mismatch():
    targets = [
        ClusterRotationTarget(
            qname="q",
            weight=torch.randn(4, 15),
            activations=torch.randn(8, 15),
        )
    ]
    with pytest.raises(ValueError, match="not divisible"):
        solve_cluster_rotation(
            targets, group_size=16, format_label="NVFP4", n_iters=1,
        )


def test_solve_cluster_rotation_accepts_external_permutation():
    """If an explicit permutation is provided, it's used as-is."""
    targets = _make_targets(n=1, in_features=32)
    custom_perm = torch.tensor(
        [15, 0, 14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7]
    )
    result = solve_cluster_rotation(
        targets, group_size=16, format_label="NVFP4",
        n_iters=2, permutation=custom_perm,
    )
    torch.testing.assert_close(result.permutation.cpu(), custom_perm)


def test_solve_cluster_rotation_rejects_wrong_size_permutation():
    targets = _make_targets(n=1, in_features=32)
    with pytest.raises(ValueError, match="must equal group_size"):
        solve_cluster_rotation(
            targets, group_size=16, format_label="NVFP4",
            n_iters=1, permutation=torch.arange(8),
        )


def test_cluster_rotation_result_log_dict_has_expected_keys():
    targets = _make_targets(n=1, in_features=32)
    result = solve_cluster_rotation(
        targets, group_size=16, format_label="NVFP4", n_iters=2,
    )
    d = result.to_log_dict()
    expected_keys = {
        "applied", "G", "permutation_swaps", "solver_seconds",
        "orthogonality_err", "init_strategy", "n_iters",
        "baseline_score", "rotated_score", "relative_gain",
    }
    assert expected_keys.issubset(d.keys())
    assert d["G"] == 16
    assert isinstance(d["applied"], bool)
    assert isinstance(d["permutation_swaps"], int)


# ---------------------------------------------------------------------------
# Log emission
# ---------------------------------------------------------------------------


def test_emit_cluster_decision_writes_jsonl(tmp_path: Path):
    record = ClusterDecisionRecord(
        cluster_key="model.layers.0.attn.residual",
        insertion_kind="residual",
        rotation={"applied": True, "G": 16, "relative_gain": 0.12},
        candidates={
            "no_rot+NVFP4": {"fisher_mse": 0.02, "bpp": 4.0},
            "rot+NVFP4": {"fisher_mse": 0.015, "bpp": 4.0},
        },
        allocator_pick="rot+NVFP4",
        render_gates=[{"mechanism": "hadamard_duquant", "accepted": True}],
    )
    out = tmp_path / "decisions.jsonl"
    emit_cluster_decision(record, out, append=False)
    emit_cluster_decision(record, out, append=True)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert obj["cluster_key"] == "model.layers.0.attn.residual"
        assert obj["allocator_pick"] == "rot+NVFP4"
        assert obj["insertion_kind"] == "residual"


def test_emit_ship_summary_writes_pretty_json(tmp_path: Path):
    record = ShipSummaryRecord(
        model="Qwen3.5-0.8B",
        target_bpp=4.5,
        actual_bpp=4.487,
        format_distribution={"NVFP4": 89, "MXFP8_E4M3": 5,
                             "FP8_E4M3": 2, "BF16": 4},
        rotation_distribution={"with_rotation": 76, "no_rotation": 24},
        insertion_coverage={"residual": 24, "v_o": 24,
                            "down_proj": 24, "attn_out": 0},
        metrics={
            "calibration_kl_n8_s512": 0.082,
            "wikitext2_ppl": 12.34,
            "c4_ppl": 18.91,
            "bf16_argmax_agreement": 0.972,
        },
        vs_baseline_no_rotation={
            "kl_delta": -0.027,
            "wikitext2_ppl_delta": -0.45,
        },
    )
    out = tmp_path / "ship.json"
    emit_ship_summary(record, out)
    obj = json.loads(out.read_text())
    assert obj["model"] == "Qwen3.5-0.8B"
    assert obj["target_bpp"] == 4.5
    assert obj["format_distribution"]["NVFP4"] == 89
    assert obj["vs_baseline_no_rotation"]["kl_delta"] == pytest.approx(-0.027)


# ---------------------------------------------------------------------------
# Render mechanism registration
# ---------------------------------------------------------------------------


def test_hadamard_duquant_mechanism_is_registered():
    """The mechanism is registered at import time via _register_builtins."""
    mechs = registered_render_mechanisms()
    assert "hadamard_duquant" in mechs
    spec = mechs["hadamard_duquant"]
    assert spec.phase == 20
    assert spec.exclusive_group == "activation_weight_fold"
    assert spec.gate_metric == "fisher_output_mse"
    assert spec.scope == "fused_sibling_group"


def test_hadamard_duquant_orders_before_others():
    """Phase 20 means hadamard_duquant runs before phase 30+ mechanisms."""
    plan = resolve_render_mechanism_order(
        ["hadamard_duquant", "gptq", "scale_sweep", "four_over_six"],
    )
    names = list(plan.names())
    assert plan.errors == ()
    assert names[0] == "hadamard_duquant"
    assert names.index("four_over_six") < names.index("gptq")
    assert names.index("gptq") < names.index("scale_sweep")
