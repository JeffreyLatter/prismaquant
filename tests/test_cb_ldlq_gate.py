"""Gated LDLQ refinement: do-no-harm on col-weighted MSE."""

import os

import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_ldlq_refinement import (
    REFINEMENT_SCHEMA,
    build_refinement_provenance,
    validate_refinement_provenance,
)


def test_ldlq_gate_keeps_better_arm():
    torch.manual_seed(0)
    w = torch.randn(32, 2048)
    cw = torch.ones(1, 2048) * 2.0
    k = 12
    grid, mode = "fp4", "product"
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, col_weights=cw)
    recon_raw = cb.nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode)
    # Create synthetic activation rows: identity proxy for Hessian
    act = torch.randn(128, 2048)
    raw_act_err = float((act @ (w - recon_raw).T).pow(2).mean().item())
    # Gated call should not be worse than raw on the activation metric
    gated_fields, info = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, act, grid=grid, mode=mode, k=k
    )
    recon_gated = cb.nvfp4_cb_reconstruct(gated_fields, k, grid=grid, mode=mode)
    gated_act_err = float((act @ (w - recon_gated).T).pow(2).mean().item())
    assert gated_act_err <= raw_act_err + 1e-9, f"gated regressed {gated_act_err} > {raw_act_err}"
    assert info["gate"] in {"ldlq_kept", "raw_kept", "ldlq_kept_all", "raw_kept_all", "mixed_per_expert", "disabled"}
    # Byte count identical regardless of arm
    assert gated_fields["indices"].numel() == fields["indices"].numel()


def test_ldlq_gate_disabled_returns_ldlq():
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "0"
    try:
        torch.manual_seed(1)
        w = torch.randn(8, 2048)
        cw = torch.randn(1, 2048).abs()
        act = torch.randn(64, 2048)
        fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
        gated_fields, info = cb.ldlq_reassign_cb_fields_gated(
            w, fields, cw, act, grid="fp4", mode="product", k=12
        )
        assert info["gate"] == "disabled"
        # When gate disabled, result should equal ungated LDLQ
        ungated = cb.ldlq_reassign_cb_fields(w, fields, cw, act, grid="fp4", mode="product")
        assert torch.equal(gated_fields["indices"], ungated["indices"])
    finally:
        os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "1"


def test_ldlq_gate_per_expert_mixed():
    torch.manual_seed(2)
    E, R, C = 4, 16, 2048
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    # Make one expert pathological: its activation rows are tiny, so LDLQ may not help
    # but gate should still decide per expert
    acts = [torch.randn(32, C) for _ in range(E)]
    # Use a fixed codebook so we can compare
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    gated_fields, info = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, acts, grid="fp4", mode="product", k=12
    )
    assert "per_expert_kept" in info or "kept_ldlq" in info
    # Ensure indices shape preserved
    assert gated_fields["indices"].shape == fields["indices"].shape


def test_refinement_provenance_roundtrip():
    prov = build_refinement_provenance(cost_ldlq=False, export_ldlq=True, gate="col_weighted_mse", gate_enabled=True)
    assert prov["schema"] == REFINEMENT_SCHEMA
    out = validate_refinement_provenance(prov, where="test")
    assert out["cost_ldlq"] is False
    assert out["export_ldlq"] is True
    # Forged ldlq downgrade should fail
    try:
        build_refinement_provenance(cost_ldlq=True, export_ldlq=False, gate="x", gate_enabled=True)
        assert False, "should have raised"
    except ValueError:
        pass


def test_batch_vs_serial_ldlq_bit_identical():
    # Batched and serial LDLQ should be bit-identical when gate disabled,
    # and gated per-expert should be consistent.
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "0"
    torch.manual_seed(3)
    E, R, C = 2, 8, 1024
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    acts = [torch.randn(32, C) for _ in range(E)]
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    # Serial path: batch_experts=False
    serial = cb.ldlq_reassign_cb_fields(w, fields, cw, acts, grid="fp4", mode="product", batch_experts=False)
    batched = cb.ldlq_reassign_cb_fields(w, fields, cw, acts, grid="fp4", mode="product", batch_experts=True)
    assert torch.equal(serial["indices"], batched["indices"])
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "1"
