"""Checkpoint-based packed-expert imatrix synthesis (moe_imatrix):
gate_up = module-input pool, down_proj = routed per-expert replay — the
entries the raw act-cache harvest can never contain, required by the CB
exporter (no silent RTN) and the local packed-expert cost (lockstep)."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.moe_imatrix import synthesize_packed_expert_col_weights

HID, INTER, E = 16, 8, 2


class _IdentityProfile:
    def source_tensor_name(self, name: str) -> str:
        return name


@pytest.fixture()
def ckpt(tmp_path):
    torch.manual_seed(3)
    model_dir = tmp_path / "model"
    act_dir = tmp_path / "act"
    model_dir.mkdir()
    act_dir.mkdir()
    tensors = {
        "model.layers.0.mlp.gate.weight": torch.randn(E, HID),
    }
    for e in range(E):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = \
            torch.randn(INTER, HID)
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = \
            torch.randn(INTER, HID)
    save_file(tensors, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(
        {"num_experts_per_tok": 1, "norm_topk_prob": True}))
    torch.save({"inputs": torch.randn(64, HID),
                "name": "model.layers.0.mlp.experts"},
               act_dir / "model__layers__0__mlp__experts.pt")
    # A dense Linear act entry that must be ignored.
    torch.save({"inputs": torch.randn(64, HID),
                "name": "model.layers.0.self_attn.q_proj"},
               act_dir / "model__layers__0__self_attn__q_proj.pt")
    return model_dir, act_dir


def test_synthesizes_gateup_and_down(ckpt):
    model_dir, act_dir = ckpt
    cw: dict = {}
    added = synthesize_packed_expert_col_weights(
        model_dir, act_dir, cw, profile=_IdentityProfile(), device="cpu")
    assert set(added) == {"model.layers.0.mlp.experts.gate_up_proj",
                         "model.layers.0.mlp.experts.down_proj"}
    gu = cw["model.layers.0.mlp.experts.gate_up_proj"]
    dn = cw["model.layers.0.mlp.experts.down_proj"]
    assert gu.shape == (1, 1, HID) and bool((gu > 0).all())
    assert dn.shape == (E, 1, INTER) and bool((dn > 0).all())
    # No entry for the dense Linear (not a per-expert module).
    assert "model.layers.0.self_attn.q_proj.down_proj" not in cw


def test_respects_existing_entries(ckpt):
    model_dir, act_dir = ckpt
    pre = torch.ones(E, 1, INTER)
    cw = {"model.layers.0.mlp.experts.gate_up_proj": torch.ones(1, 1, HID),
          "model.layers.0.mlp.experts.down_proj": pre}
    added = synthesize_packed_expert_col_weights(
        model_dir, act_dir, cw, profile=_IdentityProfile(), device="cpu")
    assert added == []
    assert torch.equal(cw["model.layers.0.mlp.experts.down_proj"], pre)


def test_missing_router_is_loud(ckpt, tmp_path):
    model_dir, act_dir = ckpt
    # Rebuild the checkpoint without the router weight.
    tensors = {}
    for e in range(E):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = \
            torch.randn(INTER, HID)
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = \
            torch.randn(INTER, HID)
    m2 = tmp_path / "model2"
    m2.mkdir()
    save_file(tensors, str(m2 / "model.safetensors"))
    (m2 / "config.json").write_text(json.dumps({"num_experts_per_tok": 1}))
    with pytest.raises(ValueError, match="router weight"):
        synthesize_packed_expert_col_weights(
            m2, act_dir, {}, profile=_IdentityProfile(), device="cpu")
