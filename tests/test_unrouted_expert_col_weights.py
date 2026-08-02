"""Declared rule for never-routed routed-expert Linears (DSv4-Flash class)."""
from __future__ import annotations

import pytest
import torch

from prismaquant.moe_imatrix import synthesize_unrouted_expert_col_weights


def _stats(**seen: int) -> dict[str, dict[str, int]]:
    return {k.replace("__", "."): {"n_tokens_seen": v} for k, v in seen.items()}


def test_unrouted_expert_inherits_layer_routed_mean() -> None:
    stats = _stats(**{
        "model__layers__3__mlp__experts__0__gate_proj": 5,
        "model__layers__3__mlp__experts__1__gate_proj": 7,
        "model__layers__3__mlp__experts__2__gate_proj": 0,
    })
    cw = {
        "model.layers.3.mlp.experts.0.gate_proj": torch.tensor([1.0, 3.0]),
        "model.layers.3.mlp.experts.1.gate_proj": torch.tensor([3.0, 5.0]),
    }
    report = synthesize_unrouted_expert_col_weights(stats, cw)

    assert report["names"] == ["model.layers.3.mlp.experts.2.gate_proj"]
    assert report["rule"] == "unrouted_expert_neutral_prior:layer_routed_mean"
    assert torch.equal(
        cw["model.layers.3.mlp.experts.2.gate_proj"], torch.tensor([2.0, 4.0]))


def test_donors_are_scoped_to_the_same_layer_and_projection() -> None:
    stats = _stats(**{
        "model__layers__3__mlp__experts__0__gate_proj": 5,
        "model__layers__3__mlp__experts__0__down_proj": 5,
        "model__layers__4__mlp__experts__0__gate_proj": 5,
        "model__layers__3__mlp__experts__1__down_proj": 0,
    })
    cw = {
        "model.layers.3.mlp.experts.0.gate_proj": torch.tensor([9.0]),
        "model.layers.3.mlp.experts.0.down_proj": torch.tensor([2.0, 2.0]),
        "model.layers.4.mlp.experts.0.gate_proj": torch.tensor([100.0]),
    }
    synthesize_unrouted_expert_col_weights(stats, cw)

    # down_proj in layer 3 only — never the gate_proj shape, never layer 4.
    assert torch.equal(
        cw["model.layers.3.mlp.experts.1.down_proj"], torch.tensor([2.0, 2.0]))


def test_a_fully_cold_layer_refuses_instead_of_inventing() -> None:
    stats = _stats(**{"model__layers__9__mlp__experts__0__gate_proj": 0})
    with pytest.raises(ValueError, match="no routed sibling expert"):
        synthesize_unrouted_expert_col_weights(stats, {})


def test_routed_and_non_expert_entries_are_untouched() -> None:
    stats = _stats(**{
        "model__layers__3__mlp__experts__0__gate_proj": 5,
        "model__layers__3__self_attn__wq_a": 11,
    })
    cw = {"model.layers.3.mlp.experts.0.gate_proj": torch.tensor([1.0])}
    report = synthesize_unrouted_expert_col_weights(stats, cw)

    assert report["names"] == []
    assert set(cw) == {"model.layers.3.mlp.experts.0.gate_proj"}
