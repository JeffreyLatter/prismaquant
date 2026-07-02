"""Empirical packed-expert unit-KL cost (the AURA-MoE hybrid's expert leg).

Pins: unit KL measured per serving unit and split across members by
n_params; FP8 stays in the menu (measured, not banned); BF16 rows are
passthrough-zero; weights restored after measurement; hybrid merge refuses
double-costed names; backfill only adds missing rows.
"""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from prismaquant.expert_empirical_cost import (
    backfill_missing_from_base,
    measure_expert_unit_costs,
    merge_cost_payloads,
)

from test_packed_expert_cross_domain_gate import TinyLM


class TinyCausal(nn.Module):
    """TinyLM + head so the KL harness sees ``.logits``."""

    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.inner = TinyLM(vocab=vocab, hidden=hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.vocab = vocab

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.inner(input_ids)
        logits = self.head(h).reshape(input_ids.shape[0], -1, self.vocab)
        return types.SimpleNamespace(logits=logits)


EXPERT_NAMES = {
    "inner.mlp.experts.gate_up_proj",
    "inner.mlp.experts.down_proj",
}


def test_measure_unit_costs_on_tiny_packed_moe():
    torch.manual_seed(7)
    model = TinyCausal().eval()
    calib = torch.randint(0, 32, (2, 24))
    before = {
        n: getattr(model.inner.mlp.experts, a).detach().clone()
        for n, a in (("gate_up", "gate_up_proj"), ("down", "down_proj"))
    }

    stats, costs, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "FP8_DYNAMIC", "BF16"],
        expert_chunk=1, progress=False)

    assert set(stats) == EXPERT_NAMES
    assert set(costs) == EXPERT_NAMES
    (unit,) = unit_kls.values()
    assert unit["NVFP4"] > 0.0
    assert unit["FP8_E4M3"] >= 0.0
    # FP8 error should be well below NVFP4 on the same unit.
    assert unit["FP8_E4M3"] < unit["NVFP4"]
    for name in EXPERT_NAMES:
        row = costs[name]
        assert set(row) == {"NVFP4", "FP8_E4M3", "BF16"}
        assert row["BF16"]["predicted_dloss"] == 0.0
        assert row["NVFP4"]["cost_source"] == "empirical_unit_kl"
        assert stats[name]["h_trace"] == 0.0
    # Member shares re-assemble exactly one unit KL per format.
    for fmt in ("NVFP4", "FP8_E4M3"):
        total = sum(costs[n][fmt]["predicted_dloss"] for n in EXPERT_NAMES)
        assert total == pytest.approx(unit[fmt], rel=1e-6)
    # In-place quantize/restore left the model untouched.
    assert torch.equal(
        model.inner.mlp.experts.gate_up_proj.detach(), before["gate_up"])
    assert torch.equal(
        model.inner.mlp.experts.down_proj.detach(), before["down"])


def test_merge_refuses_double_costed_names():
    base = {"stats": {"a": {}}, "costs": {"a": {"NVFP4": {}}}}
    with pytest.raises(RuntimeError, match="collision"):
        merge_cost_payloads(
            base, {"a": {}}, {"a": {"NVFP4": {}}}, formats=["NVFP4", "BF16"])


def test_merge_and_backfill():
    base = {
        "stats": {"lin": {"h_trace": 1.0}},
        "costs": {"lin": {"NVFP4": {"predicted_dloss": 0.5}}},
    }
    merged = merge_cost_payloads(
        base,
        {"experts.down_proj": {"h_trace": 0.0}},
        {"experts.down_proj": {"NVFP4": {"predicted_dloss": 0.1}}},
        formats=["NVFP4", "FP8_DYNAMIC", "BF16"],
    )
    assert set(merged["costs"]) == {"lin", "experts.down_proj"}
    assert merged["formats"] == ["NVFP4", "FP8_E4M3", "BF16"]

    base_cost = {
        "stats": {"mtp.fc": {"h_trace": 2.0}},
        "costs": {
            "mtp.fc": {"NVFP4": {"predicted_dloss": 9.0}},
            "lin": {"NVFP4": {"predicted_dloss": 777.0}},  # must NOT override
        },
    }
    added = backfill_missing_from_base(merged, base_cost)
    assert added == ["mtp.fc"]
    assert merged["costs"]["lin"]["NVFP4"]["predicted_dloss"] == 0.5
    assert merged["costs"]["mtp.fc"]["NVFP4"]["predicted_dloss"] == 9.0
    assert merged["stats"]["mtp.fc"]["h_trace"] == 2.0
