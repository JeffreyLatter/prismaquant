from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.allocator_solver import Candidate
from prismaquant.propagated_cost import (
    FrozenBudgetError,
    L3NeighborhoodEntry,
    build_l3_candidates,
    measure_propagated_costs,
    select_formats_for_l3,
    select_l3_neighborhood,
    solve_frozen_l3_neighborhood,
)


def _specs():
    return [fr.get_format(n) for n in ("NVFP4", "MXFP8", "BF16")]


def _stat(n_params=128 * 128):
    return {
        "n_params": n_params,
        "in_features": 128,
        "out_features": 128,
        "h_trace": 1.0,
        "_memory_bytes_by_format": {
            "NVFP4": int(n_params * 4 / 8),
            "MXFP8": int(n_params * 8 / 8),
            "BF16": int(n_params * 16 / 8),
        },
    }


class _AmplifyingToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2, bias=False)
        self.l2 = nn.Linear(2, 2, bias=False)
        self.l3 = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.l1.weight.copy_(torch.eye(2))
            self.l2.weight.copy_(10.0 * torch.eye(2))
            self.l3.weight.copy_(torch.eye(2))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return SimpleNamespace(logits=x)


def _zero_spec():
    return fr.FormatSpec(
        name="ZERO4",
        weight_bits=4,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_zero",
        quantize_dequantize=lambda w: torch.zeros_like(w),
    )


def test_select_formats_uses_current_neighbors_and_bf16():
    stats = {"layer": _stat()}
    costs = {
        "layer": {
            "NVFP4": {"predicted_dloss": 1.05},
            "MXFP8": {"predicted_dloss": 1.00},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    assignment = {"layer": "MXFP8"}

    got = select_formats_for_l3(stats, costs, assignment, "layer", _specs())

    assert got == ("NVFP4", "MXFP8", "BF16")


def test_select_l3_neighborhood_caps_and_keeps_safety_layers():
    stats = {f"layer{i}": _stat() for i in range(20)}
    costs = {}
    assignment = {}
    for i in range(20):
        name = f"layer{i}"
        assignment[name] = "MXFP8"
        current = 100.0 if i == 19 else 1.0
        cheaper = current * (1.04 if i < 12 else 2.0)
        costs[name] = {
            "NVFP4": {"predicted_dloss": cheaper},
            "MXFP8": {"predicted_dloss": current},
            "BF16": {"predicted_dloss": 0.0},
        }

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        _specs(),
        min_fraction=0.05,
        max_fraction=0.10,
        safety_fraction=0.05,
    )

    assert len(selected) == 2
    assert "layer19" in {entry.name for entry in selected}
    assert any("high_l2_cost" in entry.reasons for entry in selected)


def test_build_l3_candidates_uses_propagated_end_kl_only():
    stats = {"layer": _stat()}
    propagated = {
        "layer": {
            "NVFP4": {"propagated_end_kl": 0.5, "downstream_output_mse": 10.0},
            "MXFP8": {"output_mse": 0.01},
            "BF16": {"propagated_end_kl": 0.0},
        }
    }

    cands = build_l3_candidates(stats, propagated, _specs())

    assert [c.fmt for c in cands["layer"]] == ["NVFP4", "BF16"]
    assert [c.predicted_dloss for c in cands["layer"]] == [0.5, 0.0]


def test_solve_frozen_l3_neighborhood_respects_remaining_budget():
    stats = {name: _stat(n_params=100) for name in ("a", "b", "frozen")}
    assignment = {"a": "MXFP8", "b": "MXFP8", "frozen": "MXFP8"}
    candidates = {
        "a": [
            Candidate("NVFP4", 4.0, 50, 3.0),
            Candidate("MXFP8", 8.0, 100, 1.0),
            Candidate("BF16", 16.0, 200, 0.0),
        ],
        "b": [
            Candidate("NVFP4", 4.0, 50, 0.1),
            Candidate("MXFP8", 8.0, 100, 1.0),
            Candidate("BF16", 16.0, 200, 0.0),
        ],
    }

    solved, chosen = solve_frozen_l3_neighborhood(
        stats,
        assignment,
        candidates,
        _specs(),
        target_bits=8.0,
        bit_precision=0.5,
    )

    assert solved["frozen"] == "MXFP8"
    assert solved["a"] == "MXFP8"
    assert solved["b"] == "NVFP4"
    assert chosen["b"].fmt == "NVFP4"


def test_solve_frozen_l3_neighborhood_rejects_over_budget_frozen_choices():
    stats = {name: _stat(n_params=100) for name in ("open", "frozen")}
    assignment = {"open": "MXFP8", "frozen": "BF16"}
    candidates = {"open": [Candidate("NVFP4", 4.0, 50, 0.0)]}

    with pytest.raises(FrozenBudgetError, match="frozen L2 choices already exceed"):
        solve_frozen_l3_neighborhood(
            stats,
            assignment,
            candidates,
            _specs(),
            target_bits=4.0,
            bit_precision=0.5,
        )


def test_measure_propagated_costs_pairs_candidate_with_target_bf16_baseline(tmp_path):
    model = _AmplifyingToy().eval()
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    neighborhood = [
        L3NeighborhoodEntry(
            name="l1",
            current_format="BF16",
            formats=("ZERO4", "BF16"),
            margin=0.0,
            l2_current_cost=0.0,
        )
    ]
    calib = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)

    costs = measure_propagated_costs(
        model,
        assignment,
        neighborhood,
        calib,
        [_zero_spec(), fr.get_format("BF16")],
        work_root=tmp_path,
        max_lanes_per_batch=4,
    )

    zero = costs["l1"]["ZERO4"]
    assert zero["propagated_end_kl"] > 0.1
    assert zero["downstream_output_mse"] > 0.0
    assert costs["l1"]["BF16"]["propagated_end_kl"] == 0.0
