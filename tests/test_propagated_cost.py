from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant import propagated_cost as pc
from prismaquant.allocator_solver import Candidate
from prismaquant.propagated_cost import (
    FrozenBudgetError,
    L3UnsupportedTargetError,
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


def _cost_table(current=1.0, cheaper=1.04):
    return {
        "NVFP4": {"predicted_dloss": cheaper},
        "MXFP8": {"predicted_dloss": current},
        "BF16": {"predicted_dloss": 0.0},
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


def _identity8_spec():
    return fr.FormatSpec(
        name="IDENT8",
        weight_bits=8,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_identity",
        quantize_dequantize=lambda w: w.clone(),
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
        assignment[name] = "NVFP4"
        costs[name] = {
            "NVFP4": {"predicted_dloss": 1.0},
            "MXFP8": {"predicted_dloss": 2.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    assignment["layer0"] = "MXFP8"
    costs["layer0"] = {
        "NVFP4": {"predicted_dloss": 1.04},
        "MXFP8": {"predicted_dloss": 1.0},
        "BF16": {"predicted_dloss": 0.0},
    }
    assignment["layer5"] = "MXFP8"
    costs["layer5"] = {
        "NVFP4": {"predicted_dloss": 2.0},
        "MXFP8": {"predicted_dloss": 1.0},
        "BF16": {"predicted_dloss": 0.0},
    }
    costs["layer19"] = {
        "NVFP4": {"predicted_dloss": 100.0},
        "MXFP8": {"predicted_dloss": 200.0},
        "BF16": {"predicted_dloss": 0.0},
    }

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        _specs(),
        min_fraction=0.05,
        max_fraction=0.15,
        safety_fraction=0.05,
    )

    selected_by_name = {entry.name: entry for entry in selected}
    assert len(selected) == 3
    assert "uncertain" in selected_by_name["layer0"].reasons
    assert "confident_non_cheapest" in selected_by_name["layer5"].reasons
    assert "high_l2_cost" in selected_by_name["layer19"].reasons


def test_select_l3_neighborhood_includes_confident_non_cheapest():
    stats = {"layer": _stat()}
    costs = {
        "layer": {
            "NVFP4": {"predicted_dloss": 10.0},
            "MXFP6_E3M2": {"predicted_dloss": 1.0},
            "MXFP8": {"predicted_dloss": 2.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    assignment = {"layer": "MXFP6_E3M2"}
    specs = [fr.get_format(n) for n in ("NVFP4", "MXFP6_E3M2", "MXFP8", "BF16")]

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        specs,
        min_fraction=1.0,
        max_fraction=1.0,
        safety_fraction=0.0,
    )

    assert len(selected) == 1
    assert selected[0].name == "layer"
    assert selected[0].reasons == ("confident_non_cheapest",)


def test_select_l3_neighborhood_errors_on_packed_experts():
    packed = dict(_stat())
    packed.update({
        "num_experts": 4,
        "_packed_experts_module": "model.layers.0.mlp.experts",
        "_packed_param": "gate_up_proj",
    })
    stats = {
        "model.layers.0.mlp.experts.gate_up_proj": packed,
        "model.layers.0.self_attn.q_proj": _stat(),
    }
    assignment = {name: "MXFP8" for name in stats}
    costs = {name: _cost_table() for name in stats}

    with pytest.raises(L3UnsupportedTargetError) as exc:
        select_l3_neighborhood(
            stats,
            costs,
            assignment,
            _specs(),
            min_fraction=1.0,
            max_fraction=1.0,
            safety_fraction=0.0,
        )

    assert "model.layers.0.mlp.experts.gate_up_proj" in str(exc.value)
    assert "L3 polish does not yet support packed expert tensors" in str(exc.value)


def test_select_l3_neighborhood_passes_dense_only():
    stats = {
        "model.layers.0.self_attn.q_proj": _stat(),
        "model.layers.0.self_attn.k_proj": _stat(),
    }
    assignment = {name: "MXFP8" for name in stats}
    costs = {name: _cost_table() for name in stats}

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        _specs(),
        min_fraction=1.0,
        max_fraction=1.0,
        safety_fraction=0.0,
    )

    assert {entry.name for entry in selected} == set(stats)


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
        bit_precision=0.001,
    )

    assert solved["frozen"] == "MXFP8"
    assert solved["a"] == "MXFP8"
    assert solved["b"] == "NVFP4"
    assert chosen["b"].fmt == "NVFP4"


def test_solve_frozen_l3_neighborhood_falls_back_when_precision_too_tight(monkeypatch):
    stats = {f"layer{i}": _stat(n_params=100) for i in range(15)}
    assignment = {name: "NVFP4" for name in stats}
    candidates = {
        name: [
            Candidate("NVFP4", 4.0, 50, 1.0),
            Candidate("MXFP8", 8.0, 100, 0.0),
            Candidate("BF16", 16.0, 200, 2.0),
        ]
        for name in stats
    }
    monkeypatch.setattr(pc, "solve_allocation", lambda *_args, **_kwargs: None)

    solved, chosen, meta = solve_frozen_l3_neighborhood(
        stats,
        assignment,
        candidates,
        _specs(),
        target_bits=4.4091,
        bit_precision=0.001,
        return_metadata=True,
    )

    used_bits = sum(8.0 * chosen[name].memory_bytes for name in chosen)
    total_params = sum(entry["n_params"] for entry in stats.values())
    assert used_bits <= 4.4091 * total_params + 1e-6
    assert list(solved.values()).count("MXFP8") == 1
    assert meta["frozen_dp_precision_used"] == "greedy"
    assert meta["frozen_dp_greedy"]["accepted"] == 1


def test_solve_frozen_l3_neighborhood_greedy_swaps_dominated_current(monkeypatch):
    stats = {"layer": _stat(n_params=100)}
    assignment = {"layer": "MXFP6_E3M2"}
    candidates = {
        "layer": [
            Candidate("NVFP4", 4.0, 50, 0.1),
            Candidate("MXFP6_E3M2", 6.0, 75, 1.0),
            Candidate("BF16", 16.0, 200, 2.0),
        ]
    }
    monkeypatch.setattr(pc, "solve_allocation", lambda *_args, **_kwargs: None)

    # The target is just below the minimum-bpp candidate, so the removed
    # min-bpp shortcut would have engaged at precision=0.01 before greedy.
    solved, _chosen, meta = solve_frozen_l3_neighborhood(
        stats,
        assignment,
        candidates,
        _specs(),
        target_bits=3.995,
        bit_precision=0.001,
        return_metadata=True,
    )

    assert solved["layer"] == "NVFP4"
    assert meta["frozen_dp_precision_used"] == "greedy"
    assert meta["frozen_dp_greedy"]["accepted"] == 1


def test_solve_frozen_l3_neighborhood_still_raises_when_frozen_exceeds_budget():
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


def test_toy_l3_propagation_differs_from_local_cost_and_flips_pick(tmp_path):
    model = _AmplifyingToy().eval()
    stats = {
        "l1": {
            "n_params": 4,
            "in_features": 2,
            "out_features": 2,
            "h_trace": 1.0,
            "_memory_bytes_by_format": {
                "ZERO4": 2,
                "IDENT8": 4,
                "BF16": 8,
            },
        }
    }
    l2_costs = {
        "l1": {
            "ZERO4": {"predicted_dloss": 0.01},
            "IDENT8": {"predicted_dloss": 0.02},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    assignment = {"l1": "ZERO4", "l2": "BF16", "l3": "BF16"}
    specs = [_zero_spec(), _identity8_spec(), fr.get_format("BF16")]
    selected = select_l3_neighborhood(
        stats,
        l2_costs,
        {"l1": "ZERO4"},
        specs,
        min_fraction=1.0,
        max_fraction=1.0,
    )
    calib = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)

    l3_costs = measure_propagated_costs(
        model,
        assignment,
        selected,
        calib,
        specs,
        work_root=tmp_path,
        max_lanes_per_batch=8,
    )
    l3_candidates = build_l3_candidates(stats, l3_costs, specs)
    solved, _chosen = solve_frozen_l3_neighborhood(
        stats,
        {"l1": "ZERO4"},
        l3_candidates,
        specs,
        target_bits=8.0,
        bit_precision=0.5,
    )

    assert l3_costs["l1"]["ZERO4"]["propagated_end_kl"] != pytest.approx(
        l2_costs["l1"]["ZERO4"]["predicted_dloss"]
    )
    assert l3_costs["l1"]["ZERO4"]["propagated_end_kl"] > (
        l3_costs["l1"]["IDENT8"]["propagated_end_kl"]
    )
    assert solved["l1"] == "IDENT8"
