from __future__ import annotations

import pytest

from prismaquant.allocator import Candidate
from prismaquant.allocator_prune import (
    aggregate_moe_candidates,
    apply_consensus_prune,
    apply_global_prune_ratio,
    apply_nested_global_prune_ratio,
    build_prune_manifest,
    expand_moe_assignment,
)
from prismaquant.expert_prune import ExpertPruneDisabledError
import prismaquant.format_registry as fr


def _stats_costs_candidates():
    name = "model.layers.0.mlp.experts.0.gate_proj"
    stats = {
        name: {
            "h_trace": 0.1,
            "w_max_abs": 1.0,
            "w_norm_sq": 1.0,
            "n_params": 100,
            "in_features": 10,
            "out_features": 10,
        }
    }
    costs = {
        name: {
            "NVFP4": {
                "weight_mse": 0.01,
                "output_mse": 0.01,
                "predicted_dloss": 0.001,
            }
        }
    }
    candidates = {
        name: [Candidate("NVFP4", 4.25, 100, 0.001)]
    }
    return stats, costs, candidates


def test_no_prune_moe_aggregation_still_works():
    stats, costs, candidates = _stats_costs_candidates()
    stats_ext, costs_ext, cands_ext = aggregate_moe_candidates(
        stats,
        costs,
        [fr.get_format("NVFP4")],
        candidates,
        granularity="projection",
        prune_ratios=(),
    )
    assert "model.layers.0.mlp.experts.__fused__.gate_proj" in stats_ext
    assert "model.layers.0.mlp.experts.__fused__.gate_proj" in costs_ext
    assert "model.layers.0.mlp.experts.__fused__.gate_proj" in cands_ext


def test_prune_candidate_generation_throws():
    stats, costs, candidates = _stats_costs_candidates()
    with pytest.raises(ExpertPruneDisabledError, match="disabled"):
        aggregate_moe_candidates(
            stats,
            costs,
            [fr.get_format("NVFP4")],
            candidates,
            prune_ratios=(0.5,),
        )


def test_global_prune_helpers_throw_on_nonzero_or_nonempty_attempts():
    with pytest.raises(ExpertPruneDisabledError, match="disabled"):
        apply_global_prune_ratio(
            {"x": [Candidate("NVFP4", 4.25, 100, 0.0)]},
            {"x": {"num_experts": 2}},
            {"r": {0: 0.0, 1: 1.0}},
            global_ratio=0.5,
        )
    with pytest.raises(ExpertPruneDisabledError, match="disabled"):
        apply_nested_global_prune_ratio(
            {"x": [Candidate("NVFP4", 4.25, 100, 0.0)]},
            {"x": {"_num_experts_total": 2}},
            global_ratio=0.5,
        )
    with pytest.raises(ExpertPruneDisabledError, match="disabled"):
        build_prune_manifest({"x": (1,)}, {}, {})
    with pytest.raises(ExpertPruneDisabledError, match="disabled"):
        apply_consensus_prune({"x": (1,)}, {}, {}, {})
    with pytest.raises(ExpertPruneDisabledError, match="disabled"):
        expand_moe_assignment({"x": "NVFP4"}, {}, pruned_map={"x": (1,)})


def test_zero_or_empty_prune_inputs_remain_noops():
    candidates = {"x": [Candidate("NVFP4", 4.25, 100, 0.0)]}
    assert apply_global_prune_ratio(
        candidates, {"x": {"num_experts": 2}}, {}, global_ratio=0.0,
    ) == 0
    filtered, warnings = apply_nested_global_prune_ratio(
        candidates, {"x": {"_num_experts_total": 2}}, global_ratio=0.0,
    )
    assert filtered == candidates
    assert warnings == []
    assert build_prune_manifest({}, {}, {}) == ({}, [])
    assert apply_consensus_prune({}, {}, {}, {}) == {}
    assert expand_moe_assignment({"x": "NVFP4"}, {}) == {"x": "NVFP4"}
