from __future__ import annotations

import torch

from prismaquant.entmoot_router_diag import (
    choose_router_strategy,
    clustered_router_target,
    router_weight_for_strategy,
    topk_router_distribution,
)


def _entry():
    return {
        "num_experts_orig": 4,
        "num_experts_kept": 2,
        "kept_expert_ids": [0, 2],
        "orig_to_new_eid": {"0": 0, "1": 0, "2": 1, "3": 1},
        "clusters": [
            {
                "new_expert_id": 0,
                "anchor_expert_id": 0,
                "original_expert_ids": [0, 1],
                "weights": {"0": 1.0, "1": 0.0},
                "router_weights": {"0": 0.25, "1": 0.75},
            },
            {
                "new_expert_id": 1,
                "anchor_expert_id": 2,
                "original_expert_ids": [2, 3],
                "weights": {"2": 1.0, "3": 0.0},
                "router_weights": {"2": 0.5, "3": 0.5},
            },
        ],
    }


def test_topk_router_distribution_scatter_shape_and_mass():
    logits = torch.tensor([[5.0, 4.0, 0.0], [0.0, 1.0, 2.0]])
    probs = topk_router_distribution(logits, top_k=2)

    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    assert probs[0, 2] == 0.0
    assert probs[1, 0] == 0.0


def test_clustered_router_target_sums_old_expert_mass():
    logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
    target = clustered_router_target(
        logits,
        {"0": 0, "1": 0, "2": 1, "3": 1},
        num_new_experts=2,
        top_k=2,
    )

    assert torch.allclose(target, torch.tensor([[1.0, 0.0]], dtype=torch.float64))


def test_router_weight_strategy_uses_router_weights_for_weighted_average():
    old = torch.tensor([
        [1.0, 0.0],
        [3.0, 0.0],
        [0.0, 2.0],
        [0.0, 4.0],
    ])

    anchor = router_weight_for_strategy(old, _entry(), "anchor")
    weighted = router_weight_for_strategy(old, _entry(), "weighted_average")

    assert torch.allclose(anchor, torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    assert torch.allclose(weighted, torch.tensor([[2.5, 0.0], [0.0, 3.0]]))


def test_choose_router_strategy_selects_lower_kl_passing_strategy():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old_router = torch.tensor([
        [1.0, 0.0],
        [3.0, 0.0],
        [0.0, 2.0],
        [0.0, 4.0],
    ])
    choice = choose_router_strategy(
        hidden,
        old_router,
        _entry(),
        top_k=2,
        top1_floor=1.0,
        topk_floor=1.0,
        kl_cap=10.0,
    )

    assert choice.selected_strategy == "weighted_average"
    assert {m.strategy for m in choice.metrics} == {"anchor", "weighted_average"}
