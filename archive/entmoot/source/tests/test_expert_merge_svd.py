from __future__ import annotations

import json

import torch

from prismaquant.entmoot import (
    build_router_id_merge_plan,
    build_expert_merge_plan,
    cosine_from_gram,
    expert_weight_feature,
    gram_from_features,
    merge_plan_manifest,
    rank_expert_subsumability,
    save_merge_manifest,
    select_rrqr_anchors_from_gram,
)
from prismaquant.schemas import validate_merge_manifest_payload


def test_rank_subsumability_marks_duplicate_as_subsumed():
    features = {
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([1.0, 0.0, 0.0]),
        2: torch.tensor([0.0, 1.0, 0.0]),
    }
    ids, gram = gram_from_features(features)
    ranked = rank_expert_subsumability(
        gram, ids, {0: 10.0, 1: 1.0, 2: 5.0}, basis_size=1, ridge=0.0,
    )

    by_id = {r.expert_id: r for r in ranked}
    assert by_id[1].neighbor_expert_ids == (0,)
    assert by_id[1].subsumed_fraction == 1.0
    assert by_id[1].predicted_merge_loss == 0.0
    assert by_id[2].subsumed_fraction == 0.0
    assert by_id[2].predicted_merge_loss == 5.0


def test_rank_uses_marginal_value_for_unique_experts():
    ids, gram = gram_from_features([
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ])
    ranked = rank_expert_subsumability(
        gram, ids, {0: 100.0, 1: 1.0}, basis_size=1, ridge=0.0,
    )

    assert [r.expert_id for r in ranked] == [1, 0]
    assert ranked[0].predicted_merge_loss == 1.0
    assert ranked[1].predicted_merge_loss == 100.0


def test_merge_plan_clusters_duplicate_before_unique_expert():
    features = {
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([1.0, 0.0, 0.0]),
        2: torch.tensor([0.0, 1.0, 0.0]),
    }
    ids, gram = gram_from_features(features)
    plan = build_expert_merge_plan(
        gram,
        ids,
        {0: 10.0, 1: 1.0, 2: 9.0},
        target_experts=2,
        router_qname="model.layers.0.mlp.gate",
        ridge=0.0,
    )

    clusters = [set(c.original_expert_ids) for c in plan.clusters]
    assert {0, 1} in clusters
    assert {2} in clusters
    assert plan.num_experts_orig == 3
    assert plan.num_experts_kept == 2
    assert plan.total_predicted_merge_loss == 0.0

    manifest = plan.to_manifest()
    entry = manifest["model.layers.0.mlp.gate"]
    assert entry["orig_to_new_eid"]["0"] == entry["orig_to_new_eid"]["1"]
    assert entry["orig_to_new_eid"]["2"] != entry["orig_to_new_eid"]["0"]


def test_merge_plan_normalizes_cluster_weights_by_marginal_value():
    ids, gram = gram_from_features({
        7: torch.tensor([1.0, 0.0]),
        9: torch.tensor([1.0, 0.0]),
    })
    plan = build_expert_merge_plan(
        gram, ids, {7: 3.0, 9: 1.0}, target_experts=1, ridge=0.0,
    )

    weights = dict(plan.clusters[0].weights)
    assert weights == {7: 0.75, 9: 0.25}
    assert plan.clusters[0].anchor_expert_id == 7


def test_expert_weight_feature_is_projection_order_stable_and_normalized():
    a = expert_weight_feature({
        "down_proj": torch.tensor([[3.0, 0.0]]),
        "gate_proj": torch.tensor([[0.0, 4.0]]),
    })
    b = expert_weight_feature({
        "gate_proj": torch.tensor([[0.0, 2.0]]),
        "down_proj": torch.tensor([[6.0, 0.0]]),
    })
    ids, gram = gram_from_features({0: a, 1: b})
    cosine = cosine_from_gram(gram)

    assert torch.allclose(cosine, torch.ones(2, 2, dtype=torch.float64))


def test_save_merge_manifest_writes_json(tmp_path):
    ids, gram = gram_from_features({
        0: torch.tensor([1.0]),
        1: torch.tensor([1.0]),
    })
    plan = build_expert_merge_plan(
        gram, ids, {0: 1.0, 1: 1.0}, target_experts=1,
        router_qname="router",
    )
    out = tmp_path / "merge_manifest.json"
    save_merge_manifest([plan], out)

    loaded = json.loads(out.read_text())
    assert loaded == merge_plan_manifest([plan])
    assert loaded["router"]["method"] == "entmoot_svd_gram_v1"


def test_rrqr_anchor_selection_keeps_diverse_experts():
    ids, gram = gram_from_features({
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([0.9, 0.1, 0.0]),
        2: torch.tensor([0.0, 1.0, 0.0]),
        3: torch.tensor([0.0, 0.0, 1.0]),
    })

    anchors = select_rrqr_anchors_from_gram(gram, ids, target_experts=3)

    assert anchors == [0, 2, 3]


def test_router_id_plan_identity_keeps_rejected_expert():
    features = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.98, 0.02]),
        2: torch.tensor([0.0, 1.0]),
        3: torch.tensor([-1.0, 0.0]),
    }

    plan = build_router_id_merge_plan(
        features,
        {0: 10.0, 1: 1.0, 2: 9.0, 3: 1.0},
        target_experts=2,
        router_qname="router",
        activation_accept_threshold=0.01,
    )
    entry = plan.to_manifest()["router"]
    validate_merge_manifest_payload(plan.to_manifest(), "entmoot_router_id_v1.json")

    assert entry["method"] == "entmoot_router_id_v1"
    assert entry["router_strategy"] == "anchor"
    assert entry["num_experts_kept"] == 3
    assert entry["orig_to_new_eid"]["0"] == entry["orig_to_new_eid"]["1"]
    assert entry["orig_to_new_eid"]["3"] != entry["orig_to_new_eid"]["0"]

    decisions = {d["expert_id"]: d for d in entry["expert_decisions"]}
    assert decisions[1]["decision"] == "accept"
    assert decisions[3]["decision"] == "reject"
    identity_cluster = next(
        c for c in entry["clusters"] if c["anchor_expert_id"] == 3
    )
    assert identity_cluster["merge_action"] == "identity"
    assert identity_cluster["weights"] == {"3": 1.0}


def test_router_id_plan_uses_one_hot_expert_weights_and_router_mass_weights():
    features = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.99, 0.01]),
    }
    plan = build_router_id_merge_plan(
        features,
        {0: 1.0, 1: 1.0},
        target_experts=1,
        routed_mass={0: 1.0, 1: 3.0},
        activation_accept_threshold=0.01,
    )
    cluster = plan.clusters[0].to_dict()

    assert cluster["weights"] == {"0": 1.0, "1": 0.0}
    assert cluster["router_weights"] == {"0": 0.25, "1": 0.75}


def test_router_id_plan_can_use_measured_anchor_residuals():
    features = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
    }
    plan = build_router_id_merge_plan(
        features,
        {0: 1.0, 1: 1.0},
        target_experts=1,
        router_qname="router",
        activation_accept_threshold=0.01,
        anchor_residuals={(1, 0): 0.001},
        candidate_anchor_ids={1: [0]},
    )
    entry = plan.to_manifest()["router"]
    decisions = {d["expert_id"]: d for d in entry["expert_decisions"]}

    assert decisions[1]["decision"] == "accept"
    assert decisions[1]["activation_energy_relative"] == 0.001


def test_router_id_plan_can_emit_synthesized_pair_weights():
    features = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
    }
    plan = build_router_id_merge_plan(
        features,
        {0: 1.0, 1: 1.0},
        target_experts=1,
        router_qname="router",
        activation_accept_threshold=0.01,
        anchor_residuals={(1, 0): 0.001},
        candidate_anchor_ids={1: [0]},
        synthesis_weights={(1, 0): {0: 0.7, 1: 0.3}},
    )
    entry = plan.to_manifest()["router"]
    cluster = entry["clusters"][0]
    decisions = {d["expert_id"]: d for d in entry["expert_decisions"]}

    assert entry["num_experts_kept"] == 1
    assert cluster["weights"] == {"0": 0.7, "1": 0.3}
    assert decisions[1]["decision"] == "accept"
    assert decisions[1]["synthesis_weights"] == {"0": 0.7, "1": 0.3}


def test_router_id_plan_can_emit_tensor_specific_synthesis_weights():
    features = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
    }
    plan = build_router_id_merge_plan(
        features,
        {0: 1.0, 1: 1.0},
        target_experts=1,
        router_qname="router",
        activation_accept_threshold=0.01,
        anchor_residuals={(1, 0): 0.001},
        candidate_anchor_ids={1: [0]},
        synthesis_weights={(1, 0): {0: 0.55, 1: 0.45}},
        tensor_synthesis_weights={
            (1, 0): {
                "gate_up_proj": {0: 0.8, 1: 0.2},
                "down_proj": {0: 0.3, 1: 0.7},
            }
        },
    )
    cluster = plan.to_manifest()["router"]["clusters"][0]

    assert cluster["weights"] == {"0": 0.55, "1": 0.45}
    assert cluster["tensor_weights"]["gate_up_proj"] == {"0": 0.8, "1": 0.2}
    assert cluster["tensor_weights"]["down_proj"] == {"0": 0.3, "1": 0.7}


def test_router_id_synthesis_accepts_at_most_one_drop_per_anchor():
    features = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
        2: torch.tensor([0.0, 0.9]),
    }
    plan = build_router_id_merge_plan(
        features,
        {0: 1.0, 1: 1.0, 2: 1.0},
        target_experts=1,
        router_qname="router",
        activation_accept_threshold=0.01,
        anchor_residuals={(1, 0): 0.002, (2, 0): 0.001},
        candidate_anchor_ids={1: [0], 2: [0]},
        synthesis_weights={
            (1, 0): {0: 0.7, 1: 0.3},
            (2, 0): {0: 0.8, 2: 0.2},
        },
    )
    entry = plan.to_manifest()["router"]
    decisions = {d["expert_id"]: d for d in entry["expert_decisions"]}

    assert decisions[2]["decision"] == "accept"
    assert decisions[1]["decision"] == "reject"
    assert decisions[1]["rejection_reason"] == (
        "synthesis candidate conflicted with a lower-residual pair"
    )
    assert entry["num_experts_kept"] == 2


def test_router_id_plan_no_accepted_merges_preserves_original_order():
    features = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
        2: torch.tensor([-1.0, 0.0]),
    }
    plan = build_router_id_merge_plan(
        features,
        {0: 1.0, 1: 1.0, 2: 1.0},
        target_experts=1,
        router_qname="router",
        activation_accept_threshold=0.0,
    )
    entry = plan.to_manifest()["router"]

    assert entry["num_experts_kept"] == 3
    assert entry["kept_expert_ids"] == [0, 1, 2]
    assert entry["orig_to_new_eid"] == {"0": 0, "1": 1, "2": 2}
    assert [c["weights"] for c in entry["clusters"]] == [
        {"0": 1.0},
        {"1": 1.0},
        {"2": 1.0},
    ]
