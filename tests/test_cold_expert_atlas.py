from __future__ import annotations

import pickle

import torch

from scripts.cold_expert_atlas import (
    derive_cold_experts,
    invert_tid2eid,
    router_lens_topk,
)


def test_tid2eid_inversion_retains_empty_experts():
    mapping = torch.tensor([[2, 0], [0, 1], [2, 1]], dtype=torch.int64)

    assert invert_tid2eid(mapping, num_experts=4) == {
        0: [0, 1],
        1: [1, 2],
        2: [0, 2],
        3: [],
    }


def test_router_lens_topk_matches_fabricated_affinities():
    gate_rows = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    embeddings = torch.tensor(
        [[1.0, 0.0], [3.0, 2.0], [-2.0, -4.0], [0.5, -3.0]]
    )

    values, indices = router_lens_topk(
        gate_rows, embeddings, top_k=2, batch_size=1
    )

    assert indices.tolist() == [[1, 0], [2, 3]]
    torch.testing.assert_close(values, torch.tensor([[3.0, 1.0], [4.0, 3.0]]))


def test_cold_list_derivation_from_cost_table_pickle(tmp_path):
    def qname(expert: int, projection: str) -> str:
        return f"model.layers.3.mlp.experts.{expert}.{projection}"

    costs = {}
    stats = {}
    for expert, h_trace, rows in ((7, 0.0, 0), (8, 1.25, 3)):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = qname(expert, projection)
            costs[name] = {"BF16": {"n_activation_rows": rows}}
            stats[name] = {"h_trace": h_trace}
    path = tmp_path / "cost.pkl"
    with path.open("wb") as handle:
        pickle.dump({"costs": costs, "stats": stats}, handle)

    cold = derive_cold_experts(path)

    assert [(item["layer"], item["expert_id"]) for item in cold] == [(3, 7)]
    assert cold[0]["h_trace"] == 0.0
    assert cold[0]["n_activation_rows"] == 0
    assert cold[0]["qnames"] == [
        qname(7, "gate_proj"),
        qname(7, "up_proj"),
        qname(7, "down_proj"),
    ]
