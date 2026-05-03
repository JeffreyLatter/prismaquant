from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from prismaquant.entmoot_collector import (
    EntmootActivationCollector,
    LayerSketchBuffer,
    load_collector_state,
)


class TinyPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 4, 2))
        self.down_proj = nn.Parameter(torch.zeros(2, 2, 2))
        self.act_fn = nn.Identity()
        with torch.no_grad():
            # Expert 0: gate=[x0,x1], up=[x0,x1], down identity.
            self.gate_up_proj[0, 0, 0] = 1.0
            self.gate_up_proj[0, 1, 1] = 1.0
            self.gate_up_proj[0, 2, 0] = 1.0
            self.gate_up_proj[0, 3, 1] = 1.0
            self.down_proj[0] = torch.eye(2)
            # Expert 1: same, but output doubled.
            self.gate_up_proj[1] = self.gate_up_proj[0]
            self.down_proj[1] = 2.0 * torch.eye(2)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        mask = mask.permute(2, 1, 0)
        for eid in torch.nonzero(mask.sum(dim=(-1, -2)) > 0).flatten().tolist():
            top_pos, tok = torch.where(mask[eid])
            cur = hidden_states[tok]
            gate, up = torch.nn.functional.linear(cur, self.gate_up_proj[eid]).chunk(2, -1)
            out = torch.nn.functional.linear(gate * up, self.down_proj[eid])
            final.index_add_(0, tok, out * top_k_weights[tok, top_pos, None])
        return final


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.experts = TinyPackedExperts()


def test_layer_sketch_buffer_records_weighted_output_features():
    buf = LayerSketchBuffer("router", num_experts=2, max_samples_per_expert=4)
    buf.add_router_batch(torch.ones(3, 2))
    buf.add_expert_batch(
        0,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
        torch.tensor([1.0, 3.0]),
    )

    ids, features = buf.output_feature_matrix()

    assert ids == [0, 1]
    assert torch.allclose(features[0], torch.tensor([0.5, 3.0], dtype=torch.float64))
    assert torch.allclose(features[1], torch.zeros(2, dtype=torch.float64))
    assert buf.total_tokens == 3
    assert buf.stats()[0].samples == 2


def test_layer_sketch_buffer_round_trips_saved_state(tmp_path):
    buf = LayerSketchBuffer("router", num_experts=2, max_samples_per_expert=4)
    buf.add_router_batch(torch.ones(3, 2))
    buf.add_expert_batch(
        1,
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([0.5]),
    )
    path = tmp_path / "collector.pt"
    torch.save({
        "format": "entmoot_activation_collector_v1",
        "layers": {"router": buf.state_dict()},
    }, path)

    loaded = load_collector_state(path)

    assert set(loaded) == {"router"}
    assert loaded["router"].total_tokens == 3
    assert loaded["router"].stats()[1].samples == 1
    _ids, features = loaded["router"].output_feature_matrix()
    assert torch.allclose(features[1], torch.tensor([3.0, 4.0], dtype=torch.float64))


def test_qwen_packed_collector_patch_preserves_forward_and_collects_samples():
    model = TinyModel()
    experts = model.layers[0].mlp.experts
    hidden = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    topk_i = torch.tensor([[0, 1], [1, 0]])
    topk_w = torch.tensor([[0.75, 0.25], [0.60, 0.40]])
    expected = experts(hidden, topk_i, topk_w)

    collector = EntmootActivationCollector(
        model,
        packed_moe_blocks=[{
            "router_qname": "layers.0.mlp.gate",
            "experts_qname": "layers.0.mlp.experts",
            "num_experts": 2,
        }],
        max_samples_per_expert=8,
    )
    try:
        got = experts(hidden, topk_i, topk_w)
    finally:
        collector.remove_hooks()

    assert torch.allclose(got, expected)
    layer = collector.layers["layers.0.mlp.gate"]
    assert layer.total_tokens == 2
    assert layer.stats()[0].samples == 2
    assert layer.stats()[1].samples == 2
    assert layer.stats()[0].routed_mass == pytest.approx(1.15)
    assert layer.stats()[1].routed_mass == pytest.approx(0.85)
