import pytest
import torch
import torch.nn as nn

from prismaquant.aura_cost import _guard_packed_expert_coverage


class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 32, 16))
        self.down_proj = nn.Parameter(torch.zeros(2, 32, 32))


class _PackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = _PackedExperts()


def test_aura_guard_rejects_packed_experts_by_default():
    with pytest.raises(RuntimeError, match="packed-MoE expert costs"):
        _guard_packed_expert_coverage(_PackedModel())


def test_aura_guard_requires_explicit_omission_for_packed_experts():
    omitted = _guard_packed_expert_coverage(
        _PackedModel(),
        allow_omission=True,
    )

    assert omitted == [
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    ]


def test_aura_guard_allows_dense_only_models():
    model = nn.Sequential(nn.Linear(16, 16, bias=False))

    assert _guard_packed_expert_coverage(model) == []
