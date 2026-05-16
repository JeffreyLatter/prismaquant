import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import is_fused_moe_experts
from prismaquant.build_rtn_cache import iter_quantizable_tensors


class _ToyPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 32, 32))
        self.down_proj = nn.Parameter(torch.zeros(2, 32, 32))
        self.kernel = nn.Parameter(torch.zeros(2, 32, 32))


class _ToyPackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = _ToyPackedExperts()


class _NoPackedProfile:
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset()


class _CustomPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13 = nn.Parameter(torch.zeros(2, 64, 32))
        self.w2 = nn.Parameter(torch.zeros(2, 32, 32))
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 64, 32))


class _CustomPackedProfile:
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset({"w13", "w2"})


class _LegacyContainerProfile:
    def packed_expert_module_class_names(self) -> frozenset[str]:
        return frozenset({"LegacyPackedExperts"})


LegacyPackedExperts = type("LegacyPackedExperts", (nn.Module,), {})


def test_iter_quantizable_tensors_covers_generic_packed_experts():
    model = _ToyPackedModel()

    yielded = list(iter_quantizable_tensors(model))
    names = {name for name, _mod, _attr in yielded}
    attrs = {attr for _name, _mod, attr in yielded}

    assert names == {
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    }
    assert attrs == {"down_proj", "gate_up_proj"}


def test_iter_quantizable_tensors_respects_profile_packed_names():
    model = nn.Module()
    model.experts = _CustomPackedExperts()

    yielded = list(iter_quantizable_tensors(model, _CustomPackedProfile()))
    names = {name for name, _mod, _attr in yielded}

    assert names == {"experts.w13", "experts.w2"}


def test_iter_quantizable_tensors_allows_profile_to_disable_packed_names():
    model = _ToyPackedModel()

    yielded = list(iter_quantizable_tensors(model, _NoPackedProfile()))

    assert yielded == []


def test_legacy_fused_expert_class_names_are_profile_owned():
    module = LegacyPackedExperts()

    assert is_fused_moe_experts(module, _LegacyContainerProfile())
    assert not is_fused_moe_experts(module)
