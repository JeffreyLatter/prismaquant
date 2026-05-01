import torch
import torch.nn as nn

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
