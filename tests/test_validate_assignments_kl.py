import torch
import torch.nn as nn

from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.validate_assignments_kl import _materialize_assignment_inplace


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(32, 32, bias=False)

    def forward(self, x):
        return self.l(x)


def test_materialize_assignment_inplace_uses_production_cache_tensor():
    model = _TinyModel()
    with torch.no_grad():
        model.l.weight.zero_()
    rendered = torch.full_like(model.l.weight, 3.0)
    cache = ProductionWeightCache(
        weights={("l.weight", "NVFP4"): rendered.clone()},
        levers={},
    )

    stats = _materialize_assignment_inplace(
        model,
        {"l.weight": "NVFP4", "other": "BF16"},
        cache,
    )

    torch.testing.assert_close(model.l.weight, rendered)
    assert stats["copied"] == 1
    assert stats["format_counts"] == {"NVFP4": 1}

