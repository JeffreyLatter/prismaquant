import torch
import torch.nn as nn

from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.validate_assignments_kl import (
    _assignment_cost_summary,
    _materialize_assignment_inplace,
)


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


def test_assignment_cost_summary_reports_local_mse_and_aliases():
    costs = {
        "a": {
            "NVFP4": {
                "weight_mse": 1.0,
                "output_mse": 2.0,
                "fisher_output_mse": 1.5,
                "rel_output_mse": 0.2,
                "predicted_dloss": 3.0,
            },
        },
        "b": {
            "MXFP8_E4M3": {
                "weight_mse": 0.5,
                "output_mse": 0.25,
                "fisher_output_mse": 0.10,
                "rel_output_mse": 0.025,
            },
        },
    }
    summary = _assignment_cost_summary(
        costs,
        {"a": "NVFP4", "b": "MXFP8", "c": "BF16", "missing": "NVFP4"},
    )

    assert abs(summary["output_mse_sum"] - 2.25) < 1e-12
    assert abs(summary["fisher_output_mse_sum"] - 1.60) < 1e-12
    assert abs(summary["weight_mse_sum"] - 1.5) < 1e-12
    assert abs(summary["rel_output_mse_sum"] - 0.225) < 1e-12
    assert abs(summary["predicted_dloss_sum"] - 3.0) < 1e-12
    assert summary["counts"]["output_mse"] == 3
    assert summary["counts"]["fisher_output_mse"] == 2
    assert summary["counts"]["predicted_dloss"] == 1
    assert summary["missing_count"] == 1
    assert summary["missing_sample"] == ["missing"]
