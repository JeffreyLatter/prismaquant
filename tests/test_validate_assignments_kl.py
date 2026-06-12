import torch
import torch.nn as nn

from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.validate_assignments_kl import (
    _assignment_cost_summary,
    _calibration_provenance,
    _materialize_assignment_inplace,
    _production_cache_assignment_diagnostics,
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


def test_calibration_provenance_hashes_repeats():
    repeat_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
    repeat_b = torch.tensor([[4, 5, 6]], dtype=torch.long)

    single = _calibration_provenance([repeat_a])
    combined = _calibration_provenance([repeat_a, repeat_b])

    assert single["calib_hash"] == single["calib_repeat_hashes"][0]
    assert len(combined["calib_repeat_hashes"]) == 2
    assert combined["calib_hash"] != single["calib_hash"]


def test_production_cache_assignment_diagnostics_counts_misses(monkeypatch):
    cache = ProductionWeightCache(
        weights={("l.weight", "NVFP4"): torch.zeros(1, 1)},
        levers={},
    )
    assignment = {
        "l.weight": "NVFP4",
        "missing.weight": "NVFP4",
        "source.weight": "BF16",
    }

    strict = _production_cache_assignment_diagnostics(cache, assignment)
    assert strict["required_entries"] == 2
    assert strict["cache_hit_count"] == 1
    assert strict["cache_miss_count"] == 1
    assert strict["rtn_fallback_count"] == 0
    assert strict["strict"] is True

    monkeypatch.setenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", "0")
    permissive = _production_cache_assignment_diagnostics(cache, assignment)
    assert permissive["cache_hit_count"] == 1
    assert permissive["cache_miss_count"] == 1
    assert permissive["rtn_fallback_count"] == 1
    assert permissive["strict"] is False
