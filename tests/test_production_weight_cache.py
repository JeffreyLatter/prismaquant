from __future__ import annotations

import torch
import torch.nn as nn

from prismaquant.kl_sensitivity_probe import _normalized_production_cache_levers
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.production_weight_cache import fill_production_weight_cache


def test_prefetch_loads_disk_entries_and_respects_lru(tmp_path):
    weights = {}
    tensor_nbytes = 0
    for idx, name in enumerate(("a", "b", "c")):
        tensor = torch.full((2, 2), float(idx), dtype=torch.float32)
        tensor_nbytes = tensor.numel() * tensor.element_size()
        path = tmp_path / f"{name}.pt"
        torch.save(tensor, path)
        weights[(name, "NVFP4")] = path.name

    cache = ProductionWeightCache(
        weights=weights,
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )
    cache.enable_lru(2 * tensor_nbytes)

    assert cache.prefetch(max_workers=2) == 3
    resident = sum(isinstance(value, torch.Tensor) for value in cache.weights.values())

    assert resident <= 2
    assert cache._lru_bytes <= 2 * tensor_nbytes
    assert torch.equal(cache.get("a", "NVFP4"), torch.zeros((2, 2)))


def test_production_cache_records_damp_sweep_lever(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0")
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=[],
        levers={"gptq": True},
        progress=False,
    )

    assert cache.levers["gptq_damp_sweep"] is False
    assert _normalized_production_cache_levers(
        "gptq,scale_sweep"
    )["gptq_damp_sweep"] is False

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1")
    assert _normalized_production_cache_levers(
        "gptq,scale_sweep"
    )["gptq_damp_sweep"] is True
