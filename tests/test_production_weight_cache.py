from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from prismaquant.kl_sensitivity_probe import _normalized_production_cache_levers
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.production_recache import (
    _load_assignment,
    assignment_digest,
    recache_production_weight_cache,
)


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


class _TinyChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 32)
        self.l1 = nn.Linear(32, 32, bias=False)
        self.l2 = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[0, 0] = 1.0
            self.embed.weight[1, 1] = 2.0
            self.l1.weight.copy_(torch.eye(32))
            self.l2.weight.fill_(1.0)

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        x = self.l1(x)
        return self.l2(x)


def test_production_recache_measures_quantized_upstream_activation_range():
    model = _TinyChain()
    cache = ProductionWeightCache(
        weights={("l1", "NVFP4"): 3.0 * torch.eye(32)},
        levers={"gptq": True, "scale_sweep": True},
        activation_max_abs={"l1": 2.0, "l2": 2.0},
    )
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)

    max_abs = recache_production_weight_cache(
        model,
        calib_ids,
        {"l1": "NVFP4", "l2": "BF16"},
        cache,
        include_activation_quant=False,
        progress=False,
    )

    assert max_abs["l1"] == pytest.approx(2.0)
    assert max_abs["l2"] == pytest.approx(6.0)
    assert cache.activation_max_abs["l2"] == pytest.approx(6.0)
    assert cache.metadata["activation_recache"]["status"] == "applied"
    assert cache.metadata["activation_recache"]["assignment_entries"] == 2
    assert cache.metadata["activation_recache"]["assignment_sha256"] == assignment_digest(
        {"l2": "BF16", "l1": "NVFP4"}
    )


def test_fill_production_cache_recache_requires_concrete_assignment():
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="recache_assignment"):
        fill_production_weight_cache(
            model,
            calib_ids,
            qnames=[],
            progress=False,
            recache_pass=True,
        )


def test_recache_assignment_loader_is_exporter_independent(monkeypatch, tmp_path):
    path = tmp_path / "layer_config.json"
    path.write_text(
        '{"layer.weight": {"bits": 4, "group_size": 16, "data_type": "nv_fp", '
        '"act_bits": 4, "act_group_size": 16, "act_data_type": "nv_fp"}}'
    )

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name in {"accelerate", "prismaquant.export_native_compressed"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    assert _load_assignment(path) == {"layer": "NVFP4"}
