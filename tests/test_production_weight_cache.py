from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from prismaquant.kl_sensitivity_probe import _normalized_production_cache_levers
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.production_recache import (
    _load_assignment,
    activation_max_abs_delta_summary,
    assignment_digest,
    preload_production_cache_for_assignment,
    production_cache_keys_for_assignment,
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

    assert cache.compact_for_pickle() >= 1
    assert all(not isinstance(value, torch.Tensor) for value in cache.weights.values())
    assert cache._lru_bytes == 0


def test_production_cache_resolves_format_aliases_and_sizes(tmp_path):
    tensor = torch.ones((2, 2), dtype=torch.float32)
    path = tmp_path / "layer.pt"
    torch.save(tensor, path)
    cache = ProductionWeightCache(
        weights={("layer", "MXFP8_E4M3"): path.name},
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )

    assert cache.resolve_key("layer", "MXFP8") == ("layer", "MXFP8_E4M3")
    assert ("layer", "MXFP8") in cache
    assert cache.estimate_nbytes([("layer", "MXFP8_E4M3")]) == path.stat().st_size
    assert torch.equal(cache.get("layer", "MXFP8"), tensor)


def test_recache_preload_respects_resident_budget(tmp_path):
    for name in ("a", "b"):
        torch.save(torch.ones((2, 2)), tmp_path / f"{name}.pt")
    cache = ProductionWeightCache(
        weights={
            ("a", "NVFP4"): "a.pt",
            ("b", "MXFP8_E4M3"): "b.pt",
        },
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )
    assignment = {"a": "NVFP4", "b": "MXFP8", "c": "BF16"}
    keys, missing = production_cache_keys_for_assignment(cache, assignment)
    method_keys, method_missing = cache.assignment_keys(assignment)

    assert keys == [("a", "NVFP4"), ("b", "MXFP8_E4M3")]
    assert missing == []
    assert method_keys == keys
    assert method_missing == []

    stats = cache.prefetch_assignment(
        assignment,
        max_resident_bytes=1,
        require=False,
        progress=False,
    )
    assert stats["skipped"] is True
    assert stats["loaded"] == 0

    stats = preload_production_cache_for_assignment(
        cache,
        assignment,
        max_resident_bytes=10_000_000,
        require=True,
        progress=False,
    )
    assert stats["loaded"] == 2
    assert all(isinstance(cache.weights[k], torch.Tensor) for k in keys)


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
    delta = cache.metadata["activation_recache"]["activation_max_abs_delta"]
    assert delta["n_common"] == 2
    assert delta["changed_gt_5pct"] == 1
    assert delta["ratio_p50"] == pytest.approx(2.0)


def test_activation_max_abs_delta_summary_reports_ratio_quantiles():
    summary = activation_max_abs_delta_summary(
        {"a": 1.0, "b": 2.0, "c": 4.0, "missing": 3.0},
        {"a": 1.0, "b": 3.0, "c": 2.0},
    )

    assert summary["n_common"] == 3
    assert summary["n_before"] == 4
    assert summary["n_after"] == 3
    assert summary["ratio_min"] == pytest.approx(0.5)
    assert summary["ratio_p50"] == pytest.approx(1.0)
    assert summary["ratio_max"] == pytest.approx(1.5)
    assert summary["changed_gt_1pct"] == 2
    assert summary["changed_gt_5pct"] == 2


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
