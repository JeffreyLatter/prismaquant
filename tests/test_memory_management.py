import importlib
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import iterate_perturbed_allocation as ipa
from prismaquant.perturbed_x_cache import PerturbedActivationCache
from prismaquant.propagated_cost import CUDAGraphRegistry, _CUDAGraphEntry
from prismaquant.memory_management import enforce_gpu_memory_budget


class _ManyLinear(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(64, 64, bias=False) for _ in range(count)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_frozen_weight_cache_lru_eviction(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES", "3")
    monkeypatch.setenv("PRISMAQUANT_MAX_GPU_MEM_GB", "0")
    model = _ManyLinear(8).eval()
    assignment = {f"layers.{idx}": "BF16" for idx in range(8)}
    builder = PerturbedActivationCache(
        model,
        assignment,
        tmp_path,
        input_rows=0,
        cal_hash="fixed",
    )

    with builder.frozen_weight_cache():
        pass

    assert list(builder._frozen_weight_format_cache.keys()) == [
        (f"layers.{idx}", "BF16")
        for idx in range(5, 8)
    ]
    assert builder._frozen_weight_cache_evictions == 5


def test_cuda_graph_cache_lru_eviction(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH", "3")
    registry = CUDAGraphRegistry(label="test-graph-lru", max_entries=10)
    for idx in range(8):
        registry.entries[(idx,)] = _CUDAGraphEntry(
            graph=SimpleNamespace(replay=lambda: None),
            static_args=(),
            static_kwargs={},
            static_output=None,
        )

    registry._evict_if_needed()

    assert list(registry.entries.keys()) == [(5,), (6,), (7,)]
    assert registry.eviction_count == 5


def test_cuda_graph_capture_failure_cleanup_resets_graph(monkeypatch):
    calls = []
    registry = CUDAGraphRegistry(label="test-graph-cleanup", max_entries=10)
    graph = SimpleNamespace(reset=lambda: calls.append("reset"))
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: calls.append(("sync", str(device))),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    registry._cleanup_failed_capture(
        graph,
        torch.device("cuda:0"),
        "test-capture",
    )

    assert calls == ["reset", ("sync", "cuda:0"), "empty"]


def test_phase_boundary_cleanup_called(monkeypatch):
    calls = {"empty_cache": 0}

    def _empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", _empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ipa._phase_boundary_cleanup("l2_to_l3")

    assert calls["empty_cache"] == 1


def test_memory_budget_evicts_when_low(monkeypatch):
    gib = 1024 ** 3
    mem_info = iter([(0, 2 * gib), (1536 * 1024 ** 2, 2 * gib)])

    class _Evictor:
        def __init__(self):
            self.entries = OrderedDict((idx, idx) for idx in range(3))
            self.evicted = []

        def evict_oldest_for_memory_budget(self):
            if not self.entries:
                return False
            key, _value = self.entries.popitem(last=False)
            self.evicted.append(key)
            return True

    evictor = _Evictor()
    monkeypatch.setenv("PRISMAQUANT_MAX_GPU_MEM_GB", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: next(mem_info))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    evicted = enforce_gpu_memory_budget([evictor], reason="test")

    assert evicted == 1
    assert evictor.evicted == [0]


def test_triton_warmup_compiles_kernel(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_FUSED_JIT_WARMUP", raising=False)
    module = importlib.import_module("prismaquant.kernels.nvfp4_fused")
    state = module.nvfp4_fused_warmup_state()

    assert state["attempted"] is True
    if not torch.cuda.is_available():
        assert state["skipped_reason"] == "cuda_unavailable"
        pytest.skip("CUDA unavailable")
    assert state["compiled"] is True
    assert (8, 8, 64, 16, 32, 64) in state["compiled_signatures"]
