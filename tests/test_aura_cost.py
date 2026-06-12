import pytest
import torch
import torch.nn as nn
from collections import Counter

from prismaquant.aura_cost import _guard_packed_expert_coverage
from prismaquant.aura_cost import _auto_n_chunks
from prismaquant.aura_cost import _delta_w
from prismaquant.aura_cost import _target_linears


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


def test_auto_chunk_defaults_to_accurate_fp32_sizing(monkeypatch):
    class _FakeWeight:
        def numel(self):
            return (1024 ** 3) // 8

        def element_size(self):
            return 4

    class _FakeLinear:
        weight = _FakeWeight()

    linears = {str(i): _FakeLinear() for i in range(8)}
    names = list(linears)
    monkeypatch.setattr("prismaquant.aura_cost._free_gib", lambda: 33.0)

    default_chunks = _auto_n_chunks(
        linears,
        names,
        min_free_gib=20.0,
        n_nonzero_fmts=2,
    )
    accurate_chunks = _auto_n_chunks(
        linears,
        names,
        min_free_gib=20.0,
        n_nonzero_fmts=2,
        accurate_chunk_bytes=True,
    )
    legacy_chunks = _auto_n_chunks(
        linears,
        names,
        min_free_gib=20.0,
        n_nonzero_fmts=2,
        accurate_chunk_bytes=False,
    )

    assert default_chunks == accurate_chunks
    assert accurate_chunks > legacy_chunks


def test_delta_w_records_cache_and_rtn_sources():
    class _Cache:
        def get(self, name, fmt):
            if name == "cached":
                return torch.ones(16, 16)
            return None

    counts = Counter()
    weight = torch.zeros(16, 16)

    cached = _delta_w("cached", "NVFP4", weight, _Cache(), source_counts=counts)
    rtn = _delta_w("missing", "NVFP4", weight, _Cache(), source_counts=counts)

    assert torch.equal(cached, torch.ones(16, 16))
    assert rtn is not None
    assert counts["production_cache"] == 1
    assert counts["production_cache_miss"] == 1
    assert counts["rtn"] == 1


def test_delta_w_strict_cache_miss_raises_and_counts():
    class _Cache:
        def get(self, name, fmt):
            return None

    counts = Counter()

    with pytest.raises(RuntimeError, match="require_production_cache"):
        _delta_w(
            "missing",
            "NVFP4",
            torch.zeros(2, 2),
            _Cache(),
            strict=True,
            source_counts=counts,
        )

    assert counts["production_cache_miss"] == 1


class _TinyHeadModel(nn.Module):
    def __init__(self, *, tied: bool):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.body = nn.Linear(16, 16, bias=False)
        self.lm_head = nn.Linear(16, 32, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


def test_include_lm_head_rejects_tied_embeddings():
    with pytest.raises(RuntimeError, match="tied input/output embeddings"):
        _target_linears(_TinyHeadModel(tied=True), include_lm_head=True)


def test_include_lm_head_allows_untied_embeddings():
    linears = _target_linears(_TinyHeadModel(tied=False), include_lm_head=True)

    assert "lm_head" in linears
