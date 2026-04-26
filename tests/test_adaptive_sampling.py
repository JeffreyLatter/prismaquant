"""Tests for prismaquant.adaptive_sampling."""
from __future__ import annotations

import pytest

from prismaquant.adaptive_sampling import (
    AdaptiveExpertScheduler,
    aggregate_per_domain_saliency,
    aggregate_global_saliency,
    infer_chunk_domain,
    saliency_with_policy,
    _GLOBAL_DOMAIN,
)


# ---------- domain inference ----------

def test_infer_chunk_domain_basic():
    assert infer_chunk_domain("chunk_agentic_03.jsonl") == "agentic"
    assert infer_chunk_domain("/path/to/chunk_math_07.jsonl") == "math"
    assert infer_chunk_domain("chunk_code-py_12.jsonl") == "code-py"


def test_infer_chunk_domain_global():
    assert infer_chunk_domain("chunk_00.jsonl") == _GLOBAL_DOMAIN
    assert infer_chunk_domain("chunk_15.jsonl") == _GLOBAL_DOMAIN
    assert infer_chunk_domain("foo.jsonl") == _GLOBAL_DOMAIN
    assert infer_chunk_domain("/tmp/anything.jsonl") == _GLOBAL_DOMAIN


# ---------- helper for fake chunk pickles ----------

def _make_chunk(saliency, nsamples=16, seqlen=4096):
    return {
        "expert_saliency": saliency,
        "meta": {"nsamples": nsamples, "seqlen": seqlen},
    }


# ---------- update / accumulate ----------

def test_update_recovers_token_weighted_average():
    """A 16-sample chunk and a 32-sample chunk should weight-average
    correctly: the bigger chunk's value dominates the merged saliency."""
    sched = AdaptiveExpertScheduler()
    # Same expert, different per-chunk values, same domain.
    sched.update_from_chunk_pickle(
        _make_chunk({"R0": {0: 1.0}}, nsamples=16), "agentic")
    sched.update_from_chunk_pickle(
        _make_chunk({"R0": {0: 4.0}}, nsamples=32), "agentic")
    h = sched.history["R0"][0]
    expected = (1.0 * 16 * 4096 + 4.0 * 32 * 4096) / (16 * 4096 + 32 * 4096)
    # = (1.0 + 8.0) / 3 = 3.0
    assert h.saliency("agentic") == pytest.approx(expected)
    assert h.saliency("agentic") == pytest.approx(3.0)


def test_per_domain_separation():
    sched = AdaptiveExpertScheduler()
    sched.update_from_chunk_pickle(
        _make_chunk({"R0": {0: 10.0}}), "agentic")
    sched.update_from_chunk_pickle(
        _make_chunk({"R0": {0: 1.0}}), "math")
    h = sched.history["R0"][0]
    assert h.saliency("agentic") == pytest.approx(10.0)
    assert h.saliency("math") == pytest.approx(1.0)
    # global = token-weighted across both
    assert h.saliency_global() == pytest.approx(5.5)


# ---------- expert_status / freeze logic ----------

def test_status_contested_until_min_chunks():
    sched = AdaptiveExpertScheduler(min_chunks_for_freeze=2)
    sched.update_from_chunk_pickle(_make_chunk({"R": {0: 1.0}}), "_global")
    # Single chunk → contested
    assert sched.expert_status("R", 0) == "contested"


def test_status_frozen_when_top_of_router_and_stable():
    """Expert 9 dominates a 10-expert router and is stable across chunks
    → frozen-keep."""
    sched = AdaptiveExpertScheduler(
        min_chunks_for_freeze=2, stability_threshold=0.20,
        keep_band=0.25, drop_band=0.10,
    )
    for _ in range(3):
        sal = {f"R": {i: float(i) for i in range(10)}}
        sched.update_from_chunk_pickle(_make_chunk(sal), "_global")
    # Top expert (id 9) — top of router, very stable → frozen-keep
    assert sched.expert_status("R", 9) == "frozen-keep"


def test_status_frozen_drop_at_bottom():
    sched = AdaptiveExpertScheduler(
        min_chunks_for_freeze=2, stability_threshold=0.20,
        keep_band=0.25, drop_band=0.10,
    )
    for _ in range(3):
        sal = {f"R": {i: float(i) + 1.0 for i in range(10)}}
        sched.update_from_chunk_pickle(_make_chunk(sal), "_global")
    # Bottom expert is in the bottom 10% → frozen-drop.
    assert sched.expert_status("R", 0) == "frozen-drop"


def test_status_unstable_stays_contested():
    sched = AdaptiveExpertScheduler(
        min_chunks_for_freeze=2, stability_threshold=0.05,
    )
    # Big swings → not stable
    for v in [1.0, 5.0, 1.0, 5.0]:
        sched.update_from_chunk_pickle(_make_chunk({"R": {0: v}}), "_global")
    assert sched.expert_status("R", 0) == "contested"


def test_per_domain_disagreement_keeps_contested():
    """If an expert ranks in the keep band of one domain and the drop
    band of another, it must stay contested — domain-specific load-
    bearing is exactly what we need more samples to resolve."""
    sched = AdaptiveExpertScheduler(
        min_chunks_for_freeze=2, stability_threshold=0.20,
        keep_band=0.25, drop_band=0.25,
    )
    # 4 experts. Expert 0 dominates "agentic" (rank 1.0), is bottom in
    # "math" (rank 0.0). Expert 3 the mirror image.
    for _ in range(3):
        sched.update_from_chunk_pickle(
            _make_chunk({"R": {0: 10.0, 1: 1.0, 2: 1.0, 3: 1.0}}),
            "agentic")
        sched.update_from_chunk_pickle(
            _make_chunk({"R": {0: 1.0, 1: 1.0, 2: 1.0, 3: 10.0}}),
            "math")
    # Both 0 and 3 span the keep band in one domain and the drop band in
    # another → contested.
    assert sched.expert_status("R", 0) == "contested"
    assert sched.expert_status("R", 3) == "contested"


# ---------- linear_include narrowing ----------

def test_linear_include_unchanged_when_nothing_frozen():
    sched = AdaptiveExpertScheduler()
    out = sched.linear_include_for_next_chunk(
        base_include=r"^model\.layers\.0\..*",
        expert_info={},
    )
    assert out == r"^model\.layers\.0\..*"


def test_linear_include_excludes_frozen_experts():
    sched = AdaptiveExpertScheduler(
        min_chunks_for_freeze=2, stability_threshold=0.20,
        keep_band=0.25, drop_band=0.25,
        # Disable the contested-band gate around prune_ratio so this
        # test is solely exercising the regex-narrowing logic.
        contested_band=0.0,
    )
    # 4 experts; 0 is top, others equal-bottom.
    sal = {"R": {0: 10.0, 1: 1.0, 2: 1.0, 3: 1.0}}
    for _ in range(3):
        sched.update_from_chunk_pickle(_make_chunk(sal), "_global")
    expert_info = {
        "model.layers.0.experts.0.w1": ("R", "0"),
        "model.layers.0.experts.1.w1": ("R", "1"),
        "model.layers.0.experts.2.w1": ("R", "2"),
        "model.layers.0.experts.3.w1": ("R", "3"),
    }
    pat = sched.linear_include_for_next_chunk(
        base_include=r"^model\.layers\.0\..*",
        expert_info=expert_info,
    )
    import re
    cre = re.compile(pat)
    # Expert 0 is frozen-keep (top of router, stable)
    # Experts 1-3 are tied at the bottom of router → frozen-drop
    # ALL experts have frozen → so all four expert linears are excluded.
    matched = [
        n for n in expert_info if cre.search(n)
    ]
    assert matched == []


def test_state_roundtrips_via_json():
    sched = AdaptiveExpertScheduler(prune_ratio=0.4)
    for _ in range(2):
        sched.update_from_chunk_pickle(
            _make_chunk({"R": {0: 1.0, 1: 2.0}}), "agentic")
    blob = sched.to_json()
    restored = AdaptiveExpertScheduler.from_json(blob)
    assert restored.prune_ratio == pytest.approx(0.4)
    assert restored.history["R"][0].saliency("agentic") == pytest.approx(1.0)
    assert restored.history["R"][1].saliency("agentic") == pytest.approx(2.0)
    assert restored.chunks_processed_by_domain == {"agentic": 2}


# ---------- aggregate helpers ----------

def test_aggregate_per_domain_token_weighted():
    chunks = [
        (_make_chunk({"R": {0: 1.0}}, nsamples=16), "agentic"),
        (_make_chunk({"R": {0: 5.0}}, nsamples=48), "agentic"),
        (_make_chunk({"R": {0: 2.0}}, nsamples=16), "math"),
    ]
    out = aggregate_per_domain_saliency(chunks)
    # agentic: (1*16 + 5*48) / (16+48) = (16+240)/64 = 4.0
    assert out["agentic"]["R"][0] == pytest.approx(4.0)
    # math: just the one chunk
    assert out["math"]["R"][0] == pytest.approx(2.0)


def test_aggregate_global_combines_all():
    chunks = [
        (_make_chunk({"R": {0: 1.0}}, nsamples=16), "agentic"),
        (_make_chunk({"R": {0: 3.0}}, nsamples=16), "math"),
    ]
    pd = aggregate_per_domain_saliency(chunks)
    g = aggregate_global_saliency(pd, chunks)
    # global avg over (1,3) at equal weight = 2.0
    assert g["R"][0] == pytest.approx(2.0)


# ---------- saliency_with_policy ----------

def _per_domain_fixture():
    # Expert 0: load-bearing in agentic (10), trivial in math (1).
    # Expert 1: trivial in both.
    return {
        "agentic": {"R": {0: 10.0, 1: 1.0}},
        "math":    {"R": {0:  1.0, 1: 1.0}},
    }


def test_saliency_policy_global_returns_legacy_unchanged():
    pd = _per_domain_fixture()
    legacy = {"R": {0: 5.5, 1: 1.0}}
    out = saliency_with_policy(pd, legacy, "global")
    assert out is legacy


def test_saliency_policy_union_takes_max_across_domains():
    pd = _per_domain_fixture()
    out = saliency_with_policy(pd, {}, "union")
    # Max over (10, 1) = 10 → expert 0 is well-protected from pruning
    assert out["R"][0] == pytest.approx(10.0)
    assert out["R"][1] == pytest.approx(1.0)


def test_saliency_policy_intersection_takes_min_across_domains():
    pd = _per_domain_fixture()
    out = saliency_with_policy(pd, {}, "intersection")
    # Min over (10, 1) = 1 → expert 0 is now droppable (looks low)
    assert out["R"][0] == pytest.approx(1.0)
    assert out["R"][1] == pytest.approx(1.0)


def test_saliency_policy_mean_equal_weights_per_domain():
    pd = _per_domain_fixture()
    out = saliency_with_policy(pd, {}, "mean")
    # Mean over (10, 1) = 5.5
    assert out["R"][0] == pytest.approx(5.5)
    assert out["R"][1] == pytest.approx(1.0)


def test_saliency_policy_unknown_raises():
    with pytest.raises(ValueError):
        saliency_with_policy(_per_domain_fixture(), {}, "weird")


def test_saliency_policy_falls_back_to_legacy_when_no_per_domain():
    legacy = {"R": {0: 7.0}}
    out = saliency_with_policy({}, legacy, "union")
    assert out is legacy
