from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import pytest

from prismaquant.kl_sensitivity_probe import _normalized_production_cache_levers
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.production_weight_cache import ProductionWeightCacheVariantView
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.production_weight_cache import _solve_nvfp4_activation_clip_groups
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


def test_production_cache_resolves_prismaclip_variant_as_distinct_key(tmp_path):
    base = torch.zeros((2, 2), dtype=torch.float32)
    clipped = torch.ones((2, 2), dtype=torch.float32)
    torch.save(base, tmp_path / "base.pt")
    torch.save(clipped, tmp_path / "clip.pt")
    cache = ProductionWeightCache(
        weights={
            ("layer", "NVFP4"): "base.pt",
            ("layer", "NVFP4_CLIPPED"): "clip.pt",
        },
        levers={},
        cache_dir=str(tmp_path),
    )

    assert cache.resolve_key("layer", "NVFP4") == ("layer", "NVFP4")
    assert cache.resolve_key("layer", "NVFP4_CLIPPED") == (
        "layer",
        "NVFP4_CLIPPED",
    )
    assert torch.equal(cache.get("layer", "NVFP4"), base)
    assert torch.equal(cache.get("layer", "NVFP4_CLIPPED"), clipped)


def test_production_cache_variant_view_maps_runtime_nvfp4_to_prismaclip(tmp_path):
    base = torch.zeros((2, 2), dtype=torch.float32)
    clipped = torch.ones((2, 2), dtype=torch.float32)
    torch.save(base, tmp_path / "base.pt")
    torch.save(clipped, tmp_path / "clip.pt")
    cache = ProductionWeightCache(
        weights={
            ("layer", "NVFP4"): "base.pt",
            ("layer", "NVFP4_CLIPPED"): "clip.pt",
        },
        levers={},
        cache_dir=str(tmp_path),
    )
    view = ProductionWeightCacheVariantView(
        cache,
        {"layer": "NVFP4_CLIPPED"},
    )

    assert view.resolve_key("layer", "NVFP4") == ("layer", "NVFP4_CLIPPED")
    assert torch.equal(view.get("layer", "NVFP4"), clipped)
    keys, missing = view.assignment_keys({"layer": "NVFP4"})
    assert keys == [("layer", "NVFP4_CLIPPED")]
    assert missing == []


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


def test_production_cache_file_page_prefetch_does_not_load_tensors(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    for name in ("a", "b"):
        torch.save(torch.ones((2, 2)), tmp_path / f"{name}.pt")
    cache = ProductionWeightCache(
        weights={
            ("a", "NVFP4"): "a.pt",
            ("b", "MXFP8_E4M3"): "b.pt",
        },
        levers={},
        cache_dir=str(tmp_path),
    )
    seen_paths = []

    def fake_prefetch(paths, **kwargs):
        seen_paths.extend(Path(path).name for path in paths)
        return {
            "mode": kwargs["mode"],
            "files": len(paths),
            "bytes": 123,
            "prefetched_bytes": 123,
            "skipped": False,
        }

    monkeypatch.setattr(pwc, "prefetch_files_to_page_cache", fake_prefetch)

    stats = cache.prefetch_assignment_file_pages(
        {"a": "NVFP4", "b": "MXFP8", "c": "BF16"},
        mode="require",
        progress=False,
    )

    assert sorted(seen_paths) == ["a.pt", "b.pt"]
    assert stats["keys"] == 2
    assert stats["missing"] == 0
    assert all(not isinstance(value, torch.Tensor) for value in cache.weights.values())


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


def test_production_cache_records_nvfp4_scale_rule(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    monkeypatch.setenv("PRISMAQUANT_NVFP4_SCALE_RULE", "four_over_six_mse")
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=[],
        levers={"gptq": False, "scale_sweep": False},
        progress=False,
    )

    assert cache.levers["nvfp4_scale_rule"] == "four_over_six_mse"


def test_production_cache_act_clip_solver_selects_rendered_nvfp4(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls: list[float | None] = []

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        act_clip_threshold=None,
        **_kwargs,
    ):
        calls.append(act_clip_threshold)
        if fmt.upper() != "NVFP4":
            return weight.detach().clone()
        delta = (
            3.0
            if act_clip_threshold is None
            else abs(float(act_clip_threshold) - 1.0)
        )
        return weight.detach().to(torch.float32) + delta

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False, "act_clip_solver": True},
        max_act_rows=8,
        progress=False,
    )

    meta = cache.metadata["activation_clip_solver"]
    selected = meta["selected_by_qname"]["l1"]
    group_meta = next(iter(meta["groups"].values()))
    assert meta["status"] == "applied"
    assert selected is not None
    assert group_meta["selected"] == "solved"
    assert group_meta["selected_eval_index"] is not None
    assert 1 <= group_meta["selected_eval_index"] <= group_meta["n_evals"]
    assert len(group_meta["eval_trace"]) == group_meta["n_evals"]
    assert group_meta["eval_trace"]
    assert all(
        later["best_score"] <= earlier["best_score"]
        for earlier, later in zip(
            group_meta["eval_trace"],
            group_meta["eval_trace"][1:],
        )
    )
    assert any(value is None for value in calls)
    assert any(value is not None for value in calls)
    assert torch.allclose(
        cache.get("l1", "NVFP4").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + abs(float(selected) - 1.0),
        atol=1e-2,
    )


def test_production_cache_prismaclip_candidate_preserves_baseline_nvfp4(
    monkeypatch,
):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        act_clip_threshold=None,
        **_kwargs,
    ):
        assert fmt.upper() == "NVFP4"
        delta = (
            3.0
            if act_clip_threshold is None
            else abs(float(act_clip_threshold) - 1.0)
        )
        return weight.detach().to(torch.float32) + delta

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.delenv("PRISMAQUANT_ACT_CLIP_SOLVER", raising=False)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4", "NVFP4_CLIPPED"],
        levers={"gptq": False, "scale_sweep": False},
        max_act_rows=8,
        progress=False,
    )

    meta = cache.metadata["activation_clip_solver"]
    selected = meta["selected_by_qname"]["l1"]
    assert cache.levers["act_clip_candidates"] is True
    assert meta["status"] == "applied"
    assert meta["candidate_format"] == "NVFP4_CLIPPED"
    assert selected is not None
    assert torch.allclose(
        cache.get("l1", "NVFP4").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + 3.0,
        atol=1e-6,
    )
    assert torch.allclose(
        cache.get("l1", "NVFP4_CLIPPED").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + abs(float(selected) - 1.0),
        atol=1e-2,
    )


def test_production_cache_prismaclip_rbc_is_disabled(monkeypatch):
    import pytest
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        act_clip_threshold=None,
        **_kwargs,
    ):
        if act_clip_threshold is None:
            delta = 3.0
        else:
            delta = 2.0
        return weight.detach().to(torch.float32) + delta

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_RESCALING", "none,row_rms")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")

    with pytest.raises(RuntimeError, match="PrismaClip-RBC.*disabled"):
        fill_production_weight_cache(
            model,
            calib_ids,
            qnames=["l1"],
            formats=["NVFP4"],
            levers={"gptq": False, "scale_sweep": False, "act_clip_solver": True},
            max_act_rows=8,
            progress=False,
        )


def test_production_cache_act_clip_solver_skips_tiny_configured_gain(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls: list[float | None] = []

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        act_clip_threshold=None,
        **_kwargs,
    ):
        calls.append(act_clip_threshold)
        if fmt.upper() != "NVFP4":
            return weight.detach().clone()
        delta = 1.0 if act_clip_threshold is None else 0.9995
        return weight.detach().to(torch.float32) + delta

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN", "0.002")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE", "1")

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False, "act_clip_solver": True},
        max_act_rows=8,
        progress=False,
    )

    meta = cache.metadata["activation_clip_solver"]
    group_meta = next(iter(meta["groups"].values()))
    assert meta["min_gain"] == pytest.approx(0.002)
    assert meta["prewrite_baseline"] is True
    assert meta["prewritten_baseline_qnames"] == ["l1"]
    assert meta["selected_by_qname"]["l1"] is None
    assert group_meta["selected"] == "baseline"
    assert group_meta["relative_gain"] == 0.0
    assert calls.count(None) == 1
    assert torch.allclose(
        cache.get("l1", "NVFP4").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + 1.0,
        atol=1e-6,
    )


def test_production_cache_act_clip_solver_prewrite_default_off(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)

    def fake_render_production_weight(weight, fmt, **_kwargs):
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")
    monkeypatch.delenv("PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE", raising=False)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False, "act_clip_solver": True},
        max_act_rows=8,
        progress=False,
    )

    meta = cache.metadata["activation_clip_solver"]
    assert meta["prewrite_baseline"] is False
    assert meta["prewritten_baseline_qnames"] == []


def test_prismafisherclip_requires_h_detail_dir():
    model = nn.Linear(1, 1, bias=False)
    with pytest.raises(ValueError, match="PrismaFisherClip requires h_detail_dir"):
        fill_production_weight_cache(
            model,
            torch.empty((0, 1), dtype=torch.long),
            qnames=[],
            formats=["NVFP4"],
            levers={"gptq": False, "scale_sweep": False, "fisher_clip": True},
            progress=False,
        )


def test_production_cache_act_clip_solver_holdout_rejects_overfit(monkeypatch):
    import prismaquant.production_weight_cache as pwc

    modules = {"a": nn.Linear(2, 2, bias=False)}
    activations = {
        "a": torch.tensor([
            [100.0, 0.0],  # default holdout row: index 0 mod 4
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
    }
    weights: dict[tuple[str, str], object] = {}

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        act_clip_threshold=None,
        **_kwargs,
    ):
        delta = 1.0 if act_clip_threshold is None else 0.5
        return weight.detach().to(torch.float32) + delta

    def fake_activation_output_error(weight, rendered, activations, **_kwargs):
        delta = float((rendered - weight.to(rendered.dtype)).abs().mean())
        is_holdout = float(activations.reshape(-1, activations.shape[-1])[:, 0].mean()) > 50.0
        if delta < 0.75:
            return 10.0 if is_holdout else 0.5
        return 1.0

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setattr(pwc, "_activation_output_error", fake_activation_output_error)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")

    thresholds, meta = _solve_nvfp4_activation_clip_groups(
        groups={"a": ["a"]},
        qname_to_module=modules,
        activations=activations,
        levers={"gptq": False, "scale_sweep": False},
        joint_globals={},
        activation_max_abs={"a": 100.0},
        fisher_rows=None,
        weights_out=weights,
        cache_dir_path=None,
        progress=False,
    )

    group_meta = meta["groups"]["a"]
    assert meta["holdout_enabled"] is True
    assert thresholds["a"].threshold is None
    assert meta["selected_by_qname"]["a"] is None
    assert group_meta["selected"] == "baseline"
    assert group_meta["holdout_accepted"] is False
    assert group_meta["eval_trace"][0]["holdout_score"] == pytest.approx(10.0)
    assert weights == {}


def test_activation_output_error_accepts_fisher_row_weights():
    import prismaquant.production_weight_cache as pwc

    weight = torch.zeros((1, 2), dtype=torch.float32)
    activations = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    baseline = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    candidate = torch.tensor([[0.1, 0.8]], dtype=torch.float32)

    assert pwc._activation_output_error(weight, candidate, activations) < (
        pwc._activation_output_error(weight, baseline, activations)
    )
    assert pwc._activation_output_error(
        weight,
        candidate,
        activations,
        row_weights=torch.tensor([0.1, 10.0]),
    ) > pwc._activation_output_error(
        weight,
        baseline,
        activations,
        row_weights=torch.tensor([0.1, 10.0]),
    )


def test_prismafisherclip_vetoes_unweighted_clip_choice(monkeypatch, tmp_path):
    import prismaquant.production_weight_cache as pwc

    modules = {"a": nn.Linear(2, 1, bias=False)}
    with torch.no_grad():
        modules["a"].weight.zero_()
    activations = {
        "a": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
    }
    weights: dict[tuple[str, str], object] = {}
    torch.save(
        {"g2_per_token": torch.tensor([0.1, 10.0], dtype=torch.float32)},
        tmp_path / "a.pt",
    )
    seen_render_weights: list[torch.Tensor | None] = []

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        act_clip_threshold=None,
        fisher_row_weights=None,
        **_kwargs,
    ):
        seen_render_weights.append(fisher_row_weights)
        if act_clip_threshold is None:
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        return torch.tensor([[0.1, 0.8]], dtype=torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")
    monkeypatch.setenv("PRISMAQUANT_PRISMAFISHERCLIP_MODE", "veto")

    thresholds, meta = pwc._solve_nvfp4_activation_clip_groups(
        groups={"a": ["a"]},
        qname_to_module=modules,
        activations=activations,
        levers={"gptq": False, "scale_sweep": False, "fisher_clip": True},
        joint_globals={},
        activation_max_abs={"a": 1.0},
        fisher_rows=pwc._FisherRowWeightCache(tmp_path),
        weights_out=weights,
        cache_dir_path=None,
        progress=False,
    )

    group_meta = meta["groups"]["a"]
    assert meta["method"] == "PrismaFisherClip"
    assert meta["fisher_clip_available"] is True
    assert meta["fisher_clip_render_weighted"] is False
    assert thresholds["a"].threshold is None
    assert group_meta["selected"] == "baseline"
    assert group_meta["best_score"] == pytest.approx(group_meta["baseline_score"])
    assert seen_render_weights and all(value is None for value in seen_render_weights)
    assert weights == {}


def test_prismafisherclip_default_audits_unweighted_clip_choice(
    monkeypatch, tmp_path
):
    import prismaquant.production_weight_cache as pwc

    modules = {"a": nn.Linear(2, 1, bias=False)}
    with torch.no_grad():
        modules["a"].weight.zero_()
    activations = {
        "a": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
    }
    weights: dict[tuple[str, str], object] = {}
    torch.save(
        {"g2_per_token": torch.tensor([0.1, 10.0], dtype=torch.float32)},
        tmp_path / "a.pt",
    )

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        act_clip_threshold=None,
        fisher_row_weights=None,
        **_kwargs,
    ):
        assert fisher_row_weights is None
        if act_clip_threshold is None:
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        return torch.tensor([[0.1, 0.8]], dtype=torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")

    thresholds, meta = pwc._solve_nvfp4_activation_clip_groups(
        groups={"a": ["a"]},
        qname_to_module=modules,
        activations=activations,
        levers={"gptq": False, "scale_sweep": False, "fisher_clip": True},
        joint_globals={},
        activation_max_abs={"a": 1.0},
        fisher_rows=pwc._FisherRowWeightCache(tmp_path),
        weights_out=weights,
        cache_dir_path=None,
        progress=False,
    )

    group_meta = meta["groups"]["a"]
    assert meta["method"] == "PrismaFisherClip"
    assert meta["fisher_clip_mode"] == "audit"
    assert meta["fisher_clip_available"] is True
    assert thresholds["a"].threshold is not None
    assert group_meta["selected"] == "solved"
    assert group_meta["fisher_accepted"] is None
    assert group_meta["best_fisher_score"] is not None


def test_production_cache_act_clip_solver_batches_same_shape_groups(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_BATCHED", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")

    modules = {
        "a": nn.Linear(4, 4, bias=False),
        "b": nn.Linear(4, 4, bias=False),
    }
    activations = {
        "a": torch.randn(8, 4),
        "b": torch.randn(8, 4),
    }
    weights: dict[tuple[str, str], object] = {}

    thresholds, meta = _solve_nvfp4_activation_clip_groups(
        groups={"fused": ["a", "b"]},
        qname_to_module=modules,
        activations=activations,
        levers={"gptq": False, "scale_sweep": False},
        joint_globals={},
        activation_max_abs={
            name: float(act.abs().max().item())
            for name, act in activations.items()
        },
        fisher_rows=None,
        weights_out=weights,
        cache_dir_path=None,
        progress=False,
    )

    assert thresholds["a"].threshold is None
    assert thresholds["b"].threshold is None
    assert meta["batched_same_shape"] is True
    assert meta["batched_evaluations"] >= 1
    assert meta["scalar_evaluations"] == 0
    assert sorted(meta["prewritten_baseline_qnames"]) == ["a", "b"]
    assert set(weights) == {("a", "NVFP4"), ("b", "NVFP4")}


def test_production_cache_batched_clip_baseline_matches_scalar(monkeypatch):
    torch.manual_seed(0)
    modules = {
        "a": nn.Linear(16, 8, bias=False),
        "b": nn.Linear(16, 8, bias=False),
    }
    activations = {
        "a": torch.randn(24, 16),
        "b": torch.randn(24, 16),
    }

    def solve(batched: bool) -> dict[tuple[str, str], object]:
        monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_BATCHED", "1" if batched else "0")
        monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE", "1")
        monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS", "4")
        monkeypatch.setenv("PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN", "999")
        monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1")
        weights: dict[tuple[str, str], object] = {}
        _solve_nvfp4_activation_clip_groups(
            groups={"fused": ["a", "b"]},
            qname_to_module=modules,
            activations=activations,
            levers={"gptq": True, "scale_sweep": True},
            joint_globals={},
            activation_max_abs={
                name: float(act.abs().max().item())
                for name, act in activations.items()
            },
            fisher_rows=None,
            weights_out=weights,
            cache_dir_path=None,
            progress=False,
        )
        return weights

    scalar = solve(False)
    batched = solve(True)

    for key in scalar:
        assert torch.allclose(
            scalar[key],
            batched[key],
            atol=1e-4,
            rtol=1e-3,
        ), key


def test_production_cache_passes_fisher_row_weights(monkeypatch, tmp_path):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    torch.save({"g2_per_token": torch.tensor([1.0, 2.0, 3.0])}, tmp_path / "l1.pt")
    seen: list[torch.Tensor | None] = []

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        fisher_row_weights=None,
        **_kwargs,
    ):
        seen.append(fisher_row_weights)
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False, "fisher_gptq": True},
        h_detail_dir=tmp_path,
        max_act_rows=8,
        progress=False,
    )

    assert seen and torch.equal(seen[0], torch.tensor([1.0, 2.0, 3.0]))
    assert cache.metadata["fisher_weighted_gptq"]["loaded"] == 1


def test_production_cache_fisher_rows_resolve_fused_qkv_and_gate_up(
    monkeypatch,
    tmp_path,
):
    import prismaquant.production_weight_cache as pwc

    class FusedTiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4, 32)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            layer = self.model.layers[0]
            layer.self_attn = nn.Module()
            layer.self_attn.qkv_proj = nn.Linear(32, 96, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.gate_up_proj = nn.Linear(32, 64, bias=False)

        def forward(self, input_ids, use_cache=False):
            x = self.embed(input_ids)
            layer = self.model.layers[0]
            return layer.self_attn.qkv_proj(x) + layer.mlp.gate_up_proj(x).sum() * 0

    def save_detail(name: str, values: list[float]) -> None:
        safe = pwc._FisherRowWeightCache._FNAME_SUB.sub("__", name)
        torch.save({"g2_per_token": torch.tensor(values)}, tmp_path / f"{safe}.pt")

    save_detail("model.layers.0.self_attn.q_proj", [1.0, 2.0, 3.0])
    save_detail("model.layers.0.self_attn.k_proj", [3.0, 4.0, 5.0])
    save_detail("model.layers.0.self_attn.v_proj", [5.0, 6.0, 7.0])
    save_detail("model.layers.0.mlp.gate_proj", [2.0, 4.0, 6.0])
    save_detail("model.layers.0.mlp.up_proj", [4.0, 6.0, 8.0])

    seen: dict[str, torch.Tensor | None] = {}

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        qname=None,
        fisher_row_weights=None,
        **_kwargs,
    ):
        seen[str(qname)] = fisher_row_weights
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)

    cache = fill_production_weight_cache(
        FusedTiny(),
        torch.tensor([[0, 1]], dtype=torch.long),
        qnames=[
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.mlp.gate_up_proj",
        ],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False, "fisher_gptq": True},
        h_detail_dir=tmp_path,
        max_act_rows=8,
        progress=False,
    )

    assert torch.equal(
        seen["model.layers.0.self_attn.qkv_proj"],
        torch.tensor([3.0, 4.0, 5.0]),
    )
    assert torch.equal(
        seen["model.layers.0.mlp.gate_up_proj"],
        torch.tensor([3.0, 5.0, 7.0]),
    )
    assert cache.metadata["fisher_weighted_gptq"]["loaded"] == 2
    assert cache.metadata["fisher_weighted_gptq"]["misses"] == 0


def test_production_cache_mxfp8_uses_activation_scale_sweep(monkeypatch):
    import prismaquant.export_native_compressed as enc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls = []

    def fake_mxfp8_scale_sweep(weight, activations, **_kwargs):
        calls.append(activations.shape)
        rows, cols = weight.shape
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.zeros((rows, cols // 32), dtype=torch.uint8),
            weight.detach().to(torch.float32) + 2.0,
        )

    monkeypatch.setattr(enc, "_mxfp8_scale_sweep_quantize", fake_mxfp8_scale_sweep)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["MXFP8_E4M3"],
        levers={"gptq": False, "scale_sweep": True},
        max_act_rows=8,
        progress=False,
    )

    assert calls
    assert torch.allclose(
        cache.get("l1", "MXFP8_E4M3").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + 2.0,
    )


def test_fill_production_cache_assignment_scope_only_renders_selected_formats(
    monkeypatch,
):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    seen: list[tuple[str | None, str]] = []

    def fake_render_production_weight(weight, fmt, *, qname=None, **_kwargs):
        seen.append((qname, fmt.upper()))
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1", "l2"],
        formats=["NVFP4", "MXFP8_E4M3"],
        render_assignment={"l1": "NVFP4", "l2": "BF16"},
        levers={"gptq": False, "scale_sweep": False},
        max_act_rows=8,
        progress=False,
    )

    assert seen == [("l1", "NVFP4")]
    assert ("l1", "NVFP4") in cache
    assert ("l1", "MXFP8_E4M3") not in cache
    assert ("l2", "NVFP4") not in cache
    assert cache.metadata["render_scope"] == "assignment"
    assert cache.metadata["requested_entries"] == 1


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
