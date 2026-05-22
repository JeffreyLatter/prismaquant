from __future__ import annotations

import pytest

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import build_candidates
from prismaquant.production_render_cost import (
    select_tail_from_render_scores,
    synthesize_production_render_cost_payload,
)
from prismaquant.production_weight_cache import ProductionWeightCache


def _cache_with_scores() -> ProductionWeightCache:
    return ProductionWeightCache(
        weights={},
        levers={"gptq": True, "joint_scale_opt": True},
        metadata={
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "records": {
                    "layers.0.q_proj|NVFP4": {
                        "qname": "layers.0.q_proj",
                        "format": "NVFP4",
                        "metric": "output_mse",
                        "score": 0.5,
                        "score_sum": 12.0,
                        "normalizer": 24.0,
                        "activation_rows": 3,
                    },
                    "layers.0.q_proj|MXFP8_E4M3": {
                        "qname": "layers.0.q_proj",
                        "format": "MXFP8_E4M3",
                        "metric": "output_mse",
                        "score": 0.25,
                        "score_sum": 6.0,
                        "normalizer": 24.0,
                        "activation_rows": 3,
                    },
                },
            },
        },
    )


def test_production_render_cost_uses_render_score_directly():
    baseline = {
        "formats": ["NVFP4", "MXFP8_E4M3", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "MXFP8_E4M3": {"output_mse": 88.0},
                "BF16": {"predicted_dloss": 0.0},
            },
            "layers.0.o_proj": {
                "NVFP4": {"predicted_dloss": 4.0},
                "MXFP8_E4M3": {"predicted_dloss": 2.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }

    cost = synthesize_production_render_cost_payload(
        _cache_with_scores(),
        baseline,
    )

    q = cost["costs"]["layers.0.q_proj"]
    assert q["NVFP4"]["predicted_dloss"] == 12.0
    assert q["NVFP4"]["output_mse_measured"] is False
    assert q["NVFP4"]["cost_source"] == "production_render_score"
    assert q["MXFP8_E4M3"]["predicted_dloss"] == 6.0
    assert q["BF16"]["predicted_dloss"] == 0.0

    o = cost["costs"]["layers.0.o_proj"]
    assert o["NVFP4"]["predicted_dloss"] == 4.0
    assert o["NVFP4"]["cost_source"] == "fallback_baseline"
    assert cost["meta"]["render_score_entries"] == 2
    assert cost["meta"]["fallback_entries"] == 2


def test_production_render_cost_bypasses_h_trace_proxy_in_allocator():
    baseline = {
        "formats": ["NVFP4", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }
    cost = synthesize_production_render_cost_payload(
        _cache_with_scores(),
        baseline,
    )
    stats = {
        "layers.0.q_proj": {
            "h_trace": 1000.0,
            "out_features": 32,
            "in_features": 32,
            "n_params": 1024,
        },
    }

    candidates = build_candidates(
        stats,
        cost["costs"],
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
    )
    by_fmt = {cand.fmt: cand for cand in candidates["layers.0.q_proj"]}

    assert by_fmt["NVFP4"].predicted_dloss == 12.0
    assert by_fmt["BF16"].predicted_dloss == 0.0


def test_production_render_cost_can_reject_weight_mse_fallbacks():
    cache = _cache_with_scores()
    cache.metadata["render_scores"]["records"]["layers.0.q_proj|NVFP4"][
        "metric"
    ] = "weight_mse"
    baseline = {
        "formats": ["NVFP4"],
        "costs": {"layers.0.q_proj": {"NVFP4": {"predicted_dloss": 4.0}}},
    }

    with pytest.raises(ValueError, match="non-output metrics"):
        synthesize_production_render_cost_payload(
            cache,
            baseline,
            require_output_metric=True,
        )


def test_staged_render_cost_marks_unmeasured_promotions_unavailable():
    baseline = {
        "formats": ["NVFP4", "MXFP8_E4M3", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "MXFP8_E4M3": {"predicted_dloss": 6.0},
                "BF16": {"predicted_dloss": 0.0},
            },
            "layers.0.o_proj": {
                "NVFP4": {"predicted_dloss": 4.0},
                "MXFP8_E4M3": {"predicted_dloss": 2.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }

    cost = synthesize_production_render_cost_payload(
        _cache_with_scores(),
        baseline,
        missing_render_score_policy="unavailable",
        promotion_qnames={"layers.0.q_proj"},
        bf16_policy="promotion-set",
    )

    assert cost["costs"]["layers.0.q_proj"]["NVFP4"]["predicted_dloss"] == 12.0
    assert cost["costs"]["layers.0.q_proj"]["MXFP8_E4M3"]["predicted_dloss"] == 6.0
    assert cost["costs"]["layers.0.q_proj"]["BF16"]["predicted_dloss"] == 0.0
    assert "error" in cost["costs"]["layers.0.o_proj"]["MXFP8_E4M3"]
    assert "error" in cost["costs"]["layers.0.o_proj"]["BF16"]


def test_select_tail_from_nvfp4_render_scores():
    selected, summary = select_tail_from_render_scores(
        _cache_with_scores(),
        fmt="NVFP4",
        score_field="score_sum",
        top_fraction=0.5,
    )

    assert selected == ["layers.0.q_proj"]
    assert summary["available"] == 1
    assert summary["selected"] == 1
