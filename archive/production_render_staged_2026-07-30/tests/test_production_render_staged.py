"""Staged production-render-cost tests — ARCHIVED 2026-07-30 (re-vet R17).

Lifted verbatim out of `tests/test_production_render_cost.py` when
`COST_MODE=production-render-staged` was walled. They are a **record**, not a
runnable suite: `select_tail_from_render_scores` now lives in
`../prismaquant/production_render_staged.py` and the
`missing_render_score_policy` / `promotion_qnames` / `bf16_policy` parameters
were removed from `prismaquant.production_render_cost`. Restoring the lane
means restoring both, then moving these two tests back.

`_cache_with_scores()` is the fixture from the live test file.
"""

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
