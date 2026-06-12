import re
from pathlib import Path


def _run_pipeline_script() -> str:
    return (
        Path(__file__).resolve().parent.parent / "prismaquant" / "run-pipeline.sh"
    ).read_text()


def _shell_default(script: str, name: str) -> str:
    match = re.search(rf'^\s*:\s*"\$\{{{re.escape(name)}:=([^}}]*)\}}"\s*$',
                      script, re.MULTILINE)
    assert match is not None, f"missing shell default for {name}"
    return match.group(1)


def test_production_recache_default_enabled_after_smoke_ladder():
    script = _run_pipeline_script()

    assert "PRODUCTION_CACHE:=1" in script
    assert "PRODUCTION_RECACHE:=1" in script
    assert "PRODUCTION_RECACHE=0" in script
    assert "PIPELINE_SPEC_PATH:=${WORK_DIR}/artifacts/pipeline_spec.json" in script
    assert "COST_MODE:=production-render-score" in script
    assert "PRODUCTION_CACHE_LEVERS:=gptq,static_act_order,joint_scale_opt" in script
    assert "includes static_act_order" not in script
    assert "production-render-staged|production-render-tail" in script
    assert "python3 -m prismaquant.pipeline" in script
    assert "--write-default-production" in script
    assert "--target-profile \"$TARGET_PROFILE\"" in script
    assert ': "${HADAMARD_DUQUANT' not in script
    assert "HADAMARD_DUQUANT:-" in script
    assert "archive/hdq_2026-05-14" in script
    assert "H_DETAIL_DIR" not in script
    assert "LEVER_CACHE_TAG" not in script
    assert "PROD_H_DETAIL_ARGS" not in script


def test_multi_shot_passes_is_archived_and_blocked():
    """MULTI_SHOT_PASSES>1 fails fast with a pointer to the archive after the
    cross-layer-interaction work landed null. See
    archive/multi_shot_2026-05-19/README.md for the validation record."""
    script = _run_pipeline_script()

    assert "MULTI_SHOT_PASSES" in script
    assert "archive/multi_shot_2026-05-19" in script
    assert ': "${MULTI_SHOT_PASSES' not in script  # no opt-in default; user must explicitly opt out of vanilla


def test_grouped_kl_is_archived_and_blocked():
    """COST_MODE=grouped-kl fails fast with a pointer to the archive after it
    lost the shipped vLLM A/B on Qwen3.6-27B. See
    archive/grouped_kl_2026-05-28/README.md for the validation record."""
    script = _run_pipeline_script()

    # grouped-kl is now a fail-fast dispatch arm pointing at the archive.
    assert "archive/grouped_kl_2026-05-28" in script
    # It is no longer advertised as a valid COST_MODE in the catch-all error.
    assert (
        "COST_MODE must be local, production-render-score, or production-render-staged"
        in script
    )
    # The grouped-kl measurement invocation and its env knobs are gone.
    assert "prismaquant.grouped_kl_cost" not in script
    assert "GROUPED_KL_NSAMPLES" not in script
    assert "GROUPED_KL_MAX_LANES" not in script
    # production-render-score remains the default cost mode.
    assert "COST_MODE:=production-render-score" in script


def test_mse_promotion_fails_fast_when_inert():
    script = _run_pipeline_script()

    assert "MSE_PROMOTION requires SELECTION_MODE=validated-surrogate" in script
    assert "MSE_PROMOTION requires PRODUCTION_CACHE=1" in script
    assert "would be inert under SELECTION_MODE=$SELECTION_MODE" in script


def test_core_recipe_defaults_are_pinned():
    script = _run_pipeline_script()

    assert _shell_default(script, "FORMATS") == "NVFP4,FP8_DYNAMIC,BF16"
    assert _shell_default(script, "TARGET_BITS") == "4.75"
    assert _shell_default(script, "SELECTION_MODE") == "surrogate"
