"""SMRF pipeline component contract.

The archived SMRF/PrismaSCOUT path generated the only published PrismaSCOUT
deliverable, but its archived orchestration is too coupled to restore as-is.
This module exposes the useful part as an opt-in pipeline component: generate
candidate assignments from a decision-unit cost payload, then validate and
select using the shared measured-KL frontier tooling.
"""
from __future__ import annotations

from prismaquant.pipeline import (
    ArtifactSpec,
    MetricGateSpec,
    PipelineComponentSpec,
    PipelineStageSpec,
    ResourceContract,
)


ARCHIVE_ROOT = "archive/cross_layer_2026-05-09"
RUNTIME_MODULE = "prismaquant.research_components.smrf_runtime"


def smrf_component_spec() -> PipelineComponentSpec:
    """Return the opt-in SMRF research component contract."""

    artifacts = (
        ArtifactSpec(
            "smrf_cost_payload",
            "decision_unit_cost_payload",
            version="prismaquant.block_clado.v1",
            provided=True,
            description=(
                "Decision-unit additive cost payload consumed by the SMRF "
                "candidate generator. Currently supplied explicitly by the run."
            ),
        ),
        ArtifactSpec(
            "smrf_candidate_archive",
            "surrogate_frontier",
            version="prismaquant.smrf.archive.v1",
            description="SMRF surrogate archive and kneedle/neighbor candidates.",
        ),
        ArtifactSpec(
            "smrf_candidate_assignments",
            "candidate_layer_configs",
            version="prismaquant.smrf.candidates.v1",
            description="Assignment manifest emitted for measured KL validation.",
        ),
        ArtifactSpec(
            "smrf_validation_metrics",
            "validation_metrics",
            version="prismaquant.smrf.validate.v1",
            description="Real KL/bpp rows for SMRF candidates.",
        ),
        ArtifactSpec(
            "smrf_layer_assignment",
            "layer_config",
            description="Selected SMRF assignment candidate, still opt-in.",
        ),
    )
    gates = (
        MetricGateSpec(
            name="gate.research.smrf.last_token_kl",
            metric="last_token_kl",
            mode="any",
            direction="lower_is_better",
            description=(
                "At least one SMRF candidate must improve measured validation "
                "KL before it can replace the baseline assignment."
            ),
        ),
    )
    stages = (
        PipelineStageSpec(
            name="research.smrf.generate",
            component=f"{RUNTIME_MODULE}:generate_archive_candidates",
            inputs=("smrf_cost_payload",),
            outputs=("smrf_candidate_archive", "smrf_candidate_assignments"),
            tags=("research", "smrf", "candidate_generation", "cpu_solver"),
            metadata={
                "runtime_module": RUNTIME_MODULE,
                "archive_path": f"{ARCHIVE_ROOT}/iterate_perturbed_allocation.py",
                "archive_entrypoints": [
                    "solve_l3_pareto_archive_assignments",
                    "run_knee_archive_search",
                ],
                "default_beam_per_bin": 4,
                "default_validation_candidates": 9,
                "uses_real_validation_for_promotion": True,
            },
            description=(
                "Generate SMRF candidate assignments from additive decision-unit "
                "costs. This stage does not measure KL or promote assignments."
            ),
        ),
        PipelineStageSpec(
            name="research.smrf.validate",
            component="prismaquant.validate_assignments_kl:main",
            inputs=(
                "source_model",
                "calibration_batch",
                "smrf_candidate_assignments",
                "resident_production_weight_cache",
            ),
            outputs=("smrf_validation_metrics",),
            gates=("gate.research.smrf.last_token_kl",),
            resources=(
                ResourceContract(
                    resource="rendered_weights",
                    owner="ProductionWeightCache",
                    residency="required",
                ),
                ResourceContract(
                    resource="perturbed_activations",
                    owner="PerturbedActivationCache",
                    residency="optional",
                    required=False,
                ),
            ),
            tags=("research", "smrf", "validation", "gpu_bound"),
            metadata={
                "runtime_module": "prismaquant.validate_assignments_kl",
                "selection_metric": "last_token_kl",
                "requires_production_weight_cache": True,
            },
            description="Validate SMRF candidates with measured KL.",
        ),
        PipelineStageSpec(
            name="research.smrf.select",
            component="prismaquant.select_validated_frontier:main",
            inputs=("smrf_validation_metrics",),
            outputs=("smrf_layer_assignment",),
            tags=("research", "smrf", "selection"),
            metadata={
                "metric": "last_token_kl",
                "mode": "kneedle",
                "promotion_required": False,
            },
            description=(
                "Select a measured frontier point without automatically "
                "replacing the production assignment."
            ),
        ),
    )
    return PipelineComponentSpec(
        id="smrf",
        artifacts=artifacts,
        gates=gates,
        stages=stages,
        insert_after="cache.prefetch_assignment",
        status="research",
        default_enabled=False,
        description=(
            "Archived SMRF/PrismaSCOUT candidate generation exposed as an "
            "opt-in pipeline component backed by shared KL validation."
        ),
        metadata={
            "archive_root": ARCHIVE_ROOT,
            "runtime_module": RUNTIME_MODULE,
            "implementation_status": "ported_archive_candidate_generator",
            "production_default": False,
            "historical_deliverable": (
                "Qwen3.6-27B PrismaSCOUT no-seed knee archive at "
                "5.309499798305769 bpp"
            ),
            "live_replacements": {
                "decision_units": "prismaquant.decision_units",
                "candidate_generation": RUNTIME_MODULE,
                "kl_validation": "prismaquant.validate_assignments_kl",
                "frontier_selection": "prismaquant.select_validated_frontier",
                "production_cache": "prismaquant.production_weight_cache",
            },
            "known_porting_work": [
                "port SMRF/L3 cost measurement onto a GPU-bound, "
                "ProductionWeightCache-resident pipeline stage",
                "compare modern SMRF candidates against standard PQ using the "
                "same calibration set, bpp accounting, and vLLM gate",
            ],
        },
    )
