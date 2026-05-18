"""Block-CLADO pipeline component contract.

The executable Block-CLADO implementation remains archived under
``archive/cross_layer_2026-05-09``.  This module only declares how that
research path plugs into the typed pipeline: artifacts, gates, cache ownership,
and archive entrypoints.  It deliberately does not import the archived code.
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
RUNTIME_MODULE = "prismaquant.research_components.block_clado_runtime"


def block_clado_component_spec() -> PipelineComponentSpec:
    """Return the opt-in Block-CLADO research component contract."""

    artifacts = (
        ArtifactSpec(
            "block_clado_payload",
            "block_clado_payload",
            version="prismaquant.block_clado.v1",
            description="Measured unary and intra-block pair interaction costs.",
        ),
        ArtifactSpec(
            "block_clado_frontier",
            "surrogate_frontier",
            version="prismaquant.block_clado.sweep.v1",
            description="Lambda-sweep or budget-solve frontier from the CLADO solver.",
        ),
        ArtifactSpec(
            "block_clado_candidate_assignments",
            "candidate_layer_configs",
            version="prismaquant.block_clado.kneedle.v1",
            description="Kneedle and neighbor assignments expanded to per-Linear formats.",
        ),
        ArtifactSpec(
            "block_clado_validation_metrics",
            "validation_metrics",
            version="prismaquant.block_clado.validate.v1",
            description="Real-KL validation rows for CLADO candidates.",
        ),
        ArtifactSpec(
            "block_clado_layer_assignment",
            "layer_config",
            description="Selected CLADO assignment candidate, still opt-in.",
        ),
    )
    gates = (
        MetricGateSpec(
            name="gate.research.block_clado.real_kl",
            metric="real_kl",
            mode="any",
            direction="lower_is_better",
            description=(
                "At least one CLADO candidate must improve measured KL before "
                "it can replace the baseline assignment."
            ),
        ),
    )
    stages = (
        PipelineStageSpec(
            name="research.block_clado.measure",
            component=f"{RUNTIME_MODULE}:collect_block_clado",
            inputs=(
                "source_model",
                "model_graph",
                "calibration_batch",
                "layer_assignment",
                "resident_production_weight_cache",
            ),
            outputs=("block_clado_payload",),
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
            tags=("research", "block_clado", "measurement", "gpu_bound"),
            metadata={
                "schema": "prismaquant.block_clado.v1",
                "runtime_module": RUNTIME_MODULE,
                "archive_path": f"{ARCHIVE_ROOT}/measure_block_clado.py",
                "center_assignment_input": "layer_assignment",
                "requires_production_weight_cache": True,
            },
            description=(
                "Measure Block-CLADO unary and intra-block pair terms using "
                "the existing production weight cache."
            ),
        ),
        PipelineStageSpec(
            name="research.block_clado.solve",
            component=f"{RUNTIME_MODULE}:sweep_payload",
            inputs=("block_clado_payload",),
            outputs=("block_clado_frontier",),
            tags=("research", "block_clado", "cpu_solver"),
            metadata={
                "runtime_module": RUNTIME_MODULE,
                "archive_path": f"{ARCHIVE_ROOT}/block_clado.py",
                "solver_modes": ["lambda_sweep", "budget"],
            },
            description="Solve the Block-CLADO surrogate frontier.",
        ),
        PipelineStageSpec(
            name="research.block_clado.kneedle",
            component=f"{RUNTIME_MODULE}:kneedle_payloads",
            inputs=("block_clado_payload", "block_clado_frontier"),
            outputs=("block_clado_candidate_assignments",),
            tags=("research", "block_clado", "candidate_generation"),
            metadata={
                "runtime_module": RUNTIME_MODULE,
                "archive_path": f"{ARCHIVE_ROOT}/block_clado.py",
                "default_neighbors": 2,
            },
            description="Expand kneedle and neighbor frontier points to layer configs.",
        ),
        PipelineStageSpec(
            name="research.block_clado.validate",
            component="prismaquant.validate_assignments_kl:main",
            inputs=(
                "source_model",
                "calibration_batch",
                "block_clado_candidate_assignments",
                "resident_production_weight_cache",
            ),
            outputs=("block_clado_validation_metrics",),
            gates=("gate.research.block_clado.real_kl",),
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
            tags=("research", "block_clado", "validation", "gpu_bound"),
            metadata={
                "runtime_module": "prismaquant.validate_assignments_kl",
                "archive_path": f"{ARCHIVE_ROOT}/validate_block_clado.py",
                "selection_metric": "real_kl",
            },
            description="Validate CLADO candidates with measured KL.",
        ),
        PipelineStageSpec(
            name="research.block_clado.select",
            component="pipeline:select_lowest_metric_candidate",
            inputs=("block_clado_validation_metrics",),
            outputs=("block_clado_layer_assignment",),
            tags=("research", "block_clado", "selection"),
            metadata={
                "metric": "real_kl",
                "output_assignment_key": "assignment",
                "promotion_required": False,
            },
            description=(
                "Select the best validated CLADO candidate without replacing "
                "the production assignment automatically."
            ),
        ),
    )
    return PipelineComponentSpec(
        id="block_clado",
        artifacts=artifacts,
        gates=gates,
        stages=stages,
        insert_after="cache.prefetch_assignment",
        status="research",
        default_enabled=False,
        description=(
            "Archived Block-CLADO measurement/solve/validation path exposed "
            "as an opt-in pipeline component."
        ),
        metadata={
            "archive_root": ARCHIVE_ROOT,
            "runtime_module": RUNTIME_MODULE,
            "implementation_status": "ported_measure_solve_kneedle",
            "production_default": False,
            "live_replacements": {
                "decision_units": "prismaquant.decision_units",
                "kl_validation": "prismaquant.validate_assignments_kl",
                "production_cache": "prismaquant.production_weight_cache",
            },
            "known_porting_work": [
                "define the assignment promotion gate before export consumes "
                "block_clado_layer_assignment",
                "port iterate_block_clado sandwich orchestration onto the "
                "component runtime if multi-round CLADO becomes useful",
            ],
        },
    )
