"""Typed pipeline, artifact, resource, and gate contracts.

This module is intentionally descriptive first.  It lets existing PrismaQuant
stages advertise their inputs, outputs, gates, and cache ownership without
replacing the production implementations.  Execution stays in the current
GPU-bound probe/cache/export paths until call sites opt into these contracts.
"""
from __future__ import annotations

import math
import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


APPROVED_RESOURCE_OWNERS: dict[str, frozenset[str]] = {
    "rendered_weights": frozenset({"ProductionWeightCache"}),
    "perturbed_activations": frozenset({
        "PerturbedActivationCache",
        "StreamingActivationCache",
    }),
    "streaming_model_weights": frozenset({"StreamingModelPrefetch"}),
}


@dataclass(frozen=True)
class ArtifactSpec:
    """One typed artifact that can enter or leave a pipeline stage."""

    name: str
    kind: str
    version: str = "v1"
    description: str = ""
    resident: bool = False
    provided: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("artifact name must be non-empty")
        if not self.kind:
            raise ValueError(f"{self.name}: artifact kind must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactSpec":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            version=str(payload.get("version", "v1")),
            description=str(payload.get("description", "")),
            resident=bool(payload.get("resident", False)),
            provided=bool(payload.get("provided", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "description": self.description,
            "resident": self.resident,
            "provided": self.provided,
        }


@dataclass(frozen=True)
class ResourceContract:
    """Cache/prefetch ownership required by a stage.

    ``resource`` is the data class being managed, for example
    ``rendered_weights``.  ``owner`` is the implementation that owns residency.
    Validation rejects unapproved owners for resources covered by PrismaQuant's
    one-cache rule.
    """

    resource: str
    owner: str
    residency: str = "none"
    required: bool = True
    gpu_bound: bool = True
    fail_fast: bool = True

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("resource contract requires a resource name")
        if not self.owner:
            raise ValueError(f"{self.resource}: resource owner must be non-empty")
        if self.residency not in {"none", "optional", "required"}:
            raise ValueError(
                f"{self.resource}: invalid residency {self.residency!r}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceContract":
        return cls(
            resource=str(payload["resource"]),
            owner=str(payload["owner"]),
            residency=str(payload.get("residency", "none")),
            required=bool(payload.get("required", True)),
            gpu_bound=bool(payload.get("gpu_bound", True)),
            fail_fast=bool(payload.get("fail_fast", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "owner": self.owner,
            "residency": self.residency,
            "required": self.required,
            "gpu_bound": self.gpu_bound,
            "fail_fast": self.fail_fast,
        }


@dataclass(frozen=True)
class MetricDecision:
    key: str
    accepted: bool
    baseline: float | None
    candidate: float | None
    delta: float | None
    relative_gain: float | None
    reason: str


@dataclass(frozen=True)
class GateEvaluation:
    gate_name: str
    passed: bool
    decisions: tuple[MetricDecision, ...]

    def accepted_keys(self) -> tuple[str, ...]:
        return tuple(decision.key for decision in self.decisions if decision.accepted)

    def rejected_keys(self) -> tuple[str, ...]:
        return tuple(
            decision.key for decision in self.decisions if not decision.accepted
        )


@dataclass(frozen=True)
class MetricGateSpec:
    """A configurable metric gate for global or per-item decisions.

    Examples:
      - global KL gate: candidate ``end_kl`` must be lower than baseline.
      - local render gate: per-Linear ``output_mse`` must improve; accepted
        keys are the Linears that should receive the candidate transform.
    """

    name: str
    metric: str
    direction: str = "lower_is_better"
    mode: str = "all"
    min_absolute_delta: float = 0.0
    min_relative_gain: float = 0.0
    require_improvement: bool = True
    max_absolute_regression: float = 0.0
    max_relative_regression: float = 0.0
    missing: str = "fail"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name must be non-empty")
        if not self.metric:
            raise ValueError(f"{self.name}: metric must be non-empty")
        if self.direction not in {"lower_is_better", "higher_is_better"}:
            raise ValueError(f"{self.name}: invalid direction {self.direction!r}")
        if self.mode not in {"all", "any", "per_item"}:
            raise ValueError(f"{self.name}: invalid mode {self.mode!r}")
        if self.missing not in {"fail", "skip", "pass"}:
            raise ValueError(f"{self.name}: invalid missing policy {self.missing!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricGateSpec":
        return cls(
            name=str(payload["name"]),
            metric=str(payload["metric"]),
            direction=str(payload.get("direction", "lower_is_better")),
            mode=str(payload.get("mode", "all")),
            min_absolute_delta=float(payload.get("min_absolute_delta", 0.0)),
            min_relative_gain=float(payload.get("min_relative_gain", 0.0)),
            require_improvement=bool(payload.get("require_improvement", True)),
            max_absolute_regression=float(
                payload.get("max_absolute_regression", 0.0)
            ),
            max_relative_regression=float(
                payload.get("max_relative_regression", 0.0)
            ),
            missing=str(payload.get("missing", "fail")),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "direction": self.direction,
            "mode": self.mode,
            "min_absolute_delta": self.min_absolute_delta,
            "min_relative_gain": self.min_relative_gain,
            "require_improvement": self.require_improvement,
            "max_absolute_regression": self.max_absolute_regression,
            "max_relative_regression": self.max_relative_regression,
            "missing": self.missing,
            "description": self.description,
        }

    def evaluate(
        self,
        *,
        baseline: Mapping[str, Any] | float | int,
        candidate: Mapping[str, Any] | float | int,
        keys: Iterable[str] | None = None,
    ) -> GateEvaluation:
        baseline_values = _metric_values(baseline, self.metric)
        candidate_values = _metric_values(candidate, self.metric)
        if keys is None:
            eval_keys = tuple(
                sorted(set(baseline_values) | set(candidate_values))
            )
        else:
            eval_keys = tuple(str(key) for key in keys)
        decisions = tuple(
            self._decision_for(
                key,
                baseline_values.get(key),
                candidate_values.get(key),
            )
            for key in eval_keys
        )
        if self.mode == "all":
            passed = bool(decisions) and all(d.accepted for d in decisions)
        elif self.mode == "any":
            passed = any(d.accepted for d in decisions)
        else:
            passed = True
        return GateEvaluation(
            gate_name=self.name,
            passed=bool(passed),
            decisions=decisions,
        )

    def _decision_for(
        self,
        key: str,
        baseline: float | None,
        candidate: float | None,
    ) -> MetricDecision:
        if baseline is None or candidate is None:
            accepted = self.missing == "pass"
            reason = "missing"
            return MetricDecision(
                key=key,
                accepted=accepted,
                baseline=baseline,
                candidate=candidate,
                delta=None,
                relative_gain=None,
                reason=reason,
            )
        if not math.isfinite(float(baseline)) or not math.isfinite(float(candidate)):
            return MetricDecision(
                key=key,
                accepted=False,
                baseline=float(baseline),
                candidate=float(candidate),
                delta=None,
                relative_gain=None,
                reason="non_finite",
            )
        if self.direction == "lower_is_better":
            delta = float(baseline) - float(candidate)
        else:
            delta = float(candidate) - float(baseline)
        denom = max(abs(float(baseline)), 1e-30)
        relative_gain = delta / denom
        accepted = (
            delta > 0.0
            and delta >= float(self.min_absolute_delta)
            and relative_gain >= float(self.min_relative_gain)
        )
        if not accepted and not self.require_improvement:
            regression = max(-delta, 0.0)
            relative_regression = regression / denom
            abs_budget = (
                float(self.max_absolute_regression)
                if self.max_absolute_regression > 0.0
                else float("inf")
            )
            accepted = (
                regression <= abs_budget
                and relative_regression <= float(self.max_relative_regression)
            )
        if accepted:
            reason = "improved" if delta > 0.0 else "within_regression_budget"
        elif delta <= 0.0:
            reason = "regressed_or_tied"
        elif delta < float(self.min_absolute_delta):
            reason = "below_min_absolute_delta"
        else:
            reason = "below_min_relative_gain"
        return MetricDecision(
            key=key,
            accepted=bool(accepted),
            baseline=float(baseline),
            candidate=float(candidate),
            delta=float(delta),
            relative_gain=float(relative_gain),
            reason=reason,
        )


@dataclass(frozen=True)
class PipelineStageSpec:
    """One pluggable stage in a PrismaQuant pipeline."""

    name: str
    component: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    gates: tuple[str, ...] = ()
    resources: tuple[ResourceContract, ...] = ()
    tags: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if not self.component:
            raise ValueError(f"{self.name}: component must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineStageSpec":
        return cls(
            name=str(payload["name"]),
            component=str(payload["component"]),
            inputs=tuple(str(v) for v in payload.get("inputs", ())),
            outputs=tuple(str(v) for v in payload.get("outputs", ())),
            gates=tuple(str(v) for v in payload.get("gates", ())),
            resources=tuple(
                ResourceContract.from_dict(entry)
                for entry in payload.get("resources", ())
            ),
            tags=tuple(str(v) for v in payload.get("tags", ())),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "gates": list(self.gates),
            "resources": [resource.to_dict() for resource in self.resources],
            "tags": list(self.tags),
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PipelineValidation:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PipelineComponentSpec:
    """A named, opt-in pipeline extension.

    Components are contract fragments: they can declare artifacts, gates, and
    stages without taking over core execution.  This is the integration point
    for archived or experimental methods that need to be wired into the
    pluggable pipeline while remaining explicit and off by default.
    """

    id: str
    stages: tuple[PipelineStageSpec, ...]
    artifacts: tuple[ArtifactSpec, ...] = ()
    gates: tuple[MetricGateSpec, ...] = ()
    insert_after: str | None = None
    status: str = "research"
    default_enabled: bool = False
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("component id must be non-empty")
        if self.status not in {
            "research",
            "candidate",
            "production_recipe",
            "default_on",
        }:
            raise ValueError(f"{self.id}: invalid component status {self.status!r}")
        if self.default_enabled and self.status in {"research", "candidate"}:
            raise ValueError(
                f"{self.id}: {self.status} components must be opt-in"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineComponentSpec":
        return cls(
            id=str(payload["id"]),
            stages=tuple(
                PipelineStageSpec.from_dict(entry)
                for entry in payload.get("stages", ())
            ),
            artifacts=tuple(
                ArtifactSpec.from_dict(entry)
                for entry in payload.get("artifacts", ())
            ),
            gates=tuple(
                MetricGateSpec.from_dict(entry)
                for entry in payload.get("gates", ())
            ),
            insert_after=(
                None
                if payload.get("insert_after") is None
                else str(payload.get("insert_after"))
            ),
            status=str(payload.get("status", "research")),
            default_enabled=bool(payload.get("default_enabled", False)),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "default_enabled": self.default_enabled,
            "insert_after": self.insert_after,
            "metadata": dict(self.metadata),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "gates": [gate.to_dict() for gate in self.gates],
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class PipelineSpec:
    """A declarative PrismaQuant pipeline plan."""

    id: str
    stages: tuple[PipelineStageSpec, ...]
    artifacts: tuple[ArtifactSpec, ...] = ()
    gates: tuple[MetricGateSpec, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineSpec":
        return cls(
            id=str(payload["id"]),
            stages=tuple(
                PipelineStageSpec.from_dict(entry)
                for entry in payload.get("stages", ())
            ),
            artifacts=tuple(
                ArtifactSpec.from_dict(entry)
                for entry in payload.get("artifacts", ())
            ),
            gates=tuple(
                MetricGateSpec.from_dict(entry)
                for entry in payload.get("gates", ())
            ),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "metadata": dict(self.metadata),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "gates": [gate.to_dict() for gate in self.gates],
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def artifact_map(self) -> dict[str, ArtifactSpec]:
        return {artifact.name: artifact for artifact in self.artifacts}

    def gate_map(self) -> dict[str, MetricGateSpec]:
        return {gate.name: gate for gate in self.gates}

    def validate(self) -> PipelineValidation:
        errors: list[str] = []
        warnings: list[str] = []
        artifact_names = [artifact.name for artifact in self.artifacts]
        errors.extend(_duplicates("artifact", artifact_names))
        gate_names = [gate.name for gate in self.gates]
        errors.extend(_duplicates("gate", gate_names))
        stage_names = [stage.name for stage in self.stages]
        errors.extend(_duplicates("stage", stage_names))

        declared = set(artifact_names)
        available = {
            artifact.name
            for artifact in self.artifacts
            if artifact.provided
        }
        produced: set[str] = set()
        known_gates = set(gate_names)
        for stage in self.stages:
            for gate in stage.gates:
                if gate not in known_gates:
                    errors.append(f"{stage.name}: unknown gate {gate!r}")
            for input_name in stage.inputs:
                if input_name not in available:
                    errors.append(
                        f"{stage.name}: input {input_name!r} is not available"
                    )
                if input_name not in declared and input_name not in produced:
                    warnings.append(
                        f"{stage.name}: input {input_name!r} is not declared"
                    )
            for output_name in stage.outputs:
                if output_name in produced:
                    errors.append(
                        f"{stage.name}: output {output_name!r} is produced twice"
                    )
                produced.add(output_name)
                available.add(output_name)
            for resource in stage.resources:
                allowed = APPROVED_RESOURCE_OWNERS.get(resource.resource)
                if allowed is not None and resource.owner not in allowed:
                    errors.append(
                        f"{stage.name}: {resource.resource} must use one of "
                        f"{sorted(allowed)}, got {resource.owner!r}"
                    )
                if (
                    resource.required
                    and resource.residency == "required"
                    and not resource.fail_fast
                ):
                    errors.append(
                        f"{stage.name}: required resident {resource.resource} "
                        "must fail fast on miss"
                    )
                if resource.required and not resource.gpu_bound:
                    warnings.append(
                        f"{stage.name}: required resource {resource.resource} "
                        "is not marked GPU-bound"
                    )
        return PipelineValidation(
            errors=tuple(errors),
            warnings=tuple(warnings),
        )


_STAGES: dict[str, PipelineStageSpec] = {}
_COMPONENTS: dict[str, PipelineComponentSpec] = {}
_BUILTINS_REGISTERED = False
_BUILTIN_COMPONENTS_REGISTERED = False


def register_pipeline_stage(spec: PipelineStageSpec) -> None:
    _STAGES[spec.name] = spec


def register_pipeline_component(spec: PipelineComponentSpec) -> None:
    _COMPONENTS[spec.id] = spec


def pipeline_stage(name: str) -> PipelineStageSpec:
    _ensure_builtins_registered()
    return _STAGES[str(name)]


def pipeline_component(name: str) -> PipelineComponentSpec:
    _ensure_components_registered()
    return _COMPONENTS[str(name)]


def registered_pipeline_stages() -> Mapping[str, PipelineStageSpec]:
    _ensure_builtins_registered()
    return dict(_STAGES)


def registered_pipeline_components() -> Mapping[str, PipelineComponentSpec]:
    _ensure_components_registered()
    return dict(_COMPONENTS)


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: pipeline spec must be a JSON object")
    return PipelineSpec.from_dict(payload)


def write_pipeline_spec(spec: PipelineSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(spec.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def parse_render_mechanisms(
    enabled: str | Iterable[str] | None,
    *,
    disabled: str | Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Normalize comma-separated render mechanism config.

    Production paths historically spell render levers as env vars such as
    ``PRODUCTION_CACHE_LEVERS=gptq,static_act_order,joint_scale_opt``.  The
    pipeline contract stores the resolved mechanism list so the run artifact
    records the same plugins the cache fill will execute.
    """

    requested = _split_csv(enabled)
    blocked = set(_split_csv(disabled))
    if "none" in requested:
        requested = ()
    return tuple(name for name in requested if name != "none" and name not in blocked)


def production_pipeline_spec_from_config(
    *,
    render_mechanisms: str | Iterable[str] | None = None,
    disabled_render_mechanisms: str | Iterable[str] | None = None,
    model_path: str | None = None,
    work_dir: str | None = None,
    formats: str | None = None,
    target_bits: float | None = None,
    target_profile: str | None = None,
    calibration_modality: str | None = None,
    selection_mode: str | None = None,
    production_cache: str | bool | None = None,
    production_recache: str | bool | None = None,
    components: Iterable[str | PipelineComponentSpec] | None = None,
) -> PipelineSpec:
    """Build the production contract for one configured run."""

    mechanisms = parse_render_mechanisms(
        render_mechanisms,
        disabled=disabled_render_mechanisms,
    )
    spec = default_production_pipeline_spec(render_mechanisms=mechanisms)
    omitted_stages: list[str] = []
    if str(selection_mode or "").strip().lower() == "surrogate":
        omitted_stages.append("validate.kl")
    # run-pipeline.sh records the vLLM smoke command for manual execution; it
    # does not execute that stage as part of the default production run.
    omitted_stages.append("validate.vllm_smoke")
    if omitted_stages:
        omitted = set(omitted_stages)
        spec = PipelineSpec(
            id=spec.id,
            artifacts=spec.artifacts,
            gates=spec.gates,
            stages=tuple(stage for stage in spec.stages if stage.name not in omitted),
            description=spec.description,
            metadata=dict(spec.metadata),
        )
    component_specs = tuple(_resolve_pipeline_component(c) for c in (components or ()))
    if component_specs:
        spec = compose_pipeline_spec(spec, component_specs)
    ordered_mechanisms = tuple(
        str(stage.metadata["mechanism"])
        for stage in spec.stages
        if stage.name.startswith("render.") and "mechanism" in stage.metadata
    )
    metadata = {
        "render_mechanisms": list(ordered_mechanisms),
        "model_path": model_path,
        "work_dir": work_dir,
        "formats": formats,
        "target_bits": target_bits,
        "target_profile": target_profile,
        "calibration_modality": calibration_modality,
        "selection_mode": selection_mode,
        "production_cache": production_cache,
        "production_recache": production_recache,
    }
    if omitted_stages:
        metadata["omitted_unexecuted_stages"] = list(omitted_stages)
    if component_specs:
        metadata["components"] = list(spec.metadata.get("components", ()))
    return PipelineSpec(
        id=spec.id,
        artifacts=spec.artifacts,
        gates=spec.gates,
        stages=spec.stages,
        description=spec.description,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def compose_pipeline_spec(
    base: PipelineSpec,
    components: Iterable[str | PipelineComponentSpec],
) -> PipelineSpec:
    """Return ``base`` plus opt-in component contract fragments."""

    artifacts = list(base.artifacts)
    gates = list(base.gates)
    stages = list(base.stages)
    artifact_by_name = {artifact.name: artifact for artifact in artifacts}
    gate_by_name = {gate.name: gate for gate in gates}
    enabled_components: list[dict[str, Any]] = []

    for raw_component in components:
        component = _resolve_pipeline_component(raw_component)
        for artifact in component.artifacts:
            existing = artifact_by_name.get(artifact.name)
            if existing is not None:
                if existing.to_dict() != artifact.to_dict():
                    raise ValueError(
                        f"{component.id}: artifact {artifact.name!r} conflicts "
                        "with the base pipeline"
                    )
                continue
            artifacts.append(artifact)
            artifact_by_name[artifact.name] = artifact
        for gate in component.gates:
            existing = gate_by_name.get(gate.name)
            if existing is not None:
                if existing.to_dict() != gate.to_dict():
                    raise ValueError(
                        f"{component.id}: gate {gate.name!r} conflicts "
                        "with the base pipeline"
                    )
                continue
            gates.append(gate)
            gate_by_name[gate.name] = gate

        insert_at = len(stages)
        if component.insert_after:
            for idx, stage in enumerate(stages):
                if stage.name == component.insert_after:
                    insert_at = idx + 1
                    break
            else:
                raise ValueError(
                    f"{component.id}: insert_after stage "
                    f"{component.insert_after!r} was not found"
                )
        stages[insert_at:insert_at] = component.stages
        enabled_components.append({
            "id": component.id,
            "status": component.status,
            "default_enabled": component.default_enabled,
        })

    metadata = dict(base.metadata)
    metadata["components"] = list(metadata.get("components", ())) + enabled_components
    return PipelineSpec(
        id=base.id,
        artifacts=tuple(artifacts),
        gates=tuple(gates),
        stages=tuple(stages),
        description=base.description,
        metadata=metadata,
    )


def render_mechanism_stage_specs(enabled: Iterable[str]) -> tuple[PipelineStageSpec, ...]:
    """Expose registered render mechanisms as pipeline stages."""

    requested = tuple(enabled)
    if not requested:
        return ()

    from .render_score import resolve_render_mechanism_order

    plan = resolve_render_mechanism_order(requested)
    if plan.errors:
        raise ValueError("; ".join(plan.errors))
    stages: list[PipelineStageSpec] = []
    current_input = "render.baseline_weight"
    for spec in plan.ordered:
        output = f"render.after.{spec.name}"
        stages.append(PipelineStageSpec(
            name=f"render.{spec.name}",
            component=f"render_score:{spec.name}",
            inputs=(
                current_input,
                "render.reference_weight",
                "render.activation_rows",
            ),
            outputs=(output,),
            gates=(f"gate.render.{spec.gate_metric}",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="required",
            ),),
            tags=("render", spec.operation, spec.scope),
            metadata={
                "mechanism": spec.name,
                "operation": spec.operation,
                "scope": spec.scope,
                "phase": spec.phase,
                "gate_metric": spec.gate_metric,
            },
            description=spec.description,
        ))
        current_input = output
    return tuple(stages)


def default_production_pipeline_spec(
    *,
    render_mechanisms: Iterable[str] = (
        "four_over_six",
        "static_act_order",
        "joint_scale_opt",
        "gptq",
    ),
) -> PipelineSpec:
    """Return a declarative view of the current production pipeline."""

    artifacts = (
        ArtifactSpec(
            "source_model",
            "hf_checkpoint",
            description="Source HF checkpoint",
            provided=True,
        ),
        ArtifactSpec("model_graph", "model_structure_graph"),
        ArtifactSpec("calibration_batch", "calibration_rows", provided=True),
        ArtifactSpec("probe_stats", "probe_payload"),
        ArtifactSpec("quant_costs", "cost_payload"),
        ArtifactSpec("layer_assignment", "layer_config"),
        ArtifactSpec("production_weight_cache", "production_weight_cache"),
        ArtifactSpec(
            "resident_production_weight_cache",
            "production_weight_cache",
            resident=True,
        ),
        ArtifactSpec("kl_metrics", "validation_metrics"),
        ArtifactSpec("compressed_artifact", "hf_checkpoint"),
        ArtifactSpec("vllm_smoke", "validation_metrics"),
        ArtifactSpec("render.baseline_weight", "tensor", provided=True),
        ArtifactSpec("render.reference_weight", "tensor", provided=True),
        ArtifactSpec("render.activation_rows", "tensor", provided=True),
    )
    gates = (
        MetricGateSpec(
            name="gate.render.output_mse",
            metric="output_mse",
            mode="per_item",
            direction="lower_is_better",
        ),
        MetricGateSpec(
            name="gate.render.fisher_output_mse",
            metric="fisher_output_mse",
            mode="per_item",
            direction="lower_is_better",
        ),
        MetricGateSpec(
            name="gate.validation.end_kl",
            metric="end_kl",
            mode="all",
            direction="lower_is_better",
        ),
    )
    stages = [
        PipelineStageSpec(
            name="model.structure_graph",
            component="model_profiles.structure:build_model_graph",
            inputs=("source_model",),
            outputs=("model_graph",),
            tags=("model_structure",),
        ),
        PipelineStageSpec(
            name="probe.sensitivity",
            component="incremental_probe",
            inputs=("source_model", "model_graph", "calibration_batch"),
            outputs=("probe_stats",),
            resources=(ResourceContract(
                resource="streaming_model_weights",
                owner="StreamingModelPrefetch",
                residency="required",
            ),),
            tags=("probe", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="measure.quant_cost",
            component="incremental_measure_quant_cost",
            inputs=("source_model", "model_graph", "probe_stats"),
            outputs=("quant_costs",),
            resources=(ResourceContract(
                resource="streaming_model_weights",
                owner="StreamingModelPrefetch",
                residency="required",
            ),),
            tags=("cost", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="allocate.assignment",
            component="allocator",
            inputs=("model_graph", "probe_stats", "quant_costs"),
            outputs=("layer_assignment",),
            tags=("allocator", "cpu_solver"),
        ),
        PipelineStageSpec(
            name="cache.fill_production_weights",
            component="production_weight_cache:fill_production_weight_cache",
            inputs=(
                "source_model",
                "model_graph",
                "calibration_batch",
                "layer_assignment",
            ),
            outputs=("production_weight_cache",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="optional",
            ),),
            tags=("cache", "render", "gpu_bound"),
        ),
        *render_mechanism_stage_specs(render_mechanisms),
        PipelineStageSpec(
            name="cache.prefetch_assignment",
            component="ProductionWeightCache.prefetch_assignment",
            inputs=("production_weight_cache", "layer_assignment"),
            outputs=("resident_production_weight_cache",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="required",
            ),),
            tags=("cache", "prefetch", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="validate.kl",
            component="validate_assignments_kl",
            inputs=(
                "source_model",
                "calibration_batch",
                "layer_assignment",
                "resident_production_weight_cache",
            ),
            outputs=("kl_metrics",),
            gates=("gate.validation.end_kl",),
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
                ),
            ),
            tags=("validation", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="export.native_compressed",
            component="export_native_compressed",
            inputs=(
                "source_model",
                "layer_assignment",
                "resident_production_weight_cache",
            ),
            outputs=("compressed_artifact",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="required",
            ),),
            tags=("export", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="validate.vllm_smoke",
            component="validation_harness:vllm_smoke",
            inputs=("compressed_artifact",),
            outputs=("vllm_smoke",),
            tags=("vllm", "validation", "gpu_bound"),
        ),
    ]
    return PipelineSpec(
        id="prismaquant.production.v1",
        artifacts=artifacts,
        gates=gates,
        stages=tuple(stages),
        description="Current production flow expressed as typed contracts.",
    )


def _metric_values(
    payload: Mapping[str, Any] | float | int,
    metric: str,
) -> dict[str, float]:
    if isinstance(payload, (float, int)):
        return {"__global__": float(payload)}
    out: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            if metric not in value:
                continue
            raw = value[metric]
        elif key == metric:
            raw = value
            key = "__global__"
        else:
            raw = value
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _duplicates(label: str, values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return [f"duplicate {label}: {value}" for value in sorted(dupes)]


def _split_csv(values: str | Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values: Iterable[str] = values.split(",")
    else:
        raw_values = values
    out: list[str] = []
    for raw in raw_values:
        for value in str(raw).split(","):
            name = value.strip()
            if name and name not in out:
                out.append(name)
    return tuple(out)


def _register_builtins() -> None:
    for stage in default_production_pipeline_spec(render_mechanisms=()).stages:
        register_pipeline_stage(stage)


def _register_builtin_components() -> None:
    # Research components live in archive until explicitly revived.  The
    # component registry remains available for programmatic opt-in specs, but
    # production imports do not load shelved cross-layer methods.
    return


def _ensure_builtins_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _register_builtins()
    _BUILTINS_REGISTERED = True


def _ensure_components_registered() -> None:
    global _BUILTIN_COMPONENTS_REGISTERED
    if _BUILTIN_COMPONENTS_REGISTERED:
        return
    _register_builtin_components()
    _BUILTIN_COMPONENTS_REGISTERED = True


def _resolve_pipeline_component(
    component: str | PipelineComponentSpec,
) -> PipelineComponentSpec:
    if not isinstance(component, str) and hasattr(component, "id"):
        return component
    return pipeline_component(str(component))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write or validate PrismaQuant pipeline contracts."
    )
    ap.add_argument(
        "--write-default-production",
        metavar="PATH",
        help="Write the configured production PipelineSpec JSON to PATH.",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated or loaded PipelineSpec and fail on errors.",
    )
    ap.add_argument(
        "--input",
        metavar="PATH",
        help="Validate an existing PipelineSpec JSON instead of generating one.",
    )
    ap.add_argument("--render-mechanisms", default="")
    ap.add_argument("--disable-render-mechanisms", default="")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--formats", default=None)
    ap.add_argument("--target-bits", type=float, default=None)
    ap.add_argument("--target-profile", default=None)
    ap.add_argument("--calibration-modality", default=None)
    ap.add_argument("--selection-mode", default=None)
    ap.add_argument("--production-cache", default=None)
    ap.add_argument("--production-recache", default=None)
    ap.add_argument(
        "--include-component",
        action="append",
        default=[],
        help="Opt-in pipeline component id to compose into the contract.",
    )
    ap.add_argument(
        "--list-components",
        action="store_true",
        help="List registered opt-in pipeline components and exit.",
    )
    args = ap.parse_args(argv)

    if args.list_components:
        for component in registered_pipeline_components().values():
            print(
                f"{component.id}\t{component.status}\t"
                f"default_enabled={int(component.default_enabled)}"
            )
        return 0

    if args.input:
        spec = load_pipeline_spec(args.input)
    else:
        spec = production_pipeline_spec_from_config(
            render_mechanisms=args.render_mechanisms,
            disabled_render_mechanisms=args.disable_render_mechanisms,
            model_path=args.model_path,
            work_dir=args.work_dir,
            formats=args.formats,
            target_bits=args.target_bits,
            target_profile=args.target_profile,
            calibration_modality=args.calibration_modality,
            selection_mode=args.selection_mode,
            production_cache=args.production_cache,
            production_recache=args.production_recache,
            components=args.include_component,
        )

    validation = spec.validate()
    if args.validate and validation.errors:
        for error in validation.errors:
            print(f"[pipeline-spec] ERROR: {error}")
        return 2
    for warning in validation.warnings:
        print(f"[pipeline-spec] WARN: {warning}")

    if args.write_default_production:
        write_pipeline_spec(spec, args.write_default_production)
        print(f"[pipeline-spec] wrote {args.write_default_production}")
    elif not args.input:
        print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
    elif validation.ok:
        print(f"[pipeline-spec] valid: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
