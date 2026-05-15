"""ReSpinQuant compatibility checks for PrismaQuant.

ReSpinQuant's useful idea is layer-wise residual-basis rotation: use a
different orthogonal basis around different transformer layers, then repair
the residual-stream basis mismatch with a small transition operator. That is
not the same deployment contract as HALO. HALO is kernel-free because one
global residual basis makes every residual identity path remain an identity.

PrismaQuant's production artifacts must load in vanilla vLLM with ordinary
compressed-tensors metadata. This module therefore implements the safety
boundary first: it can describe and gate layer-wise ReSpin-style basis plans,
and it refuses plans that require a runtime residual adapter unless the caller
explicitly asks for research-only behavior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence


IDENTITY_BASIS = "identity"
GLOBAL_BASIS = "global"


class ReSpinRuntimeAdapterRequired(RuntimeError):
    """Raised when a ReSpin-style plan cannot be folded into vanilla vLLM."""


@dataclass(frozen=True)
class ReSpinLayerBasis:
    """Residual-stream basis entering and leaving one transformer layer."""

    layer_index: int
    input_basis: str = IDENTITY_BASIS
    output_basis: str = IDENTITY_BASIS
    enabled: bool = False

    @property
    def layer_transition_required(self) -> bool:
        return self.input_basis != self.output_basis


@dataclass(frozen=True)
class ReSpinTransition:
    """A residual-basis transition that would need runtime work."""

    kind: str
    layer_index: int
    from_basis: str
    to_basis: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReSpinPlanAnalysis:
    """Kernel-free feasibility result for a ReSpin-style basis plan."""

    layers: tuple[ReSpinLayerBasis, ...]
    transitions: tuple[ReSpinTransition, ...]
    kernel_free: bool
    equivalent: str
    reason: str

    @property
    def requires_runtime_adapter(self) -> bool:
        return bool(self.transitions)

    def require_kernel_free(self) -> "ReSpinPlanAnalysis":
        if self.requires_runtime_adapter:
            raise ReSpinRuntimeAdapterRequired(self.reason)
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_free": bool(self.kernel_free),
            "requires_runtime_adapter": bool(self.requires_runtime_adapter),
            "equivalent": self.equivalent,
            "reason": self.reason,
            "layers": [asdict(layer) for layer in self.layers],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _basis(value: str | None) -> str:
    text = (value or IDENTITY_BASIS).strip()
    return text or IDENTITY_BASIS


def _parse_layer_set(enabled_layers: Iterable[int] | str | None,
                     n_layers: int) -> set[int]:
    if enabled_layers is None:
        return set()
    if isinstance(enabled_layers, str):
        raw = enabled_layers.strip().lower()
        if raw in {"", "none", "off", "false", "0"}:
            return set()
        if raw == "all":
            return set(range(n_layers))
        values: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo = int(lo_s)
                hi = int(hi_s)
                values.update(range(lo, hi + 1))
            else:
                values.add(int(part))
        return {idx for idx in values if 0 <= idx < n_layers}
    return {int(idx) for idx in enabled_layers if 0 <= int(idx) < n_layers}


def analyze_residual_basis_plan(
    layers: Sequence[ReSpinLayerBasis],
) -> ReSpinPlanAnalysis:
    """Return whether a residual-basis plan is kernel-free.

    A vanilla Transformer residual path contains identity additions. If a
    layer changes residual basis, the identity residual branch would need a
    transition matrix ``T = R_out R_in^T``. Likewise, adjacent layers whose
    output/input bases differ need an inter-layer transition. Those
    transitions are runtime work unless every connected residual segment uses
    the same basis, reducing the plan to identity or a HALO-like global arm.
    """

    ordered = tuple(sorted(layers, key=lambda item: item.layer_index))
    transitions: list[ReSpinTransition] = []
    for layer in ordered:
        in_basis = _basis(layer.input_basis)
        out_basis = _basis(layer.output_basis)
        if in_basis != out_basis:
            transitions.append(ReSpinTransition(
                kind="within_residual_layer",
                layer_index=layer.layer_index,
                from_basis=in_basis,
                to_basis=out_basis,
            ))
    for prev, nxt in zip(ordered, ordered[1:]):
        prev_out = _basis(prev.output_basis)
        next_in = _basis(nxt.input_basis)
        if prev_out != next_in:
            transitions.append(ReSpinTransition(
                kind="between_layers",
                layer_index=nxt.layer_index,
                from_basis=prev_out,
                to_basis=next_in,
            ))

    kernel_free = not transitions
    unique_bases = {
        _basis(layer.input_basis) for layer in ordered
    } | {
        _basis(layer.output_basis) for layer in ordered
    }
    if not ordered or unique_bases == {IDENTITY_BASIS}:
        equivalent = "identity"
    elif kernel_free and len(unique_bases) == 1:
        equivalent = "global_rotation"
    elif kernel_free:
        equivalent = "disconnected_segments"
    else:
        equivalent = "runtime_residual_adapter"

    if kernel_free:
        reason = (
            "No residual-basis transitions are required; this plan can be "
            "represented by an identity or global residual-basis arm."
        )
    else:
        reason = (
            "Layer-wise ReSpinQuant changes residual bases. Vanilla vLLM has "
            "no residual transition operator, so this plan needs runtime "
            "adapter work and is not a production-safe compressed-tensors "
            "export."
        )
    return ReSpinPlanAnalysis(
        layers=ordered,
        transitions=tuple(transitions),
        kernel_free=kernel_free,
        equivalent=equivalent,
        reason=reason,
    )


def make_layerwise_respin_plan(
    n_layers: int,
    enabled_layers: Iterable[int] | str | None = "all",
    *,
    input_basis: str = IDENTITY_BASIS,
    rotation_prefix: str = "R",
) -> ReSpinPlanAnalysis:
    """Build a ReSpin-style layer-wise plan and analyze its exportability."""

    if n_layers < 0:
        raise ValueError(f"n_layers must be non-negative, got {n_layers}")
    enabled = _parse_layer_set(enabled_layers, n_layers)
    layers: list[ReSpinLayerBasis] = []
    current_basis = _basis(input_basis)
    for idx in range(n_layers):
        if idx in enabled:
            out_basis = f"{rotation_prefix}{idx}"
            layers.append(ReSpinLayerBasis(
                layer_index=idx,
                input_basis=current_basis,
                output_basis=out_basis,
                enabled=True,
            ))
            current_basis = out_basis
        else:
            layers.append(ReSpinLayerBasis(
                layer_index=idx,
                input_basis=current_basis,
                output_basis=current_basis,
                enabled=False,
            ))
    return analyze_residual_basis_plan(layers)


def make_global_rotation_plan(n_layers: int, *,
                              basis: str = GLOBAL_BASIS) -> ReSpinPlanAnalysis:
    """Return the kernel-free global-basis plan equivalent to HALO/QuaRot."""

    if n_layers < 0:
        raise ValueError(f"n_layers must be non-negative, got {n_layers}")
    basis = _basis(basis)
    return analyze_residual_basis_plan(tuple(
        ReSpinLayerBasis(
            layer_index=idx,
            input_basis=basis,
            output_basis=basis,
            enabled=True,
        )
        for idx in range(n_layers)
    ))


def assert_kernel_free(plan: ReSpinPlanAnalysis, *,
                       allow_runtime_adapter: bool = False) -> ReSpinPlanAnalysis:
    """Fail fast unless a ReSpin plan preserves the vanilla vLLM graph."""

    if not allow_runtime_adapter:
        plan.require_kernel_free()
    return plan


def hidden_layers_from_config(model_or_config: str | Path | dict) -> int:
    """Extract decoder layer count from common HuggingFace config layouts."""

    if isinstance(model_or_config, (str, Path)):
        path = Path(model_or_config)
        config_path = path / "config.json" if path.is_dir() else path
        data = json.loads(config_path.read_text())
    else:
        data = dict(model_or_config)

    candidates = (
        data.get("num_hidden_layers"),
        data.get("num_layers"),
        (data.get("text_config") or {}).get("num_hidden_layers")
        if isinstance(data.get("text_config"), dict) else None,
        (data.get("language_model_config") or {}).get("num_hidden_layers")
        if isinstance(data.get("language_model_config"), dict) else None,
        (data.get("llm_config") or {}).get("num_hidden_layers")
        if isinstance(data.get("llm_config"), dict) else None,
    )
    for value in candidates:
        if isinstance(value, int) and value >= 0:
            return value
    raise ValueError("could not determine num_hidden_layers from config")

