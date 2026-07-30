"""Config-backed serving/runtime constraint profiles.

Serving profiles capture backend-specific legality that should not live as
architecture branches in the allocator: format menus, kernel shape limits, and
other runtime constraints.  The allocator still performs cheap local checks,
but the policy comes from JSON specs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import import_module
from importlib import resources
from typing import Any, Collection, Mapping

from . import format_registry as fr


SCHEMA = "prismaquant.serving_profile.v1"


@dataclass(frozen=True)
class ServingFormatDecision:
    legal: bool
    reason: str | None = None
    detail: str = ""
    rule: str | None = None


@dataclass(frozen=True)
class NameCondition:
    contains: str | None = None
    not_contains: str | None = None
    prefix: str | None = None
    regex: str | None = None
    not_regex: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NameCondition":
        return cls(
            contains=_optional_str(payload.get("contains")),
            not_contains=_optional_str(payload.get("not_contains")),
            prefix=_optional_str(payload.get("prefix")),
            regex=_optional_str(payload.get("regex")),
            not_regex=_optional_str(payload.get("not_regex")),
        )

    def matches(self, name: str) -> bool:
        if self.contains is not None and self.contains not in name:
            return False
        if self.not_contains is not None and self.not_contains in name:
            return False
        if self.prefix is not None and not name.startswith(self.prefix):
            return False
        if self.regex is not None and re.search(self.regex, name) is None:
            return False
        if self.not_regex is not None and re.search(self.not_regex, name) is not None:
            return False
        return True


@dataclass(frozen=True)
class ServingFormatRule:
    id: str
    when: NameCondition = field(default_factory=NameCondition)
    allow_formats: tuple[str, ...] = ()
    deny_formats: tuple[str, ...] = ()
    reason: str = "profile_mismatch"
    detail: str = ""
    # Target-class scoping: "all" (default), "packed_experts" (rank-3 stacked
    # MoE tensors only), or "dense" (everything else). Lets a container
    # declare capabilities that differ between dense Linears and packed
    # expert stacks (e.g. nvfp4_cb carries no stock-CT packed-MoE emission).
    scope: str = "all"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServingFormatRule":
        scope = str(payload.get("scope", "all"))
        if scope not in ("all", "packed_experts", "dense"):
            raise ValueError(
                f"format rule {payload.get('id')!r}: unknown scope {scope!r} "
                f"(expected all|packed_experts|dense)")
        return cls(
            id=str(payload["id"]),
            when=NameCondition.from_dict(payload.get("when") or {}),
            allow_formats=tuple(str(v) for v in payload.get("allow_formats", ())),
            deny_formats=tuple(str(v) for v in payload.get("deny_formats", ())),
            reason=str(payload.get("reason", "profile_mismatch")),
            detail=str(payload.get("detail", "")),
            scope=scope,
        )

    def check(self, qname: str, fmt: str,
              packed_expert: bool | None = None) -> ServingFormatDecision | None:
        if self.scope == "packed_experts" and packed_expert is not True:
            return None
        if self.scope == "dense" and packed_expert is True:
            return None
        if not self.when.matches(qname):
            return None
        if self.allow_formats and not _format_in(fmt, self.allow_formats):
            return ServingFormatDecision(
                False,
                self.reason,
                self.detail,
                self.id,
            )
        if self.deny_formats and _format_in(fmt, self.deny_formats):
            return ServingFormatDecision(
                False,
                self.reason,
                self.detail,
                self.id,
            )
        return ServingFormatDecision(True, rule=self.id)


@dataclass(frozen=True)
class ShapeRule:
    id: str
    formats: tuple[str, ...]
    when: NameCondition = field(default_factory=NameCondition)
    min_in_features: int | None = None
    min_out_features: int | None = None
    in_features_multiple_of: int | None = None
    out_features_multiple_of: int | None = None
    reason: str = "kernel_shape"
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShapeRule":
        return cls(
            id=str(payload["id"]),
            formats=tuple(str(v) for v in payload.get("formats", ())),
            when=NameCondition.from_dict(payload.get("when") or {}),
            min_in_features=_optional_int(payload.get("min_in_features")),
            min_out_features=_optional_int(payload.get("min_out_features")),
            in_features_multiple_of=_optional_int(
                payload.get("in_features_multiple_of")
            ),
            out_features_multiple_of=_optional_int(
                payload.get("out_features_multiple_of")
            ),
            reason=str(payload.get("reason", "kernel_shape")),
            detail=str(payload.get("detail", "")),
        )

    def check(
        self,
        fmt: str,
        *,
        qname: str | None = None,
        in_features: int,
        out_features: int,
    ) -> ServingFormatDecision | None:
        if not _format_in(fmt, self.formats):
            return None
        if not self.when.matches(qname or ""):
            return None
        legal = True
        if self.min_in_features is not None and in_features < self.min_in_features:
            legal = False
        if self.min_out_features is not None and out_features < self.min_out_features:
            legal = False
        if (
            self.in_features_multiple_of is not None
            and in_features % self.in_features_multiple_of != 0
        ):
            legal = False
        if (
            self.out_features_multiple_of is not None
            and out_features % self.out_features_multiple_of != 0
        ):
            legal = False
        if legal:
            return ServingFormatDecision(True, rule=self.id)
        detail = self.detail or (
            f"{fmt} kernel does not support "
            f"(out_features={out_features}, in_features={in_features})"
        )
        if self.detail:
            detail = (
                f"{detail} "
                f"(out_features={out_features}, in_features={in_features})"
            )
        return ServingFormatDecision(False, self.reason, detail, self.id)


@dataclass(frozen=True)
class RuntimeShapeValidatorRule:
    id: str
    formats: tuple[str, ...]
    when: NameCondition = field(default_factory=NameCondition)
    callable_path: str | None = None
    optional: bool = True
    reason: str = "kernel_shape"
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeShapeValidatorRule":
        return cls(
            id=str(payload["id"]),
            formats=tuple(str(v) for v in payload.get("formats", ())),
            when=NameCondition.from_dict(payload.get("when") or {}),
            callable_path=_optional_str(
                payload.get("callable") or payload.get("callable_path")
            ),
            optional=bool(payload.get("optional", True)),
            reason=str(payload.get("reason", "kernel_shape")),
            detail=str(payload.get("detail", "")),
        )

    def check(
        self,
        fmt: str,
        *,
        qname: str | None = None,
        in_features: int,
        out_features: int,
    ) -> ServingFormatDecision | None:
        if not _format_in(fmt, self.formats):
            return None
        if not self.when.matches(qname or ""):
            return None
        verdict = _runtime_shape_validator_accepts(
            self.id,
            fmt,
            in_features=in_features,
            out_features=out_features,
            callable_path=self.callable_path,
        )
        if verdict is None:
            if self.optional:
                return ServingFormatDecision(True, rule=self.id)
            return ServingFormatDecision(
                False,
                self.reason,
                self.detail or f"runtime shape validator {self.id!r} unavailable",
                self.id,
            )
        if verdict:
            return ServingFormatDecision(True, rule=self.id)
        detail = self.detail or (
            f"{fmt} runtime validator {self.id} rejected "
            f"(out_features={out_features}, in_features={in_features})"
        )
        return ServingFormatDecision(False, self.reason, detail, self.id)


@dataclass(frozen=True)
class ExportLaneSpec:
    """The artifact container a serving lane ships through, plus the
    *exporter's own declaration* of what it can emit.

    A serving profile's format menu and its lane's exporter must not be
    able to disagree: a rung the exporter cannot emit is not "denied by
    policy", it is structurally unavailable, and the allocator must never
    be able to spend a bit budget on it (a recent bit-exact re-encode
    short-circuit prices weight-lossless A16 rungs at dloss 0.0 — the
    unbeatable global minimum — so an unexportable-but-legal rung is
    actively attractive to the DP).

    ``codec_formats_from`` is a tuple of ``module:ATTR`` paths whose
    attribute is an *iterable of format names the exporter itself
    declares it can emit* (a dict's keys or a set both count).  Nothing is
    duplicated here: the vLLM lane points at
    ``export_native_compressed.EXPORTABLE_FORMATS`` (that exporter's own
    declaration — its ``FORMAT_SCHEME`` metadata table, CLAUDE.md gate
    #9's "correctly represented in compressed-tensors metadata", already
    unioned with its container passthroughs), and the GGUF lane points at
    ``gguf_formats.GGUF_BLOCK_BYTES`` (the ggml type table
    ``export_gguf``/``export_gguf_direct`` gate on directly).

    ``passthrough_formats`` covers formats a container emits *without* a
    codec entry, for lanes whose declaration is a bare codec table that
    cannot contain them: BF16 is written as plain container floats
    (safetensors bf16 / GGUF F16-F32) and goes on the checkpoint's
    ``ignore`` list rather than into ``config_groups``.  It stays per-lane
    because passthrough is a container fact — FP8_SOURCE is a
    verbatim-copy passthrough on the compressed-tensors lane but has no
    ggml type at all — and it is empty for a lane like
    ``compressed_tensors`` whose exporter folds its own passthroughs into
    the constant it declares.
    """

    id: str
    exporter: str = ""
    codec_formats_from: tuple[str, ...] = ()
    passthrough_formats: tuple[str, ...] = ()
    reason: str = "exporter_cannot_emit"
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExportLaneSpec":
        return cls(
            id=str(payload["id"]),
            exporter=str(payload.get("exporter", "")),
            codec_formats_from=tuple(
                str(v) for v in payload.get("codec_formats_from", ())
            ),
            passthrough_formats=tuple(
                str(v) for v in payload.get("passthrough_formats", ())
            ),
            reason=str(payload.get("reason", "exporter_cannot_emit")),
            detail=str(payload.get("detail", "")),
        )

    def emittable_formats(self) -> frozenset[str]:
        """Canonical format names this lane's exporter can emit."""
        cached = _EMITTABLE_CACHE.get(self)
        if cached is None:
            names: set[str] = set()
            for path in self.codec_formats_from:
                names |= _declared_exporter_formats(path, self.id)
            if not names:
                raise RuntimeError(
                    f"serving profile export lane {self.id!r} declares no "
                    f"emittable formats (codec_formats_from="
                    f"{list(self.codec_formats_from)!r}). A lane with an "
                    f"empty menu would deny every format; declare the "
                    f"exporter's own format table instead."
                )
            names |= {
                fr.canonical_format_name(name)
                for name in self.passthrough_formats
            }
            cached = frozenset(names)
            _EMITTABLE_CACHE[self] = cached
        return cached

    def check(self, fmt: str) -> ServingFormatDecision:
        emittable = self.emittable_formats()
        if _format_in(fmt, emittable):
            return ServingFormatDecision(True, rule=self.id)
        detail = self.detail or (
            f"{fr.canonical_format_name(fmt)} has no emit path in this "
            f"lane's exporter"
        )
        return ServingFormatDecision(
            False,
            self.reason,
            f"{detail} (lane={self.id}"
            + (f", exporter={self.exporter}" if self.exporter else "")
            + f", emittable={sorted(emittable)})",
            self.id,
        )


@dataclass(frozen=True)
class RuntimePackageSpec:
    id: str
    module: str | None = None
    version: str | None = None
    pip_packages: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    optional: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePackageSpec":
        return cls(
            id=str(payload["id"]),
            module=_optional_str(payload.get("module")),
            version=_optional_str(payload.get("version")),
            pip_packages=tuple(str(v) for v in payload.get("pip_packages", ())),
            env=tuple(
                (str(key), str(value))
                for key, value in (payload.get("env") or {}).items()
            ),
            optional=bool(payload.get("optional", True)),
        )

    def env_dict(self) -> dict[str, str]:
        return dict(self.env)


@dataclass(frozen=True)
class ServingProfile:
    id: str
    runtime: str = ""
    extends: tuple[str, ...] = ()
    format_rules: tuple[ServingFormatRule, ...] = ()
    shape_rules: tuple[ShapeRule, ...] = ()
    runtime_shape_validators: tuple[RuntimeShapeValidatorRule, ...] = ()
    runtime_packages: tuple[RuntimePackageSpec, ...] = ()
    description: str = ""
    # The artifact container this profile ships through. Bounds the format
    # menu by what the lane's exporter declares it can emit (see
    # ExportLaneSpec). Inherited from `extends` when not declared locally.
    export_lane: ExportLaneSpec | None = None
    # Declared exemption from the export-lane bound: this profile
    # constrains *emulation / kernel* legality only and does not
    # correspond to an artifact container, so no exporter bounds its
    # menu. Deliberately true for `research`, which exists so research
    # rungs with no served path (MXFP6, INT4_W4A16_g128, the A16 family)
    # stay measurable. False is the fail-closed default: a new serving
    # profile must name its export lane or declare itself emulation-only.
    emulation_only: bool = False
    # Capability: the serving lane can load DIFFERENT expert schemes for
    # different projection roles of one MoE layer (gate/up vs down). True
    # for GGUF (expert tensors are stacked PER projection, so each stacked
    # tensor carries its own ggml type); false for vLLM compressed-tensors
    # packed MoE, where CompressedTensorsMoEMethod selects ONE scheme per
    # FusedMoE layer. Gates --packed-role-split.
    supports_per_role_expert_schemes: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServingProfile":
        schema = str(payload.get("schema", SCHEMA))
        if schema != SCHEMA:
            raise ValueError(f"unsupported serving-profile schema: {schema!r}")
        return cls(
            id=str(payload["id"]),
            runtime=str(payload.get("runtime", "")),
            extends=tuple(str(v) for v in payload.get("extends", ())),
            format_rules=tuple(
                ServingFormatRule.from_dict(entry)
                for entry in payload.get("format_rules", ())
            ),
            shape_rules=tuple(
                ShapeRule.from_dict(entry)
                for entry in payload.get("shape_rules", ())
            ),
            runtime_shape_validators=tuple(
                RuntimeShapeValidatorRule.from_dict(entry)
                for entry in payload.get("runtime_shape_validators", ())
            ),
            runtime_packages=tuple(
                RuntimePackageSpec.from_dict(entry)
                for entry in payload.get("runtime_packages", ())
            ),
            description=str(payload.get("description", "")),
            export_lane=(
                ExportLaneSpec.from_dict(payload["export_lane"])
                if payload.get("export_lane")
                else None
            ),
            emulation_only=bool(payload.get("emulation_only", False)),
            supports_per_role_expert_schemes=bool(
                payload.get("supports_per_role_expert_schemes", False)
            ),
        )

    def check_format(self, qname: str | None, fmt: str,
                     packed_expert: bool | None = None
                     ) -> ServingFormatDecision:
        name = qname or ""
        for rule in self.format_rules:
            decision = rule.check(name, fmt, packed_expert=packed_expert)
            if decision is not None and not decision.legal:
                return decision
        # Structural bound, applied after the profile's own policy rules so
        # an explicitly-denied format keeps its policy attribution: the
        # lane's exporter has no emit path for this format, so no allow/deny
        # list may admit it. Fixing the menu disagreement at the root means
        # the profile *cannot* widen past the exporter, rather than a
        # hand-maintained deny list mirroring the exporter's branches.
        if self.export_lane is not None:
            decision = self.export_lane.check(fmt)
            if not decision.legal:
                return decision
        return ServingFormatDecision(True)

    def runtime_package(self, package_id: str) -> RuntimePackageSpec | None:
        for package in reversed(self.runtime_packages):
            if package.id == package_id:
                return package
        return None

    def check_shape(
        self,
        fmt: str,
        *,
        qname: str | None = None,
        in_features: int,
        out_features: int,
    ) -> ServingFormatDecision:
        for rule in self.runtime_shape_validators:
            decision = rule.check(
                fmt,
                qname=qname,
                in_features=in_features,
                out_features=out_features,
            )
            if decision is not None and not decision.legal:
                return decision
        for rule in self.shape_rules:
            decision = rule.check(
                fmt,
                qname=qname,
                in_features=in_features,
                out_features=out_features,
            )
            if decision is not None and not decision.legal:
                return decision
        return ServingFormatDecision(True)


_CACHE: dict[str, ServingProfile] = {}
_EMITTABLE_CACHE: dict["ExportLaneSpec", frozenset[str]] = {}


def _declared_exporter_formats(path: str, lane_id: str) -> set[str]:
    """Read an exporter's own format declaration at ``module:ATTR``.

    Imported lazily and cached by the caller: the compressed-tensors
    exporter imports this module, so a module-scope import would be
    circular, and the GGUF codec tables pull torch.
    """
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except ImportError as exc:  # pragma: no cover - environment breakage
        raise RuntimeError(
            f"serving profile export lane {lane_id!r} declares "
            f"{path!r} but {module_name!r} could not be imported "
            f"({exc!r}); the lane's format menu cannot be bounded by its "
            f"exporter."
        ) from exc
    try:
        declared = getattr(module, attr_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"serving profile export lane {lane_id!r} declares "
            f"{path!r} but {module_name!r} has no attribute "
            f"{attr_name!r}. The lane's emittable-format menu is derived "
            f"from the exporter's own declaration — update the profile "
            f"spec to the declaration's new name rather than hand-listing "
            f"formats here."
        ) from exc
    try:
        names = [str(name) for name in declared]
    except TypeError as exc:
        raise RuntimeError(
            f"serving profile export lane {lane_id!r}: {path!r} is not "
            f"iterable ({type(declared).__name__}); expected a container of "
            f"format names (a dict keyed by format name counts)."
        ) from exc
    return {fr.canonical_format_name(name) for name in names}


def serving_profile_names() -> tuple[str, ...]:
    root = resources.files("prismaquant").joinpath("serving_profile_specs")
    names: list[str] = []
    try:
        for resource in root.iterdir():
            if resource.name.endswith(".json"):
                names.append(resource.name[:-5])
    except FileNotFoundError:
        pass
    return tuple(sorted(set(names) | {"research"}))


def load_serving_profile(profile_id: str | None) -> ServingProfile:
    profile_name = str(profile_id or "research")
    if profile_name in _CACHE:
        return _CACHE[profile_name]
    profile = _load_serving_profile_uncached(profile_name)
    if profile.extends:
        bases = tuple(load_serving_profile(base) for base in profile.extends)
        profile = ServingProfile(
            id=profile.id,
            runtime=profile.runtime,
            extends=profile.extends,
            format_rules=tuple(
                rule
                for base in bases
                for rule in base.format_rules
            ) + profile.format_rules,
            shape_rules=tuple(
                rule
                for base in bases
                for rule in base.shape_rules
            ) + profile.shape_rules,
            runtime_shape_validators=tuple(
                rule
                for base in bases
                for rule in base.runtime_shape_validators
            ) + profile.runtime_shape_validators,
            runtime_packages=tuple(
                package
                for base in bases
                for package in base.runtime_packages
            ) + profile.runtime_packages,
            description=profile.description,
            export_lane=(
                profile.export_lane
                or next(
                    (base.export_lane for base in bases if base.export_lane),
                    None,
                )
            ),
            # Emulation-only is NOT inherited: a lane that extends the
            # research profile's kernel-shape rules is still a shipping
            # lane and must declare its exporter.
            emulation_only=profile.emulation_only,
            supports_per_role_expert_schemes=(
                profile.supports_per_role_expert_schemes
                or any(
                    base.supports_per_role_expert_schemes for base in bases
                )
            ),
        )
    _CACHE[profile_name] = profile
    return profile


def resolve_target_profile(
    profile=None,
    requested: str | None = None,
    *,
    default: str = "research",
) -> str:
    """Resolve the serving/backend constraint profile for a run.

    Explicit CLI/API input wins. Otherwise a model profile may declare its
    default serving profile in the structure spec. The fallback is the
    research profile, which only carries generic kernel-shape rules.
    """
    if requested:
        return str(requested)
    getter = getattr(profile, "serving_profile_id", None)
    if callable(getter):
        try:
            profile_id = getter()
            if profile_id:
                return str(profile_id)
        except Exception:
            pass
    return str(default)


def require_lane_supported(
    profile,
    export_container: str | None,
    *,
    flag: str = "EXPORT_CONTAINER",
):
    """Preflight: refuse an export lane the *architecture* has not declared.

    Lane eligibility is a model-profile property (`supported_export_lanes()`),
    not an operator preference. The CB lane needs a gridbook loader keyed to
    the architecture's expert layout and the GGUF lane needs a llama.cpp-side
    arch; where that wiring is missing, nothing fails. The run completes, the
    exporter writes bytes, and the server loads uninitialised expert memory —
    the observed failure mode is *coherent-looking garbage generation*, not a
    crash (commit `9a79963`, Laguna, 93% of parameters). One quantization
    cycle on a 100 GB-class model is the cost of finding that out downstream,
    so it is refused up front against the declared set.

    Undeclared architectures support the native compressed-tensors lane only,
    which is what all of them have ever shipped through — so this is strictly
    additive: no run that is legal today becomes illegal.

    Returns the canonical lane id.
    """
    from .model_profiles.structure import (
        DEFAULT_EXPORT_LANE,
        canonical_export_lane,
    )

    requested = str(export_container or DEFAULT_EXPORT_LANE)
    try:
        lane = canonical_export_lane(requested)
    except ValueError as exc:
        raise SystemExit(f"[preflight] ERROR: {flag}: {exc}") from None

    getter = getattr(profile, "supported_export_lanes", None)
    if not callable(getter):
        return lane
    supported = tuple(getter())
    if lane in supported:
        return lane

    name = getattr(profile, "name", None) or type(profile).__name__
    preferred = getattr(profile, "preferred_export_lane", None)
    preferred_lane = preferred() if callable(preferred) else DEFAULT_EXPORT_LANE
    raise SystemExit(
        f"[preflight] ERROR: {flag}={lane!r} is not a declared lane for "
        f"architecture {name!r}. Declared: {list(supported)} "
        f"(preferred: {preferred_lane!r}). An undeclared lane does not fail "
        "loudly at serve time — the missing per-architecture loader means the "
        "runtime serves uninitialised weights and generates coherent-looking "
        "garbage. If this architecture really is wired for this lane, declare "
        f"it in model_profiles/specs/{name}.json (`supported_lanes`) together "
        "with the loader wiring that makes it true."
    )


def require_per_role_expert_scheme_support(
    profile_id: str | None,
    *,
    flag: str = "--packed-role-split",
) -> ServingProfile:
    """Hard gate: the resolved serving profile must DECLARE per-role
    expert scheme support before a per-role expert split is legal.

    A gate_up/down role split emits different formats for different
    projections of the SAME MoE layer. That is only loadable when the
    serving lane keys expert schemes per projection — GGUF does (expert
    tensors are stacked per projection, each stacked tensor carries its
    own ggml type). vLLM's compressed-tensors packed-MoE path does not:
    CompressedTensorsMoEMethod selects ONE scheme per FusedMoE layer, so
    a role-split checkpoint (e.g. gate_up=NVFP4 with down=FP8) cannot be
    loaded. Profiles opt in with ``supports_per_role_expert_schemes``.
    """
    resolved = str(profile_id or "research")
    try:
        profile = load_serving_profile(resolved)
    except FileNotFoundError:
        raise SystemExit(
            f"[alloc] ERROR: {flag} was requested, but the target profile "
            f"{resolved!r} is unknown."
        )
    if not profile.supports_per_role_expert_schemes:
        raise SystemExit(
            f"[alloc] ERROR: {flag} was requested, but the resolved "
            f"serving profile {resolved!r} does not declare "
            "supports_per_role_expert_schemes. A per-role split emits "
            "different expert formats for gate_up vs down projections of "
            "the SAME MoE layer; this profile's serving lane loads every "
            "projection of a layer's experts under ONE scheme (vLLM's "
            "CompressedTensorsMoEMethod selects one scheme per FusedMoE "
            "layer), so the checkpoint would be unservable. Use a profile "
            "whose lane keys expert schemes per projection (e.g. "
            "--target-profile gguf), or drop the flag."
        )
    return profile


def check_serving_format(
    profile_id: str | None,
    qname: str | None,
    fmt: str,
    packed_expert: bool | None = None,
) -> ServingFormatDecision:
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        return ServingFormatDecision(
            False,
            "profile_mismatch",
            f"unknown target profile {profile_id!r}",
        )
    return profile.check_format(qname, fmt, packed_expert=packed_expert)


def lane_emittable_formats(profile_id: str | None) -> frozenset[str] | None:
    """Formats the profile's export lane can emit, or None when the
    profile declares no lane (emulation-only, e.g. ``research``)."""
    profile = load_serving_profile(profile_id)
    if profile.export_lane is None:
        return None
    return profile.export_lane.emittable_formats()


def check_serving_shape(
    profile_id: str | None,
    fmt: str,
    *,
    qname: str | None = None,
    in_features: int,
    out_features: int,
) -> ServingFormatDecision:
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        profile = load_serving_profile("research")
    return profile.check_shape(
        fmt,
        qname=qname,
        in_features=in_features,
        out_features=out_features,
    )


def _load_serving_profile_uncached(profile_id: str) -> ServingProfile:
    resource = resources.files("prismaquant").joinpath(
        "serving_profile_specs", f"{profile_id}.json"
    )
    text = resource.read_text(encoding="utf-8")
    return ServingProfile.from_dict(json.loads(text))


def _format_in(fmt: str, names: Collection[str]) -> bool:
    candidates = {fmt, fr.canonical_format_name(fmt), *fr.aliases_for(fmt)}
    return bool(candidates.intersection(names))


def _runtime_shape_validator_accepts(
    validator_id: str,
    fmt: str,
    *,
    in_features: int,
    out_features: int,
    callable_path: str | None = None,
) -> bool | None:
    path = callable_path or _LEGACY_RUNTIME_VALIDATORS.get(validator_id)
    if not path:
        return None
    validator = _load_runtime_validator(path)
    return validator(fmt, in_features=in_features, out_features=out_features)


_LEGACY_RUNTIME_VALIDATORS = {
    "flashinfer_mxfp8_problem_size": (
        "prismaquant.runtime_shape_validators:"
        "flashinfer_mxfp8_problem_size_accepts"
    ),
}


def _load_runtime_validator(callable_path: str):
    if ":" in callable_path:
        module_name, attr_name = callable_path.split(":", 1)
    else:
        module_name, attr_name = callable_path.rsplit(".", 1)
    module = import_module(module_name)
    validator = getattr(module, attr_name)
    if not callable(validator):
        raise TypeError(f"runtime validator {callable_path!r} is not callable")
    return validator


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
