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
from typing import Any, Mapping

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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServingFormatRule":
        return cls(
            id=str(payload["id"]),
            when=NameCondition.from_dict(payload.get("when") or {}),
            allow_formats=tuple(str(v) for v in payload.get("allow_formats", ())),
            deny_formats=tuple(str(v) for v in payload.get("deny_formats", ())),
            reason=str(payload.get("reason", "profile_mismatch")),
            detail=str(payload.get("detail", "")),
        )

    def check(self, qname: str, fmt: str) -> ServingFormatDecision | None:
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
        )

    def check_format(self, qname: str | None, fmt: str) -> ServingFormatDecision:
        name = qname or ""
        for rule in self.format_rules:
            decision = rule.check(name, fmt)
            if decision is not None and not decision.legal:
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


def check_serving_format(
    profile_id: str | None,
    qname: str | None,
    fmt: str,
) -> ServingFormatDecision:
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        return ServingFormatDecision(
            False,
            "profile_mismatch",
            f"unknown target profile {profile_id!r}",
        )
    return profile.check_format(qname, fmt)


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


def _format_in(fmt: str, names: tuple[str, ...]) -> bool:
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
