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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NameCondition":
        return cls(
            contains=_optional_str(payload.get("contains")),
            not_contains=_optional_str(payload.get("not_contains")),
            prefix=_optional_str(payload.get("prefix")),
            regex=_optional_str(payload.get("regex")),
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
        in_features: int,
        out_features: int,
    ) -> ServingFormatDecision | None:
        if not _format_in(fmt, self.formats):
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
        return ServingFormatDecision(False, self.reason, detail, self.id)


@dataclass(frozen=True)
class ServingProfile:
    id: str
    runtime: str = ""
    extends: tuple[str, ...] = ()
    format_rules: tuple[ServingFormatRule, ...] = ()
    shape_rules: tuple[ShapeRule, ...] = ()
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
            description=str(payload.get("description", "")),
        )

    def check_format(self, qname: str | None, fmt: str) -> ServingFormatDecision:
        name = qname or ""
        for rule in self.format_rules:
            decision = rule.check(name, fmt)
            if decision is not None and not decision.legal:
                return decision
        return ServingFormatDecision(True)

    def check_shape(
        self,
        fmt: str,
        *,
        in_features: int,
        out_features: int,
    ) -> ServingFormatDecision:
        for rule in self.shape_rules:
            decision = rule.check(
                fmt,
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
            description=profile.description,
        )
    _CACHE[profile_name] = profile
    return profile


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
    in_features: int,
    out_features: int,
) -> ServingFormatDecision:
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        profile = load_serving_profile("research")
    return profile.check_shape(
        fmt,
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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
