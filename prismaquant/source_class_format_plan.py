"""Exact source-payload format plans for split-menu cost campaigns.

The plan is deliberately independent of routed-expert discovery.  Each
Linear's menu is derived from the source representation recorded by the
checkpoint census and the allocator's existing exact-byte source-rate gate.
The ``expert`` / ``nonexpert`` names are only the campaign CLI's labels for
the two expected source classes; they are never predicates over a qname,
tensor rank, or architecture.

No candidate is silently filtered to make a declared menu fit.  The planner
first derives the complete legal family from the format registry, then
requires that set to equal one of the two declared menus.  A third source
class therefore fails before any render instead of doing illegal work or
truncating a legal candidate set.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    SOURCE_BPP_EXCEEDED_REASON,
    _source_bpp_applicability,
    source_footprint_owner_for_kind,
)
from prismaquant.allocator_solver import _shape_from_stats
from prismaquant.cost_stage_checkpoint import atomic_write_bytes, canonical_json
from prismaquant.nvfp4_cb_footprint import is_cb_format


FORMAT_PLAN_SCHEMA = "prismaquant.source_class_format_plan.v1"
FORMAT_PLAN_SELECTION_RULE = (
    "complete registered family filtered only by exact integer source payload; "
    "derived set must equal one declared menu"
)
EXPERT_MENU = "expert"
NONEXPERT_MENU = "nonexpert"
_MENU_IDS = (EXPERT_MENU, NONEXPERT_MENU)


@dataclass(frozen=True)
class UnitFormatPlan:
    qname: str
    menu_id: str
    source_kind: str
    shape: tuple[int, ...]
    source_payload_bytes: int
    source_bpp_numerator_bits: int
    bpp_denominator_params: int

    @property
    def source_bpp(self) -> float:
        return (
            float(self.source_bpp_numerator_bits)
            / max(int(self.bpp_denominator_params), 1)
        )


@dataclass(frozen=True)
class SourceClassFormatPlan:
    menus: Mapping[str, tuple[str, ...]]
    units: Mapping[str, UnitFormatPlan]
    serving_groups: tuple[tuple[str, ...], ...]
    identity_sha256: str

    def formats_for(self, qname: str) -> tuple[str, ...]:
        try:
            unit = self.units[str(qname)]
        except KeyError as exc:
            raise KeyError(
                f"format plan has no unit {qname!r}; refusing an unplanned "
                "render"
            ) from exc
        return tuple(self.menus[unit.menu_id])

    def menu_id_for(self, qname: str) -> str:
        try:
            return self.units[str(qname)].menu_id
        except KeyError as exc:
            raise KeyError(
                f"format plan has no unit {qname!r}; refusing an unplanned "
                "render"
            ) from exc

    def formats_by_qname(self) -> dict[str, tuple[str, ...]]:
        return {
            qname: self.formats_for(qname)
            for qname in sorted(self.units)
        }

    def qnames_for_menu(self, menu_id: str) -> tuple[str, ...]:
        if menu_id not in self.menus:
            raise KeyError(f"unknown format-plan menu {menu_id!r}")
        return tuple(
            qname
            for qname, unit in sorted(self.units.items())
            if unit.menu_id == menu_id
        )

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": FORMAT_PLAN_SCHEMA,
            "selection_rule": FORMAT_PLAN_SELECTION_RULE,
            "menus": {
                menu_id: list(self.menus[menu_id]) for menu_id in _MENU_IDS
            },
            "units": {
                qname: {
                    "menu_id": unit.menu_id,
                    "source_kind": unit.source_kind,
                    "shape": list(unit.shape),
                    "source_payload_bytes": unit.source_payload_bytes,
                    "source_bpp_numerator_bits": (
                        unit.source_bpp_numerator_bits
                    ),
                    "bpp_denominator_params": unit.bpp_denominator_params,
                }
                for qname, unit in sorted(self.units.items())
            },
            "serving_groups": [
                list(component) for component in self.serving_groups
            ],
        }
        body["identity_sha256"] = _plan_digest(body)
        return body


def _plan_digest(body: Mapping[str, object]) -> str:
    digest_body = {
        str(key): value
        for key, value in body.items()
        if str(key) != "identity_sha256"
    }
    canonical = canonical_json(digest_body, where="source-class format plan")
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_format_menu(raw: str | Sequence[str], *, where: str) -> tuple[str, ...]:
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    canonical: list[str] = []
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        try:
            name = fr.get_format(token).name
        except KeyError as exc:
            raise ValueError(f"{where} contains unknown format {token!r}") from exc
        if name in canonical:
            raise ValueError(
                f"{where} contains duplicate canonical format {name!r}"
            )
        canonical.append(name)
    if not canonical:
        raise ValueError(f"{where} is empty")
    return tuple(canonical)


def _complete_family(
    expert_formats: tuple[str, ...],
    nonexpert_formats: tuple[str, ...],
) -> tuple[str, ...]:
    declared = tuple(dict.fromkeys((*expert_formats, *nonexpert_formats)))
    families = {fr.get_format(name).family for name in declared}
    if len(families) != 1:
        raise ValueError(
            "source-class menus must describe one registered format family; "
            f"got {sorted(families)!r}"
        )
    family = next(iter(families))
    registered = tuple(spec.name for spec in fr.list_formats(family))
    registered_set = set(registered)
    declared_set = set(declared)
    if declared_set != registered_set:
        missing = sorted(registered_set - declared_set)
        extra = sorted(declared_set - registered_set)
        raise ValueError(
            "source-class menus must cover the complete registered family; "
            f"family={family!r} missing={missing} extra={extra}. Refusing "
            "demand- or disk-driven candidate truncation."
        )
    # The registry sorts by effective rate.  Use it as the canonical ordering
    # so plan identity cannot depend on how two equivalent CLI strings happen
    # to interleave their shared formats.
    return registered


def _validate_declared_menus(
    expert_formats: tuple[str, ...],
    nonexpert_formats: tuple[str, ...],
    family_formats: tuple[str, ...],
) -> None:
    expert_set = set(expert_formats)
    nonexpert_set = set(nonexpert_formats)
    if not expert_set < nonexpert_set:
        raise ValueError(
            "expert-formats must be a strict subset of nonexpert-formats; "
            "the split represents lower- and higher-source-rate classes"
        )
    if nonexpert_set != set(family_formats):
        missing = sorted(set(family_formats) - nonexpert_set)
        extra = sorted(nonexpert_set - set(family_formats))
        raise ValueError(
            "nonexpert-formats must retain the complete registered family; "
            f"missing={missing} extra={extra}. Refusing truncation."
        )
    expected_expert_order = tuple(
        name for name in family_formats if name in expert_set
    )
    expected_nonexpert_order = tuple(
        name for name in family_formats if name in nonexpert_set
    )
    if expert_formats != expected_expert_order:
        raise ValueError(
            "expert-formats must follow the registry's increasing-rate order; "
            f"expected={list(expected_expert_order)}"
        )
    if nonexpert_formats != expected_nonexpert_order:
        raise ValueError(
            "nonexpert-formats must follow the registry's increasing-rate "
            f"order; expected={list(expected_nonexpert_order)}"
        )


def _serving_components(profile, qnames: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Union fused and packed groups exactly as serving promotion does."""
    names = tuple(sorted(dict.fromkeys(str(name) for name in qnames)))
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[tuple[str, str], list[str]] = {}
    for qname in names:
        for kind, accessor in (
            ("fused", "fused_sibling_group"),
            ("packed", "packed_expert_format_group"),
        ):
            method = getattr(profile, accessor, None)
            if not callable(method):
                raise RuntimeError(
                    f"profile {type(profile).__name__} is missing callable "
                    f"{accessor}(); source-class planning cannot validate "
                    "serving-atomic groups"
                )
            try:
                key = method(qname)
            except Exception as exc:
                raise RuntimeError(
                    f"profile {type(profile).__name__}.{accessor}({qname!r}) "
                    "failed during source-class planning"
                ) from exc
            if key is not None:
                groups.setdefault((kind, str(key)), []).append(qname)

    for members in groups.values():
        if len(members) < 2:
            continue
        first = members[0]
        for member in members[1:]:
            union(first, member)
    components: dict[str, list[str]] = {}
    for qname in names:
        components.setdefault(find(qname), []).append(qname)
    return tuple(
        tuple(sorted(members))
        for members in sorted(components.values(), key=lambda item: min(item))
        if len(members) > 1
    )


def build_source_class_format_plan(
    stats: Mapping[str, Mapping[str, object]],
    source_manifest: Mapping[str, str],
    profile,
    *,
    expert_formats: str | Sequence[str],
    nonexpert_formats: str | Sequence[str],
    cb_serialization_context,
) -> SourceClassFormatPlan:
    """Derive one exact full-family menu per source-payload class.

    ``stats`` must carry exact Linear shapes and ``source_manifest`` must be a
    complete checkpoint census in the allocator's recipe namespace.  The
    existing allocator source-rate predicate is called for every registered
    family member; this module intentionally contains no bpp formula.
    """
    expert_menu = parse_format_menu(expert_formats, where="expert-formats")
    nonexpert_menu = parse_format_menu(
        nonexpert_formats, where="nonexpert-formats"
    )
    family_formats = _complete_family(expert_menu, nonexpert_menu)
    _validate_declared_menus(expert_menu, nonexpert_menu, family_formats)
    if any(is_cb_format(name) for name in family_formats):
        if cb_serialization_context is None:
            raise ValueError(
                "source-class CB planning requires an exact "
                "CBSerializationContext; the allocator deliberately defers "
                "CB source-rate checks without it"
            )

    units: dict[str, UnitFormatPlan] = {}
    for qname in sorted(str(name) for name in stats):
        row = stats[qname]
        if not isinstance(row, Mapping):
            raise ValueError(f"probe stats row {qname!r} is not an object")
        shape = tuple(int(dim) for dim in _shape_from_stats(dict(row)))
        if len(shape) < 2 or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"{qname}: source-class planning needs an exact rank>=2 "
                f"Linear shape, got {shape}"
            )
        if qname not in source_manifest:
            raise ValueError(
                f"{qname}: source census is missing this planned unit; "
                "refusing an unverified source-rate class"
            )
        source_kind = str(source_manifest[qname])
        if source_footprint_owner_for_kind(source_kind) is None:
            raise ValueError(
                f"{qname}: source_kind={source_kind!r} has no exact source "
                "footprint owner; refusing source-class planning"
            )

        verdicts = {
            format_name: _source_bpp_applicability(
                shape,
                fr.get_format(format_name),
                qname=qname,
                source_kind=source_kind,
                cb_serialization_context=cb_serialization_context,
            )
            for format_name in family_formats
        }
        unexpected_reasons = sorted({
            verdict.reason
            for verdict in verdicts.values()
            if not verdict.legal
            and verdict.reason != SOURCE_BPP_EXCEEDED_REASON
        })
        if unexpected_reasons:
            sample = next(
                verdict.detail
                for verdict in verdicts.values()
                if not verdict.legal
                and verdict.reason != SOURCE_BPP_EXCEEDED_REASON
            )
            raise ValueError(
                f"{qname}: exact source-rate derivation failed with "
                f"{unexpected_reasons}: {sample}"
            )
        legal_formats = tuple(
            name for name in family_formats if verdicts[name].legal
        )
        if legal_formats == expert_menu:
            menu_id = EXPERT_MENU
        elif legal_formats == nonexpert_menu:
            menu_id = NONEXPERT_MENU
        else:
            raise ValueError(
                f"{qname}: source_kind={source_kind!r} derives legal family "
                f"{list(legal_formats)}, which matches neither declared menu. "
                "Refusing both illegal work and candidate truncation; declare "
                "the missing source class explicitly."
            )

        provenance = next(
            (
                verdict.provenance
                for verdict in verdicts.values()
                if isinstance(verdict.provenance, Mapping)
                and verdict.provenance.get("source_payload_bytes") is not None
            ),
            None,
        )
        if provenance is None:
            raise RuntimeError(
                f"{qname}: source-rate gate returned no exact payload "
                "provenance"
            )
        n_params = int(provenance["bpp_denominator_params"])
        if n_params != math.prod(shape):
            raise RuntimeError(
                f"{qname}: source-rate provenance parameter denominator "
                f"{n_params} disagrees with shape {shape}"
            )
        units[qname] = UnitFormatPlan(
            qname=qname,
            menu_id=menu_id,
            source_kind=source_kind,
            shape=shape,
            source_payload_bytes=int(provenance["source_payload_bytes"]),
            source_bpp_numerator_bits=int(
                provenance["source_bpp_numerator_bits"]
            ),
            bpp_denominator_params=n_params,
        )

    if not units:
        raise ValueError("source-class format plan has no units")
    unexpected_source_names = sorted(set(source_manifest) - set(units))
    # A source census normally includes only quantizable recipe qnames, but it
    # may legitimately carry pinned/passthrough entries.  They are not planned
    # and are intentionally ignored; completeness is checked in the other
    # direction above for every unit that will be rendered.
    del unexpected_source_names

    serving_groups = _serving_components(profile, tuple(units))
    for members in serving_groups:
        menu_ids = {units[qname].menu_id for qname in members}
        if len(menu_ids) != 1:
            detail = {
                qname: {
                    "menu_id": units[qname].menu_id,
                    "source_kind": units[qname].source_kind,
                }
                for qname in members
            }
            raise ValueError(
                "source-class format plan would split one fused/packed "
                f"serving component across menus: {detail}. Refusing to "
                "intersect or truncate the group."
            )

    provisional = SourceClassFormatPlan(
        menus={
            EXPERT_MENU: expert_menu,
            NONEXPERT_MENU: nonexpert_menu,
        },
        units=units,
        serving_groups=serving_groups,
        identity_sha256="",
    )
    body = provisional.to_dict()
    return SourceClassFormatPlan(
        menus=provisional.menus,
        units=provisional.units,
        serving_groups=provisional.serving_groups,
        identity_sha256=str(body["identity_sha256"]),
    )


def write_format_plan(plan: SourceClassFormatPlan, path: str | Path) -> None:
    body = plan.to_dict()
    expected = str(body["identity_sha256"])
    if plan.identity_sha256 and plan.identity_sha256 != expected:
        raise ValueError(
            "source-class format plan identity changed before publication: "
            f"stored={plan.identity_sha256} current={expected}"
        )
    encoded = json.dumps(
        body,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    atomic_write_bytes(Path(path), encoded)


def load_format_plan(path: str | Path) -> SourceClassFormatPlan:
    plan_path = Path(path)
    try:
        raw = json.loads(plan_path.read_text())
    except Exception as exc:
        raise ValueError(f"format plan {plan_path} is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"format plan {plan_path} is not an object")
    if raw.get("schema") != FORMAT_PLAN_SCHEMA:
        raise ValueError(
            f"format plan {plan_path} has unsupported schema "
            f"{raw.get('schema')!r}"
        )
    expected_digest = _plan_digest(raw)
    if raw.get("identity_sha256") != expected_digest:
        raise ValueError(
            f"format plan {plan_path} identity mismatch: stored="
            f"{raw.get('identity_sha256')!r} current={expected_digest!r}"
        )
    raw_menus = raw.get("menus")
    if not isinstance(raw_menus, Mapping):
        raise ValueError(f"format plan {plan_path} has no menus object")
    menus = {
        menu_id: parse_format_menu(
            raw_menus.get(menu_id, ()), where=f"format plan {menu_id} menu"
        )
        for menu_id in _MENU_IDS
    }
    family_formats = _complete_family(
        menus[EXPERT_MENU], menus[NONEXPERT_MENU]
    )
    _validate_declared_menus(
        menus[EXPERT_MENU], menus[NONEXPERT_MENU], family_formats
    )

    raw_units = raw.get("units")
    if not isinstance(raw_units, Mapping) or not raw_units:
        raise ValueError(f"format plan {plan_path} has no units")
    units: dict[str, UnitFormatPlan] = {}
    for raw_qname, value in raw_units.items():
        qname = str(raw_qname)
        if not isinstance(value, Mapping):
            raise ValueError(f"format plan unit {qname!r} is not an object")
        menu_id = str(value.get("menu_id", ""))
        if menu_id not in menus:
            raise ValueError(
                f"format plan unit {qname!r} has unknown menu {menu_id!r}"
            )
        shape = tuple(int(dim) for dim in value.get("shape", ()))
        if len(shape) < 2 or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"format plan unit {qname!r} has invalid shape {shape}"
            )
        denominator = int(value.get("bpp_denominator_params", 0))
        if denominator != math.prod(shape):
            raise ValueError(
                f"format plan unit {qname!r} has a parameter denominator "
                f"that disagrees with shape {shape}"
            )
        source_payload_bytes = int(value.get("source_payload_bytes", 0))
        source_bits = int(value.get("source_bpp_numerator_bits", 0))
        if source_payload_bytes <= 0 or source_bits != 8 * source_payload_bytes:
            raise ValueError(
                f"format plan unit {qname!r} has inconsistent source bytes"
            )
        units[qname] = UnitFormatPlan(
            qname=qname,
            menu_id=menu_id,
            source_kind=str(value.get("source_kind", "")),
            shape=shape,
            source_payload_bytes=source_payload_bytes,
            source_bpp_numerator_bits=source_bits,
            bpp_denominator_params=denominator,
        )

    raw_groups = raw.get("serving_groups", ())
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        raise ValueError(f"format plan {plan_path} serving_groups is invalid")
    serving_groups: list[tuple[str, ...]] = []
    for index, value in enumerate(raw_groups):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(
                f"format plan serving group {index} is not a sequence"
            )
        members = tuple(str(name) for name in value)
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(
                f"format plan serving group {index} is malformed: {members}"
            )
        missing = sorted(set(members) - set(units))
        if missing:
            raise ValueError(
                f"format plan serving group {index} names missing units "
                f"{missing}"
            )
        menu_ids = {units[name].menu_id for name in members}
        if len(menu_ids) != 1:
            raise ValueError(
                f"format plan serving group {index} straddles menus "
                f"{sorted(menu_ids)}"
            )
        serving_groups.append(members)

    return SourceClassFormatPlan(
        menus=menus,
        units=units,
        serving_groups=tuple(serving_groups),
        identity_sha256=expected_digest,
    )


__all__ = [
    "EXPERT_MENU",
    "FORMAT_PLAN_SCHEMA",
    "NONEXPERT_MENU",
    "SourceClassFormatPlan",
    "UnitFormatPlan",
    "build_source_class_format_plan",
    "load_format_plan",
    "parse_format_menu",
    "write_format_plan",
]
