"""Small refinement-unit helpers for local per-layer reconstruction tools."""
from __future__ import annotations

from dataclasses import dataclass

from .allocator import Candidate, _group_by_profile


@dataclass(frozen=True)
class UnitOption:
    fmt: str
    bits_total: float
    predicted_dloss: float


@dataclass
class RefinementUnit:
    key: str
    members: tuple[str, ...]
    base_fmt: str
    base_member_fmts: tuple[tuple[str, str], ...]
    options: tuple[UnitOption, ...]

    @property
    def option_map(self) -> dict[str, UnitOption]:
        return {opt.fmt: opt for opt in self.options}


def _block_group_for_name(name: str, present: set[str]) -> tuple[str, ...] | None:
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    prefix = ".".join(parts[:3])
    leaf = parts[-1]
    if parts[3] == "self_attn" and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        members = tuple(
            sorted(
                f"{prefix}.self_attn.{proj}"
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
                if f"{prefix}.self_attn.{proj}" in present
            )
        )
        return members if len(members) > 1 else None
    if parts[3] == "mlp" and leaf in {"gate_proj", "up_proj", "down_proj"}:
        members = tuple(
            sorted(
                f"{prefix}.mlp.{proj}"
                for proj in ("gate_proj", "up_proj", "down_proj")
                if f"{prefix}.mlp.{proj}" in present
            )
        )
        return members if len(members) > 1 else None
    return None


def _layer_group_for_name(name: str, present: set[str]) -> tuple[str, ...] | None:
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    prefix = ".".join(parts[:3]) + "."
    members = tuple(sorted(n for n in present if n.startswith(prefix)))
    return members if len(members) > 1 else None


_SIBLING_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (".self_attn.", ("q_proj", "k_proj", "v_proj")),
    (".mlp.", ("gate_proj", "up_proj")),
    (".mlp.shared_expert.", ("gate_proj", "up_proj")),
    (".linear_attn.", ("in_proj_qkv", "in_proj_z")),
    (".linear_attn.", ("in_proj_a", "in_proj_b")),
)


def _name_pattern_siblings(name: str, present: set[str]) -> tuple[str, ...] | None:
    for parent_marker, leaves in _SIBLING_PATTERNS:
        idx = name.rfind(parent_marker)
        if idx < 0:
            continue
        parent = name[:idx + len(parent_marker)]
        leaf = name[idx + len(parent_marker):]
        if leaf not in leaves:
            continue
        members = tuple(sorted(
            f"{parent}{cand}" for cand in leaves if f"{parent}{cand}" in present
        ))
        if len(members) > 1:
            return members
    return None


def _unit_groups(names: list[str], unit_scope: str = "sibling") -> list[tuple[str, ...]]:
    present = set(names)
    from .model_profiles import DefaultProfile

    sibling_key_to_names = _group_by_profile(list(present), DefaultProfile())
    name_to_fusion = {
        name: tuple(sorted(members))
        for members in sibling_key_to_names.values()
        for name in members
    }

    groups: dict[tuple[str, ...], tuple[str, ...]] = {}
    for name in names:
        if ".__fused__." in name:
            key = (name,)
        else:
            key = None
            if unit_scope == "layer":
                key = _layer_group_for_name(name, present)
            if unit_scope in {"block", "hybrid"}:
                key = _block_group_for_name(name, present)
            if unit_scope == "layer" and key is None:
                key = _layer_group_for_name(name, present)
            if key is None:
                sibs = name_to_fusion.get(name)
                if sibs is not None and len(sibs) > 1:
                    key = sibs
            if key is None:
                sibs = _name_pattern_siblings(name, present)
                if sibs is not None:
                    key = sibs
            if key is None:
                key = (name,)
            else:
                key = tuple(sorted(set(key)))
        groups[key] = tuple(sorted(set(key)))
    return sorted(groups.values())


def build_refinement_units(
    stats: dict,
    candidates: dict[str, list[Candidate]],
    assignment: dict[str, str],
    unit_scope: str = "sibling",
) -> list[RefinementUnit]:
    units = []
    for members in _unit_groups(list(assignment.keys()), unit_scope=unit_scope):
        base_fmts = {assignment[m] for m in members}
        base_member_fmts = tuple((member, assignment[member]) for member in members)
        heterogeneous_base = len(base_fmts) != 1
        base_fmt = "__base__" if heterogeneous_base else next(iter(base_fmts))
        fmt_sets = [
            {cand.fmt for cand in candidates[m]}
            for m in members
            if m in candidates
        ]
        if not fmt_sets:
            continue
        shared = set.intersection(*fmt_sets)
        options = []
        if heterogeneous_base:
            bits_total = 0.0
            predicted = 0.0
            for member in members:
                cand = next(c for c in candidates[member] if c.fmt == assignment[member])
                n_params = stats[member]["n_params"]
                bits_total += cand.bits_per_param * n_params
                predicted += cand.predicted_dloss
            options.append(UnitOption(
                fmt="__base__",
                bits_total=bits_total,
                predicted_dloss=predicted,
            ))
        for fmt in shared:
            bits_total = 0.0
            predicted = 0.0
            for member in members:
                n_params = stats[member]["n_params"]
                cand = next(c for c in candidates[member] if c.fmt == fmt)
                bits_total += cand.bits_per_param * n_params
                predicted += cand.predicted_dloss
            options.append(UnitOption(
                fmt=fmt,
                bits_total=bits_total,
                predicted_dloss=predicted,
            ))
        options.sort(key=lambda opt: (opt.bits_total, opt.predicted_dloss, opt.fmt))
        if not options:
            continue
        units.append(RefinementUnit(
            key="|".join(members),
            members=members,
            base_fmt=base_fmt,
            base_member_fmts=base_member_fmts,
            options=tuple(options),
        ))
    return units


def select_critical_units(units: list[RefinementUnit], top_n: int) -> list[RefinementUnit]:
    scored = []
    for unit in units:
        opt_map = unit.option_map
        base = opt_map[unit.base_fmt]
        cheapest = min(unit.options, key=lambda opt: (opt.bits_total, opt.predicted_dloss))
        gain = max(cheapest.predicted_dloss - base.predicted_dloss, 0.0)
        scored.append((gain, base.predicted_dloss, unit.key, unit))
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [row[-1] for row in scored[:top_n]]


def expand_unit_assignment(
    units: list[RefinementUnit],
    choices: dict[str, str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    by_key = {unit.key: unit for unit in units}
    for key, fmt in choices.items():
        unit = by_key[key]
        if fmt == "__base__":
            out.update(unit.base_member_fmts)
        else:
            for member in unit.members:
                out[member] = fmt
    return out
