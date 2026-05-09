"""Block-restricted CLADO surrogate for mixed-precision allocation.

Background
----------

CLADO (arXiv:2307.05657, "Mixed-Precision Quantization for Deep Vision Models
with Integer Quadratic Programming") models the loss-Hessian factorisation

    L(w + Δw) ≈ L(w) + g_w·Δw + 1/2·Δw^T H_w Δw

with the cross-layer block H_ij explicitly measured via the four-term identity

    Ω_ij(Δ_i, Δ_j) ≈ L(w + Δ_i + Δ_j) + L(w) − L(w + Δ_i) − L(w + Δ_j).

Full-coverage CLADO requires O(|𝔹|² · I²) loss evaluations.  At LLM scale that
is intractable (~700 GPU-hours for a Qwen 27B-class model).

Block-CLADO restricts the pair set to within-architectural-block edges only.
Empirically, transformer LayerNorms reset error magnitude between blocks, so
cross-block Ω_ij is small relative to within-block.  Coverage drops from
O(I²) to O(L · K²) where K is the number of fused-sibling groups inside one
block (~4 for standard q/k/v + o + gate/up + down transformers), which makes
the surrogate measurable in a few minutes on small LLMs.

Two adaptations vs. classical CLADO
-----------------------------------

1. Loss is teacher-student KL evaluated at the BF16 reference state.  This
   makes the linear term in the Taylor expansion exactly zero (KL is minimised
   when distributions match) and the quadratic operator the categorical Fisher
   instead of the empirical task-loss Hessian.  Side benefit: KL(teacher‖teacher)
   = 0 is free, so the four-term identity collapses to three measured terms.

2. The bit-width assignment problem is solved per-block by exact enumeration
   of all fused-group format combinations, then composed across blocks via a
   multi-choice knapsack DP / Lagrangian λ-sweep.  The integer quadratic
   programming step CLADO uses is unnecessary because each block's clique has
   small treewidth.

Schema
------

The portable artifact ``prismaquant.block_clado.v1`` is a JSON document::

    {
      "schema": "prismaquant.block_clado.v1",
      "blocks": {
        "<block_id>": {
          "units": {
            "<unit_name>": {
              "options": {
                "<format>": {"omega_ii": float,
                             "bits_per_param": float,
                             "memory_bytes": int}
              }
            }
          },
          "pairs": [
            {
              "unit_a": str, "unit_b": str,
              "omega_ij": {"<fmt_a>__<fmt_b>": float}
            }
          ]
        }
      },
      "singletons": {
        "<unit_name>": { ... same as block.units.<unit_name> }
      },
      "meta": { ... model, calibration metadata ... }
    }
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr
from .allocator_candidates import check_format_applicability
from .allocator_solver import _shape_from_stats


SCHEMA = "prismaquant.block_clado.v1"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatCost:
    """One format choice for one fused-group decision unit."""

    fmt: str
    omega_ii: float
    bits_per_param: float
    memory_bytes: int

    @property
    def bits_total(self) -> float:
        return float(self.memory_bytes) * 8.0


@dataclass(frozen=True)
class DecisionUnit:
    """A fused-sibling group treated as a single allocator decision."""

    name: str
    block_id: str
    member_qnames: tuple[str, ...]
    options: tuple[FormatCost, ...]


@dataclass(frozen=True)
class BlockPair:
    """A measured intra-block interaction between two decision units."""

    unit_a: str
    unit_b: str
    block_id: str
    omega_ij: dict[tuple[str, str], float]


@dataclass(frozen=True)
class BlockSolution:
    """Outcome of solving one block at a single (cost, bits) state."""

    block_id: str
    assignment: dict[str, str]
    cost: float
    bits_total: float


@dataclass(frozen=True)
class GlobalSolveResult:
    assignment: dict[str, str]
    cost_total: float
    bits_total: float
    bpp: float
    lambda_used: float | None
    per_block_costs: dict[str, float]
    per_block_bits: dict[str, float]


def unit_is_bf16_pinned(
    unit: DecisionUnit,
    pin_to_bf16: Sequence[str] = ("lm_head",),
) -> bool:
    """Return True when a decision unit should only expose BF16.

    Matching uses dotted-path tokens rather than substring matching, so a
    pin token such as ``lm_head`` will not match ``lm_head_norm``.
    """
    pin_tokens = [str(tok) for tok in (pin_to_bf16 or []) if str(tok)]
    if not pin_tokens:
        return False
    for candidate in (unit.name, *unit.member_qnames):
        parts = str(candidate).split(".")
        if any(tok in parts for tok in pin_tokens):
            return True
    return False


def apply_bf16_pins_to_units(
    blocks: Mapping[str, Sequence[DecisionUnit]],
    singletons: Sequence[DecisionUnit],
    *,
    pin_to_bf16: Sequence[str] = ("lm_head",),
) -> tuple[dict[str, list[DecisionUnit]], list[DecisionUnit]]:
    """Restrict pinned units to their BF16 option before measurement/solve."""

    def _pin(unit: DecisionUnit) -> DecisionUnit:
        if not unit_is_bf16_pinned(unit, pin_to_bf16):
            return unit
        bf16_options = tuple(
            opt for opt in unit.options
            if fr.canonical_format_name(opt.fmt) == "BF16"
        )
        if not bf16_options:
            return unit
        return DecisionUnit(
            name=unit.name,
            block_id=unit.block_id,
            member_qnames=unit.member_qnames,
            options=bf16_options,
        )

    pinned_blocks = {
        str(block_id): [_pin(unit) for unit in units]
        for block_id, units in blocks.items()
    }
    pinned_singletons = [_pin(unit) for unit in singletons]
    return pinned_blocks, pinned_singletons


# ---------------------------------------------------------------------------
# Block enumeration
# ---------------------------------------------------------------------------


def block_id_from_qname(qname: str) -> str:
    """Return the architectural block ID a quantizable Linear belongs to.

    Standard transformer layouts use ``model.layers.<i>.<...>``; everything
    else (lm_head, embeddings, MTP heads, etc.) is its own singleton block.
    """
    parts = qname.split(".")
    for idx, token in enumerate(parts):
        if token == "layers" and idx + 1 < len(parts):
            try:
                int(parts[idx + 1])
            except ValueError:
                continue
            # Use the first I+1 components as the block id, e.g.
            # 'model.layers.3'.
            return ".".join(parts[: idx + 2])
    return qname  # singleton block: lm_head, embeddings, etc.


def fused_group_key(profile, qname: str) -> str:
    """Profile-aware fused-sibling group key, or the bare qname if none."""
    try:
        if profile is not None:
            group = profile.fused_sibling_group(qname)
            if group:
                return group
    except Exception:
        pass
    return qname


def _recipe_name(full_name: str) -> str:
    return full_name[:-7] if full_name.endswith(".weight") else full_name


def _enumerate_quantizable_linears(model) -> list[str]:
    from .build_rtn_cache import iter_quantizable_tensors
    names: list[str] = []
    for full_name, _module, _attr in iter_quantizable_tensors(model):
        names.append(_recipe_name(full_name))
    return sorted(set(names))


def _shape_of_param(model, qname: str) -> tuple[int, ...] | None:
    from .build_rtn_cache import iter_quantizable_tensors
    target = qname
    for full_name, module, attr in iter_quantizable_tensors(model):
        if _recipe_name(full_name) == target:
            param = getattr(module, attr, None)
            if param is None:
                return None
            return tuple(int(v) for v in param.shape)
    return None


def _prod(seq: Iterable[int]) -> int:
    out = 1
    for v in seq:
        out *= int(v)
    return out


def discover_units(
    model,
    profile,
    formats: Sequence["fr.FormatSpec"],
) -> tuple[
    dict[str, list["DecisionUnit"]],
    list["DecisionUnit"],
    dict[str, int],
]:
    """Build the {block_id → [DecisionUnit]} dict directly from the model.

    No measurement — just sibling-fusion + format-menu enumeration.  Each
    unit's ``options`` carry ``omega_ii=0.0`` because the polish pipeline
    gates on real-KL and doesn't need the surrogate.

    Returns:
      blocks:     transformer-block decision units
      singletons: lm_head/embed/MTP/etc. (one option set per unit)
      n_params_by_unit: param counts for telemetry / total-bpp computation
    """
    qnames = _enumerate_quantizable_linears(model)
    groups: dict[str, list[str]] = defaultdict(list)
    for qname in qnames:
        key = fused_group_key(profile, qname)
        groups[key].append(qname)

    blocks: dict[str, list[DecisionUnit]] = defaultdict(list)
    singletons: list[DecisionUnit] = []
    n_params_by_unit: dict[str, int] = {}

    for group_name, members in sorted(groups.items()):
        members = sorted(set(members))
        block_ids = [block_id_from_qname(m) for m in members]
        block_id = max(set(block_ids), key=block_ids.count) if block_ids else group_name
        member_shapes_by_name: dict[str, tuple[int, ...]] = {}
        for member in members:
            shape = _shape_of_param(model, member)
            if shape is None:
                continue
            member_shapes_by_name[member] = shape
        if not member_shapes_by_name:
            continue
        member_shapes = list(member_shapes_by_name.values())
        n_params_unit = sum(int(_prod(s)) for s in member_shapes)
        n_params_by_unit[group_name] = n_params_unit

        options = []
        for spec in formats:
            spec_canon = fr.canonical_format_name(spec.name)
            if not all(
                check_format_applicability(
                    shape,
                    spec,
                    qname=member,
                    target_profile="research",
                ).legal
                for member, shape in member_shapes_by_name.items()
            ):
                continue
            mem_bytes = sum(spec.memory_bytes_for_shape(s) for s in member_shapes)
            bits_per_param = 8.0 * mem_bytes / max(n_params_unit, 1)
            options.append(FormatCost(
                fmt=spec_canon,
                omega_ii=0.0,
                bits_per_param=float(bits_per_param),
                memory_bytes=int(mem_bytes),
            ))
        if not options:
            continue
        options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
        unit = DecisionUnit(
            name=group_name,
            block_id=block_id,
            member_qnames=tuple(members),
            options=tuple(options),
        )
        if block_id == group_name and ".layers." not in block_id:
            singletons.append(unit)
        else:
            blocks[block_id].append(unit)

    pruned_blocks: dict[str, list[DecisionUnit]] = {}
    for block_id, units in blocks.items():
        if len(units) == 1 and ".layers." not in block_id:
            singletons.append(units[0])
        else:
            pruned_blocks[block_id] = units
    return pruned_blocks, singletons, n_params_by_unit


def floor_assignment(
    model,
    profile,
    formats: Sequence["fr.FormatSpec"],
    *,
    pin_to_bf16: Sequence[str] = ("lm_head",),
) -> tuple[dict[str, str], list["DecisionUnit"]]:
    """Build the all-min-bpp assignment, pinning specified qnames to BF16.

    Returns ``(per_linear_assignment, units)`` where ``units`` is the flat
    list usable by ``coord_descent_polish``.  The min-bpp format is chosen
    per-unit (typically NVFP4 if available); pinned units use BF16
    regardless of menu.
    """
    blocks, singletons, _np = discover_units(model, profile, formats)
    units: list[DecisionUnit] = []
    for unit_list in blocks.values():
        units.extend(unit_list)
    units.extend(singletons)

    pin_tokens = list(pin_to_bf16 or [])
    assignment: dict[str, str] = {}

    def _is_pinned(unit: DecisionUnit) -> bool:
        # MED-5: exact-token match on dotted-path components, not substring.
        # Substring matching catches false positives like "lm_head_norm".
        for tok in pin_tokens:
            if tok in unit.name.split("."):
                return True
            for m in unit.member_qnames:
                if tok in m.split("."):
                    return True
        return False

    for unit in units:
        if _is_pinned(unit):
            chosen_fmt = "BF16"
        else:
            chosen_fmt = min(unit.options, key=lambda opt: opt.bits_per_param).fmt
        for member in unit.member_qnames:
            assignment[member] = chosen_fmt
    return assignment, units


def collapse_assignment_to_units(
    base_assignment: Mapping[str, str],
    units: Sequence[DecisionUnit],
) -> dict[str, str]:
    """For a per-Linear assignment, derive the per-unit assignment.

    All members of one fused group must share a format; if they don't (the
    base assignment violates the unit grouping), pick the format used by
    the lexicographically first member.  Members not in any unit are dropped.
    """
    out: dict[str, str] = {}
    for unit in units:
        chosen: str | None = None
        for member in unit.member_qnames:
            fmt = base_assignment.get(member)
            if fmt is None:
                continue
            chosen = fr.canonical_format_name(fmt)
            break
        if chosen is None:
            chosen = unit.options[0].fmt
        out[unit.name] = chosen
    return out


def expand_unit_assignment(
    unit_assignment: Mapping[str, str],
    units: Sequence[DecisionUnit],
) -> dict[str, str]:
    """Broadcast a per-unit assignment back to per-Linear member names."""
    by_name = {unit.name: unit for unit in units}
    out: dict[str, str] = {}
    for unit_name, fmt in unit_assignment.items():
        unit = by_name.get(str(unit_name))
        canonical = fr.canonical_format_name(str(fmt))
        if unit is None:
            out[str(unit_name)] = canonical
            continue
        for member in unit.member_qnames:
            out[member] = canonical
    return out


# ---------------------------------------------------------------------------
# Payload IO
# ---------------------------------------------------------------------------


def _pair_key(fmt_a: str, fmt_b: str) -> str:
    return f"{fr.canonical_format_name(fmt_a)}__{fr.canonical_format_name(fmt_b)}"


def _parse_pair_key(key: str) -> tuple[str, str]:
    if "__" not in key:
        raise ValueError(f"invalid pair key: {key!r}")
    a, b = key.split("__", 1)
    return fr.canonical_format_name(a), fr.canonical_format_name(b)


def center_kl_from_payload(payload: Mapping) -> float:
    """Return ``KL(x_c)`` for the payload, or 0.0 if uncentered."""
    return float((payload.get("meta") or {}).get("center_kl", 0.0))


def units_and_pairs_to_payload(
    *,
    blocks: Mapping[str, Sequence[DecisionUnit]],
    singletons: Sequence[DecisionUnit],
    pairs_by_block: Mapping[str, Sequence[BlockPair]],
    meta: Mapping | None = None,
) -> dict:
    """Render the in-memory units/pairs into the portable JSON payload."""
    out_blocks = {}
    for block_id, units in blocks.items():
        unit_payload = {}
        for unit in units:
            unit_payload[unit.name] = {
                "members": list(unit.member_qnames),
                "options": {
                    cost.fmt: {
                        "omega_ii": float(cost.omega_ii),
                        "bits_per_param": float(cost.bits_per_param),
                        "memory_bytes": int(cost.memory_bytes),
                    }
                    for cost in unit.options
                },
            }
        pair_payload = []
        for pair in pairs_by_block.get(block_id, ()):  # type: ignore[arg-type]
            pair_payload.append({
                "unit_a": pair.unit_a,
                "unit_b": pair.unit_b,
                "omega_ij": {
                    _pair_key(a, b): float(v)
                    for (a, b), v in pair.omega_ij.items()
                },
            })
        out_blocks[block_id] = {
            "units": unit_payload,
            "pairs": pair_payload,
        }
    singleton_payload = {}
    for unit in singletons:
        singleton_payload[unit.name] = {
            "block_id": unit.block_id,
            "members": list(unit.member_qnames),
            "options": {
                cost.fmt: {
                    "omega_ii": float(cost.omega_ii),
                    "bits_per_param": float(cost.bits_per_param),
                    "memory_bytes": int(cost.memory_bytes),
                }
                for cost in unit.options
            },
        }
    payload = {
        "schema": SCHEMA,
        "blocks": out_blocks,
        "singletons": singleton_payload,
    }
    if meta:
        payload["meta"] = dict(meta)
    return payload


def parse_payload(payload: Mapping) -> tuple[
    dict[str, list[DecisionUnit]],
    list[DecisionUnit],
    dict[str, list[BlockPair]],
]:
    """Re-hydrate units, singletons, and intra-block pairs from JSON."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported block-CLADO schema: {payload.get('schema')!r}")
    blocks: dict[str, list[DecisionUnit]] = {}
    pairs_by_block: dict[str, list[BlockPair]] = {}
    for block_id, block_payload in (payload.get("blocks") or {}).items():
        units: list[DecisionUnit] = []
        for unit_name, unit_payload in (block_payload.get("units") or {}).items():
            options = []
            for fmt, entry in unit_payload["options"].items():
                options.append(FormatCost(
                    fmt=fr.canonical_format_name(fmt),
                    omega_ii=float(entry["omega_ii"]),
                    bits_per_param=float(entry["bits_per_param"]),
                    memory_bytes=int(entry["memory_bytes"]),
                ))
            options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
            units.append(DecisionUnit(
                name=str(unit_name),
                block_id=str(block_id),
                member_qnames=tuple(
                    unit_payload.get("members")
                    or unit_payload.get("member_qnames")
                    or [unit_name]
                ),
                options=tuple(options),
            ))
        units.sort(key=lambda unit: unit.name)
        blocks[str(block_id)] = units
        pairs: list[BlockPair] = []
        for pair_payload in block_payload.get("pairs") or []:
            omega = {}
            for key, value in (pair_payload.get("omega_ij") or {}).items():
                fmt_a, fmt_b = _parse_pair_key(str(key))
                omega[(fmt_a, fmt_b)] = float(value)
            pairs.append(BlockPair(
                unit_a=str(pair_payload["unit_a"]),
                unit_b=str(pair_payload["unit_b"]),
                block_id=str(block_id),
                omega_ij=omega,
            ))
        pairs_by_block[str(block_id)] = pairs
    singletons: list[DecisionUnit] = []
    for unit_name, unit_payload in (payload.get("singletons") or {}).items():
        options = []
        for fmt, entry in unit_payload["options"].items():
            options.append(FormatCost(
                fmt=fr.canonical_format_name(fmt),
                omega_ii=float(entry["omega_ii"]),
                bits_per_param=float(entry["bits_per_param"]),
                memory_bytes=int(entry["memory_bytes"]),
            ))
        options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
        singletons.append(DecisionUnit(
            name=str(unit_name),
            block_id=str(unit_payload.get("block_id") or unit_name),
            member_qnames=tuple(
                unit_payload.get("members")
                or unit_payload.get("member_qnames")
                or [unit_name]
            ),
            options=tuple(options),
        ))
    singletons.sort(key=lambda unit: unit.name)
    return blocks, singletons, pairs_by_block


def load_payload(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Per-block exact enumeration + Pareto filtering
# ---------------------------------------------------------------------------


def score_block_assignment(
    units: Sequence[DecisionUnit],
    assignment: Mapping[str, str],
    pairs: Sequence[BlockPair],
) -> tuple[float, float]:
    """Return ``(cost, bits_total)`` for one block-level assignment."""
    by_unit_format = {unit.name: {opt.fmt: opt for opt in unit.options} for unit in units}
    cost = 0.0
    bits = 0.0
    for unit in units:
        opt = by_unit_format[unit.name][assignment[unit.name]]
        cost += opt.omega_ii
        bits += opt.bits_total
    for pair in pairs:
        fmt_a = assignment[pair.unit_a]
        fmt_b = assignment[pair.unit_b]
        # Pair payloads are stored with one ordering only; check both keys.
        omega = pair.omega_ij.get((fmt_a, fmt_b))
        if omega is None:
            omega = pair.omega_ij.get((fmt_b, fmt_a))
        if omega is None:
            # Missing entry — treat as 0 (no measured interaction).
            omega = 0.0
        cost += float(omega)
    return float(cost), float(bits)


def enumerate_block_states(
    units: Sequence[DecisionUnit],
    pairs: Sequence[BlockPair],
    *,
    max_states: int | None = 65_536,
) -> list[BlockSolution]:
    """Enumerate every fused-group format tuple in a block, return Pareto-only.

    A state is dominated if some other state has both lower cost AND fewer
    bits.  We keep the Pareto frontier so that the global multi-choice
    knapsack only considers efficient candidates per block.
    """
    if not units:
        return []
    # Cap total enumeration to guard against pathologically large blocks.
    total_combinations = 1
    for unit in units:
        total_combinations *= max(len(unit.options), 1)
    if max_states is not None and total_combinations > int(max_states):
        raise ValueError(
            f"block has {total_combinations} format tuples > max {max_states};"
            " refusing to enumerate.  Reduce format menu or split block."
        )

    block_id = units[0].block_id
    all_states: list[BlockSolution] = []
    option_lists = [unit.options for unit in units]

    def recurse(idx: int, partial: dict[str, str]) -> None:
        if idx == len(units):
            cost, bits = score_block_assignment(units, partial, pairs)
            all_states.append(BlockSolution(
                block_id=block_id,
                assignment=dict(partial),
                cost=float(cost),
                bits_total=float(bits),
            ))
            return
        unit = units[idx]
        for option in option_lists[idx]:
            partial[unit.name] = option.fmt
            recurse(idx + 1, partial)
            del partial[unit.name]

    recurse(0, {})

    # Pareto filter: sort by bits ascending then cost ascending; keep states
    # whose cost strictly improves the running minimum.
    all_states.sort(key=lambda s: (s.bits_total, s.cost))
    pareto: list[BlockSolution] = []
    best_cost = math.inf
    for state in all_states:
        if state.cost < best_cost - 1e-12:
            pareto.append(state)
            best_cost = state.cost
    return pareto


def enumerate_singleton_states(
    unit: DecisionUnit,
) -> list[BlockSolution]:
    """A singleton unit (lm_head, embeddings, etc.) has no pair terms."""
    states = []
    for option in unit.options:
        states.append(BlockSolution(
            block_id=unit.block_id,
            assignment={unit.name: option.fmt},
            cost=float(option.omega_ii),
            bits_total=float(option.bits_total),
        ))
    states.sort(key=lambda s: (s.bits_total, s.cost))
    pareto: list[BlockSolution] = []
    best_cost = math.inf
    for state in states:
        if state.cost < best_cost - 1e-12:
            pareto.append(state)
            best_cost = state.cost
    return pareto


# ---------------------------------------------------------------------------
# Global multi-choice knapsack across blocks
# ---------------------------------------------------------------------------


def _per_block_lambda_pick(
    block_states: Sequence[BlockSolution],
    lambda_penalty: float,
) -> BlockSolution:
    """Pick the block state minimising ``cost + λ · bits``."""
    best: BlockSolution | None = None
    best_score = math.inf
    for state in block_states:
        score = state.cost + float(lambda_penalty) * state.bits_total
        if score < best_score - 1e-12:
            best_score = score
            best = state
    if best is None:
        raise ValueError("empty block state set")
    return best


def solve_lagrangian(
    block_states: Mapping[str, Sequence[BlockSolution]],
    *,
    lambda_penalty: float,
) -> GlobalSolveResult:
    """Per-block independent λ-sweep selection.

    Each block independently chooses the state minimising ``cost + λ·bits``.
    Across blocks the choices are independent, so the sweep is O(blocks ·
    states_per_block) — i.e. ~thousands of operations even at LLM scale.
    """
    assignment: dict[str, str] = {}
    cost_total = 0.0
    bits_total = 0.0
    per_block_costs: dict[str, float] = {}
    per_block_bits: dict[str, float] = {}
    for block_id, states in block_states.items():
        pick = _per_block_lambda_pick(states, lambda_penalty)
        assignment.update(pick.assignment)
        cost_total += pick.cost
        bits_total += pick.bits_total
        per_block_costs[block_id] = pick.cost
        per_block_bits[block_id] = pick.bits_total
    return GlobalSolveResult(
        assignment=dict(assignment),
        cost_total=float(cost_total),
        bits_total=float(bits_total),
        bpp=0.0,  # filled by caller from total params
        lambda_used=float(lambda_penalty),
        per_block_costs=per_block_costs,
        per_block_bits=per_block_bits,
    )


def lambda_sweep(
    block_states: Mapping[str, Sequence[BlockSolution]],
    *,
    lambda_min: float,
    lambda_max: float,
    n_lambdas: int,
    log_scale: bool = True,
) -> list[GlobalSolveResult]:
    """Sweep λ across a range and return the unique-by-bits frontier."""
    if n_lambdas <= 0:
        raise ValueError("n_lambdas must be positive")
    if lambda_min < 0 or lambda_max <= lambda_min:
        raise ValueError("require 0 ≤ lambda_min < lambda_max")
    if log_scale and lambda_min <= 0:
        # Tiny floor so log-scale doesn't explode.
        lambda_min = max(lambda_min, 1e-30)
    lambdas: list[float] = []
    if log_scale:
        log_lo = math.log10(lambda_min)
        log_hi = math.log10(lambda_max)
        for k in range(n_lambdas):
            t = k / max(n_lambdas - 1, 1)
            lambdas.append(10.0 ** (log_lo + t * (log_hi - log_lo)))
    else:
        for k in range(n_lambdas):
            t = k / max(n_lambdas - 1, 1)
            lambdas.append(lambda_min + t * (lambda_max - lambda_min))
    seen_keys: set[tuple[float, float]] = set()
    results: list[GlobalSolveResult] = []
    for lam in lambdas:
        result = solve_lagrangian(block_states, lambda_penalty=lam)
        key = (round(result.bits_total, 6), round(result.cost_total, 9))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        results.append(result)
    results.sort(key=lambda r: (r.bits_total, r.cost_total))
    return results


def solve_budget(
    block_states: Mapping[str, Sequence[BlockSolution]],
    *,
    bits_budget: float,
    bit_precision_bits: float = 1.0,
) -> GlobalSolveResult | None:
    """Multi-choice knapsack: minimise cost s.t. Σ bits ≤ ``bits_budget``.

    ``bit_precision_bits`` controls the bin width of the budget DP.  Larger
    values are faster; smaller values give more accurate budget enforcement.
    Default 1 bit/param (across all params) is plenty for kneedle search.
    """
    if not block_states:
        raise ValueError("no block states supplied")
    if bits_budget <= 0:
        raise ValueError("bits_budget must be positive")

    bits_min = sum(min(s.bits_total for s in states) for states in block_states.values())
    if bits_min > bits_budget + 1e-6:
        return None

    bin_width = max(float(bit_precision_bits), 1e-9)
    bins = max(int(math.ceil(bits_budget / bin_width)) + 1, 2)
    INF = math.inf

    # dp[b] = (cost, prev_bin, prev_choice) where prev_choice is the index of
    # the state chosen for the *previous* block.  We carry prev_bin so the
    # backtrack reconstructs assignments precisely.
    block_ids = list(block_states.keys())
    state_lists = [list(block_states[bid]) for bid in block_ids]

    dp_prev_cost = [INF] * bins
    dp_prev_cost[0] = 0.0
    # Backpointers: (prev_bin, choice_idx) per block per bin.
    backpointers: list[list[tuple[int, int] | None]] = []

    for state_list in state_lists:
        next_cost = [INF] * bins
        bp_layer: list[tuple[int, int] | None] = [None] * bins
        for prev_bin, prev_cost in enumerate(dp_prev_cost):
            if prev_cost == INF:
                continue
            for choice_idx, state in enumerate(state_list):
                bin_increment = int(math.floor(state.bits_total / bin_width))
                next_bin = prev_bin + bin_increment
                if next_bin >= bins:
                    continue
                candidate_cost = prev_cost + state.cost
                if candidate_cost < next_cost[next_bin] - 1e-12:
                    next_cost[next_bin] = candidate_cost
                    bp_layer[next_bin] = (prev_bin, choice_idx)
        dp_prev_cost = next_cost
        backpointers.append(bp_layer)

    # Find the best feasible final bin.
    best_bin = -1
    best_cost = INF
    for b, cost in enumerate(dp_prev_cost):
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_bin = b
    if best_bin < 0 or best_cost == INF:
        return None

    # Backtrack to recover the chosen state per block.
    assignment: dict[str, str] = {}
    cost_total = 0.0
    bits_total = 0.0
    per_block_costs: dict[str, float] = {}
    per_block_bits: dict[str, float] = {}
    cur_bin = best_bin
    for level in range(len(state_lists) - 1, -1, -1):
        entry = backpointers[level][cur_bin]
        if entry is None:
            return None
        prev_bin, choice_idx = entry
        state = state_lists[level][choice_idx]
        block_id = block_ids[level]
        assignment.update(state.assignment)
        cost_total += state.cost
        bits_total += state.bits_total
        per_block_costs[block_id] = state.cost
        per_block_bits[block_id] = state.bits_total
        cur_bin = prev_bin

    return GlobalSolveResult(
        assignment=dict(assignment),
        cost_total=float(cost_total),
        bits_total=float(bits_total),
        bpp=0.0,
        lambda_used=None,
        per_block_costs=per_block_costs,
        per_block_bits=per_block_bits,
    )


# ---------------------------------------------------------------------------
# Driver helpers — load payload, build block_states, run sweep
# ---------------------------------------------------------------------------


def build_block_states(
    payload: Mapping,
    *,
    max_states_per_block: int | None = 65_536,
) -> dict[str, list[BlockSolution]]:
    """Wrap parse_payload + per-block enumeration into one call."""
    blocks, singletons, pairs_by_block = parse_payload(payload)
    block_states: dict[str, list[BlockSolution]] = {}
    for block_id, units in blocks.items():
        states = enumerate_block_states(
            units,
            pairs_by_block.get(block_id, []),
            max_states=max_states_per_block,
        )
        block_states[block_id] = states
    for unit in singletons:
        block_states[unit.block_id] = enumerate_singleton_states(unit)
    return block_states


def total_param_count(payload: Mapping) -> int:
    """Sum n_params across all units in the payload (units carry no shape;
    use the BF16 option's memory_bytes to back-derive)."""
    total = 0
    for block in (payload.get("blocks") or {}).values():
        for unit in (block.get("units") or {}).values():
            options = unit.get("options") or {}
            ref = options.get("BF16")
            if ref is None:
                # Fall back to whichever option exists.
                ref = next(iter(options.values()))
            mem = int(ref["memory_bytes"])
            bpp = float(ref["bits_per_param"])
            if bpp > 0:
                total += int(round(mem * 8.0 / bpp))
    for unit in (payload.get("singletons") or {}).values():
        options = unit.get("options") or {}
        ref = options.get("BF16") or next(iter(options.values()))
        mem = int(ref["memory_bytes"])
        bpp = float(ref["bits_per_param"])
        if bpp > 0:
            total += int(round(mem * 8.0 / bpp))
    return total


def fill_bpp(result: GlobalSolveResult, total_params: int) -> GlobalSolveResult:
    if total_params <= 0:
        return result
    return GlobalSolveResult(
        assignment=result.assignment,
        cost_total=result.cost_total,
        bits_total=result.bits_total,
        bpp=result.bits_total / float(total_params),
        lambda_used=result.lambda_used,
        per_block_costs=result.per_block_costs,
        per_block_bits=result.per_block_bits,
    )


# ---------------------------------------------------------------------------
# Kneedle extraction + per-Linear assignment expansion
# ---------------------------------------------------------------------------


def _normalise(values: Sequence[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def kneedle_pick(
    points: Sequence[tuple[float, float]],
) -> tuple[int, float, bool]:
    """Pick the kneedle (max-perpendicular-distance) point from ``(x, y)``.

    Inputs are interpreted as ``(bpp, cost)``; the function picks the index
    on the (bpp ascending, cost descending) frontier that maximises the
    perpendicular distance to the secant line connecting the endpoints.

    Returns ``(index, score, endpoint_fallback)``.
    """
    if len(points) < 3:
        # Degenerate: fall through to the middle point.
        return len(points) // 2, 0.0, True
    pts = sorted(points, key=lambda xy: xy[0])
    xs = _normalise([p[0] for p in pts])
    ys_raw = _normalise([p[1] for p in pts])
    # Cost is the y axis we want LOW, so flip for kneedle scoring.
    ys = [1.0 - v for v in ys_raw]
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denom = max(((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5, 1e-12)
    best_score = -math.inf
    best_idx = 0
    for idx, (x, y) in enumerate(zip(xs, ys)):
        score = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        if score > best_score:
            best_score = score
            best_idx = idx
    endpoint = best_idx in {0, len(pts) - 1}
    if endpoint:
        best_idx = len(pts) // 2
    # Map back to original index ordering.
    sorted_to_orig = [points.index(pt) for pt in pts]
    return sorted_to_orig[best_idx], float(best_score), bool(endpoint)


def expand_sweep_row_to_linear_assignment(
    payload: Mapping,
    unit_assignment: Mapping[str, str],
) -> dict[str, str]:
    """Convert a sweep row's per-unit assignment into per-Linear members."""
    blocks, singletons, _pairs = parse_payload(payload)
    units: list[DecisionUnit] = []
    for unit_list in blocks.values():
        units.extend(unit_list)
    units.extend(singletons)
    return expand_unit_assignment(unit_assignment, units)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block-CLADO solver")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="λ-sweep over a payload")
    sweep.add_argument("--payload", required=True)
    sweep.add_argument("--lambda-min", type=float, default=1e-12)
    sweep.add_argument("--lambda-max", type=float, default=1e-3)
    sweep.add_argument("--n-lambdas", type=int, default=41)
    sweep.add_argument("--output", required=True)

    budget = sub.add_parser("budget", help="exact-budget knapsack on a payload")
    budget.add_argument("--payload", required=True)
    budget.add_argument("--target-bpp", type=float, required=True)
    budget.add_argument("--bit-precision-bits", type=float, default=1.0)
    budget.add_argument("--output", required=True)

    knee = sub.add_parser(
        "kneedle",
        help="extract kneedle from a sweep, write per-Linear assignment JSONs",
    )
    knee.add_argument("--payload", required=True)
    knee.add_argument("--sweep", required=True)
    knee.add_argument("--output-dir", required=True)
    knee.add_argument(
        "--n-neighbors",
        type=int,
        default=2,
        help="Also write N nearest-neighbor frontier points either side",
    )

    args = parser.parse_args(argv)
    payload = load_payload(args.payload)
    block_states = build_block_states(payload)
    total_params = total_param_count(payload)

    if args.command == "sweep":
        results = lambda_sweep(
            block_states,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            n_lambdas=args.n_lambdas,
        )
        rows = [
            {
                "lambda": r.lambda_used,
                "bits_total": r.bits_total,
                "bpp": r.bits_total / float(total_params) if total_params else 0.0,
                "cost_total": r.cost_total,
                "assignment": r.assignment,
            }
            for r in results
        ]
        Path(args.output).write_text(
            json.dumps({"schema": "prismaquant.block_clado.sweep.v1",
                        "rows": rows,
                        "total_params": int(total_params)},
                       indent=2)
        )
        print(f"[block-clado] λ-sweep wrote {len(rows)} unique frontier points to {args.output}")
        return 0

    if args.command == "budget":
        bits_budget = float(args.target_bpp) * float(total_params)
        result = solve_budget(
            block_states,
            bits_budget=bits_budget,
            bit_precision_bits=args.bit_precision_bits,
        )
        if result is None:
            raise RuntimeError("infeasible budget")
        result = fill_bpp(result, total_params)
        Path(args.output).write_text(json.dumps({
            "schema": "prismaquant.block_clado.budget.v1",
            "bits_total": result.bits_total,
            "bpp": result.bpp,
            "cost_total": result.cost_total,
            "assignment": result.assignment,
        }, indent=2))
        print(f"[block-clado] budget solve achieved bpp={result.bpp:.4f}, cost={result.cost_total:.6g}")
        return 0

    if args.command == "kneedle":
        with Path(args.sweep).open("r", encoding="utf-8") as fh:
            sweep = json.load(fh)
        rows = sweep.get("rows") or []
        if not rows:
            raise RuntimeError("sweep file has no rows")

        # Restrict kneedle picking to the meaningful (positive-cost) region.
        # Negative surrogate cost lives above the trust region of the second-
        # order Taylor approximation; ignore it for the elbow.
        positive_rows = [r for r in rows if r["cost_total"] > 0.0]
        if len(positive_rows) < 3:
            positive_rows = rows
        points = [(float(r["bpp"]), float(r["cost_total"])) for r in positive_rows]
        idx, score, endpoint = kneedle_pick(points)
        chosen = positive_rows[idx]

        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        # Sort full sweep by bpp for neighbour lookup.
        sweep_sorted = sorted(rows, key=lambda r: r["bpp"])
        # Find chosen in sorted order.
        chosen_bpp = float(chosen["bpp"])
        chosen_sorted_idx = min(
            range(len(sweep_sorted)),
            key=lambda i: abs(sweep_sorted[i]["bpp"] - chosen_bpp),
        )
        n_neighbors = max(int(args.n_neighbors), 0)
        neighbour_indices = list(range(
            max(chosen_sorted_idx - n_neighbors, 0),
            min(chosen_sorted_idx + n_neighbors + 1, len(sweep_sorted)),
        ))
        wrote = []
        for sort_idx in neighbour_indices:
            row = sweep_sorted[sort_idx]
            label = (
                "kneedle"
                if sort_idx == chosen_sorted_idx
                else f"neighbor_bpp_{row['bpp']:.4f}".replace(".", "p")
            )
            assignment = expand_sweep_row_to_linear_assignment(payload, row["assignment"])
            row_payload = {
                "schema": "prismaquant.block_clado.kneedle.v1",
                "label": label,
                "bpp": float(row["bpp"]),
                "bits_total": float(row["bits_total"]),
                "surrogate_cost": float(row["cost_total"]),
                "lambda": float(row["lambda"]),
                "assignment": assignment,
            }
            out_path = out_root / f"{label}.json"
            out_path.write_text(json.dumps(row_payload, indent=2) + "\n")
            wrote.append({
                "label": label,
                "bpp": float(row["bpp"]),
                "surrogate_cost": float(row["cost_total"]),
                "path": str(out_path),
            })

        summary = {
            "schema": "prismaquant.block_clado.kneedle_summary.v1",
            "kneedle_score": float(score),
            "endpoint_fallback": bool(endpoint),
            "chosen": {
                "bpp": float(chosen["bpp"]),
                "surrogate_cost": float(chosen["cost_total"]),
                "lambda": float(chosen["lambda"]),
            },
            "candidates": wrote,
            "frontier_size_used": len(positive_rows),
            "frontier_size_total": len(rows),
        }
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"[block-clado] kneedle bpp={chosen['bpp']:.4f} "
            f"cost={chosen['cost_total']:.6g} score={score:.4f} "
            f"endpoint_fallback={endpoint}"
        )
        print(f"[block-clado] wrote {len(wrote)} candidate(s) to {out_root}")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
