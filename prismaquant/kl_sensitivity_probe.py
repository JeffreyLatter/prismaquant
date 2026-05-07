"""Lane-batched KL sensitivity probe and assignment frontier builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    check_format_applicability,
    _scan_source_dtype_manifest,
)
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    iter_quantizable_tensors,
)
from prismaquant.iterate_perturbed_allocation import (
    _assignment_digest,
    measure_assignment_kl,
)
from prismaquant.measure_adjoint_l3 import load_wikitext_calibration_windowed
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.perturbed_x_cache import (
    calibration_data_hash,
    stage_text_only_under_work_root,
)
from prismaquant.propagated_cost import KLScope, measure_lane_batched_kl_deltas


SCHEMA = "prismaquant.kl_sensitivity_probe.v1"


@dataclass(frozen=True)
class LinearTarget:
    qname: str
    shape: tuple[int, int]
    n_params: int
    pinned: bool = False


@dataclass(frozen=True)
class ProbeRow:
    qname: str
    format: str
    shape: tuple[int, int]
    bits_baseline: float
    bits_format: float
    bits_delta: float
    candidate_kl: float
    sensitivity: float
    sem: float | None = None

    def to_json(self, decision_unit: str | None = None) -> dict:
        payload = {
            "qname": self.qname,
            "format": self.format,
            "shape": list(self.shape),
            "bits_baseline": float(self.bits_baseline),
            "bits_format": float(self.bits_format),
            "bits_delta": float(self.bits_delta),
            "candidate_kl": float(self.candidate_kl),
            "sensitivity": float(self.sensitivity),
            "sem": self.sem,
        }
        if decision_unit is not None:
            payload["decision_unit"] = decision_unit
        return payload


@dataclass(frozen=True)
class UnitOption:
    unit: str
    fmt: str
    members: tuple[str, ...]
    bits_total: float
    bits_delta: float
    gain: float


@dataclass(frozen=True)
class FrontierPoint:
    budget_bits: float
    bits_total: float
    bits_delta: float
    gain: float
    predicted_kl: float
    unit_assignment: dict[str, str]
    assignment: dict[str, str]
    promotion_count: int

    def to_json(self) -> dict:
        return {
            "budget_bits": float(self.budget_bits),
            "bits_total": float(self.bits_total),
            "bits_delta": float(self.bits_delta),
            "gain": float(self.gain),
            "predicted_kl": float(self.predicted_kl),
            "unit_assignment": dict(sorted(self.unit_assignment.items())),
            "assignment": dict(sorted(self.assignment.items())),
            "assignment_hash": _assignment_digest(self.assignment),
            "promotion_count": int(self.promotion_count),
        }


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _git_commit() -> str | None:
    repo = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _json_digest(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_format_menu(formats_arg: str) -> list[fr.FormatSpec]:
    if formats_arg.strip().lower() == "registry":
        raw = fr.list_formats()
    else:
        raw = [
            fr.get_format(part.strip())
            for part in formats_arg.split(",")
            if part.strip()
        ]
    by_name: dict[str, fr.FormatSpec] = {}
    for spec in raw:
        canonical = fr.canonical_format_name(spec.name)
        by_name.setdefault(canonical, fr.get_format(canonical))
    return sorted(
        by_name.values(),
        key=lambda spec: (spec.effective_bits, spec.name),
    )


def _parse_pins(values: Sequence[str] | None) -> set[str]:
    pins: set[str] = set()
    for value in values or ():
        for part in str(value).split(","):
            part = part.strip()
            if part:
                pins.add(part)
    return pins


def _is_pinned(qname: str, pins: set[str]) -> bool:
    tokens = set(qname.split("."))
    return any(pin == qname or pin in tokens or qname.endswith(pin) for pin in pins)


def _linear_targets(model: nn.Module, pins: set[str]) -> list[LinearTarget]:
    targets: list[LinearTarget] = []
    seen: set[str] = set()
    for full_name, module, attr in iter_quantizable_tensors(model):
        if attr != "weight" or not isinstance(module, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if qname in seen:
            continue
        seen.add(qname)
        shape = tuple(int(dim) for dim in module.weight.shape)
        if len(shape) != 2:
            continue
        targets.append(
            LinearTarget(
                qname=qname,
                shape=(int(shape[0]), int(shape[1])),
                n_params=int(module.weight.numel()),
                pinned=_is_pinned(qname, pins),
            )
        )
    return sorted(targets, key=lambda target: (target.pinned, target.qname))


def _format_bits(spec: fr.FormatSpec, shape: tuple[int, int]) -> float:
    return float(8 * spec.memory_bytes_for_shape(tuple(shape)))


def _format_counts(assignment: Mapping[str, str]) -> dict[str, int]:
    counts = Counter(fr.canonical_format_name(fmt) for fmt in assignment.values())
    return dict(sorted(counts.items()))


def _decision_unit_for(profile, qname: str) -> str:
    if profile is None:
        return qname
    try:
        group = profile.fused_sibling_group(qname)
    except Exception:
        group = None
    return str(group) if group is not None else qname


def _build_unit_options(
    rows: Sequence[ProbeRow],
    targets: Sequence[LinearTarget],
    *,
    floor_format: str,
    floor_assignment: Mapping[str, str],
    profile,
) -> tuple[dict[str, list[UnitOption]], dict[str, str], dict[str, list[str]]]:
    rows_by_qname_fmt = {(row.qname, row.format): row for row in rows}
    members_by_unit: dict[str, list[str]] = defaultdict(list)
    target_by_qname = {target.qname: target for target in targets}
    for target in targets:
        if target.pinned:
            continue
        members_by_unit[_decision_unit_for(profile, target.qname)].append(target.qname)

    unit_for_qname: dict[str, str] = {}
    options_by_unit: dict[str, list[UnitOption]] = {}
    missing_by_unit: dict[str, list[str]] = {}
    for unit, members_unsorted in sorted(members_by_unit.items()):
        members = tuple(sorted(members_unsorted))
        for member in members:
            unit_for_qname[member] = unit
        formats = sorted(
            set.intersection(*[
                {
                    fmt
                    for (qname, fmt), _row in rows_by_qname_fmt.items()
                    if qname == member
                }
                for member in members
            ])
        )
        if floor_format not in formats:
            missing_by_unit[unit] = [floor_format]
            continue
        options: list[UnitOption] = []
        baseline_bits = sum(
            rows_by_qname_fmt[(member, floor_format)].bits_format
            for member in members
        )
        for fmt in formats:
            member_rows = [rows_by_qname_fmt[(member, fmt)] for member in members]
            bits_total = sum(row.bits_format for row in member_rows)
            gain = sum(row.sensitivity for row in member_rows)
            options.append(
                UnitOption(
                    unit=unit,
                    fmt=fmt,
                    members=members,
                    bits_total=float(bits_total),
                    bits_delta=float(bits_total - baseline_bits),
                    gain=float(gain),
                )
            )
        options.sort(key=lambda option: (option.bits_total, -option.gain, option.fmt))
        options_by_unit[unit] = options
    del target_by_qname, floor_assignment
    return options_by_unit, unit_for_qname, missing_by_unit


def _expand_unit_assignment(
    unit_assignment: Mapping[str, str],
    unit_members: Mapping[str, Sequence[str]],
    floor_assignment: Mapping[str, str],
) -> dict[str, str]:
    assignment = dict(floor_assignment)
    for unit, fmt in unit_assignment.items():
        for qname in unit_members.get(unit, (unit,)):
            assignment[qname] = fmt
    return assignment


def _promotion_count(
    assignment: Mapping[str, str],
    floor_assignment: Mapping[str, str],
    specs_by_name: Mapping[str, fr.FormatSpec],
) -> int:
    count = 0
    for qname, fmt in assignment.items():
        base = floor_assignment.get(qname)
        if base is None:
            continue
        fmt_c = fr.canonical_format_name(fmt)
        base_c = fr.canonical_format_name(base)
        if fmt_c == base_c:
            continue
        fmt_bits = specs_by_name[fmt_c].effective_bits
        base_bits = specs_by_name[base_c].effective_bits
        if fmt_bits >= base_bits - 1e-12:
            count += 1
    return count


def solve_multi_choice_frontier(
    options_by_unit: Mapping[str, Sequence[UnitOption]],
    *,
    floor_assignment: Mapping[str, str],
    floor_kl: float,
    constant_bits: float = 0.0,
    budget_points: int = 64,
    bit_precision_bits: float | None = None,
) -> list[FrontierPoint]:
    """Solve a multi-choice knapsack at multiple budgets.

    The DP maximizes measured gain subject to a bit budget, with one selected
    format per decision unit.  The bit axis is discretized; tests can pass
    ``bit_precision_bits=1`` for exact synthetic cases.
    """
    if not options_by_unit:
        return []
    units = list(sorted(options_by_unit))
    state_lists = [list(options_by_unit[unit]) for unit in units]
    min_bits_by_unit = [min(option.bits_total for option in opts) for opts in state_lists]
    min_bits = float(sum(min_bits_by_unit))
    max_bits = float(sum(max(option.bits_total for option in opts) for opts in state_lists))
    floor_bits = float(sum(
        next(
            option.bits_total
            for option in options_by_unit[unit]
            if option.fmt == fr.canonical_format_name(floor_assignment.get(option.members[0], ""))
        )
        for unit in units
    ))
    budget_lo = float(constant_bits + floor_bits)
    budget_hi = float(constant_bits + max_bits)
    if budget_hi <= budget_lo:
        budgets = [budget_lo]
    else:
        n_points = max(int(budget_points), 2)
        budgets = [
            budget_lo + (budget_hi - budget_lo) * idx / (n_points - 1)
            for idx in range(n_points)
        ]
    budgets = sorted(set(float(b) for b in budgets))

    dynamic_range = max(max_bits - min_bits, 0.0)
    if bit_precision_bits is None:
        bit_precision_bits = max(dynamic_range / max(int(budget_points) * 4, 1), 1.0)
    bin_width = max(float(bit_precision_bits), 1e-9)
    max_budget_delta = max(max(budgets) - constant_bits - min_bits, 0.0)
    max_bin = int(math.ceil(max_budget_delta / bin_width)) + 2

    dp: dict[int, tuple[float, int | None, int | None]] = {0: (0.0, None, None)}
    layers: list[dict[int, tuple[float, int, int]]] = []
    for level, opts in enumerate(state_lists):
        next_dp: dict[int, tuple[float, int | None, int | None]] = {}
        layer_bp: dict[int, tuple[float, int, int]] = {}
        unit_min = min_bits_by_unit[level]
        for prev_bin, (prev_gain, _prev_prev, _prev_choice) in dp.items():
            for choice_idx, option in enumerate(opts):
                delta_bits = max(option.bits_total - unit_min, 0.0)
                inc = int(math.ceil(delta_bits / bin_width - 1e-12))
                new_bin = prev_bin + inc
                if new_bin > max_bin:
                    continue
                gain = prev_gain + option.gain
                old = next_dp.get(new_bin)
                if old is None or gain > old[0] + 1e-12:
                    next_dp[new_bin] = (gain, prev_bin, choice_idx)
                    layer_bp[new_bin] = (gain, prev_bin, choice_idx)
        dp = next_dp
        layers.append(layer_bp)
    if not dp:
        return []

    unit_members = {
        unit: tuple(state_lists[idx][0].members)
        for idx, unit in enumerate(units)
    }
    specs_by_name = {
        fr.canonical_format_name(spec.name): spec
        for spec in fr.list_formats()
    }

    def reconstruct(final_bin: int, budget: float) -> FrontierPoint | None:
        choices: list[UnitOption] = []
        cur_bin = final_bin
        for level in range(len(layers) - 1, -1, -1):
            entry = layers[level].get(cur_bin)
            if entry is None:
                return None
            _gain, prev_bin, choice_idx = entry
            choices.append(state_lists[level][choice_idx])
            cur_bin = prev_bin
        choices.reverse()
        unit_assignment = {
            unit: choice.fmt for unit, choice in zip(units, choices, strict=True)
        }
        bits_units = sum(choice.bits_total for choice in choices)
        gain = sum(choice.gain for choice in choices)
        assignment = _expand_unit_assignment(
            unit_assignment,
            unit_members,
            floor_assignment,
        )
        return FrontierPoint(
            budget_bits=float(budget),
            bits_total=float(constant_bits + bits_units),
            bits_delta=float((constant_bits + bits_units) - budget_lo),
            gain=float(gain),
            predicted_kl=float(floor_kl - gain),
            unit_assignment=unit_assignment,
            assignment=assignment,
            promotion_count=_promotion_count(assignment, floor_assignment, specs_by_name),
        )

    points: list[FrontierPoint] = []
    seen_assignments: set[str] = set()
    for budget in budgets:
        budget_delta = max(float(budget) - constant_bits - min_bits, 0.0)
        budget_bin = min(int(math.floor(budget_delta / bin_width + 1e-12)), max_bin)
        feasible = [
            (gain, bin_idx)
            for bin_idx, (gain, _prev, _choice) in dp.items()
            if bin_idx <= budget_bin
        ]
        if not feasible:
            continue
        _gain, best_bin = max(feasible, key=lambda item: (item[0], -item[1]))
        point = reconstruct(best_bin, budget)
        if point is None:
            continue
        digest = _assignment_digest(point.assignment)
        if digest in seen_assignments:
            continue
        seen_assignments.add(digest)
        points.append(point)

    points.sort(key=lambda point: (point.bits_total, -point.gain))
    pareto: list[FrontierPoint] = []
    best_gain = -math.inf
    for point in points:
        if point.gain > best_gain + 1e-12 or not pareto:
            pareto.append(point)
            best_gain = point.gain
    return pareto


def choose_kneedle_point(frontier: Sequence[FrontierPoint]) -> int:
    if not frontier:
        return -1
    if len(frontier) == 1:
        return 0
    xs = [point.bits_total for point in frontier]
    ys = [point.gain for point in frontier]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        return 0
    scores = [
        ((y - ymin) / (ymax - ymin)) - ((x - xmin) / (xmax - xmin))
        for x, y in zip(xs, ys)
    ]
    return max(range(len(scores)), key=lambda idx: (scores[idx], frontier[idx].gain))


def run_probe(args: argparse.Namespace) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.time()
    work_root = Path(args.work_root or Path(args.output).parent)
    work_root.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    staged = stage_text_only_under_work_root(args.model, work_root)
    local_only = bool(args.local_files_only or Path(staged).exists())

    tokenizer = AutoTokenizer.from_pretrained(
        staged,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    calib_ids = load_wikitext_calibration_windowed(
        tokenizer,
        args.n_calib_samples,
        args.calib_seqlen,
        split=args.calib_split,
        seed=args.calib_seed,
    )
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": local_only,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    try:
        profile = detect_profile(args.model)
    except Exception:
        profile = DefaultProfile()

    pins = _parse_pins(args.pin)
    floor_format = fr.canonical_format_name(args.floor_format)
    floor_spec = fr.get_format(floor_format)
    requested_specs = _canonical_format_menu(args.formats)
    if floor_format not in {fr.canonical_format_name(spec.name) for spec in requested_specs}:
        requested_specs = sorted(
            [*requested_specs, floor_spec],
            key=lambda spec: (spec.effective_bits, spec.name),
        )
    requested_formats = [fr.canonical_format_name(spec.name) for spec in requested_specs]

    targets = _linear_targets(model, pins)
    if args.max_targets is not None:
        targets = targets[: max(int(args.max_targets), 0)]
    source_manifest = _scan_source_dtype_manifest(args.model, profile=profile)

    floor_assignment: dict[str, str] = {}
    skipped: list[dict] = []
    for target in targets:
        if target.pinned:
            floor_assignment[target.qname] = "BF16"
            continue
        verdict = check_format_applicability(
            target.shape,
            floor_spec,
            qname=target.qname,
            source_kind=source_manifest.get(target.qname),
            target_profile=args.target_profile,
        )
        if verdict.legal:
            floor_assignment[target.qname] = floor_format
        else:
            floor_assignment[target.qname] = "BF16"
            skipped.append({
                "qname": target.qname,
                "format": floor_format,
                "shape": list(target.shape),
                "reason": "floor_" + str(verdict.reason or "not_applicable"),
                "detail": verdict.detail,
            })

    print(
        f"[kl-probe] targets={len(targets)} floor={floor_format} "
        f"formats={requested_formats} kl_scope={args.kl_scope}",
        flush=True,
    )
    ref_log_probs = cache_reference_log_probs(model, calib_ids, device)
    floor_kl = measure_assignment_kl(
        model,
        floor_assignment,
        calib_ids,
        ref_log_probs,
        work_root=work_root,
        profile=profile,
        use_frozen_weight_cache=True,
        rng_seed=0,
        kl_scope=args.kl_scope,
    )
    print(f"[kl-probe] floor_kl={floor_kl:.8g}", flush=True)

    rows: list[ProbeRow] = []
    candidate_flips: list[tuple[str, str]] = []
    candidate_meta: list[tuple[LinearTarget, fr.FormatSpec, float, float]] = []
    measured_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()

    for target in targets:
        baseline_fmt = floor_assignment.get(target.qname, "BF16")
        if target.pinned:
            skipped.append({
                "qname": target.qname,
                "format": "*",
                "shape": list(target.shape),
                "reason": "pinned",
                "detail": "pinned by --pin",
            })
            continue
        if baseline_fmt != floor_format:
            continue
        baseline_bits = _format_bits(floor_spec, target.shape)
        for spec in requested_specs:
            fmt = fr.canonical_format_name(spec.name)
            verdict = check_format_applicability(
                target.shape,
                spec,
                qname=target.qname,
                source_kind=source_manifest.get(target.qname),
                target_profile=args.target_profile,
            )
            if not verdict.legal:
                skipped_counter[fmt] += 1
                skipped.append({
                    "qname": target.qname,
                    "format": fmt,
                    "shape": list(target.shape),
                    "reason": verdict.reason or "not_applicable",
                    "detail": verdict.detail,
                })
                continue
            bits_format = _format_bits(fr.get_format(fmt), target.shape)
            if fmt == floor_format:
                rows.append(
                    ProbeRow(
                        qname=target.qname,
                        format=fmt,
                        shape=target.shape,
                        bits_baseline=baseline_bits,
                        bits_format=bits_format,
                        bits_delta=0.0,
                        candidate_kl=float(floor_kl),
                        sensitivity=0.0,
                    )
                )
                measured_counter[fmt] += 1
            else:
                candidate_flips.append((target.qname, fmt))
                candidate_meta.append((target, fr.get_format(fmt), baseline_bits, bits_format))

    print(f"[kl-probe] measuring {len(candidate_flips)} candidate flips", flush=True)
    candidate_kls = measure_lane_batched_kl_deltas(
        model,
        floor_assignment,
        candidate_flips,
        calib_ids,
        ref_log_probs,
        work_root=work_root,
        max_lanes_per_batch=args.max_lanes_per_batch,
        profile=profile,
        replay_cache=None,
        kl_scope=args.kl_scope,
    )
    for (target, spec, baseline_bits, bits_format), candidate_kl in zip(
        candidate_meta,
        candidate_kls,
        strict=True,
    ):
        fmt = fr.canonical_format_name(spec.name)
        if not math.isfinite(float(candidate_kl)):
            raise RuntimeError(f"non-finite KL for {target.qname} {fmt}: {candidate_kl}")
        rows.append(
            ProbeRow(
                qname=target.qname,
                format=fmt,
                shape=target.shape,
                bits_baseline=baseline_bits,
                bits_format=bits_format,
                bits_delta=bits_format - baseline_bits,
                candidate_kl=float(candidate_kl),
                sensitivity=float(floor_kl - candidate_kl),
            )
        )
        measured_counter[fmt] += 1

    rows.sort(key=lambda row: (row.qname, row.format))
    options_by_unit, unit_for_qname, missing_units = _build_unit_options(
        rows,
        targets,
        floor_format=floor_format,
        floor_assignment=floor_assignment,
        profile=profile,
    )
    constant_bits = sum(
        _format_bits(fr.get_format(fmt), target.shape)
        for target in targets
        if target.pinned
        for fmt in [floor_assignment.get(target.qname, "BF16")]
    )
    frontier = solve_multi_choice_frontier(
        options_by_unit,
        floor_assignment=floor_assignment,
        floor_kl=float(floor_kl),
        constant_bits=constant_bits,
        budget_points=args.selection_budget_points,
        bit_precision_bits=args.selection_bit_precision,
    )
    knee_idx = choose_kneedle_point(frontier)
    chosen = frontier[knee_idx] if knee_idx >= 0 else None
    if chosen is None:
        chosen_assignment = dict(floor_assignment)
        chosen_unit_assignment: dict[str, str] = {}
        promotion_count = 0
    else:
        chosen_assignment = chosen.assignment
        chosen_unit_assignment = chosen.unit_assignment
        promotion_count = chosen.promotion_count

    row_json = [
        row.to_json(decision_unit=unit_for_qname.get(row.qname))
        for row in rows
    ]
    payload = {
        "schema": SCHEMA,
        "version": 1,
        "git_commit": _git_commit(),
        "model_path": str(Path(args.model).expanduser()),
        "staged_model_path": staged,
        "profile": getattr(profile, "name", type(profile).__name__),
        "target_profile": args.target_profile,
        "kl_scope": args.kl_scope,
        "calibration": {
            "split": args.calib_split,
            "n_calib_samples": int(args.n_calib_samples),
            "seqlen": int(args.calib_seqlen),
            "seed": int(args.calib_seed),
            "hash": calibration_data_hash(calib_ids),
        },
        "floor": {
            "format": floor_format,
            "pinned": sorted(pins),
            "assignment_hash": _assignment_digest(floor_assignment),
            "assignment_sha256": _json_digest(floor_assignment),
            "kl": float(floor_kl),
            "bits_total": float(sum(row.bits_baseline for row in rows if row.format == floor_format) + constant_bits),
            "format_counts": _format_counts(floor_assignment),
        },
        "formats": {
            "requested": requested_formats,
            "measured": dict(sorted(measured_counter.items())),
            "skipped": dict(sorted(skipped_counter.items())),
            "skip_reasons": dict(sorted(Counter(item["reason"] for item in skipped).items())),
        },
        "skipped": skipped,
        "rows": row_json,
        "selection": {
            "budget_points_requested": int(args.selection_budget_points),
            "bit_precision_bits": args.selection_bit_precision,
            "frontier": [point.to_json() for point in frontier],
            "knee_index": int(knee_idx),
            "chosen": (
                frontier[knee_idx].to_json()
                if knee_idx >= 0
                else {
                    "assignment": dict(sorted(chosen_assignment.items())),
                    "assignment_hash": _assignment_digest(chosen_assignment),
                    "promotion_count": int(promotion_count),
                }
            ),
            "chosen_unit_assignment": dict(sorted(chosen_unit_assignment.items())),
            "missing_units": missing_units,
        },
        "chosen_assignment": dict(sorted(chosen_assignment.items())),
        "diagnostics": {
            "elapsed_seconds": float(time.time() - start),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "python": platform.python_version(),
            "max_lanes_per_batch": int(args.max_lanes_per_batch),
            "target_count": int(len(targets)),
            "candidate_flip_count": int(len(candidate_flips)),
            "row_count": int(len(rows)),
            "source_manifest_entries": int(len(source_manifest)),
            "production_cache_used": False,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"[kl-probe] wrote {output}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe per-Linear KL sensitivity across legal formats",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--floor-format", default="NVFP4")
    parser.add_argument(
        "--formats",
        default="registry",
        help="'registry' or a comma-separated format list",
    )
    parser.add_argument("--pin", action="append", default=[])
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--n-calib-samples", type=int, default=128)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument(
        "--kl-scope",
        choices=["last_token", "full_sequence"],
        default="last_token",
    )
    parser.add_argument("--max-lanes-per-batch", type=int, default=32)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--target-profile",
        choices=["research", "vllm_qwen3_5_packed_moe"],
        default="research",
    )
    parser.add_argument("--selection-budget-points", type=int, default=64)
    parser.add_argument("--selection-bit-precision", type=float, default=None)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
