"""Lane-batched KL sensitivity probe and assignment frontier builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import pickle
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    check_format_applicability,
    _scan_source_dtype_manifest,
)
from prismaquant.build_rtn_cache import (
    iter_quantizable_tensors,
)
from prismaquant.iterate_perturbed_allocation import (
    _assignment_digest,
    measure_assignment_kl,
)
from prismaquant.layer_state_cache import LayerHiddenStateCache
from prismaquant.memory_management import phase_boundary_memory_cleanup
from prismaquant.measure_adjoint_l3 import load_wikitext_calibration_windowed
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.perturbed_x_cache import (
    calibration_data_hash,
    stage_text_only_under_work_root,
)
from prismaquant.propagated_cost import (
    KLScope,
    layer_depth,
    measure_lane_batched_kl_deltas,
    measure_override_set_kl,
)
from prismaquant.production_weight_cache import fill_production_weight_cache


SCHEMA = "prismaquant.kl_sensitivity_probe.v2"
PRODUCTION_CACHE_SCHEMA = "prismaquant.production_weight_cache.v1"


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
    measured_kl: float | None = None
    measured_gain: float | None = None

    def to_json(self) -> dict:
        predicted = float(self.predicted_kl)
        return {
            "budget_bits": float(self.budget_bits),
            "bits_total": float(self.bits_total),
            "bits_delta": float(self.bits_delta),
            "gain": float(self.gain),
            "predicted_gain_first_order": float(self.gain),
            "predicted_kl": predicted,
            "predicted_kl_first_order": predicted,
            "predicted_kl_clamped": max(predicted, 0.0),
            "measured_kl": (
                None if self.measured_kl is None else float(self.measured_kl)
            ),
            "measured_gain": (
                None if self.measured_gain is None else float(self.measured_gain)
            ),
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


def _parse_levers(value: str | None) -> dict[str, bool]:
    enabled = {
        part.strip()
        for part in str(value or "").split(",")
        if part.strip()
    }
    return {name: True for name in enabled}


def _normalized_production_cache_levers(value: str | None) -> dict[str, bool]:
    levers = _parse_levers(value)
    levers.setdefault("gptq", True)
    levers.setdefault("scale_sweep", True)
    levers.setdefault("awq", False)
    levers.setdefault("awq_round", False)
    return dict(sorted(levers.items()))


def _production_cache_entries_digest(cache: object) -> str:
    entries: list[dict[str, object]] = []
    weights = getattr(cache, "weights", {}) or {}
    cache_dir = getattr(cache, "cache_dir", None)
    for key, value in sorted(weights.items()):
        qname, fmt = key
        item: dict[str, object] = {
            "qname": str(qname),
            "format": str(fmt),
        }
        if isinstance(value, torch.Tensor):
            item.update({
                "storage": "tensor",
                "dtype": str(value.dtype).replace("torch.", ""),
                "shape": [int(dim) for dim in value.shape],
                "numel": int(value.numel()),
            })
        else:
            path = str(value)
            resolved = Path(path)
            if cache_dir and not resolved.is_absolute():
                resolved = Path(cache_dir) / path
            size = resolved.stat().st_size if resolved.is_file() else None
            item.update({
                "storage": "file",
                "path": path,
                "bytes": None if size is None else int(size),
            })
        entries.append(item)
    return _json_digest({"entries": entries})


def _production_cache_expected_metadata(
    args: argparse.Namespace,
    calib_ids: torch.Tensor,
    qnames: Sequence[str],
    formats: Sequence[str],
    source_manifest: Mapping[str, str | None],
) -> dict[str, object]:
    qname_list = sorted(str(q) for q in qnames)
    fmt_list = sorted(fr.canonical_format_name(fmt) for fmt in formats)
    relevant_source = {
        qname: source_manifest.get(qname)
        for qname in qname_list
    }
    payload: dict[str, object] = {
        "schema": PRODUCTION_CACHE_SCHEMA,
        "model_path": str(Path(args.model).expanduser()),
        "target_profile": str(args.target_profile),
        "calibration": {
            "split": str(args.calib_split),
            "n_calib_samples": int(calib_ids.size(0)),
            "seqlen": int(calib_ids.size(1)) if calib_ids.dim() >= 2 else None,
            "seed": int(args.calib_seed),
            "hash": calibration_data_hash(calib_ids),
        },
        "formats": fmt_list,
        "qname_count": int(len(qname_list)),
        "qnames_sha256": _json_digest({"qnames": qname_list}),
        "source_manifest_sha256": _json_digest(relevant_source),
        "levers": _normalized_production_cache_levers(
            args.production_cache_levers
        ),
        "max_act_rows": int(args.production_cache_max_act_rows),
    }
    payload["identity_sha256"] = _json_digest(payload)
    return payload


def _attach_production_cache_metadata(
    cache: object,
    expected: Mapping[str, object],
) -> dict[str, object]:
    metadata = dict(expected)
    metadata["entries_sha256"] = _production_cache_entries_digest(cache)
    metadata["manifest_sha256"] = _json_digest({
        "identity_sha256": metadata.get("identity_sha256"),
        "entries_sha256": metadata.get("entries_sha256"),
    })
    setattr(cache, "metadata", metadata)
    return metadata


def _production_cache_metadata_diag(
    metadata: Mapping[str, object] | None,
    expected: Mapping[str, object],
    *,
    status: str,
    validated: bool,
) -> dict[str, object]:
    return {
        "status": status,
        "validated": bool(validated),
        "schema": None if metadata is None else metadata.get("schema"),
        "cache_identity_sha256": (
            None if metadata is None else metadata.get("identity_sha256")
        ),
        "expected_cache_identity_sha256": expected.get("identity_sha256"),
        "entries_sha256": (
            None if metadata is None else metadata.get("entries_sha256")
        ),
        "manifest_sha256": (
            None if metadata is None else metadata.get("manifest_sha256")
        ),
    }


def _validate_production_cache_metadata(
    cache: object,
    expected: Mapping[str, object],
) -> dict[str, object]:
    metadata = getattr(cache, "metadata", None)
    if not isinstance(metadata, Mapping):
        return _production_cache_metadata_diag(
            None,
            expected,
            status="legacy_missing",
            validated=False,
        )
    actual_identity = metadata.get("identity_sha256")
    expected_identity = expected.get("identity_sha256")
    if actual_identity != expected_identity:
        raise RuntimeError(
            "production weight cache identity mismatch; "
            f"expected={expected_identity} actual={actual_identity}"
        )
    actual_entries = metadata.get("entries_sha256")
    computed_entries = _production_cache_entries_digest(cache)
    if actual_entries != computed_entries:
        raise RuntimeError(
            "production weight cache entry manifest mismatch; "
            f"expected={actual_entries} actual={computed_entries}"
        )
    return _production_cache_metadata_diag(
        metadata,
        expected,
        status="validated",
        validated=True,
    )


def _production_cache_formats(
    requested_formats: Sequence[str],
    floor_format: str,
) -> list[str]:
    formats = {
        fr.canonical_format_name(fmt)
        for fmt in [*requested_formats, floor_format]
    }
    return sorted(
        fmt
        for fmt in formats
        if fmt not in {"BF16", "MXFP8", "MXFP8_E4M3"}
    )


def _production_cache_qnames(
    targets: Sequence[LinearTarget],
    formats: Sequence[str],
    *,
    source_manifest: Mapping[str, str | None],
    target_profile: str,
) -> list[str]:
    qnames: list[str] = []
    for target in targets:
        if target.pinned:
            continue
        for fmt in formats:
            spec = fr.get_format(fmt)
            verdict = check_format_applicability(
                target.shape,
                spec,
                qname=target.qname,
                source_kind=source_manifest.get(target.qname),
                target_profile=target_profile,
            )
            if verdict.legal:
                qnames.append(target.qname)
                break
    return sorted(set(qnames))


def _prepare_production_weight_cache(
    args: argparse.Namespace,
    model: nn.Module,
    calib_ids: torch.Tensor,
    targets: Sequence[LinearTarget],
    requested_formats: Sequence[str],
    floor_format: str,
    source_manifest: Mapping[str, str | None],
    *,
    work_root: Path,
) -> tuple[object | None, dict]:
    diag = {
        "mode": str(args.candidate_recipe),
        "path": None,
        "cache_dir": None,
        "formats": [],
        "qname_count": 0,
        "entries": 0,
        "levers": {},
        "lru_gb": None,
        "prefetch": str(args.production_cache_prefetch),
        "built": False,
        "expected_cache_identity_sha256": None,
        "metadata": {"status": "not_used", "validated": False},
    }
    if args.candidate_recipe == "raw":
        return None, diag

    formats = _production_cache_formats(requested_formats, floor_format)
    diag["formats"] = list(formats)
    if not formats:
        return None, diag

    cache_qnames = _production_cache_qnames(
        targets,
        formats,
        source_manifest=source_manifest,
        target_profile=args.target_profile,
    )
    diag["qname_count"] = int(len(cache_qnames))
    if not cache_qnames:
        return None, diag
    expected_metadata = _production_cache_expected_metadata(
        args,
        calib_ids,
        cache_qnames,
        formats,
        source_manifest,
    )
    diag["expected_cache_identity_sha256"] = expected_metadata["identity_sha256"]

    cache = None
    cache_path: Path | None = None
    if args.production_weight_cache:
        cache_path = Path(args.production_weight_cache)
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        print(
            f"[kl-probe] loaded production weight cache "
            f"{cache_path} entries={len(cache)}",
            flush=True,
        )
    else:
        cache_dir = (
            Path(args.production_cache_dir)
            if args.production_cache_dir
            else work_root / "production_weight_cache"
        )
        cache_path = (
            Path(args.production_cache_output)
            if args.production_cache_output
            else work_root / "production_weight_cache.pkl"
        )
        print(
            f"[kl-probe] building production candidate cache "
            f"qnames={len(cache_qnames)} formats={formats} dir={cache_dir}",
            flush=True,
        )
        cache = fill_production_weight_cache(
            model,
            calib_ids,
            cache_qnames,
            formats=formats,
            levers=_parse_levers(args.production_cache_levers),
            max_act_rows=int(args.production_cache_max_act_rows),
            cache_dir=cache_dir,
        )
        metadata = _attach_production_cache_metadata(cache, expected_metadata)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        diag["built"] = True
        diag["metadata"] = _production_cache_metadata_diag(
            metadata,
            expected_metadata,
            status="built",
            validated=True,
        )
        print(
            f"[kl-probe] wrote production weight cache manifest {cache_path}",
            flush=True,
        )

    if cache is None:
        return None, diag

    if args.production_cache_dir_override:
        cache.relocate(args.production_cache_dir_override)
    setattr(cache, "_prismaquant_prefetch_policy", args.production_cache_prefetch)
    if args.production_weight_cache:
        diag["metadata"] = _validate_production_cache_metadata(
            cache,
            expected_metadata,
        )
        if not diag["metadata"].get("validated"):
            print(
                "[kl-probe] WARNING: loaded production cache has no v1 "
                "metadata; shard presence and coverage will be checked, but "
                "calibration/source/lever identity is unverified",
                flush=True,
            )

    cache_dir_value = getattr(cache, "cache_dir", None)
    diag["path"] = str(cache_path) if cache_path is not None else None
    diag["cache_dir"] = str(cache_dir_value or "")
    diag["entries"] = int(len(cache))
    diag["levers"] = dict(getattr(cache, "levers", {}) or {})
    if cache_dir_value:
        verify = cache.verify_files()
        missing = verify.get("missing", [])
        if missing:
            raise RuntimeError(
                "production weight cache references missing shard files; "
                f"missing={len(missing)} sample={missing[:5]}"
            )

    if cache_qnames and not args.allow_incomplete_production_cache:
        cache.validate_coverage(cache_qnames, formats)

    lru_gb = float(args.production_cache_lru_gb)
    if lru_gb > 0 and hasattr(cache, "enable_lru"):
        cache.enable_lru(int(lru_gb * 1024 ** 3))
        diag["lru_gb"] = lru_gb
    if args.production_cache_prefetch == "all" and hasattr(cache, "prefetch"):
        cache.prefetch()
    return cache, diag


def _decision_unit_for(profile, qname: str) -> str:
    if profile is None:
        return qname
    try:
        group = profile.fused_sibling_group(qname)
    except Exception:
        group = None
    return str(group) if group is not None else qname


def _members_by_decision_unit(
    targets: Sequence[LinearTarget],
    profile,
    *,
    include_pinned: bool = False,
) -> dict[str, tuple[str, ...]]:
    members_by_unit: dict[str, list[str]] = defaultdict(list)
    for target in targets:
        if target.pinned and not include_pinned:
            continue
        members_by_unit[_decision_unit_for(profile, target.qname)].append(
            target.qname
        )
    return {
        unit: tuple(sorted(members))
        for unit, members in sorted(members_by_unit.items())
    }


def _format_effective_bits(fmt: str) -> float:
    return float(fr.get_format(fr.canonical_format_name(fmt)).effective_bits)


def _highest_precision_format(formats: Sequence[str]) -> str:
    if not formats:
        return "BF16"
    return max(
        (fr.canonical_format_name(fmt) for fmt in formats),
        key=lambda fmt: (_format_effective_bits(fmt), fmt),
    )


def _coerce_assignment_to_fused_units(
    assignment: Mapping[str, str],
    targets: Sequence[LinearTarget],
    profile,
) -> dict[str, str]:
    coherent = {
        str(qname): fr.canonical_format_name(fmt)
        for qname, fmt in assignment.items()
    }
    for _unit, members in _members_by_decision_unit(
        targets, profile, include_pinned=True,
    ).items():
        present = [member for member in members if member in coherent]
        if len(present) <= 1:
            continue
        fmts = [coherent[member] for member in present]
        if len(set(fmts)) <= 1:
            continue
        chosen = _highest_precision_format(fmts)
        for member in present:
            coherent[member] = chosen
    return coherent


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


def _fused_assignment_violations(
    assignment: Mapping[str, str],
    targets: Sequence[LinearTarget],
    profile,
) -> dict[str, dict[str, str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for target in targets:
        groups[_decision_unit_for(profile, target.qname)].append(target.qname)

    violations: dict[str, dict[str, str]] = {}
    for unit, members in sorted(groups.items()):
        if len(members) <= 1:
            continue
        member_formats = {
            member: fr.canonical_format_name(assignment.get(member, "BF16"))
            for member in sorted(members)
            if member in assignment
        }
        if len(set(member_formats.values())) > 1:
            violations[unit] = member_formats
    return violations


def _assert_fused_assignment_coherent(
    assignment: Mapping[str, str],
    targets: Sequence[LinearTarget],
    profile,
    *,
    label: str,
) -> None:
    violations = _fused_assignment_violations(assignment, targets, profile)
    if violations:
        sample = dict(list(violations.items())[:5])
        raise RuntimeError(
            f"{label} violates vLLM packed-module format coherence for "
            f"{len(violations)} fused decision unit(s).  Sample: {sample}"
        )


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


def choose_kneedle_point(
    frontier: Sequence[FrontierPoint],
    *,
    use_measured: bool = False,
) -> int:
    if not frontier:
        return -1
    if len(frontier) == 1:
        return 0
    xs = [point.bits_total for point in frontier]
    if use_measured:
        measured = [point.measured_gain for point in frontier]
        if any(value is None or not math.isfinite(float(value)) for value in measured):
            raise ValueError("use_measured=True requires finite measured_gain values")
        ys = [float(value) for value in measured if value is not None]
    else:
        ys = [point.gain for point in frontier]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        return 0
    scores = [
        ((y - ymin) / (ymax - ymin)) - ((x - xmin) / (xmax - xmin))
        for x, y in zip(xs, ys)
    ]
    return max(range(len(scores)), key=lambda idx: (scores[idx], ys[idx]))


def _model_hidden_size(model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _replay_cache_window_size(
    model: nn.Module,
    calib_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
    window_arg: str,
    max_cache_gb: float,
    max_lanes_per_batch: int,
    max_effective_batch: int,
) -> int:
    total = int(calib_ids.size(0))
    if total <= 0:
        return 0
    raw = str(window_arg or "auto").strip().lower()
    if raw not in {"", "auto"}:
        return max(min(int(raw), total), 1)

    # The hidden-state cache can be much smaller than the transient attention
    # workspace needed to populate/replay it.  Bound auto windows by the
    # effective replay batch (rows × lanes), which is the relevant OOM axis on
    # UMA systems.
    lane_cap = max(
        int(max_effective_batch) // max(int(max_lanes_per_batch), 1),
        1,
    )
    hidden_size = _model_hidden_size(model)
    if hidden_size is None or calib_ids.dim() < 2:
        return min(total, lane_cap)
    try:
        layer_count = len(LayerHiddenStateCache(model).layers)
    except Exception:
        return 0
    bytes_per_row = (
        int(calib_ids.size(1))
        * int(hidden_size)
        * int(torch.empty((), dtype=dtype).element_size())
        * max(int(layer_count), 1)
    )
    if bytes_per_row <= 0:
        return min(total, lane_cap)
    budget = max(float(max_cache_gb), 0.0) * (1024 ** 3)
    if budget <= 0:
        return min(total, 1)
    cache_cap = max(int(budget // bytes_per_row), 1)
    return max(min(total, cache_cap, lane_cap), 1)


def _candidate_measurement_order(
    candidate_flips: Sequence[tuple[str, str]],
) -> list[int]:
    return sorted(
        range(len(candidate_flips)),
        key=lambda idx: (
            layer_depth(candidate_flips[idx][0]) is None,
            (
                layer_depth(candidate_flips[idx][0])
                if layer_depth(candidate_flips[idx][0]) is not None
                else 1_000_000
            ),
            str(candidate_flips[idx][0]),
            str(candidate_flips[idx][1]),
        ),
    )


def _override_measurement_order(
    candidate_overrides: Sequence[Mapping[str, str]],
) -> list[int]:
    def _key(idx: int) -> tuple:
        override = candidate_overrides[idx]
        depths = [layer_depth(name) for name in override]
        depth = (
            min(d for d in depths if d is not None)
            if depths and all(d is not None for d in depths)
            else None
        )
        return (
            depth is None,
            depth if depth is not None else 1_000_000,
            tuple(sorted((str(name), str(fmt)) for name, fmt in override.items())),
        )

    return sorted(range(len(candidate_overrides)), key=_key)


def _weight_session_mode_enabled(
    args: argparse.Namespace,
    production_weight_cache,
) -> bool:
    mode = str(getattr(args, "weight_session", "auto")).strip().lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return production_weight_cache is not None


@contextmanager
def _external_weight_management(enabled: bool):
    previous = os.environ.get("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT")
    if enabled:
        os.environ["PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT"] = "1"
    try:
        yield
    finally:
        if enabled:
            if previous is None:
                os.environ.pop("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", None)
            else:
                os.environ["PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT"] = previous


def _populate_replay_cache(
    model: nn.Module,
    assignment: Mapping[str, str],
    calib_window: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_activation_quant: bool,
    production_weight_cache=None,
) -> LayerHiddenStateCache | None:
    try:
        cache = LayerHiddenStateCache(model)
    except (AttributeError, TypeError, ValueError):
        return None
    rng_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        rng_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(0)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        cache.populate(
            assignment,
            calib_window,
            device=str(device),
            dtype=dtype,
            include_activation_quant=include_activation_quant,
            production_weight_cache=production_weight_cache,
        )
    return cache


def measure_candidate_overrides(
    model: nn.Module,
    floor_assignment: Mapping[str, str],
    candidate_overrides: list[Mapping[str, str]],
    calib_ids: torch.Tensor,
    ref_log_probs: Sequence[torch.Tensor],
    *,
    work_root: Path,
    profile,
    kl_scope: KLScope,
    max_lanes_per_batch: int,
    calib_microbatch_size: int,
    include_activation_quant: bool,
    use_cuda_graphs: bool,
    use_tail_replay: bool,
    replay_cache_window: str,
    replay_cache_max_gb: float,
    replay_cache_max_effective_batch: int,
    dtype: torch.dtype,
    production_weight_cache=None,
    source_weight_resolver=None,
) -> list[float]:
    if not candidate_overrides:
        return []

    order = _override_measurement_order(candidate_overrides)
    ordered_overrides = [candidate_overrides[idx] for idx in order]
    inverse = [0] * len(order)
    for ordered_idx, original_idx in enumerate(order):
        inverse[original_idx] = ordered_idx

    if not use_tail_replay:
        ordered_values = measure_override_set_kl(
            model,
            floor_assignment,
            ordered_overrides,
            calib_ids,
            list(ref_log_probs),
            work_root=work_root,
            max_lanes_per_batch=max_lanes_per_batch,
            profile=profile,
            replay_cache=None,
            kl_scope=kl_scope,
            calib_microbatch_size=calib_microbatch_size,
            include_activation_quant=include_activation_quant,
            use_cuda_graphs=use_cuda_graphs,
            production_weight_cache=production_weight_cache,
            source_weight_resolver=source_weight_resolver,
        )
        return [ordered_values[inverse[idx]] for idx in range(len(inverse))]

    device = next(model.parameters()).device
    window = _replay_cache_window_size(
        model,
        calib_ids,
        dtype=dtype,
        window_arg=replay_cache_window,
        max_cache_gb=replay_cache_max_gb,
        max_lanes_per_batch=max_lanes_per_batch,
        max_effective_batch=replay_cache_max_effective_batch,
    )
    if window <= 0:
        print(
            "[kl-probe] tail replay unavailable for this model; falling back "
            "to full-forward unit probing",
            flush=True,
        )
        return measure_candidate_overrides(
            model,
            floor_assignment,
            candidate_overrides,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
            kl_scope=kl_scope,
            max_lanes_per_batch=max_lanes_per_batch,
            calib_microbatch_size=calib_microbatch_size,
            include_activation_quant=include_activation_quant,
            use_cuda_graphs=use_cuda_graphs,
            use_tail_replay=False,
            replay_cache_window=replay_cache_window,
            replay_cache_max_gb=replay_cache_max_gb,
            replay_cache_max_effective_batch=replay_cache_max_effective_batch,
            dtype=dtype,
            production_weight_cache=production_weight_cache,
            source_weight_resolver=source_weight_resolver,
        )

    print(
        f"[kl-probe] tail replay enabled for unit candidates "
        f"(window={window}, max_cache_gb={float(replay_cache_max_gb):.1f})",
        flush=True,
    )
    totals = [0.0 for _ in ordered_overrides]
    count = 0
    total_windows = int(math.ceil(int(calib_ids.size(0)) / float(window)))
    for window_idx, start in enumerate(range(0, int(calib_ids.size(0)), window), start=1):
        end = min(start + window, int(calib_ids.size(0)))
        calib_window = calib_ids[start:end]
        ref_window = list(ref_log_probs[start:end])
        print(
            f"[kl-probe] populating tail replay cache "
            f"{window_idx}/{total_windows} rows={end - start}",
            flush=True,
        )
        replay_cache = _populate_replay_cache(
            model,
            floor_assignment,
            calib_window,
            device=device,
            dtype=dtype,
            include_activation_quant=include_activation_quant,
            production_weight_cache=production_weight_cache,
        )
        if replay_cache is None:
            print(
                "[kl-probe] tail replay cache build failed; falling back "
                "to full-forward unit probing",
                flush=True,
            )
            return measure_candidate_overrides(
                model,
                floor_assignment,
                candidate_overrides,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
                kl_scope=kl_scope,
                max_lanes_per_batch=max_lanes_per_batch,
                calib_microbatch_size=calib_microbatch_size,
                include_activation_quant=include_activation_quant,
                use_cuda_graphs=use_cuda_graphs,
                use_tail_replay=False,
                replay_cache_window=replay_cache_window,
                replay_cache_max_gb=replay_cache_max_gb,
                replay_cache_max_effective_batch=replay_cache_max_effective_batch,
                dtype=dtype,
                production_weight_cache=production_weight_cache,
                source_weight_resolver=source_weight_resolver,
            )
        try:
            values = measure_override_set_kl(
                model,
                floor_assignment,
                ordered_overrides,
                calib_window,
                ref_window,
                work_root=work_root,
                max_lanes_per_batch=max_lanes_per_batch,
                profile=profile,
                replay_cache=replay_cache,
                kl_scope=kl_scope,
                calib_microbatch_size=1,
                include_activation_quant=include_activation_quant,
                use_cuda_graphs=use_cuda_graphs,
                use_replay_cache=True,
                production_weight_cache=production_weight_cache,
                source_weight_resolver=source_weight_resolver,
            )
        finally:
            replay_cache.invalidate()
            del replay_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = int(end - start)
        count += rows
        for idx, value in enumerate(values):
            totals[idx] += float(value) * rows
    ordered_values = [total / max(count, 1) for total in totals]
    return [ordered_values[inverse[idx]] for idx in range(len(inverse))]


def measure_candidate_flips(
    model: nn.Module,
    floor_assignment: Mapping[str, str],
    candidate_flips: list[tuple[str, str]],
    calib_ids: torch.Tensor,
    ref_log_probs: Sequence[torch.Tensor],
    *,
    work_root: Path,
    profile,
    kl_scope: KLScope,
    max_lanes_per_batch: int,
    calib_microbatch_size: int,
    include_activation_quant: bool,
    use_cuda_graphs: bool,
    use_tail_replay: bool,
    replay_cache_window: str,
    replay_cache_max_gb: float,
    replay_cache_max_effective_batch: int,
    dtype: torch.dtype,
    production_weight_cache=None,
    source_weight_resolver=None,
) -> list[float]:
    if not candidate_flips:
        return []

    order = _candidate_measurement_order(candidate_flips)
    ordered_flips = [candidate_flips[idx] for idx in order]
    inverse = [0] * len(order)
    for ordered_idx, original_idx in enumerate(order):
        inverse[original_idx] = ordered_idx

    if not use_tail_replay:
        ordered_values = measure_lane_batched_kl_deltas(
            model,
            floor_assignment,
            ordered_flips,
            calib_ids,
            list(ref_log_probs),
            work_root=work_root,
            max_lanes_per_batch=max_lanes_per_batch,
            profile=profile,
            replay_cache=None,
            kl_scope=kl_scope,
            calib_microbatch_size=calib_microbatch_size,
            include_activation_quant=include_activation_quant,
            use_cuda_graphs=use_cuda_graphs,
            production_weight_cache=production_weight_cache,
            source_weight_resolver=source_weight_resolver,
        )
        return [ordered_values[inverse[idx]] for idx in range(len(inverse))]

    device = next(model.parameters()).device
    window = _replay_cache_window_size(
        model,
        calib_ids,
        dtype=dtype,
        window_arg=replay_cache_window,
        max_cache_gb=replay_cache_max_gb,
        max_lanes_per_batch=max_lanes_per_batch,
        max_effective_batch=replay_cache_max_effective_batch,
    )
    if window <= 0:
        print(
            "[kl-probe] tail replay unavailable for this model; falling back "
            "to full-forward candidate probing",
            flush=True,
        )
        return measure_candidate_flips(
            model,
            floor_assignment,
            candidate_flips,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
            kl_scope=kl_scope,
            max_lanes_per_batch=max_lanes_per_batch,
            calib_microbatch_size=calib_microbatch_size,
            include_activation_quant=include_activation_quant,
            use_cuda_graphs=use_cuda_graphs,
            use_tail_replay=False,
            replay_cache_window=replay_cache_window,
            replay_cache_max_gb=replay_cache_max_gb,
            replay_cache_max_effective_batch=replay_cache_max_effective_batch,
            dtype=dtype,
            production_weight_cache=production_weight_cache,
            source_weight_resolver=source_weight_resolver,
        )

    print(
        f"[kl-probe] tail replay enabled "
        f"(window={window}, max_cache_gb={float(replay_cache_max_gb):.1f})",
        flush=True,
    )
    totals = [0.0 for _ in ordered_flips]
    count = 0
    total_windows = int(math.ceil(int(calib_ids.size(0)) / float(window)))
    for window_idx, start in enumerate(range(0, int(calib_ids.size(0)), window), start=1):
        end = min(start + window, int(calib_ids.size(0)))
        calib_window = calib_ids[start:end]
        ref_window = list(ref_log_probs[start:end])
        print(
            f"[kl-probe] populating tail replay cache "
            f"{window_idx}/{total_windows} rows={end - start}",
            flush=True,
        )
        replay_cache = _populate_replay_cache(
            model,
            floor_assignment,
            calib_window,
            device=device,
            dtype=dtype,
            include_activation_quant=include_activation_quant,
            production_weight_cache=production_weight_cache,
        )
        if replay_cache is None:
            print(
                "[kl-probe] tail replay cache build failed; falling back "
                "to full-forward candidate probing",
                flush=True,
            )
            return measure_candidate_flips(
                model,
                floor_assignment,
                candidate_flips,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
                kl_scope=kl_scope,
                max_lanes_per_batch=max_lanes_per_batch,
                calib_microbatch_size=calib_microbatch_size,
                include_activation_quant=include_activation_quant,
                use_cuda_graphs=use_cuda_graphs,
                use_tail_replay=False,
                replay_cache_window=replay_cache_window,
                replay_cache_max_gb=replay_cache_max_gb,
                replay_cache_max_effective_batch=replay_cache_max_effective_batch,
                dtype=dtype,
                production_weight_cache=production_weight_cache,
                source_weight_resolver=source_weight_resolver,
            )
        try:
            values = measure_lane_batched_kl_deltas(
                model,
                floor_assignment,
                ordered_flips,
                calib_window,
                ref_window,
                work_root=work_root,
                max_lanes_per_batch=max_lanes_per_batch,
                profile=profile,
                replay_cache=replay_cache,
                kl_scope=kl_scope,
                calib_microbatch_size=1,
                include_activation_quant=include_activation_quant,
                use_cuda_graphs=use_cuda_graphs,
                use_replay_cache=True,
                production_weight_cache=production_weight_cache,
                source_weight_resolver=source_weight_resolver,
            )
        finally:
            replay_cache.invalidate()
            del replay_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = int(end - start)
        count += rows
        for idx, value in enumerate(values):
            totals[idx] += float(value) * rows
    ordered_values = [total / max(count, 1) for total in totals]
    return [ordered_values[inverse[idx]] for idx in range(len(inverse))]


def measure_frontier_points(
    model: nn.Module,
    frontier: Sequence[FrontierPoint],
    calib_ids: torch.Tensor,
    ref_log_probs: Sequence[torch.Tensor],
    *,
    floor_kl: float,
    floor_assignment: Mapping[str, str],
    work_root: Path,
    profile,
    kl_scope: KLScope,
    include_activation_quant: bool = True,
    use_cuda_graphs: bool = False,
    production_weight_cache=None,
    weight_session=None,
) -> list[FrontierPoint]:
    measured: list[FrontierPoint] = []
    total = len(frontier)
    floor_digest = _assignment_digest(floor_assignment)
    try:
        for idx, point in enumerate(frontier, start=1):
            print(
                f"[kl-probe] measuring frontier point {idx}/{total} "
                f"promotions={point.promotion_count}",
                flush=True,
            )
            if _assignment_digest(point.assignment) == floor_digest:
                if weight_session is not None:
                    weight_session.apply_assignment(floor_assignment)
                measured.append(
                    replace(
                        point,
                        measured_kl=float(floor_kl),
                        measured_gain=0.0,
                    )
                )
                continue
            if weight_session is not None:
                changed = weight_session.apply_assignment(point.assignment)
                print(
                    f"[kl-probe] frontier point {idx}/{total} "
                    f"materialized_changes={changed}",
                    flush=True,
                )
            with _external_weight_management(weight_session is not None):
                kl = measure_assignment_kl(
                    model,
                    point.assignment,
                    calib_ids,
                    ref_log_probs,
                    work_root=work_root,
                    profile=profile,
                    use_frozen_weight_cache=weight_session is None,
                    rng_seed=0,
                    kl_scope=kl_scope,
                    include_activation_quant=include_activation_quant,
                    stream_ref_log_probs=kl_scope == "full_sequence",
                    use_cuda_graphs=use_cuda_graphs,
                    production_weight_cache=production_weight_cache,
                )
            measured.append(
                replace(
                    point,
                    measured_kl=float(kl),
                    measured_gain=float(floor_kl - kl),
                )
            )
    finally:
        if weight_session is not None:
            weight_session.apply_assignment(floor_assignment)
    return measured


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
    if str(args.attn_implementation).lower() != "auto":
        load_kwargs["attn_implementation"] = str(args.attn_implementation)
    if device.type == "cuda":
        load_kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    print(f"[kl-probe] attention_implementation={attn_impl}", flush=True)

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
    if args.target_offset:
        targets = targets[max(int(args.target_offset), 0):]
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
    floor_assignment = _coerce_assignment_to_fused_units(
        floor_assignment,
        targets,
        profile,
    )
    production_weight_cache, production_cache_diag = _prepare_production_weight_cache(
        args,
        model,
        calib_ids,
        targets,
        requested_formats,
        floor_format,
        source_manifest,
        work_root=work_root,
    )

    print(
        f"[kl-probe] targets={len(targets)} floor={floor_format} "
        f"formats={requested_formats} kl_scope={args.kl_scope} "
        f"candidate_recipe={args.candidate_recipe}",
        flush=True,
    )
    # Build the BF16-teacher logprob cache.  On GB10/UMA, CPU tensors and CUDA
    # tensors draw from the same 128 GB physical pool, so "move it to CPU" is
    # not a capacity solution; it only changes allocator locality.  The
    # vocab-sized fp32 logprob tensor for one (n_samples, seqlen) pair is
    # n_samples × seqlen × |V| × 4 bytes — for Qwen at 128×2048 / 152K vocab
    # that is ~160 GB, larger than the shared pool either way.
    #
    # For ``kl_scope == "last_token"``: store only [:, -1:, :] per sample,
    # which is ~78 MB total for the above example.  For ``full_sequence``,
    # this exact in-memory cache is deliberately not the scalable design; the
    # next version should use a bounded LRU/prefetch teacher cache or streamed
    # teacher recompute.
    print(f"[kl-probe] caching reference logprobs (kl_scope={args.kl_scope})", flush=True)
    ref_log_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(calib_ids.size(0)):
            batch = calib_ids[i:i + 1].to(device)
            logits = model(batch).logits
            if args.kl_scope == "last_token":
                logits = logits[:, -1:, :]
            # Stage outside the CUDA allocator during the build loop.  This
            # does not reduce total UMA usage; it only keeps allocator pressure
            # lower for the small last-token cache case.
            ref_log_probs.append(
                torch.nn.functional.log_softmax(logits.float(), dim=-1).cpu()
            )
            del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device.type == "cuda":
        if args.kl_scope == "last_token":
            ref_log_probs = [t.to(device) for t in ref_log_probs]
        else:
            # Keeping this as CPU tensors does not solve memory on UMA.  It is
            # only a temporary compatibility path for small full-sequence
            # smokes until a bounded LRU/prefetch teacher path lands.
            print(
                "[kl-probe] WARNING: full_sequence teacher cache still scales "
                "as n_calib × seqlen × vocab × fp32 in the shared UMA pool. "
                "Use smaller calibration until bounded LRU/prefetch teacher "
                "cache or streamed recompute lands.",
                flush=True,
            )

    weight_session = None
    weight_session_diag: dict[str, object] = {
        "enabled": False,
        "mode": str(getattr(args, "weight_session", "auto")),
    }
    if _weight_session_mode_enabled(args, production_weight_cache):
        from prismaquant.weight_session import WeightSession

        snapshot_dir = work_root / "weight_session_snapshots"
        print(
            "[kl-probe] initializing WeightSession for floor materialization "
            f"(snapshots={snapshot_dir})",
            flush=True,
        )
        weight_session = WeightSession(
            model,
            production_weight_cache=production_weight_cache,
            snapshot_dir=str(snapshot_dir),
        )
        weight_session.initialize(floor_assignment, [])
        weight_session_diag = {
            "enabled": True,
            "mode": str(getattr(args, "weight_session", "auto")),
            "snapshot_dir": str(snapshot_dir),
            "diagnostics": weight_session.diagnostics(),
        }
        print(
            "[kl-probe] WeightSession initialized "
            f"{weight_session.diagnostics()}",
            flush=True,
        )
        phase_boundary_memory_cleanup("after_weight_session_initialize")

    source_weight_resolver = (
        weight_session.format_weight if weight_session is not None else None
    )
    with _external_weight_management(weight_session is not None):
        floor_kl = measure_assignment_kl(
            model,
            floor_assignment,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
            use_frozen_weight_cache=weight_session is None,
            rng_seed=0,
            kl_scope=args.kl_scope,
            include_activation_quant=not args.no_activation_quant,
            stream_ref_log_probs=args.kl_scope == "full_sequence",
            use_cuda_graphs=bool(args.enable_cuda_graphs),
            production_weight_cache=production_weight_cache,
        )
    print(f"[kl-probe] floor_kl={floor_kl:.8g}", flush=True)
    phase_boundary_memory_cleanup("after_floor_kl")

    rows: list[ProbeRow] = []
    candidate_overrides: list[dict[str, str]] = []
    candidate_meta: list[
        tuple[str, tuple[str, ...], tuple[LinearTarget, ...], str, float, float]
    ] = []
    measured_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()
    target_by_qname = {target.qname: target for target in targets}
    members_by_unit = _members_by_decision_unit(targets, profile)
    unit_for_qname: dict[str, str] = {}
    options_by_unit: dict[str, list[UnitOption]] = {}
    missing_units: dict[str, list[str]] = {}

    for target in targets:
        if target.pinned:
            skipped.append({
                "qname": target.qname,
                "format": "*",
                "shape": list(target.shape),
                "reason": "pinned",
                "detail": "pinned by --pin",
            })

    for unit, members in members_by_unit.items():
        member_targets = tuple(target_by_qname[member] for member in members)
        for member in members:
            unit_for_qname[member] = unit
        baseline_formats = {
            fr.canonical_format_name(floor_assignment.get(member, "BF16"))
            for member in members
        }
        if len(baseline_formats) != 1:
            missing_units[unit] = sorted(baseline_formats)
            continue
        baseline_fmt = next(iter(baseline_formats))
        baseline_spec = fr.get_format(baseline_fmt)
        baseline_bits = sum(
            _format_bits(baseline_spec, target.shape)
            for target in member_targets
        )
        options_by_unit[unit] = [
            UnitOption(
                unit=unit,
                fmt=baseline_fmt,
                members=members,
                bits_total=float(baseline_bits),
                bits_delta=0.0,
                gain=0.0,
            )
        ]
        for target in member_targets:
            bits_format = _format_bits(baseline_spec, target.shape)
            rows.append(
                ProbeRow(
                    qname=target.qname,
                    format=baseline_fmt,
                    shape=target.shape,
                    bits_baseline=bits_format,
                    bits_format=bits_format,
                    bits_delta=0.0,
                    candidate_kl=float(floor_kl),
                    sensitivity=0.0,
                )
            )
            measured_counter[baseline_fmt] += 1

        for spec in requested_specs:
            fmt = fr.canonical_format_name(spec.name)
            if fmt == baseline_fmt:
                continue
            legal = True
            bits_total = 0.0
            for target in member_targets:
                verdict = check_format_applicability(
                    target.shape,
                    spec,
                    qname=target.qname,
                    source_kind=source_manifest.get(target.qname),
                    target_profile=args.target_profile,
                )
                if not verdict.legal:
                    legal = False
                    skipped_counter[fmt] += 1
                    skipped.append({
                        "qname": target.qname,
                        "format": fmt,
                        "shape": list(target.shape),
                        "reason": verdict.reason or "not_applicable",
                        "detail": verdict.detail,
                    })
                    continue
                bits_total += _format_bits(fr.get_format(fmt), target.shape)
            if not legal:
                continue
            candidate_overrides.append({member: fmt for member in members})
            candidate_meta.append(
                (
                    unit,
                    members,
                    member_targets,
                    fmt,
                    float(baseline_bits),
                    float(bits_total),
                )
            )

    # UMA-safe calibration microbatching.  CPU/CUDA tensors draw from the same
    # physical pool on GB10, and the current estimator does not model attention
    # or per-layer activation workspaces.  Keep the default streaming
    # microbatch at 1; explicit integer overrides are profiling-only.
    if args.calib_microbatch in (None, "auto", "AUTO"):
        chosen_calib_micro = 1
    else:
        chosen_calib_micro = max(int(args.calib_microbatch), 1)
    print(
        f"[kl-probe] measuring {len(candidate_overrides)} unit candidates "
        f"(max_lanes_per_batch={args.max_lanes_per_batch}, "
        f"calib_microbatch_size={chosen_calib_micro})",
        flush=True,
    )
    # Tail replay creates a fresh hidden-state cache per calibration window.
    # Graph keys include that cache identity, so candidate replay graphs are
    # effectively single-use captures.  Keep CUDA graphs for measured frontier
    # assignment KL, where each assignment replays the same graph across many
    # calibration rows.
    candidate_cuda_graphs = bool(args.enable_cuda_graphs and args.no_tail_replay)
    if weight_session is not None:
        weight_session.apply_assignment(floor_assignment)
    with _external_weight_management(weight_session is not None):
        candidate_kls = measure_candidate_overrides(
            model,
            floor_assignment,
            candidate_overrides,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
            kl_scope=args.kl_scope,
            max_lanes_per_batch=args.max_lanes_per_batch,
            calib_microbatch_size=chosen_calib_micro,
            include_activation_quant=not args.no_activation_quant,
            use_cuda_graphs=candidate_cuda_graphs,
            use_tail_replay=not args.no_tail_replay,
            replay_cache_window=args.replay_cache_window,
            replay_cache_max_gb=float(args.replay_cache_max_gb),
            replay_cache_max_effective_batch=int(args.replay_cache_max_effective_batch),
            dtype=dtype,
            production_weight_cache=production_weight_cache,
            source_weight_resolver=source_weight_resolver,
        )
    for (
        unit,
        members,
        member_targets,
        fmt,
        baseline_bits,
        bits_total,
    ), candidate_kl in zip(
        candidate_meta,
        candidate_kls,
        strict=True,
    ):
        if not math.isfinite(float(candidate_kl)):
            raise RuntimeError(f"non-finite KL for {unit} {fmt}: {candidate_kl}")
        gain = float(floor_kl - candidate_kl)
        options_by_unit[unit].append(
            UnitOption(
                unit=unit,
                fmt=fmt,
                members=members,
                bits_total=float(bits_total),
                bits_delta=float(bits_total - baseline_bits),
                gain=float(gain),
            )
        )
        per_member_gain = gain / max(len(member_targets), 1)
        for target in member_targets:
            baseline_fmt = floor_assignment.get(target.qname, "BF16")
            baseline_bits_member = _format_bits(
                fr.get_format(baseline_fmt),
                target.shape,
            )
            bits_format = _format_bits(fr.get_format(fmt), target.shape)
            rows.append(
                ProbeRow(
                    qname=target.qname,
                    format=fmt,
                    shape=target.shape,
                    bits_baseline=baseline_bits_member,
                    bits_format=bits_format,
                    bits_delta=bits_format - baseline_bits_member,
                    candidate_kl=float(candidate_kl),
                    sensitivity=float(per_member_gain),
                )
            )
            measured_counter[fmt] += 1

    rows.sort(key=lambda row: (row.qname, row.format))
    for options in options_by_unit.values():
        options.sort(key=lambda option: (option.bits_total, -option.gain, option.fmt))
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
    measured_frontier = False
    if args.measure_frontier_kl and frontier:
        frontier = measure_frontier_points(
            model,
            frontier,
            calib_ids,
            ref_log_probs,
            floor_kl=float(floor_kl),
            floor_assignment=floor_assignment,
            work_root=work_root,
            profile=profile,
            kl_scope=args.kl_scope,
            include_activation_quant=not args.no_activation_quant,
            use_cuda_graphs=bool(args.enable_cuda_graphs),
            production_weight_cache=production_weight_cache,
            weight_session=weight_session,
        )
        measured_frontier = True
    knee_idx = choose_kneedle_point(frontier, use_measured=measured_frontier)
    chosen = frontier[knee_idx] if knee_idx >= 0 else None
    if chosen is None:
        chosen_assignment = dict(floor_assignment)
        chosen_unit_assignment: dict[str, str] = {}
        promotion_count = 0
    else:
        chosen_assignment = chosen.assignment
        chosen_unit_assignment = chosen.unit_assignment
        promotion_count = chosen.promotion_count
    _assert_fused_assignment_coherent(
        floor_assignment,
        targets,
        profile,
        label="floor_assignment",
    )
    _assert_fused_assignment_coherent(
        chosen_assignment,
        targets,
        profile,
        label="chosen_assignment",
    )

    row_json = [
        row.to_json(decision_unit=unit_for_qname.get(row.qname))
        for row in rows
    ]
    floor_bits_total = sum(
        _format_bits(fr.get_format(floor_assignment.get(target.qname, "BF16")), target.shape)
        for target in targets
    )
    payload = {
        "schema": SCHEMA,
        "version": 2,
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
            "bits_total": float(floor_bits_total),
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
            "frontier_kl_measured": bool(measured_frontier),
            "kneedle_gain_source": (
                "measured_gain" if measured_frontier else "predicted_gain_first_order"
            ),
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
            "attention_implementation": attn_impl,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "python": platform.python_version(),
            "max_lanes_per_batch": int(args.max_lanes_per_batch),
            "target_count": int(len(targets)),
            "candidate_flip_count": int(len(candidate_overrides)),
            "candidate_unit_count": int(len(candidate_overrides)),
            "row_count": int(len(rows)),
            "source_manifest_entries": int(len(source_manifest)),
            "production_cache_used": bool(production_weight_cache is not None),
            "production_weight_cache": production_cache_diag,
            "weight_session": weight_session_diag,
            "include_activation_quant": bool(not args.no_activation_quant),
            "calib_microbatch_size": int(chosen_calib_micro),
            "cuda_graphs_enabled": bool(args.enable_cuda_graphs),
            "candidate_cuda_graphs_enabled": bool(candidate_cuda_graphs),
            "tail_replay_enabled": bool(not args.no_tail_replay),
            "replay_cache_window": str(args.replay_cache_window),
            "replay_cache_max_gb": float(args.replay_cache_max_gb),
            "replay_cache_max_effective_batch": int(
                args.replay_cache_max_effective_batch
            ),
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
        default="NVFP4,MXFP8_E4M3,BF16",
        help=(
            "'registry' or a comma-separated format list. Defaults to the "
            "shipping triad: NVFP4, MXFP8_E4M3, BF16."
        ),
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
    parser.add_argument(
        "--max-lanes-per-batch",
        type=int,
        default=4,
        help=(
            "Ceiling for lane-batched candidate evaluation. The probe queries "
            "free GPU memory and steps this down to whatever fits "
            "(_adjust_l3_max_lanes_for_memory), but the estimator cannot yet "
            "bound model activation workspaces on UMA. Default 4 is the "
            "streaming-safe probe setting; raise only for profiling."
        ),
    )
    parser.add_argument(
        "--calib-microbatch",
        default="auto",
        help=(
            "Calibration microbatch size — how many calibration rows are "
            "stacked into each forward. 'auto' (default) is 1 for streaming "
            "UMA safety; an integer pins the size explicitly for profiling."
        ),
    )
    parser.add_argument(
        "--enable-cuda-graphs",
        action="store_true",
        help=(
            "Enable CUDA graphs inside probe KL measurement. Disabled by "
            "default because the probe visits many unique lane/assignment "
            "states and graph registries can retain too much UMA memory."
        ),
    )
    parser.add_argument(
        "--no-tail-replay",
        action="store_true",
        help=(
            "Disable decoder-tail replay for candidate sensitivity probing. "
            "Tail replay caches floor-assignment layer inputs and replays only "
            "the affected layer suffix, preserving downstream interactions "
            "while avoiding repeated prefix forwards."
        ),
    )
    parser.add_argument(
        "--replay-cache-window",
        default="auto",
        help=(
            "Calibration rows per tail-replay cache window. 'auto' sizes the "
            "window from --replay-cache-max-gb, model dimensions, and the "
            "effective replay batch cap."
        ),
    )
    parser.add_argument(
        "--replay-cache-max-gb",
        type=float,
        default=8.0,
        help=(
            "Approximate hidden-state cache budget for auto tail-replay "
            "windows. This is a streaming cache budget, not a hard total "
            "process-memory cap."
        ),
    )
    parser.add_argument(
        "--replay-cache-max-effective-batch",
        type=int,
        default=16,
        help=(
            "Auto replay window cap for rows × lane_count. This bounds the "
            "transient attention workspace that hidden-cache byte estimates "
            "do not capture."
        ),
    )
    parser.add_argument(
        "--no-activation-quant",
        action="store_true",
        help=(
            "Skip activation quantization while probing weight-format "
            "sensitivity. This keeps the format bit accounting unchanged, "
            "but avoids the lane-broadcast activation RTN hot path."
        ),
    )
    parser.add_argument(
        "--candidate-recipe",
        choices=["production", "raw"],
        default="production",
        help=(
            "Candidate weight recipe used for sensitivity measurement. "
            "'production' renders each unit/format with the same GPTQ/"
            "scale-sweep cache used by export; 'raw' is the older RTN/QDQ "
            "diagnostic path."
        ),
    )
    parser.add_argument(
        "--production-weight-cache",
        default=None,
        help=(
            "Optional pickled ProductionWeightCache manifest. When omitted "
            "and --candidate-recipe=production, the probe builds a streamed "
            "cache under --production-cache-dir."
        ),
    )
    parser.add_argument(
        "--production-cache-dir",
        default=None,
        help=(
            "Directory for streamed per-Linear production weight shards. "
            "Defaults to <work-root>/production_weight_cache."
        ),
    )
    parser.add_argument(
        "--production-cache-output",
        default=None,
        help=(
            "Where to write the ProductionWeightCache manifest when the "
            "probe builds it. Defaults to <work-root>/production_weight_cache.pkl."
        ),
    )
    parser.add_argument(
        "--production-cache-dir-override",
        default=None,
        help=(
            "Relocate a loaded disk-streamed production cache to this shard "
            "directory before measuring."
        ),
    )
    parser.add_argument(
        "--production-cache-levers",
        default="gptq,scale_sweep",
        help=(
            "Comma-separated production cleanup levers for on-the-fly cache "
            "builds. Default: gptq,scale_sweep."
        ),
    )
    parser.add_argument(
        "--production-cache-max-act-rows",
        type=int,
        default=512,
        help=(
            "Maximum activation rows stored per Linear while rendering the "
            "production weight cache."
        ),
    )
    parser.add_argument(
        "--production-cache-lru-gb",
        type=float,
        default=16.0,
        help=(
            "LRU budget for lazily loaded production-cache shards. Use 0 to "
            "disable eviction. The cache still streams to disk during build."
        ),
    )
    parser.add_argument(
        "--production-cache-prefetch",
        choices=["batch", "all", "none"],
        default="batch",
        help=(
            "Production-cache prefetch policy. 'batch' prefetches each lane "
            "batch just before use; 'all' materializes every shard within "
            "the LRU budget; 'none' only lazy-loads on demand."
        ),
    )
    parser.add_argument(
        "--weight-session",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Materialize the floor assignment once with WeightSession and "
            "spill BF16 source snapshots to disk. Auto enables this whenever "
            "a production weight cache is active."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-production-cache",
        action="store_true",
        help=(
            "Allow raw fallback for missing production cache entries. Default "
            "is to fail coverage before measurement."
        ),
    )
    parser.add_argument(
        "--no-measure-frontier-kl",
        dest="measure_frontier_kl",
        action="store_false",
        help=(
            "Select the knee from the first-order additive frontier only. "
            "By default the probe measures each frontier assignment end-to-end "
            "and runs kneedle on the measured interaction-aware curve."
        ),
    )
    parser.set_defaults(measure_frontier_kl=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help=(
            "Transformers attention backend for model load. Accepts built-ins "
            "such as sdpa/flash_attention_2 and Kernel Hub backends such as "
            "kernels-community/flash-attn2. Default sdpa avoids the slow eager "
            "attention path on long-sequence replay workloads."
        ),
    )
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
    parser.add_argument(
        "--target-offset",
        type=int,
        default=0,
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
