"""Screened proposal generation for adjoint L3 allocation.

The low-rank adjoint objective is useful as a proposal model, but it can
over-trust cancellation between quantization errors.  This module writes a
small, diverse set of candidate assignments that can be validated by the real
KL path before composing a final allocation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import adjoint_l3 as l3a
from prismaquant import format_registry as fr


SCHEMA = "prismaquant.adjoint_l3.screen.v1"


def layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if not match:
        return None
    return int(match.group(1))


def module_kind(name: str) -> str:
    if "mlp.gate_up_proj" in name:
        return "mlp.gate_up"
    if "mlp.down_proj" in name:
        return "mlp.down"
    if "linear_attn.in_proj_qkvz" in name:
        return "linear_attn.in_qkvz"
    if "linear_attn.out_proj" in name:
        return "linear_attn.out"
    if "self_attn.qkv_proj" in name:
        return "self_attn.qkv"
    if "self_attn.o_proj" in name:
        return "self_attn.o"
    return "other"


def _slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return out[:96] or "candidate"


def reference_upgrade_rows(
    units: Sequence[l3a.AdjointUnit],
    rank: int,
    reference_assignment: Mapping[str, str],
    *,
    group_members: Mapping[str, Sequence[str]] | None = None,
    forbid_downgrades: bool = True,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
) -> list[dict]:
    """Return all one-unit moves from a reference assignment.

    Rows are scored in the reference context.  Negative deltas mean the move
    improves the surrogate objective; they should still be validated with the
    measured KL path before shipping.
    """

    current = {
        unit.name: fr.canonical_format_name(reference_assignment[unit.name])
        for unit in units
        if unit.name in reference_assignment
    }
    if len(current) != len(units):
        missing = sorted(unit.name for unit in units if unit.name not in current)
        raise ValueError(f"reference assignment missing {len(missing)} units: {missing[:3]}")
    base_score = l3a.score_adjoint_assignment(
        units,
        rank,
        current,
        low_rank_weight=low_rank_weight,
        diagonal_weight=diagonal_weight,
    )
    by_format = {
        unit.name: {option.fmt: option for option in unit.options}
        for unit in units
    }
    rows = []
    for unit in units:
        old_fmt = current[unit.name]
        old_option = by_format[unit.name].get(old_fmt)
        if old_option is None:
            continue
        for option in unit.options:
            if option.fmt == old_fmt:
                continue
            if forbid_downgrades and option.bits_total < old_option.bits_total:
                continue
            trial = dict(current)
            trial[unit.name] = option.fmt
            score = l3a.score_adjoint_assignment(
                units,
                rank,
                trial,
                low_rank_weight=low_rank_weight,
                diagonal_weight=diagonal_weight,
            )
            layer = layer_index(unit.name)
            rows.append({
                "name": unit.name,
                "members": list(group_members.get(unit.name, (unit.name,))) if group_members else [unit.name],
                "module_kind": module_kind(unit.name),
                "layer_index": layer,
                "from_format": old_fmt,
                "to_format": option.fmt,
                "delta_objective": float(score[0] - base_score[0]),
                "delta_diagonal": float(score[1] - base_score[1]),
                "delta_low_rank": float(score[2] - base_score[2]),
                "delta_bits": float(option.bits_total - old_option.bits_total),
                "from_bits": float(old_option.bits_total),
                "to_bits": float(option.bits_total),
            })
    return rows


def _add_unique(
    selected: list[dict],
    rows: Sequence[dict],
    *,
    bucket: str,
    limit: int,
) -> None:
    existing = {(row["name"], row["to_format"]) for row in selected}
    added = 0
    for row in rows:
        key = (row["name"], row["to_format"])
        if key in existing:
            continue
        item = dict(row)
        item["selection_bucket"] = bucket
        selected.append(item)
        existing.add(key)
        added += 1
        if added >= limit:
            break


def select_diverse_upgrade_rows(
    rows: Sequence[Mapping],
    *,
    max_candidates: int = 32,
    per_bucket: int = 8,
) -> list[dict]:
    """Select a diverse shortlist from one-unit surrogate moves."""

    rows = [dict(row) for row in rows]
    if not rows:
        return []
    max_layer = max((row.get("layer_index") or 0) for row in rows)

    def objective(row: Mapping) -> tuple:
        return (
            float(row["delta_objective"]),
            float(row["delta_bits"]),
            str(row["name"]),
            str(row["to_format"]),
        )

    def diagonal(row: Mapping) -> tuple:
        return (
            float(row["delta_diagonal"]),
            float(row["delta_bits"]),
            str(row["name"]),
            str(row["to_format"]),
        )

    def late_objective(row: Mapping) -> tuple:
        layer = row.get("layer_index")
        depth_gap = max_layer - int(layer if layer is not None else 0)
        return (
            float(row["delta_objective"]) + 0.002 * float(depth_gap),
            float(row["delta_bits"]),
            str(row["name"]),
            str(row["to_format"]),
        )

    def efficient(row: Mapping) -> tuple:
        return (
            float(row["delta_objective"]) / max(float(row["delta_bits"]), 1.0),
            float(row["delta_bits"]),
            str(row["name"]),
            str(row["to_format"]),
        )

    selected: list[dict] = []
    _add_unique(selected, sorted(rows, key=objective), bucket="objective", limit=per_bucket)
    _add_unique(selected, sorted(rows, key=diagonal), bucket="diagonal", limit=per_bucket)
    _add_unique(selected, sorted(rows, key=late_objective), bucket="late_objective", limit=per_bucket)
    _add_unique(selected, sorted(rows, key=efficient), bucket="efficient", limit=per_bucket)
    for fmt in sorted({str(row["to_format"]) for row in rows}):
        fmt_rows = [row for row in rows if row["to_format"] == fmt]
        _add_unique(
            selected,
            sorted(fmt_rows, key=objective),
            bucket=f"{fmt.lower()}_objective",
            limit=max(1, per_bucket // 2),
        )
    return selected[:max_candidates]


def write_candidate_assignments(
    *,
    output_dir: str | Path,
    base_assignment: Mapping[str, str],
    rows: Sequence[Mapping],
) -> list[dict]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, row in enumerate(rows, 1):
        label = (
            f"one_{idx:03d}_{row.get('selection_bucket', 'proposal')}_"
            f"{row['to_format']}_{row['name']}"
        )
        label = _slug(label)
        assignment = dict(base_assignment)
        for member in row["members"]:
            assignment[str(member)] = str(row["to_format"])
        path = out_root / f"{label}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(assignment, fh, indent=2, sort_keys=True)
        record = dict(row)
        record["label"] = label
        record["assignment_path"] = str(path)
        records.append(record)
    return records


def validation_rows_by_label(path: str | Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"validation JSON {path} has no results list")
    return {str(row["label"]): dict(row) for row in results if "label" in row}


def write_combo_assignments_from_validation(
    *,
    output_dir: str | Path,
    base_assignment: Mapping[str, str],
    proposals: Sequence[Mapping],
    validation_results: Mapping[str, Mapping],
    base_label: str,
    max_positive_moves: int = 8,
    max_combo_size: int = 3,
    max_combos: int = 32,
    max_move_delta_bpp: float | None = None,
    max_combo_delta_bpp: float | None = None,
) -> list[dict]:
    """Write combinations of individually validated positive moves."""

    if base_label not in validation_results:
        raise ValueError(f"base label {base_label!r} not present in validation results")
    base_result = validation_results[base_label]
    base_kl = float(base_result["last_token_kl"])
    base_bpp = float(base_result.get("bpp", 0.0) or 0.0)
    measured = []
    for row in proposals:
        label = str(row.get("label", ""))
        result = validation_results.get(label)
        if not result:
            continue
        delta_kl = float(result["last_token_kl"]) - base_kl
        if delta_kl >= 0.0:
            continue
        item = dict(row)
        item["measured_delta_kl"] = float(delta_kl)
        item["measured_kl"] = float(result["last_token_kl"])
        item["measured_bpp"] = float(result.get("bpp", 0.0) or 0.0)
        item["measured_delta_bpp"] = (
            float(item["measured_bpp"] - base_bpp) if base_bpp > 0.0 else 0.0
        )
        if (
            max_move_delta_bpp is not None
            and item["measured_delta_bpp"] > float(max_move_delta_bpp)
        ):
            continue
        if "to_format" not in item and "to" in item:
            item["to_format"] = item["to"]
        if "from_format" not in item and "from" in item:
            item["from_format"] = item["from"]
        if "name" not in item and "unit" in item:
            item["name"] = item["unit"]
        if "members" not in item:
            item["members"] = [item["name"]]
        measured.append(item)
    measured.sort(key=lambda row: (row["measured_delta_kl"], row["delta_bits"]))
    measured = measured[:max_positive_moves]

    combos = []
    for size in range(1, max_combo_size + 1):
        for combo in itertools.combinations(measured, size):
            touched: set[str] = set()
            conflict = False
            for row in combo:
                members = {str(member) for member in row["members"]}
                if touched & members:
                    conflict = True
                    break
                touched.update(members)
            if conflict:
                continue
            sum_delta_bpp = float(sum(row["measured_delta_bpp"] for row in combo))
            if (
                max_combo_delta_bpp is not None
                and sum_delta_bpp > float(max_combo_delta_bpp)
            ):
                continue
            combos.append({
                "moves": list(combo),
                "sum_measured_delta_kl": float(sum(row["measured_delta_kl"] for row in combo)),
                "sum_measured_delta_bpp": sum_delta_bpp,
                "sum_delta_bits": float(sum(float(row["delta_bits"]) for row in combo)),
            })
    combos.sort(key=lambda row: (row["sum_measured_delta_kl"], row["sum_delta_bits"]))
    combos = combos[:max_combos]

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, combo in enumerate(combos, 1):
        label = _slug(
            "combo_%03d_%s" % (
                idx,
                "_".join(str(row["label"]) for row in combo["moves"]),
            )
        )
        assignment = dict(base_assignment)
        for row in combo["moves"]:
            for member in row["members"]:
                assignment[str(member)] = str(row["to_format"])
        path = out_root / f"{label}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(assignment, fh, indent=2, sort_keys=True)
        record = {
            "label": label,
            "assignment_path": str(path),
            "sum_measured_delta_kl": combo["sum_measured_delta_kl"],
            "sum_measured_delta_bpp": combo["sum_measured_delta_bpp"],
            "sum_delta_bits": combo["sum_delta_bits"],
            "moves": [dict(row) for row in combo["moves"]],
        }
        records.append(record)
    return records


def propose(
    *,
    adjoint_costs: str | Path,
    probe: str | Path,
    base_assignment_path: str | Path,
    output_dir: str | Path,
    model: str | None = None,
    formats: str = "NVFP4,MXFP8_E4M3,BF16",
    fused_groups: bool = False,
    max_candidates: int = 32,
    per_bucket: int = 8,
) -> dict:
    payload = l3a.load_adjoint_l3_payload(adjoint_costs)
    stats = l3a._load_probe_stats(probe)
    specs = l3a._parse_formats(formats)
    units = l3a.adjoint_units_from_payload(payload, stats=stats, formats=specs)
    group_members = None
    solve_units = units
    if fused_groups:
        if model:
            from prismaquant.model_profiles import detect_profile

            profile = detect_profile(model)
        else:
            from prismaquant.model_profiles import DefaultProfile

            profile = DefaultProfile()
        solve_units, group_members = l3a.group_adjoint_units_by_profile(units, profile)

    base_assignment = l3a._load_assignment_json(base_assignment_path)
    reference = l3a.collapse_assignment_to_solve_units(
        base_assignment,
        solve_units,
        group_members,
    )
    rows = reference_upgrade_rows(
        solve_units,
        int(payload["rank"]),
        reference,
        group_members=group_members,
        forbid_downgrades=True,
    )
    selected = select_diverse_upgrade_rows(
        rows,
        max_candidates=max_candidates,
        per_bucket=per_bucket,
    )
    candidates = write_candidate_assignments(
        output_dir=output_dir,
        base_assignment=base_assignment,
        rows=selected,
    )
    manifest = {
        "schema": SCHEMA,
        "mode": "proposals",
        "adjoint_costs": str(adjoint_costs),
        "probe": str(probe),
        "base_assignment": str(base_assignment_path),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    out_path = Path(output_dir) / "manifest.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def combos_from_validation(
    *,
    proposal_manifest_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    base_label: str = "old_5p2631",
    max_positive_moves: int = 8,
    max_combo_size: int = 3,
    max_combos: int = 32,
    max_move_delta_bpp: float | None = None,
    max_combo_delta_bpp: float | None = None,
) -> dict:
    with Path(proposal_manifest_path).open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_assignment_path = manifest.get("base_assignment")
    if not base_assignment_path:
        raise ValueError("proposal manifest lacks base_assignment")
    base_assignment = l3a._load_assignment_json(base_assignment_path)
    validation = validation_rows_by_label(validation_path)
    combos = write_combo_assignments_from_validation(
        output_dir=output_dir,
        base_assignment=base_assignment,
        proposals=manifest.get("candidates", []),
        validation_results=validation,
        base_label=base_label,
        max_positive_moves=max_positive_moves,
        max_combo_size=max_combo_size,
        max_combos=max_combos,
        max_move_delta_bpp=max_move_delta_bpp,
        max_combo_delta_bpp=max_combo_delta_bpp,
    )
    out = {
        "schema": SCHEMA,
        "mode": "combos",
        "proposal_manifest": str(proposal_manifest_path),
        "validation": str(validation_path),
        "base_label": base_label,
        "max_move_delta_bpp": max_move_delta_bpp,
        "max_combo_delta_bpp": max_combo_delta_bpp,
        "combo_count": len(combos),
        "combos": combos,
    }
    out_path = Path(output_dir) / "manifest.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose", help="write one-unit candidate assignments")
    p.add_argument("--adjoint-costs", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--base-assignment", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--formats", default="NVFP4,MXFP8_E4M3,BF16")
    p.add_argument("--fused-groups", action="store_true")
    p.add_argument("--max-candidates", type=int, default=32)
    p.add_argument("--per-bucket", type=int, default=8)

    c = sub.add_parser("combos", help="write combinations from validated proposals")
    c.add_argument("--proposal-manifest", required=True)
    c.add_argument("--validation", required=True)
    c.add_argument("--output-dir", required=True)
    c.add_argument("--base-label", default="old_5p2631")
    c.add_argument("--max-positive-moves", type=int, default=8)
    c.add_argument("--max-combo-size", type=int, default=3)
    c.add_argument("--max-combos", type=int, default=32)
    c.add_argument(
        "--max-move-delta-bpp",
        type=float,
        default=None,
        help="drop one-unit moves whose measured bpp increase exceeds this value",
    )
    c.add_argument(
        "--max-combo-delta-bpp",
        type=float,
        default=None,
        help="drop generated combos whose summed one-unit measured bpp increase exceeds this value",
    )

    args = parser.parse_args(argv)
    if args.command == "propose":
        manifest = propose(
            adjoint_costs=args.adjoint_costs,
            probe=args.probe,
            base_assignment_path=args.base_assignment,
            output_dir=args.output_dir,
            model=args.model,
            formats=args.formats,
            fused_groups=args.fused_groups,
            max_candidates=args.max_candidates,
            per_bucket=args.per_bucket,
        )
        print(
            f"[adjoint-l3-screen] wrote {args.output_dir} "
            f"candidates={manifest['candidate_count']}",
            flush=True,
        )
        return 0
    if args.command == "combos":
        manifest = combos_from_validation(
            proposal_manifest_path=args.proposal_manifest,
            validation_path=args.validation,
            output_dir=args.output_dir,
            base_label=args.base_label,
            max_positive_moves=args.max_positive_moves,
            max_combo_size=args.max_combo_size,
            max_combos=args.max_combos,
            max_move_delta_bpp=args.max_move_delta_bpp,
            max_combo_delta_bpp=args.max_combo_delta_bpp,
        )
        print(
            f"[adjoint-l3-screen] wrote {args.output_dir} "
            f"combos={manifest['combo_count']}",
            flush=True,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
