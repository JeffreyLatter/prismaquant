"""Generate an adjoint-sketch L3 surrogate frontier."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import adjoint_l3 as l3a
from prismaquant import format_registry as fr
from prismaquant.propagated_cost import assignment_bit_total


def _bpp_label(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _normalise(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _segmented_kneedle_point(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, bool]:
    pts = sorted(points)
    if len(pts) < 3:
        mid = pts[len(pts) // 2]
        return mid[0], mid[1], 0.0, True
    xs = _normalise([point[0] for point in pts])
    ys_raw = _normalise([point[1] for point in pts])
    ys = [1.0 - value for value in ys_raw]
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denom = max(((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5, 1e-12)
    scored = []
    for idx, (x, y) in enumerate(zip(xs, ys)):
        score = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        scored.append((score, idx))
    score, idx = max(scored)
    endpoint = idx in {0, len(pts) - 1}
    if endpoint:
        idx = len(pts) // 2
    return pts[idx][0], pts[idx][1], float(score), endpoint


def _target_bpps(args) -> list[float]:
    if args.target_full_bpps:
        values = l3a._parse_float_list(args.target_full_bpps)
        return sorted(dict.fromkeys(float(value) for value in values))
    if args.bpp_min is None or args.bpp_max is None or args.bpp_step is None:
        raise ValueError("provide --target-full-bpps or --bpp-min/--bpp-max/--bpp-step")
    if args.bpp_step <= 0:
        raise ValueError("--bpp-step must be positive")
    values = []
    value = float(args.bpp_min)
    stop = float(args.bpp_max)
    while value <= stop + 1e-9:
        values.append(round(value, 10))
        value += float(args.bpp_step)
    return values


def solve_frontier(
    *,
    adjoint_costs: str | Path,
    probe: str | Path,
    base_assignment_path: str | Path,
    target_bpps: Sequence[float],
    output_dir: str | Path,
    model: str | None = None,
    formats: str = "NVFP4,MXFP8_E4M3,BF16",
    fused_groups: bool = False,
    diagonal_floor_frac: float | None = None,
    mse_diagonal_floor_frac: float | None = None,
    lambdas: Sequence[float] | None = None,
    seed_from_base: bool = True,
    seed_assignment_paths: Sequence[str | Path] = (),
    max_changed_units: int | None = None,
    change_penalty: float = 0.0,
    forbid_reference_downgrades: bool = False,
    max_passes: int = 16,
) -> dict:
    payload = l3a.retune_adjoint_diagonal_costs(
        l3a.load_adjoint_l3_payload(adjoint_costs),
        diagonal_floor_frac=diagonal_floor_frac,
        mse_diagonal_floor_frac=mse_diagonal_floor_frac,
    )
    stats = l3a._load_probe_stats(probe)
    specs = l3a._parse_formats(formats)
    specs_by_name = {fr.canonical_format_name(spec.name): spec for spec in specs}
    units = l3a.adjoint_units_from_payload(payload, stats=stats, formats=specs)
    rank = int(payload["rank"])
    solve_units = units
    group_members = None
    profile_name = None
    if fused_groups:
        if model:
            from prismaquant.model_profiles import detect_profile

            profile = detect_profile(model)
        else:
            from prismaquant.model_profiles import DefaultProfile

            profile = DefaultProfile()
        profile_name = getattr(profile, "name", type(profile).__name__)
        solve_units, group_members = l3a.group_adjoint_units_by_profile(units, profile)

    base_assignment = l3a._load_assignment_json(base_assignment_path)
    raw_unit_names = {unit.name for unit in units}
    all_names = sorted(set(stats) & set(base_assignment))
    if not all_names:
        raise ValueError("base assignment and probe stats have no shared names")
    fixed_names = [name for name in all_names if name not in raw_unit_names]
    fixed_assignment = {name: base_assignment[name] for name in fixed_names}
    fixed_bits = assignment_bit_total(stats, fixed_assignment, specs_by_name)
    total_params = sum(int(stats[name].get("n_params", 0) or 0) for name in all_names)
    if total_params <= 0:
        raise ValueError("probe stats/base assignment have zero total params")

    initial_assignments: list[Mapping[str, str] | None] = [None]
    if seed_from_base:
        initial_assignments.append(l3a.collapse_assignment_to_solve_units(
            base_assignment,
            solve_units,
            group_members,
        ))
    for path in seed_assignment_paths:
        initial_assignments.append(l3a.collapse_assignment_to_solve_units(
            l3a._load_assignment_json(path),
            solve_units,
            group_members,
        ))
    initial_assignments = [item for item in initial_assignments if item is None or item]
    reference_assignment = l3a.collapse_assignment_to_solve_units(
        base_assignment,
        solve_units,
        group_members,
    )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for target_bpp in target_bpps:
        label = _bpp_label(float(target_bpp))
        target_bits = float(target_bpp) * float(total_params) - float(fixed_bits)
        if target_bits < 0.0:
            raise ValueError(f"target bpp {target_bpp} leaves negative solved budget")
        result = l3a.solve_low_rank_budget_sweep(
            solve_units,
            rank,
            target_bits_total=target_bits,
            lambdas=lambdas,
            initial_assignments=initial_assignments,
            reference_assignment=reference_assignment,
            max_changed_units=max_changed_units,
            change_penalty=change_penalty,
            forbid_reference_downgrades=forbid_reference_downgrades,
            max_passes=max_passes,
        )
        raw_assignment = result.assignment
        target_feasible = result.bits_total <= float(target_bits) + 1e-6
        if group_members is not None:
            raw_assignment = l3a.expand_grouped_assignment(result.assignment, group_members)
        full_assignment = dict(base_assignment)
        full_assignment.update(raw_assignment)
        full_bits = fixed_bits + result.bits_total
        achieved_bpp = full_bits / float(total_params)
        counts = dict(Counter(full_assignment.values()))
        meta = {
            "target_full_bpp": float(target_bpp),
            "achieved_full_bpp": float(achieved_bpp),
            "solved_full_bits_total": float(full_bits),
            "target_solved_bits_total": float(target_bits),
            "achieved_solved_bits_total": float(result.bits_total),
            "target_feasible": bool(target_feasible),
            "fixed_bits": float(fixed_bits),
            "fixed_entry_count": len(fixed_names),
            "total_param_count": int(total_params),
            "fused_groups": bool(fused_groups),
            "profile": profile_name,
        }
        assignment_path = out_root / f"assignment_bpp_{label}.json"
        full_assignment_path = out_root / f"full_assignment_bpp_{label}.json"
        move_report_path = out_root / f"moves_bpp_{label}.json"
        assignment_path.write_text(
            json.dumps(l3a.result_to_json_dict(result, assignment=raw_assignment, meta=meta), indent=2)
            + "\n"
        )
        full_assignment_path.write_text(
            json.dumps(
                l3a.result_to_json_dict(
                    result,
                    assignment=full_assignment,
                    meta={**meta, "assignment_scope": "base_plus_adjoint_overlay"},
                ),
                indent=2,
            )
            + "\n"
        )
        moves = l3a.build_move_report(
            solve_units,
            rank,
            result.assignment,
            base_assignment,
            group_members=group_members,
        )
        move_report_path.write_text(
            json.dumps(
                {
                    "schema": "prismaquant.adjoint_l3.move_report.v1",
                    "move_count": len(moves),
                    "moves": moves,
                },
                indent=2,
            )
            + "\n"
        )
        rows.append({
            "target_bpp": float(target_bpp),
            "achieved_bpp": float(achieved_bpp),
            "objective": float(result.objective),
            "diagonal_cost": float(result.diagonal_cost),
            "low_rank_cost": float(result.low_rank_cost),
            "bits_total": float(result.bits_total),
            "target_feasible": bool(target_feasible),
            "moves": int(result.moves),
            "passes": int(result.passes),
            "changed_units": result.changed_units,
            "assignment_path": str(assignment_path),
            "full_assignment_path": str(full_assignment_path),
            "move_report_path": str(move_report_path),
            "format_counts": counts,
        })

    knee = None
    if len(rows) >= 3:
        knee_bpp, knee_objective, score, endpoint = _segmented_kneedle_point(
            [(row["achieved_bpp"], row["objective"]) for row in rows]
        )
        chosen = min(rows, key=lambda row: abs(row["achieved_bpp"] - knee_bpp))
        knee = {
            "mode": "surrogate_adjoint_kneedle",
            "achieved_bpp": float(knee_bpp),
            "objective": float(knee_objective),
            "kneedle_score": float(score),
            "endpoint_fallback": bool(endpoint),
            "assignment_path": chosen["assignment_path"],
            "full_assignment_path": chosen["full_assignment_path"],
            "move_report_path": chosen["move_report_path"],
        }

    summary = {
        "schema": "prismaquant.adjoint_l3.frontier.v1",
        "rows": rows,
        "knee": knee,
        "meta": {
            "adjoint_costs": str(adjoint_costs),
            "probe": str(probe),
            "base_assignment": str(base_assignment_path),
            "formats": [spec.name for spec in specs],
            "rank": rank,
            "raw_unit_count": len(units),
            "solve_unit_count": len(solve_units),
            "seed_assignment_count": len(initial_assignments) - 1,
            "lambdas": list(lambdas) if lambdas is not None else None,
            "max_changed_units": (
                int(max_changed_units) if max_changed_units is not None else None
            ),
            "change_penalty": float(change_penalty),
            "forbid_reference_downgrades": bool(forbid_reference_downgrades),
            "objective_metric": payload.get("meta", {}).get("objective_metric"),
            "curvature": payload.get("meta", {}).get("curvature"),
            "direction_mode": payload.get("direction_mode"),
            "fisher_temperature": payload.get("meta", {}).get("fisher_temperature"),
            "fisher_token_scope": payload.get("meta", {}).get("fisher_token_scope"),
            "fisher_probe_distribution": payload.get("meta", {}).get(
                "fisher_probe_distribution"
            ),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_root / "frontier.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "target_bpp",
            "achieved_bpp",
            "objective",
            "diagonal_cost",
            "low_rank_cost",
            "bits_total",
            "target_feasible",
            "moves",
            "passes",
            "changed_units",
            "assignment_path",
            "full_assignment_path",
            "move_report_path",
            "format_counts",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["format_counts"] = json.dumps(row["format_counts"], sort_keys=True)
            writer.writerow(csv_row)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an adjoint L3 bpp frontier")
    parser.add_argument("--adjoint-costs", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--formats", default="NVFP4,MXFP8_E4M3,BF16")
    parser.add_argument("--fused-groups", action="store_true")
    parser.add_argument("--target-full-bpps", default=None)
    parser.add_argument("--bpp-min", type=float, default=None)
    parser.add_argument("--bpp-max", type=float, default=None)
    parser.add_argument("--bpp-step", type=float, default=None)
    parser.add_argument("--diagonal-floor-frac", type=float, default=None)
    parser.add_argument("--mse-diagonal-floor-frac", type=float, default=None)
    parser.add_argument("--lambdas", default=None)
    parser.add_argument("--seed-assignment", action="append", default=[])
    parser.add_argument("--no-seed-from-base", action="store_true")
    parser.add_argument("--max-changed-units", type=int, default=None)
    parser.add_argument("--change-penalty", type=float, default=0.0)
    parser.add_argument("--forbid-reference-downgrades", action="store_true")
    parser.add_argument("--max-passes", type=int, default=16)
    args = parser.parse_args(argv)

    summary = solve_frontier(
        adjoint_costs=args.adjoint_costs,
        probe=args.probe,
        base_assignment_path=args.base_assignment,
        target_bpps=_target_bpps(args),
        output_dir=args.output_dir,
        model=args.model,
        formats=args.formats,
        fused_groups=args.fused_groups,
        diagonal_floor_frac=args.diagonal_floor_frac,
        mse_diagonal_floor_frac=args.mse_diagonal_floor_frac,
        lambdas=l3a._parse_float_list(args.lambdas),
        seed_from_base=not args.no_seed_from_base,
        seed_assignment_paths=args.seed_assignment,
        max_changed_units=args.max_changed_units,
        change_penalty=args.change_penalty,
        forbid_reference_downgrades=args.forbid_reference_downgrades,
        max_passes=args.max_passes,
    )
    print(
        f"[adjoint-l3-frontier] wrote {args.output_dir} "
        f"points={len(summary['rows'])} "
        f"knee={summary.get('knee', {}).get('achieved_bpp') if summary.get('knee') else 'n/a'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
