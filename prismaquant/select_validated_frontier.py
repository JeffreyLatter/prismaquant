"""Select a measured frontier point from assignment-KL validation output."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import format_registry as fr


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _load_assignment(path: str | Path) -> dict[str, str]:
    payload = _load_json(path)
    raw = payload.get("assignment") if isinstance(payload, Mapping) else None
    if raw is None and isinstance(payload, Mapping):
        raw = payload
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected assignment JSON object")
    return {
        str(name): str(fmt).strip().upper()
        for name, fmt in raw.items()
        if str(name).strip()
    }


def _layer_config_from_assignment(assignment: Mapping[str, str]) -> dict:
    out = {}
    for name, fmt in sorted(assignment.items()):
        out[str(name)] = fr.get_format(str(fmt).strip().upper()).autoround_config()
    return out


def _kneedle_convex_decreasing(points: Sequence[Mapping[str, float]]) -> int:
    """Return knee index for points sorted by increasing bpp, decreasing KL."""
    if len(points) < 3:
        return min(
            range(len(points)),
            key=lambda i: (float(points[i]["kl"]), float(points[i]["bpp"])),
        )
    xs = [float(p["bpp"]) for p in points]
    ys = [float(p["kl"]) for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return min(range(len(points)), key=lambda i: (ys[i], xs[i]))
    x_norm = [(x - xmin) / (xmax - xmin) for x in xs]
    y_norm = [(y - ymin) / (ymax - ymin) for y in ys]
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    return min(range(len(diffs)), key=lambda i: diffs[i])


def measured_frontier(results: Sequence[Mapping]) -> list[dict]:
    """Return non-dominated measured KL/bpp points sorted by bpp.

    A point is dominated when a lower-or-equal bpp assignment already has
    lower-or-equal KL. Kneedle should operate on this measured lower envelope,
    not on noisy interior points.
    """
    rows: list[dict] = []
    for row in results:
        kl = row.get("last_token_kl", row.get("kl"))
        bpp = row.get("bpp")
        path = row.get("path")
        label = row.get("label")
        if kl is None or bpp is None or path is None:
            continue
        kl_f = float(kl)
        bpp_f = float(bpp)
        if not (math.isfinite(kl_f) and math.isfinite(bpp_f)):
            continue
        rows.append({
            "label": str(label or Path(str(path)).stem),
            "path": str(path),
            "kl": kl_f,
            "bpp": bpp_f,
            "format_counts": dict(row.get("format_counts", {}) or {}),
            "changed_vs_base": int(row.get("changed_vs_base", 0) or 0),
            "mse": dict(row.get("mse", {}) or {}),
        })
    rows.sort(key=lambda r: (r["bpp"], r["kl"], r["label"]))
    frontier: list[dict] = []
    best_kl = float("inf")
    for row in rows:
        if row["kl"] < best_kl:
            frontier.append(row)
            best_kl = row["kl"]
    return frontier


def select_frontier_point(
    results: Sequence[Mapping],
    *,
    mode: str = "kneedle",
) -> tuple[dict, list[dict]]:
    frontier = measured_frontier(results)
    if not frontier:
        raise ValueError("no finite measured KL/bpp points found")
    if mode == "best-kl":
        idx = min(range(len(frontier)), key=lambda i: (frontier[i]["kl"], frontier[i]["bpp"]))
    elif mode == "lowest-bpp":
        idx = 0
    elif mode == "kneedle":
        idx = _kneedle_convex_decreasing(frontier)
    else:
        raise ValueError(f"unknown selection mode {mode!r}")
    return frontier[idx], frontier


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a measured-kneedle assignment from validate_assignments_kl output",
    )
    parser.add_argument("--validation-json", required=True)
    parser.add_argument(
        "--mode",
        choices=("kneedle", "best-kl", "lowest-bpp"),
        default="kneedle",
    )
    parser.add_argument("--output-layer-config", required=True)
    parser.add_argument("--output-assignment", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(argv)

    payload = _load_json(args.validation_json)
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list):
        raise ValueError("--validation-json must contain a results list")

    selected, frontier = select_frontier_point(results, mode=args.mode)
    assignment = _load_assignment(selected["path"])
    layer_config = _layer_config_from_assignment(assignment)

    layer_config_path = Path(args.output_layer_config)
    layer_config_path.parent.mkdir(parents=True, exist_ok=True)
    layer_config_path.write_text(json.dumps(layer_config, indent=2, sort_keys=True) + "\n")

    assignment_payload = {
        "schema": "prismaquant.validated_frontier_assignment.v1",
        "selection_mode": args.mode,
        "selected": selected,
        "assignment": dict(sorted(assignment.items())),
    }
    assignment_path = Path(args.output_assignment)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(json.dumps(assignment_payload, indent=2, sort_keys=True) + "\n")

    summary = {
        "schema": "prismaquant.validated_frontier_selection.v1",
        "validation_json": str(Path(args.validation_json)),
        "selection_mode": args.mode,
        "selected": selected,
        "frontier": frontier,
        "n_results": len(results),
        "n_frontier": len(frontier),
        "output_layer_config": str(layer_config_path),
        "output_assignment": str(assignment_path),
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    mse = selected.get("mse", {}) if isinstance(selected, Mapping) else {}
    mse_msg = ""
    if isinstance(mse, Mapping) and mse.get("output_mse_sum") is not None:
        mse_msg = f" output_mse={float(mse['output_mse_sum']):.6g}"
    print(
        "[frontier-select] selected "
        f"{selected['label']} bpp={selected['bpp']:.6f} "
        f"KL={selected['kl']:.8g}{mse_msg} mode={args.mode}",
        flush=True,
    )
    print(f"[frontier-select] layer_config -> {layer_config_path}", flush=True)
    print(f"[frontier-select] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
