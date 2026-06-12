"""Select a measured frontier point from assignment-KL validation output."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.layer_config import canonicalize_format
from prismaquant.saturation_select import find_saturation_bpp


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _saturation_pick(frontier: Sequence[Mapping], z: float) -> tuple[int, dict]:
    """Saturation B* over the measured frontier (the unconstrained selector).

    Builds the bpp grid + a (kl, kl_stderr) lookup from the measured lower
    envelope and runs ``find_saturation_bpp``: B* is the lowest bpp whose KL is
    within z * combined stderr of the highest-bpp asymptote. Returns the chosen
    frontier index and the full saturation result (trace/slopes/measured) for
    the summary. ``no_noise_floor`` is set when the frontier carries no positive
    per-bpp stderr (single-rep validation): the band is then 0, so B* collapses
    to the asymptote (ship the most bits) — a safe but uninformative degenerate
    that the caller must surface (run validation with --calib-repeats>=4).
    """
    grid = [float(r["bpp"]) for r in frontier]
    kl_by = {float(r["bpp"]): float(r["kl"]) for r in frontier}

    def _se(r):
        se = r.get("kl_stderr")
        try:
            se = float(se)
        except (TypeError, ValueError):
            return 0.0
        return se if math.isfinite(se) and se > 0.0 else 0.0

    se_by = {float(r["bpp"]): _se(r) for r in frontier}
    result = find_saturation_bpp(
        grid, lambda b: (kl_by[b], se_by[b]), z=z,
    )
    result["no_noise_floor"] = not any(v > 0.0 for v in se_by.values())
    bstar = result["bpp"]
    idx = min(range(len(frontier)), key=lambda i: abs(float(frontier[i]["bpp"]) - bstar))
    return idx, result


def _load_assignment(path: str | Path) -> dict[str, str]:
    payload = _load_json(path)
    raw = payload.get("assignment") if isinstance(payload, Mapping) else None
    if raw is None and isinstance(payload, Mapping):
        raw = payload
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected assignment JSON object")
    # Entries may be format-name strings ({qname: "NVFP4"}) or AutoRound-style
    # dicts ({qname: {"data_type": "nv_fp", "bits": 4, ...}}); str().upper() on
    # a dict silently fabricates a garbage format name. Strings go through the
    # registry canonicalizer (which keeps FP8_SOURCE & friends); dicts go
    # through the layer-config parser. Unknown names still fail loudly at
    # fr.get_format in _layer_config_from_assignment.
    return {
        str(name): (
            fr.canonical_format_name(fmt.strip().upper())
            if isinstance(fmt, str)
            else canonicalize_format(fmt)
        )
        for name, fmt in raw.items()
        if str(name).strip()
    }


def _layer_config_from_assignment(assignment: Mapping[str, str]) -> dict:
    out = {}
    for name, fmt in sorted(assignment.items()):
        out[str(name)] = fr.get_format(str(fmt).strip().upper()).autoround_config()
    return out


def _log_error_values(values: Sequence[float]) -> list[float]:
    finite_positive = [
        float(value) for value in values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not finite_positive:
        return [0.0 for _ in values]
    floor = max(min(finite_positive) * 1.0e-6, 1.0e-300)
    return [math.log10(max(float(value), floor)) for value in values]


def _kneedle_convex_decreasing(
    points: Sequence[Mapping[str, float]],
    *,
    log_error: bool = True,
) -> int:
    """Return knee index for points sorted by increasing bpp, decreasing KL."""
    if len(points) < 3:
        return min(
            range(len(points)),
            key=lambda i: (float(points[i]["kl"]), float(points[i]["bpp"])),
        )
    xs = [float(p["bpp"]) for p in points]
    ys = [float(p["kl"]) for p in points]
    if log_error:
        ys = _log_error_values(ys)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return min(range(len(points)), key=lambda i: (ys[i], xs[i]))
    x_norm = [(x - xmin) / (xmax - xmin) for x in xs]
    y_norm = [(y - ymin) / (ymax - ymin) for y in ys]
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    return min(range(len(diffs)), key=lambda i: diffs[i])


def kneedle_comparison(points: Sequence[Mapping[str, float]]) -> dict:
    if len(points) < 3:
        return {"enabled": False, "reason": "too_few_frontier_points"}

    def _record(mode: str, idx: int) -> dict:
        row = points[idx]
        return {
            "mode": mode,
            "label": row.get("label"),
            "bpp": float(row["bpp"]),
            "kl": float(row["kl"]),
            "index": int(idx),
        }

    log_idx = _kneedle_convex_decreasing(points, log_error=True)
    raw_idx = _kneedle_convex_decreasing(points, log_error=False)
    return {
        "enabled": True,
        "primary": "log_error",
        "log_error": _record("log_error", log_idx),
        "raw_linear": _record("raw_linear", raw_idx),
    }


def _row_metric(row: Mapping, metric: str) -> float | None:
    candidates: tuple[str, ...]
    if metric == "ucb":
        candidates = ("kl_ucb", "validation_kl_ucb", "last_token_kl_ucb", "last_token_kl", "kl")
    else:
        candidates = ("last_token_kl", "validation_kl", "kl")
    for key in candidates:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def measured_rows(
    results: Sequence[Mapping],
    *,
    metric: str = "kl",
) -> list[dict]:
    """Return finite measured KL/bpp rows sorted by bpp."""
    rows: list[dict] = []
    for row in results:
        kl = _row_metric(row, metric)
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
            # The surrogate the allocator optimized. validate_assignments_kl emits
            # it nested as mse.predicted_dloss_sum; a top-level surrogate_loss (e.g.
            # test fixtures or legacy rows) takes precedence when present. Without
            # this fallback the surrogate-vs-KL Spearman below silently never fires.
            "surrogate_loss": (
                row.get("surrogate_loss")
                if row.get("surrogate_loss") is not None
                else (row.get("mse") or {}).get("predicted_dloss_sum")
            ),
            "kl_repeats": list(row.get("kl_repeats", []) or []),
            "kl_std": row.get("kl_std"),
            "kl_stderr": row.get("kl_stderr"),
            "kl_ucb": row.get("kl_ucb", row.get("validation_kl_ucb")),
        })
    rows.sort(key=lambda r: (r["bpp"], r["kl"], r["label"]))
    return rows


def measured_frontier(
    results: Sequence[Mapping],
    *,
    metric: str = "kl",
    kl_noise_floor: float = 0.0,
) -> list[dict]:
    """Return non-dominated measured KL/bpp points sorted by bpp.

    A point is dominated when a lower-or-equal bpp assignment already has
    lower-or-equal KL. Kneedle should operate on this measured lower envelope,
    not on noisy interior points.
    """
    rows = measured_rows(results, metric=metric)
    frontier: list[dict] = []
    best_kl = float("inf")
    floor = max(float(kl_noise_floor), 0.0)
    for row in rows:
        if row["kl"] < best_kl - floor - 1e-12:
            frontier.append(row)
            best_kl = row["kl"]
    return frontier


def practical_knee(
    frontier: Sequence[Mapping],
    *,
    rel_eps: float = 0.005,
    abs_eps: float = 0.0,
    kl_noise_floor: float = 0.0,
) -> dict | None:
    if not frontier:
        return None
    best = min(frontier, key=lambda row: (float(row["kl"]), float(row["bpp"])))
    tol = max(
        float(abs_eps),
        float(kl_noise_floor),
        abs(float(best["kl"])) * max(float(rel_eps), 0.0),
    )
    eligible = [
        row for row in frontier
        if float(row["kl"]) <= float(best["kl"]) + tol + 1e-12
    ]
    chosen = min(eligible, key=lambda row: (float(row["bpp"]), float(row["kl"])))
    out = dict(chosen)
    out["best_kl_label"] = best["label"]
    out["best_kl"] = float(best["kl"])
    out["tolerance"] = float(tol)
    return out


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = [0.0 for _ in ordered]
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and ordered[end][0] == ordered[idx][0]:
            end += 1
        rank = (idx + end - 1) / 2.0
        for _value, original_idx in ordered[idx:end]:
            ranks[original_idx] = rank
        idx = end
    return ranks


def spearman_rank_correlation(rows: Sequence[Mapping]) -> float | None:
    paired = [
        (float(row["surrogate_loss"]), float(row["kl"]))
        for row in rows
        if row.get("surrogate_loss") is not None
        and math.isfinite(float(row["surrogate_loss"]))
        and math.isfinite(float(row["kl"]))
    ]
    if len(paired) < 3:
        return None
    xs = _rank([item[0] for item in paired])
    ys = _rank([item[1] for item in paired])
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return float(num / (den_x * den_y))


def worst_rank_inversion(rows: Sequence[Mapping]) -> dict | None:
    """Surface the single most-misranked pair of measured rows.

    Uses the same (surrogate_loss, kl) pairing as ``spearman_rank_correlation``
    so the two agree on which rows count. Returns the pair whose surrogate-rank
    ordering disagrees most strongly with the measured-KL-rank ordering, i.e.
    the pair maximizing ``rank_kl_gap`` among discordant pairs. This checks
    surrogate-vs-KL fidelity only; it says nothing about held-out PPL, which is
    measured post-export and is not joined here. Returns ``None`` when fewer than
    three usable pairs exist (same guard as the Spearman).
    """
    paired = [
        {
            "label": str(row.get("label") or row.get("path") or f"point[{idx}]"),
            "surrogate_loss": float(row["surrogate_loss"]),
            "kl": float(row["kl"]),
        }
        for idx, row in enumerate(rows)
        if row.get("surrogate_loss") is not None
        and math.isfinite(float(row["surrogate_loss"]))
        and math.isfinite(float(row["kl"]))
    ]
    if len(paired) < 3:
        return None
    sur_ranks = _rank([p["surrogate_loss"] for p in paired])
    kl_ranks = _rank([p["kl"] for p in paired])

    worst = None
    worst_gap = 0.0
    for i in range(len(paired)):
        for j in range(i + 1, len(paired)):
            # Discordant: surrogate orders i,j one way, KL the other way.
            sur_order = sur_ranks[i] - sur_ranks[j]
            kl_order = kl_ranks[i] - kl_ranks[j]
            if sur_order == 0.0 or kl_order == 0.0:
                continue
            if (sur_order > 0) == (kl_order > 0):
                continue  # concordant, no inversion
            gap = abs(kl_ranks[i] - kl_ranks[j])
            if gap > worst_gap:
                worst_gap = gap
                worst = (i, j)
    if worst is None:
        return None

    i, j = worst
    # Order the reported pair so "better" = lower surrogate_loss (predicted best).
    a, b = (paired[i], paired[j]) if paired[i]["surrogate_loss"] <= paired[j]["surrogate_loss"] else (paired[j], paired[i])
    direction = "worse" if a["kl"] > b["kl"] else "better"
    verdict = (
        f"surrogate ranked '{a['label']}' better than '{b['label']}' "
        f"(predicted_dloss {a['surrogate_loss']:.6g} < {b['surrogate_loss']:.6g}) "
        f"but measured KL was {direction} ({a['kl']:.6g} vs {b['kl']:.6g})"
    )
    return {
        "predicted_best_label": a["label"],
        "predicted_best_surrogate_loss": a["surrogate_loss"],
        "predicted_best_kl": a["kl"],
        "predicted_worse_label": b["label"],
        "predicted_worse_surrogate_loss": b["surrogate_loss"],
        "predicted_worse_kl": b["kl"],
        "rank_gap": float(worst_gap),
        "verdict": verdict,
    }


def leave_one_out_kneedle_diagnostic(
    frontier: Sequence[Mapping],
    selected: Mapping,
    *,
    tolerance_bpp: float = 0.1,
    kl_noise_floor: float = 0.0,
) -> dict:
    if len(frontier) < 4:
        return {"enabled": False, "reason": "too_few_frontier_points"}
    selected_bpp = float(selected["bpp"])
    selected_kl = float(selected["kl"])
    picks: list[dict] = []
    for idx in range(len(frontier)):
        subset = [dict(row) for j, row in enumerate(frontier) if j != idx]
        if len(subset) < 3:
            continue
        chosen = subset[_kneedle_convex_decreasing(subset)]
        picks.append({
            "dropped_label": frontier[idx]["label"],
            "selected_label": chosen["label"],
            "bpp": float(chosen["bpp"]),
            "kl": float(chosen["kl"]),
        })
    if not picks:
        return {"enabled": False, "reason": "no_leave_one_out_picks"}
    max_bpp_shift = max(abs(row["bpp"] - selected_bpp) for row in picks)
    max_kl_shift = max(abs(row["kl"] - selected_kl) for row in picks)
    stable = (
        max_bpp_shift <= max(float(tolerance_bpp), 0.0) + 1e-12
        and max_kl_shift <= max(float(kl_noise_floor), 0.0) + 1e-12
    )
    return {
        "enabled": True,
        "stable": bool(stable),
        "max_bpp_shift": float(max_bpp_shift),
        "max_kl_shift": float(max_kl_shift),
        "tolerance_bpp": float(tolerance_bpp),
        "kl_noise_floor": float(kl_noise_floor),
        "picks": picks,
    }


def select_frontier_point(
    results: Sequence[Mapping],
    *,
    mode: str = "kneedle",
    metric: str = "kl",
    kl_noise_floor: float = 0.0,
    practical_rel_eps: float = 0.005,
    practical_abs_eps: float = 0.0,
    knee_tolerance_bpp: float = 0.1,
    unstable_policy: str = "keep-kneedle",
    sat_z: float = 2.0,
) -> tuple[dict, list[dict]]:
    frontier = measured_frontier(
        results,
        metric=metric,
        kl_noise_floor=kl_noise_floor,
    )
    if not frontier:
        raise ValueError("no finite measured KL/bpp points found")
    if mode == "best-kl":
        idx = min(range(len(frontier)), key=lambda i: (frontier[i]["kl"], frontier[i]["bpp"]))
    elif mode == "saturation":
        idx, _sat = _saturation_pick(frontier, sat_z)
    elif mode == "lowest-bpp":
        idx = 0
    elif mode == "practical-knee":
        practical = practical_knee(
            frontier,
            rel_eps=practical_rel_eps,
            abs_eps=practical_abs_eps,
            kl_noise_floor=kl_noise_floor,
        )
        idx = next(
            i for i, row in enumerate(frontier)
            if practical is not None and row["label"] == practical["label"]
        )
    elif mode == "kneedle":
        idx = _kneedle_convex_decreasing(frontier)
        diagnostic = leave_one_out_kneedle_diagnostic(
            frontier,
            frontier[idx],
            tolerance_bpp=knee_tolerance_bpp,
            kl_noise_floor=kl_noise_floor,
        )
        if diagnostic.get("enabled") and not diagnostic.get("stable", True):
            if unstable_policy == "best-kl":
                idx = min(range(len(frontier)), key=lambda i: (frontier[i]["kl"], frontier[i]["bpp"]))
            elif unstable_policy == "practical-knee":
                practical = practical_knee(
                    frontier,
                    rel_eps=practical_rel_eps,
                    abs_eps=practical_abs_eps,
                    kl_noise_floor=kl_noise_floor,
                )
                idx = next(
                    i for i, row in enumerate(frontier)
                    if practical is not None and row["label"] == practical["label"]
                )
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
        choices=("kneedle", "best-kl", "lowest-bpp", "practical-knee", "saturation"),
        default="kneedle",
        help="Frontier pick. 'saturation' = unconstrained bit-rate selector: "
             "lowest bpp whose KL is within --sat-z stderr of the high-bpp "
             "asymptote (needs a real per-bpp stderr, i.e. validate with "
             "--calib-repeats>=4). 'kneedle' is axis-dependent and a diagnostic "
             "on a log-linear RD curve.",
    )
    parser.add_argument(
        "--metric",
        choices=("kl", "ucb"),
        default="kl",
        help="Metric used for frontier construction. 'ucb' uses kl_ucb when present.",
    )
    parser.add_argument("--kl-noise-floor", type=float, default=0.0)
    parser.add_argument("--sat-z", type=float, default=2.0,
                        help="Significance multiplier on the combined per-bpp "
                             "stderr for --mode saturation (2.0 ~= 95%).")
    parser.add_argument("--practical-rel-eps", type=float, default=0.005)
    parser.add_argument("--practical-abs-eps", type=float, default=0.0)
    parser.add_argument("--knee-tolerance-bpp", type=float, default=0.1)
    parser.add_argument(
        "--unstable-policy",
        choices=("keep-kneedle", "best-kl", "practical-knee"),
        default="keep-kneedle",
    )
    parser.add_argument("--output-layer-config", required=True)
    parser.add_argument("--output-assignment", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(argv)

    payload = _load_json(args.validation_json)
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list):
        raise ValueError("--validation-json must contain a results list")

    selected, frontier = select_frontier_point(
        results,
        mode=args.mode,
        metric=args.metric,
        kl_noise_floor=args.kl_noise_floor,
        practical_rel_eps=args.practical_rel_eps,
        practical_abs_eps=args.practical_abs_eps,
        knee_tolerance_bpp=args.knee_tolerance_bpp,
        unstable_policy=args.unstable_policy,
        sat_z=args.sat_z,
    )
    saturation = None
    if args.mode == "saturation":
        if args.metric == "ucb":
            # The band is z * combined stderr; with metric=ucb the frontier 'kl'
            # is already mean+k*stderr, so the band would double-count the noise.
            # Saturation wants the raw mean — warn rather than silently inflate.
            print("[frontier-select] WARNING: --mode saturation with --metric "
                  "ucb double-counts uncertainty (UCB already folds stderr into "
                  "kl, and the saturation band re-adds it); use --metric kl.",
                  flush=True)
        _sidx, saturation = _saturation_pick(frontier, args.sat_z)
    practical = practical_knee(
        frontier,
        rel_eps=args.practical_rel_eps,
        abs_eps=args.practical_abs_eps,
        kl_noise_floor=args.kl_noise_floor,
    )
    knee_cmp = kneedle_comparison(frontier)
    loo = (
        leave_one_out_kneedle_diagnostic(
            frontier,
            selected,
            tolerance_bpp=args.knee_tolerance_bpp,
            kl_noise_floor=args.kl_noise_floor,
        )
        if args.mode == "kneedle"
        else {"enabled": False, "reason": "mode_not_kneedle"}
    )
    diagnostic_rows = measured_rows(results, metric=args.metric)
    rank_corr = spearman_rank_correlation(diagnostic_rows)
    worst_inversion = worst_rank_inversion(diagnostic_rows)
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
        "metric": args.metric,
        "selected": selected,
        "frontier": frontier,
        "practical_knee": practical,
        "kneedle_comparison": knee_cmp,
        "leave_one_out": loo,
        "saturation": saturation,
        "surrogate_spearman": rank_corr,
        "surrogate_worst_rank_inversion": worst_inversion,
        "kl_noise_floor": float(args.kl_noise_floor),
        "practical_rel_eps": float(args.practical_rel_eps),
        "practical_abs_eps": float(args.practical_abs_eps),
        "unstable_policy": args.unstable_policy,
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
    if saturation is not None:
        print(
            "[frontier-select] saturation B*="
            f"{saturation['bpp']:.6f} (KL={saturation['kl_at_bstar']:.8g}, "
            f"asymptote@{saturation['asymptote_bpp']:.4f}="
            f"{saturation['kl_asymptote']:.8g}, z={saturation['z']}, "
            f"{saturation['n_measurements']} probes)",
            flush=True,
        )
        if saturation.get("no_noise_floor"):
            print(
                "[frontier-select] WARNING: frontier has no positive per-bpp "
                "stderr -> saturation band is 0 and B* collapsed to the "
                "asymptote (most bits). Re-run validate_assignments_kl with "
                "--calib-repeats>=4 for a real noise floor.",
                flush=True,
            )
    if knee_cmp.get("enabled"):
        log_k = knee_cmp["log_error"]
        raw_k = knee_cmp["raw_linear"]
        print(
            "[frontier-select] kneedle log-error="
            f"{log_k['label']}@{log_k['bpp']:.6f} "
            f"raw-linear={raw_k['label']}@{raw_k['bpp']:.6f}",
            flush=True,
        )
    if rank_corr is not None:
        print(
            "[frontier-select] surrogate-vs-KL fidelity: "
            f"spearman={rank_corr:.4f} (1.0=perfect, surrogate-vs-KL only)",
            flush=True,
        )
        if worst_inversion is not None:
            print(
                f"[frontier-select] worst rank-inversion: {worst_inversion['verdict']}",
                flush=True,
            )
    else:
        print(
            "[frontier-select] surrogate-vs-KL fidelity: unavailable "
            "(need >=3 measured points carrying predicted_dloss_sum)",
            flush=True,
        )
    print(f"[frontier-select] layer_config -> {layer_config_path}", flush=True)
    print(f"[frontier-select] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
