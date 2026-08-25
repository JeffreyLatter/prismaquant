#!/usr/bin/env python3
"""Analyze a multi-shot validation run.

Reads:
  <base>/baseline/artifacts/validated_frontier_kl.json
  <base>/comparison/kl_repeats.json   (preferred)  OR
  <base>/comparison/multishot_kl.json (single-rep fallback)
  <base>/multi_shot_manifest.json (in each multishot-* dir, for assignment provenance)

Prints a comparison table and writes a structured summary to
<base>/comparison/analysis.json + analysis.txt.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


def _budget_from_label(label: str) -> float | None:
    """Try a few patterns to extract a budget from a validation result label.

    Handles:
      - 'allocator_target_5p0000_achieved_...'  (orchestrator validated-surrogate manifest)
      - 'baseline__allocator_target_5p0000_achieved_...' (prefixed by orchestrator)
      - 'baseline_5.0' / 'multishot_5.0'        (manual labels)
    """
    m = re.search(r"target_([0-9p]+)", label)
    if m:
        return float(m.group(1).replace("p", "."))
    m = re.match(r"(?:baseline|multishot)_+([0-9.]+)$", label)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _by_budget(results, key: str):
    out: dict[float, dict] = {}
    for r in results:
        label = r["label"]
        tb = _budget_from_label(label)
        if tb is None:
            continue
        if key == "baseline":
            out[tb] = r
        elif key == "multishot":
            out[tb] = r
        elif key == "both":
            if label.startswith("baseline"):
                out.setdefault(tb, {})["baseline"] = r
            elif label.startswith("multishot"):
                out.setdefault(tb, {})["multishot"] = r
            elif label.startswith("allocator_target"):
                # bare orchestrator label without prefix → assume baseline
                out.setdefault(tb, {})["baseline"] = r
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base_dir", help="run directory: multi-shot-validate-*/")
    p.add_argument(
        "--repeats-source",
        choices=("kl_repeats", "multishot_kl", "auto"),
        default="auto",
        help="Which step-3 output to read. Default auto-picks kl_repeats if present.",
    )
    args = p.parse_args(argv)

    base = Path(args.base_dir).resolve()
    baseline_kl_path = base / "baseline/artifacts/validated_frontier_kl.json"
    comparison_dir = base / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Auto-pick the paired source. Try these in order:
    #   - comparison/kl_comparison.json (orchestrator output — both arms in one file)
    #   - comparison/kl_repeats.json    (manual rerun naming)
    #   - comparison/multishot_kl.json  (single-shot orchestrator output) +
    #                                   baseline/.../validated_frontier_kl.json
    kl_comparison_path = comparison_dir / "kl_comparison.json"
    kl_repeats_path = comparison_dir / "kl_repeats.json"
    multishot_kl_path = comparison_dir / "multishot_kl.json"
    if args.repeats_source == "kl_repeats" or (
        args.repeats_source == "auto" and kl_comparison_path.is_file()
    ):
        src = kl_comparison_path
        mode = "paired"
    elif args.repeats_source == "kl_repeats" or (
        args.repeats_source == "auto" and kl_repeats_path.is_file()
    ):
        src = kl_repeats_path
        mode = "paired"
    else:
        if not multishot_kl_path.is_file():
            raise SystemExit(
                f"None of {kl_comparison_path}, {kl_repeats_path}, "
                f"{multishot_kl_path} exists; nothing to analyze."
            )
        src = multishot_kl_path
        mode = "two-file"

    output_lines: list[str] = []

    def emit(line: str = ""):
        output_lines.append(line)
        print(line)

    emit(f"Multi-shot validation analysis")
    emit(f"Run: {base.name}")
    emit(f"Source: {src.name} (mode={mode})")
    emit()

    if mode == "paired":
        # Single file contains both arms' results
        data = json.loads(src.read_text())
        per = _by_budget(data.get("results", []), "both")
        cal_info = data.get("calibration", {})
        nsamp = cal_info.get("n_calib_samples")
        seqlen = cal_info.get("calib_seqlen")
        reps = cal_info.get("calib_repeats")
        emit(f"Calibration: N={nsamp} T={seqlen} × {reps} repeats")
    else:
        # Two-file mode: baseline from validated_frontier_kl, multishot from multishot_kl
        baseline_data = json.loads(baseline_kl_path.read_text())
        multishot_data = json.loads(src.read_text())
        baseline_per = _by_budget(baseline_data.get("results", []), "baseline")
        multishot_per = _by_budget(multishot_data.get("results", []), "multishot")
        per = {}
        for b, r in baseline_per.items():
            per.setdefault(b, {})["baseline"] = r
        for b, r in multishot_per.items():
            per.setdefault(b, {})["multishot"] = r
        cal_b = baseline_data.get("calibration", {})
        cal_m = multishot_data.get("calibration", {})
        emit(f"Calibration: baseline N={cal_b.get('n_calib_samples')} T={cal_b.get('calib_seqlen')} × {cal_b.get('calib_repeats')} repeats")
        emit(f"             multishot N={cal_m.get('n_calib_samples')} T={cal_m.get('calib_seqlen')} × {cal_m.get('calib_repeats')} repeats")

    emit()
    emit("=" * 92)
    emit(f"{'budget':>6} | {'baseline KL':>21} | {'multishot KL':>21} | {'ΔKL':>11} {'pct':>7} {'z':>6}")
    emit(f"{'-'*6:>6}-+-{'-'*21:>21}-+-{'-'*21:>21}-+-{'-'*11:>11} {'-'*7:>7} {'-'*6:>6}")
    n_wins = 0
    n_significant = 0
    total_pct = 0.0
    summary_rows: list[dict] = []
    paired_blocks: list[str] = []

    for budget in sorted(per.keys()):
        b = per[budget].get("baseline")
        m = per[budget].get("multishot")
        if not b or not m:
            emit(f"{budget:>6.2f} | (incomplete: baseline={bool(b)} multishot={bool(m)})")
            continue
        bk = b["last_token_kl"]
        bs = b.get("kl_stderr", 0.0)
        mk = m["last_token_kl"]
        ms = m.get("kl_stderr", 0.0)
        bv = b.get("kl_repeats", [bk])
        mv = m.get("kl_repeats", [mk])

        delta = mk - bk
        pct = 100.0 * delta / bk if bk else 0.0
        z = 0.0
        d_serr = 0.0
        diffs: list[float] = []
        if len(bv) > 1 and len(mv) == len(bv):
            diffs = [mi - bi for bi, mi in zip(bv, mv)]
            if len(diffs) > 1 and statistics.stdev(diffs) > 0:
                d_serr = statistics.stdev(diffs) / (len(diffs) ** 0.5)
                z = (sum(diffs) / len(diffs)) / d_serr
        elif len(bv) > 1 and len(mv) == 1:
            # baseline has repeats but multishot is single — fall back to unpaired
            pooled = b.get("kl_std", 0.0)
            z = delta / pooled if pooled else 0.0

        win_marker = "✓" if delta < 0 else "✗"
        sig_marker = " **" if abs(z) >= 2.0 else ""
        emit(
            f"{budget:>6.2f} | {bk:>10.5f} ± {bs:.5f} | "
            f"{mk:>10.5f} ± {ms:.5f} | "
            f"{delta:>+11.5f} {pct:>+6.2f}% {z:>+6.2f} {win_marker}{sig_marker}"
        )
        total_pct += pct
        if delta < 0:
            n_wins += 1
        if abs(z) >= 2.0:
            n_significant += 1
        summary_rows.append({
            "budget": budget,
            "baseline_kl_mean": bk,
            "baseline_kl_stderr": bs,
            "baseline_repeats": bv,
            "multishot_kl_mean": mk,
            "multishot_kl_stderr": ms,
            "multishot_repeats": mv,
            "delta_kl_mean": delta,
            "delta_kl_pct": pct,
            "paired_diff_stderr": d_serr,
            "paired_z": z,
            "paired_diffs": diffs,
            "win": delta < 0,
            "significant_wz2": abs(z) >= 2.0,
            "baseline_format_counts": b.get("format_counts"),
            "multishot_format_counts": m.get("format_counts"),
        })
        if diffs:
            paired_blocks.append(
                f"  {budget}: paired Δ = [" +
                ", ".join(f"{d:+.4f}" for d in diffs) +
                f"]  (wins: {sum(1 for d in diffs if d < 0)}/{len(diffs)})"
            )

    n_total = len(summary_rows)
    emit()
    if n_total:
        emit(f"Wins: {n_wins}/{n_total}  |  Mean ΔKL%: {total_pct/n_total:+.2f}%  |  |z|≥2: {n_significant}/{n_total}")
    if paired_blocks:
        emit()
        emit("Paired Δ (multishot − baseline per calibration repeat):")
        for ln in paired_blocks:
            emit(ln)

    # Format-counts diff (how much did the assignment actually move?)
    if summary_rows:
        emit()
        emit("Assignment movement (format counts):")
        for row in summary_rows:
            bc = row["baseline_format_counts"] or {}
            mc = row["multishot_format_counts"] or {}
            keys = sorted(set(bc) | set(mc))
            diff_str = "  ".join(
                f"{k}: {bc.get(k,0)}→{mc.get(k,0)} ({mc.get(k,0)-bc.get(k,0):+d})"
                for k in keys
            )
            emit(f"  {row['budget']}: {diff_str}")

    summary = {
        "schema": "prismaquant.multi_shot.analysis.v1",
        "base_dir": str(base),
        "source": str(src),
        "mode": mode,
        "n_budgets": n_total,
        "n_wins": n_wins,
        "n_significant_wins": n_significant,
        "mean_delta_kl_pct": (total_pct / n_total) if n_total else 0.0,
        "rows": summary_rows,
    }
    out_json = comparison_dir / "analysis.json"
    out_txt = comparison_dir / "analysis.txt"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    out_txt.write_text("\n".join(output_lines) + "\n")
    print()
    print(f"[analyze] wrote {out_json}")
    print(f"[analyze] wrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
