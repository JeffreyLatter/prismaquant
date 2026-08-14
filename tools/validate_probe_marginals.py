#!/usr/bin/env python3
"""Validate the probe's marginal emission against a REAL probe run.

The marginal vectors (`fisher_row`, `fisher_col`, `g_sq_sum`, `act_sq_sum`,
`act_absmax`) are emitted by `incremental_probe.py` by default since 2026-08-14.
Until this tool is run they have only ever been exercised by synthetic unit
tests -- no real model, no real hooks, no real bf16 accumulation.

This checks the one property that must hold for the Sensitivity Card to mean
anything:

    sum_o fisher_row[o]  ==  sum_i fisher_col[i]  ==  h_trace_raw

all three being the same scalar `sum_{o,i} H[o,i]` summed in a different order.
The card's own `validate()` enforces it with a fixed tolerance; the point of
running it here is to discover the *empirical* tolerance on a real bf16 probe,
because the resident accumulation path is only exact at the deferred-sync sites.

Usage:
    python3 tools/validate_probe_marginals.py --probe WORK/artifacts/probe.pkl
    python3 tools/validate_probe_marginals.py --probe ... --json report.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from typing import Any

import numpy as np


VECTOR_KEYS = ("fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax")


def _load(path: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _stats_of(probe: dict[str, Any]) -> dict[str, Any]:
    for key in ("stats", "layer_stats", "linears"):
        node = probe.get(key)
        if isinstance(node, dict) and node:
            return node
    raise SystemExit(
        f"could not find a stats mapping in probe (top-level keys: "
        f"{sorted(probe)[:20]})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="path to probe.pkl")
    ap.add_argument("--json", default=None, help="write a JSON report here")
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="tolerance to REPORT against; this tool never fails on it, it "
        "measures the real distribution so a defensible value can be chosen",
    )
    args = ap.parse_args()

    probe = _load(args.probe)
    stats = _stats_of(probe)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, s in stats.items():
        if not isinstance(s, dict):
            continue
        present = [k for k in VECTOR_KEYS if s.get(k) is not None]
        if not present:
            missing.append(name)
            continue

        rec: dict[str, Any] = {"name": name, "vectors": present}

        row = s.get("fisher_row")
        col = s.get("fisher_col")
        h_raw = s.get("h_trace_raw")
        if row is not None and col is not None:
            row = np.asarray(row, dtype=np.float64)
            col = np.asarray(col, dtype=np.float64)
            r_sum = float(row.sum())
            c_sum = float(col.sum())
            rec["row_sum"] = r_sum
            rec["col_sum"] = c_sum
            rec["shape"] = [int(row.size), int(col.size)]
            denom = max(abs(r_sum), abs(c_sum), 1e-300)
            rec["row_vs_col_rel"] = abs(r_sum - c_sum) / denom
            if h_raw is not None and float(h_raw) != 0.0:
                h_raw = float(h_raw)
                rec["h_trace_raw"] = h_raw
                rec["row_vs_htrace_rel"] = abs(r_sum - h_raw) / max(abs(h_raw), 1e-300)
                rec["col_vs_htrace_rel"] = abs(c_sum - h_raw) / max(abs(h_raw), 1e-300)
            # non-negativity: every H[o,i] is a sum of squares
            rec["row_min"] = float(row.min())
            rec["col_min"] = float(col.min())
            rec["row_nonfinite"] = int((~np.isfinite(row)).sum())
            rec["col_nonfinite"] = int((~np.isfinite(col)).sum())
            # how much of the mass is in the tail -- the reason VECTOR_DTYPE
            # is float32 and not float16
            if r_sum > 0:
                srt = np.sort(row)[::-1]
                csum = np.cumsum(srt) / r_sum
                rec["row_frac_for_99pct"] = float(
                    (np.searchsorted(csum, 0.99) + 1) / max(1, row.size)
                )

        for k in ("g_sq_sum", "act_sq_sum", "act_absmax"):
            v = s.get(k)
            if v is not None:
                v = np.asarray(v, dtype=np.float64)
                rec[f"{k}_min"] = float(v.min())
                rec[f"{k}_nonfinite"] = int((~np.isfinite(v)).sum())
        rows.append(rec)

    if not rows:
        print("FAIL: no unit in this probe carries any marginal vector.")
        print("      Was the probe run with marginals enabled? "
              "(--emit-marginals / PRISMAQUANT_PROBE_MARGINALS, default on)")
        return 2

    def _col(key: str) -> list[tuple[str, float]]:
        """(name, value) pairs, kept together so an index can never misalign."""
        return [
            (r["name"], r[key])
            for r in rows
            if key in r and math.isfinite(r[key])
        ]

    rc = _col("row_vs_col_rel")
    rh = _col("row_vs_htrace_rel")
    ch = _col("col_vs_htrace_rel")

    summary = {
        "probe": args.probe,
        "units_total": len(stats),
        "units_with_marginals": len(rows),
        "units_without_marginals": len(missing),
        "reported_rtol": args.rtol,
    }

    def _stat(pairs: list[tuple[str, float]]) -> dict[str, Any]:
        if not pairs:
            return {"n": 0}
        names = [p[0] for p in pairs]
        arr = np.asarray([p[1] for p in pairs])
        worst = int(np.argmax(arr))  # index into `pairs`, so the name matches
        return {
            "n": int(arr.size),
            "max": float(arr.max()),
            "p99": float(np.percentile(arr, 99)),
            "median": float(np.median(arr)),
            "n_over_rtol": int((arr > args.rtol).sum()),
            "worst_unit": names[worst],
        }

    summary["identity"] = {
        "row_vs_col": _stat(rc),
        "row_vs_htrace": _stat(rh),
        "col_vs_htrace": _stat(ch),
    }
    summary["negatives"] = {
        "row": sum(1 for r in rows if r.get("row_min", 0.0) < 0.0),
        "col": sum(1 for r in rows if r.get("col_min", 0.0) < 0.0),
        "g_sq_sum": sum(1 for r in rows if r.get("g_sq_sum_min", 0.0) < 0.0),
        "act_sq_sum": sum(1 for r in rows if r.get("act_sq_sum_min", 0.0) < 0.0),
    }
    summary["nonfinite"] = {
        "row": sum(r.get("row_nonfinite", 0) for r in rows),
        "col": sum(r.get("col_nonfinite", 0) for r in rows),
        "g_sq_sum": sum(r.get("g_sq_sum_nonfinite", 0) for r in rows),
        "act_sq_sum": sum(r.get("act_sq_sum_nonfinite", 0) for r in rows),
    }
    tail = [v for _, v in _col("row_frac_for_99pct")]
    if tail:
        summary["row_frac_of_outputs_holding_99pct_of_fisher"] = {
            "median": float(np.median(tail)),
            "max": float(np.max(tail)),
        }

    print(json.dumps(summary, indent=2))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"summary": summary, "units": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")

    # Verdict: the identity is the load-bearing property.  Report, don't assume.
    worst = max(
        [s.get("max", 0.0) for s in summary["identity"].values() if s.get("n")],
        default=0.0,
    )
    bad = (
        summary["nonfinite"]["row"]
        or summary["nonfinite"]["col"]
        or summary["negatives"]["row"]
        or summary["negatives"]["col"]
    )
    print(
        f"\nworst marginal-identity relative deviation on a REAL probe: {worst:.3e}"
    )
    if bad:
        print("FAIL: non-finite or negative entries in a Fisher marginal.")
        return 2
    if summary["units_without_marginals"]:
        print(
            f"NOTE: {summary['units_without_marginals']} unit(s) carry no marginals "
            "-- check whether that accumulation site ran."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
