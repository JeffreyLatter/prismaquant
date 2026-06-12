"""AURA additivity gate: observe the cross-layer residual, per artifact.

AURA's one structural assumption is that per-Linear KL contributions add.
The 2026-06-09 measurements say that assumption is sound but has a boundary:
the fp32 residual is +5–12% of full KL on 0.6B (growing with total distortion)
and is a DIFFUSE sum of micro-couplings — only 3/1180 pairs significant — so
the single assignment-level residual

    residual = measured_end_KL(assignment) − Σ_i predicted_dloss_i

is the entire cross-layer story worth measuring. This module computes it with
an honest stderr and turns the trust-region question ("are we somewhere AURA's
assumptions hold?") into a per-run, recorded diagnostic instead of a
paper-level claim. A residual that blows out flags a regime departure
(low bpp, routing, new arch) before anyone trusts the allocation.

Stderr correctness: cost rows share the same K probes, so row errors are
correlated and √Σσ² understates the sum's noise. When the cost rows carry
``x2_per_probe`` (probe-aligned raw samples), the gate forms the per-probe
sums S_k = Σ_i x²_{i,k} and takes the exact stderr of 0.5·mean_k(S_k);
otherwise it falls back to the independence approximation and says so.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence

ZERO_COST_SOURCES = {"aura_passthrough_zero"}


def additivity_gate(
    cost_payload: Mapping,
    assignment: Mapping[str, str],
    measured_kl: float,
    *,
    measured_kl_stderr: float = 0.0,
) -> dict:
    """Compare an assignment's measured end-KL to AURA's additive prediction.

    Returns a dict with the predicted sum, its stderr (exact per-probe when
    available), the residual, the residual's z-score, and coverage accounting.
    Uncovered members (assignment entries with no cost row) are LISTED, never
    silently dropped — a large uncovered set invalidates the comparison.
    """
    costs = cost_payload["costs"]
    covered: list[tuple[str, str]] = []
    uncovered: list[str] = []
    zero_rows = 0
    per_probe_ok = True
    n_probes = int(cost_payload.get("n_probes", 0))

    for name, fmt in assignment.items():
        fmt = str(fmt).strip().upper()
        row = costs.get(name, {}).get(fmt)
        if row is None:
            uncovered.append(f"{name}|{fmt}")
            continue
        if row.get("cost_source") in ZERO_COST_SOURCES or (
                row.get("predicted_dloss", 0.0) == 0.0
                and "x2_per_probe" not in row):
            zero_rows += 1
            continue
        covered.append((name, fmt))
        if "x2_per_probe" not in row or len(row["x2_per_probe"]) != n_probes:
            per_probe_ok = False

    predicted_sum = sum(
        float(costs[n][f]["predicted_dloss"]) for n, f in covered)

    if covered and per_probe_ok and n_probes >= 2:
        # Exact correlated-sum stderr: per-probe totals across all rows.
        s = [0.0] * n_probes
        for n, f in covered:
            for k, x2 in enumerate(costs[n][f]["x2_per_probe"]):
                s[k] += x2
        mean_s = sum(s) / n_probes
        var_s = sum((v - mean_s) ** 2 for v in s) / (n_probes - 1)
        predicted_stderr = 0.5 * math.sqrt(var_s / n_probes)
        stderr_method = "per_probe_exact"
    else:
        predicted_stderr = math.sqrt(sum(
            float(costs[n][f].get("predicted_dloss_stderr", 0.0)) ** 2
            for n, f in covered))
        stderr_method = "independence_lower_bound"

    residual = float(measured_kl) - predicted_sum
    denom = math.sqrt(predicted_stderr ** 2 + float(measured_kl_stderr) ** 2)
    z = residual / denom if denom > 0 else float("inf") if residual else 0.0

    return {
        "schema": "prismaquant.aura_additivity_gate.v1",
        "measured_kl": float(measured_kl),
        "measured_kl_stderr": float(measured_kl_stderr),
        "predicted_sum": predicted_sum,
        "predicted_stderr": predicted_stderr,
        "stderr_method": stderr_method,
        "residual": residual,
        "residual_over_measured": (
            residual / measured_kl if measured_kl else 0.0),
        "residual_z": z,
        "n_covered": len(covered),
        "n_zero_cost": zero_rows,
        "uncovered": sorted(uncovered),
        "n_probes": n_probes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="AURA additivity gate: measured vs Σ predicted end-KL")
    p.add_argument("--costs", required=True, help="AURA cost.pkl")
    p.add_argument("--assignment", required=True,
                   help="layer_config.json (format-name or AutoRound dicts)")
    p.add_argument("--measured-kl", required=True, type=float)
    p.add_argument("--measured-kl-stderr", type=float, default=0.0)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    from prismaquant.layer_config import load_assignment
    with open(args.costs, "rb") as fh:
        payload = pickle.load(fh)
    assignment = load_assignment(args.assignment)
    result = additivity_gate(
        payload, assignment, args.measured_kl,
        measured_kl_stderr=args.measured_kl_stderr)
    text = json.dumps(result, indent=1)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
