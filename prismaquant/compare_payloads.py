"""Compare two Block-CLADO payloads (e.g., four-term vs Output-Fisher).

Usage::

    python -m prismaquant.compare_payloads \\
        --left  path/to/block_clado.json    \\
        --right path/to/output_fisher.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from prismaquant import block_clado as bc


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _collect_unary(payload: dict) -> dict[tuple[str, str], float]:
    blocks, singletons, _ = bc.parse_payload(payload)
    out: dict[tuple[str, str], float] = {}
    for unit_list in blocks.values():
        for unit in unit_list:
            for opt in unit.options:
                out[(unit.name, opt.fmt)] = float(opt.omega_ii)
    for unit in singletons:
        for opt in unit.options:
            out[(unit.name, opt.fmt)] = float(opt.omega_ii)
    return out


def _collect_pairs(payload: dict) -> dict[tuple[str, str, str, str], float]:
    _blocks, _singletons, pairs_by_block = bc.parse_payload(payload)
    out: dict[tuple[str, str, str, str], float] = {}
    for plist in pairs_by_block.values():
        for pair in plist:
            for (fa, fb), value in pair.omega_ij.items():
                out[(pair.unit_a, pair.unit_b, fa, fb)] = float(value)
    return out


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    rank_a = sorted(range(len(a)), key=lambda i: a[i])
    rank_b = sorted(range(len(b)), key=lambda i: b[i])
    rank_a_pos = {orig: r for r, orig in enumerate(rank_a)}
    rank_b_pos = {orig: r for r, orig in enumerate(rank_b)}
    n = len(a)
    d2 = sum((rank_a_pos[i] - rank_b_pos[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--label-left", default="left")
    parser.add_argument("--label-right", default="right")
    args = parser.parse_args(argv)

    left = _load(args.left)
    right = _load(args.right)

    left_unary = _collect_unary(left)
    right_unary = _collect_unary(right)
    common_keys = sorted(left_unary.keys() & right_unary.keys())

    print(
        f"=== unary Ω_ii: {args.label_left} vs {args.label_right} "
        f"({len(common_keys)} common) ==="
    )
    if common_keys:
        deltas = []
        ratios = []
        l_vals = []
        r_vals = []
        for k in common_keys:
            l = left_unary[k]
            r = right_unary[k]
            l_vals.append(l)
            r_vals.append(r)
            deltas.append(r - l)
            denom = max(abs(l), 1e-12)
            ratios.append((r - l) / denom)
        d_mean = statistics.fmean(deltas)
        d_std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        rho = _spearman(l_vals, r_vals)
        print(f"  mean Δ = {d_mean:+.6g}  std Δ = {d_std:.6g}  Spearman ρ = {rho:.3f}")
        # 5 largest absolute differences for inspection
        ranked = sorted(zip(common_keys, deltas, ratios), key=lambda x: -abs(x[1]))
        print(f"  Top 5 by |Δ|:")
        for (unit_name, fmt), d, ratio in ranked[:5]:
            l = left_unary[(unit_name, fmt)]
            r = right_unary[(unit_name, fmt)]
            print(
                f"    {unit_name:60s} @{fmt}  "
                f"L={l:+.4g}  R={r:+.4g}  Δ={d:+.4g}  Δ/L={ratio:+.2%}"
            )

    left_pairs = _collect_pairs(left)
    right_pairs = _collect_pairs(right)
    common_pair_keys = sorted(left_pairs.keys() & right_pairs.keys())
    print(f"\n=== pair Ω_ij ({len(common_pair_keys)} common) ===")
    if common_pair_keys:
        deltas = []
        l_vals = []
        r_vals = []
        for k in common_pair_keys:
            l = left_pairs[k]
            r = right_pairs[k]
            l_vals.append(l)
            r_vals.append(r)
            deltas.append(r - l)
        d_mean = statistics.fmean(deltas)
        d_std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        rho = _spearman(l_vals, r_vals)
        print(f"  mean Δ = {d_mean:+.6g}  std Δ = {d_std:.6g}  Spearman ρ = {rho:.3f}")
        ranked = sorted(zip(common_pair_keys, deltas), key=lambda x: -abs(x[1]))
        print(f"  Top 5 by |Δ|:")
        for (a, b, fa, fb), d in ranked[:5]:
            l = left_pairs[(a, b, fa, fb)]
            r = right_pairs[(a, b, fa, fb)]
            print(
                f"    {a:40s} × {b:40s} @({fa},{fb})  "
                f"L={l:+.4g}  R={r:+.4g}  Δ={d:+.4g}"
            )

    print(f"\n=== meta ===")
    for label, p in [(args.label_left, left), (args.label_right, right)]:
        m = p.get("meta", {})
        method = m.get("method", "four_term")
        elapsed = m.get("elapsed_seconds", 0)
        print(
            f"  {label:>12s}: method={method:>13s} "
            f"elapsed={float(elapsed):.1f}s "
            f"forwards={m.get('n_perturbation_forwards', '?')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
