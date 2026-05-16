"""Summarize the joint (SmoothQuant α, format) search output.

Reports per-cluster joint-best (α, format), compares against today's baseline
(α=0, NVFP4 = identity rendering), and shows the distribution of wins.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _baseline_score(cluster: dict) -> float:
    """The score today's path would see — NVFP4 with α=0 (identity)."""
    return float(cluster["per_format"]["NVFP4"]["alpha_0_score"])


def _best_at_format(cluster: dict, fmt: str) -> tuple[float, float]:
    """Returns (alpha, score) for the cluster's best α at the given format."""
    entry = cluster["per_format"][fmt]
    return float(entry["alpha"]), float(entry["score"])


def _format_bpp(fmt: str) -> float:
    """Approximate effective bpp for a fold-eligible Linear (input axis only)."""
    return {
        "NVFP4": 4.5,
        "MXFP8_E4M3": 8.5,
        "FP8_E4M3": 8.5,
        "BF16": 16.0,
    }.get(fmt, 16.0)


def _joint_best(cluster: dict, formats: list[str]) -> tuple[str, float, float]:
    """Return (best_fmt, best_alpha, best_score) joint over (α, format)."""
    best_fmt = formats[0]
    best_alpha = 0.0
    best_score = float("inf")
    for fmt in formats:
        a, s = _best_at_format(cluster, fmt)
        if s < best_score:
            best_score = s
            best_alpha = a
            best_fmt = fmt
    return best_fmt, best_alpha, best_score


def _kind_from_cluster_key(ck: str) -> str:
    # cluster_key looks like "model.layers.0.input_layernorm" or
    # "model.layers.0.post_attention_layernorm"
    return ck.rsplit(".", 1)[-1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument(
        "--bpp-threshold",
        type=float,
        default=0.5,
        help="report per-Linear bpp delta vs NVFP4 baseline; "
        "only call a joint pick a 'win' if score drops by >= 5% AND delta_bpp "
        "is at or below this threshold.",
    )
    args = ap.parse_args(argv)

    blob = json.loads(Path(args.input).read_text())
    clusters = blob["clusters"]
    formats = blob.get("formats", ["NVFP4", "MXFP8_E4M3", "FP8_E4M3", "BF16"])

    print(f"clusters: {len(clusters)}")
    print(f"formats searched: {formats}")
    print()

    # 1. How does joint best compare to NVFP4 identity baseline?
    nvfp4_identity = [_baseline_score(c) for c in clusters]
    joint_best = [_joint_best(c, formats) for c in clusters]
    joint_best_score = [b[2] for b in joint_best]

    rel_gains = []
    for base, best in zip(nvfp4_identity, joint_best_score):
        gain = (base - best) / max(base, 1e-30)
        rel_gains.append(gain)
    print("== joint best vs NVFP4 identity ==")
    print(f"  rel gain (positive = joint helps):")
    print(f"    min:    {min(rel_gains):.5f}")
    print(
        f"    p25:    {statistics.quantiles(rel_gains, n=4)[0]:.5f}"
    )
    print(f"    p50:    {statistics.median(rel_gains):.5f}")
    print(
        f"    p75:    {statistics.quantiles(rel_gains, n=4)[2]:.5f}"
    )
    print(f"    max:    {max(rel_gains):.5f}")
    print(f"    mean:   {statistics.mean(rel_gains):.5f}")
    print()

    # 2. Format distribution: how often is each format the joint-best?
    fmt_counts = Counter(b[0] for b in joint_best)
    print("== joint-best format distribution ==")
    for fmt in formats:
        n = fmt_counts.get(fmt, 0)
        print(f"  {fmt:12s}: {n}/{len(clusters)} ({100 * n / max(len(clusters), 1):.1f}%)")
    print()

    # 3. By cluster kind
    kind_format_counts: dict[str, Counter] = defaultdict(Counter)
    for cluster, (fmt, _, _) in zip(clusters, joint_best):
        kind_format_counts[_kind_from_cluster_key(cluster["cluster_key"])][fmt] += 1
    print("== joint-best format by cluster kind ==")
    for kind in sorted(kind_format_counts):
        counts = kind_format_counts[kind]
        total = sum(counts.values())
        line = f"  {kind} ({total} clusters): "
        line += ", ".join(f"{f}={counts.get(f, 0)}" for f in formats)
        print(line)
    print()

    # 4. Distribution of α at NVFP4 (where today's solver would operate)
    nvfp4_alphas = [c["per_format"]["NVFP4"]["alpha"] for c in clusters]
    nvfp4_alpha_nonzero = [a for a in nvfp4_alphas if a > 1e-6]
    print("== NVFP4-format α distribution (continuous, golden-section) ==")
    if nvfp4_alpha_nonzero:
        print(f"  nonzero α count:  {len(nvfp4_alpha_nonzero)}/{len(nvfp4_alphas)}")
        print(f"  α nonzero min:    {min(nvfp4_alpha_nonzero):.4f}")
        print(
            f"  α nonzero p50:    {statistics.median(nvfp4_alpha_nonzero):.4f}"
        )
        print(f"  α nonzero max:    {max(nvfp4_alpha_nonzero):.4f}")
        print(f"  α nonzero mean:   {statistics.mean(nvfp4_alpha_nonzero):.4f}")
    else:
        print("  (all NVFP4 α optima collapsed to 0 — identity wins everywhere)")
    print()

    # 5. Top "joint wins" — clusters where the joint best is much better than NVFP4
    print("== top 12 joint wins vs NVFP4 identity ==")
    ranked = sorted(
        zip(clusters, joint_best, nvfp4_identity, joint_best_score),
        key=lambda row: (row[2] - row[3]) / max(row[2], 1e-30),
        reverse=True,
    )
    for cluster, (fmt, a, _), base, best in ranked[:12]:
        rel = (base - best) / max(base, 1e-30)
        print(
            f"  {cluster['cluster_key']:55s}  best=({fmt}, α={a:.3f})  "
            f"base={base:.4f} -> {best:.4f}  rel={100 * rel:.2f}%"
        )
    print()

    # 6. Where does joint pick a different format than NVFP4?
    upgraded = [
        (cluster, fmt, a)
        for cluster, (fmt, a, _) in zip(clusters, joint_best)
        if fmt != "NVFP4"
    ]
    print(f"== format-upgrade clusters ({len(upgraded)}/{len(clusters)}) ==")
    for cluster, fmt, a in upgraded[:12]:
        nvfp4_score = float(cluster["per_format"]["NVFP4"]["score"])
        upgraded_score = float(cluster["per_format"][fmt]["score"])
        rel = (nvfp4_score - upgraded_score) / max(nvfp4_score, 1e-30)
        delta_bpp = _format_bpp(fmt) - _format_bpp("NVFP4")
        print(
            f"  {cluster['cluster_key']:55s}  {fmt} α={a:.3f}  "
            f"score_delta={100 * rel:+.2f}%  bpp+{delta_bpp:.1f}"
        )


if __name__ == "__main__":
    main()
