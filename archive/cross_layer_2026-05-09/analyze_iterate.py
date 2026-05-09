"""Pretty-print summary across iterate_block_clado iterations.

Reads ``<output_root>/summary.json`` plus the per-iteration validation/
polish JSONs and emits a compact table comparing surrogate kneedle vs
real-KL best-validated vs polished outcomes per round.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    root = Path(args.output_root)
    summary = json.loads((root / "summary.json").read_text())
    iters = summary.get("iterations", [])
    if not iters:
        raise RuntimeError(f"no iterations in {root / 'summary.json'}")

    print(f"=== iterate_block_clado @ {root} ===")
    print(f"{'iter':>4s}  {'centered_at':>16s}  {'kneedle bpp/cost':>22s}  "
          f"{'best_val bpp/KL':>22s}  {'polished KL':>11s}  "
          f"{'polish steps':>13s}  {'elapsed':>8s}")
    for r in iters:
        print(
            f"{r['iteration']:>4d}  {r['centered_at']:>16s}  "
            f"{r['kneedle_bpp']:6.4f}/{r['kneedle_surrogate_cost']:+8.4f}  "
            f"{r['best_validated_bpp']:6.4f}/{r['best_validated_kl']:6.4f}  "
            f"{r['polished_kl']:11.4f}  "
            f"{r['polish_steps']:>13d}  "
            f"{r['elapsed_seconds']:7.1f}s"
        )

    # Detail per iteration: validation rows + polish trace.
    for r in iters:
        ipath = root / f"iter_{r['iteration']}"
        if not ipath.exists():
            continue
        print()
        print(f"--- iter {r['iteration']} validation cone ---")
        validation = json.loads((ipath / "validation.json").read_text())
        for row in validation["rows"]:
            knee = " ←kneedle" if row["is_kneedle"] else ""
            counts = dict(sorted(Counter(row["assignment"].values()).items()))
            print(
                f"  bpp={row['bpp']:.4f}  surrogate={row['surrogate_cost']:+.4f}  "
                f"real_kl={row['real_kl']:.4f}  counts={counts}{knee}"
            )
        polish = json.loads((ipath / "polish.json").read_text())
        if polish.get("steps"):
            print(f"--- iter {r['iteration']} polish trace "
                  f"(initial {polish['initial_kl']:.4f} → final {polish['final_kl']:.4f}) ---")
            for s in polish["steps"]:
                print(
                    f"  pass {s['pass_index']}  {s['unit']}  {s['from_fmt']}→{s['to_fmt']}  "
                    f"KL {s['kl_before']:.4f} → {s['kl_after']:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
