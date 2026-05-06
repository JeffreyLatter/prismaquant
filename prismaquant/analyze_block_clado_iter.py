"""Compare per-iteration results from iterate_block_clado.

Reads the run's ``summary.json`` plus per-iteration ``polish.json`` and
``validation.json`` files; renders a textual report quantifying the
benefit of each iteration of sandwich + polish.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _spearman(values_x, values_y) -> float:
    """Spearman ρ for paired sequences."""
    n = len(values_x)
    if n < 2:
        return 0.0
    rx = {id(v): i for i, v in enumerate(sorted(zip(values_x, values_y), key=lambda t: t[0]))}
    ry = {id(v): i for i, v in enumerate(sorted(zip(values_x, values_y), key=lambda t: t[1]))}
    pairs = list(zip(values_x, values_y))
    d2 = sum((rx[id(p)] - ry[id(p)]) ** 2 for p in pairs)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze iterate-block-clado run")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run = Path(args.run_dir)

    summary = json.loads((run / "summary.json").read_text())
    iters = summary.get("iterations", [])
    print(f"Iterations: {len(iters)}")
    print(f"{'iter':>4}  {'centered_at':>20}  {'kneedle bpp':>11}  "
          f"{'best valid KL':>13}  {'polished KL':>11}  "
          f"{'polish steps':>12}  {'elapsed':>10}")
    for r in iters:
        print(
            f"{r['iteration']:>4}  {str(r['centered_at']):>20}  "
            f"{r['kneedle_bpp']:>11.4f}  {r['best_validated_kl']:>13.4f}  "
            f"{r['polished_kl']:>11.4f}  {r['polish_steps']:>12d}  "
            f"{r['elapsed_seconds']:>9.1f}s"
        )
    if (best := summary.get("best_overall")):
        print()
        print(f"Best overall: iter {best['iteration']} "
              f"polished_kl={best['polished_kl']:.4f} "
              f"@ bpp={best['best_validated_bpp']:.4f}")

    print()
    # Surrogate-real correlation per iteration
    print("Per-iteration surrogate quality (Spearman ρ on the validation cone):")
    for iter_idx in range(len(iters)):
        iter_dir = run / f"iter_{iter_idx}"
        validation_path = iter_dir / "validation.json"
        if not validation_path.exists():
            continue
        validation = json.loads(validation_path.read_text())
        rows = validation.get("rows", [])
        if len(rows) < 3:
            continue
        rho = _spearman(
            [r["surrogate_cost"] for r in rows],
            [r["real_kl"] for r in rows],
        )
        kls = [r["real_kl"] for r in rows]
        kl_var = (max(kls) - min(kls))
        print(f"  iter {iter_idx}: ρ = {rho:+.3f}  "
              f"KL spread = [{min(kls):.4f}, {max(kls):.4f}]  "
              f"(spread = {kl_var:.4f})")

    print()
    # Polish trajectory per iteration
    print("Per-iteration polish trajectory:")
    for iter_idx in range(len(iters)):
        polish_path = run / f"iter_{iter_idx}" / "polish.json"
        if not polish_path.exists():
            continue
        polish = json.loads(polish_path.read_text())
        steps = polish.get("steps", [])
        improvement = polish.get("improvement", 0.0)
        print(f"  iter {iter_idx}: initial {polish['initial_kl']:.4f} → "
              f"final {polish['final_kl']:.4f} (Δ {improvement:+.4f}) "
              f"in {len(steps)} accepted moves, "
              f"{polish['n_kl_measurements']} KL measurements")
        for step in steps:
            print(f"      pass {step['pass_index']}: {step['unit']} "
                  f"{step['from_fmt']}→{step['to_fmt']} "
                  f"KL {step['kl_before']:.4f} → {step['kl_after']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
