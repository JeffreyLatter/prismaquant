"""Dense Pareto cone via exact-budget knapsack at specified bpp targets.

The λ-sweep produces a discrete Pareto frontier; adjacent λ values often pick
the same per-block options, so the frontier is sparse where the user wants
fine-grained control.  ``solve_budget`` is a multi-choice knapsack DP that
hits any reachable target bpp; running it at a list of bpps gives a much
denser real-KL surface for kneedle picking.

Output layout matches what ``validate_block_clado`` expects (one JSON per
candidate with ``label``, ``bpp``, ``assignment``, ``surrogate_cost``).
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from prismaquant.block_clado import (
    build_block_states,
    expand_sweep_row_to_linear_assignment,
    fill_bpp,
    load_payload,
    solve_budget,
    total_param_count,
)


def _label_for_bpp(bpp: float) -> str:
    return f"budget_bpp_{bpp:.4f}".replace(".", "p")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dense exact-budget cone")
    p.add_argument("--payload", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--bpps",
        required=True,
        help="comma-separated list of target bpps, e.g. 4.50,4.55,4.60,...",
    )
    p.add_argument(
        "--bit-precision-bits",
        type=float,
        default=None,
        help="DP bin width in bits (auto: total_params / 4 → ~0.25 bpp)",
    )
    p.add_argument(
        "--pin",
        nargs="*",
        default=["lm_head"],
        help="Decision-unit name fragments to pin to BF16 (default lm_head; "
        "vLLM ParallelLMHead doesn't accept compressed-tensors layout).",
    )
    args = p.parse_args(argv)

    payload = load_payload(args.payload)
    block_states = build_block_states(payload)
    total_params = total_param_count(payload)

    pin_tokens = [t for t in (args.pin or []) if t]
    if pin_tokens:
        n_pinned = 0
        for block_id, states in list(block_states.items()):
            should_pin = any(t in block_id for t in pin_tokens) or any(
                any(t in qn for qn in s.assignment.keys()) for s in states
                for t in pin_tokens
            )
            if not should_pin:
                continue
            bf16_only = [
                s for s in states
                if all(v == "BF16" for v in s.assignment.values())
            ]
            if not bf16_only:
                print(
                    f"[dense-cone] WARNING: pinned block {block_id!r} has no "
                    f"BF16-only state; leaving unconstrained",
                    flush=True,
                )
                continue
            block_states[block_id] = bf16_only
            n_pinned += 1
            print(
                f"[dense-cone] pinned {block_id} to BF16 ({len(states)} → 1 state)",
                flush=True,
            )
        if n_pinned == 0:
            print(f"[dense-cone] no blocks matched pin tokens {pin_tokens}",
                  flush=True)

    # Auto bin width: target ~20,000 bins regardless of model size.  Bin
    # width must be << bits of the smallest decision unit's cheapest state
    # or that unit's bit_increment rounds to 0 in the DP.  20,000 bins
    # gives ~5e-4 bpp resolution and ~25M ops for 4B-class models.
    avg_bits_per_param_at_floor = 4.0  # roughly NVFP4
    bin_w = float(args.bit_precision_bits) if args.bit_precision_bits else max(
        total_params * avg_bits_per_param_at_floor / 20000.0, 1.0
    )

    target_bpps = sorted({float(x.strip()) for x in args.bpps.split(",") if x.strip()})

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for tgt_bpp in target_bpps:
        bits_budget = float(tgt_bpp) * float(total_params)
        result = solve_budget(
            block_states,
            bits_budget=bits_budget,
            bit_precision_bits=bin_w,
        )
        if result is None:
            print(f"[dense-cone] bpp={tgt_bpp:.4f}: INFEASIBLE", flush=True)
            continue
        result = fill_bpp(result, total_params)
        per_linear = expand_sweep_row_to_linear_assignment(payload, result.assignment)
        label = _label_for_bpp(result.bpp)
        out_path = out_root / f"{label}.json"
        out_path.write_text(json.dumps({
            "schema": "prismaquant.block_clado.kneedle.v1",
            "label": label,
            "bpp": float(result.bpp),
            "bits_total": float(result.bits_total),
            "surrogate_cost": float(result.cost_total),
            "lambda": 0.0,
            "target_bpp": float(tgt_bpp),
            "assignment": per_linear,
        }, indent=2) + "\n")
        rows.append({
            "target_bpp": float(tgt_bpp),
            "achieved_bpp": float(result.bpp),
            "surrogate_cost": float(result.cost_total),
            "label": label,
        })
        print(
            f"[dense-cone] target={tgt_bpp:.4f}  achieved={result.bpp:.4f}  "
            f"surrogate_cost={result.cost_total:+.4f}",
            flush=True,
        )

    summary = out_root / "summary.json"
    summary.write_text(json.dumps({
        "schema": "prismaquant.block_clado.dense_cone.v1",
        "total_params": int(total_params),
        "bin_precision_bits": float(bin_w),
        "candidates": rows,
    }, indent=2) + "\n")
    print(f"[dense-cone] wrote {len(rows)} candidates to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
