"""Merge two Block-CLADO payloads, taking unary from one and pairs from another.

The hypothesis: Output-Fisher unary (cheap, accurate to second order, PSD)
+ four-term identity pairs (capture higher-order finite-difference effects)
should yield a more predictive surrogate than either alone.

Both payloads must be on the same model and calibration set, with the
same fused-sibling block structure.  The merged payload uses the
``prismaquant.block_clado.v1`` schema with ``meta.method="hybrid"``.

Usage::

    python -m prismaquant.merge_payloads \\
        --unary  output_fisher.json   \\
        --pairs  block_clado.json     \\
        --output hybrid.json
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from prismaquant import block_clado as bc
from prismaquant import format_registry as fr


def merge_payloads(unary_payload: dict, pair_payload: dict) -> dict:
    """Build a merged payload with unary Ω_ii from ``unary_payload`` and
    pair Ω_ij from ``pair_payload``.

    Validates that the block / unit structure agrees between the two.
    """
    left_blocks, left_singles, _left_pairs = bc.parse_payload(unary_payload)
    right_blocks, right_singles, right_pairs = bc.parse_payload(pair_payload)

    # Sanity: same set of blocks
    if set(left_blocks) != set(right_blocks):
        raise ValueError(
            f"block sets disagree: "
            f"unary has {sorted(left_blocks.keys())[:3]}... "
            f"pairs has {sorted(right_blocks.keys())[:3]}..."
        )

    # Build the new units: unary_payload's options carry the Ω_ii values
    # but we keep the same fused-group/qname structure.
    blocks_out: dict[str, list[bc.DecisionUnit]] = {}
    for block_id, left_units in left_blocks.items():
        right_units = {u.name: u for u in right_blocks[block_id]}
        new_units = []
        for unit in left_units:
            if unit.name not in right_units:
                raise ValueError(
                    f"unit {unit.name} missing from pair payload in block {block_id}"
                )
            r_unit = right_units[unit.name]
            if set(opt.fmt for opt in unit.options) != set(opt.fmt for opt in r_unit.options):
                raise ValueError(
                    f"unit {unit.name} has different format options between payloads"
                )
            # Use unary_payload's options unchanged (Ω_ii from unary)
            new_units.append(unit)
        blocks_out[block_id] = new_units

    singletons_out: list[bc.DecisionUnit] = []
    right_singles_by_name = {u.name: u for u in right_singles}
    for unit in left_singles:
        # Singletons have no pair edges, so just keep unary.  Skip if
        # missing from the pair payload (warn) — the pair payload may
        # not have lm_head etc. listed as singletons consistently.
        singletons_out.append(unit)

    # Use right_pairs (from the pair_payload) as-is.
    pairs_out: dict[str, list[bc.BlockPair]] = {
        bid: list(right_pairs.get(bid, []))
        for bid in blocks_out
    }

    left_meta = unary_payload.get("meta", {})
    right_meta = pair_payload.get("meta", {})
    meta = {
        "elapsed_seconds": float(left_meta.get("elapsed_seconds", 0.0))
                          + float(right_meta.get("elapsed_seconds", 0.0)),
        "n_calib_samples": left_meta.get("n_calib_samples"),
        "calib_seqlen": left_meta.get("calib_seqlen"),
        "formats": left_meta.get("formats"),
        "objective_metric": "hybrid_output_fisher_unary_plus_four_term_pairs",
        "loss": "teacher_student_kl",
        "method": "hybrid",
        "unary_method": left_meta.get("method", "unknown"),
        "pair_method": right_meta.get("method", "unknown"),
        "block_count": len(blocks_out),
        "singleton_count": len(singletons_out),
        "centered": False,
        "center_kl": 0.0,
    }
    return bc.units_and_pairs_to_payload(
        blocks=blocks_out,
        singletons=singletons_out,
        pairs_by_block=pairs_out,
        meta=meta,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unary", required=True, help="Source for Ω_ii values")
    parser.add_argument("--pairs", required=True, help="Source for Ω_ij values")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    unary = json.loads(Path(args.unary).read_text())
    pairs = json.loads(Path(args.pairs).read_text())
    merged = merge_payloads(unary, pairs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(
        f"[merge] wrote {out_path} "
        f"unary_from={merged['meta']['unary_method']} "
        f"pairs_from={merged['meta']['pair_method']} "
        f"blocks={merged['meta']['block_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
