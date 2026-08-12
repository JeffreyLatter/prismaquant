#!/usr/bin/env python
"""Bank accepted burn shards + books for the ON-LAW routed rungs (K28/K32).

WHY THIS EXISTS
---------------
The 112.69 GB DSv4-Flash allocation puts routed experts on FP8_CB_K28/K32.
Those are the only two rungs that are simultaneously

  * ON-LAW      -- k % 4 == 0, the gridbook K1.2 fused mid-M prefill law, whose
                   `type_size = 4k` packed-B TMA box needs a 16-byte multiple.
                   A uniform decode at an off-law rung is WRONG, not merely
                   unaligned.
  * BANKABLE    -- cb_banked_books.ROUTED_MOE_CBL_BANK_RUNGS caps routed books
                   at K28-K33 (the byte-exact source-payload ceiling).

Rendering a routed expert at rung K requires an ACCEPTED BURN SHARD whose
`content_guard` digests match the tensors the exporter is about to render:
`load_banked_cbl_book` never searches a directory, never trains, and never
falls back to a lattice.  A Lloyd book file on disk is NOT sufficient -- the
shard is the contract, the book is what the shard points at.

The a-fast burn banked full 129/129 coverage only at its two interpolation
anchors K29 and K35, and BOTH are off-law (29%4=1, 35%4=3).  Every other rung's
*cost* came from the law `logy = a0 + a1*K + phi_{K%4}`, and an interpolated
cost is not a book.  On-law coverage as measured 2026-08-11:

    K28: 27/129 accepted cells  ->  102 missing
    K32: 29/129 accepted cells  ->  100 missing

WHAT THIS TOOL DOES NOT DO
--------------------------
It does not measure allocator costs.  Costs come from the anchored-AURA
supersurrogate (probe of the underlying model, mapped onto formats through the
fitted error terms) -- this is deliberately NOT a measure-and-burn campaign.
This tool exists only to make the on-law rungs RENDERABLE.

WHY ONE RUNG PER CHAIN
----------------------
`_run_chain` over multiple rungs feeds each rung's reconstruction forward as the
next rung's `embed` (minchain) predecessor, and the per-expert winner is then
whichever arm scores lower weight_mse.  A cell whose winner is `embed` does not
carry the certified `cbl_poolb` measurement stamp, and `load_banked_cbl_book`
rejects it.  Running each rung as its OWN single-rung chain leaves no
predecessor, so the free (CBL) arm is the only arm and every banked cell is the
certified pooled-Lloyd product book.  The weights are loaded once per
(layer, projection) and reused across both rungs, so this costs I/O nothing.

RESUME
------
`_run_chain` validates and reuses any existing cell whose identity still
matches, so re-running is safe and skips completed work.  Cells are keyed on the
full 256-expert stack (`expert_ids=all_experts`, `full_encode_rungs=rung`),
which is exactly the digest the exporter computes.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import torch

from tools import dsv4_afast_burn as burn
from tools.dsv4_afast_campaign import load_layer_identity, load_projection

# The two on-law, bankable routed rungs.  Not configurable by accident: any
# other value is either off-law (breaks the fused mid-M decode) or outside
# ROUTED_MOE_CBL_BANK_RUNGS (unbankable).  --rungs exists to burn them one at a
# time, never to introduce a third.
ONLAW_ROUTED_RUNGS = (28, 32)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def acceptable_cell(path: Path) -> bool:
    """True when this existing shard can back a routed book as-is.

    `load_banked_cbl_book` validates the pass-tag SCHEMA, not the pass-tag
    value, so a scout/primary/backstop cell is as good as a full-layer one
    provided it clears the two things the loader actually checks:

      * the content_guard is keyed on the FULL 256-expert stack -- the a-fast
        campaign full-encoded only its `full_encode_rungs`, so a cell at an
        off-schedule rung may have been written under a sampled expert slice;
      * the measurement is the certified `cbl_poolb` arm.  Cells carrying
        `incumbent_sweep_noldlq` (the disabled-rung lattice policy) or no
        semantics stamp at all exist at these rungs and must be re-burned.
    """
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return False
    identity = payload.get("identity") or {}
    guard = identity.get("content_guard") or {}
    shape = guard.get("source_shape") or []
    if len(shape) != 3 or int(shape[0]) != burn.EXPERT_COUNT:
        return False
    if len(identity.get("encoded_expert_ids") or []) != burn.EXPERT_COUNT:
        return False
    semantics = (
        (payload.get("cell") or {}).get("timing") or {}
    ).get("measurement_semantics") or {}
    return (
        semantics.get("encoder") == "cbl_poolb"
        and semantics.get("ldlq") is False
        and semantics.get("scale_policy") == "one_shot_cand0"
    )


def accepted_shard(layer: int, projection: str, rung: int) -> Path | None:
    """The shard that can back this cell today, across every pass tag."""
    for tag in burn.BURN_PASS_TAGS.values():
        path = burn._burn_cell_path(layer, projection, tag, rung)
        if acceptable_cell(path):
            return path
    return None


def _missing_rungs(layer: int, projection: str, rungs) -> tuple[int, ...]:
    """Rungs with no acceptable cell yet for this (layer, projection)."""
    return tuple(
        rung for rung in rungs
        if accepted_shard(layer, projection, rung) is None
    )


def _burn_cell(*, layer, projection, data, rung) -> dict:
    """One single-rung chain -> one accepted, full-stack-keyed CBL cell."""
    cells = burn._run_chain(
        layer=layer,
        projection=projection,
        pass_tag=burn.BURN_PASS_TAGS["full_layer"],
        data=data,
        rungs=(rung,),
        expert_ids=tuple(range(burn.EXPERT_COUNT)),
        replay=False,
        full_encode_rungs=(rung,),
    )
    return cells[rung]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rungs", default=",".join(map(str, ONLAW_ROUTED_RUNGS)),
        help="comma-separated subset of the on-law routed rungs",
    )
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=burn.LAYER_COUNT)
    parser.add_argument(
        "--manifest",
        default=str(burn.RUN_ROOT / "onlaw-books" / "ONLAW_BOOK_BURN.jsonl"),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report the work list and exit without touching the GPU",
    )
    args = parser.parse_args()

    rungs = tuple(int(r) for r in args.rungs.split(",") if r.strip())
    illegal = [r for r in rungs if r not in ONLAW_ROUTED_RUNGS]
    if illegal:
        raise SystemExit(
            f"REFUSE: {illegal} is not an on-law bankable routed rung; "
            f"legal set is {list(ONLAW_ROUTED_RUNGS)} (k%4==0 AND <=K33)"
        )

    layers = range(args.start_layer, min(args.end_layer, burn.LAYER_COUNT))
    worklist = [
        (layer, projection, missing)
        for layer in layers
        for projection in PROJECTIONS
        if (missing := _missing_rungs(layer, projection, rungs))
    ]
    total_cells = sum(len(m) for _, _, m in worklist)
    print(
        f"[onlaw-books] {total_cells} cells to burn across "
        f"{len(worklist)} (layer, projection) loads; rungs={list(rungs)}",
        flush=True,
    )
    if args.dry_run:
        for layer, projection, missing in worklist[:12]:
            print(f"    L{layer:02d} {projection}: K{list(missing)}")
        if len(worklist) > 12:
            print(f"    ... {len(worklist) - 12} more")
        return 0
    if not worklist:
        print("[onlaw-books] nothing to do; every on-law cell is banked")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("REFUSE: on-law book burn requires CUDA")
    burn._configure()
    # `_configure()` is tuned for the a-fast COST burn, which runs the
    # throughput LDLQ route (16 feeder threads).  These books are canonical
    # packed ARTIFACTS -- the exporter renders through them -- and
    # `_validate_packed_ldlq_route_env` refuses every byte-changing packed
    # route, feeder threads included.  Overriding after `_configure()` so the
    # artifact ABI wins over the measurement tuning; the other three knobs
    # (batch-experts on, expert-batch 16, one stream) are already canonical.
    os.environ["PRISMAQUANT_CB_LDLQ_FEEDER_THREADS"] = "0"

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with burn.COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    model_to_shard, model_to_ckpt = burn._build_weight_map(str(burn.SOURCE))
    scale_map = burn._build_fp8_scale_inv_map(str(burn.SOURCE))

    started = time.time()
    done = 0
    with manifest_path.open("a") as manifest:
        for layer, projection, missing in worklist:
            _, verified = load_layer_identity(layer)
            data = load_projection(
                layer, projection, device=torch.device("cuda:0"),
                identity=verified["identity"], all_col_weights=all_col_weights,
                model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
                scale_map=scale_map,
            )
            try:
                for rung in missing:
                    cell_started = time.time()
                    cell = _burn_cell(
                        layer=layer, projection=projection,
                        data=data, rung=rung,
                    )
                    done += 1
                    semantics = (cell.get("timing") or {}).get(
                        "measurement_semantics"
                    ) or {}
                    row = {
                        "layer": layer, "projection": projection, "rung": rung,
                        "shard": str(burn._burn_cell_path(
                            layer, projection,
                            burn.BURN_PASS_TAGS["full_layer"], rung,
                        )),
                        "book_sha256": semantics.get("book_sha256"),
                        "book_path": semantics.get("book_path"),
                        "encoder": semantics.get("encoder"),
                        "ldlq": semantics.get("ldlq"),
                        "scale_policy": semantics.get("scale_policy"),
                        "warm_state_outcome": (
                            cell.get("timing") or {}
                        ).get("warm_state_outcome"),
                        "seconds": round(time.time() - cell_started, 2),
                    }
                    manifest.write(json.dumps(row, sort_keys=True) + "\n")
                    manifest.flush()
                    rate = (time.time() - started) / done
                    print(
                        f"[onlaw-books] {done}/{total_cells} "
                        f"L{layer:02d} {projection} K{rung} "
                        f"{row['seconds']}s enc={row['encoder']} "
                        f"eta={(total_cells - done) * rate / 3600:.2f}h",
                        flush=True,
                    )
            finally:
                del data
                torch.cuda.empty_cache()

    print(
        f"[onlaw-books] DONE {done} cells in "
        f"{(time.time() - started) / 3600:.2f}h -> {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
