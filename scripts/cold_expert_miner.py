#!/usr/bin/env python3
"""Future GPU phase for finding corpus contexts that activate cold experts.

TODO(GPU): run a production-faithful, truncated forward over a user-selected
corpus for each target layer L. Reuse PrismaQuant's streaming model loader and
activation/prefetch machinery; keep layers 0..L and the current corpus batch
resident/prefetched; record top-k router membership (not merely router scores)
for the cold expert ids; preserve token offsets, decoded context windows,
router ranks/scores, layer/expert ids, corpus provenance, and model/calibration
hashes; shard results incrementally; fail closed on cache misses or inadequate
GPU residency; and validate hits by replaying selected windows through the same
truncated path. Do not implement this by loading the full model or by adding a
parallel cache.

This file intentionally contains no model, tensor, CUDA, or container code.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cold-experts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="3-42", help="layer range or comma-separated ids")
    parser.add_argument("--router-top-k", type=int, default=6)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        "TODO(GPU): implement a resident-prefetched truncated forward through layers <= L; "
        "scan corpus batches for top-k router membership of cold experts; persist context, "
        "rank/score, corpus/model hashes, and resumable shards; replay hits for validation. "
        f"Parsed request: model={args.model}, layers={args.layers}, top_k={args.router_top_k}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
