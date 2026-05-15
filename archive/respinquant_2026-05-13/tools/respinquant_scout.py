#!/usr/bin/env python3
"""Report whether a ReSpinQuant-style basis plan is vanilla-vLLM safe."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.respinquant import (
    assert_kernel_free,
    hidden_layers_from_config,
    make_global_rotation_plan,
    make_layerwise_respin_plan,
    ReSpinRuntimeAdapterRequired,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="HF model directory or config.json")
    source.add_argument("--layers", type=int, help="Number of decoder layers")
    parser.add_argument(
        "--mode",
        choices=("layerwise", "global"),
        default="layerwise",
        help="Basis plan to analyze. layerwise is ReSpinQuant-style; global is HALO-like.",
    )
    parser.add_argument(
        "--enable-layers",
        default="all",
        help="Layer set for layerwise mode: all, none, comma list, or ranges like 0-3,12.",
    )
    parser.add_argument(
        "--allow-runtime-adapter",
        action="store_true",
        help="Do not fail when the plan needs a runtime residual adapter.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    n_layers = args.layers if args.layers is not None else hidden_layers_from_config(args.model)
    if args.mode == "global":
        plan = make_global_rotation_plan(n_layers)
    else:
        plan = make_layerwise_respin_plan(n_layers, args.enable_layers)

    text = plan.to_json(indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    else:
        print(text)

    try:
        assert_kernel_free(plan, allow_runtime_adapter=args.allow_runtime_adapter)
    except ReSpinRuntimeAdapterRequired as exc:
        print(f"[respinquant-scout] rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
