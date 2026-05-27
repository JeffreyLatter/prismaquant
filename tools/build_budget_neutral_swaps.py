#!/usr/bin/env python3
"""Build budget-neutral swap candidates for empirical KL probing."""
from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant.budget_swaps import build_budget_neutral_swaps
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles import DefaultProfile, detect_profile


def _load_pickle_mapping(path: str | Path, key: str) -> Mapping[str, object]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} did not contain mapping key {key!r}")
    return value


def _load_costs(path: str | Path) -> Mapping[str, object]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    costs = payload.get("costs")
    if isinstance(costs, Mapping):
        return costs
    return payload


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--sensitivity-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--formats", default="NVFP4,MXFP8_E4M3,FP8_E4M3,BF16")
    parser.add_argument("--categories", default="linear_attn,self_attn,shared_expert")
    parser.add_argument("--group-by", default="serving_unit")
    parser.add_argument(
        "--promotion-target-format",
        default="next_higher",
        help="Format to promote sensitive units to, or 'next_higher'.",
    )
    parser.add_argument("--max-promotions", type=int, default=32)
    parser.add_argument("--max-demotions-per-swap", type=int, default=4)
    parser.add_argument("--max-swaps", type=int, default=64)
    parser.add_argument("--demotion-start-window", type=int, default=8)
    parser.add_argument("--max-net-bpp-increase", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        profile = detect_profile(args.model) if args.model else DefaultProfile()
    except Exception:
        profile = DefaultProfile()

    assignment = load_assignment(args.base_assignment)
    costs = _load_costs(args.costs)
    stats = _load_pickle_mapping(args.probe, "stats")
    report = json.loads(Path(args.sensitivity_report).read_text())
    payload = build_budget_neutral_swaps(
        assignment,
        costs=costs,
        stats=stats,
        propagated_report=report,
        formats=_split_csv(args.formats),
        categories=_split_csv(args.categories),
        profile=profile,
        group_by=args.group_by,
        promotion_target_format=args.promotion_target_format,
        max_promotions=int(args.max_promotions),
        max_demotions_per_swap=int(args.max_demotions_per_swap),
        max_swaps=int(args.max_swaps),
        demotion_start_window=int(args.demotion_start_window),
        max_net_bpp_increase=float(args.max_net_bpp_increase),
    )
    payload.update({
        "base_assignment": str(args.base_assignment),
        "costs": str(args.costs),
        "probe": str(args.probe),
        "sensitivity_report": str(args.sensitivity_report),
        "model": args.model,
    })
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} swaps={payload['swap_count']} "
        f"promotions={payload['promotion_candidate_count']} "
        f"demotions={payload['demotion_candidate_count']}",
        flush=True,
    )
    if payload["swaps"]:
        first = payload["swaps"][0]
        print(
            "top swap: "
            f"{first['key']} net_bits={first['net_bits_delta']:.0f} "
            f"score={first['score']:.6g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
