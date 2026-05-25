#!/usr/bin/env python3
"""Build an MSE-driven promoted layer_config from an existing assignment."""
from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant.layer_config import load_assignment
from prismaquant.mse_promotion import (
    build_mse_promotion_assignment,
    layer_config_from_assignment,
)


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


def _parse_categories(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote high-output-MSE groups in an existing layer_config, "
            "ranking candidates by MSE removed per added bit."
        )
    )
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output-layer-config", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--categories",
        default="linear_attn,self_attn",
        help="Comma-separated semantic categories eligible for promotion.",
    )
    parser.add_argument("--target-format", default="BF16")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-bpp-delta", type=float, default=None)
    budget.add_argument("--target-bpp", type=float, default=None)
    parser.add_argument(
        "--group-by",
        default="layer_category",
        choices=("name", "layer_category", "category"),
    )
    parser.add_argument(
        "--metric",
        default="output_mse_per_bit",
        choices=(
            "output_mse_per_bit",
            "output_mse",
            "predicted_dloss_per_bit",
            "weight_mse_per_bit",
        ),
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    result = build_mse_promotion_assignment(
        load_assignment(args.base_assignment),
        costs=_load_costs(args.costs),
        stats=_load_pickle_mapping(args.probe, "stats"),
        categories=_parse_categories(args.categories),
        target_format=args.target_format,
        max_bpp_delta=args.max_bpp_delta,
        target_bpp=args.target_bpp,
        group_by=args.group_by,
        metric=args.metric,
    )
    layer_config = layer_config_from_assignment(result["assignment"])
    report = result["report"]

    layer_config_path = Path(args.output_layer_config)
    report_path = Path(args.output_report)
    layer_config_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    layer_config_path.write_text(json.dumps(layer_config, indent=2, sort_keys=True) + "\n")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {layer_config_path}")
    print(f"wrote {report_path}")
    print(
        f"base_bpp={report['base_bpp']:.6f} "
        f"promoted_bpp={report['promoted_bpp']:.6f} "
        f"delta={report['actual_bpp_delta']:.6f}"
    )
    print(
        f"selected_groups={report['selected_group_count']} "
        f"selected_members={report['selected_member_count']} "
        f"output_mse_removed={report['selected_output_mse_removed']:.6g} "
        f"({report['selected_output_mse_removed_pct']:.2f}%)"
    )
    print("top selected:")
    for row in report["selected"][: max(int(args.top), 0)]:
        print(
            f"  {row['key']}: members={row['member_count']} "
            f"mse={row['output_mse_removed']:.6g} "
            f"bpp_delta={row['bpp_delta']:.6f} "
            f"score={row['score']:.6g} formats={row['current_formats']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
