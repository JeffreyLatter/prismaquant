#!/usr/bin/env python3
"""Summarize assignment coverage of propagated-sensitive units."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from prismaquant.layer_config import load_assignment
from prismaquant.sensitivity_coverage import summarize_sensitivity_coverage


def _parse_top_ns(value: str) -> list[int]:
    top_ns = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not top_ns:
        raise ValueError("at least one top-N value is required")
    return top_ns


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Given a propagated sensitivity report and a layer_config, show "
            "whether the top-sensitive serving units are still protected."
        )
    )
    parser.add_argument("--sensitivity-report", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--top-ns", default="10,20,40")
    parser.add_argument("--order-by", default="propagated_kl_per_added_bit")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.sensitivity_report).read_text())
    assignment = load_assignment(args.assignment)
    summary = summarize_sensitivity_coverage(
        report,
        assignment,
        top_ns=_parse_top_ns(args.top_ns),
        order_by=args.order_by,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out}")

    print(f"rows={summary['row_count']} order_by={summary['order_by']}")
    for n, row in summary["top"].items():
        print(
            f"top{n}: all_bf16={row['all_bf16_units']} "
            f"no_nvfp4={row['no_nvfp4_units']} "
            f"nvfp4={row['nvfp4_units']} "
            f"missing={row['missing_units']} "
            f"formats={row['format_units']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
