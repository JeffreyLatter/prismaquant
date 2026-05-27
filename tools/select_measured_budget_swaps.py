#!/usr/bin/env python3
"""Build a layer_config from empirically measured budget-neutral swaps."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from prismaquant.budget_swaps import select_measured_budget_swaps
from prismaquant.layer_config import load_assignment
from prismaquant.mse_promotion import layer_config_from_assignment


def _measured_rows(payload: dict) -> list[dict]:
    rows = payload.get("ranked")
    if not isinstance(rows, list):
        rows = payload.get("rows", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument("--measurement-report", required=True)
    parser.add_argument("--output-layer-config", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--min-kl-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum positive KL improvement required per swap. "
            "Default 0 accepts any negative delta."
        ),
    )
    parser.add_argument(
        "--max-net-bits-increase",
        type=float,
        default=0.0,
        help=(
            "Allowed cumulative net bit increase. Default 0 keeps the selected "
            "set budget-neutral or bit-saving."
        ),
    )
    parser.add_argument(
        "--max-base-drift",
        type=float,
        default=None,
        help="Optional cap on paired swap-vs-base KL drift.",
    )
    parser.add_argument(
        "--max-swaps",
        type=int,
        default=0,
        help="Maximum selected swaps. Default 0 means no explicit cap.",
    )
    args = parser.parse_args(argv)

    assignment = load_assignment(args.base_assignment)
    measurement = json.loads(Path(args.measurement_report).read_text())
    result = select_measured_budget_swaps(
        assignment,
        _measured_rows(measurement),
        min_kl_improvement=float(args.min_kl_improvement),
        max_net_bits_increase=float(args.max_net_bits_increase),
        max_base_drift=args.max_base_drift,
        max_swaps=int(args.max_swaps),
    )

    layer_config_path = Path(args.output_layer_config)
    report_path = Path(args.output_report)
    layer_config_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    layer_config_path.write_text(
        json.dumps(
            layer_config_from_assignment(result["assignment"]),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        **{k: v for k, v in result.items() if k != "assignment"},
        "base_assignment": args.base_assignment,
        "measurement_report": args.measurement_report,
        "output_layer_config": str(layer_config_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {layer_config_path}")
    print(f"wrote {report_path}")
    print(
        f"selected={result['selected_count']} "
        f"members={result['selected_member_count']} "
        f"net_bits={result['selected_net_bits_delta']:.0f} "
        f"estimated_delta_kl={result['selected_delta_kl_vs_bf16_sum']:.8g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
