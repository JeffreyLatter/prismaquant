#!/usr/bin/env python3
"""Summarize empirical sensitivity from a measured KL probe."""
from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant.sensitivity_response import (
    build_response_report,
    write_category_csv,
    write_unit_csv,
)


def _load_pickle_mapping(path: str | None, key: str) -> Mapping[str, object] | None:
    if not path:
        return None
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} did not contain mapping key {key!r}")
    return value


def _load_costs(path: str | None) -> Mapping[str, object] | None:
    if not path:
        return None
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    costs = payload.get("costs")
    if isinstance(costs, Mapping):
        return costs
    return payload


def _default_path(source: Path, suffix: str) -> Path:
    return source.with_name(source.stem + suffix)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert measured kl_sensitivity_probe rows into quantitative "
            "unit/category sensitivity coefficients."
        )
    )
    parser.add_argument("--kl-probe", required=True, help="kl_sensitivity_probe JSON")
    parser.add_argument("--costs", default=None, help="Optional cost.pkl for error denominators")
    parser.add_argument("--probe", default=None, help="Optional sensitivity probe pickle for stats")
    parser.add_argument("--target-format", default="BF16")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--units-csv", default=None)
    parser.add_argument("--categories-csv", default=None)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args(argv)

    source = Path(args.kl_probe)
    payload = json.loads(source.read_text())
    report = build_response_report(
        payload,
        costs=_load_costs(args.costs),
        stats=_load_pickle_mapping(args.probe, "stats"),
        target_format=args.target_format,
    )

    output_json = Path(args.output_json) if args.output_json else _default_path(
        source,
        f".sensitivity_{str(args.target_format).lower()}.json",
    )
    units_csv = Path(args.units_csv) if args.units_csv else _default_path(
        source,
        f".sensitivity_{str(args.target_format).lower()}_units.csv",
    )
    categories_csv = (
        Path(args.categories_csv)
        if args.categories_csv
        else _default_path(source, f".sensitivity_{str(args.target_format).lower()}_categories.csv")
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_unit_csv(units_csv, report["units"])
    write_category_csv(categories_csv, report["categories"])

    print(f"wrote {output_json}")
    print(f"wrote {units_csv}")
    print(f"wrote {categories_csv}")
    print("top categories:")
    for row in report["categories"][: max(int(args.top), 0)]:
        print(
            f"  {row['category']}: "
            f"gain={float(row['kl_gain_sum']):.6g} "
            f"pos_gain={float(row['positive_kl_gain_sum']):.6g} "
            f"gain/Gbit={row['kl_gain_per_gbit']} "
            f"enrich_vs_mse={row['positive_gain_enrichment_vs_output_mse']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
