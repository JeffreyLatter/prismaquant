#!/usr/bin/env python3
"""Build allocator cost pickles from a propagated sensitivity report."""
from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant.propagated_sensitivity_costs import (
    apply_propagated_sensitivity_penalty,
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


def _load_cost_payload(path: str | Path) -> tuple[Mapping[str, object], dict | None]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    if isinstance(payload.get("costs"), Mapping):
        return payload["costs"], dict(payload)
    return payload, None


def _parse_scales(value: str) -> list[float]:
    scales = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not scales:
        raise ValueError("at least one scale is required")
    return scales


def _scale_slug(scale: float) -> str:
    text = f"{float(scale):g}".replace("-", "m").replace(".", "p")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inject propagated serving-unit sensitivity into render-cost "
            "pickles so the regular allocator can account for cross-layer "
            "error amplification."
        )
    )
    parser.add_argument("--costs", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--sensitivity-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", default="0.25,0.5,1,2,5,10")
    parser.add_argument("--output-prefix", default="cost_propagated_scale")
    parser.add_argument("--score-field", default="propagated_kl")
    parser.add_argument("--target-format", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)

    costs, payload = _load_cost_payload(args.costs)
    stats = _load_pickle_mapping(args.probe, "stats")
    report_path = Path(args.sensitivity_report)
    report = json.loads(report_path.read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for scale in _parse_scales(args.scales):
        adjusted_costs, summary = apply_propagated_sensitivity_penalty(
            costs,
            stats=stats,
            report=report,
            scale=scale,
            target_format=args.target_format,
            score_field=args.score_field,
        )
        out_payload = adjusted_costs
        if payload is not None:
            out_payload = dict(payload)
            out_payload["costs"] = adjusted_costs
            meta = dict(out_payload.get("meta") or {})
            meta["propagated_sensitivity_costs"] = summary
            out_payload["meta"] = meta

        out_path = out_dir / f"{args.output_prefix}_{_scale_slug(scale)}.pkl"
        with out_path.open("wb") as fh:
            pickle.dump(out_payload, fh)
        row = {
            "scale": float(scale),
            "path": str(out_path),
            "adjusted_entries": summary["adjusted_entries"],
            "skipped": summary["skipped"],
            "total_scaled_member_penalty": summary["total_scaled_member_penalty"],
        }
        outputs.append(row)
        print(
            f"scale={scale:g} wrote {out_path} "
            f"adjusted={row['adjusted_entries']} "
            f"skipped={row['skipped']} "
            f"penalty={row['total_scaled_member_penalty']:.6g}",
            flush=True,
        )

    manifest = {
        "schema": "prismaquant.propagated_serving_unit_cost_sweep.v1",
        "base_cost": str(args.costs),
        "probe": str(args.probe),
        "source_report": str(report_path),
        "scales": _parse_scales(args.scales),
        "measured_units": len(report.get("rows", ())),
        "score_field": str(args.score_field),
        "target_format": args.target_format or report.get("target_format"),
        "total_unscaled_propagated_kl": float(
            sum(float(row.get(args.score_field, 0.0) or 0.0) for row in report.get("rows", ()))
        ),
        "penalty_distribution": (
            "member added-bit share; format local-output-mse ratio to current "
            "format; fused unit sums once after allocator aggregation"
        ),
        "outputs": outputs,
    }
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "propagated_cost_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
