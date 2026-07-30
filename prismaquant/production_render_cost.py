#!/usr/bin/env python3
"""Build allocator costs from production-rendered reconstruction losses.

The local cost path estimates layer damage from a separate RTN-style cost
measurement and then, for measured output MSE, multiplies by
``0.5 * h_trace``.  Production cache fill already renders the weights the
export will ship and records the scorer objective used to accept GPTQ/JSO
and scale-sweep candidates.  This module turns those rendered scores into
allocator-compatible ``predicted_dloss`` entries.

The synthesized entries intentionally set ``output_mse_measured=False`` so
``allocator_candidates.cost_entry_predicted_dloss`` consumes
``predicted_dloss`` directly instead of applying the diagonal-Fisher proxy
again.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import format_registry as fr


SCHEMA = "prismaquant.production_render_score_cost.v1"


def canonical_cost_name(qname: str) -> str:
    name = str(qname)
    if name.endswith(".weight"):
        name = name[:-len(".weight")]
    prefix = "model.language_model."
    if name.startswith(prefix):
        name = "model." + name[len(prefix):]
    return name


def _record_key(qname: str, fmt: str) -> str:
    return f"{qname}|{fmt.upper()}"


def _load_pickle(path: str | Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _cache_render_score_records(cache: object) -> dict[tuple[str, str], dict]:
    meta = getattr(cache, "metadata", None)
    if not isinstance(meta, Mapping):
        return {}
    scores = meta.get("render_scores")
    if not isinstance(scores, Mapping):
        return {}
    records = scores.get("records")
    if not isinstance(records, Mapping):
        return {}

    out: dict[tuple[str, str], dict] = {}
    for raw_key, raw_record in records.items():
        if not isinstance(raw_record, Mapping):
            continue
        qname = canonical_cost_name(str(raw_record.get("qname", "")))
        fmt = fr.canonical_format_name(str(raw_record.get("format", "")))
        if not qname or not fmt:
            parts = str(raw_key).rsplit("|", 1)
            if len(parts) == 2:
                qname = canonical_cost_name(parts[0])
                fmt = fr.canonical_format_name(parts[1])
        if not qname or not fmt:
            continue
        out[(qname, fmt)] = dict(raw_record)
    return out


def _calibration_hashes(*sources: object) -> list[str]:
    """Union of R14 calibration identities carried by upstream artifacts.

    This module never sees calibration text — its cost comes from render
    scores the production cache already recorded — so the hash it stamps is
    inherited from the cache metadata (stamped by ``build_production_cache``)
    and from the baseline cost payload's meta/provenance. Returns ``[]`` when
    no upstream stamped one, which keeps the downstream disjointness check
    inert on pre-R14 artifacts rather than guessing.
    """
    found: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        single = source.get("calib_hash")
        if isinstance(single, str) and single:
            found.add(single)
        many = source.get("calib_hashes")
        if isinstance(many, Sequence) and not isinstance(many, (str, bytes)):
            found.update(str(item) for item in many if item)
    return sorted(found)


def _lookup_record(
    records: Mapping[tuple[str, str], Mapping],
    qname: str,
    fmt: str,
) -> Mapping | None:
    cname = canonical_cost_name(qname)
    for alias in fr.aliases_for(fmt):
        record = records.get((cname, fr.canonical_format_name(alias)))
        if record is not None:
            return record
    return None


def _score_value(record: Mapping, field: str) -> float | None:
    value = record.get(field)
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out) or out < 0.0:
        return None
    return out


def load_qnames_file(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        canonical_cost_name(line.strip())
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _production_cost_entry(
    record: Mapping,
    *,
    score_field: str,
) -> dict | None:
    metric = str(record.get("metric", "render_score"))
    if score_field in {"weight_mse_sum", "weight_mse"}:
        weight_mse = _score_value(record, "weight_mse")
        if weight_mse is None:
            return None
        # Hand the allocator the production-rendered weight_mse and leave
        # predicted_dloss/output_mse off the entry so its existing
        # h_trace * weight_mse fallback fires.
        return {
            "weight_mse": float(weight_mse),
            "output_mse_measured": False,
            "cost_source": "production_render_weight_mse",
            "render_score_metric": metric,
            "render_score": _score_value(record, "score"),
            "render_score_sum": _score_value(record, "score_sum"),
            "render_score_normalizer": _score_value(record, "normalizer"),
            "render_activation_rows": int(record.get("activation_rows", 0) or 0),
            "raw_render_metric": str(record.get("raw_render_metric", "")),
            "raw_render_score": _score_value(record, "raw_render_score"),
            "raw_render_score_sum": _score_value(record, "raw_render_score_sum"),
            "weight_mse_sum": _score_value(record, "weight_mse_sum"),
            "n_weights": int(record.get("n_weights", 0) or 0),
            "activation_quantized": bool(record.get("activation_quantized", False)),
            "activation_clipped": bool(record.get("activation_clipped", False)),
        }
    if score_field == "output_mse":
        # Use the production-rendered per-element output_mse (unweighted,
        # equals raw_render_score after Fisher row-weighting was dropped).
        # output_mse_measured=True triggers the allocator's path 1
        # (h_trace * output_mse), matching the original prismaquant cost
        # objective but on production-quality rendered weights instead of
        # naive RTN.
        output_mse = _score_value(record, "raw_render_score")
        if output_mse is None:
            output_mse = _score_value(record, "score")
        if output_mse is None:
            return None
        return {
            "output_mse": float(output_mse),
            "output_mse_measured": True,
            "weight_mse": float(_score_value(record, "weight_mse") or 0.0),
            "cost_source": "production_render_output_mse",
            "render_score_metric": metric,
            "render_score": _score_value(record, "score"),
            "render_score_sum": _score_value(record, "score_sum"),
            "render_score_normalizer": _score_value(record, "normalizer"),
            "render_activation_rows": int(record.get("activation_rows", 0) or 0),
            "raw_render_metric": str(record.get("raw_render_metric", "")),
            "raw_render_score": _score_value(record, "raw_render_score"),
            "raw_render_score_sum": _score_value(record, "raw_render_score_sum"),
            "weight_mse_sum": _score_value(record, "weight_mse_sum"),
            "n_weights": int(record.get("n_weights", 0) or 0),
            "activation_quantized": bool(record.get("activation_quantized", False)),
            "activation_clipped": bool(record.get("activation_clipped", False)),
        }
    score = _score_value(record, score_field)
    mean_score = _score_value(record, "score")
    if score is None:
        return None
    return {
        "predicted_dloss": float(score),
        "weight_mse": float(_score_value(record, "weight_mse") or 0.0),
        "output_mse": float(mean_score if mean_score is not None else 0.0),
        "rel_output_mse": 0.0,
        "output_mse_measured": False,
        "cost_source": "production_render_score",
        "render_score_metric": metric,
        "render_score": float(mean_score if mean_score is not None else score),
        "render_score_sum": _score_value(record, "score_sum"),
        "render_score_normalizer": _score_value(record, "normalizer"),
        "render_activation_rows": int(record.get("activation_rows", 0) or 0),
        "raw_render_metric": str(record.get("raw_render_metric", "")),
        "raw_render_score": _score_value(record, "raw_render_score"),
        "raw_render_score_sum": _score_value(record, "raw_render_score_sum"),
        "activation_quantized": bool(record.get("activation_quantized", False)),
        "activation_clipped": bool(record.get("activation_clipped", False)),
    }


def synthesize_production_render_cost_payload(
    production_cache: object,
    baseline_cost_payload: Mapping,
    *,
    formats: Sequence[str] | None = None,
    score_field: str = "score_sum",
    source_label: str | None = None,
    require_render_scores: bool = False,
    require_output_metric: bool = False,
    missing_render_score_policy: str = "fallback",
    promotion_qnames: set[str] | None = None,
    bf16_policy: str = "all",
) -> dict:
    records = _cache_render_score_records(production_cache)
    baseline_costs = dict(baseline_cost_payload["costs"])
    output_formats = [
        fr.canonical_format_name(str(fmt))
        for fmt in (
            formats
            if formats is not None
            else baseline_cost_payload.get("formats", [])
        )
    ]
    output_formats = list(dict.fromkeys(output_formats))

    output_costs: dict[str, dict[str, dict]] = {}
    render_entries = 0
    fallback_entries = 0
    missing: list[dict[str, str]] = []
    non_output_metric: list[dict[str, str]] = []

    for qname, per_name_raw in baseline_costs.items():
        cname = canonical_cost_name(str(qname))
        per_name = dict(per_name_raw)
        synthesized: dict[str, dict] = {}
        for fmt in output_formats:
            fmt_c = fr.canonical_format_name(fmt)
            if fmt_c == "BF16":
                if (
                    bf16_policy == "promotion-set"
                    and promotion_qnames is not None
                    and cname not in promotion_qnames
                ):
                    synthesized[fmt_c] = {
                        "error": "bf16_not_in_staged_promotion_set",
                        "cost_source": "unavailable_staged_bf16",
                    }
                    continue
                synthesized[fmt_c] = {
                    "predicted_dloss": 0.0,
                    "weight_mse": 0.0,
                    "output_mse": 0.0,
                    "rel_output_mse": 0.0,
                    "output_mse_measured": False,
                    "cost_source": "bf16_zero",
                }
                continue

            record = _lookup_record(records, qname, fmt_c)
            if record is not None:
                metric = str(record.get("metric", ""))
                if require_output_metric and metric not in {
                    "output_mse",
                    "fisher_output_mse",
                }:
                    non_output_metric.append({
                        "qname": str(qname),
                        "format": fmt_c,
                        "metric": metric,
                    })
                else:
                    entry = _production_cost_entry(
                        record,
                        score_field=score_field,
                    )
                    if entry is not None:
                        synthesized[fmt_c] = entry
                        render_entries += 1
                        continue

            missing.append({"qname": str(qname), "format": fmt_c})
            if missing_render_score_policy == "unavailable":
                synthesized[fmt_c] = {
                    "error": "missing production render score",
                    "cost_source": "unavailable_missing_render_score",
                }
                fallback_entries += 1
                continue
            fallback = None
            for alias in fr.aliases_for(fmt_c):
                if alias in per_name:
                    fallback = dict(per_name[alias])
                    break
            if fallback is None and fmt_c in per_name:
                fallback = dict(per_name[fmt_c])
            if fallback is None:
                fallback = {"error": "missing production render score"}
            else:
                fallback["cost_source"] = fallback.get(
                    "cost_source",
                    "fallback_baseline",
                )
            synthesized[fmt_c] = fallback
            fallback_entries += 1
        output_costs[str(qname)] = synthesized

    if (require_render_scores or missing_render_score_policy == "error") and missing:
        sample = ", ".join(
            f"{row['qname']}@{row['format']}" for row in missing[:8]
        )
        raise ValueError(
            f"missing {len(missing)} production render scores; sample={sample}"
        )
    if require_output_metric and non_output_metric:
        sample = ", ".join(
            f"{row['qname']}@{row['format']}:{row['metric']}"
            for row in non_output_metric[:8]
        )
        raise ValueError(
            "production render scores fell back to non-output metrics for "
            f"{len(non_output_metric)} entries; sample={sample}"
        )

    calib_hashes = _calibration_hashes(
        getattr(production_cache, "metadata", None),
        baseline_cost_payload.get("meta"),
        baseline_cost_payload.get("provenance"),
    )
    return {
        "schema": SCHEMA,
        "costs": output_costs,
        "formats": output_formats,
        "meta": {
            # R14: inherited calibration identity — see _calibration_hashes.
            "calib_hashes": calib_hashes,
            "calib_hash": calib_hashes[0] if len(calib_hashes) == 1 else None,
            "production_cache_source": source_label,
            "baseline_schema": baseline_cost_payload.get("schema"),
            "baseline_meta": baseline_cost_payload.get("meta"),
            "score_field": score_field,
            "missing_render_score_policy": missing_render_score_policy,
            "bf16_policy": bf16_policy,
            "promotion_qnames": (
                int(len(promotion_qnames))
                if promotion_qnames is not None else None
            ),
            "render_score_entries": int(render_entries),
            "fallback_entries": int(fallback_entries),
            "available_render_scores": int(len(records)),
            "missing_render_score_entries": int(len(missing)),
            "cost_semantics": (
                "predicted_dloss is copied directly from the production "
                "render score field; output_mse_measured is false so the "
                "allocator does not multiply by h_trace"
            ),
        },
    }


def select_tail_from_render_scores(
    production_cache: object,
    *,
    fmt: str = "NVFP4",
    score_field: str = "score_sum",
    top_fraction: float = 0.30,
    min_score: float | None = None,
    min_count: int = 1,
    max_count: int | None = None,
) -> tuple[list[str], dict[str, object]]:
    records = _cache_render_score_records(production_cache)
    fmt_c = fr.canonical_format_name(fmt)
    rows: list[tuple[str, float]] = []
    skipped = 0
    for (qname, record_fmt), record in records.items():
        if fr.canonical_format_name(record_fmt) != fmt_c:
            continue
        score = _score_value(record, score_field)
        if score is None:
            skipped += 1
            continue
        rows.append((qname, score))
    rows.sort(key=lambda item: item[1], reverse=True)
    if not rows:
        return [], {
            "format": fmt_c,
            "score_field": score_field,
            "available": 0,
            "selected": 0,
            "skipped": int(skipped),
        }

    frac = max(0.0, min(1.0, float(top_fraction)))
    target = max(int(math.ceil(len(rows) * frac)), int(min_count))
    if max_count is not None and max_count > 0:
        target = min(target, int(max_count))
    selected_rows = rows[:target]
    if min_score is not None:
        selected_rows = [
            row for row in selected_rows
            if float(row[1]) >= float(min_score)
        ]
    selected = [qname for qname, _score in selected_rows]
    scores = [score for _qname, score in rows]
    summary = {
        "format": fmt_c,
        "score_field": score_field,
        "available": int(len(rows)),
        "selected": int(len(selected)),
        "skipped": int(skipped),
        "top_fraction": float(frac),
        "min_score": min_score,
        "min_count": int(min_count),
        "max_count": max_count,
        "threshold_score": (
            float(selected_rows[-1][1]) if selected_rows else None
        ),
        "max_score": float(scores[0]),
        "min_score_available": float(scores[-1]),
        "selected_qnames_sample": selected[:8],
    }
    return selected, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize allocator costs from ProductionWeightCache render scores",
    )
    parser.add_argument("--production-cache", required=True)
    parser.add_argument("--baseline-cost", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--formats",
        default=None,
        help="Comma-separated formats. Defaults to baseline cost formats.",
    )
    parser.add_argument(
        "--score-field",
        choices=("output_mse", "weight_mse", "weight_mse_sum", "score_sum", "score"),
        default="output_mse",
        help="Cost source to feed the allocator. output_mse (default) hands "
        "the allocator the production-rendered per-element output_mse with "
        "output_mse_measured=True, so it computes h_trace * output_mse "
        "(the original prismaquant cost objective on production-quality "
        "rendered weights). weight_mse / weight_mse_sum emit weight_mse and "
        "let the allocator's h_trace * weight_mse fallback fire. score_sum / "
        "score emit the local render-gate score directly as predicted_dloss "
        "(legacy production-render-score behavior).",
    )
    parser.add_argument(
        "--require-render-scores",
        action="store_true",
        help="Fail instead of falling back to baseline costs when a non-BF16 "
        "format lacks a production render score.",
    )
    parser.add_argument(
        "--require-output-metric",
        action="store_true",
        help="Fail if any consumed render score is a weight_mse fallback "
        "instead of output_mse/fisher_output_mse.",
    )
    parser.add_argument(
        "--missing-render-score-policy",
        choices=("fallback", "unavailable", "error"),
        default="fallback",
        help="How to handle non-BF16 formats without production render "
        "scores. staged mode should use unavailable so unmeasured promotions "
        "do not fall back to proxy costs.",
    )
    parser.add_argument(
        "--promotion-qnames-file",
        default=None,
        help="Optional qname allowlist for staged promotions. Used with "
        "--bf16-policy=promotion-set.",
    )
    parser.add_argument(
        "--bf16-policy",
        choices=("all", "promotion-set"),
        default="all",
        help="Whether BF16 is available for every qname or only qnames listed "
        "in --promotion-qnames-file.",
    )
    parser.add_argument(
        "--select-tail-output",
        default=None,
        help="Write a newline-delimited high-error qname tail selected from "
        "the input production cache.",
    )
    parser.add_argument("--select-tail-summary", default=None)
    parser.add_argument("--select-tail-format", default="NVFP4")
    parser.add_argument("--select-tail-top-fraction", type=float, default=0.30)
    parser.add_argument("--select-tail-min-score", type=float, default=None)
    parser.add_argument("--select-tail-min-count", type=int, default=1)
    parser.add_argument("--select-tail-max-count", type=int, default=None)
    args = parser.parse_args(argv)

    cache = _load_pickle(args.production_cache)
    if args.select_tail_output:
        selected, summary = select_tail_from_render_scores(
            cache,
            fmt=args.select_tail_format,
            score_field=args.score_field,
            top_fraction=args.select_tail_top_fraction,
            min_score=args.select_tail_min_score,
            min_count=args.select_tail_min_count,
            max_count=args.select_tail_max_count,
        )
        tail_path = Path(args.select_tail_output)
        tail_path.parent.mkdir(parents=True, exist_ok=True)
        tail_path.write_text("".join(f"{qname}\n" for qname in selected))
        print(
            f"[production-render-cost] selected {len(selected)} "
            f"{args.select_tail_format} high-error qnames -> {tail_path}",
            flush=True,
        )
        if args.select_tail_summary:
            summary_path = Path(args.select_tail_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        if not args.output:
            return 0

    if not args.baseline_cost:
        raise SystemExit("--baseline-cost is required when writing --output")
    if not args.output:
        raise SystemExit("--output is required unless only selecting a tail")
    baseline = _load_pickle(args.baseline_cost)
    formats = (
        [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
        if args.formats else None
    )
    payload = synthesize_production_render_cost_payload(
        cache,
        baseline,
        formats=formats,
        score_field=args.score_field,
        source_label=str(args.production_cache),
        require_render_scores=bool(args.require_render_scores),
        require_output_metric=bool(args.require_output_metric),
        missing_render_score_policy=str(args.missing_render_score_policy),
        promotion_qnames=load_qnames_file(args.promotion_qnames_file),
        bf16_policy=str(args.bf16_policy),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta = payload["meta"]
    print(
        f"[production-render-cost] wrote {output_path} "
        f"(render_entries={meta['render_score_entries']} "
        f"fallback_entries={meta['fallback_entries']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
