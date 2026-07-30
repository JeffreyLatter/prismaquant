"""Staged (two-pass) production-render cost — ARCHIVED 2026-07-30.

This module is the record of `COST_MODE=production-render-staged` (alias
`production-render-tail`), walled by architecture re-vet proposal **R17**. It
is NOT imported by the live tree; `run-pipeline.sh` fails fast with `exit 2`
if the mode is selected. See the banner README next to this file for the
killing measurement and the durable lesson.

The lane worked in two passes:

  1. render **NVFP4 only** for every quantizable Linear,
  2. rank those Linears by their NVFP4 render score and take the top
     `--select-tail-top-fraction` as a "promotion tail",
  3. render the *higher* formats for the tail alone, and
  4. synthesize the allocator cost with `bf16_policy="promotion-set"` +
     `missing_render_score_policy="unavailable"` so the un-rendered
     (non-tail) Linears carried an explicit `error` entry instead of a
     baseline-proxy fallback.

`select_tail_from_render_scores` below is step 2 verbatim as it stood at
`8e071bf`. Steps 3-4's staged-only parameters
(`missing_render_score_policy`, `promotion_qnames`, `bf16_policy`) were
removed from `prismaquant.production_render_cost` in the same commit; the
verbatim bodies of those branches are reproduced at the bottom of this file
so the lane can be reconstructed without archaeology.

To resurrect: restore the parameters on
`synthesize_production_render_cost_payload`, re-add the ten CLI arguments
listed in the README, restore the `[2b]`-`[2e]` staged block in
`run-pipeline.sh`, and delete the `exit 2` gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence  # noqa: F401  (record fidelity)

from prismaquant import format_registry as fr
from prismaquant.production_render_cost import (
    _cache_render_score_records,
    _score_value,
)


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


# ---------------------------------------------------------------------------
# Removed staged branches of prismaquant.production_render_cost, verbatim.
# ---------------------------------------------------------------------------
#
# synthesize_production_render_cost_payload signature additions:
#
#     missing_render_score_policy: str = "fallback",
#     promotion_qnames: set[str] | None = None,
#     bf16_policy: str = "all",
#
# BF16 arm of the per-format loop (before the unconditional bf16_zero entry):
#
#             if fmt_c == "BF16":
#                 if (
#                     bf16_policy == "promotion-set"
#                     and promotion_qnames is not None
#                     and cname not in promotion_qnames
#                 ):
#                     synthesized[fmt_c] = {
#                         "error": "bf16_not_in_staged_promotion_set",
#                         "cost_source": "unavailable_staged_bf16",
#                     }
#                     continue
#
# Missing-render-score arm (before the alias fallback search):
#
#             if missing_render_score_policy == "unavailable":
#                 synthesized[fmt_c] = {
#                     "error": "missing production render score",
#                     "cost_source": "unavailable_missing_render_score",
#                 }
#                 fallback_entries += 1
#                 continue
#
# Strictness check:
#
#     if (require_render_scores or missing_render_score_policy == "error") \
#             and missing:
#
# meta keys:
#
#             "missing_render_score_policy": missing_render_score_policy,
#             "bf16_policy": bf16_policy,
#             "promotion_qnames": (
#                 int(len(promotion_qnames))
#                 if promotion_qnames is not None else None
#             ),
#
# main() tail dispatch (before the --baseline-cost requirement check):
#
#     if args.select_tail_output:
#         selected, summary = select_tail_from_render_scores(
#             cache,
#             fmt=args.select_tail_format,
#             score_field=args.score_field,
#             top_fraction=args.select_tail_top_fraction,
#             min_score=args.select_tail_min_score,
#             min_count=args.select_tail_min_count,
#             max_count=args.select_tail_max_count,
#         )
#         tail_path = Path(args.select_tail_output)
#         tail_path.parent.mkdir(parents=True, exist_ok=True)
#         tail_path.write_text("".join(f"{qname}\n" for qname in selected))
#         print(
#             f"[production-render-cost] selected {len(selected)} "
#             f"{args.select_tail_format} high-error qnames -> {tail_path}",
#             flush=True,
#         )
#         if args.select_tail_summary:
#             summary_path = Path(args.select_tail_summary)
#             summary_path.parent.mkdir(parents=True, exist_ok=True)
#             summary_path.write_text(
#                 json.dumps(summary, indent=2, sort_keys=True))
#         if not args.output:
#             return 0
