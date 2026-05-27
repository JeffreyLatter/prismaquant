"""Candidate construction and coupled-candidate aggregation."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr
from .allocator_solver import Candidate, _shape_from_stats, predicted_dloss
from .serving_profiles import (
    check_serving_format,
    check_serving_shape,
)


PASSTHROUGH_SOURCE_REQUIREMENTS: dict[str, str] = {
    "FP8_SOURCE": "fp8",
    "BF16": "bf16",
}

def _is_passthrough_format(format_name: str) -> bool:
    return format_name in PASSTHROUGH_SOURCE_REQUIREMENTS


def _passthrough_source_ok(
    format_name: str,
    source_kind: str | None,
) -> bool:
    required = PASSTHROUGH_SOURCE_REQUIREMENTS.get(format_name)
    if required is None:
        return True
    if source_kind is None:
        return format_name == "BF16"
    return source_kind == required


@dataclass(frozen=True)
class FormatApplicability:
    legal: bool
    reason: str | None = None
    detail: str = ""


def _profile_allows_format(
    target_profile: str | None,
    name: str | None,
    fmt: str,
) -> FormatApplicability:
    decision = check_serving_format(target_profile, name, fmt)
    return FormatApplicability(
        decision.legal,
        decision.reason,
        decision.detail,
    )


def _format_kernel_supports_shape(fmt_name: str, in_features: int,
                                  out_features: int) -> bool:
    """Return True if the runtime kernel can handle this Linear shape."""
    return check_serving_shape(
        "research",
        fmt_name,
        in_features=in_features,
        out_features=out_features,
    ).legal


def check_format_applicability(
    linear_shape: tuple[int, ...],
    format_spec_or_name: fr.FormatSpec | str,
    *,
    qname: str | None = None,
    source_kind: str | None = None,
    target_profile: str | None = None,
) -> FormatApplicability:
    """Return whether a Linear shape can legally use a format.

    The verdict captures all cheap preflight constraints that otherwise show
    up later as allocator-invalid choices or RTN/kernel crashes: source
    passthrough integrity, serving profile restrictions, group divisibility,
    and known runtime kernel shape rules.
    """
    try:
        spec = (
            format_spec_or_name
            if isinstance(format_spec_or_name, fr.FormatSpec)
            else fr.get_format(str(format_spec_or_name))
        )
    except KeyError as exc:
        return FormatApplicability(False, "unknown_format", str(exc))
    fmt = fr.canonical_format_name(spec.name)
    shape = tuple(int(dim) for dim in linear_shape)
    if len(shape) < 2:
        return FormatApplicability(
            False,
            "shape_rank",
            f"expected a Linear weight shape with rank >= 2, got {shape}",
        )
    out_features = int(shape[-2])
    in_features = int(shape[-1])

    if (
        _is_passthrough_format(fmt)
        and not _passthrough_source_ok(fmt, source_kind)
    ):
        required = PASSTHROUGH_SOURCE_REQUIREMENTS.get(fmt)
        return FormatApplicability(
            False,
            "source_dtype_mismatch",
            f"{fmt} requires source_kind={required!r}, got {source_kind!r}",
        )

    profile_verdict = _profile_allows_format(target_profile, qname, fmt)
    if not profile_verdict.legal:
        return profile_verdict

    if (
        spec.group_size > 0
        and int(spec.group_size) < in_features
        and in_features % int(spec.group_size) != 0
    ):
        return FormatApplicability(
            False,
            "group_divisibility",
            f"group_size={spec.group_size} does not divide in_features="
            f"{in_features}",
        )
    if spec.scale_block_shape is not None:
        block_rows, block_cols = spec.scale_block_shape
        if out_features % int(block_rows) != 0 or in_features % int(block_cols) != 0:
            return FormatApplicability(
                False,
                "scale_block_divisibility",
                f"scale_block_shape={spec.scale_block_shape} does not divide "
                f"(out_features={out_features}, in_features={in_features})",
            )

    shape_decision = check_serving_shape(
        target_profile,
        fmt,
        qname=qname,
        in_features=in_features,
        out_features=out_features,
    )
    if not shape_decision.legal:
        return FormatApplicability(
            False,
            shape_decision.reason or "kernel_shape",
            shape_decision.detail,
        )
    return FormatApplicability(True)


def check_stats_format_applicability(
    stats_entry: dict,
    format_spec_or_name: fr.FormatSpec | str,
    *,
    qname: str | None = None,
    source_kind: str | None = None,
    target_profile: str | None = None,
) -> FormatApplicability:
    """Stats-entry wrapper for ``check_format_applicability``.

    This is the path allocator-like code should use when it only has the
    probe stats table.  Rank-1 legacy stats do not carry enough shape
    information for kernel preflight, so they remain admissible and the
    exporter keeps the final safety check.
    """
    shape = _shape_from_stats(dict(stats_entry))
    if len(shape) < 2:
        return FormatApplicability(True)
    return check_format_applicability(
        shape,
        format_spec_or_name,
        qname=qname,
        source_kind=source_kind,
        target_profile=target_profile,
    )


def _flashinfer_kernel_accepts(fmt_name: str, in_features: int,
                               out_features: int) -> bool | None:
    """Compatibility wrapper for the config-backed FlashInfer validator."""
    from .runtime_shape_validators import flashinfer_mxfp8_problem_size_accepts

    return flashinfer_mxfp8_problem_size_accepts(
        fmt_name,
        in_features=in_features,
        out_features=out_features,
    )


def _stats_indicates_packed_expert(stats_entry: dict) -> bool:
    """True for probe entries representing a 3D packed-expert tensor."""
    return bool(
        stats_entry.get("_packed_experts_module")
        or stats_entry.get("_packed_param")
        or int(stats_entry.get("num_experts", 0) or 0) > 0
    )


def _has_measured_output_mse(stats_entry: dict, cost_entry: dict) -> bool:
    """Whether ``output_mse`` is a real joint-output measurement.

    Packed experts historically stored ``output_mse=0.0`` as a placeholder
    because the routed expert forward was not reconstructed offline. That
    placeholder must not outrank the scalar predicted_dloss / weight_mse path.
    """
    if "output_mse" not in cost_entry:
        return False
    if cost_entry.get("output_mse_measured") is False:
        return False
    if (_stats_indicates_packed_expert(stats_entry)
            and float(cost_entry.get("output_mse", 0.0)) == 0.0
            and ("predicted_dloss" in cost_entry or "weight_mse" in cost_entry)):
        return False
    return True


def cost_entry_uses_measured_output_mse(
    stats_entry: dict,
    cost_entry: dict,
) -> bool:
    """Whether ``cost_entry_predicted_dloss`` will read ``output_mse``."""
    return _has_measured_output_mse(stats_entry, cost_entry)


def cost_entry_source(stats_entry: dict, cost_entry: dict) -> str:
    """Return the named cost source the allocator will use for one row."""
    explicit = cost_entry.get("cost_source")
    if isinstance(explicit, str) and explicit:
        return explicit
    if _has_measured_output_mse(stats_entry, cost_entry):
        if (
            _fisher_output_mse_allocator_enabled()
            and "fisher_output_mse" in cost_entry
        ):
            return "fisher_output_mse"
        return "output_mse"
    if "predicted_dloss" in cost_entry:
        return "predicted_dloss"
    return "weight_mse"


def _fisher_output_mse_allocator_enabled() -> bool:
    value = os.environ.get("PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def cost_entry_predicted_dloss(
    stats_entry: dict,
    cost_entry: dict,
    *,
    gain: float = 1.0,
) -> float:
    """Return the allocator's authoritative Δloss for one cost entry."""
    if _has_measured_output_mse(stats_entry, cost_entry):
        if (
            _fisher_output_mse_allocator_enabled()
            and "fisher_output_mse" in cost_entry
        ):
            return predicted_dloss(
                stats_entry["h_trace"],
                float(cost_entry["fisher_output_mse"]),
                gain=gain,
            )
        return predicted_dloss(
            stats_entry["h_trace"],
            float(cost_entry["output_mse"]),
            gain=gain,
        )
    if "predicted_dloss" in cost_entry:
        return float(cost_entry["predicted_dloss"]) * float(gain)
    return predicted_dloss(
        stats_entry["h_trace"],
        float(cost_entry.get("weight_mse", 0.0)),
        gain=gain,
    )


def build_candidates(stats: dict, costs: dict, formats: list[fr.FormatSpec],
                     calibrated_gains: dict[str, float] | None = None,
                     source_manifest: dict[str, str] | None = None,
                     target_profile: str | None = None,
                     mask_records: list[dict] | None = None,
                     ) -> dict[str, list[Candidate]]:
    """Build runtime-legal format candidates for every measured Linear.

    This is the optimizer's first legality gate. Export keeps a final
    defensive check for stale or hand-written recipes, but the DP must never
    see choices that the selected serving profile cannot run.
    """
    gains = calibrated_gains or {}
    out: dict[str, list[Candidate]] = {}
    masked: dict[tuple[str, str], list[str]] = {}
    source_counts: Counter[str] = Counter()
    for name, s in stats.items():
        if name not in costs:
            continue
        shape = _shape_from_stats(s)
        in_features = int(s.get("in_features", 0) or 0)
        out_features = int(s.get("out_features", 0) or 0)
        source_kind = (source_manifest or {}).get(name)
        cands = []
        for spec in formats:
            entry = None
            entry_fmt = spec.name
            for candidate_name in fr.aliases_for(spec.name):
                if candidate_name in costs[name]:
                    entry = costs[name][candidate_name]
                    entry_fmt = candidate_name
                    break
            if entry is None or "error" in entry:
                continue
            verdict = check_stats_format_applicability(
                s,
                spec,
                qname=name,
                source_kind=source_kind,
                target_profile=target_profile,
            )
            if not verdict.legal:
                if mask_records is not None:
                    mask_records.append({
                        "qname": name,
                        "format": spec.name,
                        "reason": verdict.reason or "not_applicable",
                        "detail": verdict.detail,
                        "shape": [out_features, in_features],
                        "out_features": out_features,
                        "in_features": in_features,
                        "source_kind": source_kind,
                    })
                masked.setdefault(
                    (spec.name, verdict.reason or "not_applicable"),
                    [],
                ).append(name)
                continue
            gain = float(gains.get(spec.name, gains.get(entry_fmt, 1.0)))
            # Always use measured joint output perturbation when available.
            # Packed experts can carry an unmeasured output_mse placeholder;
            # cost_entry_predicted_dloss falls back to predicted_dloss or
            # weight_mse for those entries.
            predicted = cost_entry_predicted_dloss(s, entry, gain=gain)
            source_counts[cost_entry_source(s, entry)] += 1
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=spec.effective_bits_for_shape(shape),
                memory_bytes=spec.memory_bytes_for_shape(shape),
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            out[name] = cands
    if masked:
        for (fmt, reason), names in sorted(masked.items()):
            print(
                f"[alloc] format-applicability: {len(names)} Linear(s) "
                f"dropped {fmt} reason={reason} (sample: {names[:3]})",
                flush=True,
            )
    if source_counts:
        summary = ", ".join(
            f"{source}={count}" for source, count in sorted(source_counts.items())
        )
        print(f"[alloc] cost-source usage: {summary}", flush=True)
    return out


def summarize_applicability_masks(records: list[dict]) -> dict:
    """Summarize format candidates removed before the optimizer sees them.

    The allocator's legality gate is part of the optimization layer: illegal
    candidates are excluded before DP, rather than caught later by export.
    This summary is intentionally small enough to save beside Pareto curves
    while still preserving exact qnames and kernel shapes for debugging.
    """
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_shape: dict[tuple[str, str], dict[tuple[int, int], dict]] = defaultdict(dict)
    for rec in records:
        fmt = str(rec.get("format", ""))
        reason = str(rec.get("reason", "not_applicable"))
        summary[fmt][reason] += 1
        out_features = int(rec.get("out_features", 0) or 0)
        in_features = int(rec.get("in_features", 0) or 0)
        shape_key = (out_features, in_features)
        bucket_key = (fmt, reason)
        bucket = by_shape[bucket_key].setdefault(shape_key, {
            "shape": [out_features, in_features],
            "count": 0,
            "sample": [],
            "detail": rec.get("detail", ""),
        })
        bucket["count"] += 1
        if len(bucket["sample"]) < 8:
            bucket["sample"].append(rec.get("qname", ""))

    shape_payload: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (fmt, reason), shapes in by_shape.items():
        shape_payload[fmt][reason] = sorted(
            shapes.values(),
            key=lambda row: (-int(row["count"]), row["shape"]),
        )

    return {
        "summary": {
            fmt: dict(sorted(reason_counts.items()))
            for fmt, reason_counts in sorted(summary.items())
        },
        "by_shape": {
            fmt: {
                reason: rows
                for reason, rows in sorted(reason_map.items())
            }
            for fmt, reason_map in sorted(shape_payload.items())
        },
        "records": sorted(
            records,
            key=lambda row: (
                str(row.get("format", "")),
                str(row.get("reason", "")),
                int(row.get("out_features", 0) or 0),
                int(row.get("in_features", 0) or 0),
                str(row.get("qname", "")),
            ),
        ),
    }


_FUSED_SIBLING_MARKER = ".__siblings__."


def aggregate_fused_siblings(
    stats: dict,
    costs: dict,
    formats: list[fr.FormatSpec],
    candidates: dict[str, list[Candidate]],
    profile,
    calibrated_gains: dict[str, float] | None = None,
) -> tuple[dict, dict, dict]:
    """Aggregate fused siblings into single DP items."""
    if profile is None:
        return stats, costs, candidates

    gains = calibrated_gains or {}
    grouped: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for name in candidates:
        if ".__fused__." in name:
            ungrouped.append(name)
            continue
        try:
            key = profile.fused_sibling_group(name)
        except Exception:
            key = None
        if key is None:
            ungrouped.append(name)
            continue
        grouped.setdefault(key, []).append(name)

    for key in list(grouped.keys()):
        if len(grouped[key]) < 2:
            ungrouped.extend(grouped.pop(key))

    if not grouped:
        return stats, costs, candidates

    stats_ext = {n: stats[n] for n in ungrouped}
    costs_ext = {n: costs.get(n, {}) for n in ungrouped}
    candidates_ext = {n: candidates[n] for n in ungrouped}

    for key, members in grouped.items():
        members = sorted(members)
        safe_key = key.replace(".", "__")
        super_name = f"{members[0].rsplit('.', 1)[0]}{_FUSED_SIBLING_MARKER}{safe_key}"

        n_params = sum(stats[m]["n_params"] for m in members)
        sum_h = sum(stats[m]["h_trace"] for m in members)
        d_out = int(stats[members[0]].get("out_features", 0) or 0)
        d_in = int(stats[members[0]].get("in_features", 0) or 0)

        stats_ext[super_name] = {
            "h_trace": sum_h,
            "h_trace_raw": sum(stats[m].get("h_trace_raw", 0.0) for m in members),
            "h_w2_sum": sum(stats[m].get("h_w2_sum", 0.0) for m in members),
            "w_max_abs": max(stats[m].get("w_max_abs", 0.0) for m in members),
            "w_norm_sq": sum(stats[m].get("w_norm_sq", 0.0) for m in members),
            "n_params": n_params,
            "in_features": d_in,
            "out_features": d_out,
            "n_tokens_seen": sum(stats[m].get("n_tokens_seen", 0) for m in members),
            "_fused_siblings": members,
            "_memory_bytes_by_format": {},
        }

        super_cost = {}
        for spec in formats:
            missing = [m for m in members
                       if spec.name not in costs.get(m, {})
                       or "error" in costs.get(m, {}).get(spec.name, {})]
            if missing:
                super_cost[spec.name] = {"error": "partial"}
                continue
            sum_pred = 0.0
            for m in members:
                c = costs[m][spec.name]
                # Mirrors build_candidates, including unmeasured packed
                # output_mse fallback.
                sum_pred += cost_entry_predicted_dloss(stats[m], c)
            effective_mse = sum_pred / (0.5 * sum_h) if sum_h > 0 else 0.0
            super_cost[spec.name] = {
                "weight_mse": effective_mse,
                "predicted_dloss": sum_pred,
            }
        costs_ext[super_name] = super_cost

        member_format_sets = [
            {c.fmt for c in candidates.get(m, [])}
            for m in members
        ]
        if member_format_sets:
            member_format_intersection = set.intersection(*member_format_sets)
        else:
            member_format_intersection = set()

        cands = []
        for spec in formats:
            if spec.name not in member_format_intersection:
                continue
            entry = super_cost.get(spec.name)
            if entry is None or "error" in entry:
                continue
            total_bytes = 0
            for m in members:
                shape = _shape_from_stats(stats[m])
                total_bytes += spec.memory_bytes_for_shape(shape)
            bits_per_param = 8.0 * total_bytes / max(n_params, 1)
            stats_ext[super_name]["_memory_bytes_by_format"][spec.name] = total_bytes
            gain = float(gains.get(spec.name, 1.0))
            predicted = entry["predicted_dloss"] * gain
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=bits_per_param,
                memory_bytes=total_bytes,
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            candidates_ext[super_name] = cands

    return stats_ext, costs_ext, candidates_ext


def expand_fused_sibling_assignment(assignment: dict[str, str],
                                    stats_ext: dict) -> dict[str, str]:
    """Broadcast a fused-sibling super-item assignment back to members."""
    out = {}
    for name, fmt in assignment.items():
        if _FUSED_SIBLING_MARKER in name:
            members = stats_ext[name].get("_fused_siblings", [])
            for m in members:
                out[m] = fmt
        else:
            out[name] = fmt
    return out


def _scan_source_dtype_manifest(
    model_path: str,
    profile=None,
) -> dict[str, str]:
    """Classify source Linear weights as ``fp8`` or ``bf16`` for passthrough gating."""
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    weight_map = {}
    if idx_path.exists():
        try:
            with open(idx_path) as f:
                weight_map = json.load(f).get("weight_map", {})
        except Exception:
            weight_map = {}
    bases: dict[str, set[str]] = {}
    for key in weight_map:
        for suffix in (".weight_scale_inv", ".weight"):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                bases.setdefault(base, set()).add(suffix[1:])
                break

    def _strip_weight_suffix(name: str) -> str:
        return name[:-7] if name.endswith(".weight") else name

    def _to_recipe_name(ck_base: str) -> str:
        weight_key = f"{ck_base}.weight"
        if profile is not None:
            mapper = getattr(profile, "checkpoint_to_live_name", None)
            if callable(mapper):
                try:
                    live_param = mapper(weight_key, multimodal=False)
                except TypeError:
                    live_param = mapper(weight_key)
                except Exception:
                    live_param = None
                if live_param is None:
                    return ""
                live_qname = _strip_weight_suffix(str(live_param))
                recipe_mapper = getattr(profile, "live_to_recipe_name", None)
                if callable(recipe_mapper):
                    try:
                        return str(recipe_mapper(live_qname))
                    except Exception:
                        return live_qname
                return live_qname
        if (ck_base.startswith("model.visual.")
                or ck_base.startswith("model.audio_tower.")
                or ck_base.startswith("model.vision_tower.")
                or ck_base.startswith("model.embed_vision.")
                or ck_base.startswith("model.embed_audio.")
                or ck_base.startswith("mtp.")):
            return ""
        if ck_base.startswith("model.language_model."):
            return "model." + ck_base[len("model.language_model."):]
        return ck_base

    manifest: dict[str, str] = {}
    for base, suffixes in bases.items():
        if "weight" not in suffixes:
            continue
        source_kind = "fp8" if "weight_scale_inv" in suffixes else "bf16"
        recipe_name = _to_recipe_name(base)
        if not recipe_name:
            continue
        manifest[recipe_name] = source_kind
    fp8_pairs = None
    if profile is not None:
        pairs_fn = getattr(profile, "fp8_scale_pairs", None)
        if callable(pairs_fn):
            try:
                fp8_pairs = pairs_fn(model_path)
            except Exception:
                fp8_pairs = None
    if fp8_pairs:
        recipe_mapper = getattr(profile, "live_to_recipe_name", None)
        for live_param in fp8_pairs:
            live_qname = _strip_weight_suffix(str(live_param))
            if callable(recipe_mapper):
                try:
                    live_qname = str(recipe_mapper(live_qname))
                except Exception:
                    pass
            manifest[live_qname] = "fp8"
    return manifest
