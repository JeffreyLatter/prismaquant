"""Empirical packed-MoE expert costs for the AURA hybrid recipe.

AURA's smooth per-Linear cost is route-flip-blind on routed experts (Step A,
2026-06-29: Spearman drops 0.45->0.35 under faithful dW; predicted NVFP4/FP8
ratios 2-49x vs measured 1.1-1.5x), so expert costs are MEASURED, not
modeled: per MoE layer the serving unit = all packed expert tensors of that
module (they must share one format — vLLM FusedMoE constraint), and the unit
cost of a format is the end-to-end mean-token KL(BF16 || unit-quantized)
with everything else left at source precision. The unit KL is split across
the member tensors proportionally to n_params so the allocator's per-member
aggregation charges it exactly once.

The quantizer is plain RTN ``quantize_dequantize`` from the format registry —
the same estimator contract as the AURA non-expert cost (RTN-vs-GPTQ dW is a
wash at fp4 and RTN is *better* at fp8 on the served 27B A/B); the deliberate
GPTQ render happens later in the production cache, and real-KL frontier
selection (M4) judges the actual rendered bytes.

FP8 stays IN the expert menu (standing decision 2026-06-29): it is
Pareto-dominated on routed experts (~1.3x lower KL for 2x bits), and the
right place for that fact to act is the allocator's DP + the real-KL
frontier — not a hardcoded ban here.

This module also performs the hybrid merge that previously lived as a
one-off in /home/rob/dq-runs/aura-35b/: ``--merge-base`` unions these expert
rows into an AURA (non-expert) cost payload, and ``--backfill-base`` copies
rows for any name the merged payload still lacks (MTP / visual sidecars the
AURA pass never sees) from the baseline incremental cost.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.emu_forward_kl import _qdq_accepts_col_weights

SCHEMA = "prismaquant.expert_empirical_cost.v1"
PASSTHROUGH_FORMATS = {"BF16", "FP8_SOURCE"}
# CB families render the WHOLE stack in one qdq call (the export convention:
# fp4 derives one per-stack global; fp8 per-row scales) — measured render ==
# shipped bytes, never chunked (moe_cb_design.md §3).
_CB_FAMILIES = {"nvfp4_cb", "fp8_cb"}
# Both CB families carry a k-rung ladder (k index bits per 8-weight vector);
# the RD-law fit is holdout-gated per unit, so admitting FP8_CB costs nothing
# when the law fails there (falls back to full measurement).
_CB_K_RE = re.compile(r"^(?:NVFP4|FP8)_CB_[KS](\d+)$")
# RD-law ladder interpolation (moe_cb_design.md §3.4): D(k) = C * 2^(-k/4),
# validated +-3% on weighted-recon at 0.6B but UNPROVEN on unit-KL — so it is
# opt-in and holdout-gated PER UNIT (a failed holdout falls back to full
# measurement for that unit).
_LADDER_SLOPE_BITS = 0.25


def _log(msg: str) -> None:
    print(f"[expert-cost {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _canon_formats(formats: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for raw in formats:
        name = fr.canonical_format_name(str(raw).strip())
        if name and name not in seen:
            seen.append(name)
    return seen


@torch.no_grad()
def _baseline_logprobs(model, calib_ids: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for i in range(calib_ids.shape[0]):
        logits = model(calib_ids[i:i + 1]).logits.float()
        out.append(F.log_softmax(logits, dim=-1).cpu())
    return out


@torch.no_grad()
def _expert_sample_idx(num_experts: int, sample: int) -> torch.Tensor | None:
    """Deterministic stratified expert subsample (the local cost path's
    linspace pattern — even coverage of the expert index range)."""
    if sample <= 0 or num_experts <= sample:
        return None
    return torch.linspace(0, num_experts - 1, sample).round().long().unique()


def _quantize_unit_inplace(
    mod,
    param_names: Sequence[str],
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    unit_qname: str = "",
    sample_idx: torch.Tensor | None = None,
) -> None:
    """Render every member of one expert serving unit in-place in ``fmt``.

    CB families use the imatrix-WEIGHTED VQ render on the whole stack (the
    exporter convention — measuring an unweighted render while the exporter
    ships weighted bytes is the rendering-confound class, so a CB format
    with no col_weights entry for a member hard-fails; the encode tier is
    inherited from PRISMAQUANT_CB_ENCODE_TIER via the registry closure).

    ``sample_idx`` (expert subsampling): quantize ONLY those expert slices,
    leaving the rest BF16 — the caller extrapolates the partial unit KL.
    Each sampled expert's render is identical to its full-stack render (the
    CB encode is per-expert-row independent; per-expert col_weights slices
    keep the weighted-render contract), so sampling changes COVERAGE, never
    the bytes measured for a covered expert.
    """
    spec = fr.get_format(fmt)
    qdq = spec.quantize_dequantize
    weighted = _qdq_accepts_col_weights(spec)
    for pn in param_names:
        w = getattr(mod, pn).data
        full = f"{unit_qname}.{pn}" if unit_qname else pn
        if spec.family in _CB_FAMILIES:
            cw = (col_weights or {}).get(full)
            if weighted and cw is None:
                raise ValueError(
                    f"{full}: CB format {fmt} needs a col_weights entry — "
                    f"the deliberate CB render is imatrix-weighted; an "
                    f"unweighted unit-KL would measure bytes the exporter "
                    f"never ships (pass --col-weights)")
            cw_dev = cw.to(w.device)
            if sample_idx is not None:
                idx = sample_idx.to(w.device)
                cw_s = (cw_dev[idx] if cw_dev.ndim >= 3
                        and cw_dev.shape[0] == w.shape[0] else cw_dev)
                w[idx] = qdq(
                    w[idx].float(), col_weights=cw_s).to(w.dtype)
            else:
                w.copy_(qdq(
                    w.float(), col_weights=cw_dev).to(w.dtype))
        elif spec.family == "nv":
            # NV formats derive one per-TENSOR global scale from
            # whatever slice they are given, while export ships one
            # global PER EXPERT. Chunk-batching would share a global
            # across the chunk and make the measured KL depend on the
            # --expert-chunk knob; quantize per expert slice instead
            # (mirrors measure_quant_cost._batched_quantize, which does
            # the per-slice loop for exactly this reason).
            experts = (sample_idx.tolist() if sample_idx is not None
                       else range(w.shape[0]))
            for e in experts:
                w[e] = qdq(w[e].float()).to(w.dtype)
        else:
            # Scale-local formats are chunk-invariant, so batching is
            # safe: FP8_E4M3/FP8_E5M2 reshape to (-1, in) and scale each
            # output row independently (fp8_dynamic_weight_qdq), and
            # group/block-scaled formats (MX) never cross the expert
            # boundary within a row.
            if sample_idx is not None:
                idx = sample_idx.to(w.device)
                w[idx] = qdq(w[idx].float()).to(w.dtype)
            else:
                for e in range(0, w.shape[0], expert_chunk):
                    w[e:e + expert_chunk] = qdq(
                        w[e:e + expert_chunk].float()).to(w.dtype)


@torch.no_grad()
def _unit_kl(
    model,
    calib_ids: torch.Tensor,
    baseline: list[torch.Tensor],
    mod,
    param_names: Sequence[str],
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    unit_qname: str = "",
    sample_idx: torch.Tensor | None = None,
) -> float:
    """Mean-token KL(BF16 || model-with-this-unit-quantized).

    With ``sample_idx``, only those expert slices are quantized (and cloned
    for restore) — the caller owns the extrapolation to the full unit."""
    if sample_idx is None:
        originals = {pn: getattr(mod, pn).data.clone() for pn in param_names}
    else:
        originals = {pn: getattr(mod, pn).data[
            sample_idx.to(getattr(mod, pn).device)].clone()
            for pn in param_names}
    try:
        _quantize_unit_inplace(
            mod, param_names, fmt, expert_chunk=expert_chunk,
            col_weights=col_weights, unit_qname=unit_qname,
            sample_idx=sample_idx)
        total = 0.0
        n_tok = 0
        for i in range(calib_ids.shape[0]):
            lp = F.log_softmax(model(calib_ids[i:i + 1]).logits.float(), -1)
            bl = baseline[i].to(lp.device)
            kl = (bl.exp() * (bl - lp)).sum(-1)
            total += float(kl.sum().item())
            n_tok += kl.numel()
        return total / max(n_tok, 1)
    finally:
        for pn in param_names:
            w = getattr(mod, pn).data
            if sample_idx is None:
                w.copy_(originals[pn])
            else:
                w[sample_idx.to(w.device)] = originals[pn]


def _cb_ladder_split(measured_fmts: Sequence[str]):
    """Split a CB k-rung ladder into (anchors, holdout, predicted) for RD-law
    interpolation. Returns None when the ladder is too short to pay
    (< 4 rungs: anchors+holdout would measure everything anyway). At exactly
    4 rungs (the shipped FP8 menu) the two extremes anchor the line and one
    middle rung is the holdout, predicting the other (25% fewer encodes);
    at >= 5 rungs three anchors give a least-squares fit."""
    kmap = {f: int(m.group(1)) for f in measured_fmts
            if (m := _CB_K_RE.match(f))}
    if len(kmap) < 4:
        return None
    by_k = sorted(kmap, key=kmap.get)
    if len(by_k) == 4:
        anchors = [by_k[0], by_k[-1]]
    else:
        anchors = [by_k[0], by_k[len(by_k) // 2], by_k[-1]]
    rest = [f for f in by_k if f not in anchors]
    holdout = rest[len(rest) // 2]
    predicted = [f for f in rest if f != holdout]
    if not predicted:
        return None
    return kmap, anchors, holdout, predicted


def _cb_ladder_fit(kls: Mapping[str, float], kmap: Mapping[str, int],
                   anchors: Sequence[str], holdout: str,
                   predicted: Sequence[str], tol: float):
    """Fit log2 D = a - b*k on the anchors (least squares; with 2 anchors the
    line is exact — the slope is fitted, not assumed, so the fp8 ladder does
    not inherit the fp4-calibrated 1/4-per-k decay). Accept iff the holdout's
    relative error <= tol. Returns (predicted_kls | None, holdout_rel_err)."""
    xs = [float(kmap[f]) for f in anchors]
    ys = [math.log2(max(kls[f], 1e-12)) for f in anchors]
    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    b = (-sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
         if denom > 0 else _LADDER_SLOPE_BITS)
    a = my + b * mx

    def pred(f):
        return float(2.0 ** (a - b * kmap[f]))

    rel = abs(pred(holdout) - kls[holdout]) / max(kls[holdout], 1e-12)
    if rel > tol:
        return None, rel
    return {f: pred(f) for f in predicted}, rel


def measure_expert_unit_costs(
    model,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    ladder_interp: bool = False,
    ladder_tol: float = 0.10,
    expert_sample: int = 0,
    max_units: int = 0,
    unit_filter: str | None = None,
) -> tuple[dict, dict, dict]:
    """Measure per-serving-unit empirical KL costs for packed-MoE experts.

    Returns ``(stats, costs, unit_kls)`` where stats/costs are
    allocator-payload row dicts keyed by full member names and ``unit_kls``
    maps ``experts_qname -> {fmt: unit_kl}``.
    """
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )

    menu = _canon_formats(formats)
    measured_fmts = [f for f in menu if f not in PASSTHROUGH_FORMATS]
    units = [
        (qn, m) for qn, m in model.named_modules()
        if _is_packed_experts_module(m, profile)
    ]
    if unit_filter:
        pat = re.compile(unit_filter)
        units = [(qn, m) for qn, m in units if pat.search(qn)]
    if max_units > 0:
        units = units[:max_units]
    if progress:
        _log(f"{len(units)} expert serving units; measured formats: "
             f"{measured_fmts} (menu {menu})"
             + (f"; expert_sample={expert_sample}" if expert_sample else ""))
    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    if not units or not measured_fmts:
        return stats, costs, unit_kls

    baseline = _baseline_logprobs(model, calib_ids)
    ladder = _cb_ladder_split(measured_fmts) if ladder_interp else None
    for qn, mod in units:
        pnames = list(_packed_experts_param_names(mod, profile))
        n_params_unit = sum(int(getattr(mod, pn).numel()) for pn in pnames)
        num_experts = int(getattr(mod, pnames[0]).shape[0])
        # One stratified subsample SHARED across every format of the unit, so
        # inter-format comparability (what the allocator consumes) is exact
        # even under sampling; the extrapolation to the full unit rides on
        # cross-expert additivity (validated fp32-additive in this repo) and
        # is scaled by expert count (uniform stacks).
        sample_idx = _expert_sample_idx(num_experts, expert_sample)
        kl_scale = (float(num_experts) / float(sample_idx.numel())
                    if sample_idx is not None else 1.0)

        def kl_of(fmt):
            return kl_scale * _unit_kl(
                model, calib_ids, baseline, mod, pnames, fmt,
                expert_chunk=expert_chunk, col_weights=col_weights,
                unit_qname=qn, sample_idx=sample_idx)

        if ladder is None:
            kls = {fmt: kl_of(fmt) for fmt in measured_fmts}
        else:
            kmap, anchors, holdout, predicted = ladder
            kls = {fmt: kl_of(fmt) for fmt in measured_fmts
                   if fmt not in predicted}
            pred_kls, rel = _cb_ladder_fit(
                kls, kmap, anchors, holdout, predicted, ladder_tol)
            if pred_kls is None:
                # Holdout gate FAILED for this unit: fall back to full
                # measurement (the law is recon-validated, KL-unproven).
                if progress:
                    _log(f"  {qn}: ladder holdout rel_err {rel:.1%} > "
                         f"{ladder_tol:.0%} — measuring all rungs")
                kls.update({fmt: kl_of(fmt) for fmt in predicted})
                kls["_ladder"] = {"accepted": False,
                                  "holdout_rel_err": round(rel, 4)}
            else:
                kls.update(pred_kls)
                kls["_ladder"] = {
                    "accepted": True, "holdout_rel_err": round(rel, 4),
                    "anchors": anchors, "holdout": holdout,
                    "predicted": predicted,
                }
        ladder_meta = kls.pop("_ladder", None)
        unit_kls[qn] = dict(kls)
        if ladder_meta is not None:
            unit_kls[qn]["_ladder"] = ladder_meta
        if sample_idx is not None:
            unit_kls[qn]["_sampling"] = {
                "num_experts": num_experts,
                "sampled": int(sample_idx.numel()),
                "scale": round(kl_scale, 4),
            }
        for pn in pnames:
            tensor = getattr(mod, pn)
            npm = int(tensor.numel())
            full = f"{qn}.{pn}" if qn else pn
            shape = list(tensor.shape)
            stats[full] = {
                # h_trace is meaningless for an empirically-costed unit; the
                # allocator consumes predicted_dloss directly. 0.0 marks
                # "do not fall back to h_trace x weight_mse" for this row.
                "h_trace": 0.0,
                "n_params": npm,
                "in_features": int(shape[2]),
                "out_features": int(shape[1]),
                "num_experts": num_experts,
                "_packed_experts_module": qn,
                "_packed_param": pn,
                "n_probes": 0,
            }
            row: dict = {}
            for fmt in measured_fmts:
                # Split the UNIT cost across members by n_params so the
                # per-member sum re-assembles exactly one unit KL.
                row[fmt] = {
                    "predicted_dloss": kls[fmt] * npm / n_params_unit,
                    "cost_source": "empirical_unit_kl",
                    "output_mse_measured": False,
                }
            for fmt in menu:
                if fmt in PASSTHROUGH_FORMATS:
                    row[fmt] = {
                        "predicted_dloss": 0.0,
                        "cost_source": "passthrough_zero",
                        "output_mse_measured": False,
                    }
            costs[full] = row
        if progress:
            _log(f"  {qn}: " + "  ".join(
                f"{fmt} unit KL = {kls[fmt]:.4e}" for fmt in measured_fmts)
                + f"  (n_params={n_params_unit / 1e6:.0f}M, "
                  f"experts={num_experts})")
    return stats, costs, unit_kls


def merge_cost_payloads(
    base: Mapping[str, object],
    expert_stats: Mapping[str, object],
    expert_costs: Mapping[str, object],
    *,
    formats: Sequence[str],
    replace_experts: bool = False,
) -> dict:
    """Union base non-expert rows with empirical expert rows.

    AURA lane (``replace_experts=False``): collisions are an error —
    aura_cost must have been run with ``--allow-packed-expert-omission``
    (its guard fail-fasts otherwise), so no name may be costed by both
    estimators.

    CB lane (``replace_experts=True``): the COST_MODE=local payload DOES
    cost the expert stacks (smoothly — route-flip-blind); those rows are
    REPLACED by the empirical ones and recorded in provenance, non-expert
    rows stay untouched (moe_cb_design.md §3).
    """
    merged = dict(base)
    base_stats = dict(base.get("stats", {}) or {})
    base_costs = dict(base.get("costs", {}) or {})
    overlap = set(base_costs) & set(expert_costs)
    if overlap and not replace_experts:
        raise RuntimeError(
            f"hybrid merge collision: {len(overlap)} names costed by BOTH "
            f"the base payload and the expert empirical pass (e.g. "
            f"{sorted(overlap)[:3]}). The base run must omit packed experts "
            f"(or pass replace_experts for the CB-lane replace semantics).")
    if overlap:
        for name in overlap:
            base_costs.pop(name)
            base_stats.pop(name, None)
        prov = dict(merged.get("provenance", {}) or {})
        prov["replaced_smooth_expert_rows"] = sorted(overlap)
        merged["provenance"] = prov
    base_stats.update(expert_stats)
    base_costs.update(expert_costs)
    merged["stats"] = base_stats
    merged["costs"] = base_costs
    merged["schema"] = SCHEMA
    merged["formats"] = _canon_formats(formats)
    return merged


def backfill_missing_from_base(
    payload: dict,
    base_cost: Mapping[str, object],
) -> list[str]:
    """Copy rows for names the payload lacks from the baseline cost pkl.

    Covers MTP / visual sidecars the AURA pass never sees (the synthesized
    MTP module lives outside the CausalLM the cost harness loads). Returns
    the backfilled names, and records them in provenance for honesty: these
    rows carry the baseline estimator, not the AURA adjoint.
    """
    base_costs = dict(base_cost.get("costs", {}) or {})
    base_stats = dict(base_cost.get("stats", {}) or {})
    added: list[str] = []
    for name, row in base_costs.items():
        if name in payload["costs"]:
            continue
        payload["costs"][name] = row
        if name in base_stats and name not in payload["stats"]:
            payload["stats"][name] = base_stats[name]
        added.append(name)
    return sorted(added)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Empirical packed-MoE expert cost (+ hybrid merge)")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--formats", default="NVFP4,FP8_DYNAMIC,BF16",
        help="Expert format menu. Non-passthrough formats are measured; "
        "BF16/FP8_SOURCE rows are passthrough-zero.")
    p.add_argument("--n-calib-samples", type=int, default=16)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument(
        "--dataset", default=None,
        help="Optional calibration source (HF id, .jsonl, .txt) via "
        "sensitivity_probe.load_calibration; default is the WikiText "
        "windowed loader (matches aura_cost).")
    p.add_argument("--expert-chunk", type=int, default=16,
                   help="Experts quantized per in-place RTN chunk.")
    p.add_argument(
        "--merge-base", default=None,
        help="AURA non-expert cost pkl to union the expert rows into "
        "(the hybrid recipe). Output = merged payload.")
    p.add_argument(
        "--backfill-base", default=None,
        help="Baseline incremental cost pkl; rows for names still missing "
        "after the merge (MTP/visual sidecars) are copied from it.")
    p.add_argument(
        "--replace-experts", action="store_true",
        help="CB-lane merge semantics: the COST_MODE=local base payload "
        "costs expert stacks smoothly (route-flip-blind); REPLACE those "
        "rows with the empirical ones (recorded in provenance) instead of "
        "treating the collision as an error.")
    p.add_argument(
        "--col-weights", default=None,
        help="Pickle {qname: per-input-column importance} (the CB "
        "exporter's imatrix). REQUIRED when the menu contains CB formats: "
        "their deliberate render is imatrix-weighted, and the measured "
        "unit-KL must be of the bytes the exporter ships.")
    p.add_argument(
        "--cb-ladder-interp", action="store_true",
        help="RD-law ladder interpolation for NVFP4_CB_K rungs (measure "
        "anchors + holdout, predict the rest; holdout-gated PER UNIT). "
        "Also enabled by PRISMAQUANT_CB_LADDER_INTERP=1. Default OFF — "
        "the law is recon-validated but KL-unproven (encode_tiers.md §B).")
    p.add_argument("--ladder-holdout-tol", type=float, default=0.10,
                   help="Max holdout relative error to accept a unit's "
                   "ladder fit; above it the unit measures every rung.")
    p.add_argument(
        "--expert-sample", type=int, default=0,
        help="Quantize only a stratified subsample of N experts per unit "
        "and extrapolate the unit KL by expert count (cross-expert "
        "additivity). One shared subsample across all formats of a unit "
        "keeps inter-format comparability exact. 0 = full stack "
        "(default). Cuts the encode volume ~E/N (the 35B/300B cost-stage "
        "wall); export still encodes every expert exactly.")
    p.add_argument(
        "--max-units", type=int, default=0,
        help="Measure only the first N units (0 = all). Validation/"
        "sharding aid.")
    p.add_argument(
        "--unit-filter", default=None,
        help="Regex on the experts qname; only matching units are "
        "measured. Validation/sharding aid.")
    p.add_argument("--device", default="cuda")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("expert_empirical_cost", args.device)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.build_rtn_cache import stage_multimodal
    from prismaquant.model_profiles import detect_profile_with_warning

    staged, _cleanup = stage_multimodal(args.model)
    local_only = Path(staged).exists()
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    _log(f"loading {args.model} (staged={staged}) bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        staged, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=local_only, attn_implementation="eager",
        device_map=args.device,
    ).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    profile = detect_profile_with_warning(
        staged, entrypoint="expert-empirical-cost")

    if args.dataset:
        from prismaquant.sensitivity_probe import load_calibration
        calib = load_calibration(
            tok, args.dataset, args.n_calib_samples, args.calib_seqlen,
            calib_seed=args.calib_seed)
    else:
        from prismaquant.calibration_data import (
            load_wikitext_calibration_windowed,
        )
        calib = load_wikitext_calibration_windowed(
            tok, args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed)
    calib = calib.to(args.device)

    formats = _canon_formats(
        [f for f in args.formats.split(",") if f.strip()])
    col_weights = None
    if args.col_weights:
        with open(args.col_weights, "rb") as fh:
            col_weights = {k: torch.as_tensor(v)
                           for k, v in pickle.load(fh).items()}
    ladder_interp = bool(args.cb_ladder_interp) or (
        os.environ.get("PRISMAQUANT_CB_LADDER_INTERP", "0") == "1")
    stats, costs, unit_kls = measure_expert_unit_costs(
        model, profile, calib, formats, expert_chunk=args.expert_chunk,
        col_weights=col_weights, ladder_interp=ladder_interp,
        ladder_tol=args.ladder_holdout_tol,
        expert_sample=args.expert_sample, max_units=args.max_units,
        unit_filter=args.unit_filter)

    provenance = {
        "schema": SCHEMA,
        "git_commit": _git_commit(),
        "model": args.model,
        "dataset": args.dataset or f"wikitext:{args.calib_split}",
        "n_calib_samples": int(calib.shape[0]),
        "calib_seqlen": int(calib.shape[1]),
        "calib_seed": args.calib_seed,
        "calib_sha256": hashlib.sha256(
            calib.cpu().numpy().tobytes()).hexdigest(),
        "expert_units": len(unit_kls),
        "unit_kls": unit_kls,
        "formats_measured": [
            f for f in formats if f not in PASSTHROUGH_FORMATS],
        "col_weights": args.col_weights,
        "cb_ladder_interp": ladder_interp,
        "encode_tier": os.environ.get("PRISMAQUANT_CB_ENCODE_TIER"),
        "expert_sample": int(args.expert_sample),
        "max_units": int(args.max_units),
        "unit_filter": args.unit_filter,
    }

    if args.merge_base:
        with open(args.merge_base, "rb") as fh:
            base = pickle.load(fh)
        payload = merge_cost_payloads(
            base, stats, costs, formats=formats,
            replace_experts=bool(args.replace_experts))
        prov = dict(payload.get("provenance", {}) or {})
        prov["expert_empirical_cost"] = provenance
        prov["merge_base"] = args.merge_base
        payload["provenance"] = prov
        _log(f"merged {len(costs)} expert member rows into "
             f"{args.merge_base} ({len(payload['costs'])} total)")
    else:
        payload = {
            "schema": SCHEMA,
            "formats": formats,
            "stats": stats,
            "costs": costs,
            "provenance": provenance,
        }

    if args.backfill_base:
        with open(args.backfill_base, "rb") as fh:
            base_cost = pickle.load(fh)
        added = backfill_missing_from_base(payload, base_cost)
        prov = dict(payload.get("provenance", {}) or {})
        prov["backfilled_from_base"] = added
        prov["backfill_base"] = args.backfill_base
        payload["provenance"] = prov
        if added:
            _log(f"backfilled {len(added)} sidecar rows from "
                 f"{args.backfill_base}: {added[:5]}"
                 f"{' ...' if len(added) > 5 else ''}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"wrote {out}: {len(payload['costs'])} cost rows "
         f"({len(unit_kls)} expert units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
