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
import pickle
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from prismaquant import format_registry as fr

SCHEMA = "prismaquant.expert_empirical_cost.v1"
PASSTHROUGH_FORMATS = {"BF16", "FP8_SOURCE"}


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
def _unit_kl(
    model,
    calib_ids: torch.Tensor,
    baseline: list[torch.Tensor],
    mod,
    param_names: Sequence[str],
    fmt: str,
    *,
    expert_chunk: int = 16,
) -> float:
    """Mean-token KL(BF16 || model-with-this-unit-RTN-quantized)."""
    spec = fr.get_format(fmt)
    qdq = spec.quantize_dequantize
    originals = {pn: getattr(mod, pn).data.clone() for pn in param_names}
    try:
        for pn in param_names:
            w = getattr(mod, pn).data
            if spec.family == "nv":
                # NV formats derive one per-TENSOR global scale from
                # whatever slice they are given, while export ships one
                # global PER EXPERT. Chunk-batching would share a global
                # across the chunk and make the measured KL depend on the
                # --expert-chunk knob; quantize per expert slice instead
                # (mirrors measure_quant_cost._batched_quantize, which does
                # the per-slice loop for exactly this reason).
                for e in range(w.shape[0]):
                    w[e] = qdq(w[e].float()).to(w.dtype)
            else:
                # Scale-local formats are chunk-invariant, so batching is
                # safe: FP8_E4M3/FP8_E5M2 reshape to (-1, in) and scale each
                # output row independently (fp8_dynamic_weight_qdq), and
                # group/block-scaled formats (MX) never cross the expert
                # boundary within a row.
                for e in range(0, w.shape[0], expert_chunk):
                    w[e:e + expert_chunk] = qdq(
                        w[e:e + expert_chunk].float()).to(w.dtype)
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
            getattr(mod, pn).data.copy_(originals[pn])


def measure_expert_unit_costs(
    model,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
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
    if progress:
        _log(f"{len(units)} expert serving units; measured formats: "
             f"{measured_fmts} (menu {menu})")
    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    if not units or not measured_fmts:
        return stats, costs, unit_kls

    baseline = _baseline_logprobs(model, calib_ids)
    for qn, mod in units:
        pnames = list(_packed_experts_param_names(mod, profile))
        n_params_unit = sum(int(getattr(mod, pn).numel()) for pn in pnames)
        num_experts = int(getattr(mod, pnames[0]).shape[0])
        kls = {
            fmt: _unit_kl(
                model, calib_ids, baseline, mod, pnames, fmt,
                expert_chunk=expert_chunk)
            for fmt in measured_fmts
        }
        unit_kls[qn] = kls
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
) -> dict:
    """Union AURA non-expert rows with empirical expert rows.

    Collisions are an error: aura_cost must have been run with
    ``--allow-packed-expert-omission`` (its guard fail-fasts otherwise), so
    no name may be costed by both estimators.
    """
    merged = dict(base)
    base_stats = dict(base.get("stats", {}) or {})
    base_costs = dict(base.get("costs", {}) or {})
    overlap = set(base_costs) & set(expert_costs)
    if overlap:
        raise RuntimeError(
            f"hybrid merge collision: {len(overlap)} names costed by BOTH "
            f"the base payload and the expert empirical pass (e.g. "
            f"{sorted(overlap)[:3]}). The base run must omit packed experts.")
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


def main(argv: Sequence[str] | None = None) -> int:
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
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

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
    stats, costs, unit_kls = measure_expert_unit_costs(
        model, profile, calib, formats, expert_chunk=args.expert_chunk)

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
    }

    if args.merge_base:
        with open(args.merge_base, "rb") as fh:
            base = pickle.load(fh)
        payload = merge_cost_payloads(
            base, stats, costs, formats=formats)
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
