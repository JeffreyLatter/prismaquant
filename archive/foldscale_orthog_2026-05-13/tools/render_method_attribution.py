#!/usr/bin/env python3
"""Posthoc local-error attribution for production render mechanisms.

This reads an existing PrismaQuant run directory and replays only the local
render gates on saved activation snapshots.  It does not re-run probe, cost,
allocator, export, or KL validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import iter_quantizable_tensors, stage_multimodal
from prismaquant.calibration_data import _dtype_from_name
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.layer_config import load_assignment
from prismaquant.measure_quant_cost import ActivationIndex, HDetailIndex
from prismaquant.production_weight_cache import render_production_weight


def _load_cache(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _load_model(model_path: str, *, dtype: torch.dtype, device: torch.device) -> nn.Module:
    from transformers import AutoModelForCausalLM

    staged, cleanup = stage_multimodal(model_path)
    try:
        local_only = Path(staged).exists()
        model = AutoModelForCausalLM.from_pretrained(
            staged,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )
        model.eval().to(device)
        model.name_or_path = str(staged)
        return model
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def _selected_clip_candidates(cache) -> dict[str, tuple[float | None, str]]:
    meta = getattr(cache, "metadata", {}) or {}
    solver = meta.get("activation_clip_solver", {})
    selected = solver.get("selected_candidate_by_qname", {}) if isinstance(solver, Mapping) else {}
    out: dict[str, tuple[float | None, str]] = {}
    if not isinstance(selected, Mapping):
        return out
    for qname, entry in selected.items():
        if not isinstance(entry, Mapping):
            out[str(qname)] = (None, "none")
            continue
        threshold = entry.get("threshold")
        try:
            threshold_f = float(threshold) if threshold is not None else None
        except Exception:
            threshold_f = None
        out[str(qname)] = (threshold_f, str(entry.get("rescale", "none")))
    return out


def _clip_solver_summary(cache) -> dict[str, object]:
    meta = getattr(cache, "metadata", {}) or {}
    solver = meta.get("activation_clip_solver", {})
    if not isinstance(solver, Mapping):
        return {"enabled": False}
    groups = solver.get("groups", {})
    if not isinstance(groups, Mapping):
        return {"enabled": bool(solver.get("enabled", False)), "groups": 0}
    base = 0.0
    best = 0.0
    selected = 0
    rejected = 0
    reasons: dict[str, int] = defaultdict(int)
    for group in groups.values():
        if not isinstance(group, Mapping):
            continue
        b = float(group.get("baseline_score", 0.0) or 0.0)
        s = float(group.get("best_score", b) or b)
        base += b
        best += s
        if group.get("selected") == "solved":
            selected += 1
        else:
            rejected += 1
        reasons[str(group.get("rejection_reason") or group.get("gate_reason") or "accepted")] += 1
    gain = base - best
    return {
        "enabled": bool(solver.get("enabled", False)),
        "method": solver.get("method", "PrismaClip"),
        "metric": solver.get("objective", "unknown"),
        "groups": len(groups),
        "accepted": selected,
        "rejected": rejected,
        "baseline_score": base,
        "after_score": best,
        "score_reduction": gain,
        "relative_reduction": gain / base if base > 0.0 else 0.0,
        "reasons": dict(sorted(reasons.items())),
    }


def _assignment_nvfp4_qnames(assignment: Mapping[str, str], cache) -> list[str]:
    cache_keys = getattr(cache, "weights", {})
    cached_nvfp4 = {
        str(qname)
        for qname, fmt in cache_keys
        if str(fmt).upper() == "NVFP4"
    }
    qnames = [
        qname
        for qname, fmt in assignment.items()
        if str(fmt).upper() == "NVFP4" and qname in cached_nvfp4
    ]
    return sorted(qnames)


def _row_weights_for(
    h_detail: HDetailIndex | None,
    qname: str,
    row_indices: torch.Tensor | None,
    n_rows: int,
) -> torch.Tensor | None:
    if h_detail is None or qname not in h_detail or row_indices is None:
        return None
    blob = h_detail.load_blob(qname)
    weights = blob.get("g2_per_token") if isinstance(blob, Mapping) else None
    if not isinstance(weights, torch.Tensor) or weights.numel() == 0:
        return None
    idx = row_indices.detach().reshape(-1).to(dtype=torch.long)
    if idx.numel() < n_rows:
        return None
    idx = idx[:n_rows]
    if int(idx.min().item()) < 0 or int(idx.max().item()) >= int(weights.numel()):
        return None
    return weights.detach().reshape(-1).index_select(0, idx).to(torch.float32)


def _summarize_gate_records(records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    for record in records:
        for step in record.get("trace", []):  # type: ignore[union-attr]
            if not isinstance(step, Mapping):
                continue
            mech = str(step.get("mechanism", "unknown"))
            if mech == "baseline":
                continue
            bucket = buckets.setdefault(mech, {
                "attempts": 0,
                "accepted": 0,
                "rejected": 0,
                "score_before": 0.0,
                "score_after": 0.0,
                "actual_reduction": 0.0,
                "candidate_delta": 0.0,
                "package_accepted": 0,
                "reasons": defaultdict(int),
            })
            bucket["attempts"] = int(bucket["attempts"]) + 1
            accepted = bool(step.get("accepted", False))
            if accepted:
                bucket["accepted"] = int(bucket["accepted"]) + 1
            else:
                bucket["rejected"] = int(bucket["rejected"]) + 1
            before = float(step.get("baseline_score", 0.0) or 0.0)
            cand = float(step.get("candidate_score", before) or before)
            after = cand if accepted else before
            bucket["score_before"] = float(bucket["score_before"]) + before
            bucket["score_after"] = float(bucket["score_after"]) + after
            bucket["actual_reduction"] = float(bucket["actual_reduction"]) + (before - after)
            bucket["candidate_delta"] = float(bucket["candidate_delta"]) + (before - cand)
            reason = str(step.get("reason", "unknown"))
            bucket["reasons"][reason] += 1  # type: ignore[index]
            package = step.get("package")
            if (
                accepted
                and isinstance(package, Sequence)
                and not isinstance(package, str)
                and mech in {str(item) for item in package}
            ):
                bucket["package_accepted"] = int(bucket["package_accepted"]) + 1

            if isinstance(package, Sequence) and not isinstance(package, str):
                for item in package:
                    member = str(item)
                    if member == mech:
                        continue
                    member_bucket = buckets.setdefault(member, {
                        "attempts": 0,
                        "accepted": 0,
                        "rejected": 0,
                        "score_before": 0.0,
                        "score_after": 0.0,
                        "actual_reduction": 0.0,
                        "candidate_delta": 0.0,
                        "package_accepted": 0,
                        "reasons": defaultdict(int),
                    })
                    if accepted:
                        member_bucket["package_accepted"] = (
                            int(member_bucket["package_accepted"]) + 1
                        )
    out: dict[str, dict[str, object]] = {}
    for mech, bucket in buckets.items():
        before = float(bucket["score_before"])
        out[mech] = {
            **{
                k: v
                for k, v in bucket.items()
                if k not in {"reasons"}
            },
            "relative_reduction": (
                float(bucket["actual_reduction"]) / before if before > 0.0 else 0.0
            ),
            "candidate_relative_delta": (
                float(bucket["candidate_delta"]) / before if before > 0.0 else 0.0
            ),
            "reasons": dict(sorted(bucket["reasons"].items())),  # type: ignore[union-attr]
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--layer-config", default=None)
    parser.add_argument("--production-cache", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--nvfp4-scale-rule", default="four_over_six_mse")
    args = parser.parse_args(argv)

    device = require_cuda_hot_path("render_method_attribution")
    run_dir = Path(args.run_dir)
    layer_config = Path(args.layer_config) if args.layer_config else run_dir / "artifacts" / "layer_config.json"
    cache_path = (
        Path(args.production_cache)
        if args.production_cache else
        run_dir / "artifacts" / "production_weight_cache_fisher_clip_fisherclip_recached.pkl"
    )
    act_dir = run_dir / "act"
    h_detail_dir = run_dir / "h_detail"
    output_json = Path(args.output_json) if args.output_json else run_dir / "artifacts" / "render_method_attribution.json"
    output_csv = Path(args.output_csv) if args.output_csv else run_dir / "artifacts" / "render_method_attribution.csv"

    cache = _load_cache(cache_path)
    assignment = load_assignment(layer_config)
    qnames = _assignment_nvfp4_qnames(assignment, cache)
    if args.limit and args.limit > 0:
        qnames = qnames[:args.limit]
    if not qnames:
        raise SystemExit("no NVFP4 qnames found in assignment/cache intersection")

    dtype = _dtype_from_name(args.dtype)
    model = _load_model(args.model, dtype=dtype, device=device)
    qname_to_module = {
        (name[:-len(".weight")] if name.endswith(".weight") else name): module
        for name, module, _attr in iter_quantizable_tensors(model)
        if isinstance(module, nn.Linear)
    }
    missing_model = [q for q in qnames if q not in qname_to_module]
    if missing_model:
        raise SystemExit(f"{len(missing_model)} qnames missing from model, sample={missing_model[:3]}")

    act_index = ActivationIndex(act_dir, qnames)
    missing_act = [q for q in qnames if q not in act_index]
    if missing_act:
        raise SystemExit(f"{len(missing_act)} qnames missing activations, sample={missing_act[:3]}")
    h_detail = HDetailIndex(h_detail_dir, qnames) if h_detail_dir.is_dir() else None

    from prismaquant.export_native_compressed import _compute_nvfp4_joint_global

    joint_globals = _compute_nvfp4_joint_global(model, {q: "NVFP4" for q in qnames})
    clip_candidates = _selected_clip_candidates(cache)
    levers = {
        "gptq": True,
        "gptq_damp_sweep": True,
        "scale_sweep": True,
        "fisher_gptq": h_detail is not None,
        "fisher_clip": h_detail is not None,
        "act_clip_solver": True,
        "nvfp4_scale_rule": args.nvfp4_scale_rule,
    }

    records: list[dict[str, object]] = []
    clip_records: list[dict[str, object]] = []
    torch.set_grad_enabled(False)
    for idx, qname in enumerate(qnames, start=1):
        module = qname_to_module[qname]
        act_cpu, row_indices = act_index.load_with_row_indices(qname)
        activations = {
            qname: act_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        }
        row_weights = _row_weights_for(h_detail, qname, row_indices, int(act_cpu.shape[0]))
        if row_weights is not None:
            row_weights = row_weights.to(device=device, dtype=torch.float32, non_blocking=True)

        trace: list[dict[str, object]] = []
        render_production_weight(
            module.weight.data,
            "NVFP4",
            qname=qname,
            activations=activations,
            levers=levers,
            joint_global_real=joint_globals.get(qname),
            fisher_row_weights=row_weights,
            gate_trace=trace,
        )
        records.append({
            "qname": qname,
            "format": "NVFP4",
            "trace": trace,
        })

        threshold, rescale = clip_candidates.get(qname, (None, "none"))
        if threshold is not None:
            clip_trace: list[dict[str, object]] = []
            render_production_weight(
                module.weight.data,
                "NVFP4",
                qname=qname,
                activations=activations,
                levers=levers,
                joint_global_real=joint_globals.get(qname),
                act_clip_threshold=threshold,
                act_clip_rescale=rescale,
                fisher_row_weights=row_weights,
                gate_trace=clip_trace,
            )
            clip_records.append({
                "qname": qname,
                "format": "NVFP4",
                "clip_threshold": threshold,
                "clip_rescale": rescale,
                "trace": clip_trace,
            })

        if idx % 25 == 0 or idx == len(qnames):
            print(f"[attrib] rendered {idx}/{len(qnames)}", flush=True)

    gate_summary = _summarize_gate_records(records)
    clip_gate_summary = _summarize_gate_records(clip_records)
    clip_summary = _clip_solver_summary(cache)
    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "layer_config": str(layer_config),
        "production_cache": str(cache_path),
        "qnames": len(qnames),
        "metric_note": (
            "Local render gates use Fisher-weighted output MSE when h-detail "
            "row weights are available; PrismaFisherClip aggregate comes from "
            "production-cache solver metadata."
        ),
        "no_clip_render_gates": gate_summary,
        "selected_clip_render_gates": clip_gate_summary,
        "clip_solver": clip_summary,
        "activation_recache": (getattr(cache, "metadata", {}) or {}).get(
            "activation_recache", {}
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2))

    rows = []
    for name, bucket in sorted(gate_summary.items()):
        rows.append({
            "section": "no_clip_render_gate",
            "mechanism": name,
            "attempts": bucket.get("attempts", 0),
            "accepted": bucket.get("accepted", 0),
            "rejected": bucket.get("rejected", 0),
            "package_accepted": bucket.get("package_accepted", 0),
            "score_before": bucket.get("score_before", 0.0),
            "score_after": bucket.get("score_after", 0.0),
            "score_reduction": bucket.get("actual_reduction", 0.0),
            "relative_reduction": bucket.get("relative_reduction", 0.0),
            "candidate_relative_delta": bucket.get("candidate_relative_delta", 0.0),
        })
    if clip_summary.get("enabled"):
        rows.append({
            "section": "clip_solver",
            "mechanism": clip_summary.get("method", "PrismaClip"),
            "attempts": clip_summary.get("groups", 0),
            "accepted": clip_summary.get("accepted", 0),
            "rejected": clip_summary.get("rejected", 0),
            "package_accepted": 0,
            "score_before": clip_summary.get("baseline_score", 0.0),
            "score_after": clip_summary.get("after_score", 0.0),
            "score_reduction": clip_summary.get("score_reduction", 0.0),
            "relative_reduction": clip_summary.get("relative_reduction", 0.0),
            "candidate_relative_delta": clip_summary.get("relative_reduction", 0.0),
        })

    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "section",
            "mechanism",
            "attempts",
            "accepted",
            "rejected",
            "package_accepted",
            "score_before",
            "score_after",
            "score_reduction",
            "relative_reduction",
            "candidate_relative_delta",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[attrib] wrote {output_json}", flush=True)
    print(f"[attrib] wrote {output_csv}", flush=True)
    print(json.dumps({
        "qnames": len(qnames),
        "no_clip_render_gates": gate_summary,
        "clip_solver": clip_summary,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
