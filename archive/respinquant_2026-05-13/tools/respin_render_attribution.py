#!/usr/bin/env python3
"""Posthoc residual-basis local-error attribution on saved activations.

This tool answers two questions for a residual-basis artifact:

1. How much does the basis arm change per-Linear NVFP4 render error?
2. After that basis change, how much do local methods such as FourOverSix,
   Fisher-GPTQ/GPTQ, and scale_sweep still add?

It deliberately does not replay PrismaClip thresholds from a non-ReSpin run:
those thresholds are activation-basis dependent and must be re-solved under the
ReSpin basis before being treated as a valid subsequent method.

The current random-Givens artifacts are runtime-substrate smokes. They are not
paper-faithful ReSpinQuant unless the rotations were trained separately and the
artifact metadata says so.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.layer_config import load_assignment
from prismaquant.measure_quant_cost import ActivationIndex, HDetailIndex
from prismaquant.production_weight_cache import render_production_weight
from prismaquant.render_score import score_render_error
from tools.create_respin_equivalent_variant import (
    build_alternating_transitions,
    build_transitions_from_basis_checkpoint,
)
from tools.render_method_attribution import _row_weights_for, _summarize_gate_records


BODY_QNAME_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
ATTN_INPUT_WEIGHTS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
}
ATTN_OUTPUT_WEIGHTS = {
    "self_attn.o_proj",
    "linear_attn.out_proj",
}
MLP_INPUT_WEIGHTS = {
    "mlp.gate_proj",
    "mlp.up_proj",
}
MLP_OUTPUT_WEIGHTS = {
    "mlp.down_proj",
}


def _load_cache(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _canonical_tensor_key(qname: str) -> str:
    if qname.startswith("model.layers."):
        return "model.language_model." + qname[len("model."):] + ".weight"
    return qname + ".weight"


def _norm_key(layer: int, which: str) -> str:
    return f"model.language_model.layers.{layer}.{which}.weight"


def _weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        out: dict[str, str] = {}
        for shard in sorted(model_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    out[key] = shard.name
        return out
    payload = json.loads(index.read_text())
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, Mapping):
        return {}
    return {str(key): str(value) for key, value in weight_map.items()}


def _load_tensor_subset(model_dir: Path,
                        keys: Sequence[str]) -> dict[str, torch.Tensor]:
    wanted = set(keys)
    weight_map = _weight_map(model_dir)
    shards: dict[str, list[str]] = defaultdict(list)
    for key in wanted:
        filename = weight_map.get(key)
        if filename is not None:
            shards[filename].append(key)
    if not shards:
        for shard in sorted(model_dir.glob("*.safetensors")):
            if shard.name.startswith("prisma-residual-adapters"):
                continue
            with safe_open(shard, framework="pt", device="cpu") as handle:
                present = wanted.intersection(handle.keys())
                if present:
                    shards[shard.name].extend(sorted(present))
    tensors: dict[str, torch.Tensor] = {}
    for filename, shard_keys in shards.items():
        path = model_dir / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                if key in handle.keys():
                    tensors[key] = handle.get_tensor(key)
    missing = sorted(wanted.difference(tensors))
    if missing:
        raise FileNotFoundError(f"{len(missing)} tensors missing from {model_dir}: {missing[:5]}")
    return tensors


def _is_body_qname(qname: str) -> bool:
    return BODY_QNAME_RE.match(qname) is not None


def _layer_and_suffix(qname: str) -> tuple[int, str]:
    match = BODY_QNAME_RE.match(qname)
    if match is None:
        raise ValueError(f"not a body qname: {qname}")
    return int(match.group(1)), match.group(2)


def _activation_for_arm(qname: str,
                        x: torch.Tensor,
                        *,
                        arm: str,
                        bases: Sequence[torch.Tensor],
                        input_gammas: Mapping[int, torch.Tensor],
                        post_gammas: Mapping[int, torch.Tensor],
                        device: torch.device) -> torch.Tensor:
    layer, suffix = _layer_and_suffix(qname)
    x_gpu = x.to(device=device, dtype=torch.float32, non_blocking=True)
    if suffix in ATTN_INPUT_WEIGHTS:
        gamma = input_gammas[layer].to(device=device, dtype=torch.float32)
        x_gpu = x_gpu / gamma.clamp_min(1e-12).unsqueeze(0)
        if arm == "respin":
            x_gpu = x_gpu @ bases[layer]
        return x_gpu
    if suffix in MLP_INPUT_WEIGHTS:
        gamma = post_gammas[layer].to(device=device, dtype=torch.float32)
        x_gpu = x_gpu / gamma.clamp_min(1e-12).unsqueeze(0)
        if arm == "respin":
            x_gpu = x_gpu @ bases[layer]
        return x_gpu
    if suffix in ATTN_OUTPUT_WEIGHTS or suffix in MLP_OUTPUT_WEIGHTS:
        return x_gpu
    return x_gpu


def _compute_joint_globals(weights: Mapping[str, torch.Tensor],
                           qnames: Sequence[str],
                           assignment: Mapping[str, str],
                           *,
                           device: torch.device) -> dict[str, torch.Tensor]:
    from prismaquant.export_native_compressed import (
        _fused_dense_group,
        compute_nvfp4_global_real,
    )

    groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for qname in qnames:
        if assignment.get(qname) != "NVFP4":
            continue
        group = _fused_dense_group(qname)
        if group is not None:
            groups[group].append(qname)
    out: dict[str, torch.Tensor] = {}
    for _group, members in groups.items():
        if not members:
            continue
        candidates = [
            compute_nvfp4_global_real(
                weights[qname].to(device=device, dtype=torch.float32),
                group_size=16,
            )
            for qname in members
        ]
        joint = torch.stack(candidates).max()
        for qname in members:
            out[qname] = joint
    return out


def _static_score(weight: torch.Tensor,
                  x: torch.Tensor,
                  qname: str,
                  *,
                  joint_global: torch.Tensor | None,
                  row_weights: torch.Tensor | None,
                  device: torch.device) -> float:
    weight_gpu = weight.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    trace: list[dict[str, object]] = []
    rendered = render_production_weight(
        weight_gpu,
        "NVFP4",
        qname=qname,
        activations={qname: x},
        levers={
            "gptq": False,
            "scale_sweep": False,
            "nvfp4_scale_rule": "static_6",
        },
        joint_global_real=joint_global,
        fisher_row_weights=row_weights,
        gate_trace=trace,
    )
    return score_render_error(
        weight_gpu.to(torch.float32),
        rendered,
        x,
        row_weights=row_weights,
    )


def _final_score_from_trace(trace: Sequence[Mapping[str, object]]) -> float:
    current = None
    for step in trace:
        if step.get("mechanism") == "baseline":
            current = float(step.get("score", 0.0) or 0.0)
            continue
        if bool(step.get("accepted", False)):
            current = float(step.get("candidate_score", current or 0.0) or 0.0)
    return float(current or 0.0)


def _method_reductions(trace: Sequence[Mapping[str, object]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for step in trace:
        if step.get("mechanism") == "baseline":
            continue
        before = float(step.get("baseline_score", 0.0) or 0.0)
        cand = float(step.get("candidate_score", before) or before)
        if bool(step.get("accepted", False)):
            out[str(step.get("mechanism", "unknown"))] += before - cand
    return dict(out)


def _layer_name(layer: int) -> str:
    return f"layer_{layer:02d}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True,
                        help="Original BF16 source model.")
    parser.add_argument("--fold-model", default=None,
                        help="Fold-only ReSpin control artifact.")
    parser.add_argument("--respin-model", required=True,
                        help="ReSpin-equivalent artifact with rotated weights.")
    parser.add_argument("--layer-config", default=None)
    parser.add_argument("--production-cache", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--layer-csv", default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--angle", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotation-checkpoint", default=None,
                        help=(
                            "Optional trained rotation checkpoint. If omitted, "
                            "read prisma_respin_equivalent.rotation_checkpoint "
                            "from --respin-model/config.json when present; "
                            "otherwise use the legacy random alternating basis."
                        ))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    device = require_cuda_hot_path("respin_render_attribution")
    run_dir = Path(args.run_dir)
    model_dir = Path(args.model)
    fold_dir = Path(args.fold_model) if args.fold_model else None
    respin_dir = Path(args.respin_model)
    layer_config = (
        Path(args.layer_config)
        if args.layer_config else
        run_dir / "artifacts" / "layer_config.json"
    )
    cache_path = (
        Path(args.production_cache)
        if args.production_cache else
        run_dir / "artifacts" / "production_weight_cache_fisher_clip_fisherclip_recached.pkl"
    )
    output_json = (
        Path(args.output_json)
        if args.output_json else
        run_dir / "artifacts" / "respin_render_attribution.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv else
        run_dir / "artifacts" / "respin_render_attribution.csv"
    )
    layer_csv = (
        Path(args.layer_csv)
        if args.layer_csv else
        run_dir / "artifacts" / "respin_render_attribution_by_layer.csv"
    )

    assignment = load_assignment(layer_config)
    cache = _load_cache(cache_path)
    qnames = [
        qname
        for qname, fmt in assignment.items()
        if fmt == "NVFP4" and _is_body_qname(qname)
    ]
    cached_nvfp4 = {
        str(qname)
        for qname, fmt in getattr(cache, "weights", {})
        if str(fmt).upper() == "NVFP4"
    }
    qnames = sorted(q for q in qnames if q in cached_nvfp4)
    if args.limit and args.limit > 0:
        qnames = qnames[:args.limit]
    if not qnames:
        raise SystemExit("no body NVFP4 qnames found in assignment/cache intersection")

    act_index = ActivationIndex(run_dir / "act", qnames)
    missing_act = [q for q in qnames if q not in act_index]
    if missing_act:
        raise SystemExit(f"{len(missing_act)} qnames missing activations, sample={missing_act[:5]}")
    h_detail = HDetailIndex(run_dir / "h_detail", qnames) if (run_dir / "h_detail").is_dir() else None

    tensor_keys = [_canonical_tensor_key(q) for q in qnames]
    gamma_keys = []
    layers = sorted({_layer_and_suffix(q)[0] for q in qnames})
    for layer in layers:
        gamma_keys.append(_norm_key(layer, "input_layernorm"))
        gamma_keys.append(_norm_key(layer, "post_attention_layernorm"))

    print(f"[respin-attrib] loading {len(qnames)} weights from source/fold/respin", flush=True)
    source_tensors = _load_tensor_subset(model_dir, [*tensor_keys, *gamma_keys])
    source_weights = {
        q: source_tensors[_canonical_tensor_key(q)]
        for q in qnames
    }
    fold_weights: dict[str, torch.Tensor] | None = None
    if fold_dir is not None:
        fold_tensors = _load_tensor_subset(fold_dir, tensor_keys)
        fold_weights = {q: fold_tensors[_canonical_tensor_key(q)] for q in qnames}
    respin_tensors = _load_tensor_subset(respin_dir, tensor_keys)
    respin_weights = {
        q: respin_tensors[_canonical_tensor_key(q)]
        for q in qnames
    }
    input_gammas = {
        layer: source_tensors[_norm_key(layer, "input_layernorm")].to(torch.float32) + 1.0
        for layer in layers
    }
    post_gammas = {
        layer: source_tensors[_norm_key(layer, "post_attention_layernorm")].to(torch.float32) + 1.0
        for layer in layers
    }

    hidden_size = int(next(iter(source_weights.values())).shape[1])
    # The first qname may be an output projection with a wider input; use the
    # RMSNorm gamma width as the residual hidden size.
    hidden_size = int(next(iter(input_gammas.values())).numel())
    rotation_checkpoint = args.rotation_checkpoint
    basis_source = "random_disjoint_givens_untrained"
    config_path = respin_dir / "config.json"
    if rotation_checkpoint is None and config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text())
            meta = cfg.get("prisma_respin_equivalent", {})
            if isinstance(meta, Mapping):
                candidate = meta.get("rotation_checkpoint")
                if candidate:
                    rotation_checkpoint = str(candidate)
                    basis_source = str(meta.get("basis_source", "trained_checkpoint"))
        except Exception as exc:
            print(f"[respin-attrib] warning: failed reading {config_path}: {exc}", flush=True)
    if rotation_checkpoint:
        transitions, bases, basis_meta = build_transitions_from_basis_checkpoint(
            rotation_checkpoint,
            hidden_size,
            max(layers) + 1,
            int(args.rank),
            device=device,
            transition_mode="paper-svd",
        )
        del transitions
        print(
            f"[respin-attrib] using trained basis checkpoint {rotation_checkpoint} "
            f"({basis_meta.get('used_rotation_count')} rotations)",
            flush=True,
        )
    else:
        transitions, bases = build_alternating_transitions(
            hidden_size,
            max(layers) + 1,
            int(args.rank),
            angle=float(args.angle),
            seed=int(args.seed),
            device=device,
        )
        del transitions

    base_joint = _compute_joint_globals(source_weights, qnames, assignment, device=device)
    fold_joint = (
        _compute_joint_globals(fold_weights, qnames, assignment, device=device)
        if fold_weights is not None else
        {}
    )
    respin_joint = _compute_joint_globals(respin_weights, qnames, assignment, device=device)

    records: list[dict[str, object]] = []
    torch.set_grad_enabled(False)
    for idx, qname in enumerate(qnames, start=1):
        layer, suffix = _layer_and_suffix(qname)
        act_cpu, row_indices = act_index.load_with_row_indices(qname)
        row_weights = _row_weights_for(h_detail, qname, row_indices, int(act_cpu.shape[0]))
        if row_weights is not None:
            row_weights = row_weights.to(device=device, dtype=torch.float32, non_blocking=True)

        x_base = act_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        x_fold = _activation_for_arm(
            qname,
            act_cpu,
            arm="fold",
            bases=bases,
            input_gammas=input_gammas,
            post_gammas=post_gammas,
            device=device,
        )
        x_respin = _activation_for_arm(
            qname,
            act_cpu,
            arm="respin",
            bases=bases,
            input_gammas=input_gammas,
            post_gammas=post_gammas,
            device=device,
        )

        base_static = _static_score(
            source_weights[qname],
            x_base,
            qname,
            joint_global=base_joint.get(qname),
            row_weights=row_weights,
            device=device,
        )
        fold_static = None
        if fold_weights is not None:
            fold_static = _static_score(
                fold_weights[qname],
                x_fold,
                qname,
                joint_global=fold_joint.get(qname),
                row_weights=row_weights,
                device=device,
            )
        respin_static = _static_score(
            respin_weights[qname],
            x_respin,
            qname,
            joint_global=respin_joint.get(qname),
            row_weights=row_weights,
            device=device,
        )

        trace: list[dict[str, object]] = []
        respin_weight_gpu = respin_weights[qname].to(
            device=device,
            dtype=torch.bfloat16,
            non_blocking=True,
        )
        rendered = render_production_weight(
            respin_weight_gpu,
            "NVFP4",
            qname=qname,
            activations={qname: x_respin},
            levers={
                "gptq": True,
                "gptq_damp_sweep": True,
                "scale_sweep": True,
                "fisher_gptq": row_weights is not None,
                "fisher_clip": False,
                "act_clip_solver": False,
                "nvfp4_scale_rule": "four_over_six_mse",
            },
            joint_global_real=respin_joint.get(qname),
            fisher_row_weights=row_weights,
            gate_trace=trace,
        )
        final_score = score_render_error(
            respin_weight_gpu.to(torch.float32),
            rendered,
            x_respin,
            row_weights=row_weights,
        )
        trace_final = _final_score_from_trace(trace)
        if trace_final > 0.0:
            final_score = trace_final

        method_reductions = _method_reductions(trace)
        fold_delta = (
            float(base_static - fold_static)
            if fold_static is not None else
            None
        )
        respin_vs_base = float(base_static - respin_static)
        respin_vs_fold = (
            float(fold_static - respin_static)
            if fold_static is not None else
            None
        )
        subsequent = float(respin_static - final_score)
        records.append({
            "qname": qname,
            "layer": int(layer),
            "suffix": suffix,
            "baseline_static_score": float(base_static),
            "fold_static_score": fold_static,
            "respin_static_score": float(respin_static),
            "progressive_final_score": float(final_score),
            "fold_reduction": fold_delta,
            "respin_reduction_vs_base": respin_vs_base,
            "respin_rotation_reduction_vs_fold": respin_vs_fold,
            "subsequent_reduction_after_respin": subsequent,
            "total_reduction": float(base_static - final_score),
            "method_reductions": method_reductions,
            "trace": trace,
        })
        if idx % 25 == 0 or idx == len(qnames):
            print(f"[respin-attrib] scored {idx}/{len(qnames)}", flush=True)

    layer_summary: dict[str, dict[str, object]] = {}
    for record in records:
        layer = int(record["layer"])
        key = _layer_name(layer)
        bucket = layer_summary.setdefault(key, {
            "layer": layer,
            "qnames": 0,
            "baseline_static_score": 0.0,
            "fold_static_score": 0.0,
            "respin_static_score": 0.0,
            "progressive_final_score": 0.0,
            "fold_reduction": 0.0,
            "respin_reduction_vs_base": 0.0,
            "respin_rotation_reduction_vs_fold": 0.0,
            "subsequent_reduction_after_respin": 0.0,
            "total_reduction": 0.0,
            "method_reductions": defaultdict(float),
        })
        bucket["qnames"] = int(bucket["qnames"]) + 1
        for field in (
            "baseline_static_score",
            "respin_static_score",
            "progressive_final_score",
            "respin_reduction_vs_base",
            "subsequent_reduction_after_respin",
            "total_reduction",
        ):
            bucket[field] = float(bucket[field]) + float(record[field])
        if record.get("fold_static_score") is not None:
            bucket["fold_static_score"] = float(bucket["fold_static_score"]) + float(record["fold_static_score"])
            bucket["fold_reduction"] = float(bucket["fold_reduction"]) + float(record["fold_reduction"] or 0.0)
            bucket["respin_rotation_reduction_vs_fold"] = (
                float(bucket["respin_rotation_reduction_vs_fold"])
                + float(record["respin_rotation_reduction_vs_fold"] or 0.0)
            )
        methods = record.get("method_reductions", {})
        if isinstance(methods, Mapping):
            for name, value in methods.items():
                bucket["method_reductions"][str(name)] += float(value)  # type: ignore[index]

    layer_rows: list[dict[str, object]] = []
    all_methods = sorted({
        name
        for record in records
        for name in (
            record.get("method_reductions", {}).keys()
            if isinstance(record.get("method_reductions"), Mapping) else
            []
        )
    })
    for key in sorted(layer_summary, key=lambda item: int(item.split("_")[1])):
        bucket = layer_summary[key]
        baseline = float(bucket["baseline_static_score"])
        row = {
            "layer": bucket["layer"],
            "qnames": bucket["qnames"],
            "baseline_static_score": baseline,
            "fold_static_score": bucket["fold_static_score"],
            "respin_static_score": bucket["respin_static_score"],
            "progressive_final_score": bucket["progressive_final_score"],
            "fold_reduction": bucket["fold_reduction"],
            "respin_reduction_vs_base": bucket["respin_reduction_vs_base"],
            "respin_rotation_reduction_vs_fold": bucket["respin_rotation_reduction_vs_fold"],
            "subsequent_reduction_after_respin": bucket["subsequent_reduction_after_respin"],
            "total_reduction": bucket["total_reduction"],
            "total_relative_reduction": (
                float(bucket["total_reduction"]) / baseline if baseline > 0.0 else 0.0
            ),
        }
        method_map = bucket["method_reductions"]
        for name in all_methods:
            row[f"method_{name}_reduction"] = (
                float(method_map.get(name, 0.0))  # type: ignore[union-attr]
            )
        layer_rows.append(row)

    gate_summary = _summarize_gate_records(records)
    total_baseline = sum(float(r["baseline_static_score"]) for r in records)
    total_fold = sum(float(r["fold_static_score"] or 0.0) for r in records)
    total_respin = sum(float(r["respin_static_score"]) for r in records)
    total_final = sum(float(r["progressive_final_score"]) for r in records)
    summary = {
        "run_dir": str(run_dir),
        "model": str(model_dir),
        "fold_model": str(fold_dir) if fold_dir is not None else None,
        "respin_model": str(respin_dir),
        "basis_source": basis_source,
        "rotation_checkpoint": str(rotation_checkpoint) if rotation_checkpoint else None,
        "layer_config": str(layer_config),
        "production_cache": str(cache_path),
        "qnames": len(records),
        "rank": int(args.rank),
        "angle": float(args.angle),
        "seed": int(args.seed),
        "metric": (
            "Fisher-weighted output MSE when h-detail rows are available; "
            "otherwise output MSE. Lower is better."
        ),
        "prismaclip_note": (
            "PrismaClip thresholds are not replayed because existing "
            "thresholds were solved in the non-ReSpin activation basis."
        ),
        "totals": {
            "baseline_static_score": total_baseline,
            "fold_static_score": total_fold if fold_dir is not None else None,
            "respin_static_score": total_respin,
            "progressive_final_score": total_final,
            "fold_reduction": (
                total_baseline - total_fold if fold_dir is not None else None
            ),
            "respin_reduction_vs_base": total_baseline - total_respin,
            "respin_rotation_reduction_vs_fold": (
                total_fold - total_respin if fold_dir is not None else None
            ),
            "subsequent_reduction_after_respin": total_respin - total_final,
            "total_reduction": total_baseline - total_final,
            "total_relative_reduction": (
                (total_baseline - total_final) / total_baseline
                if total_baseline > 0.0 else
                0.0
            ),
        },
        "gate_summary_after_respin": gate_summary,
        "layers": layer_rows,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))

    with output_csv.open("w", newline="") as fh:
        fieldnames = [
            "qname",
            "layer",
            "suffix",
            "baseline_static_score",
            "fold_static_score",
            "respin_static_score",
            "progressive_final_score",
            "fold_reduction",
            "respin_reduction_vs_base",
            "respin_rotation_reduction_vs_fold",
            "subsequent_reduction_after_respin",
            "total_reduction",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})

    with layer_csv.open("w", newline="") as fh:
        base_fields = [
            "layer",
            "qnames",
            "baseline_static_score",
            "fold_static_score",
            "respin_static_score",
            "progressive_final_score",
            "fold_reduction",
            "respin_reduction_vs_base",
            "respin_rotation_reduction_vs_fold",
            "subsequent_reduction_after_respin",
            "total_reduction",
            "total_relative_reduction",
        ]
        fieldnames = [*base_fields, *[f"method_{name}_reduction" for name in all_methods]]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(layer_rows)

    print(f"[respin-attrib] wrote {output_json}", flush=True)
    print(f"[respin-attrib] wrote {output_csv}", flush=True)
    print(f"[respin-attrib] wrote {layer_csv}", flush=True)
    print(json.dumps({
        "qnames": len(records),
        "totals": summary["totals"],
        "gate_summary_after_respin": gate_summary,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
