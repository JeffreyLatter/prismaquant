"""Random-orthogonal sampling diagnostic for HDQ basin-search question.

For each sampled cluster, run:
  1. Identity-init Adam (the current production solver) — baseline.
  2. N Haar-uniform random-orthogonal-init Adam runs (varying seed).
  3. The geodesic-sweep multi-init (sylvester_t0p3, sylvester_t0p5, svd_v).

Tabulate per-cluster:
  - best random-init Adam rotated_score
  - identity-init Adam rotated_score
  - count of random inits whose solver beat identity-init
  - distribution width (std of random results)

If random sampling consistently finds rotations meaningfully better than
identity-Adam, then identity-init is missing basins and we need a better
global-search method. If not, identity-Adam is empirically near-global
and the "2% Fisher-MSE ceiling" we measured is the actual landscape
ceiling, not a search artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import stage_multimodal
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.hadamard_duquant import (
    NVFP4_GROUP_SIZE,
    default_insertion_specs,
    solve_cluster_rotation,
)
from prismaquant.run_joint_hadamard_search import (
    _ActivationCapture,
    _build_cluster_inputs,
    _resolve_module,
)
from prismaquant.sensitivity_probe import load_calibration


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Random-orthogonal HDQ diagnostic")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, help="JSONL output path")
    p.add_argument("--group-size", type=int, default=NVFP4_GROUP_SIZE)
    p.add_argument("--n-calib-samples", type=int, default=4)
    p.add_argument("--calib-seqlen", type=int, default=256)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--dataset", default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--max-act-rows", type=int, default=512)
    p.add_argument(
        "--target-clusters",
        default="",
        help="Comma-separated cluster keys to target. Empty (default) "
        "picks 3 'winners' and 3 'losers' from the 4B identity-init sidecar.",
    )
    p.add_argument(
        "--existing-sidecar",
        default="",
        help="Path to an identity-init W4A4 sidecar; used to pick winners "
        "and losers automatically when --target-clusters is empty.",
    )
    p.add_argument("--n-random-samples", type=int, default=30)
    p.add_argument("--solver-iters", type=int, default=300)
    p.add_argument("--solver-lr", type=float, default=1e-3)
    p.add_argument(
        "--solver-early-stop-patience", type=int, default=100,
    )
    p.add_argument("--body-layer-prefix", default="model.layers")
    p.add_argument("--hidden-dim", type=int, default=None)
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    staged, _cleanup = stage_multimodal(args.model)
    device = require_cuda_hot_path("random_orthogonal_diagnostic")
    local_only = Path(staged).exists()

    tokenizer = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only,
    )
    if args.dataset:
        calib_ids = load_calibration(
            tokenizer, args.dataset, args.n_calib_samples, args.calib_seqlen,
        )
    else:
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer, args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed,
        )
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": local_only,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = "cuda"
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(device)
    if device.type != "cuda":
        model.to(device)
    model.eval()

    specs = default_insertion_specs(
        model,
        group_size=int(args.group_size),
        body_layer_prefix=args.body_layer_prefix,
        hidden_dim=int(args.hidden_dim) if args.hidden_dim else None,
    )
    by_key = {s.cluster_key: s for s in specs}
    print(f"[diag] {len(specs)} insertion points discovered", flush=True)

    # Resolve target clusters: explicit list, or auto-pick from sidecar
    target_keys: list[str] = []
    if args.target_clusters.strip():
        target_keys = [s.strip() for s in args.target_clusters.split(",") if s.strip()]
    elif args.existing_sidecar:
        sc = json.load(open(args.existing_sidecar))
        deltas: list[tuple[str, float]] = []
        for ck, cd in sc["clusters"].items():
            c = cd.get("candidates", {})
            if "no_rot+NVFP4" not in c or "rot+NVFP4" not in c:
                continue
            base = c["no_rot+NVFP4"]["fisher_mse"]
            rot = c["rot+NVFP4"]["fisher_mse"]
            if base <= 0 or rot != rot:
                continue
            deltas.append((ck, (rot / base - 1.0) * 100.0))
        deltas.sort(key=lambda x: x[1])
        winners = [ck for ck, _ in deltas[:3]]  # most negative Δ
        losers = [
            ck for ck, _ in deltas
            if abs(_ - 0.0) < 0.01  # essentially-no-gain clusters
        ][:3]
        target_keys = winners + losers
        print(f"[diag] auto-picked winners: {winners}", flush=True)
        print(f"[diag] auto-picked losers:  {losers}", flush=True)
    else:
        raise SystemExit(
            "must pass --target-clusters or --existing-sidecar"
        )
    target_specs = [by_key[k] for k in target_keys if k in by_key]
    print(f"[diag] targeting {len(target_specs)} clusters", flush=True)

    # Capture activations only for target clusters
    captures: dict[str, _ActivationCapture] = {}
    handles = []
    for spec in target_specs:
        if not spec.consumer_qnames:
            continue
        first = spec.consumer_qnames[0]
        mod = _resolve_module(model, first)
        if mod is None:
            continue
        cap = _ActivationCapture(max_rows=args.max_act_rows)
        captures[spec.cluster_key] = cap
        handles.append(mod.register_forward_pre_hook(cap))
    try:
        with torch.no_grad():
            for sample in calib_ids:
                if sample.dim() == 1:
                    sample = sample.unsqueeze(0)
                sample = sample.to(device=device)
                model(sample)
    finally:
        for h in handles:
            h.remove()

    cluster_inputs = _build_cluster_inputs(model, target_specs, captures)
    print(f"[diag] populated {len(cluster_inputs)}/{len(target_specs)} clusters", flush=True)

    g = int(args.group_size)
    fh = out_path.open("w")
    try:
        for spec in target_specs:
            ck = spec.cluster_key
            if ck not in cluster_inputs:
                continue
            cins = cluster_inputs[ck]
            kind = spec.kind.value

            # 1. Identity-init baseline (the production solver path)
            r_id = solve_cluster_rotation(
                cins.targets, group_size=g, format_label="NVFP4",
                init_strategy="identity", loss_kind="w4a4",
                n_iters=int(args.solver_iters), lr=float(args.solver_lr),
                early_stop_patience=int(args.solver_early_stop_patience),
            )
            base_score = r_id.baseline_score
            id_score = r_id.rotated_score
            id_gain = (id_score / base_score - 1) * 100 if base_score > 0 else 0
            print(
                f"[diag] {kind:<10} {ck:<45} identity: rot={id_score:.4e} "
                f"gain={id_gain:+.3f}%",
                flush=True,
            )
            fh.write(json.dumps({
                "cluster": ck, "kind": kind, "init": "identity",
                "seed": 0, "rotated_score": id_score,
                "baseline_score": base_score, "gain_pct": id_gain,
            }) + "\n")

            # 2. Multi-init candidates (sylvester scales + svd_v)
            for init in ["sylvester_t0p3", "sylvester_t0p5", "svd_v", "sylvester"]:
                try:
                    r = solve_cluster_rotation(
                        cins.targets, group_size=g, format_label="NVFP4",
                        init_strategy=init, loss_kind="w4a4",
                        n_iters=int(args.solver_iters),
                        lr=float(args.solver_lr),
                        early_stop_patience=int(args.solver_early_stop_patience),
                    )
                    g_pct = (r.rotated_score / base_score - 1) * 100
                    fh.write(json.dumps({
                        "cluster": ck, "kind": kind, "init": init,
                        "seed": 0, "rotated_score": r.rotated_score,
                        "baseline_score": base_score, "gain_pct": g_pct,
                    }) + "\n")
                except Exception as e:
                    print(f"[diag]   {init} failed: {e}", file=sys.stderr)

            # 3. N random-orthogonal-init Adam runs
            random_scores: list[float] = []
            for seed in range(args.n_random_samples):
                try:
                    r = solve_cluster_rotation(
                        cins.targets, group_size=g, format_label="NVFP4",
                        init_strategy="random", seed=seed, loss_kind="w4a4",
                        n_iters=int(args.solver_iters),
                        lr=float(args.solver_lr),
                        early_stop_patience=int(args.solver_early_stop_patience),
                    )
                    g_pct = (r.rotated_score / base_score - 1) * 100
                    random_scores.append(r.rotated_score)
                    fh.write(json.dumps({
                        "cluster": ck, "kind": kind, "init": "random",
                        "seed": seed, "rotated_score": r.rotated_score,
                        "baseline_score": base_score, "gain_pct": g_pct,
                    }) + "\n")
                except Exception as e:
                    print(f"[diag]   random seed={seed} failed: {e}",
                          file=sys.stderr)

            if random_scores:
                rs = torch.tensor(random_scores)
                best_random = float(rs.min().item())
                best_random_gain = (best_random / base_score - 1) * 100
                n_beat_id = int((rs < id_score).sum().item())
                print(
                    f"[diag] {kind:<10} {ck:<45} "
                    f"best_random={best_random:.4e} gain={best_random_gain:+.3f}%  "
                    f"({n_beat_id}/{len(random_scores)} beat identity)",
                    flush=True,
                )
            fh.flush()
    finally:
        fh.close()

    print(f"[diag] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
