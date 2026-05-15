"""W4A4 geodesic-sweep diagnostic for HDQ rotation insertion.

For each sampled cluster, sweeps R(t) = orthogonalize((1-t) I + t R_init)
for t in [0, 1] and measures THREE losses at each t with production-equivalent
STE quantizers:

  - W-only: y = (x M^T) @ Q_w(W M^T)^T        // weight quant error only
  - A-only: y = Q_a(x M^T) @ (W M^T)^T        // activation quant error only
  - W4A4:   y = Q_a(x M^T) @ Q_w(W M^T)^T     // full runtime

Output: per (cluster_key, init, t, loss_type) row in JSONL, plus a
per-cluster summary printing whether the W4A4 minimum lies at t > 0
(activation benefit recoverable by adding A quant to the solver loss)
or at t = 0 (genuine NVFP4 G=16 inversion — pivot away from rotations).

Convention (matches the solver's): M is the code's matrix; runtime applies
M^T to both input activations and stored weight. The math closes for any
orthogonal M because (x M^T)(W M^T)^T = x M^T M W^T = x W^T.
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
    apply_block_rotation_input,
    calibrate_zigzag_permutation,
    default_insertion_specs,
    ste_rtn_nvfp4_per_group,
    sylvester_hadamard,
)
from prismaquant.run_joint_hadamard_search import (
    _ActivationCapture,
    _build_cluster_inputs,
    _resolve_module,
)
from prismaquant.sensitivity_probe import load_calibration


def _orthogonalize(M: torch.Tensor) -> torch.Tensor:
    """Closest orthogonal to M via QR (with diag-sign fix)."""
    q, r = torch.linalg.qr(M)
    d = torch.diag(r).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    return (q * d.unsqueeze(0)).contiguous()


def _build_init_R(
    kind: str, g: int, weights: Sequence[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Construct R_init for the sweep. Returns float32 (g, g)."""
    if kind == "sylvester":
        return sylvester_hadamard(g, device=device, dtype=torch.float32).contiguous()
    if kind == "svd_v":
        cov = torch.zeros(g, g, device=device, dtype=torch.float64)
        for w in weights:
            w64 = w.to(dtype=torch.float64)
            wb = w64.view(w64.shape[0], -1, g)
            cov = cov + torch.einsum("obg,obh->gh", wb, wb)
        cov = (cov + cov.t()) / 2
        _ev, evec = torch.linalg.eigh(cov)
        evec = evec.flip(dims=[1])
        return evec.t().to(dtype=torch.float32).contiguous()
    if kind == "random":
        gen = torch.Generator(device=device).manual_seed(0)
        A = torch.randn(g, g, generator=gen, device=device, dtype=torch.float32)
        return _orthogonalize(A)
    raise ValueError(f"unknown init kind {kind!r}")


def _interp_R(R_init: torch.Tensor, t: float) -> torch.Tensor:
    """Orthogonalize((1-t) I + t R_init). Smooth path I -> R_init in O(g)."""
    g = int(R_init.shape[0])
    I = torch.eye(g, device=R_init.device, dtype=R_init.dtype)
    if t == 0.0:
        return I.clone()
    if t == 1.0:
        return R_init.clone()
    M = (1.0 - t) * I + t * R_init
    return _orthogonalize(M)


def _row_chunked_sqsum(
    diff: torch.Tensor, row_chunk: int
) -> tuple[float, int]:
    """Sum of squares of diff, returned with the row count for normalization."""
    total = 0.0
    n = 0
    for s in range(0, diff.shape[0], row_chunk):
        chunk = diff[s : s + row_chunk]
        total += float(chunk.pow(2).sum().item())
        n += int(chunk.shape[0])
    return total, n


@torch.no_grad()
def _three_losses(
    x_orig: torch.Tensor,  # (N, cols)
    w: torch.Tensor,  # (out, cols)
    M: torch.Tensor,  # (g, g)
    g: int,
    row_chunk: int = 1024,
) -> dict[str, float]:
    """Compute W-only, A-only, W4A4 losses at the current M.

    All quantization uses ``ste_rtn_nvfp4_per_group`` (max_abs/6, E2M1
    codebook, per-G block, no scale rounding). Matches the joint-search
    STE quantizer.
    """
    # No-rotation reference output
    y_target = x_orig @ w.t()  # (N, out)

    # Rotated activations and weights (storage convention: M^T applied)
    x_rot = apply_block_rotation_input(x_orig, M.t())  # (N, cols)
    w_rot = apply_block_rotation_input(w, M.t())  # (out, cols)

    # Quantize independently
    w_q = ste_rtn_nvfp4_per_group(w_rot, group_size=g)
    # Activation quant: same per-G16 max/6 STE recipe, per-row dynamic
    x_q = ste_rtn_nvfp4_per_group(x_rot, group_size=g)

    # W-only: perfect activations, only weight quantized
    diff_w = y_target - (x_rot @ w_q.t())
    sqs_w, n_w = _row_chunked_sqsum(diff_w, row_chunk)

    # A-only: perfect weights, only activation quantized
    diff_a = y_target - (x_q @ w_rot.t())
    sqs_a, n_a = _row_chunked_sqsum(diff_a, row_chunk)

    # W4A4: full runtime
    diff_b = y_target - (x_q @ w_q.t())
    sqs_b, n_b = _row_chunked_sqsum(diff_b, row_chunk)

    n_elem = int(w.shape[0])  # output features
    return {
        "w_only": sqs_w / max(1, n_w * n_elem),
        "a_only": sqs_a / max(1, n_a * n_elem),
        "w4a4": sqs_b / max(1, n_b * n_elem),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="W4A4 geodesic sweep diagnostic")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group-size", type=int, default=NVFP4_GROUP_SIZE)
    p.add_argument("--n-calib-samples", type=int, default=16)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--dataset", default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--max-act-rows", type=int, default=2048,
                   help="More rows than the solver uses; W4A4 needs N x cols "
                        "for stable per-token block scale stats.")
    p.add_argument("--n-clusters-per-kind", type=int, default=3,
                   help="How many clusters per insertion-kind to sweep.")
    p.add_argument("--inits", default="sylvester,svd_v,random",
                   help="Comma-separated init kinds to sweep.")
    p.add_argument("--n-t-points", type=int, default=11,
                   help="Sweep points across [0, 1], evenly spaced.")
    p.add_argument("--body-layer-prefix", default="model.layers")
    p.add_argument("--hidden-dim", type=int, default=None)
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    staged, _cleanup = stage_multimodal(args.model)
    device = require_cuda_hot_path("w4a4_geodesic_sweep")
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
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
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
    print(f"[sweep] {len(specs)} insertion points discovered", flush=True)

    # Sample clusters: take first N per insertion_kind
    by_kind: dict[str, list] = {}
    for spec in specs:
        by_kind.setdefault(spec.kind.value, []).append(spec)
    selected = []
    for kind, group in by_kind.items():
        selected.extend(group[: args.n_clusters_per_kind])
    print(
        f"[sweep] sampling {len(selected)} clusters across "
        f"{len(by_kind)} insertion kinds", flush=True
    )

    # Attach captures only for the selected clusters
    captures: dict[str, _ActivationCapture] = {}
    handles = []
    for spec in selected:
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

    cluster_inputs = _build_cluster_inputs(model, selected, captures)
    print(
        f"[sweep] {len(cluster_inputs)}/{len(selected)} clusters populated; "
        f"n_act_rows = {args.max_act_rows}", flush=True
    )

    init_kinds = tuple(s.strip() for s in args.inits.split(",") if s.strip())
    ts = [i / (args.n_t_points - 1) for i in range(args.n_t_points)]
    g = int(args.group_size)

    rows = []
    with out_path.open("w") as fh:
        for spec in selected:
            ck = spec.cluster_key
            if ck not in cluster_inputs:
                continue
            cins = cluster_inputs[ck]
            kind = spec.kind.value
            # Use ONE representative consumer (first sibling) for sweep speed.
            # Siblings share x; the loss is additive across them.
            target = cins.targets[0]
            x = target.activations.to(device=device, dtype=torch.float32)
            w = target.weight.to(device=device, dtype=torch.float32)
            weights_list = [t.weight.to(device=device, dtype=torch.float32)
                            for t in cins.targets]

            # Baseline at M = I (no rotation)
            I_ = torch.eye(g, device=device, dtype=torch.float32)
            base = _three_losses(x, w, I_, g)

            for init in init_kinds:
                try:
                    R_init = _build_init_R(init, g, weights_list, device)
                except (RuntimeError, ValueError) as e:
                    print(f"[sweep] init {init!r} failed for {ck}: {e}",
                          file=sys.stderr, flush=True)
                    continue
                for t in ts:
                    M = _interp_R(R_init, float(t))
                    losses = _three_losses(x, w, M, g)
                    row = {
                        "cluster": ck,
                        "kind": kind,
                        "qname": target.qname,
                        "init": init,
                        "t": float(t),
                        "w_only": losses["w_only"],
                        "a_only": losses["a_only"],
                        "w4a4": losses["w4a4"],
                        "w_only_at_I": base["w_only"],
                        "a_only_at_I": base["a_only"],
                        "w4a4_at_I": base["w4a4"],
                    }
                    fh.write(json.dumps(row) + "\n")
                    rows.append(row)
                    print(
                        f"[sweep] {kind:10s} {ck[:40]:40s} init={init:9s} "
                        f"t={t:.2f}  w_only={losses['w_only']:.3e}  "
                        f"a_only={losses['a_only']:.3e}  "
                        f"w4a4={losses['w4a4']:.3e}",
                        flush=True,
                    )

    print(f"[sweep] wrote {len(rows)} rows to {out_path}", flush=True)

    # Quick verdict per (init, kind): is t* > 0 for the W4A4 minimum?
    from collections import defaultdict
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["init"], r["kind"], r["cluster"])].append(r)
    print("\n[sweep] per-cluster W4A4 minimum location:")
    print(f"  {'init':<10} {'kind':<10} {'cluster':<40} {'t*':>5} "
          f"{'w4a4(t*)/w4a4(0)':>20}")
    for (init, kind, cluster), curve in by_group.items():
        curve.sort(key=lambda r: r["t"])
        ws = [r["w4a4"] for r in curve]
        idx_min = ws.index(min(ws))
        t_star = curve[idx_min]["t"]
        ratio = ws[idx_min] / max(ws[0], 1e-30)
        print(f"  {init:<10} {kind:<10} {cluster[:40]:<40} {t_star:>5.2f} "
              f"{ratio:>20.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
