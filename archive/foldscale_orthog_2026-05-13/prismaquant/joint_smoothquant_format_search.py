"""Joint (SmoothQuant α, format) golden-section search per cluster.

For each fold-eligible cluster (Linears sharing a predecessor RMSNorm), this
runs a continuous-α optimization per candidate format and reports the
per-(cluster, format) (α_opt, output_mse) table. By default this is pure
isolation: no GPTQ, no scale_sweep, no fisher_gptq, no PrismaClip — just
SmoothQuant/AWQ rescale + the format's RTN. ``--render-levers`` can enable
selected downstream render passes such as GPTQ for comparison runs.

The output is a JSON record per cluster:

    {
        "cluster_key": "model.layers.0.input_layernorm",
        "members": ["model.layers.0.self_attn.q_proj", ...],
        "in_features": 896,
        "per_format": {
            "NVFP4":       {"alpha": 0.13, "score": 0.0042, "alpha_0_score": 0.0051},
            "MXFP8_E4M3":  {"alpha": 0.41, "score": 0.0011, "alpha_0_score": 0.0019},
            "FP8_E4M3":    {"alpha": 0.62, "score": 0.0004, "alpha_0_score": 0.0008},
            "BF16":        {"alpha": 0.00, "score": 0.0000, "alpha_0_score": 0.0000}
        }
    }

Allocator integration is intentionally NOT part of this script; the goal here
is to surface whether joint optimization finds materially different (α,
format) than today's separate optimization.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from prismaquant.awq import (
    awq_activation_mean,
    awq_scale_from_stats,
    awq_weight_mean,
    normalize_awq_scale,
)
from prismaquant.production_weight_cache import (
    _AWQ_INPUT_LN_LEAVES,
    _AWQ_POST_LN_LEAVES,
    _awq_format_group_size,
    _awq_group_key_from_qname,
    _fold_runtime_output_mse,
    _render_fold_scaled_for_cache,
    _smoothquant_alpha_hi_for_formats,
)
from prismaquant.sensitivity_probe import load_calibration


DEFAULT_FORMATS = ("NVFP4", "MXFP8_E4M3", "FP8_E4M3", "BF16")
SCALE_MODES = ("smoothquant_max", "awq_mean")


# ---------------------------------------------------------------------------
# Golden-section search


def golden_section_search(
    f, a: float, b: float, *, tol: float = 1e-3, max_iter: int = 30
) -> tuple[float, float]:
    """Minimize unimodal ``f`` over ``[a, b]``. Returns ``(x_opt, f_opt)``."""
    phi = (1 + 5 ** 0.5) / 2
    inv_phi = 1.0 / phi
    inv_phi2 = 1.0 / (phi * phi)

    h = float(b - a)
    if h <= tol:
        x = 0.5 * (a + b)
        return x, float(f(x))

    n = max(1, int(math.ceil(math.log(tol / h) / math.log(inv_phi))))
    n = min(n, max_iter)

    c = a + inv_phi2 * h
    d = a + inv_phi * h
    yc = float(f(c))
    yd = float(f(d))
    for _ in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h *= inv_phi
            c = a + inv_phi2 * h
            yc = float(f(c))
        else:
            a = c
            c = d
            yc = yd
            h *= inv_phi
            d = a + inv_phi * h
            yd = float(f(d))
    if yc < yd:
        return c, yc
    return d, yd


# ---------------------------------------------------------------------------
# Cluster + activation capture


class _Target:
    __slots__ = ("name", "weight", "activations")

    def __init__(self, name: str, weight: torch.Tensor, activations: torch.Tensor):
        self.name = name
        self.weight = weight
        self.activations = activations


def _collect_eligible_qnames(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    out: list[tuple[str, nn.Linear]] = []
    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = qname.rsplit(".", 1)[-1]
        if leaf in _AWQ_INPUT_LN_LEAVES or leaf in _AWQ_POST_LN_LEAVES:
            out.append((qname, mod))
    return out


def _capture_activations(
    model: nn.Module,
    qnames: list[tuple[str, nn.Linear]],
    calib_ids: torch.Tensor,
    *,
    max_rows: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run a forward pass with pre-hooks to capture per-Linear inputs."""
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(qname: str):
        def hook(_module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            captures[qname].append(x.detach().to(device=device, dtype=torch.float32).reshape(-1, x.shape[-1]))
        return hook

    for qname, mod in qnames:
        handles.append(mod.register_forward_pre_hook(_make_hook(qname)))
    try:
        with torch.no_grad():
            for i in range(calib_ids.shape[0]):
                ids = calib_ids[i : i + 1].to(device=device)
                model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out: dict[str, torch.Tensor] = {}
    for qname, chunks in captures.items():
        if not chunks:
            continue
        cat = torch.cat(chunks, dim=0)
        if cat.shape[0] > max_rows:
            stride = max(1, cat.shape[0] // max_rows)
            cat = cat[::stride][:max_rows].contiguous()
        out[qname] = cat
    return out


# ---------------------------------------------------------------------------
# SmoothQuant scale + scoring


def _smoothquant_scale(
    targets: list[_Target], alpha: float, *, eps: float = 1e-4, clamp_ratio: float = 10.0
) -> torch.Tensor:
    """SmoothQuant-style scale: max(|x|)^α / max(|W|)^(1-α), then normalize."""
    cols = int(targets[0].weight.shape[1])
    device = targets[0].weight.device
    x_max = torch.zeros(cols, device=device, dtype=torch.float32)
    w_max = torch.zeros(cols, device=device, dtype=torch.float32)
    for t in targets:
        x = t.activations.to(device=device, dtype=torch.float32).reshape(-1, cols)
        w = t.weight.to(device=device, dtype=torch.float32)
        x_max = torch.maximum(x_max, x.abs().amax(dim=0))
        w_max = torch.maximum(w_max, w.abs().amax(dim=0))
    raw = x_max.clamp_min(eps).pow(float(alpha)) / w_max.clamp_min(eps).pow(1.0 - float(alpha))
    return normalize_awq_scale(raw, eps=eps, clamp_ratio=clamp_ratio)


def _awq_mean_scale(
    targets: list[_Target],
    ratio: float,
    fmt: str,
    *,
    eps: float = 1e-4,
    clamp_ratio: float = 10.0,
) -> torch.Tensor:
    """AWQ duo-scaling: mean(|x|)^r / mean(|W_norm|)^(1-r), normalized.

    ``W_norm`` is the per-microscale-group normalized weight (each group of
    G consecutive input channels divided by the group's max-abs), so the
    statistic reflects "channel salience within microscale group" rather
    than raw max-abs. Same fold mathematics as SmoothQuant; different scale
    formula.
    """
    cols = int(targets[0].weight.shape[1])
    device = targets[0].weight.device
    group_size = max(0, int(_awq_format_group_size(fmt)))
    x_mean = awq_activation_mean(
        [t.activations for t in targets], cols, device=device, eps=eps
    )
    w_mean = awq_weight_mean(
        [t.weight for t in targets],
        [group_size for _ in targets],
        cols,
        device=device,
        eps=eps,
    )
    return awq_scale_from_stats(
        x_mean,
        w_mean,
        ratio=float(ratio),
        duo_scaling=True,
        eps=eps,
        clamp_ratio=clamp_ratio,
    )


def _scale_for_mode(
    targets: list[_Target],
    knob: float,
    fmt: str,
    mode: str,
) -> torch.Tensor:
    if mode == "smoothquant_max":
        return _smoothquant_scale(targets, knob)
    if mode == "awq_mean":
        return _awq_mean_scale(targets, knob, fmt)
    raise ValueError(f"unknown scale mode: {mode}")


_DEFAULT_RENDER_LEVERS: dict[str, object] = {
    "gptq": False,
    "scale_sweep": False,
    "awq_round": False,
    "fisher_gptq": False,
}


def _parse_render_levers(raw: str) -> dict[str, object]:
    levers = dict(_DEFAULT_RENDER_LEVERS)
    aliases = {
        "none": None,
        "identity": None,
        "rtn": None,
        "gptq": "gptq",
        "scale_sweep": "scale_sweep",
        "scale-sweep": "scale_sweep",
        "awq_round": "awq_round",
        "awq-round": "awq_round",
    }
    for item in (raw or "").split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                f"unknown render lever {item!r}; allowed: "
                "gptq, scale_sweep, awq_round"
            )
        resolved = aliases[key]
        if resolved is not None:
            levers[resolved] = True
    return levers


def _score_cluster_per_member(
    targets: list[_Target],
    knob: float,
    fmt: str,
    mode: str = "smoothquant_max",
    *,
    render_levers: dict[str, object] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return (total_score, per_qname_score) at (knob, fmt) under ``mode``.

    ``knob`` is α for ``smoothquant_max`` and r for ``awq_mean``. Mechanism
    (fold-scale) is identical across modes; only the scale formula differs.
    """
    device = targets[0].weight.device
    cols = int(targets[0].weight.shape[1])
    if knob <= 1e-9:
        scale = torch.ones(cols, device=device, dtype=torch.float32)
    else:
        scale = _scale_for_mode(targets, knob, fmt, mode)
    total = 0.0
    per_member: dict[str, float] = {}
    with torch.no_grad():
        for t in targets:
            w = t.weight.to(device=device, dtype=torch.float32)
            x = t.activations.to(device=device, dtype=torch.float32).reshape(-1, cols)
            w_scaled = w * scale.unsqueeze(0)
            a_scaled = x / scale.clamp_min(1e-12).unsqueeze(0)
            rendered = _render_fold_scaled_for_cache(
                qname=t.name,
                fmt=fmt,
                weight_scaled=w_scaled,
                activations_scaled=a_scaled,
                levers=render_levers or _DEFAULT_RENDER_LEVERS,
                joint_global_real=None,
            )
            score = _fold_runtime_output_mse(
                weight=w,
                rendered_scaled_weight=rendered.to(device=device, dtype=torch.float32),
                activations=t.activations,
                scale=scale,
                fmt=fmt,
            )
            per_member[t.name] = float(score)
            total += float(score)
    return float(total), per_member


def _score_cluster(
    targets: list[_Target],
    knob: float,
    fmt: str,
    mode: str = "smoothquant_max",
    *,
    render_levers: dict[str, object] | None = None,
) -> float:
    total, _ = _score_cluster_per_member(
        targets,
        knob,
        fmt,
        mode,
        render_levers=render_levers,
    )
    return total


# ---------------------------------------------------------------------------
# Main


def _build_clusters(
    qnames: list[tuple[str, nn.Linear]],
    activations: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, list[_Target]]:
    clusters: dict[str, list[_Target]] = defaultdict(list)
    for qname, mod in qnames:
        if qname not in activations:
            continue
        ck = _awq_group_key_from_qname(qname)
        if ck is None:
            continue
        clusters[ck].append(
            _Target(
                name=qname,
                weight=mod.weight.detach().to(device=device),
                activations=activations[qname],
            )
        )
    return clusters


def _search_cluster(
    targets: list[_Target],
    formats: list[str],
    *,
    alpha_lo: float,
    alpha_hi: float,
    tol: float,
    max_iter: int,
    modes: tuple[str, ...] = ("smoothquant_max",),
    render_levers: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Search per-(format, mode); top-level dict picks the best mode per format.

    Returned schema per format:

        {
            "alpha": <best knob>,
            "score": <best score>,
            "alpha_0_score": <identity score>,
            "identity_per_member_score": {qname: score, ...},
            "per_member_score": {qname: score, ...},  # winner's per-member
            "mode": <winning mode>,
            "per_mode": {
                mode_name: {
                    "alpha": ..., "score": ..., "per_member_score": ...,
                    "evals": int,
                },
                ...
            },
        }
    """
    out: dict[str, dict[str, object]] = {}
    render_levers = render_levers or dict(_DEFAULT_RENDER_LEVERS)
    for fmt in formats:
        if fmt.upper() == "BF16":
            score_id, per_member_id = _score_cluster_per_member(
                targets,
                0.0,
                fmt,
                render_levers=render_levers,
            )
            out[fmt] = {
                "alpha": 0.0,
                "score": float(score_id),
                "alpha_0_score": float(score_id),
                "requested_alpha_hi": float(alpha_hi),
                "effective_alpha_hi": 0.0,
                "identity_per_member_score": per_member_id,
                "per_member_score": per_member_id,
                "mode": "identity",
                "per_mode": {},
            }
            continue
        score_id, per_member_id = _score_cluster_per_member(
            targets,
            0.0,
            fmt,
            "smoothquant_max",
            render_levers=render_levers,
        )
        per_mode_results: dict[str, dict[str, object]] = {}
        for mode in modes:
            eval_calls = {"n": 0}
            mode_alpha_hi = (
                _smoothquant_alpha_hi_for_formats(
                    [fmt],
                    requested_hi=float(alpha_hi),
                )
                if mode == "smoothquant_max"
                else float(alpha_hi)
            )
            mode_alpha_lo = min(float(alpha_lo), float(mode_alpha_hi))
            if mode == "smoothquant_max" and float(mode_alpha_hi) <= 0.0:
                per_mode_results[mode] = {
                    "alpha": 0.0,
                    "score": float(score_id),
                    "evals": 0,
                    "requested_alpha_hi": float(alpha_hi),
                    "effective_alpha_hi": float(mode_alpha_hi),
                    "per_member_score": per_member_id,
                }
                continue

            def f(knob: float, _mode: str = mode) -> float:
                eval_calls["n"] += 1
                return _score_cluster(
                    targets,
                    float(knob),
                    fmt,
                    _mode,
                    render_levers=render_levers,
                )

            knob_opt, score_opt = golden_section_search(
                f, mode_alpha_lo, mode_alpha_hi, tol=tol, max_iter=max_iter
            )
            if score_id <= score_opt:
                # identity beats this mode's search
                knob_chosen = 0.0
                score_chosen = float(score_id)
            else:
                knob_chosen = float(knob_opt)
                score_chosen = float(score_opt)
            _, per_member = _score_cluster_per_member(
                targets,
                float(knob_chosen),
                fmt,
                mode,
                render_levers=render_levers,
            )
            per_mode_results[mode] = {
                "alpha": float(knob_chosen),
                "score": float(score_chosen),
                "evals": int(eval_calls["n"]),
                "requested_alpha_hi": float(alpha_hi),
                "effective_alpha_hi": float(mode_alpha_hi),
                "per_member_score": per_member,
            }
        winner = min(per_mode_results.items(), key=lambda kv: kv[1]["score"])
        winner_mode = winner[0]
        winner_data = winner[1]
        # If identity (no mode helps) still beats all modes, fall back to identity.
        if score_id <= float(winner_data["score"]):
            out[fmt] = {
                "alpha": 0.0,
                "score": float(score_id),
                "alpha_0_score": float(score_id),
                "requested_alpha_hi": float(alpha_hi),
                "effective_alpha_hi": float(winner_data["effective_alpha_hi"]),
                "identity_per_member_score": per_member_id,
                "per_member_score": per_member_id,
                "mode": "identity",
                "per_mode": per_mode_results,
            }
        else:
            out[fmt] = {
                "alpha": float(winner_data["alpha"]),
                "score": float(winner_data["score"]),
                "alpha_0_score": float(score_id),
                "requested_alpha_hi": float(alpha_hi),
                "effective_alpha_hi": float(winner_data["effective_alpha_hi"]),
                "identity_per_member_score": per_member_id,
                "per_member_score": winner_data["per_member_score"],
                "mode": winner_mode,
                "per_mode": per_mode_results,
            }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-calib-samples", type=int, default=8)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--max-act-rows", type=int, default=512)
    ap.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated formats to search per cluster",
    )
    ap.add_argument("--alpha-lo", type=float, default=0.0)
    ap.add_argument("--alpha-hi", type=float, default=1.0)
    ap.add_argument("--alpha-tol", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument(
        "--scale-modes",
        default=",".join(SCALE_MODES),
        help="Comma-separated scale-formula modes to evaluate per cluster. "
        "Available: smoothquant_max (default), awq_mean. With both, the "
        "winner per (cluster, format) is the mode with lower output MSE; "
        "per-mode results are preserved in the output JSON.",
    )
    ap.add_argument(
        "--render-levers",
        default="",
        help=(
            "Comma-separated downstream render passes to include while "
            "scoring candidates. Empty/default means RTN-only. Supported: "
            "gptq, scale_sweep, awq_round."
        ),
    )
    args = ap.parse_args(argv)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    modes = tuple(m.strip() for m in args.scale_modes.split(",") if m.strip())
    for m in modes:
        if m not in SCALE_MODES:
            raise ValueError(f"unknown scale mode {m!r}; allowed: {SCALE_MODES}")
    render_levers = _parse_render_levers(args.render_levers)
    print(f"[joint-search] formats={formats}", flush=True)
    print(f"[joint-search] scale modes={list(modes)}", flush=True)
    print(f"[joint-search] render levers={render_levers}", flush=True)
    if bool(render_levers.get("gptq", False)) and any(
        f.upper() in {"MXFP8", "MXFP8_E4M3"} for f in formats
    ):
        print(
            "[joint-search] note: export_native_compressed currently ignores "
            "gptq_enabled for MXFP8; MXFP8 rows reflect its non-GPTQ path.",
            flush=True,
        )

    print("[joint-search] loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.dataset:
        calib_ids = load_calibration(
            tokenizer, args.dataset, args.n_calib_samples, args.calib_seqlen
        )
    else:
        raise ValueError("--dataset is required for this isolation run")
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "cuda",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "requires `accelerate`" not in msg and "requires accelerate" not in msg:
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.to("cuda")
    model.eval()
    device = next(model.parameters()).device

    print("[joint-search] discovering eligible Linears ...", flush=True)
    qnames = _collect_eligible_qnames(model)
    print(f"[joint-search] {len(qnames)} eligible Linears", flush=True)

    print(
        f"[joint-search] capturing activations (n={args.n_calib_samples} "
        f"seqlen={args.calib_seqlen}) ...",
        flush=True,
    )
    t0 = time.time()
    activations = _capture_activations(
        model, qnames, calib_ids, max_rows=args.max_act_rows, device=device
    )
    print(
        f"[joint-search] captured {len(activations)} activation tensors "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    clusters = _build_clusters(qnames, activations, device)
    print(f"[joint-search] {len(clusters)} clusters", flush=True)

    results: list[dict] = []
    t0 = time.time()
    for idx, (cluster_key, targets) in enumerate(sorted(clusters.items()), start=1):
        per_fmt = _search_cluster(
            targets,
            formats,
            alpha_lo=args.alpha_lo,
            alpha_hi=args.alpha_hi,
            tol=args.alpha_tol,
            max_iter=args.max_iter,
            modes=modes,
            render_levers=render_levers,
        )
        results.append(
            {
                "cluster_key": cluster_key,
                "members": sorted(t.name for t in targets),
                "in_features": int(targets[0].weight.shape[1]),
                "per_format": per_fmt,
            }
        )
        if idx % 5 == 0 or idx == len(clusters):
            elapsed = time.time() - t0
            print(
                f"[joint-search] {idx}/{len(clusters)} clusters "
                f"({elapsed:.1f}s, {elapsed / max(idx, 1):.2f}s/cluster)",
                flush=True,
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "clusters": results,
                "formats": formats,
                "modes": list(modes),
                "render_levers": render_levers,
            },
            indent=2,
        )
    )
    print(f"[joint-search] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
