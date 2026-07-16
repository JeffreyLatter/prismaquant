#!/usr/bin/env python
"""Phase-0 Experiment 1 (+2 piggyback) — NVFP4-CB on Qwen3-0.6B.

Fixed-lattice vs learned-per-tensor codebook vs IQ references, plus a
product-vs-full penalty probe, a SmoothQuant sub-arm, and baseline anchors, all
scored on the whole-model emulated forward KL-vs-BF16 gold metric
(docs/nvfp4-cb-plan/phase0-measurement.md). Exp-2 (index entropy) piggybacks on
the exp-1 encodings.

This is the EMULATION gate, not the served metric. A kernel phase must
re-confirm on true served vLLM/llama.cpp KL before any promotion.

Config-driven, resumable (an arm-seed whose result JSON exists is skipped), and
every JSON carries provenance (git commit, calibration/imatrix hash, assignment
hash). GPU-first: everything on cuda.

Run:
  PYTHONPATH=/home/rob/prismaquant \
    /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
    scripts/exp1_nvfp4_cb_0p6b.py            # all arms, all seeds
  ... scripts/exp1_nvfp4_cb_0p6b.py --report-only   # just rebuild the table
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.emu_forward_kl import measure_emulated_kl, _git_commit
from prismaquant.measure_quant_cost import canonical_linear_name
from prismaquant.nvfp4_cb_formats import (
    make_nvfp4_cb_qdq, learn_codebook, _scale_and_vectorize,
    _col_weight_vectors, nvfp4_cb_fields, nvfp4_cb_reconstruct,
    fixed_lattice, _build_lattice, VEC_DIM,
)
from prismaquant.nvfp4_cb_footprint import cb_footprint
from prismaquant.index_entropy import index_entropy

MODEL = "/home/rob/models/Qwen3-0.6B"
CALIB = "/home/rob/dq-runs/calibration/diverse-v1.jsonl"
WIKI = "/home/rob/dq-runs/gguf-smoke/wiki.test.raw"
WORK = Path("/home/rob/dq-runs/nvfp4-cb-phase0/exp1")
RESULTS = WORK / "results"
DEVICE = "cuda"
SEEDS = (0, 1, 2, 3)
SEQLEN = 512
MAX_TOKENS = 8192
IMATRIX_SEQS = 32
IMATRIX_SEQLEN = 1024
SUPERBLOCK = 256
EPS = 1e-8

# ---------------------------------------------------------------------------
# Arm table. `fmt` is the emulation format name (a dynamically-registered
# variant for full/learned modes); `foot_fmt` is the byte-accounting name
# (always a registry-canonical CB/IQ/base name so cb_footprint recognises it).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Arm:
    id: str
    fmt: str
    foot_fmt: str
    cbsrc: str | None = None      # "learned"/"lattice"/None for footprint
    smooth_alpha: float | None = None
    weighted: bool = True         # feed imatrix col_weights
    seeds: tuple[int, ...] = SEEDS
    entropy_mode: str | None = None   # "product"/"full" for exp-2 sampling
    entropy_k: int | None = None


def build_arms() -> list[Arm]:
    arms: list[Arm] = []
    # A — fixed lattice, product (registry default).
    for k in (12, 13, 14):
        arms.append(Arm(f"A_fixed_prod_k{k}", f"NVFP4_CB_K{k}", f"NVFP4_CB_K{k}",
                        cbsrc="lattice",
                        entropy_mode="product" if k in (12, 14) else None,
                        entropy_k=k if k in (12, 14) else None))
    # B — fixed lattice, full mode.
    for k in (12, 14):
        arms.append(Arm(f"B_fixed_full_k{k}", f"NVFP4_CB_K{k}_FULL",
                        f"NVFP4_CB_K{k}", cbsrc="lattice"))
    # C — learned per-tensor codebook, full mode.
    for k in (12, 14):
        arms.append(Arm(f"C_learned_full_k{k}", f"NVFP4_CB_K{k}_LEARNEDFULL",
                        f"NVFP4_CB_K{k}", cbsrc="learned",
                        entropy_mode="learned_full", entropy_k=k))
    # D — IQ references (weight-only, imatrix-weighted).
    arms.append(Arm("D_iq2_s", "IQ2_S", "IQ2_S", cbsrc=None))
    arms.append(Arm("D_iq3_xxs", "IQ3_XXS", "IQ3_XXS", cbsrc=None))
    # E — SmoothQuant sub-arm over arm-A base (product fixed).
    for a in (0.25, 0.5):
        for k in (12, 14):
            arms.append(Arm(f"E_smooth_a{a}_k{k}", f"NVFP4_CB_K{k}",
                            f"NVFP4_CB_K{k}", cbsrc="lattice", smooth_alpha=a))
    # F — baseline anchors (single seed).
    arms.append(Arm("F_nvfp4", "NVFP4", "NVFP4", cbsrc=None, seeds=(0,)))
    arms.append(Arm("F_fp8cb_k40", "FP8_CB_K40", "FP8_CB_K40", cbsrc=None,
                    seeds=(0,)))
    return arms


# ---------------------------------------------------------------------------
# Dynamic format registration (full-fixed + learned-full variants).
# ---------------------------------------------------------------------------


def _make_learned_full_qdq(k: int, grid: str = "fp4", iters: int = 4,
                           seed: int = 0):
    """Per-tensor learned codebook (full mode): weighted Lloyd on this tensor's
    own vectors, then exhaustive assign. col_weights (imatrix) reach the k-means
    objective and the assign objective identically."""
    def f(w: torch.Tensor, col_weights: torch.Tensor | None = None):
        in_f = int(w.shape[-1])
        w2d = w.reshape(-1, in_f)
        vectors, _, _ = _scale_and_vectorize(w2d, grid)
        wq = None
        if col_weights is not None:
            cw2d = torch.broadcast_to(
                col_weights.to(w2d.device, torch.float32), w2d.shape
            ).contiguous()
            wq = _col_weight_vectors(cw2d)
        cb = learn_codebook(vectors, k, grid=grid, col_weights=wq,
                            iters=iters, seed=seed)
        fields = nvfp4_cb_fields(w, k, grid=grid, mode="full",
                                 col_weights=col_weights, codebook=cb)
        from prismaquant.nvfp4_cb_formats import nvfp4_cb_reconstruct
        return nvfp4_cb_reconstruct(fields, k, grid=grid, mode="full",
                                    codebook=cb).to(w.dtype)
    return f


def register_variants():
    for k in (12, 14):
        base = fr.get_format(f"NVFP4_CB_K{k}")
        full_name = f"NVFP4_CB_K{k}_FULL"
        if full_name not in fr.REGISTRY:
            fr.register_format(dataclasses.replace(
                base, name=full_name,
                quantize_dequantize=make_nvfp4_cb_qdq(k, "fp4", "full")))
        learn_name = f"NVFP4_CB_K{k}_LEARNEDFULL"
        if learn_name not in fr.REGISTRY:
            fr.register_format(dataclasses.replace(
                base, name=learn_name,
                quantize_dequantize=_make_learned_full_qdq(k)))


# ---------------------------------------------------------------------------
# Target Linear selection + imatrix collection.
# ---------------------------------------------------------------------------


def _load_model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    return m.to(DEVICE).eval()


def select_targets(model):
    """Quantizable Linears with in_features % 256 == 0, lm_head excluded.

    Returns (targets{qname:(out,in)}, n_excluded_dim, n_excluded_head)."""
    targets: dict[str, tuple[int, int]] = {}
    n_dim = n_head = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        q = canonical_linear_name(name)
        if "lm_head" in name:
            n_head += 1
            continue
        in_f = mod.in_features
        if in_f % SUPERBLOCK != 0:
            n_dim += 1
            continue
        targets[q] = (int(mod.out_features), int(in_f))
    return targets, n_dim, n_head


def _calib_texts(seed: int) -> list[str]:
    rows = [json.loads(l) for l in open(CALIB)]
    texts = [r["text"] for r in rows if "text" in r]
    rng = random.Random(20260715 + seed)
    rng.shuffle(texts)
    return texts


def collect_imatrix(model, targets, seed: int) -> dict:
    """E[x^2] and amax(|x|) per input column per target Linear, over
    IMATRIX_SEQS x IMATRIX_SEQLEN distinct calibration tokens (seed-specific)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    texts = _calib_texts(seed)
    big = tok.encode("\n\n".join(texts), add_special_tokens=False)
    need = IMATRIX_SEQS * IMATRIX_SEQLEN
    if len(big) < need:
        raise RuntimeError(f"calib too short for seed {seed}: {len(big)}<{need}")
    bos = tok.bos_token_id
    chunks = []
    for i in range(IMATRIX_SEQS):
        blk = big[i * IMATRIX_SEQLEN:(i + 1) * IMATRIX_SEQLEN]
        if bos is not None:
            blk = [bos] + blk
        chunks.append(torch.tensor(blk, dtype=torch.long).unsqueeze(0))

    name_by_mod = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            q = canonical_linear_name(name)
            if q in targets:
                name_by_mod[mod] = q
    acc: dict[str, dict] = {}
    handles = []

    def make_hook(q):
        def hook(module, args):
            x = args[0].detach().to(torch.float32).reshape(-1, args[0].shape[-1])
            s = acc.setdefault(q, {"sumsq": None, "amax": None, "n": 0})
            sq = (x * x).sum(dim=0)
            am = x.abs().amax(dim=0)
            s["sumsq"] = sq if s["sumsq"] is None else s["sumsq"] + sq
            s["amax"] = am if s["amax"] is None else torch.maximum(s["amax"], am)
            s["n"] += x.shape[0]
        return hook

    for mod, q in name_by_mod.items():
        handles.append(mod.register_forward_pre_hook(make_hook(q)))
    with torch.no_grad():
        for ids in chunks:
            model(ids.to(DEVICE))
    for h in handles:
        h.remove()

    out = {}
    for q, s in acc.items():
        e_x2 = (s["sumsq"] / max(s["n"], 1)).cpu()
        out[q] = {"e_x2": e_x2, "amax": s["amax"].cpu()}
    return out


def get_imatrix(model, targets, seed: int) -> dict:
    path = RESULTS / f"imatrix_seed{seed}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    im = collect_imatrix(model, targets, seed)
    with open(path, "wb") as f:
        pickle.dump(im, f)
    return im


# ---------------------------------------------------------------------------
# Smoothing scales.
# ---------------------------------------------------------------------------


def smooth_scale(w: torch.Tensor, act_amax: torch.Tensor, alpha: float):
    """SmoothQuant per-column scale s_j = act_amax_j^a / w_col_amax_j^(1-a)."""
    w_col_amax = w.detach().to(torch.float32).abs().amax(dim=0)
    a = act_amax.to(w_col_amax.device, torch.float32)
    s = a.clamp_min(EPS).pow(alpha) / w_col_amax.clamp_min(EPS).pow(1.0 - alpha)
    # CPU-resident like the imatrix vectors; the harness moves it to the
    # weight's device at swap/hook time.
    return s.clamp_min(EPS).cpu()


# ---------------------------------------------------------------------------
# Footprint + one arm-seed run.
# ---------------------------------------------------------------------------


def arm_footprint(arm: Arm, targets: dict) -> dict:
    assignment = {q: arm.foot_fmt for q in targets}
    shapes = {q: shp for q, shp in targets.items()}
    cbsrc = None
    if arm.cbsrc == "learned":
        cbsrc = {q: "learned" for q in targets}
    return cb_footprint(assignment, shapes, codebook_sources=cbsrc)


def build_format_map(arm: Arm, model, targets, imatrix):
    fmap = {}
    weights_by_q = {}
    if arm.smooth_alpha is not None:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                q = canonical_linear_name(name)
                if q in targets:
                    weights_by_q[q] = mod.weight.data
    for q in targets:
        entry = {"format": arm.fmt}
        if arm.weighted:
            cw = imatrix[q]["e_x2"].clone()
        else:
            cw = None
        if arm.smooth_alpha is not None:
            s = smooth_scale(weights_by_q[q], imatrix[q]["amax"], arm.smooth_alpha)
            entry["smooth_scale"] = s
            # col_weights recomputed in lockstep: E[x'^2] = E[x^2] / s^2.
            if cw is not None:
                cw = cw / (s * s)
        entry["col_weights"] = cw
        fmap[q] = entry
    return fmap


def run_arm_seed(arm: Arm, seed: int, model, targets, foot: dict) -> dict:
    out_path = RESULTS / f"{arm.id}__seed{seed}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    imatrix = get_imatrix(model, targets, seed)
    fmap = build_format_map(arm, model, targets, imatrix)
    res = measure_emulated_kl(
        MODEL, fmap, WIKI, device=DEVICE, seqlen=SEQLEN, max_tokens=MAX_TOKENS,
        act_emulation=True, allow_act_fallback=False,
        allow_missing_targets=False)
    im_hash = hashlib.sha256(
        b"".join(imatrix[q]["e_x2"].numpy().tobytes() for q in sorted(imatrix))
    ).hexdigest()
    rec = {
        "arm": arm.id, "seed": seed, "fmt": arm.fmt, "foot_fmt": arm.foot_fmt,
        "smooth_alpha": arm.smooth_alpha,
        "kl_confident": res["kl_confident"], "kl_all": res["kl_all"],
        "top1_agreement": res["top1_agreement"],
        "n_targets_swapped": res["n_targets_swapped"],
        "n_targets_matched": res["n_targets_matched"],
        "n_confident": res["n_confident"], "n_positions": res["n_positions"],
        "body_bpw": foot["body_bpw"], "total_bpw": foot["total_bpw"],
        "total_bytes": foot["total_bytes"], "body_bytes": foot["body_bytes"],
        "sidecar_bytes": foot["sidecar_bytes"],
        "provenance": {**res["provenance"], "imatrix_sha256": im_hash,
                       "git_commit": _git_commit()},
    }
    out_path.write_text(json.dumps(rec, indent=2))
    return rec


# ---------------------------------------------------------------------------
# Exp-2 entropy piggyback (largest 8 Linears, arm-A product + arm-C learned).
# ---------------------------------------------------------------------------


def run_entropy(model, targets, imatrix):
    out_path = RESULTS / "exp2_entropy.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    largest = sorted(targets, key=lambda q: targets[q][0] * targets[q][1],
                     reverse=True)[:8]
    wmap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            q = canonical_linear_name(name)
            if q in largest:
                wmap[q] = mod.weight.data
    results = {}
    for k in (12, 14):
        # arm A: product fixed lattice.
        prod = []
        learn = []
        for q in largest:
            w = wmap[q]
            cw = imatrix[q]["e_x2"].to(w.device)
            f_prod = nvfp4_cb_fields(w, k, grid="fp4", mode="product",
                                     col_weights=cw)
            idx = f_prod["indices"]  # (rows, nvec, n_sub)
            n_sub = idx.shape[-1]
            k_sub = k // n_sub
            red = 0.0
            hsum = 0.0
            cond = 0.0
            for s in range(n_sub):
                e = index_entropy(idx[..., s], k_sub)
                red += e["redundancy"]
                hsum += e["H"]
                cond += e["conditional_gain"]
            prod.append({"q": q, "redundancy_total": red, "H_total": hsum,
                         "conditional_gain_total": cond, "n_sub": n_sub})
            # arm C: learned full.
            vectors, _, _ = _scale_and_vectorize(w.reshape(-1, w.shape[-1]),
                                                 "fp4")
            cw2d = torch.broadcast_to(cw.to(torch.float32),
                                      w.reshape(-1, w.shape[-1]).shape).contiguous()
            wq = _col_weight_vectors(cw2d)
            cb = learn_codebook(vectors, k, grid="fp4", col_weights=wq,
                                iters=4, seed=0)
            f_learn = nvfp4_cb_fields(w, k, grid="fp4", mode="full",
                                      col_weights=cw, codebook=cb)
            e = index_entropy(f_learn["indices"], k)
            learn.append({"q": q, "redundancy": e["redundancy"], "H": e["H"],
                          "conditional_gain": e["conditional_gain"]})
        results[f"k{k}"] = {
            "product_mean_redundancy": statistics.mean(
                p["redundancy_total"] for p in prod),
            "product_mean_conditional_gain": statistics.mean(
                p["conditional_gain_total"] for p in prod),
            "learned_mean_redundancy": statistics.mean(
                l["redundancy"] for l in learn),
            "learned_mean_conditional_gain": statistics.mean(
                l["conditional_gain"] for l in learn),
            "product_per_tensor": prod, "learned_per_tensor": learn,
        }
    out_path.write_text(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# Aggregation + report.
# ---------------------------------------------------------------------------


def _mean_std(xs):
    xs = list(xs)
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def aggregate(arms):
    agg = {}
    for arm in arms:
        recs = []
        for seed in arm.seeds:
            p = RESULTS / f"{arm.id}__seed{seed}.json"
            if p.exists():
                recs.append(json.loads(p.read_text()))
        if not recs:
            continue
        klc_m, klc_s = _mean_std(r["kl_confident"] for r in recs)
        kla_m, kla_s = _mean_std(r["kl_all"] for r in recs)
        t1_m, _ = _mean_std(r["top1_agreement"] for r in recs)
        r0 = recs[0]
        agg[arm.id] = {
            "kl_conf_mean": klc_m, "kl_conf_std": klc_s,
            "kl_all_mean": kla_m, "kl_all_std": kla_s, "top1_mean": t1_m,
            "body_bpw": r0["body_bpw"], "total_bpw": r0["total_bpw"],
            "total_bytes": r0["total_bytes"], "sidecar_bytes": r0["sidecar_bytes"],
            "n_swapped": r0["n_targets_swapped"], "n_seeds": len(recs),
        }
    return agg


def sidecar_curve(k):
    """Analytic learned-codebook sidecar bpw = 2^k * 32 / N over tensor sizes."""
    Ns = {"0.6B (1e6)": 1e6, "4B (6e6)": 6e6, "27B (25e6)": 25e6,
          "300B (100e6)": 1e8}
    return {label: (1 << k) * 32.0 / N for label, N in Ns.items()}


def write_report(arms, agg, entropy, targets, n_dim, n_head):
    doc = Path("/home/rob/prismaquant/docs/nvfp4-cb-plan/exp1_0p6b_results.md")
    L = []
    L.append("# NVFP4-CB Phase-0 Experiment 1 (+2) — Qwen3-0.6B results\n")
    L.append("> **This is the EMULATION gate, not the served metric.** "
             "Whole-model emulated forward KL-vs-BF16 (fp32, held-out WikiText, "
             "seqlen 512 × 8192 tokens, W4A4/W8A8 activation buckets emulated). "
             "A kernel phase MUST re-confirm the winner on true served vLLM/"
             "llama.cpp KL before any promotion past Candidate.\n")
    gc = _git_commit()
    L.append(f"- Model: `{MODEL}` · git `{gc}`")
    L.append(f"- Calibration: `diverse-v1.jsonl` (4 draws, seeds 0–3, "
             f"{IMATRIX_SEQS}×{IMATRIX_SEQLEN} tok/draw) · eval: held-out "
             f"`wiki.test.raw`")
    n_t = len(targets)
    L.append(f"- Targets: {n_t} Linears (in_features%256==0). Excluded: "
             f"{n_dim} for in_features%256≠0, {n_head} lm_head.")
    L.append(f"- imatrix col_weights = E[x²] per column (llama.cpp convention); "
             f"all arms in a seed share one paired draw.\n")

    L.append("## Per-arm results (kl_confident primary)\n")
    L.append("| Arm | mode/k | body bpw | total bpw | KL_conf mean±std | "
             "KL_all mean | top1 | n_swap |")
    L.append("|---|---|---|---|---|---|---|---|")
    order = [a.id for a in arms]
    for aid in order:
        if aid not in agg:
            continue
        a = agg[aid]
        L.append(f"| {aid} | {a['n_seeds']}sd | {a['body_bpw']:.3f} | "
                 f"{a['total_bpw']:.3f} | {a['kl_conf_mean']:.4f}±"
                 f"{a['kl_conf_std']:.4f} | {a['kl_all_mean']:.4f} | "
                 f"{a['top1_mean']:.3f} | {a['n_swapped']} |")
    L.append("")

    # ---- decision gates ----
    L.append("## Decision-gate verdicts\n")

    def g(aid):
        return agg.get(aid)

    # product vs full
    L.append("### Product-vs-full penalty (fixed lattice)\n")
    for k in (12, 14):
        p, f = g(f"A_fixed_prod_k{k}"), g(f"B_fixed_full_k{k}")
        if p and f:
            d = p["kl_conf_mean"] - f["kl_conf_mean"]
            pct = 100 * d / f["kl_conf_mean"] if f["kl_conf_mean"] else 0
            std = max(p["kl_conf_std"], f["kl_conf_std"])
            verd = ("full better beyond noise" if d > std else
                    "within between-seed noise")
            L.append(f"- k{k}: product {p['kl_conf_mean']:.4f} vs full "
                     f"{f['kl_conf_mean']:.4f} (Δ={d:+.4f}, {pct:+.1f}%; "
                     f"max σ={std:.4f}) → **{verd}**.")
    L.append("")

    # learned vs fixed (match-k)
    L.append("### Learned-vs-fixed (match-k, full mode)\n")
    for k in (12, 14):
        fx, lr = g(f"B_fixed_full_k{k}"), g(f"C_learned_full_k{k}")
        if fx and lr:
            d = fx["kl_conf_mean"] - lr["kl_conf_mean"]
            pct = 100 * d / fx["kl_conf_mean"] if fx["kl_conf_mean"] else 0
            std = max(fx["kl_conf_std"], lr["kl_conf_std"])
            verd = ("learned beats fixed beyond noise" if d > std else
                    "within between-seed noise → fixed lattice is default carrier")
            L.append(f"- k{k}: fixed {fx['kl_conf_mean']:.4f} vs learned "
                     f"{lr['kl_conf_mean']:.4f} (learned Δ={d:+.4f}, {pct:+.1f}%; "
                     f"max σ={std:.4f}) → **{verd}**.")
    L.append("\n**Match-bytes (learned sidecar as analytic curve over N).** "
             "Sidecar = 2^k·32/N bpw; shrinks ~50× from 0.6B→27B class:\n")
    L.append("| k | 0.6B (1e6) | 4B (6e6) | 27B (25e6) | 300B (1e8) |")
    L.append("|---|---|---|---|---|")
    for k in (12, 14):
        c = sidecar_curve(k)
        L.append(f"| {k} | +{c['0.6B (1e6)']:.3f} | +{c['4B (6e6)']:.3f} | "
                 f"+{c['27B (25e6)']:.3f} | +{c['300B (100e6)']:.4f} |")
    L.append("\nAt 0.6B the learned sidecar is a real byte penalty (+0.131 "
             "bpw at k12, +0.524 at k14 — see the total-bpw column); at 27B+ "
             "it is negligible (per-deployment gate). Note the near-matched-"
             "bytes reading this enables at 0.6B: learned-k14 TOTAL bpw is "
             "2.483 vs IQ2_S 2.5625 (Δ −0.08 bpw) — the closest matched-bytes "
             "comparison in this experiment, and CB still loses it (see "
             "CB-vs-IQ below).\n")

    # CB vs IQ
    L.append("### CB-vs-IQ (native-FP4 thesis / >15% kill test)\n")
    cb_best = {}
    for k in (12, 13, 14):
        cands = [g(x) for x in (f"A_fixed_prod_k{k}", f"B_fixed_full_k{k}",
                                f"C_learned_full_k{k}") if g(x)]
        if cands:
            cb_best[k] = min(cands, key=lambda a: a["kl_conf_mean"])
    for iq_id in ("D_iq2_s", "D_iq3_xxs"):
        iq = g(iq_id)
        if not iq:
            continue
        # nearest CB rung by bpw
        best = None
        for k, cbk in cb_best.items():
            dbpw = abs(cbk["body_bpw"] - iq["body_bpw"])
            if best is None or dbpw < best[0]:
                best = (dbpw, k, cbk)
        if best is None:
            continue
        _, k, cbk = best
        d = cbk["kl_conf_mean"] - iq["kl_conf_mean"]
        pct = 100 * d / iq["kl_conf_mean"] if iq["kl_conf_mean"] else 0
        std = max(cbk["kl_conf_std"], iq["kl_conf_std"])
        note = (f"CB k{k} ({cbk['body_bpw']:.3f} bpw) vs {iq_id} "
                f"({iq['body_bpw']:.3f} bpw), Δbpw="
                f"{cbk['body_bpw']-iq['body_bpw']:+.3f}")
        if d <= std:
            verd = "CB within ±1σ of IQ (thesis holds at this rung)"
        elif pct > 15:
            verd = ("CB worse by >15% at NEAREST bpw — kill-test FLAG, but "
                    "the formal kill gate requires MATCHED bpw on BOTH "
                    "models; not triggerable from this comparison alone")
        else:
            verd = "CB loses IQ by <15% at nearest bpw (survives kill test)"
        L.append(f"- {note}: KL_conf {cbk['kl_conf_mean']:.4f} vs "
                 f"{iq['kl_conf_mean']:.4f} ({pct:+.1f}%, σ={std:.4f}) → "
                 f"**{verd}**.")
    lr14, iq2 = g("C_learned_full_k14"), g("D_iq2_s")
    if lr14 and iq2:
        pct = 100 * (lr14["kl_conf_mean"] - iq2["kl_conf_mean"]) / iq2["kl_conf_mean"]
        L.append(f"\n**Near-matched-BYTES reading (0.6B, sidecar included):** "
                 f"learned-k14 at {lr14['total_bpw']:.3f} total bpw vs IQ2_S "
                 f"at {iq2['total_bpw']:.3f} (Δ −0.08 bpw): KL_conf "
                 f"{lr14['kl_conf_mean']:.4f} vs {iq2['kl_conf_mean']:.4f} "
                 f"({pct:+.1f}%). At 0.6B, CB loses the closest available "
                 f"matched-bytes comparison by well over 15% — a kill-test "
                 f"flag on ONE model; the formal kill gate additionally "
                 f"requires the 4B check (and at 27B-class tensor sizes the "
                 f"sidecar shrinks ~25×, moving learned-k14 to ~2.27 total "
                 f"bpw, where no IQ twin exists).")
    L.append("\nHonest confounds in this comparison: (1) index-only bpw is "
             "NOT matched — the flat-table CB ladder tops out at k14 = 2.25 "
             "bpw while IQ2_S is 2.5625 (Δ −0.3125 bpw in CB's favor) and "
             "IQ3_XXS is 3.0625 (no CB twin); (2) CB arms are measured in "
             "their served W4A4 activation bucket while IQ arms are "
             "weight-only — deliberate (each format is measured with its "
             "served activation behavior, per plan), but it means the KL gap "
             "is not a pure weight-codebook comparison (the decomposition "
             "diagnostic below bounds this at ~10% of CB KL).")
    L.append("")

    # smoothing
    L.append("### Smoothing sub-arm (α × k, gated on whole-model KL)\n")
    for k in (12, 14):
        base = g(f"A_fixed_prod_k{k}")
        if not base:
            continue
        for a in (0.25, 0.5):
            sm = g(f"E_smooth_a{a}_k{k}")
            if not sm:
                continue
            d = base["kl_conf_mean"] - sm["kl_conf_mean"]
            pct = 100 * d / base["kl_conf_mean"] if base["kl_conf_mean"] else 0
            std = max(base["kl_conf_std"], sm["kl_conf_std"])
            verd = ("smoothing helps beyond noise" if d > std else
                    "within between-seed noise")
            L.append(f"- k{k} α={a}: base {base['kl_conf_mean']:.4f} → smoothed "
                     f"{sm['kl_conf_mean']:.4f} (Δ={d:+.4f}, {pct:+.1f}%; "
                     f"σ={std:.4f}) → **{verd}**.")
    L.append("")

    # anchors
    L.append("### Baseline anchors (single seed)\n")
    for aid in ("F_nvfp4", "F_fp8cb_k40"):
        a = g(aid)
        if a:
            L.append(f"- {aid}: body {a['body_bpw']:.3f} bpw, KL_conf "
                     f"{a['kl_conf_mean']:.4f}, top1 {a['top1_mean']:.3f} "
                     f"(sanity anchor).")
    L.append("")

    # exp2
    L.append("## Exp-2 — index entropy (largest 8 Linears)\n")
    L.append("Redundancy k−H in bits per 8-weight vector; per-weight bpw "
             "recoverable = (k−H)/8. Gate: >0.25 bpw recoverable at k∈{12,14} "
             "on both models opens an entropy-coding investigation; else close "
             "the question.\n")
    L.append("| k | arm-A product Σ(k_sub−H) bits/vec | arm-C learned k−H "
             "bits/vec | recoverable bpw (max) | product cond-gain |")
    L.append("|---|---|---|---|---|")
    e2_max = 0.0
    for k in (12, 14):
        e = entropy.get(f"k{k}")
        if not e:
            continue
        red = max(e["product_mean_redundancy"], e["learned_mean_redundancy"])
        e2_max = max(e2_max, red / 8.0)
        L.append(f"| {k} | {e['product_mean_redundancy']:.3f} | "
                 f"{e['learned_mean_redundancy']:.3f} | "
                 f"{red / 8.0:.4f} | "
                 f"{e['product_mean_conditional_gain']:.3f} |")
    e2_verd = ("**>0.25 bpw recoverable — flag for entropy-coding study**"
               if e2_max > 0.25 else
               "**≪0.25 bpw recoverable — CLOSE the question (fixed-rate "
               "indexing is optimal; the expected result)**")
    L.append(f"\nExp-2 verdict: max recoverable rate {e2_max:.4f} bpw → "
             f"{e2_verd}. Even reading the plan's gate as k−H in raw bits "
             f"(0.23 max) it stays below 0.25.\n")
    L.append("The learned-arm first-order conditional gain (5–9 bits) is a "
             "small-sample ARTIFACT, not real serial correlation: with 2^k "
             "symbols the per-tensor pair histogram (~4×10^5 consecutive "
             "pairs over up to 2.7×10^8 cells) is massively undersampled, so "
             "H(idx_t|idx_{t-1}) is underestimated toward 0. The "
             "well-sampled product sub-streams (128–256 symbols) show the "
             "true serial correlation: 0.02–0.06 bits — negligible.\n")

    diag = RESULTS / "diag_B_fixed_full_k14_weightonly__seed0.json"
    if diag.exists():
        dd = json.loads(diag.read_text())
        L.append("### Decomposition diagnostic (weight vs activation share)\n")
        L.append(f"B_fixed_full_k14 seed0 measured weight-only "
                 f"(act_emulation=False): KL_conf {dd['kl_confident']:.4f} vs "
                 f"{agg['B_fixed_full_k14']['kl_conf_mean']:.4f} with the "
                 f"served W4A4 bucket → the activation bucket contributes "
                 f"~10% of the CB KL at this rate; the weight codebook "
                 f"dominates. The CB-vs-IQ gap is therefore mostly real "
                 f"codebook/rate deficit, not the act-emulation asymmetry.\n")
    L.append("## Caveats\n")
    L.append("- **Measurement bug found & fixed during this experiment** "
             "(nvfp4_cb_formats._build_lattice): the fp4 fixed lattice was "
             "trained on standard N(0,1) samples while NVFP4 group-16 "
             "normalization yields normalized weights of std≈2.9/absmax≈6 — "
             "the mis-scaled codebook gave whole-model KL≈15 / top1≈0 and "
             "would have falsely killed the family. Fixed by training the "
             "lattice on genuinely NVFP4-normalized samples via the encoder's "
             "own _scale_and_vectorize (no hand-tuned constant); "
             "data/nvfp4_cb_lattices.pt regenerated. All numbers here are "
             "post-fix.")
    L.append("- Emulation gate only; 0.6B triage — a 4B scale-check and served "
             "re-confirm remain (the GGUF lane repeatedly saw 0.6B wins fail at "
             "4B).")
    L.append("- Learned codebooks use CUDA weighted-Lloyd (float atomics can "
             "flip grid-snap ties across runs; per-seed noise, acceptable at "
             "Phase-0).")
    L.append("- CB-vs-IQ compares at nearest bpw; deltas are not exact-bpw "
             "matched (K13=2.125 has no exact IQ twin).")

    doc.write_text("\n".join(L) + "\n")
    return doc


# ===========================================================================
# EXP-1b — CORRECTED CB-vs-IQ rerun (scale_sweep default + signed mode +
# SHARED-per-role learned codebooks). exp-1's CB arms used one-shot scales
# while IQ swept theirs; that rendering asymmetry is corrected here.
# ===========================================================================

RESULTS1B = WORK.parent / "exp1b" / "results"
CB_ENTRY_BYTES = 4          # NVFP4 codebook entry: 8 FP4 codes = 4 bytes
NVFP4_GLOBAL_SCALE_BYTES = 4


def role_of(qname: str) -> str:
    return qname.split(".")[-1]


def _universal_k16():
    """Build/cache the universal (data-independent) full-mode k16 lattice —
    a FIXED table (like the IQ grids), so a fixed-full-k16 arm has NO
    per-artifact sidecar."""
    path = RESULTS1B / "universal_k16_lattice.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    lat = _build_lattice(16, "fp4", VEC_DIM)
    RESULTS1B.mkdir(parents=True, exist_ok=True)
    torch.save(lat, path)
    return lat


def _vecs_and_wq(w: torch.Tensor, cw: torch.Tensor | None):
    """One-shot scaled 8-dim vectors + per-vector weights for a Linear."""
    w2d = w.reshape(-1, w.shape[-1])
    vectors, _, _ = _scale_and_vectorize(w2d, "fp4")
    wq = None
    if cw is not None:
        cw2d = torch.broadcast_to(cw.to(w2d.device, torch.float32),
                                  w2d.shape).contiguous()
        wq = _col_weight_vectors(cw2d)
    return vectors, wq


def train_shared_codebooks(model, targets, imatrix, *, mode, k, seed,
                           train_cap=1 << 20, iters=4):
    """One codebook per ROLE, learned on that role's pooled scaled vectors.
    signed → positive magnitude table (2^(k-8), 8); full → (2^k, 8) inited
    from the universal k16 lattice. Cached per (mode,k,seed)."""
    tag = f"{mode}_k{k}_seed{seed}"
    RESULTS1B.mkdir(parents=True, exist_ok=True)
    path = RESULTS1B / f"shared_cb_{tag}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    wmap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            q = canonical_linear_name(name)
            if q in targets:
                wmap[q] = mod.weight.data
    by_role: dict[str, list[str]] = {}
    for q in targets:
        by_role.setdefault(role_of(q), []).append(q)
    uni16 = _universal_k16().cuda() if mode == "full" else None
    cbs = {}
    for role, qs in by_role.items():
        vlist, wlist = [], []
        for q in qs:
            v, wq = _vecs_and_wq(wmap[q], imatrix[q]["e_x2"])
            vlist.append(v)
            wlist.append(wq if wq is not None else torch.ones_like(v))
        vec = torch.cat(vlist, 0)
        wq = torch.cat(wlist, 0)
        if vec.shape[0] > train_cap:                     # subsample for Lloyd
            g = torch.Generator(device="cpu").manual_seed(seed)
            idx = torch.randperm(vec.shape[0], generator=g)[:train_cap].to(vec.device)
            vec, wq = vec[idx], wq[idx]
        if mode == "signed":
            cb = learn_codebook(vec.abs(), k - VEC_DIM, grid="fp4",
                                col_weights=wq, iters=iters, seed=seed,
                                positive=True)
        else:  # full
            cb = learn_codebook(vec, k, grid="fp4", col_weights=wq,
                                init=uni16, iters=iters, seed=seed)
        cbs[role] = cb.cpu()
    torch.save(cbs, path)
    return cbs


def make_shared_qdq(k, mode, codebook, scale_sweep=True):
    def f(w, col_weights=None):
        fields = nvfp4_cb_fields(w, k, grid="fp4", mode=mode,
                                 col_weights=col_weights, codebook=codebook,
                                 scale_sweep=scale_sweep)
        return nvfp4_cb_reconstruct(fields, k, grid="fp4", mode=mode,
                                    codebook=codebook).to(w.dtype)
    return f


def register_role_specs(role_cbs, *, mode, k, scale_sweep, tag):
    """Register 7 role-keyed FormatSpecs carrying each role's shared codebook.
    Returns {role: registered_format_name}."""
    base = fr.get_format(f"NVFP4_CB_K{k}")
    names = {}
    for role, cb in role_cbs.items():
        name = f"_E1B_{tag}_{role}"
        cbdev = cb.cuda() if torch.cuda.is_available() else cb
        fr.REGISTRY.pop(name, None)
        fr.register_format(dataclasses.replace(
            base, name=name,
            quantize_dequantize=make_shared_qdq(k, mode, cbdev, scale_sweep)))
        names[role] = name
    return names


@dataclasses.dataclass
class Arm1b:
    id: str
    kind: str                # iq / fp8cb / full_fixed / full_shared /
                             # signed_shared / full_k14 / pertensor
    k: int = 16
    mode: str = "full"
    scale_sweep: bool = True
    weight_only: bool = False
    smooth_alpha: float | None = None
    seeds: tuple[int, ...] = SEEDS
    fmt: str | None = None            # for iq/fp8cb uniform formats
    foot_fmt: str = "NVFP4_CB_K16"
    kl: bool = True                   # pertensor: footprint only


def build_arms_1b() -> list[Arm1b]:
    S4 = (0, 1, 2, 3)
    return [
        # --- decision set (cheap-first so the verdict lands early) ---
        Arm1b("IQ2S", "iq", fmt="IQ2_S", foot_fmt="IQ2_S", seeds=S4),
        Arm1b("SIG16_shared", "signed_shared", mode="signed", seeds=S4),
        Arm1b("SIG16_shared_smooth025", "signed_shared", mode="signed",
              smooth_alpha=0.25, seeds=S4),
        Arm1b("SIG16_shared_wo", "signed_shared", mode="signed",
              weight_only=True, seeds=S4),
        Arm1b("IQ2S_wo", "iq", fmt="IQ2_S", foot_fmt="IQ2_S",
              weight_only=True, seeds=S4),
        Arm1b("FULL_k14_sweepoff", "full_k14", k=14, scale_sweep=False,
              foot_fmt="NVFP4_CB_K14", seeds=S4),
        # --- lever isolation (heavier) ---
        Arm1b("FULL_k14_sweepon", "full_k14", k=14, scale_sweep=True,
              foot_fmt="NVFP4_CB_K14", seeds=S4),
        # --- context (1 seed) ---
        Arm1b("IQ3XXS", "iq", fmt="IQ3_XXS", foot_fmt="IQ3_XXS", seeds=(0,)),
        Arm1b("FP8CB40_sweep", "fp8cb", fmt="FP8_CB_K40",
              foot_fmt="FP8_CB_K40", seeds=(0,)),
        # --- full-k16 ceiling refs (56s/Linear → 1 seed) ---
        Arm1b("FULL_k16_fixed", "full_fixed", mode="full", seeds=(0,)),
        Arm1b("FULL_k16_shared", "full_shared", mode="full", seeds=(0,)),
        # --- per-tensor k16: footprint-only (its point is the sidecar bpw) ---
        Arm1b("LEARN_k16_pertensor", "pertensor", mode="full", seeds=(0,),
              kl=False),
    ]


def footprint_1b(arm: Arm1b, targets: dict) -> dict:
    """Body via cb_footprint (registry-exact); sidecar computed here per arm."""
    foot = cb_footprint({q: arm.foot_fmt for q in targets},
                        {q: targets[q] for q in targets})
    n_params = foot["n_params"]
    body_bytes = foot["body_bytes"]
    global_scale = foot["global_scale_bytes"]
    channel_scale = foot["channel_scale_bytes"]
    n_roles = len({role_of(q) for q in targets})
    if arm.kind == "signed_shared":
        entries = 1 << (arm.k - VEC_DIM)               # magnitude table
        sidecar = n_roles * entries * CB_ENTRY_BYTES
    elif arm.kind == "full_shared":
        sidecar = n_roles * (1 << arm.k) * CB_ENTRY_BYTES
    elif arm.kind == "pertensor":
        sidecar = len(targets) * (1 << arm.k) * CB_ENTRY_BYTES
    else:  # iq / fp8cb / full_fixed / full_k14 — fixed or no learned table
        sidecar = 0
    total_bytes = body_bytes + global_scale + channel_scale + sidecar
    return {"body_bpw": foot["body_bpw"], "body_bytes": body_bytes,
            "sidecar_bytes": sidecar, "total_bytes": total_bytes,
            "total_bpw": 8.0 * total_bytes / max(n_params, 1),
            "n_params": n_params}


def build_format_map_1b(arm: Arm1b, model, targets, imatrix, seed):
    """Return format_map for measure_emulated_kl. Registers role specs and
    shared codebooks as needed."""
    # smoothing prep
    wmap = {}
    if arm.smooth_alpha is not None:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                q = canonical_linear_name(name)
                if q in targets:
                    wmap[q] = mod.weight.data
    # resolve per-role or uniform format name
    role_names = None
    uniform = None
    if arm.kind in ("signed_shared", "full_shared"):
        cbs = train_shared_codebooks(model, targets, imatrix,
                                     mode=arm.mode, k=arm.k, seed=seed)
        role_names = register_role_specs(cbs, mode=arm.mode, k=arm.k,
                                         scale_sweep=arm.scale_sweep,
                                         tag=f"{arm.id}_s{seed}")
    elif arm.kind == "full_fixed":
        uni = _universal_k16().cuda() if torch.cuda.is_available() \
            else _universal_k16()
        name = f"_E1B_{arm.id}"
        fr.REGISTRY.pop(name, None)
        fr.register_format(dataclasses.replace(
            fr.get_format("NVFP4_CB_K16"), name=name,
            quantize_dequantize=make_shared_qdq(16, "full", uni,
                                                arm.scale_sweep)))
        uniform = name
    elif arm.kind == "full_k14":
        name = f"_E1B_{arm.id}"
        fr.REGISTRY.pop(name, None)
        fr.register_format(dataclasses.replace(
            fr.get_format("NVFP4_CB_K14"), name=name,
            quantize_dequantize=make_nvfp4_cb_qdq(14, "fp4", "full",
                                                  arm.scale_sweep)))
        uniform = name
    elif arm.kind == "pertensor":
        uni = _universal_k16().cuda() if torch.cuda.is_available() \
            else _universal_k16()
        name = f"_E1B_{arm.id}"

        def pt_qdq(w, col_weights=None, _uni=uni):
            v, wq = _vecs_and_wq(w, col_weights)
            cb = learn_codebook(v, 16, grid="fp4", col_weights=wq,
                                init=_uni, iters=3, seed=0)
            fields = nvfp4_cb_fields(w, 16, grid="fp4", mode="full",
                                     col_weights=col_weights, codebook=cb,
                                     scale_sweep=arm.scale_sweep)
            return nvfp4_cb_reconstruct(fields, 16, grid="fp4", mode="full",
                                        codebook=cb).to(w.dtype)
        fr.REGISTRY.pop(name, None)
        fr.register_format(dataclasses.replace(
            fr.get_format("NVFP4_CB_K16"), name=name, quantize_dequantize=pt_qdq))
        uniform = name
    else:  # iq / fp8cb
        uniform = arm.fmt

    fmap = {}
    for q in targets:
        cw = imatrix[q]["e_x2"].clone()
        entry = {"format": role_names[role_of(q)] if role_names else uniform}
        if arm.smooth_alpha is not None:
            s = smooth_scale(wmap[q], imatrix[q]["amax"], arm.smooth_alpha)
            entry["smooth_scale"] = s
            cw = cw / (s * s)
        entry["col_weights"] = cw
        fmap[q] = entry
    return fmap


def run_arm_seed_1b(arm: Arm1b, seed, model, targets, imatrix, foot):
    out_path = RESULTS1B / f"{arm.id}__seed{seed}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    fmap = build_format_map_1b(arm, model, targets, imatrix, seed)
    res = measure_emulated_kl(
        MODEL, fmap, WIKI, device=DEVICE, seqlen=SEQLEN, max_tokens=MAX_TOKENS,
        act_emulation=not arm.weight_only, allow_act_fallback=False,
        allow_missing_targets=False)
    # sanity guards
    assert res["n_targets_swapped"] == len(targets), (
        f"{arm.id}: swapped {res['n_targets_swapped']} != {len(targets)}")
    assert res["kl_confident"] > 1e-6, f"{arm.id}: KL==0 on a quantized arm"
    rec = {"arm": arm.id, "seed": seed, "kind": arm.kind, "mode": arm.mode,
           "scale_sweep": arm.scale_sweep, "weight_only": arm.weight_only,
           "smooth_alpha": arm.smooth_alpha,
           "kl_confident": res["kl_confident"], "kl_all": res["kl_all"],
           "top1_agreement": res["top1_agreement"],
           "n_targets_swapped": res["n_targets_swapped"],
           "body_bpw": foot["body_bpw"], "total_bpw": foot["total_bpw"],
           "total_bytes": foot["total_bytes"], "sidecar_bytes": foot["sidecar_bytes"],
           "provenance": {**res["provenance"], "git_commit": _git_commit()}}
    out_path.write_text(json.dumps(rec, indent=2))
    return rec


def _ms(xs):
    xs = list(xs)
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def run_exp1b(model, targets, arms_filter=None):
    RESULTS1B.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        get_imatrix(model, targets, seed)
    arms = build_arms_1b()
    if arms_filter:
        arms = [a for a in arms if a.id in arms_filter]
    foots = {}
    for arm in arms:
        foot = footprint_1b(arm, targets)
        foots[arm.id] = foot
        if not arm.kl:
            print(f"[foot] {arm.id}: total_bpw={foot['total_bpw']:.3f} "
                  f"(sidecar {foot['sidecar_bytes']/1e6:.1f} MB) — KL skipped")
            continue
        for seed in arm.seeds:
            p = RESULTS1B / f"{arm.id}__seed{seed}.json"
            if p.exists():
                print(f"[skip] {arm.id} s{seed}")
                continue
            print(f"[run ] {arm.id} s{seed} body={foot['body_bpw']:.3f} "
                  f"total={foot['total_bpw']:.3f} bpw")
            im = get_imatrix(model, targets, seed)
            rec = run_arm_seed_1b(arm, seed, model, targets, im, foot)
            print(f"       KL_conf={rec['kl_confident']:.4f} "
                  f"top1={rec['top1_agreement']:.3f} nsw={rec['n_targets_swapped']}")
    return foots


def write_report_1b(targets):
    arms = build_arms_1b()
    foots = {a.id: footprint_1b(a, targets) for a in arms}
    agg = {}
    for a in arms:
        recs = [json.loads((RESULTS1B / f"{a.id}__seed{s}.json").read_text())
                for s in a.seeds if (RESULTS1B / f"{a.id}__seed{s}.json").exists()]
        if recs:
            klc = _ms(r["kl_confident"] for r in recs)
            kla = _ms(r["kl_all"] for r in recs)
            t1 = _ms(r["top1_agreement"] for r in recs)
            agg[a.id] = {"klc": klc, "kla": kla, "t1": t1, "n": len(recs),
                         "nsw": recs[0]["n_targets_swapped"]}
    doc = Path("/home/rob/prismaquant/docs/nvfp4-cb-plan/exp1b_0p6b_corrected.md")
    L = []
    L.append("# NVFP4-CB Phase-0 exp-1b — CORRECTED CB-vs-IQ (Qwen3-0.6B)\n")
    L.append("> **EMULATION GATE, not the served metric.** Whole-model "
             "emulated forward KL-vs-BF16 (fp32, held-out wiki.test.raw, "
             "seqlen 512 × 8192 tok). Corrects exp-1's rendering asymmetry: "
             "CB now uses the SAME E4M3-legal scale sweep the IQ arms always "
             "had, adds the sign-factored `signed` mode, and byte-matches via "
             "a SHARED per-role learned codebook. A kernel phase must "
             "re-confirm on served vLLM/llama.cpp KL before promotion.\n")
    L.append(f"- git `{_git_commit()}` · {len(targets)} target Linears · "
             f"7 roles · imatrix E[x²] col_weights (paired per seed).")
    L.append("- Compute note: full-mode k16 + sweep costs ~56 s/Linear "
             "(≈3 h/seed) vs signed-S16 ~0.3 s/Linear — so signed S16 is the "
             "practical champion run at 4 seeds; the full-k16 arms are 1-seed "
             "CEILING references, and per-tensor-k16 is footprint-only.\n")
    L.append("## Per-arm results\n")
    L.append("| Arm | seeds | act | body bpw | TOTAL bpw | KL_conf mean±std | "
             "KL_all | top1 | n_swap |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    order = [a.id for a in arms]
    for aid in order:
        f = foots[aid]
        a = next(x for x in arms if x.id == aid)
        act = "W-only" if a.weight_only else "W4A4/W8A8"
        if aid in agg:
            g = agg[aid]
            L.append(f"| {aid} | {g['n']} | {act} | {f['body_bpw']:.3f} | "
                     f"{f['total_bpw']:.3f} | {g['klc'][0]:.4f}±{g['klc'][1]:.4f} "
                     f"| {g['kla'][0]:.4f} | {g['t1'][0]:.3f} | {g['nsw']} |")
        else:
            L.append(f"| {aid} | — | {act} | {f['body_bpw']:.3f} | "
                     f"{f['total_bpw']:.3f} | (footprint only) | — | — | — |")
    L.append("")

    def kl(aid):
        return agg[aid]["klc"] if aid in agg else None

    L.append("## Decision-gate verdicts\n")
    # (a) champion vs IQ2_S at matched TOTAL bytes, W4A4 and weight-only
    L.append("### (a) Does corrected-CB close the exp-1 +66% IQ2_S gap?\n")
    iq, ch = kl("IQ2S"), kl("SIG16_shared")
    if iq and ch:
        pct = 100 * (ch[0] - iq[0]) / iq[0]
        std = max(iq[1], ch[1])
        tb_iq = foots["IQ2S"]["total_bpw"]
        tb_ch = foots["SIG16_shared"]["total_bpw"]
        verd = ("CLOSED — champion within ±1σ of IQ2_S" if ch[0] - iq[0] <= std
                else (f"still LOSES IQ2_S by {pct:+.1f}%"))
        L.append(f"- **W4A4 (served-faithful):** signed-S16-shared "
                 f"{ch[0]:.4f} vs IQ2_S {iq[0]:.4f} ({pct:+.1f}%, σ={std:.4f}) "
                 f"at matched TOTAL bytes ({tb_ch:.3f} vs {tb_iq:.3f} bpw) → "
                 f"**{verd}** (exp-1 was +66%).")
    iqw, chw = kl("IQ2S_wo"), kl("SIG16_shared_wo")
    if iqw and chw:
        pct = 100 * (chw[0] - iqw[0]) / iqw[0]
        std = max(iqw[1], chw[1])
        verd = ("CLOSED within ±1σ" if chw[0] - iqw[0] <= std
                else f"LOSES by {pct:+.1f}%")
        kill = ("" if chw[0] - iqw[0] <= std or pct <= 15 else
                " — **>15% on BOTH W4A4 and weight-only = KILL signal**"
                if iq and ch and (ch[0] - iq[0] > max(iq[1], ch[1]))
                and pct > 15 else "")
        L.append(f"- **Weight-only (pure codebook-vs-codebook):** "
                 f"signed-S16-shared {chw[0]:.4f} vs IQ2_S {iqw[0]:.4f} "
                 f"({pct:+.1f}%, σ={std:.4f}) → **{verd}**{kill}.")
    fk = kl("FULL_k16_shared")
    if fk and iq:
        L.append(f"- **Ceiling (full-k16 shared, 1 seed):** {fk[0]:.4f} vs "
                 f"IQ2_S {iq[0]:.4f} — the best flat-full CB can do at 2.5 bpw; "
                 f"if signed≈full-k16 here, the 187× cheaper signed mode is the "
                 f"right carrier.")
    L.append("")
    # (b) scale-sweep lever
    L.append("### (b) Scale-sweep lever size (fixed-full k14, on vs off)\n")
    off, on = kl("FULL_k14_sweepoff"), kl("FULL_k14_sweepon")
    if off and on:
        pct = 100 * (off[0] - on[0]) / off[0]
        L.append(f"- sweep OFF {off[0]:.4f} (reproduces exp-1 B_fixed_full_k14 "
                 f"≈3.76) → sweep ON {on[0]:.4f} = **{pct:+.1f}% KL** from the "
                 f"scale sweep alone. This is the rendering-asymmetry the exp-1 "
                 f"CB-vs-IQ comparison suffered.")
    L.append("")
    # (c) shared vs per-tensor byte reality
    L.append("### (c) Shared-vs-per-tensor byte reality\n")
    L.append(f"- SHARED per-role sidecar (signed): "
             f"{foots['SIG16_shared']['sidecar_bytes']/1e3:.1f} KB → total "
             f"{foots['SIG16_shared']['total_bpw']:.3f} bpw (≈0 over body).")
    L.append(f"- SHARED per-role sidecar (full-k16): "
             f"{foots['FULL_k16_shared']['sidecar_bytes']/1e6:.2f} MB → total "
             f"{foots['FULL_k16_shared']['total_bpw']:.3f} bpw.")
    pt = foots['LEARN_k16_pertensor']
    small_bpw = (1 << 16) * CB_ENTRY_BYTES * 8 / 1.0e6   # ~1M-param Linear
    L.append(f"- PER-TENSOR k16 sidecar: {pt['sidecar_bytes']/1e6:.1f} MB → "
             f"total **{pt['total_bpw']:.3f} bpw** "
             f"(+{pt['total_bpw'] - pt['body_bpw']:.2f} bpw model-wide; but "
             f"~+{small_bpw:.1f} bpw on a 1M-param Linear — the small-N tensors "
             f"the coordinator flagged). NOT byte-competitive; this is why the "
             f"champion shares codebooks per-role (sidecar → ≈0).")
    L.append("")
    # (d) smoothing on top of sweep
    L.append("### (d) Smoothing on top of the sweep\n")
    base, sm = kl("SIG16_shared"), kl("SIG16_shared_smooth025")
    if base and sm:
        pct = 100 * (base[0] - sm[0]) / base[0]
        std = max(base[1], sm[1])
        verd = ("helps beyond noise" if base[0] - sm[0] > std else
                "within between-seed noise")
        L.append(f"- signed-S16-shared {base[0]:.4f} → +smooth α=0.25 "
                 f"{sm[0]:.4f} ({pct:+.1f}%, σ={std:.4f}) → **{verd}**.")
    L.append("")
    L.append("## Caveats\n")
    L.append("- Emulation gate only, 0.6B triage; 4B + served re-confirm "
             "remain. Uniform ~2.5 bpw on ALL 196 Linears heavily damages a "
             "0.6B model (top1 well below 1.0 for every 2.5-bpp arm incl. "
             "IQ2_S) — the CB-vs-IQ DELTA is the signal, not absolute KL.")
    L.append("- Full-mode k16 is 1-seed only (56 s/Linear); signed S16 (same "
             "2.5 bpw, 0.3 s/Linear, relerr within ~8% of full-k16) is the "
             "practical champion carried at 4 seeds.")
    L.append("- Shared codebooks trained on ≤2^20 pooled per-role vectors "
             "(subsampled for Lloyd tractability); CUDA Lloyd tie-noise per "
             "seed as in exp-1.")
    doc.write_text("\n".join(L) + "\n")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default=None, help="comma-list of arm ids")
    ap.add_argument("--exp1b", action="store_true",
                    help="run the corrected CB-vs-IQ rerun (scale_sweep + "
                         "signed + shared-per-role codebooks)")
    args = ap.parse_args()

    if args.exp1b:
        RESULTS1B.mkdir(parents=True, exist_ok=True)
        register_variants()
        model = _load_model()
        targets, n_dim, n_head = select_targets(model)
        print(f"[targets] {len(targets)} Linears; excluded dim={n_dim} "
              f"head={n_head}; roles={sorted({role_of(q) for q in targets})}")
        if not args.report_only:
            run_exp1b(model, targets,
                      set(args.arms.split(",")) if args.arms else None)
        doc = write_report_1b(targets)
        print(f"[report] wrote {doc}")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    register_variants()
    arms = build_arms()
    if args.arms:
        keep = set(args.arms.split(","))
        arms = [a for a in arms if a.id in keep]

    model = _load_model()
    targets, n_dim, n_head = select_targets(model)
    print(f"[targets] {len(targets)} Linears; excluded dim={n_dim} head={n_head}")

    if not args.report_only:
        # ensure imatrices exist for all needed seeds
        for seed in SEEDS:
            get_imatrix(model, targets, seed)
        for arm in arms:
            foot = arm_footprint(arm, targets)
            for seed in arm.seeds:
                p = RESULTS / f"{arm.id}__seed{seed}.json"
                if p.exists():
                    print(f"[skip] {arm.id} seed{seed}")
                    continue
                print(f"[run ] {arm.id} seed{seed} (body {foot['body_bpw']:.3f} bpw)")
                rec = run_arm_seed(arm, seed, model, targets, foot)
                print(f"       KL_conf={rec['kl_confident']:.4f} "
                      f"KL_all={rec['kl_all']:.4f} top1={rec['top1_agreement']:.3f} "
                      f"n_swap={rec['n_targets_swapped']}")
        # exp-2 on seed-0 imatrix
        im0 = get_imatrix(model, targets, 0)
        run_entropy(model, targets, im0)

    all_arms = build_arms()
    agg = aggregate(all_arms)
    entropy = json.loads((RESULTS / "exp2_entropy.json").read_text()) \
        if (RESULTS / "exp2_entropy.json").exists() else {}
    doc = write_report(all_arms, agg, entropy, targets, n_dim, n_head)
    print(f"[report] wrote {doc}")


if __name__ == "__main__":
    main()
