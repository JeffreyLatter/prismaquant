"""Cross-layer sensitivity instrument: measure, don't model.

Directly measures, on a resident fp32 model with exact weight save/restore:

  * unary KL    — KL(teacher ‖ model with one Linear flipped to fmt)
  * assignment  — KL with a whole assignment applied, vs Σ unary
                  → the ADDITIVITY RESIDUAL (the cross-layer interaction term)
  * pairs       — I(i,j) = KL({i,j}) − KL(i) − KL(j), stratified by circuit
                  relation → the interaction-structure map

All KLs are full-vocab forward KL (teacher→student), fp32, mean over token
positions, with a per-window stderr so "interaction ≈ 0" is a statistical
statement. The teacher's log-probs are cached once on GPU; each flip is an
in-place quantize + bit-exact restore from a saved copy.

This is the measurement layer under the additive-cost question: AURA prices
propagation exactly and assumes interaction ≈ 0 inside a trust region; this
tool finds that region's boundary instead of asserting it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import prismaquant.format_registry as fr
from prismaquant.aura_cost import _git_commit, _free_gib


def _log(msg: str) -> None:
    print(f"[xlayer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def target_linears(model: nn.Module) -> dict[str, nn.Linear]:
    out: dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        if mod.weight.dim() == 2 and min(mod.weight.shape) >= 16:
            out[name] = mod
    return out


class TeacherCache:
    """Per-window teacher log-probs + entropy term, GPU-resident.

    KL(t‖s) per token = Σ_v p_t (log p_t − log p_s); the Σ p_t log p_t term is
    flip-independent, so we precompute p_t and the entropy scalar once and each
    evaluation only pays one student forward + one reduction.
    """

    def __init__(self, model: nn.Module, windows: torch.Tensor):
        self.p_t: list[torch.Tensor] = []
        self.neg_ent: list[torch.Tensor] = []  # Σ p_t log p_t per token
        with torch.no_grad():
            for i in range(windows.size(0)):
                logits = model(windows[i:i + 1]).logits.float()
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                self.p_t.append(p)
                self.neg_ent.append((p * logp).sum(dim=-1))

    def kl_window(self, model: nn.Module, windows: torch.Tensor,
                  i: int) -> float:
        with torch.no_grad():
            logits = model(windows[i:i + 1]).logits.float()
            logp_s = F.log_softmax(logits, dim=-1)
            cross = (self.p_t[i] * logp_s).sum(dim=-1)
            return float((self.neg_ent[i] - cross).mean().item())


class Flipper:
    """Apply/revert per-Linear RTN format flips with bit-exact restore."""

    def __init__(self, linears: dict[str, nn.Linear]):
        self.linears = linears
        self._saved: dict[str, torch.Tensor] = {}
        self._qdq_cache: dict[tuple[str, str], torch.Tensor] = {}

    def quantized_weight(self, name: str, fmt: str) -> torch.Tensor:
        key = (name, fmt)
        if key not in self._qdq_cache:
            spec = fr.get_format(fmt)
            w = self.linears[name].weight.detach()
            self._qdq_cache[key] = spec.quantize_dequantize(w.float())
            if len(self._qdq_cache) > 8:  # bounded; recompute is cheap
                self._qdq_cache.pop(next(iter(self._qdq_cache)))
        return self._qdq_cache[key]

    def apply(self, flips: Iterable[tuple[str, str]]) -> None:
        with torch.no_grad():
            for name, fmt in flips:
                w = self.linears[name].weight
                if name not in self._saved:
                    self._saved[name] = w.detach().clone()
                w.copy_(self.quantized_weight(name, fmt))

    def revert(self) -> None:
        with torch.no_grad():
            for name, saved in self._saved.items():
                self.linears[name].weight.copy_(saved)
        self._saved.clear()


def measure_subset(
    flips: Sequence[tuple[str, str]],
    flipper: Flipper,
    teacher: TeacherCache,
    model: nn.Module,
    windows: torch.Tensor,
) -> dict:
    flipper.apply(flips)
    try:
        vals = [teacher.kl_window(model, windows, i)
                for i in range(windows.size(0))]
    finally:
        flipper.revert()
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    return {"kl_mean": mean, "kl_stderr": math.sqrt(var / n),
            "kl_windows": vals}


def relation(a: str, b: str) -> str:
    """Circuit relation between two Linear names (Qwen-style naming)."""
    def parse(n):
        parts = n.split(".")
        layer = next((int(p) for p in parts if p.isdigit()), -1)
        sub = "attn" if "attn" in n else ("mlp" if "mlp" in n else "?")
        return layer, sub, parts[-1]
    la, sa, ra = parse(a)
    lb, sb, rb = parse(b)
    if la == lb and sa == sb:
        return f"same_{sa}"
    if la == lb:
        return "same_block_cross"
    if abs(la - lb) == 1:
        return "adjacent_block"
    if ra == rb:
        return "same_role_distant"
    return "distant"


def greedy_assignment(
    unary: dict[tuple[str, str], dict],
    linears: dict[str, nn.Linear],
    fmt: str,
    target_bpp: float,
    bits_lo: float = 4.5,
    bits_hi: float = 16.0,
) -> list[tuple[str, str]]:
    """Cheapest-unary-first fill into `fmt` until target bpp — the same shape
    a real 2-rung allocation takes (low-cost Linears get the low format)."""
    total = sum(m.weight.numel() for m in linears.values())
    budget = (bits_hi - target_bpp) / (bits_hi - bits_lo) * total
    order = sorted(
        (k for k in unary if k[1] == fmt),
        key=lambda k: unary[k]["kl_mean"],
    )
    out, used = [], 0
    for name, f in order:
        n = linears[name].weight.numel()
        if used + n > budget:
            continue
        out.append((name, f))
        used += n
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="cross-layer sensitivity measurement")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fmt", default="NVFP4",
                   help="flip format for unary/pairs/assignment members")
    p.add_argument("--n-windows", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--mode", default="unary+assignments",
                   choices=["unary", "unary+assignments", "pairs", "all"])
    p.add_argument("--bpp-targets", default="8,6,5,4.75,4.5",
                   help="assignment sweep targets (2-rung fmt/BF16 fill)")
    p.add_argument("--pairs-top-m", type=int, default=40,
                   help="exhaustive pairs among the top-M unary-KL Linears, "
                        "plus a stratified sample of the rest")
    p.add_argument("--pairs-sample", type=int, default=400)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="measurement dtype — bfloat16 reproduces the "
                        "bf16-floor artifact for the Q5 comparison")
    p.add_argument("--min-free-gib", type=float, default=6.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.calibration_data import load_wikitext_calibration_windowed

    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    _log(f"loading {args.model} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dt, trust_remote_code=True,
        attn_implementation="eager", device_map=args.device,
    ).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    windows = load_wikitext_calibration_windowed(
        tok, args.n_windows, args.seqlen, split=args.calib_split,
    ).to(args.device)

    linears = target_linears(model)
    flipper = Flipper(linears)
    _log(f"{len(linears)} target Linears; caching teacher log-probs "
         f"({args.n_windows}x{args.seqlen}); free={_free_gib():.1f}")
    teacher = TeacherCache(model, windows)
    if _free_gib() < args.min_free_gib:
        raise RuntimeError(f"free {_free_gib():.1f} < {args.min_free_gib} GiB")

    fmt = fr.canonical_format_name(args.fmt)
    result: dict = {
        "schema": "prismaquant.cross_layer_residual.v1",
        "provenance": {
            "model": args.model, "fmt": fmt, "dtype": args.dtype,
            "n_windows": args.n_windows, "seqlen": args.seqlen,
            "calib_split": args.calib_split, "git_commit": _git_commit(),
        },
    }

    # --- unary sweep (always needed) ---
    t0 = time.monotonic()
    unary: dict[tuple[str, str], dict] = {}
    for i, name in enumerate(linears):
        unary[(name, fmt)] = measure_subset(
            [(name, fmt)], flipper, teacher, model, windows)
        if (i + 1) % 32 == 0:
            _log(f"unary {i + 1}/{len(linears)} "
                 f"({time.monotonic() - t0:.0f}s)")
    result["unary"] = {
        f"{n}|{f}": v for (n, f), v in unary.items()
    }
    _log(f"unary sweep done in {time.monotonic() - t0:.0f}s")

    # --- assignment residuals (Q1) ---
    if args.mode in ("unary+assignments", "all"):
        rows = []
        for tgt in (float(x) for x in args.bpp_targets.split(",")):
            members = greedy_assignment(unary, linears, fmt, tgt)
            if not members:
                continue
            meas = measure_subset(members, flipper, teacher, model, windows)
            s_unary = sum(unary[m]["kl_mean"] for m in members)
            s_stderr = math.sqrt(sum(
                unary[m]["kl_stderr"] ** 2 for m in members))
            row = {
                "bpp_target": tgt, "n_members": len(members),
                "members": [m[0] for m in members],
                "kl_full": meas["kl_mean"],
                "kl_full_stderr": meas["kl_stderr"],
                # per-window values so the residual can be computed PAIRED
                # against Σ unary per window (removes shared window-difficulty
                # variance — the honest significance test for the residual)
                "kl_windows": meas["kl_windows"],
                "sum_unary": s_unary,
                "sum_unary_stderr": s_stderr,
                "residual": meas["kl_mean"] - s_unary,
                "residual_over_full": (
                    (meas["kl_mean"] - s_unary) / meas["kl_mean"]
                    if meas["kl_mean"] else 0.0),
            }
            rows.append(row)
            _log(f"assignment bpp~{tgt}: n={len(members)} "
                 f"full={row['kl_full']:.5f}±{row['kl_full_stderr']:.5f} "
                 f"Σunary={s_unary:.5f} residual={row['residual']:+.5f} "
                 f"({100 * row['residual_over_full']:+.1f}% of full)")
        result["assignments"] = rows

    # --- pairwise interactions (Q2) ---
    if args.mode in ("pairs", "all"):
        g = torch.Generator().manual_seed(args.seed)
        names = list(linears)
        by_kl = sorted(names, key=lambda n: -unary[(n, fmt)]["kl_mean"])
        top = by_kl[:args.pairs_top_m]
        pair_set = {(a, b) for ii, a in enumerate(top) for b in top[ii + 1:]}
        while len(pair_set) < args.pairs_top_m * (args.pairs_top_m - 1) // 2 \
                + args.pairs_sample and len(names) > 1:
            i = int(torch.randint(len(names), (1,), generator=g))
            j = int(torch.randint(len(names), (1,), generator=g))
            if i != j:
                pair_set.add(tuple(sorted((names[i], names[j]))))
        pairs_out = []
        t0 = time.monotonic()
        for k, (a, b) in enumerate(sorted(pair_set)):
            meas = measure_subset(
                [(a, fmt), (b, fmt)], flipper, teacher, model, windows)
            ka, kb = unary[(a, fmt)], unary[(b, fmt)]
            inter = meas["kl_mean"] - ka["kl_mean"] - kb["kl_mean"]
            stderr = math.sqrt(meas["kl_stderr"] ** 2
                               + ka["kl_stderr"] ** 2
                               + kb["kl_stderr"] ** 2)
            pairs_out.append({
                "a": a, "b": b, "relation": relation(a, b),
                "kl_joint": meas["kl_mean"], "interaction": inter,
                "interaction_stderr": stderr,
                "significant": abs(inter) > 2 * stderr,
            })
            if (k + 1) % 200 == 0:
                _log(f"pairs {k + 1}/{len(pair_set)} "
                     f"({time.monotonic() - t0:.0f}s)")
        result["pairs"] = pairs_out

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))
    _log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
