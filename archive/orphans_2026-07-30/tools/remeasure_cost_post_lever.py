"""Re-measure cost.pkl from the post-lever production weight cache.

cost.pkl as written by `measure_quant_cost` uses **bare RTN** quantization
to score every (Linear, format) — it doesn't apply the production lever
stack (GPTQ + JSO + damp_sweep). The allocator then promotes Linears to
FP8_DYNAMIC/BF16 based on pessimistic RTN-quality estimates, even though
the production stack would fix many of them at NVFP4. The result is
non-monotonic Pareto curves and over-promotion.

This tool closes the feedback loop. It reads the production-rendered
weights from cache_dir, runs one forward pass on calibration with the
bf16 reference model hooked, computes per-(Linear, format)
output_mse = ||W_q · X − W_bf16 · X||² / numel(out) using PRODUCTION
weights, and writes cost.pkl.v2. The allocator can then re-run with
post-lever costs and produce monotonic Pareto candidates.

This is the minimum viable restoration of the architectural intent
behind the archived `propagated_cost` module
(`archive/cross_layer_2026-05-09/propagated_cost.py`).
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _cache_file(cache_dir: Path, qname: str, fmt: str) -> Path:
    flat = qname.replace(".", "_")
    return cache_dir / f"{flat}__{fmt}.pt"


def _load_prod_weight(cache_dir: Path, qname: str, fmt: str) -> torch.Tensor | None:
    p = _cache_file(cache_dir, qname, fmt)
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False).to(torch.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache-dir", required=True,
                   help="Production cache dir with per-(qname, fmt).pt entries.")
    p.add_argument("--cost-pkl-in", required=True,
                   help="Original cost.pkl (RTN). Used as template — formats list,"
                   " probe shapes, meta. Re-measured output_mse overrides the RTN values.")
    p.add_argument("--cost-pkl-out", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-calib-samples", type=int, default=32)
    p.add_argument("--calib-seqlen", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--formats", default="NVFP4,FP8_DYNAMIC",
                   help="Comma-separated formats to re-measure. BF16 is a "
                   "passthrough so it stays at output_mse=0.")
    args = p.parse_args(argv)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    cache_dir = Path(args.cache_dir)

    # 1. Load original cost.pkl as template
    with open(args.cost_pkl_in, "rb") as f:
        cost_blob = pickle.load(f)
    costs = cost_blob.get("costs", {}) if isinstance(cost_blob, dict) else cost_blob
    print(f"[remeasure] cost.pkl in: {len(costs)} entries", flush=True)

    # 2. Load model
    print(f"[remeasure] loading bf16 model from {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    # Map cost.pkl qname (e.g., model.layers.X.Y) to live nn.Module
    name_to_module: dict[str, nn.Linear] = {}
    for hf_name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # cost.pkl uses model.layers.X.Y while HF state may use model.language_model.layers.X.Y
        norm = hf_name.replace("model.language_model.", "model.")
        name_to_module[norm] = mod

    # Filter: only re-measure qnames that exist both in cost.pkl AND the model
    targets = [qn for qn in costs.keys() if qn in name_to_module]
    print(f"[remeasure] targets: {len(targets)}/{len(costs)} cost.pkl entries map to model Linears", flush=True)

    # 3. Set up hooks that compute output_mse per format on-the-fly.
    #    Load production weights ON-DEMAND inside the hook (free after use).
    sums_sq: dict[tuple[str, str], float] = {}
    sums_ref_sq: dict[str, float] = {}
    counts: dict[str, int] = {}
    skipped_load_logged: set[tuple[str, str]] = set()

    device = next(model.parameters()).device

    def make_hook(qname: str, bf16_weight: torch.Tensor):
        def hook(_module, inputs, output):
            X = inputs[0].detach()  # [B, T, in_features]
            X_f = X.reshape(-1, X.shape[-1]).to(torch.float32)
            W_ref = bf16_weight.detach().to(torch.float32).to(X_f.device)
            Y_ref = X_f @ W_ref.t()
            ref_sq = float(Y_ref.pow(2).sum().item())
            n = Y_ref.shape[0] * Y_ref.shape[1]
            for fmt in formats:
                W_q = _load_prod_weight(cache_dir, qname, fmt)
                if W_q is None:
                    if (qname, fmt) not in skipped_load_logged:
                        skipped_load_logged.add((qname, fmt))
                    continue
                W_q = W_q.to(X_f.device)
                Y_q = X_f @ W_q.t()
                err = float((Y_ref - Y_q).pow(2).sum().item())
                sums_sq[(qname, fmt)] = sums_sq.get((qname, fmt), 0.0) + err
                del W_q, Y_q  # free
            sums_ref_sq[qname] = sums_ref_sq.get(qname, 0.0) + ref_sq
            counts[qname] = counts.get(qname, 0) + n
            del W_ref, Y_ref
        return hook

    hooks = []
    for qn, mod in name_to_module.items():
        if qn not in costs:
            continue
        hooks.append(mod.register_forward_hook(make_hook(qn, mod.weight.data)))
    print(f"[remeasure] hooked {len(hooks)} Linears", flush=True)

    # 5. Load calibration prompts
    prompts = []
    with open(args.dataset) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "__manifest__" in rec:
                continue
            t = rec.get("text") or rec.get("prompt") or rec.get("content")
            if t:
                prompts.append(t)
            if len(prompts) >= args.n_calib_samples:
                break
    print(f"[remeasure] {len(prompts)} prompts", flush=True)

    # 6. Forward pass
    t0 = time.time()
    for i, txt in enumerate(prompts[:args.n_calib_samples]):
        enc = tok(txt, return_tensors="pt", truncation=True,
                  max_length=args.calib_seqlen)
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(ids)
        if (i + 1) % 8 == 0:
            print(f"[remeasure] forward {i+1}/{args.n_calib_samples}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    for h in hooks:
        h.remove()
    print(f"[remeasure] forward done in {time.time()-t0:.1f}s", flush=True)

    # 7. Update cost.pkl in-place
    n_updated = 0
    for (qn, fmt), err_sum in sums_sq.items():
        n = counts.get(qn, 0)
        if n <= 0:
            continue
        output_mse = err_sum / n
        ref_sq = sums_ref_sq.get(qn, 0.0)
        rel_output_mse = err_sum / max(ref_sq, 1e-30)
        per_fmt = costs.setdefault(qn, {})
        entry = per_fmt.setdefault(fmt, {})
        entry["output_mse_rtn"] = entry.get("output_mse", entry.get("output_mse_rtn"))
        entry["output_mse"] = output_mse
        entry["rel_output_mse_rtn"] = entry.get("rel_output_mse", entry.get("rel_output_mse_rtn"))
        entry["rel_output_mse"] = rel_output_mse
        entry["output_mse_source"] = "post_lever_production"
        n_updated += 1
    print(f"[remeasure] updated {n_updated} (qname, format) entries with post-lever output_mse", flush=True)

    # 8. Write
    out_blob = dict(cost_blob) if isinstance(cost_blob, dict) else {"costs": costs}
    out_blob["costs"] = costs
    meta = out_blob.setdefault("meta", {})
    meta["post_lever_remeasure"] = {
        "n_updated": n_updated,
        "formats": formats,
        "n_calib_samples": args.n_calib_samples,
        "calib_seqlen": args.calib_seqlen,
        "model": args.model,
        "cache_dir": str(cache_dir),
    }
    with open(args.cost_pkl_out, "wb") as f:
        pickle.dump(out_blob, f)
    print(f"[remeasure] wrote post-lever cost.pkl -> {args.cost_pkl_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
