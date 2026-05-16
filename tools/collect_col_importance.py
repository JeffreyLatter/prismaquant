"""Capture per-Linear col_importance = E[X^2] from a calibration pass.

Mirrors what build_production_cache computes internally during GPTQ but
just emits the diag(H) proxy so we can score quantized weights by their
activation-weighted reconstruction error.

Output: a dict {qname: tensor[cols]} pickled to --output.
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True,
                   help="JSONL file with one prompt per line (key=text or similar).")
    p.add_argument("--n-calib-samples", type=int, default=32)
    p.add_argument("--calib-seqlen", type=int, default=1024)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    # Hook every nn.Linear on its INPUT; accumulate X^T X diagonal.
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(name):
        def hook(_module, inputs, _output):
            X = inputs[0].detach()
            # X may be [B, T, in_features] or [T, in_features] or [N, in_features]
            X = X.reshape(-1, X.shape[-1]).to(torch.float32)
            s = X.pow(2).sum(dim=0)
            if name not in sums:
                sums[name] = s
                counts[name] = X.shape[0]
            else:
                sums[name] = sums[name] + s
                counts[name] = counts[name] + X.shape[0]
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    # Load calibration prompts. JSONL — try 'text' key first, then full record str.
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
    print(f"[collect] loaded {len(prompts)} prompts", flush=True)
    if len(prompts) < args.n_calib_samples:
        print(f"[collect] WARNING: only {len(prompts)} prompts available", flush=True)

    device = next(model.parameters()).device
    for i, txt in enumerate(prompts[:args.n_calib_samples]):
        enc = tok(txt, return_tensors="pt", truncation=True,
                  max_length=args.calib_seqlen)
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(ids)
        if (i + 1) % 8 == 0:
            print(f"[collect] {i+1}/{args.n_calib_samples}", flush=True)

    for h in hooks:
        h.remove()

    # Convert to col_importance = E[X^2] = sum / count
    col_importance = {
        name: (sums[name] / max(counts[name], 1)).cpu()
        for name in sums
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(col_importance, f)
    print(f"[collect] wrote {len(col_importance)} Linears -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
