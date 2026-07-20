#!/usr/bin/env python3
"""Dense-tier A/B: vanilla NVFP4 (4.5 bpw) vs FP8-CB K36 (4.5 bpw) on Hy3.

Answers the open menu question (2026-07-20): would native NVFP4 have beaten
FP8-CB K36 on the dense/attention/shared tier at matched bits? Measures BOTH
formats through ONE code path — format_registry RTN qdq -> weight-MSE
(unweighted AND activation-column-weighted) x probe h_trace — so the
comparison is internally consistent with the allocation's local cost
convention. Reads the BF16 source directly; GPU-first.

  ab_nvfp4_vs_k36_dense.py [--work /home/rob/dq-runs/prod-hy3-nvfp4cb-2p9]
                           [--source /home/rob/dq-runs/hy3-prod/source]
                           [--limit N]   # first N dense units (debug)

Output: per-role win rates + cost-ratio geomeans + a per-unit CSV.
"""
import argparse
import glob
import json
import pickle
import struct
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "/home/rob/prismaquant")
from prismaquant.format_registry import REGISTRY, canonical_format_name  # noqa: E402
from prismaquant import nvfp4_cb_formats as cb           # noqa: E402


def load_st_headers(src):
    hdrs = {}
    for st in sorted(glob.glob(f"{src}/*.safetensors")):
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            hdrs[k] = (st, v)
    return hdrs


def load_tensor(hdrs, name):
    st, meta = hdrs[name]
    off = meta["data_offsets"]
    dt = {"BF16": torch.bfloat16, "F32": torch.float32,
          "F16": torch.float16}[meta["dtype"]]
    with open(st, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + hlen + off[0])
        buf = f.read(off[1] - off[0])
    t = torch.frombuffer(bytearray(buf), dtype=dt).reshape(meta["shape"])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9")
    ap.add_argument("--source", default="/home/rob/dq-runs/hy3-prod/source")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-first: refusing CPU run"
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.5)

    cfg = json.load(open(f"{args.work}/exported_nvfp4_cb/quant_config.json"))
    # Dense-tier CB targets (skip experts + layers.80) with their shipped rung.
    targets = {}
    for g in cfg["config_groups"].values():
        s = g.get("scheme")
        if not s:
            continue
        rung = f"{'NVFP4_CB' if s['grid']=='fp4' else 'FP8_CB'}_K{s['k']}"
        for t in g["targets"]:
            if ".experts." in t or ".layers.80." in t:
                continue
            targets[t] = rung

    with open(f"{args.work}/artifacts/probe.pkl", "rb") as f:
        probe = pickle.load(f)
    htr = probe.get("h_trace", probe) if isinstance(probe, dict) else probe
    with open(f"{args.work}/artifacts/cb_col_weights.pkl", "rb") as f:
        colw = pickle.load(f)

    spec_nvfp4 = REGISTRY[canonical_format_name("NVFP4")]
    spec_k36 = REGISTRY[canonical_format_name("FP8_CB_K36")]
    hdrs = load_st_headers(args.source)

    def cost(W, spec, cw):
        # fp32 in for BOTH formats: the CB qdq path expects fp32, and cost is
        # measured in fp32 per the cross-layer-additivity finding.
        Wf = W.float()
        q = spec.quantize_dequantize(Wf)
        d2 = (q.float() - Wf) ** 2
        um = d2.mean().item()                       # unweighted weight-MSE
        wm = um
        if cw is not None:
            w = cw.to(dev, torch.float32).clamp_min(0)
            w = w / w.mean().clamp_min(1e-30)
            wm = (d2 * w[None, :]).mean().item()    # act-col-weighted
        return um, wm

    rows = []
    items = sorted(targets.items())
    if args.limit:
        items = items[: args.limit]
    for i, (t, rung) in enumerate(items):
        wname = t + ".weight"
        if wname not in hdrs:
            print(f"[skip] {t}: no source tensor", flush=True)
            continue
        W = load_tensor(hdrs, wname).to(dev, torch.bfloat16)
        h = float(htr.get(t, htr.get(wname, 1.0))) if hasattr(htr, "get") else 1.0
        cw = colw.get(t) if hasattr(colw, "get") else None
        u_n, w_n = cost(W, spec_nvfp4, cw)
        u_k, w_k = cost(W, spec_k36, cw)
        del W
        role = ("shared" if "shared_mlp" in t else "dense/attn")
        rows.append((t, role, rung, h, u_n, u_k, w_n, w_k))
        if i % 50 == 0:
            print(f"[{i}/{len(items)}] {t}: NVFP4 {w_n:.3e} vs K36 {w_k:.3e} "
                  f"({'K36' if w_k < w_n else 'NVFP4'} wins)", flush=True)

    import csv
    out = f"{args.work}/ab_nvfp4_vs_k36.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["target", "role", "shipped_rung", "h_trace",
                     "nvfp4_mse", "k36_mse", "nvfp4_wmse", "k36_wmse"])
        wr.writerows(rows)

    for label, ui, ki in (("unweighted", 4, 5), ("act-weighted", 6, 7)):
        by = defaultdict(lambda: [0, 0, 0.0])
        for r in rows:
            b = by[r[1]]
            b[0] += 1
            if r[ki] < r[ui]:
                b[1] += 1
            b[2] += torch.log(torch.tensor(r[ki] / max(r[ui], 1e-30))).item()
        print(f"\n== {label} weight-MSE, NVFP4(4.5) vs FP8-CB K36(4.5) ==")
        for role, (n, kwins, lg) in sorted(by.items()):
            print(f"  {role:12s} n={n:4d}  K36 wins {kwins} ({100*kwins/n:.0f}%)"
                  f"  geomean cost ratio K36/NVFP4 = {torch.exp(torch.tensor(lg/n)).item():.3f}")
    print(f"\nper-unit CSV: {out}")


if __name__ == "__main__":
    main()
