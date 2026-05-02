#!/usr/bin/env bash
set -euo pipefail

ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-10}"
M="${M:-512}"
N="${N:-2560}"
K="${K:-2560}"

PYTHONPATH="${PYTHONPATH:-.}" python3 - <<'PY'
import os
import time

import torch

from prismaquant import format_registry as fr
from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul, nvfp4_pack_weight


iters = int(os.environ.get("ITERS", "100"))
warmup = int(os.environ.get("WARMUP", "10"))
M = int(os.environ.get("M", "512"))
N = int(os.environ.get("N", "2560"))
K = int(os.environ.get("K", "2560"))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the NVFP4 fused benchmark")

torch.manual_seed(0)
device = torch.device("cuda")
x = (torch.randn(M, K, device=device) * 0.05).to(torch.bfloat16)
weight = (torch.randn(N, K, device=device) * 0.05).to(torch.bfloat16)
nvfp4 = fr.get_format("NVFP4")

print(f"[bench] shape: x=({M}, {K}) weight=({N}, {K}) iters={iters}")
print("[bench] pre-quantizing weight for both paths")
q_weight = nvfp4.quantize_dequantize(weight).contiguous()
w_packed, w_scales, w_global_scale = nvfp4_pack_weight(weight)


def unfused():
    qx = nvfp4.activation_quantize_dequantize(x)
    return qx @ q_weight.t()


def fused():
    return nvfp4_fused_aw_matmul(x, w_packed, w_scales, w_global_scale)


def measure(label, fn):
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_call_ms = elapsed * 1000.0 / iters
    print(f"[bench] {label}: {per_call_ms:.3f} ms/call")
    return per_call_ms, y


unfused_ms, y_ref = measure("unfused", unfused)
fused_ms, y_fused = measure("fused", fused)
max_abs = (y_ref.float() - y_fused.float()).abs().max().item()
speedup = unfused_ms / fused_ms if fused_ms > 0 else float("inf")
print(f"[bench] speedup: {speedup:.2f}x")
print(f"[bench] max_abs_diff: {max_abs:.6g}")
PY

