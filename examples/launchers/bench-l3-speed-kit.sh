#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DEVICE="${DEVICE:-cuda}"
MAX_LANES="${MAX_LANES:-64}"
MAX_LENGTH="${MAX_LENGTH:-128}"

python3 - <<'PY'
import json
import os
import time

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.propagated_cost import L3NeighborhoodEntry, measure_propagated_costs

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:
    raise SystemExit(f"transformers is required for this benchmark: {exc}") from exc


model_name = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
device_pref = os.environ.get("DEVICE", "cuda")
device = "cuda" if device_pref != "cpu" and torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
max_lanes = int(os.environ.get("MAX_LANES", "64"))
max_length = int(os.environ.get("MAX_LENGTH", "128"))

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    trust_remote_code=True,
).to(device).eval()

linear_names = [
    name
    for name, module in model.named_modules()
    if isinstance(module, nn.Linear) and ".layers." in name
]
if not linear_names:
    raise SystemExit("could not find a decoder Linear to benchmark")
target = linear_names[0]

encoded = tokenizer(
    [
        "PrismaQuant measures propagated quantization costs.",
        "A tiny L3 benchmark checks cache and graph equivalence.",
    ],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=max_length,
)
calibration = {
    key: value.to(device)
    for key, value in encoded.items()
    if isinstance(value, torch.Tensor)
}
assignment = {target: "BF16"}
neighborhood = [
    L3NeighborhoodEntry(
        name=target,
        current_format="BF16",
        formats=("MXFP8", "BF16"),
        margin=0.0,
        l2_current_cost=0.0,
    )
]
specs = [fr.get_format("MXFP8"), fr.get_format("BF16")]


def run(label: str, *, prequant_cache: bool, cuda_graphs: bool) -> dict:
    os.environ["PRISMAQUANT_L3_PREQUANT_CACHE"] = "1" if prequant_cache else "0"
    os.environ["PRISMAQUANT_L3_CUDA_GRAPHS"] = "1" if cuda_graphs else "0"
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.monotonic()
    costs = measure_propagated_costs(
        model,
        assignment,
        neighborhood,
        calibration,
        specs,
        max_lanes_per_batch=max_lanes,
        tail_only=True,
        output_mse_names=[],
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    kl = float(costs[target]["MXFP8"]["propagated_end_kl"])
    print(f"{label}: wall={elapsed:.3f}s propagated_end_kl={kl:.12g}")
    return {"label": label, "wall": elapsed, "kl": kl}


results = [
    run("no_cache_no_graphs", prequant_cache=False, cuda_graphs=False),
    run("cache_only", prequant_cache=True, cuda_graphs=False),
    run("cache_plus_graphs", prequant_cache=True, cuda_graphs=True),
]
baseline = results[0]
for item in results[1:]:
    if abs(item["kl"] - baseline["kl"]) > 1e-8 * max(abs(baseline["kl"]), 1.0):
        raise SystemExit(
            f"KL mismatch for {item['label']}: {item['kl']} vs {baseline['kl']}"
        )
summary = {
    "model": model_name,
    "device": device,
    "target": target,
    "results": results,
    "speedups": {
        item["label"]: baseline["wall"] / item["wall"]
        for item in results[1:]
        if item["wall"] > 0
    },
}
print(json.dumps(summary, indent=2))
PY
