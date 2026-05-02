#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DEVICE="${DEVICE:-cuda}"
MAX_LANES="${MAX_LANES:-64}"
MAX_LENGTH="${MAX_LENGTH:-128}"
COORD_LINEAR_LIMIT="${COORD_LINEAR_LIMIT:-8}"
COORD_EARLY_STOP="${COORD_EARLY_STOP:-8}"
SYNTHETIC_LAYERS="${SYNTHETIC_LAYERS:-8}"
SYNTHETIC_HIDDEN="${SYNTHETIC_HIDDEN:-64}"
SYNTHETIC_SEQ_LEN="${SYNTHETIC_SEQ_LEN:-32}"

python3 - <<'PY'
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import cache_reference_log_probs
from prismaquant.iterate_perturbed_allocation import (
    coordinate_descent_polish,
    measure_assignment_kl,
)
from prismaquant.propagated_cost import L3NeighborhoodEntry, measure_propagated_costs


model_name = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
device_pref = os.environ.get("DEVICE", "cuda")
device = "cuda" if device_pref != "cpu" and torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
max_lanes = int(os.environ.get("MAX_LANES", "64"))
max_length = int(os.environ.get("MAX_LENGTH", "128"))
coord_linear_limit = int(os.environ.get("COORD_LINEAR_LIMIT", "8"))
coord_early_stop = int(os.environ.get("COORD_EARLY_STOP", "8"))
work_dir = os.environ.get("WORK_DIR", "/tmp")
os.makedirs(work_dir, exist_ok=True)


class _SyntheticBlock(nn.Module):
    def __init__(self, hidden: int, idx: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden) * (1.0 + 0.01 * idx))

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _SyntheticDecoder(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_SyntheticBlock(hidden, idx) for idx in range(layers)]
        )
        self.norm = nn.Identity()


class _SyntheticCausalLM(nn.Module):
    def __init__(self, *, vocab: int = 257, hidden: int = 64, layers: int = 8):
        super().__init__()
        self.model = _SyntheticDecoder(vocab, hidden, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


if model_name.lower() in {"synthetic", "synthetic-decoder"}:
    synthetic_layers = int(os.environ.get("SYNTHETIC_LAYERS", "8"))
    synthetic_hidden = int(os.environ.get("SYNTHETIC_HIDDEN", "64"))
    synthetic_seq_len = int(os.environ.get("SYNTHETIC_SEQ_LEN", "32"))
    model = _SyntheticCausalLM(
        hidden=synthetic_hidden,
        layers=synthetic_layers,
    ).to(device=device, dtype=dtype).eval()
    calib_ids = (
        torch.arange(2 * synthetic_seq_len, dtype=torch.long)
        .reshape(2, synthetic_seq_len)
        % 257
    ).to(device)
    encoded = {
        "input_ids": calib_ids,
        "attention_mask": torch.ones_like(calib_ids),
    }
else:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(f"transformers is required for this benchmark: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
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
    calib_ids = encoded["input_ids"].to(device)

linear_names = [
    name
    for name, module in model.named_modules()
    if isinstance(module, nn.Linear) and ".layers." in name
]
if not linear_names:
    raise SystemExit("could not find a decoder Linear to benchmark")
target = linear_names[0]

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

coord_names = (
    linear_names[-coord_linear_limit:]
    if len(linear_names) > coord_linear_limit
    else linear_names[:coord_linear_limit]
)
modules = dict(model.named_modules())
coord_assignment = {name: "BF16" for name in coord_names}
coord_stats = {}
for name in coord_names:
    weight = modules[name].weight
    coord_stats[name] = {
        "n_params": int(weight.numel()),
        "in_features": int(weight.shape[-1]),
        "out_features": int(weight.shape[-2]),
        "_memory_bytes_by_format": {
            "MXFP8": fr.get_format("MXFP8").memory_bytes_for_shape(tuple(weight.shape)),
            "BF16": fr.get_format("BF16").memory_bytes_for_shape(tuple(weight.shape)),
        },
    }
coord_l3_costs = {
    name: {
        "BF16": {"propagated_end_kl": 0.0},
        "MXFP8": {"propagated_end_kl": float(idx + 1)},
    }
    for idx, name in enumerate(coord_names)
}
coord_specs = [fr.get_format("MXFP8"), fr.get_format("BF16")]
ref_log_probs = cache_reference_log_probs(model, calib_ids, device)
coord_start_kl = measure_assignment_kl(
    model,
    coord_assignment,
    calib_ids,
    ref_log_probs,
    work_root=work_dir,
)


def run_coord(label: str, *, lane_batch: bool, replay_cache: bool) -> dict:
    os.environ["PRISMAQUANT_COORD_LANE_BATCH"] = "1" if lane_batch else "0"
    os.environ["PRISMAQUANT_COORD_REPLAY_CACHE"] = "1" if replay_cache else "0"
    if device == "cuda":
        torch.cuda.synchronize()
    coord_start = time.monotonic()
    polished, coord_final_kl, coord_meta = coordinate_descent_polish(
        model,
        coord_assignment,
        coord_l3_costs,
        coord_specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=coord_stats,
        work_root=work_dir,
        current_kl=coord_start_kl,
        return_metadata=True,
        early_stop_streak=coord_early_stop,
        max_lanes_per_batch=max_lanes,
        emit=print,
        anchor_label=label,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    coord_elapsed = time.monotonic() - coord_start
    coord_summary = {
        "label": label,
        "linear_count": len(coord_names),
        "wall": coord_elapsed,
        "start_kl": float(coord_start_kl),
        "final_kl": float(coord_final_kl),
        "meta": coord_meta,
        "accepted_formats": {
            fmt: sum(1 for value in polished.values() if value == fmt)
            for fmt in sorted(set(polished.values()))
        },
    }
    print(
        f"{label}: wall={coord_elapsed:.3f}s "
        f"final_kl={float(coord_final_kl):.12g} "
        f"flips={coord_meta['flips_committed']}"
    )
    return coord_summary


coord_results = [
    run_coord("coord_sequential", lane_batch=False, replay_cache=False),
    run_coord("coord_lane_batched_no_cache", lane_batch=True, replay_cache=False),
    run_coord("coord_lane_batched_replay_cache", lane_batch=True, replay_cache=True),
]
coord_baseline = coord_results[0]
coord_speedups = {
    item["label"]: coord_baseline["wall"] / item["wall"]
    for item in coord_results[1:]
    if item["wall"] > 0
}
print(json.dumps({"coord_results": coord_results, "coord_speedups": coord_speedups}, indent=2))
PY
