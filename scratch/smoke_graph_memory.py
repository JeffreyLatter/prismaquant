#!/usr/bin/env python3
"""Manual CUDA graph memory smoke for the L2/L3 graph stack.

This script never downloads models. Set PRISMAQUANT_SMOKE_MODEL to a local path
or cached HF repo ID, or let it probe a few small Qwen IDs with
local_files_only=True.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import cache_reference_log_probs
from prismaquant.iterate_perturbed_allocation import (
    coordinate_descent_polish,
    measure_assignment_kl,
)
from prismaquant.measure_quant_cost import ActivationIndex, run_cost_pass
from prismaquant.memory_management import report_graph_memory
from prismaquant.perturbed_x_cache import capture_perturbed_activation_cache
from prismaquant.propagated_cost import L3NeighborhoodEntry, measure_propagated_costs


MODEL_CANDIDATES = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2-0.5B",
)


def _set_smoke_env() -> None:
    # Shared graph pools are incompatible with no-clone; see clone warning.
    defaults = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PRISMAQUANT_FUSED_KERNEL_NVFP4": "1",
        "PRISMAQUANT_GRAPH_AUDIT": "1",
        "PRISMAQUANT_GRAPH_SHARED_POOL": "1",
        "PRISMAQUANT_GRAPH_OUTPUT_CLONE": "1",
        "PRISMAQUANT_KL_CUDA_GRAPHS": "1",
        "PRISMAQUANT_L3_CUDA_GRAPHS": "1",
        "PRISMAQUANT_COORD_LANE_BATCH": "1",
        "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS": "1",
        "PRISMAQUANT_COORD_REPLAY_CACHE": "1",
        "PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE": "2",
        "PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE": "2",
        "PRISMAQUANT_VALIDATION_CUDA_GRAPH_CACHE_SIZE": "2",
        "PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH": "2",
        "PRISMAQUANT_SMOKE_SEED": "12345",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def _pin_rng(seed: int) -> None:
    # Note: torch.use_deterministic_algorithms(True) was tried here but
    # produces NaN with the fused NVFP4 Triton kernel. Seeding alone is
    # enough to give bit-identical coord descent decisions across
    # shared-vs-private graph pool runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_local_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    requested = os.environ.get("PRISMAQUANT_SMOKE_MODEL")
    candidates = [requested] if requested else list(MODEL_CANDIDATES)
    last_error = None
    for model_id in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=True,
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            ).eval()
            return model_id, tokenizer, model.cuda()
        except Exception as exc:
            last_error = exc
    print(
        "SKIP: no local small Qwen model available "
        f"(last_error={type(last_error).__name__}: {last_error})"
    )
    return None, None, None


def _calibration_ids(tokenizer) -> torch.Tensor:
    text = [
        "PrismaQuant graph memory smoke.",
        "CUDA graph replay should reuse a shared pool.",
    ]
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(os.environ.get("PRISMAQUANT_SMOKE_SEQLEN", "32")),
    )
    return encoded.input_ids[: int(os.environ.get("PRISMAQUANT_SMOKE_SAMPLES", "2"))]


def _pick_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    preferred = []
    fallback = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        out_features, in_features = module.weight.shape
        if in_features % 16 != 0:
            continue
        item = (name, module)
        if ".layers." in name or "model.layers" in name:
            preferred.append(item)
        else:
            fallback.append(item)
    candidates = preferred or fallback
    if not candidates:
        raise RuntimeError("no CUDA/NVFP4-compatible nn.Linear found")
    return candidates[0]


def _stats_for(name: str, module: nn.Linear, specs: list[fr.FormatSpec]) -> dict:
    shape = tuple(module.weight.shape)
    memory_by_format = {
        spec.name: int(spec.memory_bytes_for_shape(shape))
        for spec in specs
    }
    return {
        name: {
            "n_params": int(module.weight.numel()),
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "h_trace": 1.0,
            "_memory_bytes_by_format": memory_by_format,
        }
    }


def _phase(records: list[dict], label: str, fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = int(torch.cuda.memory_allocated())
    start = time.monotonic()
    result = fn()
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    after = int(torch.cuda.memory_allocated())
    records.append({
        "label": label,
        "before": before,
        "peak": peak,
        "after": after,
        "delta_peak": peak - before,
        "elapsed": time.monotonic() - start,
    })
    report_graph_memory(label)
    return result


def main() -> int:
    _set_smoke_env()
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return 0

    # Pinned for shared-vs-private graph pool determinism check.
    # Keep calibration data and coord descent RNG-aligned across modes.
    # Do not relax without re-verifying both graph pool configurations.
    _pin_rng(int(os.environ["PRISMAQUANT_SMOKE_SEED"]))

    model_id, tokenizer, model = _load_local_model()
    if model is None:
        return 0

    work_root = Path(tempfile.mkdtemp(prefix="prismaquant_graph_smoke_", dir="scratch"))
    records: list[dict] = []
    try:
        calib_ids = _calibration_ids(tokenizer).to("cuda")
        target_name, target_module = _pick_linear(model)
        specs = [fr.get_format("NVFP4"), fr.get_format("MXFP8"), fr.get_format("BF16")]
        stats = _stats_for(target_name, target_module, specs)
        assignment = {target_name: "NVFP4"}
        ref_log_probs = _phase(
            records,
            "reference-log-probs",
            lambda: cache_reference_log_probs(model, calib_ids, torch.device("cuda")),
        )

        cache_dir = work_root / "activation_cache_iter_01"
        _phase(
            records,
            "tiny-l2-cache",
            lambda: capture_perturbed_activation_cache(
                model,
                assignment,
                calib_ids,
                cache_dir,
                input_rows=8,
            ),
        )
        act_cache = ActivationIndex(cache_dir, stats.keys())
        missing = [name for name in stats if name not in act_cache]
        l2_costs = _phase(
            records,
            "tiny-l2-cost",
            lambda: run_cost_pass(
                model,
                act_cache,
                set(stats),
                missing,
                specs,
                model_id,
                "<smoke>",
                "cuda",
                torch.bfloat16,
                "batched",
                1,
                str(work_root / "costs_iter_01.pkl"),
            ),
        )
        current_kl = _phase(
            records,
            "assignment-kl-graph",
            lambda: measure_assignment_kl(
                model,
                assignment,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
            ),
        )
        neighborhood = [
            L3NeighborhoodEntry(
                name=target_name,
                current_format="NVFP4",
                formats=("MXFP8", "NVFP4", "BF16"),
                margin=0.0,
                l2_current_cost=0.0,
            )
        ]
        l3_costs = _phase(
            records,
            "l3-propagated-graph",
            lambda: measure_propagated_costs(
                model,
                assignment,
                neighborhood,
                calib_ids,
                specs,
                work_root=work_root,
                max_lanes_per_batch=2,
                tail_only=True,
                output_mse_names=[],
            ),
        )
        _phase(
            records,
            "coord-descent-replay-graph",
            lambda: coordinate_descent_polish(
                model,
                assignment,
                l3_costs or l2_costs,
                specs,
                16.0,
                calib_ids,
                ref_log_probs,
                stats=stats,
                work_root=work_root,
                current_kl=float(current_kl),
                max_passes=1,
                early_stop_streak=2,
                max_lanes_per_batch=2,
                emit=print,
            ),
        )

        print(f"model={model_id} target={target_name}")
        print("phase,before_gb,peak_gb,after_gb,delta_peak_gb,elapsed_s")
        for row in records:
            print(
                f"{row['label']},"
                f"{row['before'] / 1024 ** 3:.4f},"
                f"{row['peak'] / 1024 ** 3:.4f},"
                f"{row['after'] / 1024 ** 3:.4f},"
                f"{row['delta_peak'] / 1024 ** 3:.4f},"
                f"{row['elapsed']:.3f}"
            )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
