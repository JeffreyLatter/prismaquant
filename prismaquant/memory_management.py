"""Shared memory controls for cache-heavy quantization passes."""
from __future__ import annotations

import gc
import os
import sys
import weakref
from typing import Iterable

import torch


_BUDGET_EVICTORS: "weakref.WeakSet[object]" = weakref.WeakSet()


class GPUMemoryBudgetExceeded(RuntimeError):
    """Raised when cache eviction cannot bring CUDA memory under budget."""


def env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return max(parsed, 0)


def register_budget_evictor(evictor: object) -> None:
    try:
        _BUDGET_EVICTORS.add(evictor)
    except TypeError:
        pass


def unregister_budget_evictor(evictor: object) -> None:
    try:
        _BUDGET_EVICTORS.discard(evictor)
    except TypeError:
        pass


def max_gpu_memory_bytes() -> int | None:
    gb = env_float("PRISMAQUANT_MAX_GPU_MEM_GB", 85.0)
    if gb <= 0.0:
        return None
    return int(gb * 1024 ** 3)


def cuda_memory_info(device: torch.device | None = None) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def _gb(num_bytes: int | float) -> float:
    return float(num_bytes) / float(1024 ** 3)


def _unique_evictors(evictors: Iterable[object]) -> list[object]:
    out: list[object] = []
    seen: set[int] = set()
    for evictor in evictors:
        if evictor is None:
            continue
        key = id(evictor)
        if key in seen:
            continue
        seen.add(key)
        out.append(evictor)
    return out


def _drop_released_cuda_memory(*, synchronize: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if synchronize:
            torch.cuda.synchronize()


def enforce_gpu_memory_budget(
    evictors: Iterable[object] = (),
    *,
    device: torch.device | None = None,
    reason: str = "",
) -> int:
    """Evict oldest registered cache entries until used CUDA memory is in budget.

    ``torch.cuda.mem_get_info`` reports driver-visible free memory. We compare
    ``total - free`` against ``PRISMAQUANT_MAX_GPU_MEM_GB`` so the budget acts
    as a hard ceiling even when PyTorch's caching allocator is holding blocks.
    """
    budget_bytes = max_gpu_memory_bytes()
    if budget_bytes is None:
        return 0
    info = cuda_memory_info(device)
    if info is None:
        return 0
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes
    if used_bytes <= budget_bytes:
        return 0

    candidates = _unique_evictors([*evictors, *_BUDGET_EVICTORS])
    evicted = 0
    while used_bytes > budget_bytes:
        progress = False
        for evictor in candidates:
            evict_one = getattr(evictor, "evict_oldest_for_memory_budget", None)
            if not callable(evict_one):
                continue
            if evict_one():
                evicted += 1
                progress = True
                _drop_released_cuda_memory()
                info = cuda_memory_info(device)
                if info is None:
                    return evicted
                free_bytes, total_bytes = info
                used_bytes = total_bytes - free_bytes
                if used_bytes <= budget_bytes:
                    break
        if not progress:
            detail = f" during {reason}" if reason else ""
            raise GPUMemoryBudgetExceeded(
                "CUDA memory budget exceeded"
                f"{detail}: used={_gb(used_bytes):.2f}GB "
                f"budget={_gb(budget_bytes):.2f}GB "
                f"total={_gb(total_bytes):.2f}GB. "
                "No registered cache entries remain to evict; the model or "
                "other allocations exceed PRISMAQUANT_MAX_GPU_MEM_GB."
            )
    return evicted


def phase_boundary_memory_cleanup(label: str | None = None) -> None:
    """Release allocator-held memory and collect Python garbage at phase edges."""
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if label:
            print(
                f"[memory] cleanup {label}: empty_cache failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
