"""Shared layer-config parsing helpers.

PrismaQuant writes allocator assignments in a few shapes: shorthand strings,
integer bit widths, and AutoRound-style dictionaries. Keep the production
parser in one place so export, recache, KL validation, and small tools cannot
silently disagree about the same recipe.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from prismaquant.schemas import validate_layer_config_payload


def strip_weight(name: str) -> str:
    """Normalize tensor names to module qnames."""
    return name[:-len(".weight")] if name.endswith(".weight") else name


# GGUF k-quant + IQ lane (llama.cpp / vLLM-GGUF serving). Kept as an explicit
# literal so this module stays torch-free; pinned to gguf_formats.GGUF_BLOCK_BYTES
# by test_gguf_formats.test_layer_config_gguf_names_stay_in_sync.
_GGUF_FORMAT_NAMES = frozenset(
    {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
     "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL"}
)

# NVFP4-CB / FP8-CB vector-quantization codebook lane (custom vLLM plugin;
# pinned to format_registry by test_nvfp4_cb_formats). ALL integer rungs per
# STANDARDS.md: NVFP4_CB_K12..K24, NVFP4_CB_S13..S16 (sign-magnitude),
# FP8_CB_K28..K48 (the 2026-07-21 all-integer ladder — a stale step-4 set
# here crashed the first full-ladder 27B export on cb_k=47).
_NVFP4_CB_FORMAT_NAMES = frozenset(
    {f"NVFP4_CB_K{k}" for k in range(12, 25)}
    | {f"NVFP4_CB_S{k}" for k in (13, 14, 15, 16)}
    | {f"FP8_CB_K{k}" for k in range(28, 49)}
)


def canonicalize_format(entry: dict | str | int) -> str:
    """Map a layer-config entry to an export/runtime format name.

    This parser is runtime-neutral: research formats such as E5M2 are
    canonicalized here, then serving/export profiles decide whether they are
    legal for a concrete backend.
    """
    if isinstance(entry, dict):
        dt = entry.get("data_type")
        bits = int(entry.get("bits", 0))
        if dt == "gguf":
            gguf_type = str(entry.get("gguf_type", "")).upper()
            if gguf_type not in _GGUF_FORMAT_NAMES:
                raise ValueError(f"unsupported gguf scheme: {entry!r}")
            return gguf_type
        if dt == "nvfp4_cb":
            rung = "S" if str(entry.get("cb_mode", "")) == "signed" else "K"
            name = f"NVFP4_CB_{rung}{int(entry['cb_k'])}"
            if name not in _NVFP4_CB_FORMAT_NAMES:
                raise ValueError(f"unsupported nvfp4_cb scheme: {entry!r}")
            return name
        if dt == "fp8_cb":
            name = f"FP8_CB_K{int(entry['cb_k'])}"
            if name not in _NVFP4_CB_FORMAT_NAMES:
                raise ValueError(f"unsupported fp8_cb scheme: {entry!r}")
            return name
        if dt == "nv_fp" and bits == 4:
            return "NVFP4"
        if dt == "mx_fp" and bits == 4:
            return "MXFP4"
        if dt == "mx_fp" and bits == 8:
            elt = str(entry.get("weight_element_dtype", "fp8_e4m3")).lower()
            if elt == "fp8_e5m2":
                return "MXFP8_E5M2"
            return "MXFP8_E4M3"
        if dt in ("float", "bfloat16") and bits in (16, 0):
            return "BF16"
        if dt == "fp8_e4m3" and bits == 8:
            group_size = int(entry.get("group_size", 0))
            if group_size == 128:
                return "FP8_SOURCE"
            if group_size == 32:
                return "MXFP8_E4M3"
            if group_size in (0, -1):
                return "FP8_E4M3"
            return "MXFP8_E4M3"
        if dt == "fp8_e5m2" and bits == 8:
            return "FP8_E5M2"
        if dt == "mx_fp" and bits == 6:
            elt = str(entry.get("weight_element_dtype", "fp6_e3m2")).lower()
            if elt == "fp6_e2m3":
                return "MXFP6_E2M3"
            return "MXFP6_E3M2"
        if dt == "fp6_e3m2" and bits == 6:
            return "MXFP6_E3M2"
        if dt == "fp6_e2m3" and bits == 6:
            return "MXFP6_E2M3"
        raise ValueError(f"unsupported scheme: {entry!r}")
    if isinstance(entry, str):
        value = entry.lower()
        if value.upper() in _GGUF_FORMAT_NAMES:
            return value.upper()
        if value.upper() in _NVFP4_CB_FORMAT_NAMES:
            return value.upper()
        if value in ("nvfp4", "fp4", "4"):
            return "NVFP4"
        if value in ("mxfp4", "mx_fp4"):
            return "MXFP4"
        if value in ("mxfp8", "mxfp8_e4m3"):
            return "MXFP8_E4M3"
        if value in ("fp8", "fp8_dynamic", "fp8_e4m3", "fp8_e4m3fn", "8"):
            return "FP8_E4M3"
        if value in ("mxfp8_e5m2", "mx_fp8_e5m2"):
            return "MXFP8_E5M2"
        if value in ("fp8_e5m2", "fp8_e5m2fn"):
            return "FP8_E5M2"
        if value in ("bf16", "bfloat16", "16"):
            return "BF16"
        raise ValueError(f"unsupported format string: {entry!r}")
    if isinstance(entry, int):
        if entry <= 4:
            return "NVFP4"
        if entry <= 8:
            return "FP8_E4M3"
        return "BF16"
    raise ValueError(f"unsupported scheme: {entry!r}")


def canonicalize_assignment(raw: Mapping) -> dict[str, str]:
    return {
        strip_weight(str(name)): canonicalize_format(entry)
        for name, entry in raw.items()
    }


def load_assignment(path: str | Path) -> dict[str, str]:
    path = Path(path)
    payload = json.loads(path.read_text())
    validate_layer_config_payload(payload, str(path))
    return canonicalize_assignment(payload)
