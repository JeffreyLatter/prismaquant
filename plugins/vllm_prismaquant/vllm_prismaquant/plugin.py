"""vLLM registration hook.

Unlike the GGUF plugin we need **no** model-loader / config-parser / engine-arg
monkeypatches: the artifact is a standard HF dir (config.json + safetensors), so
vLLM's normal load path applies. We only register the quantization config; the
shared codebooks are read from the `codebook_file` sidecar by the config itself
(via `get_current_vllm_config()`), and the packed custom op is registered on
import of `.ops`. Zero vLLM-core surface (serving-kernel.md §2).
"""
from __future__ import annotations

from vllm.model_executor.layers.quantization import register_quantization_config

from . import ops  # noqa: F401  (registers the prismaquant::cb_gemm custom op)
from .config import PrismaQuantConfig


def register() -> None:
    try:
        register_quantization_config("prismaquant")(PrismaQuantConfig)
    except ValueError:
        # Already registered (idempotent across repeated plugin loads).
        pass
