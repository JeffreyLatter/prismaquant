"""Fail-fast guard against the Python torch fallback for Qwen3.5/3.6
linear-attention forwards.

The transformers Qwen3.5/3.6 modeling code silently falls back to a pure-
Python torch implementation of the gated delta rule when ``causal-conv1d``
and ``flash-linear-attention`` are not installed.  The fallback prints
*one* warning at first forward and is then invisible — and it is roughly
5-10x slower than the optimized kernel path.  Polish on a 27B model
that should take 30 minutes ends up taking 5 hours.

This module exposes a single function ``require_fast_kernels`` that
hard-fails at startup if those packages are missing on a model that
would use them.  Override with the env var
``PRISMAQUANT_ALLOW_PYTORCH_FALLBACK=1`` for the rare case you really
do want to debug-run on the slow path.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from prismaquant.memory_management import env_truthy as _env_truthy


_QWEN_HYBRID_FAST_KERNEL_REQUIREMENTS = (
    ("causal_conv1d", "causal-conv1d (Dao-AILab/causal-conv1d)"),
    ("fla", "flash-linear-attention (fla-org/flash-linear-attention)"),
)


def require_fast_kernels(model_id_or_path: str | None = None) -> None:
    """Raise if the optimized linear-attention kernels are missing on a
    model that would use them.

    ``model_id_or_path`` is consulted to skip the check on architectures
    that do not use the gated delta rule (e.g. DeepSeek, Mistral, Gemma);
    if None, the check fires unconditionally.

    Bypass with ``PRISMAQUANT_ALLOW_PYTORCH_FALLBACK=1`` (intended for
    explicit debug runs only — never as a default).
    """
    if _env_truthy("PRISMAQUANT_ALLOW_PYTORCH_FALLBACK"):
        return

    requirements = _requirements_for_model(model_id_or_path)
    if not requirements:
        return

    missing: list[str] = []
    for module_name, package_name in requirements:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        raise RuntimeError(
            "Required fast-kernel packages missing for this model's "
            "linear-attention forwards: " + ", ".join(missing) + ". "
            "The Python torch fallback is ~5-10x slower than the "
            "optimized kernel path and silently destroys polish wall-"
            "time measurability.  Install both with:\n"
            "  pip install causal-conv1d flash-linear-attention\n"
            "Or set PRISMAQUANT_ALLOW_PYTORCH_FALLBACK=1 to bypass "
            "this guard (debug runs only — never in production)."
        )


def _requirements_for_model(
    model_id_or_path: str | None,
) -> tuple[tuple[str, str], ...]:
    if model_id_or_path is None:
        return _QWEN_HYBRID_FAST_KERNEL_REQUIREMENTS

    path = Path(str(model_id_or_path))
    if (path / "config.json").is_file():
        try:
            from prismaquant.model_profiles import detect_profile

            return tuple(detect_profile(str(path)).fast_kernel_requirements())
        except Exception:
            return ()

    # Backward-compatible fallback for remote IDs or nonlocal shorthand paths.
    lower = str(model_id_or_path).lower()
    is_hybrid_qwen = any(
        tok in lower for tok in (
            "qwen3.5", "qwen3p5", "qwen35",
            "qwen3.6", "qwen3p6", "qwen36",
        )
    )
    return _QWEN_HYBRID_FAST_KERNEL_REQUIREMENTS if is_hybrid_qwen else ()
