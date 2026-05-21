"""Vendored modeling code for architecture gaps and local compatibility fixes.

Currently:
- transformers_deepseek_v4/ — from transformers PR #45643 (Arthur Zucker's
  add-deepseek-v4 branch, vendored 2026-04-27). Required because DSv4 is
  the test target for the prismaquant DSv4-Flash-Base run.
- transformers_qwen3/ — local Qwen3 copy with RoPE cos/sin precomputed at
  init time to avoid deterministic cuBLAS NaNs in per-forward RoPE matmul.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).parent


_QWEN3_REGISTERED = False
_DSV4_REGISTERED = False


def register_qwen3() -> None:
    """Route Qwen3 causal-LM loads to PrismaQuant's vendored model.

    Idempotent — safe to call on any PrismaQuant import or profile detect.
    Uses the upstream Qwen3Config and only replaces the causal-LM model class.
    """
    global _QWEN3_REGISTERED
    if _QWEN3_REGISTERED:
        return

    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from .transformers_qwen3.modeling_qwen3 import Qwen3ForCausalLM

    AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)
    _QWEN3_REGISTERED = True


def register_deepseek_v4() -> None:
    """Register vendored DeepSeek-V4 modeling code with transformers.

    Idempotent — safe to call multiple times. After this returns:
    - `transformers.models.deepseek_v4` is importable
    - `AutoConfig.from_pretrained` resolves model_type='deepseek_v4'
    - `AutoModelForCausalLM.from_pretrained` instantiates DeepseekV4ForCausalLM

    Transformers v5+ ships a native DSv4 implementation whose attention
    uses `q_a_proj`/`kv_proj`/`o_a_proj` layout, while DSv4-Flash's
    checkpoint files (and our profile's name bridge) carry
    `wq_a`/`wkv`/`wo_a`. Letting upstream's modeling load would yield a
    skeleton whose live names don't match the safetensors map, so the
    install resolver misses and `set_module_tensor_to_device` falls back
    and crashes with `AttributeError: ... has no attribute 'wq_a'`.
    The earlier `if pkg_name in sys.modules: return` short-circuit
    silently lost this race — replaced by an explicit `_DSV4_REGISTERED`
    sentinel so we always override the upstream slot.
    """
    global _DSV4_REGISTERED
    if _DSV4_REGISTERED:
        return

    pkg_name = "transformers.models.deepseek_v4"

    # Ensure parent package is importable.
    importlib.import_module("transformers.models")

    # Extend transformers' ALLOWED_LAYER_TYPES to accept DSv4's new attention
    # types. PR #45643 lands these in transformers.configuration_utils; on
    # 5.5.4 we monkey-patch the tuple. Idempotent: re-running just rebuilds
    # the same set.
    from transformers import configuration_utils as _cu
    _DSV4_TYPES = ("compressed_sparse_attention", "heavily_compressed_attention")
    _existing = tuple(_cu.ALLOWED_LAYER_TYPES)
    _to_add = tuple(t for t in _DSV4_TYPES if t not in _existing)
    if _to_add:
        _cu.ALLOWED_LAYER_TYPES = _existing + _to_add

    # Register sqrtsoftplus in ACT2FN (used by DSv4's TopKRouter scoring_func).
    # PR #45643 adds this to transformers.activations; we patch it here.
    import torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from transformers.activations import ACT2FN

    class _SqrtSoftplus(_nn.Module):
        def forward(self, x):
            return torch.sqrt(_F.softplus(x))

    if "sqrtsoftplus" not in ACT2FN:
        ACT2FN["sqrtsoftplus"] = _SqrtSoftplus

    vendor_dir = _VENDOR_DIR / "transformers_deepseek_v4"
    if not (vendor_dir / "__init__.py").exists():
        raise RuntimeError(f"vendored DSv4 missing at {vendor_dir}")

    # Purge any cached upstream DSv4 submodules. If any prior call (e.g.
    # `AutoTokenizer.from_pretrained` resolving the tokenizer via the
    # config) loaded upstream's `transformers.models.deepseek_v4` and its
    # submodules, those submodule entries remain in sys.modules after we
    # overwrite the parent slot below. Our subsequent
    # `from transformers.models.deepseek_v4.configuration_deepseek_v4
    # import DeepseekV4Config` would then return the *upstream* class
    # (cached, no `num_hash_layers`) which we'd then register with
    # AutoConfig — pairing upstream config with vendored modeling and
    # crashing on `config.num_hash_layers` at MoE block init.
    _stale_prefix = pkg_name + "."
    for _k in [k for k in sys.modules if k == pkg_name or k.startswith(_stale_prefix)]:
        del sys.modules[_k]

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(vendor_dir / "__init__.py"),
        submodule_search_locations=[str(vendor_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to build import spec for {pkg_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)

    # Register with the auto-classes so AutoConfig / AutoModelForCausalLM work.
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
    AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)

    # Swap DeepseekV4Experts for the per-expert ModuleList variant so
    # prismaquant's per-Linear Fisher hooks observe each routed expert
    # individually. Must run AFTER the modeling module exec_module above
    # but BEFORE any SparseMoeBlock is instantiated. vLLM serving uses
    # the packed PR #40860 format directly; this only affects in-process
    # model construction in the probe path.
    from .dsv4_probe_experts import enable_per_expert_experts
    enable_per_expert_experts()

    _DSV4_REGISTERED = True
