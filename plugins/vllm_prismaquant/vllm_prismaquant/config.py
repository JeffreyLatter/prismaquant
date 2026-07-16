"""``PrismaQuantConfig`` — the vLLM quantization config for the NVFP4-CB /
FP8-CB out-of-tree lane (docs/nvfp4-cb-plan/serving-kernel.md §2, LAYOUT.md §4).

Consumes the ``quantization_config`` that the exporter inlines into
``config.json`` (config_groups + ignore + a ``codebook_file`` pointer). vLLM
auto-detects us from ``quant_method == "prismaquant"`` and calls
``from_config`` with that dict — so everything the plugin needs is in the dict;
no model-path plumbing is required at config time. The shared codebooks are
loaded lazily (once) from the sidecar safetensors, resolving the model dir via
``get_current_vllm_config()`` at weight-load time.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

# vLLM fuses these siblings into one module; the config's packed_modules_mapping
# is not yet populated when get_quant_method runs during layer construction, so
# we carry the standard mapping as a fallback (export guarantees fused siblings
# share one CB scheme, so resolving via any shard is correct).
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


class PrismaQuantConfig(QuantizationConfig):
    """Per-layer dispatch to CB decode / unquantized / (stub) stock-CT."""

    def __init__(self, config_groups: dict, ignore: list[str],
                 codebook_file: str) -> None:
        super().__init__()
        self.config_groups = config_groups
        self.ignore = list(ignore or [])
        self.codebook_file = codebook_file
        # module name -> scheme dict
        self.target_scheme: dict[str, dict] = {}
        for g in config_groups.values():
            sch = g["scheme"]
            for t in g["targets"]:
                self.target_scheme[t] = sch
        self._codebooks: dict[str, torch.Tensor] | None = None

    def __repr__(self) -> str:
        return (f"PrismaQuantConfig(groups={len(self.config_groups)}, "
                f"targets={len(self.target_scheme)})")

    @classmethod
    def get_name(cls):
        return "prismaquant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # NOT fp16-forced (unlike GGUF). CB layers run bf16 activations; the
        # W4A4/W8A8 activation bucket is emulated inside the linear method.
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Blackwell-class target; keep permissive so the correctness prototype
        # is not gated out on the dev box.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PrismaQuantConfig":
        if "config_groups" not in config:
            raise ValueError(
                "prismaquant quantization_config is missing 'config_groups'. "
                "The exporter must inline the full quant config into "
                "config.json['quantization_config'] (see serve export driver).")
        return cls(config["config_groups"], config.get("ignore", []),
                   config.get("codebook_file", "cb_codebooks.pqcb"))

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, **kwargs):
        if user_quant == "prismaquant":
            return "prismaquant"
        if hf_quant_cfg is not None and \
                hf_quant_cfg.get("quant_method") == "prismaquant":
            return "prismaquant"
        return None

    # -- codebook sidecar (loaded once, shared across all layers) ------------
    def get_codebooks(self) -> dict[str, torch.Tensor]:
        if self._codebooks is None:
            from safetensors.torch import load_file
            from vllm.config import get_current_vllm_config
            model_dir = get_current_vllm_config().model_config.model
            path = os.path.join(model_dir, self.codebook_file)
            self._codebooks = load_file(path)
        return self._codebooks

    # -- per-prefix scheme resolution (handles vLLM fused qkv/gate_up) -------
    def _is_ignored(self, prefix: str) -> bool:
        return any(ig in prefix for ig in self.ignore)

    def _scheme_for_prefix(self, prefix: str) -> dict | None:
        if prefix in self.target_scheme:
            return self.target_scheme[prefix]
        leaf = prefix.split(".")[-1]
        pmm = getattr(self, "packed_modules_mapping", {}) or {}
        shard_leaves = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf)
        if shard_leaves:
            schemes = []
            for shard_leaf in shard_leaves:
                sp = prefix[: -len(leaf)] + shard_leaf
                if sp in self.target_scheme:
                    schemes.append(self.target_scheme[sp])
            if schemes:
                # Fused siblings must share the DECODE format (grid/mode/k/…);
                # their per-role codebook_ref legitimately differs (handled by
                # cb_row_offset at apply time), so compare only format keys.
                fmt_keys = ("grid", "mode", "k", "n_sub", "type_size")
                sig = {kk: schemes[0][kk] for kk in fmt_keys}
                for s in schemes[1:]:
                    if {kk: s[kk] for kk in fmt_keys} != sig:
                        raise ValueError(
                            f"fused module {prefix} maps to mixed CB decode "
                            "formats — export union-find should prevent this")
                return schemes[0]
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> "QuantizeMethodBase | None":
        from .linear import PrismaQuantCBLinearMethod

        if isinstance(layer, LinearBase):
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            scheme = self._scheme_for_prefix(prefix)
            if scheme is not None:
                return PrismaQuantCBLinearMethod(self, scheme, prefix)
            # plain NVFP4 / FP8 in a mixed container would delegate to stock
            # compressed-tensors here — not needed for the uniform prototypes.
            raise NotImplementedError(
                f"{prefix!r}: non-CB quantized Linear encountered; stock "
                "compressed-tensors delegation is intentionally unimplemented "
                "in this prototype (uniform CB artifacts only).")
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        # Only remap the ignore list. target_scheme stays keyed by the ORIGINAL
        # per-role HF names (q_proj/k_proj/v_proj/gate_proj/up_proj): the vLLM
        # fusion (q_proj -> qkv_proj) is a *stacked* mapping, which
        # `_scheme_for_prefix` already resolves by expanding the fused prefix
        # back to its shards. Applying the stacked mapper to the keys would
        # collapse the three per-role codebooks into one and return generators.
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
