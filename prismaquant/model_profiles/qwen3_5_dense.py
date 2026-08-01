"""Qwen3.5 / Qwen3.6 dense profile.

Covers:
  - Qwen3_5ForConditionalGeneration (multimodal, dense MLP, MTP retained)
  - Qwen3_5ForCausalLM (text-only, dense MLP, MTP retained)

Dense variants keep the same hybrid DeltaNet + full-attention layer-mix
and the same MTP head as the MoE sibling, but the per-layer MLP is a
plain gate/up/down Linear stack instead of an experts bank. This
profile inherits body/visual/MTP naming + vLLM remap logic from
Qwen3_5Profile and flips off the MoE-specific hooks so the allocator
and export pipeline treat every MLP as a regular dense Linear.
"""
from __future__ import annotations

from .qwen3_5 import Qwen3_5Profile


class Qwen3_5DenseProfile(Qwen3_5Profile):

    # Detection priority (lower = consulted first): must precede Qwen3_5Profile — the dense arch is a subset of the 3.5/3.6 family.
    priority = 100

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        # Catch the dense arch before the MoE catch-all in Qwen3_5Profile.
        # Dense: Qwen3_5ForConditionalGeneration / Qwen3_5ForCausalLM
        # MoE:   Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM
        if model_type in {"qwen3_5", "qwen3_6"} and not any(
            "Moe" in arch for arch in architectures
        ):
            return True
        for arch in architectures:
            if "Moe" in arch:
                return False
            if arch.startswith("Qwen3_5For") or arch.startswith("Qwen3_6For"):
                return True
            if arch.startswith("Qwen3.5For") or arch.startswith("Qwen3.6For"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3_5_dense"

    def vllm_architecture_class(self) -> str | None:
        # Dense vLLM class. If vLLM doesn't ship it on the host, the base
        # profile falls back to this profile's declarative structure spec.
        return "Qwen3_5ForConditionalGeneration"

    # ------------------------------------------------------------
    # MTP — inherited from Qwen3_5Profile.
    #
    # This profile used to carry a third near-copy of the MTP module
    # (dead: production imported `mtp_module.MtpModule` directly, so
    # nothing but the offline validator ever reached it). `MtpModule`
    # picks `Qwen3_5DecoderLayer` / `Qwen3_5RMSNorm` from the config —
    # exactly what this override built — so inheriting it is what the
    # shipped path already did. Removed 2026-07-30 (audit R12).
    # ------------------------------------------------------------
