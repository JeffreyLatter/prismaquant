"""Qwen3 (original) profile — dense transformer, no MoE, no MTP, no visual.

Covers:
  - Qwen3ForCausalLM  (Qwen3-0.6B, 1.5B, 4B, 8B, 14B, 32B)

Distinct from Qwen3.5/3.6:
  - No DeltaNet hybrid — full attention on every layer
  - No Multi-Token-Prediction head
  - No MoE (Qwen3 MoE variants use `Qwen3MoeForCausalLM` — separate profile)

vLLM's `Qwen3ForCausalLM` has:
  - packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
                              'gate_up_proj': ['gate_proj', 'up_proj']}
  - no hf_to_vllm_mapper (identity naming)

So the profile is nearly a pass-through: fused-sibling derivation comes
from the base class via packed_modules_mapping, and name remap is the
identity.
"""
from __future__ import annotations

from .base import ModelProfile


class Qwen3Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        # Must distinguish from Qwen3.5/3.6 (which use Qwen3_5For... or
        # Qwen3_6For...) and from Qwen3 MoE (Qwen3MoeForCausalLM).
        if model_type == "qwen3":
            return True
        for arch in architectures:
            # Exact match — anything with a separator (Qwen3_5, Qwen3Moe)
            # belongs to another profile.
            if arch == "Qwen3ForCausalLM":
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3"

    def vllm_architecture_class(self) -> str | None:
        return "Qwen3ForCausalLM"

    def register_vendored_modeling(self) -> None:
        """Install the vendored Qwen3 causal-LM class with cached RoPE."""
        from ..vendored import register_qwen3
        register_qwen3()
