"""Qwen3.5 / Qwen3.6 MoE profile.

Covers:
  - Qwen3_5MoeForConditionalGeneration (multimodal, MoE)
  - Qwen3_5MoeForCausalLM (text-only MoE)
  - Qwen3_5MoeTextModel  (headless)

The canonical ``Qwen/Qwen3.6-35B-A3B`` checkpoint deliberately belongs to
this producer family: its outer config is ``qwen3_5_moe`` with architecture
``Qwen3_5MoeForConditionalGeneration``.  Its routed experts are already
packed as ``gate_up_proj`` / ``down_proj`` tensors (256 experts, top-8), while
the one shared expert per layer remains split as gate/up/down Linears.  Keep
the producer id ``qwen3_5``: that is the id declared by Gridbook's serving
contract, and inventing a release-name ``qwen3_6`` id would make an otherwise
supported checkpoint fail closed at the repository boundary.

The two naming conventions PrismaQuant must juggle:

  | where                    | body                                         |
  |--------------------------|----------------------------------------------|
  | HF multimodal source     | model.language_model.layers.X.*              |
  | vLLM scheme-dispatch     | language_model.model.layers.X.*              |
  | HF text-only / lm_head   | lm_head                                      |
  | vLLM scheme-dispatch     | language_model.lm_head                       |
  | MTP source               | mtp.layers.0.*   (mtp.fc, mtp.norm, ...)     |
  | vLLM MTP scheme-dispatch | mtp.layers.0.*   (IDENTITY — mtp. → model.   |
  |                          |                    remap only at weight-load)|

Visual encoder blocks pass through as BF16 (no real calibration yet).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .base import ModelProfile


class Qwen3_5Profile(ModelProfile):

    # Detection priority (lower = consulted first): the 3.5/3.6 MoE catch-all.
    priority = 110

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5",
            "qwen3_6_moe",
            "qwen3_6_moe_text",
            "qwen3_6",
        }:
            return True
        for arch in architectures:
            if arch.startswith("Qwen3_5") or arch.startswith("Qwen3.5") \
                    or arch.startswith("Qwen3_6") or arch.startswith("Qwen3.6"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3_5"

    def vllm_architecture_class(self) -> str | None:
        """vLLM class to read `packed_modules_mapping` +
        `hf_to_vllm_mapper` from. The base class auto-derives
        `fused_sibling_group()` and the body-part of
        `to_vllm_internal_name()` from these two attributes. We only
        override `to_vllm_internal_name()` below to handle the MTP
        prefix specially."""
        return "Qwen3_5MoeForConditionalGeneration"

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        return True

    def mtp_layer_count(self, cfg: dict) -> int:
        # Use base implementation first.
        n = super().mtp_layer_count(cfg)
        if n > 0:
            return n
        # Qwen3.6 uses `mtp_num_hidden_layers` on text_config. Covered
        # above. If still zero, scan safetensors as a last resort.
        return 0  # caller can scan safetensors separately if desired

    def build_mtp_module(self, text_config) -> nn.Module:
        """Return the Qwen3.5/3.6 MTP replica. See `MtpModule` below;
        `Qwen3_5DenseProfile` inherits this unchanged because
        `MtpModule` picks the dense-vs-MoE decoder class from the
        config at construction time."""
        return MtpModule(text_config)

    def mtp_objective_example(self) -> str:
        return ("CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2}) — "
                "the aux-loss Qwen3.5/3.6 MTP was trained under.")


# ---------------------------------------------------------------------------
# MTP module
#
# Transformers v5 ships no MTP module for these models (the top-level
# PreTrainedModel has `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`,
# so MTP weights are silently dropped on load). MTP is a vLLM-only runtime
# feature. To get real Fisher stats / cost measurements / export on MTP
# Linears we synthesize one here from HF primitives.
#
# Moved verbatim out of the former top-level `prismaquant/mtp_module.py`
# (deleted 2026-07-30, audit R12) so MTP construction goes through the
# profile like every other architecture-specific decision. The generic
# halves — reading `mtp.*` out of safetensors and loading them into the
# module, including the packed-expert fold — now live on `ModelProfile`
# as `read_mtp_source_state_dict()` / `load_mtp_state_dict()`.
# ---------------------------------------------------------------------------

def _build_single_layer_config(text_config):
    """Return a `Qwen3_5MoeTextConfig` (or compatible) with exactly one
    decoder layer of type 'full_attention'. This matches vLLM's MTP:
    one full-attention decoder block per MTP step.

    `copy.deepcopy` is used so the body's config is untouched and
    gradient checkpointing state on the original model doesn't leak."""
    cfg = copy.deepcopy(text_config)
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    return cfg


class MtpModule(nn.Module):
    """Mirrors `vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor`
    but built on HF primitives so Fisher hooks and autograd work normally.

    Satisfies `ModelProfile.build_mtp_module`'s naming contract: wrapped
    in a parent named `mtp`, its parameters come out as `mtp.fc.weight`,
    `mtp.layers.0.self_attn.q_proj.weight`, ... — the recipe names.

    Dense vs MoE is selected from the config at construction time:
    `Qwen3_5MoeDecoderLayer.__init__` eagerly reads `num_experts_per_tok`,
    which dense configs don't define, so we must route to
    `Qwen3_5DecoderLayer` for Qwen3.5/3.6 dense checkpoints."""

    def __init__(self, text_config):
        super().__init__()
        mtp_cfg = _build_single_layer_config(text_config)
        hidden = mtp_cfg.hidden_size
        eps = mtp_cfg.rms_norm_eps

        is_moe = (
            getattr(mtp_cfg, "num_experts", 0)
            or getattr(mtp_cfg, "num_local_experts", 0)
            or getattr(mtp_cfg, "num_experts_per_tok", 0)
        )
        if is_moe:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDecoderLayer as _DecoderLayer,
                Qwen3_5MoeRMSNorm as _RMSNorm,
            )
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer as _DecoderLayer,
                Qwen3_5RMSNorm as _RMSNorm,
            )

        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        self.layers = nn.ModuleList([_DecoderLayer(mtp_cfg, layer_idx=0)])
        self.norm = _RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_hidden = _RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_embedding = _RMSNorm(hidden, eps=eps)

    def forward(self,
                inputs_embeds: torch.Tensor,
                body_hidden_states: torch.Tensor,
                position_embeddings,
                causal_mask,
                position_ids):
        e = self.pre_fc_norm_embedding(inputs_embeds)
        h = self.pre_fc_norm_hidden(body_hidden_states)
        h = torch.cat([e, h], dim=-1)
        h = self.fc(h)
        h = self.layers[0](
            hidden_states=h,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if isinstance(h, tuple):
            h = h[0]
        h = self.norm(h)
        return h
