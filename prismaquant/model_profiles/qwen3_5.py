"""Qwen3.5 / Qwen3.6 MoE profile.

Covers:
  - Qwen3_5MoeForConditionalGeneration (multimodal, MoE)
  - Qwen3_5MoeForCausalLM (text-only MoE)
  - Qwen3_5MoeTextModel  (headless)

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

import torch.nn as nn

from .base import ModelProfile


class Qwen3_5Profile(ModelProfile):

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
        """Mirror vLLM's `Qwen3_5MultiTokenPredictor.forward`:

            e = pre_fc_norm_embedding(inputs_embeds)
            h = pre_fc_norm_hidden(body_hidden_states)
            h = fc(cat([e, h]))
            h = layers[0](h, pos_embeddings, causal_mask)
            h = norm(h)
            return h

        Implemented on HF primitives so Fisher hooks and autograd work
        normally."""
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm,
        )
        mtp_cfg = _single_layer_full_attention_config(text_config)
        hidden = mtp_cfg.hidden_size
        eps = mtp_cfg.rms_norm_eps

        class _MtpModule(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.fc = nn.Linear(hidden * 2, hidden, bias=False)
                self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, layer_idx=0)])
                self.norm = Qwen3_5MoeRMSNorm(hidden, eps=eps)
                self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, eps=eps)
                self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, eps=eps)

            def forward(self, inputs_embeds, body_hidden_states,
                        position_embeddings, causal_mask, position_ids):
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

        import torch
        return _MtpModule(mtp_cfg)

    def mtp_objective_example(self) -> str:
        return ("CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2}) — "
                "the aux-loss Qwen3.5/3.6 MTP was trained under.")


def _single_layer_full_attention_config(text_config):
    """Shallow-copy the text config and pin it to one full-attention
    decoder layer — that's what vLLM's MTP uses."""
    cfg = copy.deepcopy(text_config)
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    return cfg
