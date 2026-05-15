"""Gemma 4 profile (Google's multimodal family — text + vision + audio).

Covers:
  - Gemma4ForConditionalGeneration (multimodal MoE + dense, all sizes)
  - Gemma4ForCausalLM (text-only)

Almost entirely vLLM-metadata-derived — Gemma 4 has a clean
`packed_modules_mapping` (`qkv_proj`, `gate_up_proj`) and a standard
`hf_to_vllm_mapper` that matches Qwen3.5/3.6's body-prefix convention.
No MTP heads (not in vLLM's speculative registry at this vLLM version),
so PrismaQuant doesn't need a custom MTP forward builder.

Source passthrough prefixes cover the three modality towers (vision,
audio, and their embedding projectors) — these pass through as BF16
until we wire real multimodal calibration, matching the Qwen3.6 visual
encoder policy.

Minimal size: ~30 lines. Everything else inherits from base.
"""
from __future__ import annotations

from .base import ModelProfile


class Gemma4Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"gemma4", "gemma4_text"}:
            return True
        for arch in architectures:
            if arch.startswith("Gemma4"):
                return True
        return False

    @property
    def name(self) -> str:
        return "gemma4"

    def vllm_architecture_class(self) -> str:
        # `Gemma4ForConditionalGeneration` exposes the full multimodal
        # prefix map (vision_tower, audio_tower, embed_vision,
        # embed_audio, language_model). Auto-derived
        # `fused_sibling_group` and `to_vllm_internal_name` inherit
        # from base — no overrides needed.
        return "Gemma4ForConditionalGeneration"

    # ------------------------------------------------------------
    # MoE (only on MoE variants like gemma-4-26b-a4b)
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        return super().packed_expert_param_names()

    def per_expert_moe_regex(self) -> str | None:
        return super().per_expert_moe_regex()

    # ------------------------------------------------------------
    # Source passthrough (multimodal towers stay BF16 for v1)
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        return super().source_passthrough_prefixes()

    def stage_text_only_strip_keys(self) -> tuple[str, ...]:
        return (
            "vision_config", "audio_config", "speech_config",
            "image_token_id", "video_token_id", "audio_token_id",
            "vision_start_token_id", "vision_end_token_id",
        )

    def visual_config_key(self) -> str:
        return "vision_config"

    # `on_disk_expert_qname` intentionally NOT overridden: vLLM's
    # `Gemma4TextModel.load_weights` already runs a substring remap
    # `.experts.{id}.{proj}` → `.moe.experts.{id}.{proj}` (see
    # `vllm.model_executor.models.gemma4.py:1554`). Emitting the HF
    # naming (no `.moe.`) lets vLLM's own remap path land the per-expert
    # tensors correctly on `FusedMoE.w13_weight` / `w2_weight`.
    # Overriding to inject `.moe.` ourselves produces a double `.moe.`
    # after vLLM's remap runs — verified experimentally.

    def live_to_recipe_name(self, live_qname: str) -> str:
        """Gemma 4 loads as Gemma4ForConditionalGeneration at export
        time (multimodal) with body Linears at
        `model.language_model.layers.X.*`. The probe-time recipe uses
        flat `model.layers.X.*` from text-only staging.

        Additionally, the live multimodal model wraps MoE experts
        in a `moe.` submodule (`...layers.X.moe.experts.Y.*`) while
        the HF text-only class exposes them directly
        (`...layers.X.experts.Y.*`). Strip both prefixes to match the
        recipe keys."""
        return super().live_to_recipe_name(live_qname)

    def export_tensor_name(self, model_qname: str) -> str:
        """Keep body/expert export keys in recipe form.

        Gemma 4's vLLM weight iterator performs its own body and
        `.experts` -> `.moe.experts` remaps. Source lookup still uses the
        declarative `recipe_to_source` rules, but export must not pre-apply
        those remaps or vLLM sees doubled `.moe.` prefixes.
        """
        if (
            model_qname.startswith("model.layers.")
            or model_qname.startswith("model.embed_tokens")
            or model_qname.startswith("model.norm")
        ):
            return model_qname
        return super().export_tensor_name(model_qname)

    def stage_text_only_promote_inner_model_type(self) -> bool:
        # `Gemma4ForCausalLM.config: Gemma4TextConfig`. We need
        # `model_type: gemma4_text` on the staged config so AutoConfig
        # picks Gemma4TextConfig (and its flat schema matches the
        # text-only checkpoint tensors' shapes).
        return True

    def visual_layer_prefix(self) -> str:
        # Gemma 4's vision tower uses the HF naming
        # `model.vision_tower.vision_model.encoder.layers.X.*`.
        # We only use this prefix for the probe's extended shard
        # regexes; the actual probe path for visual blocks is
        # deferred pending real multimodal calibration.
        return "model.vision_tower.vision_model.encoder.layers"
