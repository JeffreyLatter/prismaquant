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

    # `on_disk_expert_qname` intentionally NOT overridden: vLLM's
    # `Gemma4TextModel.load_weights` already runs a substring remap
    # `.experts.{id}.{proj}` → `.moe.experts.{id}.{proj}` (see
    # `vllm.model_executor.models.gemma4.py:1554`). Emitting the HF
    # naming (no `.moe.`) lets vLLM's own remap path land the per-expert
    # tensors correctly on `FusedMoE.w13_weight` / `w2_weight`.
    # Overriding to inject `.moe.` ourselves produces a double `.moe.`
    # after vLLM's remap runs — verified experimentally.

    def init_rotaries(self, rotary, cfg, device, dtype) -> bool:
        """Gemma 4's text rotary is multi-layer-type: it registers one
        ``<layer_type>_inv_freq`` buffer per entry in ``config.layer_types``,
        with *mixed* rope types (e.g. ``sliding_attention``=default,
        ``full_attention``=proportional). The generic single-rope fallback in
        ``_init_rotary_inplace`` calls ``compute_default_rope_parameters(cfg,
        device)`` with no ``layer_type`` → ``KeyError: None`` on
        ``config.rope_parameters[layer_type]`` (issue #6).

        Re-run the rotary's own ``__init__`` on the real device: it rebuilds
        every ``<layer_type>_inv_freq`` / ``<layer_type>_attention_scaling``
        with the correct per-type rope init function (proportional / linear /
        default, plus any per-type kwargs). A hand-rolled
        ``compute_default_rope_parameters`` loop would silently apply the
        *default* formula to the proportional layer and produce wrong
        frequencies."""
        if getattr(rotary, "layer_types", None) is None:
            return False
        if getattr(cfg, "rope_parameters", None) is None:
            return False
        try:
            type(rotary).__init__(rotary, cfg, device=device)
        except Exception:
            return False
        return True

    # ------------------------------------------------------------
    # Cross-layer KV sharing.  Gemma4's last `num_kv_shared_layers`
    # attention layers have no k/v_proj — they reuse the K/V computed by
    # the last non-shared layer of their `layer_type`, passed via a
    # `shared_kv_states` dict the model forward threads through every layer.
    # ------------------------------------------------------------
    def new_forward_pass_state(self) -> dict:
        return {"shared_kv_states": {}}

    def capture_forward_pass_state(self, pass_state: dict):
        """After phase-1's sequential forward, `shared_kv_states[type]` holds
        the (full-length) K/V the shared layers reuse. Snapshot to CPU."""
        skv = (pass_state or {}).get("shared_kv_states") or {}
        out = {}
        for lt, kv in skv.items():
            try:
                out[lt] = tuple(t.detach().to("cpu") for t in kv)
            except Exception:
                pass
        return out

    def isolated_layer_pass_state(self, captured, layer) -> dict:
        """For an isolated (phase-3) layer forward: a shared layer needs its
        type's captured K/V (the attention moves them to the right device
        itself); a non-shared layer just needs a writable dict to store into.
        Always returns a `shared_kv_states` dict so the layer never sees
        `None`."""
        attn = getattr(layer, "self_attn", None)
        if getattr(attn, "is_kv_shared_layer", False) and captured:
            lt = getattr(attn, "layer_type", None)
            kv = captured.get(lt)
            if kv is not None:
                return {"shared_kv_states": {lt: kv}}
        return {"shared_kv_states": {}}

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
