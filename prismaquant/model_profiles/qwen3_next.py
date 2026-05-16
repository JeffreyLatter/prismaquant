"""Qwen3-Coder-Next (qwen3_next) profile.

Covers:
  - Qwen3NextForCausalLM

Architecture: hybrid GDN (Gated Delta Networks) + sparse MoE.

  Layer types (from config.layer_types, full_attention_interval=4):
    linear_attention  — DeltaNet SSM layers (layers 0,1,2,4,5,6,...,44,45,46)
    full_attention    — standard GQA layers (layers 3,7,11,...,47)

  Linear attention:
    model.layers.X.linear_attn.in_proj_qkvz   — packed Q/K/V/Z projection
    model.layers.X.linear_attn.in_proj_ba     — packed B/A projection
    model.layers.X.linear_attn.out_proj

  Full attention:
    model.layers.X.self_attn.{q|k|v}_proj  (fused as qkv_proj by vLLM)
    model.layers.X.self_attn.o_proj

  MoE (ALL layers — mlp_only_layers=[]):
    model.layers.X.mlp.experts.gate_up_proj  — 3D packed [E, gate+up, in]
    model.layers.X.mlp.experts.down_proj     — 3D packed [E, in, out]
    model.layers.X.mlp.shared_expert.{gate|up|down}_proj
    model.layers.X.mlp.gate                  — router (small Linear)

  Checkpoint format (per-expert 2D, NOT packed):
    model.layers.X.mlp.experts.N.gate_proj.weight  — shape (moe_intermediate, hidden)
    model.layers.X.mlp.experts.N.up_proj.weight
    model.layers.X.mlp.experts.N.down_proj.weight
    (512 experts per layer, 48 layers = 73,728 per-expert tensors)

  The HF model (Qwen3NextExperts) holds these as 3D nn.Parameter directly:
    gate_up_proj: (512, 1024, 2048)
    down_proj:    (512, 2048, 512)

  The streaming loader packs them via pack_checkpoint_expert_tensors().
  packed_expert_param_names() returns empty because the validate check
  inspects safetensors keys (which are per-expert), not the live model.

  No MTP — no mtp.* tensors in checkpoint.
  No visual encoder.
  Identity naming — no language_model. infix, no hf_to_vllm_mapper.

Differences from Qwen3.5/3.6:
  - Checkpoint has per-expert 2D tensors; live HF model has 3D packed
    Qwen3NextExperts. The streaming loader must pack on the fly.
  - DeltaNet layers (linear_attn.*) carry projections not present in the
    Qwen3.5/3.6 profile; their fused-sibling singletons come from the
    vLLM packed_modules_mapping which lists them as identity mappings.
  - 512 experts per layer (vs 64 for Qwen3.6-35B-A3B). The allocator's
    aggregate_moe_candidates with granularity="layer" (triggered by the
    vllm_qwen3_5_packed_moe allocator profile) collapses them to one DP
    item per projection type per layer.
"""
from __future__ import annotations

import re

from .base import ModelProfile


_QWEN3_NEXT_FALLBACK_PACKED_MODULES = {
    # Full-attention layers
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    # MLP: shared expert gate+up fused
    "gate_up_proj": ["gate_proj", "up_proj"],
    # DeltaNet linear-attention layers — already single-tensor in checkpoint;
    # vLLM lists them as single-element packed mappings (identity) so the
    # allocator treats each as its own singleton sibling group.
    "in_proj_qkvz": ["in_proj_qkvz"],
    "in_proj_ba": ["in_proj_ba"],
}


class Qwen3NextProfile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type == "qwen3_next":
            return True
        for arch in architectures:
            if arch.startswith("Qwen3Next"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3_next"

    def serving_profile_id(self) -> str | None:
        return "vllm_qwen3_5_packed_moe"

    def vllm_architecture_class(self) -> str | None:
        return "Qwen3NextForCausalLM"

    def register_vendored_modeling(self) -> None:
        # DeltaNet (linear_attn) layers use FLA's Triton backward kernel when
        # _flash_linear_attention_available is True. PrismaQuant's
        # ACTIVATION_ROWS_LIMIT slicing produces strided views that violate the
        # kernel's alignment requirements, causing "misaligned address" on the
        # backward pass. Force the eager fallback on all modeling modules that
        # bake in this flag at import time.
        import importlib
        for _mod in (
            "transformers.models.qwen3_5.modeling_qwen3_5",
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            "transformers.models.qwen3_next.modeling_qwen3_next",
        ):
            try:
                m = importlib.import_module(_mod)
                if getattr(m, "_flash_linear_attention_available", False):
                    m._flash_linear_attention_available = False
            except ImportError:
                pass

    def fused_sibling_group(self, linear_qname: str) -> str | None:
        if self._fused_matcher is None:
            self._ensure_vllm_class()
            from .vllm_registry import (
                fused_sibling_matcher_from_packed_mapping,
                packed_modules_mapping_from_class,
            )
            pm = (
                packed_modules_mapping_from_class(self._vllm_cls)
                or _QWEN3_NEXT_FALLBACK_PACKED_MODULES
            )
            self._fused_matcher = fused_sibling_matcher_from_packed_mapping(pm)
        return self._fused_matcher(linear_qname)

    # ----------------------------------------------------------------
    # MoE — per-expert 2D on disk, 3D packed in live HF model
    # ----------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        # The live HF model (Qwen3NextExperts) holds the expert weights as 3D
        # nn.Parameters: gate_up_proj [E, 2·d_ffn, d_model] and
        # down_proj [E, d_model, d_ffn].  These names are what the probe's
        # install_packed_expert_hooks and the export's step-3d use to detect
        # and quantize the packed expert blocks.
        #
        # Note: the CHECKPOINT stores per-expert 2D tensors (experts.N.gate_proj
        # etc.), not the packed 3D tensors.  The validate_native_export check
        # that inspects safetensors keys already handles the empty-set case
        # gracefully (returns True / "profile declares no packed-expert names"),
        # so returning the correct live-model names here does NOT cause a false
        # validate failure.
        return frozenset({"gate_up_proj", "down_proj"})

    def pack_checkpoint_expert_tensors(
        self, layer_prefix: str, tensors: dict
    ) -> dict:
        """Pack 512 per-expert 2D checkpoint tensors into two 3D params.

        The Qwen3Next checkpoint stores each expert separately:
          model.layers.X.mlp.experts.N.{gate|up|down}_proj.weight

        The HF model's Qwen3NextExperts holds 3D nn.Parameters:
          model.layers.X.mlp.experts.gate_up_proj  [E, gate_out+up_out, in]
          model.layers.X.mlp.experts.down_proj     [E, in, out]

        Packing must happen before _fast_install so the install resolver
        (built from named_parameters() on the meta skeleton) can find
        the packed keys.
        """
        import torch

        experts_prefix = f"{layer_prefix}mlp.experts."
        pat = re.compile(
            rf"^{re.escape(experts_prefix)}(\d+)\.(gate|up|down)_proj\.weight$"
        )
        gate_projs: dict[int, torch.Tensor] = {}
        up_projs: dict[int, torch.Tensor] = {}
        down_projs: dict[int, torch.Tensor] = {}
        per_expert_keys: list[str] = []

        for key in tensors:
            m = pat.match(key)
            if m is None:
                continue
            eid = int(m.group(1))
            proj = m.group(2)
            per_expert_keys.append(key)
            if proj == "gate":
                gate_projs[eid] = tensors[key]
            elif proj == "up":
                up_projs[eid] = tensors[key]
            else:
                down_projs[eid] = tensors[key]

        if not gate_projs:
            return tensors

        num_e = max(max(gate_projs), max(down_projs)) + 1
        gate_up = torch.stack(
            [torch.cat([gate_projs[e], up_projs[e]], dim=0)
             for e in range(num_e)],
            dim=0,
        )
        down = torch.stack([down_projs[e] for e in range(num_e)], dim=0)

        per_expert_set = set(per_expert_keys)
        result = {k: v for k, v in tensors.items() if k not in per_expert_set}
        packed = experts_prefix.rstrip(".")
        result[f"{packed}.gate_up_proj"] = gate_up
        result[f"{packed}.down_proj"] = down
        return result

    def per_expert_moe_regex(self) -> str | None:
        # Qwen3Next names are identity (no language_model. prefix remap).
        # At compressed-tensors scheme dispatch the per-expert Linear qnames
        # are:  model.layers.<L>.mlp.experts.<E>.{gate|up|down}_proj
        # This regex is the catch-all so config_groups covers all 512×48
        # expert projections without explicit enumeration.
        return (r"re:^model[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    def split_packed_experts_for_format(self, fmt: str) -> bool:
        # The exporter reads 3D packed params from the live model and must
        # split them into per-expert 2D tensors for vLLM's compressed-tensors
        # weight loader, which expects per-expert layout.
        return True

    # ----------------------------------------------------------------
    # No MTP
    # ----------------------------------------------------------------
    def has_mtp(self) -> bool:
        return False

    def per_expert_mtp_regex(self) -> str | None:
        return None

    # ----------------------------------------------------------------
    # Naming — identity (no language_model. infix, no hf_to_vllm_mapper)
    # ----------------------------------------------------------------
    # to_vllm_internal_name: base-class auto-derives from hf_to_vllm_mapper
    # which is {} for Qwen3NextForCausalLM → identity.  No override needed.

    # live_to_recipe_name: identity.  No override needed.

    # ----------------------------------------------------------------
    # No visual encoder, no passthrough prefixes
    # ----------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        return ()
