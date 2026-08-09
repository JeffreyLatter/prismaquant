"""DeepSeek-V4-Flash / -Flash-Base profile.

Covers:
  - DeepseekV4ForCausalLM as released in DeepSeek-V4-Flash-0731: ~285 B params
    total by checkpoint arithmetic, of which 281,263,734,784 are quantizable
    across 33,325 probeable Linears (probe-measured 2026-08-02, 16x512
    diverse-v1); 43 layers, 256 routed + 1 shared expert, top-k=6, hybrid
    HCA/CSA attention with compressor + indexer, MTP head, hyper-connections,
    hash-routed first 3 MoE blocks. The 671 B figure that previously stood
    here is the DeepSeek-V3-family headline total and does not describe
    Flash; the DeepSeek family ships several differently-sized generations
    and variants, so size claims in this profile are per-checkpoint from the
    probe inventory, never the family headline.

The DSv4 checkpoint uses a non-standard naming convention compared to the
transformers `DeepseekV4Model` live module names. PrismaQuant must bridge
both directions:

  | live (transformers)                     | checkpoint (safetensors)             |
  |-----------------------------------------|--------------------------------------|
  | model.embed_tokens.weight               | embed.weight                         |
  | model.norm.weight                       | norm.weight                          |
  | lm_head.weight                          | head.weight                          |
  | model.layers.N.self_attn.X              | layers.N.attn.X                      |
  | model.layers.N.self_attn.compressor.X   | layers.N.attn.compressor.X           |
  | model.layers.N.mlp.gate.weight          | layers.N.ffn.gate.weight             |
  | model.layers.N.mlp.experts.gate_up_proj | layers.N.ffn.experts.{0..255}.{w1,w3}|
  | model.layers.N.mlp.experts.down_proj    | layers.N.ffn.experts.{0..255}.w2     |
  | model.layers.N.mlp.shared_experts.X     | layers.N.ffn.shared_experts.X        |
  | model.layers.N.attn_hc.{base,fn,scale}  | layers.N.hc_attn_{base,fn,scale}     |
  | model.layers.N.ffn_hc.{base,fn,scale}   | layers.N.hc_ffn_{base,fn,scale}      |
  | model.mtp.0.X                           | mtp.0.X                              |

Also:
  - shared experts use HF-style `gate_proj`/`up_proj`/`down_proj` in live, but
    the checkpoint stores them as `w1`/`w2`/`w3` (Mixtral convention)
  - routed experts are PACKED into `gate_up_proj` (E, 2*I, H) and
    `down_proj` (E, H, I) in the live module; checkpoint stores
    per-expert separately

Important: vLLM main landed DSv4 support today (PR #40860). Once the
container is rebuilt, set `vllm_architecture_class()` to `"DeepseekV4ForCausalLM"`
to enable scheme-dispatch + packed-modules-mapping autoderivation. Until
then we run with `None` and the base class gracefully degrades.
"""
from __future__ import annotations

import re

from .base import ModelProfile


class DeepseekV4Profile(ModelProfile):

    # Detection priority (lower = consulted first): disjoint.
    priority = 170

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"deepseek_v4", "deepseek-v4"}:
            return True
        for arch in architectures:
            if arch.startswith("DeepseekV4") or arch.startswith("DeepSeek-V4"):
                return True
        return False

    @property
    def name(self) -> str:
        return "deepseek_v4"

    def vllm_architecture_class(self) -> str | None:
        # vLLM main has DSv4 (PR #40860 merged 2026-04-27). Once
        # vllm-fresh-b12x:latest is rebuilt, returning the class name
        # here unlocks autoderivation of fused-sibling promotion +
        # name-remapper from vLLM's class-attribute metadata. For now
        # (probe-only path) we return None and rely on the local
        # fallback.
        return None

    def mtp_layer_count(self, cfg: dict) -> int:
        # Honor the standard config field; DSv4-Flash sets it to 1.
        return int(
            cfg.get("num_nextn_predict_layers")
            or cfg.get("num_mtp_layers")
            or 0
        )

    # ------------------------------------------------------------
    # MTP — DSv4-Flash has 1 nextn-predict block, NOT quantized (yet)
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        # The hy_v3 route (audit R12, 2026-07-30). Until DSv4's nextn
        # block is actually quantized, PrismaQuant does not probe, cost
        # or render it: `checkpoint_to_live_name` already drops `mtp.*`
        # so probe/cost/allocator never see it, and the export ships it
        # VERBATIM via `passthrough_prefixes` (`"mtp."`, declared in
        # specs/deepseek_v4.json) so vLLM's nextn spec decode still
        # loads.
        #
        # This replaces a latent contradiction: `has_mtp -> True` with
        # `build_mtp_module -> None` meant the three production MTP
        # sites, which imported the Qwen3.5-specific module directly,
        # would have handed DSv4 a Qwen3.5 decoder layer. Standing up a
        # real DSv4 MTP module is a `build_mtp_module()` override plus
        # flipping this back to True — nothing else has to change.
        return False

    # ------------------------------------------------------------
    # Streaming-probe adapters (refactor #32)
    #
    # Centralizes all DSv4-specific streaming logic that was previously
    # scattered across layer_streaming / streaming_model / incremental_probe.
    # ------------------------------------------------------------

    def checkpoint_to_live_name(self, k: str, *,
                                multimodal: bool = False) -> str | None:
        """DSv4-Flash checkpoint → transformers live qname.

        DSv4's checkpoint uses a flat, abbreviated naming convention
        (`embed.weight`, `layers.5.attn.wkv.weight`) that doesn't match
        the transformers `DeepseekV4ForCausalLM` live module names.
        This is the inverse of `source_tensor_name`.

        Drops:
          - MTP block (`mtp.0.*`) — handled by separate MTP synthesis
          - `.weight_scale_inv` (legacy MiniMax-style FP8 sibling; DSv4
             uses `.scale` siblings handled via fp8_scale_pairs)
          - FP8 block-scale `.scale` siblings of `.weight` keys
            (consumed by the FP8 dequant pass)
          - Compressor + indexer keys (skipped at probe time per the
            modeling patch in vendored/transformers_deepseek_v4)
          - Standalone `.scale` top-level entries with no paired weight
        """
        if k.endswith(".weight_scale_inv"):
            return None
        # DSv4 stores FP8 block-scale siblings as `.scale` (paired with
        # `.weight`). The dequant pass reads these directly via
        # `fp8_scale_pairs`; they must NOT also flow through the body
        # weight map (the live Linear has no `.scale` parameter).
        if k.endswith(".scale"):
            return None
        if k.startswith("mtp."):
            return None
        if k.startswith("hc_head_"):
            # Top-level HyperHead params. Map to model.hc_head.{hc_X}
            # so the multi-stream collapse fires with real weights.
            if k == "hc_head_base":
                return "model.hc_head.hc_base"
            if k == "hc_head_fn":
                return "model.hc_head.hc_fn"
            if k == "hc_head_scale":
                return "model.hc_head.hc_scale"
            return None
        if k in ("head.scale", "embed.scale"):
            return None
        if k == "embed.weight":
            return "model.embed_tokens.weight"
        if k == "head.weight":
            return "lm_head.weight"
        if k == "norm.weight":
            return "model.norm.weight"

        m = re.match(r"^layers\.(\d+)\.(.+)$", k)
        if m:
            layer_idx, leaf = m.group(1), m.group(2)

            # Routed experts → per-expert ModuleList (set up by
            # `enable_per_expert_experts()` in vendored/dsv4_probe_experts).
            m_exp = re.match(r"^ffn\.experts\.(\d+)\.(w1|w2|w3)(.*)$", leaf)
            if m_exp:
                exp_idx, leaf_w, suffix = m_exp.group(1), m_exp.group(2), m_exp.group(3)
                leaf_proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[leaf_w]
                return f"model.layers.{layer_idx}.mlp.experts.{exp_idx}.{leaf_proj}{suffix}"

            # --- PATCH 02: Compressor + indexer — KEEP (faithful) ---
            # Prior drop (return None) was tied to modeling:608-625 probe_mode skip.
            # Faithful forward needs these weights live (model.py:285-440).
            # Proven mapping per port: local_checkpoint_to_live_name (port:138-148):
            #   layers.N.attn.compressor.{wkv,wgate,ape,norm.weight}
            #     → model.layers.N.self_attn.compressor.{wkv,wgate,position_bias,kv_norm.weight}
            #   layers.N.attn.indexer.{wqb,weights_proj} + inner compressor
            # This mirrors DESIGN_NOTES §9 weight-map variant.
            if leaf.startswith("attn.compressor."):
                # ape in checkpoint is position_bias in live (modeling:428)
                rest = leaf[len("attn.compressor."):]
                rest = rest.replace("ape", "position_bias").replace("norm.weight", "kv_norm.weight")
                return f"model.layers.{layer_idx}.self_attn.compressor." + rest
            if leaf.startswith("attn.indexer."):
                sub = leaf[len("attn.indexer."):]
                if sub.startswith("compressor."):
                    rest = sub[len("compressor."):]
                    rest = rest.replace("ape", "position_bias").replace("norm.weight", "kv_norm.weight")
                    return f"model.layers.{layer_idx}.self_attn.indexer.compressor." + rest
                # top-level indexer: wq_b, weights_proj etc
                # checkpoint: layers.N.attn.indexer.wq_b.weight → live: model.layers.N.self_attn.indexer.wq_b.weight
                # but vendored Indexer expects .wq_b / .weights_proj under indexer
                # Also handle legacy compressor.indexer_compressor alias (port:1038)
                return f"model.layers.{layer_idx}.self_attn.indexer." + sub

            # `attn.attn_sink` → `self_attn.sinks` (PR #45643's per-head
            # bias buffer attribute name).
            if leaf == "attn.attn_sink":
                new_leaf = "self_attn.sinks"
            elif leaf.startswith("attn."):
                new_leaf = "self_attn." + leaf[len("attn."):]
            elif leaf.startswith("ffn."):
                new_leaf = "mlp." + leaf[len("ffn."):]
            elif leaf.startswith("attn_norm"):
                # Layer-level pre-attention RMSNorm.
                new_leaf = "input_layernorm" + leaf[len("attn_norm"):]
            elif leaf.startswith("ffn_norm"):
                # Layer-level pre-MLP RMSNorm.
                new_leaf = "post_attention_layernorm" + leaf[len("ffn_norm"):]
            elif leaf.startswith("hc_attn_"):
                new_leaf = "attn_hc." + leaf[len("hc_attn_"):]
            elif leaf.startswith("hc_ffn_"):
                new_leaf = "ffn_hc." + leaf[len("hc_ffn_"):]
            else:
                new_leaf = leaf

            # Shared experts: gate_proj/up_proj/down_proj → w1/w3/w2 in
            # the source. Mirror is done via the regex below; we apply
            # the inverse here (source w1/w3/w2 → live gate/up/down_proj).
            new_leaf = re.sub(
                r"(mlp\.shared_experts)\.w1(\.)", r"\1.gate_proj\2", new_leaf)
            new_leaf = re.sub(
                r"(mlp\.shared_experts)\.w2(\.)", r"\1.down_proj\2", new_leaf)
            new_leaf = re.sub(
                r"(mlp\.shared_experts)\.w3(\.)", r"\1.up_proj\2", new_leaf)

            return f"model.layers.{layer_idx}.{new_leaf}"

        return None

    def fp8_scale_pairs(self, model_path: str
                        ) -> dict[str, tuple[str, str]] | None:
        """DSv4 stores F8_E8M0 `.scale` siblings (not `.weight_scale_inv`).
        Build the `{live_weight_qname: (shard_path, scale_ckpt_key)}`
        map by pairing every `.scale` ckpt key with its `.weight`
        sibling, then routing the weight key through `checkpoint_to_live_name`."""
        import json as _json
        import os as _os
        from safetensors import safe_open as _safe_open

        index_file = _os.path.join(model_path, "model.safetensors.index.json")
        if _os.path.exists(index_file):
            with open(index_file) as f:
                raw = _json.load(f)["weight_map"]
        else:
            single = _os.path.join(model_path, "model.safetensors")
            if not _os.path.exists(single):
                return {}
            with _safe_open(single, framework="pt") as f:
                raw = {k: single for k in f.keys()}

        weight_keys = {k for k in raw if k.endswith(".weight")}
        out: dict[str, tuple[str, str]] = {}
        for ck_key, shard in raw.items():
            if not ck_key.endswith(".scale"):
                continue
            base = ck_key[: -len(".scale")]
            weight_ck = base + ".weight"
            if weight_ck not in weight_keys:
                continue
            weight_live = self.checkpoint_to_live_name(weight_ck)
            if weight_live is None:
                continue
            out[weight_live] = (_os.path.join(model_path, shard), ck_key)
        return out

    def head_resident_extra_prefixes(self, root) -> list[str]:
        """DSv4 has a HyperHead module at `model.hc_head` that collapses
        multi-stream → single-stream before the final norm. Its parameters
        must load with the head-resident batch."""
        if hasattr(root, "model") and hasattr(root.model, "hc_head"):
            return ["model.hc_head."]
        if hasattr(root, "hc_head"):
            return ["hc_head."]
        return []

    def init_rotaries(self, rotary, cfg, device, dtype) -> bool:
        """DSv4 uses Gemma3's multi-layer-type rotary pattern: separate
        `main_inv_freq` and `compress_inv_freq` buffers + matching
        `<name>_attention_scaling` attributes."""
        import torch as _torch
        layer_types = getattr(rotary, "layer_types", None)
        if not (layer_types and getattr(cfg, "rope_parameters", None) is not None):
            return False
        try:
            rope_init_fn = rotary.compute_default_rope_parameters
        except AttributeError:
            return False
        for layer_type in layer_types:
            params = cfg.rope_parameters.get(layer_type)
            if params is None:
                continue
            try:
                inv_freq_lt, scaling_lt = rope_init_fn(
                    cfg, device, layer_type=layer_type)
            except TypeError:
                inv_freq_lt, scaling_lt = rope_init_fn(cfg, device)
            rotary.register_buffer(
                f"{layer_type}_inv_freq",
                inv_freq_lt.to(dtype=_torch.float32, device=device),
                persistent=False,
            )
            rotary.register_buffer(
                f"{layer_type}_original_inv_freq",
                inv_freq_lt.to(dtype=_torch.float32, device=device).clone(),
                persistent=False,
            )
            setattr(rotary, f"{layer_type}_attention_scaling", scaling_lt)
        return True

    def expand_hidden_for_layers(self, hidden, base_model):
        """Expand single-stream `[B, T, H]` to multi-stream
        `[B, T, hc_mult, H]` (mirrors `DeepseekV4Model.forward`)."""
        hc_mult = getattr(base_model.config, "hc_mult", None)
        if hc_mult is None or hc_mult <= 1:
            return hidden
        return hidden.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()

    def collapse_hidden_after_layers(self, hidden, base_model):
        """Collapse multi-stream `[B, T, hc_mult, H]` back to `[B, T, H]`
        via `base_model.hc_head(hidden)` before the final norm."""
        if hidden.dim() == 4 and hasattr(base_model, "hc_head"):
            return base_model.hc_head(hidden)
        return hidden

    def extra_layer_kwargs(self, *, input_ids=None) -> dict:
        """DSv4 hash-routed layers (first `num_hash_layers`) consume
        `input_ids` for the `tid2eid` lookup. Other layers ignore the
        kwarg via **kwargs absorption."""
        return {"input_ids": input_ids} if input_ids is not None else {}

    def register_vendored_modeling(self) -> None:
        """Install the vendored DSv4 modeling code with transformers
        + apply the three monkey-patches (ALLOWED_LAYER_TYPES,
        sqrtsoftplus ACT2FN, per-expert experts swap)."""
        from ..vendored import register_deepseek_v4
        register_deepseek_v4()
