"""DeepSeek-V4-Flash / -Flash-Base profile.

Covers:
  - DeepseekV4ForCausalLM (671 B params total, 256 routed + 1 shared expert,
    top-k=6, hybrid HCA/CSA attention with compressor + indexer, MTP head,
    hyper-connections, hash-routed first 3 MoE blocks).

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

import torch.nn as nn

from .base import ModelProfile


_DSV4_PACKED_MODULES_FALLBACK = {
    # The wq_a+wq_b and wo_a+wo_b pairs are LoRA-style decompositions
    # (compressed-rank linears), not fused siblings — they sit on opposite
    # sides of a norm/activation, so they DON'T promote to a single fused
    # module. Same for wkv (single, not q/k/v split).
    # Routed experts use w1/w2/w3 in the checkpoint (Mixtral convention),
    # but the live module packs gate+up.
}

# Per-leaf rename: HF live (gate_proj/up_proj/down_proj) ↔ source (w1/w3/w2).
# Used for shared_experts (which still has separate per-leaf modules).
_SHARED_EXPERT_LEAF_RENAME = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}

# Per-prefix rename inside each decoder layer.
_LAYER_INFIX_RENAME = (
    ("self_attn", "attn"),
    ("mlp", "ffn"),
)

# HC modules use compact `hc_attn_*` / `hc_ffn_*` flat keys in the checkpoint
# rather than nested submodule keys. Map e.g.
#   `model.layers.5.attn_hc.base` → `layers.5.hc_attn_base`.
_HC_MODULE_RENAME = (
    ("attn_hc.", "hc_attn_"),
    ("ffn_hc.", "hc_ffn_"),
)


class DeepseekV4Profile(ModelProfile):

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

    # ------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------
    def body_layer_prefix(self) -> str:
        # Checkpoint uses `layers.N.*`, not `model.layers.N.*`.
        return "layers"

    def lm_head_name(self) -> str:
        return "head"

    def mtp_layer_prefix(self) -> str:
        # Single MTP block stored as `mtp.0.*` (no intervening "layers").
        return "mtp"

    def mtp_layer_count(self, cfg: dict) -> int:
        # Honor the standard config field; DSv4-Flash sets it to 1.
        return int(
            cfg.get("num_nextn_predict_layers")
            or cfg.get("num_mtp_layers")
            or 0
        )

    # ------------------------------------------------------------
    # Fused siblings
    # ------------------------------------------------------------
    def fused_sibling_group(self, linear_qname: str) -> str | None:
        # DSv4-Flash has no Q/K/V or gate/up fused siblings at the live
        # Linear level: attention uses the `wkv` combined projection
        # (no separate K/V), Q goes through wq_a + wq_b LoRA, and the
        # routed experts are already packed (gate+up are stored as a
        # single gate_up_proj tensor). Shared experts have separate
        # gate_proj/up_proj/down_proj Linears, but they're not fused
        # in vLLM's compressed-tensors dispatch — return None for all.
        return None

    # ------------------------------------------------------------
    # MoE — packed experts
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        # PR #45643 packs routed experts into two 3D Parameters per
        # SparseMoeBlock: `mlp.experts.gate_up_proj` (E, 2*I, H) and
        # `mlp.experts.down_proj` (E, H, I).
        return frozenset({"gate_up_proj", "down_proj"})

    def per_expert_moe_regex(self) -> str | None:
        # Form vLLM's compressed-tensors dispatcher will see after
        # the container is rebuilt with PR #40860. Adjust if vLLM
        # uses a different expert leaf naming on the dispatch side.
        return (r"re:^model[.]layers[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    def split_packed_experts_for_format(self, fmt: str) -> bool:
        # Mirror Qwen3.5: always emit per-expert Linears at export time
        # so the latching is_fused_expert flag in vLLM's load_weights
        # never trips on mixed-format MoE artifacts.
        return True

    # ------------------------------------------------------------
    # MTP — DSv4-Flash has 1 nextn-predict block
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        return True

    def per_expert_mtp_regex(self) -> str | None:
        return (r"re:^model[.]mtp[.][0-9]+"
                r"[.]mlp[.]experts[.][0-9]+[.](gate|up|down)_proj$")

    def build_mtp_module(self, text_config) -> nn.Module | None:
        # Defer MTP module standup until first need. The transformers
        # vendored class includes MTP wiring inside the main model;
        # we'll subclass and isolate it when the export pipeline
        # actually needs to hook it.
        return None

    # ------------------------------------------------------------
    # Name remap (live transformers → on-disk checkpoint)
    # ------------------------------------------------------------
    def source_tensor_name(self, model_qname: str) -> str:
        """Translate transformers live qnames to DSv4 checkpoint keys.

        DSv4 storage uses a flatter, abbreviated naming:
          - top-level `model.` prefix is dropped
          - `self_attn` → `attn`, `mlp` → `ffn`
          - `lm_head` → `head`, `embed_tokens` → `embed`
          - hyper-connections collapse `attn_hc.X` / `ffn_hc.X` to
            `hc_attn_X` / `hc_ffn_X`
          - shared expert leafs gate_proj/up_proj/down_proj → w1/w3/w2
        """
        name = model_qname

        # Top-level strip + lm_head rename.
        if name == "lm_head.weight":
            return "head.weight"
        if name == "model.embed_tokens.weight":
            return "embed.weight"
        if name == "model.norm.weight":
            return "norm.weight"

        # Drop the leading `model.` prefix for body content.
        if name.startswith("model."):
            name = name[len("model."):]

        # Apply per-decoder-layer infix renames.
        for src, dst in _LAYER_INFIX_RENAME:
            # Only inside `layers.<N>.<infix>` boundary, never as a
            # bare substring elsewhere.
            name = re.sub(
                rf"(^|\.)layers\.(\d+)\.{re.escape(src)}\.",
                rf"\1layers.\2.{dst}.",
                name,
            )

        # Hyper-connection compaction — runs after layer infix.
        for src, dst in _HC_MODULE_RENAME:
            name = name.replace("." + src, "." + dst)
            name = re.sub(rf"(^|\.)layers\.(\d+)\.{re.escape(src)}",
                          rf"\1layers.\2.{dst}", name)

        # Shared expert leaf rename (gate_proj/up_proj/down_proj → w1/w3/w2).
        # Only inside ffn.shared_experts.<leaf>.weight, not the routed
        # experts' packed tensors (which use gate_up_proj/down_proj).
        m = re.match(r"^(.*ffn\.shared_experts\.)([^.]+)(\..*)$", name)
        if m:
            leaf = m.group(2)
            if leaf in _SHARED_EXPERT_LEAF_RENAME:
                name = m.group(1) + _SHARED_EXPERT_LEAF_RENAME[leaf] + m.group(3)

        return name

    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        # Until vLLM container is rebuilt and we wire the dispatch
        # remapper, this is identity — the export step writes
        # checkpoint-form names. Revisit when adding
        # vllm_architecture_class().
        return checkpoint_name

    def live_to_recipe_name(self, live_qname: str) -> str:
        return live_qname

    # ------------------------------------------------------------
    # Source-passthrough hints
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        # Tiny F32 / BF16 params that should never go through quantization:
        # attn_sink, hc_*, ape (compressor abs-pos-embed), tid2eid (hash
        # token-id → expert-id table), norms.
        return (
            "attn_sink",
            "hc_",
            "compressor.ape",
            "tid2eid",
            "kv_norm",
            "q_norm",
            "norm.",
        )
