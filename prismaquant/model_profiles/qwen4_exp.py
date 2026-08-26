"""Qwen3.8-Flash-Next (qwen4_exp) profile.

Covers:
  - Qwen4ExpForConditionalGeneration (multimodal, MoE, "experimental preview
    of the architecture that will underpin Qwen4" per the model card)

`transformers` only gained `qwen4_exp` support on 2026-08-26 (merged same
day as this profile was written; first available in the v5.16.1 PyPI
release). No vLLM support exists yet for this architecture, so
`vllm_architecture_class()` returns None — `fused_sibling_group()` /
`packed_expert_param_names()` / naming / `passthrough_prefixes` all fall
back to this profile's declarative twin, `specs/qwen4_exp.json` (see
`ModelProfile.fused_sibling_group`/`packed_expert_param_names`/
`source_passthrough_prefixes` in base.py — each already reads
`structure_spec()` when no vLLM class is available).

Architecture, from the modeling source
(transformers.models.qwen4_exp.modeling_qwen4_exp) and the model card:

  Layer types (config.layer_types, full_attention_interval=4):
    linear_attention  — Gated DeltaNet layers (3 of every 4)
    full_attention    — Qwen Sparse Attention (QSA) layers, with an
                         extra `self_attn.indexer.index_qk_proj` Linear
                         (MQA-style block-selection indexer)

  Every layer (both kinds) carries a SparseMoeBlock:
    mlp.gate                      — router (raw nn.Parameter, not nn.Linear
                                     — already outside the Linear inventory)
    mlp.experts.{gate_up,down}_proj — 3D packed nn.Parameter (512 experts)
    mlp.shared_expert.{gate|up|down}_proj — one dense MLP shared expert

  Every layer also carries TWO "Gated Residual" (hyper-connection) blocks
  (attn_hyper_connection, mlp_hyper_connection), each with
  input_mix_weight_down/up + block_inject_weight Linears — a mechanism with
  no precedent in any other profile in this repo. Left as ordinary
  quantizable Linears (auto-discovered) pending evidence they need pinning.

  Layer 2 (config.ple_layer_ids=[2]) additionally carries a PLE block
  (`ple.key_proj`/`ple.value_proj` Linears + a 20M-row n-gram
  `nn.Embedding` — the embedding is not a Linear so it's already outside
  the quantizable inventory without any profile action).

  MTP: config declares `mtp_num_hidden_layers=1`, but transformers ships no
  MTP module for this class (same situation as Qwen3.5/3.6 — see
  `qwen3_5.py`'s `MtpModule` docstring) and, unlike Qwen3.5/3.6, this
  profile does NOT yet synthesize one: I could not confirm the MTP
  checkpoint's exact tensor names against real weights (the download's
  `model.safetensors.index.json` had not landed yet when this profile was
  written) or find any reference implementation to build the forward pass
  against (no HF module, no vLLM predictor — this architecture is too new
  for either to exist). `has_mtp()` is deliberately left at the base
  default (False); the `mtp.` prefix in this profile's `passthrough_prefixes`
  ships any MTP tensors present as unquantized BF16 passthrough instead of
  guessing a forward pass that could silently mis-measure Fisher/cost on
  those Linears. Revisit once real weights can be inspected.

  Similarly, `pack_checkpoint_expert_tensors` is NOT overridden: whether the
  on-disk checkpoint stores experts pre-packed 3D (needing no packing, most
  likely given the model card claims plain `from_pretrained` compatibility)
  or per-expert 2D (needing Qwen3-Next-style packing) could not be
  confirmed without the safetensors index. The base no-op default is safe
  either way until this is verified against real tensor names.
"""
from __future__ import annotations

from .base import ModelProfile


class Qwen4ExpProfile(ModelProfile):

    # Detection priority (lower = consulted first): distinct model_type
    # "qwen4_exp", no overlap with any other profile's match criteria.
    # 85, not 90 (Qwen3NextProfile's value) — ties in _REGISTERED order are
    # avoidable here since the model_types never collide, so avoid one.
    priority = 85

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"qwen4_exp", "qwen4_exp_text"}:
            return True
        for arch in architectures:
            if arch.startswith("Qwen4Exp"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen4_exp"

    def vllm_architecture_class(self) -> str | None:
        # No vLLM support exists yet for this architecture (added to
        # transformers 2026-08-26). Returning None makes fused_sibling_group
        # / packed_expert_param_names / naming / passthrough_prefixes fall
        # back to this profile's declarative twin, specs/qwen4_exp.json.
        return None
