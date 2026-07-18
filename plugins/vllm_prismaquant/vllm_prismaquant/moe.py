"""``PrismaQuantCBMoEMethod`` — FusedMoE serving for stacked CB expert weights
(docs/nvfp4-cb-plan/moe_cb_design.md §4, LAYOUT.md §3 stacked layout).

Each expert stack ships as ONE tensor per role: ``<q>.cb_qweight`` uint8
``(E, out, (in/256)·type_size)`` (+ fp8 ``<q>.weight_scale`` ``(E, out)``), where
``cb_qweight[e]`` is exactly the dense §1 superblock layout. All experts of a
stack share one format + one codebook (per-layer uniformity, union-find at
export; asserted here). Serving mirrors ``GGUFMoEMethod``: register w13/w2 expert
buffers, then a per-expert **transient** decode (one expert's ``[out, in]`` bf16
tile live at a time — INV-1, the dense transient pattern extended to experts).

  w13 = fused gate_up_proj : (E, 2·inter, hidden)  -> cb_qweight (E, 2·inter, bytes)
  w2  = down_proj          : (E, hidden, inter)    -> cb_qweight (E, hidden, bytes)

NOTE (post-27B GPU/vLLM validation): the FusedMoE weight-loader wiring and the
routed forward are exercised by the synthetic-MoE serve smoke, deferred to the
first idle GPU window (resource-discipline hold). The decode math per expert is
the dense path (bit-exact CPU/triton-tested in test_cb_kernels / test_two_tier_v2);
this file adds the expert-stack loop + buffer mapping. CPU unit tests below pin
the buffer shapes, w13/w2 split, and per-layer uniformity.
"""
from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.utils import set_weight_attrs

from . import codec
from .expand import expand_cb_to_value, expand_fp4_v2_to_weight


def _row_bytes(in_features: int, type_size: int) -> int:
    return (in_features // codec.SUPERBLOCK) * type_size


class PrismaQuantCBMoEMethod(FusedMoEMethodBase):
    """CB decode for RoutedExperts (FusedMoE) — one uniform CB format per layer."""

    def __init__(self, quant_config, moe: FusedMoEConfig, scheme: dict,
                 prefix: str) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.scheme = scheme
        self.prefix = prefix
        self.is_fp4 = scheme["grid"] == "fp4"
        self.k = int(scheme["k"])
        self.n_sub = int(scheme["n_sub"])
        self.type_size = int(scheme["type_size"])
        sc = scheme.get("scale_coding")
        if isinstance(sc, dict):
            self.is_v2 = sc.get("kind") == codec.SCALE_CODING_TWO_TIER
            self._sub_table = sc.get("table") or codec.TWO_TIER_SUB_TABLE
        else:
            self.is_v2 = (sc == codec.SCALE_CODING_TWO_TIER)
            self._sub_table = codec.TWO_TIER_SUB_TABLE if self.is_v2 else None
        if self.is_fp4 and not self.is_v2:
            # fp4-v1 expert transient is a follow-up (no compose-during-expand);
            # export MoE experts as fp8-CB or fp4 two-tier v2.
            raise NotImplementedError(
                f"{prefix}: fp4 MoE experts require two-tier v2 scale coding "
                "(fp4-v1 expert transient not yet implemented)")

    # -- weight buffers (stacked experts) ------------------------------------
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        del params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        E = num_experts
        inter = intermediate_size_per_partition
        layer._cb_hidden = hidden_size
        layer._cb_inter = inter
        # w13 = gate_up: out=2*inter, in=hidden.  w2 = down: out=hidden, in=inter.
        w13 = torch.nn.Parameter(torch.empty(
            E, 2 * inter, _row_bytes(hidden_size, self.type_size),
            dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(w13, {"weight_loader": weight_loader,
                               "is_transposed": False})
        set_weight_attrs(w13, extra_weight_attrs)
        layer.register_parameter("w13_cb_qweight", w13)

        w2 = torch.nn.Parameter(torch.empty(
            E, hidden_size, _row_bytes(inter, self.type_size),
            dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(w2, {"weight_loader": weight_loader,
                              "is_transposed": False})
        set_weight_attrs(w2, extra_weight_attrs)
        layer.register_parameter("w2_cb_qweight", w2)

        if not self.is_fp4:                       # fp8: per-(expert, out) scale
            w13s = torch.nn.Parameter(
                torch.empty(E, 2 * inter, dtype=torch.float32),
                requires_grad=False)
            set_weight_attrs(w13s, {"weight_loader": weight_loader})
            set_weight_attrs(w13s, extra_weight_attrs)
            layer.register_parameter("w13_weight_scale", w13s)
            w2s = torch.nn.Parameter(
                torch.empty(E, hidden_size, dtype=torch.float32),
                requires_grad=False)
            set_weight_attrs(w2s, {"weight_loader": weight_loader})
            set_weight_attrs(w2s, extra_weight_attrs)
            layer.register_parameter("w2_weight_scale", w2s)

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig | None:
        return None

    # -- per-stack codebook / compose + uniformity assert --------------------
    def process_weights_after_loading(self, layer: torch.nn.Module):
        dev = layer.w13_cb_qweight.device
        E = layer.w13_cb_qweight.shape[0]
        codebooks = self.quant_config.get_codebooks()
        ref = self.scheme["codebook_ref"]
        names = ref if isinstance(ref, list) else [ref]
        subs = [codebooks[n].to(dev) for n in names]
        layer._cb_flat = codec.build_flat_codebook(subs)
        layer._cb_row0 = torch.zeros(1, dtype=torch.int32, device=dev)
        if self.is_v2:
            layer._cb_compose = codec.build_compose_table(
                self._sub_table).to(dev)
        else:
            layer._cb_compose = torch.zeros(1, dtype=torch.float32, device=dev)
        # Per-layer uniformity: one format for all experts (union-find at
        # export). The stacked buffer is single-format by construction; assert
        # the byte width matches the scheme so a mis-exported stack fails loudly.
        exp_w13 = _row_bytes(layer._cb_hidden, self.type_size)
        exp_w2 = _row_bytes(layer._cb_inter, self.type_size)
        assert layer.w13_cb_qweight.shape[2] == exp_w13, (
            f"{self.prefix}: w13 byte width {layer.w13_cb_qweight.shape[2]} != "
            f"{exp_w13} (type_size/uniformity mismatch)")
        assert layer.w2_cb_qweight.shape[2] == exp_w2
        layer._cb_E = E

    # -- per-expert decode to a bounded transient [out, in] bf16 -------------
    def _decode_expert(self, layer, which: str, e: int) -> torch.Tensor:
        """Decode ONE expert's CB weight to a bf16 ``[out, in]`` transient
        (INV-1: one expert live at a time). fp8: value × per-channel scale;
        fp4 v2: value × composed group scale."""
        qw = getattr(layer, f"{which}_cb_qweight")[e]          # (out, bytes)
        out = qw.shape[0]
        # w13 in=hidden (gate_up), w2 in=inter (down).
        in_f = layer._cb_hidden if which == "w13" else layer._cb_inter
        qwp = codec.pad_qweight(qw.contiguous())
        row0 = torch.zeros(out, dtype=torch.int32, device=qw.device)
        if self.is_fp4:                                        # fp4 v2
            W = expand_fp4_v2_to_weight(
                qwp, layer._cb_flat, row0, layer._cb_compose,
                out, in_f, self.k, self.n_sub, self.type_size)
        else:                                                  # fp8
            val = expand_cb_to_value(qwp, layer._cb_flat, row0,
                                     out, in_f, self.k, self.n_sub,
                                     self.type_size, is_fp4=False)
            ws = getattr(layer, f"{which}_weight_scale")[e].to(torch.float32)
            W = (val.float() * ws[:, None]).to(torch.bfloat16)
        return W                                               # (out, in) bf16

    def apply(self, layer: RoutedExperts, x: torch.Tensor,
              topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts, shared_experts_input) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input unsupported for CB MoE")
        act = MoEActivation.from_str(layer.activation.value)
        num_tokens = x.shape[0]
        out = torch.zeros_like(x)
        # Grouped by expert: decode each routed expert once (transient), matmul
        # the tokens routed to it, combine with the router weight. bf16 MMA.
        for e in range(layer._cb_E):
            sel = (topk_ids == e)
            if not bool(sel.any()):
                continue
            tok_idx, slot = torch.where(sel)                   # tokens -> expert e
            xe = codec.fp4_group16_act_qdq(x[tok_idx]) if self.is_fp4 \
                else codec.fp8_dynamic_act_qdq(x[tok_idx])
            xe = xe.to(torch.bfloat16)
            W13 = self._decode_expert(layer, "w13", e)         # (2*inter, hidden)
            gate_up = torch.nn.functional.linear(xe, W13)      # (n_e, 2*inter)
            del W13
            d = gate_up.shape[-1] // 2
            a = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype,
                            device=gate_up.device)
            apply_moe_activation(act, a, gate_up)              # silu(gate)*up
            aq = (codec.fp4_group16_act_qdq(a) if self.is_fp4
                  else codec.fp8_dynamic_act_qdq(a)).to(torch.bfloat16)
            W2 = self._decode_expert(layer, "w2", e)           # (hidden, inter)
            oe = torch.nn.functional.linear(aq, W2)            # (n_e, hidden)
            del W2
            oe = oe * topk_weights[tok_idx, slot][:, None].to(oe.dtype)
            out.index_add_(0, tok_idx, oe.to(out.dtype))
        return out
