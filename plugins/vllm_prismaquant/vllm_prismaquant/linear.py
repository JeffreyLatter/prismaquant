"""``PrismaQuantCBLinearMethod`` — weight loading + apply for CB Linears.

Load-time (LAYOUT.md §3): a byte-shaped uint8 ``cb_qweight`` per Linear, an
fp8-only per-output-channel ``weight_scale``, and the model-level shared
``cb_codebook.*`` sidecars (loaded once by the config). Apply: emulate the
served W4A4/W8A8 activation bucket, then the Triton decode-GEMM custom op.

Fused vLLM modules (qkv_proj, gate_up_proj) hold several roles' output rows in
one weight; per-role shared codebooks are concatenated and addressed by a
per-output-row offset (``cb_row_offset``) so fusion stays correct.
"""
from __future__ import annotations

import os

import torch
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

from . import codec
from .expand import expand_cb_to_value, expand_fp4_v2_to_weight
from .ops import cb_gemm

# Fallback fused mapping if the config's packed_modules_mapping is unset.
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

# M-gate for the CB dispatch (GGUF's mmvq_safe pattern, quantization/linear.py
# :34-57): M<=threshold is the decode regime -> keep the bf16-MMA Triton
# decode-GEMM; M>threshold is prefill -> transiently expand FP8_CB to a native
# fp8 tile and hit vLLM's stock W8A8 fp8 GEMM (native tensor cores). NVFP4_CB
# stays on the Triton path either way (transient FP4 needs FP4-MMA, out of
# scope). 16 mirrors the decode/prefill split the decode kernel already tiles at.
# Env-overridable so the prefill A/B (old Triton path vs transient native GEMM)
# is a serve-flag toggle: set PRISMAQUANT_PREFILL_M_THRESHOLD huge to force the
# Triton decode path at prefill (isolates the transient-expansion lever).
PREFILL_M_THRESHOLD = int(os.environ.get("PRISMAQUANT_PREFILL_M_THRESHOLD", "16"))


@register_weight_loader_v2_supported_method
class PrismaQuantCBLinearMethod(LinearMethodBase):
    def __init__(self, quant_config, scheme: dict, prefix: str) -> None:
        self.quant_config = quant_config
        self.scheme = scheme
        self.prefix = prefix
        self.is_fp4 = scheme["grid"] == "fp4"
        self.k = int(scheme["k"])
        self.n_sub = int(scheme["n_sub"])
        self.type_size = int(scheme["type_size"])
        # Two-tier v2 scale coding (fp4 only) — absence of scale_coding ⇒ v1.
        sc = scheme.get("scale_coding")
        if isinstance(sc, dict):
            self.is_v2 = sc.get("kind") == codec.SCALE_CODING_TWO_TIER
            self._sub_table = sc.get("table") or codec.TWO_TIER_SUB_TABLE
        elif isinstance(sc, str):
            self.is_v2 = sc == codec.SCALE_CODING_TWO_TIER
            self._sub_table = codec.TWO_TIER_SUB_TABLE
        else:
            self.is_v2 = False
            self._sub_table = None
        if self.is_v2 and not self.is_fp4:
            raise ValueError(f"{prefix}: two-tier scale coding is fp4-only")

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        K = input_size_per_partition
        if K % codec.SUPERBLOCK != 0:
            raise ValueError(f"{self.prefix}: in_features {K} not a multiple of "
                             f"{codec.SUPERBLOCK}")
        rows = sum(output_partition_sizes)
        row_bytes = (K // codec.SUPERBLOCK) * self.type_size
        layer.logical_widths = list(output_partition_sizes)
        layer._cb_input_size = K

        cb_qweight = ModelWeightParameter(
            data=torch.empty(rows, row_bytes, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("cb_qweight", cb_qweight)

        if not self.is_fp4:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(rows, dtype=torch.float32),
                output_dim=0, weight_loader=weight_loader)
            layer.register_parameter("weight_scale", weight_scale)

    # -- shard-role resolution for a (possibly fused) layer -----------------
    def _shard_roles(self):
        leaf = self.prefix.split(".")[-1]
        pmm = getattr(self.quant_config, "packed_modules_mapping", {}) or {}
        shard_leaves = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf) or [leaf]
        prefixes = [self.prefix[: -len(leaf)] + sl for sl in shard_leaves]
        # Keep only shards that are actual CB targets (all, for uniform arts).
        return [p for p in prefixes if p in self.quant_config.target_scheme]

    def process_weights_after_loading(self, layer):
        dev = layer.cb_qweight.device
        codebooks = self.quant_config.get_codebooks()

        # Build the concatenated flat codebook + per-row base offset.
        shard_prefixes = self._shard_roles()
        widths = layer.logical_widths
        if len(shard_prefixes) != len(widths):
            # Non-fused / single-role Linear.
            shard_prefixes = [self.prefix] if self.prefix in \
                self.quant_config.target_scheme else shard_prefixes
        blocks, row_offsets, cb_total = [], [], None
        offset_rows = 0
        for i, sp in enumerate(shard_prefixes):
            ref = self.quant_config.target_scheme[sp]["codebook_ref"]
            names = ref if isinstance(ref, list) else [ref]
            subs = [codebooks[n].to(dev) for n in names]
            flat = codec.build_flat_codebook(subs)
            if cb_total is None:
                cb_total = flat.numel()
            blocks.append(flat)
            w = widths[i]
            row_offsets.append(torch.full((w,), i * cb_total,
                                          dtype=torch.int32, device=dev))
            offset_rows += w
        cb_flat = torch.cat(blocks).contiguous()
        cb_row_offset = torch.cat(row_offsets).contiguous()

        qw = layer.cb_qweight.data
        layer._cb_qw_padded = codec.pad_qweight(qw)
        layer._cb_flat = cb_flat
        layer._cb_row_offset = cb_row_offset
        dummy = torch.zeros(1, dtype=torch.float32, device=dev)
        if self.is_fp4 and self.is_v2:
            # v2: NO resident fp32 plane (spec §4/G4). The kernel composes the
            # E4M3 scales in-register from the packed 9 bytes via this (256,16)
            # table; the 9-byte plane stays inside cb_qweight.
            layer._cb_compose = codec.build_compose_table(self._sub_table).to(dev)
            layer._cb_scale = dummy
        elif self.is_fp4:
            layer._cb_scale = codec.decode_fp4_scale_plane(qw, self.k).to(dev)
            layer._cb_compose = dummy
        else:
            layer._cb_scale = layer.weight_scale.data.reshape(-1).to(
                torch.float32)
            layer._cb_compose = dummy
        layer._cb_N = qw.shape[0]
        layer._cb_K = layer._cb_input_size

    def apply(self, layer, x, bias=None):
        N, K = layer._cb_N, layer._cb_K
        M = x.reshape(-1, K).shape[0]
        # Decode regime (M small), plus fp4-v1 which has no transient path yet
        # (its v1 e4m3 plane is not composed during expansion) — Triton decode.
        if M <= PREFILL_M_THRESHOLD or (self.is_fp4 and not self.is_v2):
            xq = (codec.fp4_group16_act_qdq(x) if self.is_fp4
                  else codec.fp8_dynamic_act_qdq(x))
            y = cb_gemm(xq, layer._cb_qw_padded, layer._cb_flat,
                        layer._cb_row_offset, layer._cb_scale,
                        layer._cb_compose, N, K, self.k, self.n_sub,
                        self.type_size, self.is_fp4, self.is_v2)
            if bias is not None:
                y = y + bias
            return y

        if self.is_fp4:
            # fp4 v2 prefill: transiently expand to a bf16 weight (value ×
            # composed E4M3 v2 scale) and run one cuBLAS GEMM, amortising the
            # decode over M — the fp4 counterpart of the fp8 transient. INV-1:
            # the [N,K] tile is bounded to one layer, freed per forward. (bf16
            # MMA — INV-2 waived; the FP4-MMA CUTLASS prefill is prototype iii.)
            import torch.nn.functional as F
            xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
            W = expand_fp4_v2_to_weight(
                layer._cb_qw_padded, layer._cb_flat, layer._cb_row_offset,
                layer._cb_compose, N, K, self.k, self.n_sub, self.type_size)
            y = F.linear(xq, W)
            del W
            if bias is not None:
                y = y + bias
            return y

        # FP8_CB prefill (M large): transiently expand THIS layer's packed
        # weight into a native fp8 tile and call vLLM's stock per-channel W8A8
        # fp8 GEMM (native tensor cores), then free the tile. An expanded
        # FP8_CB weight IS a standard per-channel fp8 checkpoint (codebook
        # values on the e4m3 grid; layer.weight_scale per output channel).
        # INV-1: the [N,K] tile is bounded to one layer (expand -> GEMM ->
        # free), never resident/model-wide (the NVINT2 OOM trap). `ops` is
        # imported lazily so the module still imports without vLLM (venv tests).
        import vllm._custom_ops as ops
        W_value = expand_cb_to_value(
            layer._cb_qw_padded, layer._cb_flat, layer._cb_row_offset,
            N, K, self.k, self.n_sub, self.type_size, self.is_fp4)  # [N,K] bf16
        # Lossless: every codebook value is already on the e4m3 grid.
        W_e4m3 = W_value.to(torch.float8_e4m3fn)
        x2 = x.reshape(-1, K)
        xq, sa = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
        # scale_b is the per-output-channel weight scale as [N, 1] (matches
        # vLLM's stock per-channel fp8 scheme; verified against a fp32 dequant
        # reference in tests/test_transient_fp8.py::test_transient_gemm_*).
        ws = layer._cb_scale.reshape(N, 1)
        out = ops.cutlass_scaled_mm(xq, W_e4m3.t(), sa, ws, torch.bfloat16, bias)
        del W_value, W_e4m3
        return out.reshape(*x.shape[:-1], N)
