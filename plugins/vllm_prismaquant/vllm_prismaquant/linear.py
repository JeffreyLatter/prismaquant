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
from .ops import cb_gemm

# Fallback fused mapping if the config's packed_modules_mapping is unset.
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


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
        if self.is_fp4:
            layer._cb_scale = codec.decode_fp4_scale_plane(qw, self.k).to(dev)
        else:
            layer._cb_scale = layer.weight_scale.data.reshape(-1).to(
                torch.float32)
        layer._cb_N = qw.shape[0]
        layer._cb_K = layer._cb_input_size

    def apply(self, layer, x, bias=None):
        if self.is_fp4:
            xq = codec.fp4_group16_act_qdq(x)
        else:
            xq = codec.fp8_dynamic_act_qdq(x)
        y = cb_gemm(xq, layer._cb_qw_padded, layer._cb_flat,
                    layer._cb_row_offset, layer._cb_scale,
                    layer._cb_N, layer._cb_K, self.k, self.n_sub,
                    self.type_size, self.is_fp4)
        if bias is not None:
            y = y + bias
        return y
