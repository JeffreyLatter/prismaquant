"""format_registry.py — extensible quantization format catalog.

A FormatSpec describes everything the pipeline needs to treat a format
uniformly:

  - name                    canonical identifier
  - weight_bits             weight element width
  - group_size              per-group scale granularity; 0 = per-channel
  - weight_element_dtype    torch dtype used on disk
  - scale_bits              bits consumed by per-group scales
  - scale_dtype_name        human-readable scale dtype
  - effective_bits          total bits/param (weight + scale amortized)
  - autoround_config()      dict AutoRound consumes via --layer_config
  - quantize_dequantize(w)  apply RTN in-place (for closed-loop MSE)
  - min_capability_sm       minimum SM arch (useful for hardware filter)

Users register new formats with @register_format. New hardware formats
(e.g. a future MXFP6 variant or Ada W4A8) can be added without touching
core code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

import torch
from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales

from prismaquant.fp8_dynamic import (
    fp8_dynamic_activation_qdq_vllm,
    fp8_dynamic_weight_qdq,
)
from prismaquant.gguf_formats import make_gguf_qdq
from prismaquant.nvfp4_cb_formats import make_nvfp4_cb_qdq
from prismaquant.mx_formats import (
    e8m0_to_scale,
    mxfp8_e4m3_activation_qdq_vllm,
    mxfp8_e4m3_weight_qdq,
)


@dataclass
class FormatSpec:
    name: str
    weight_bits: int
    group_size: int          # 0 means per-channel; >=1 = block size
    scale_bits: int          # bits per scale element
    scale_dtype_name: str    # "fp8_e4m3", "uint8_e8m0", "fp32", ...
    weight_element_dtype: str   # "fp4_e2m1", "fp8_e4m3", "fp6_e3m2", "int4", ...
    scale_block_shape: tuple[int, int] | None = None
    act_bits: int | None = None   # None = no activation quant (W8A16)
    act_dtype_name: str | None = None
    act_group_size: int | None = None
    family: str = "generic"       # "nv", "mx", "int", "fp"
    min_capability_sm: int = 80   # minimum CUDA compute capability
    # The autoround layer_config dict. Leave as callable for lazy config.
    autoround_config: Callable[[], dict] = field(default=lambda: {})
    # RTN quantize+dequantize, returns the rounded tensor (same shape+dtype).
    quantize_dequantize: Callable[[torch.Tensor], torch.Tensor] = field(default=lambda x: x)
    # Optional activation RTN path. Formats with A16 / BF16 activations should
    # leave this as identity; W4A4/W8A8 style formats should provide the
    # matching activation-side quantizer so functional-cost measurement reflects
    # the actual serving bucket rather than weight-only error.
    activation_quantize_dequantize: Callable[[torch.Tensor], torch.Tensor] = field(
        default=lambda x: x
    )

    @property
    def effective_bits(self) -> float:
        """Average bits per parameter accounting for scales."""
        # Backward-compatible fallback when no layer shape is available.
        if self.group_size == 0:
            if self.scale_bits == 0:
                return float(self.weight_bits)
            # True overhead depends on the layer shape. Keep a small, explicit
            # fallback here so older code doesn't crash, but new allocation code
            # should call effective_bits_for_shape().
            return float(self.weight_bits) + 0.02
        if self.scale_block_shape is not None:
            rows, cols = self.scale_block_shape
            return float(self.weight_bits) + float(self.scale_bits) / (rows * cols)
        return float(self.weight_bits) + float(self.scale_bits) / self.group_size

    def scale_count_for_shape(self, shape: tuple[int, ...]) -> int:
        """Return the number of scale values needed for a tensor shape.

        Assumptions:
          - For block/group quantization (group_size > 0), groups are taken
            along the innermost dimension and repeated for every outer row.
          - For per-channel formats (group_size == 0), one scale is used per
            output channel / row, which for Linear weights is shape[0].
        """
        if len(shape) == 0:
            return 0
        if self.scale_bits == 0:
            return 0
        if self.scale_block_shape is not None:
            if len(shape) < 2:
                n_params = int(shape[0])
                rows, cols = self.scale_block_shape
                return math.ceil(n_params / (rows * cols))
            rows, cols = self.scale_block_shape
            outer = int(math.prod(shape[:-2])) if len(shape) > 2 else 1
            return (
                outer
                * math.ceil(int(shape[-2]) / rows)
                * math.ceil(int(shape[-1]) / cols)
            )
        if self.group_size == 0:
            return int(shape[0]) if len(shape) >= 1 else 1
        if len(shape) == 1:
            n_params = int(shape[0])
            return math.ceil(n_params / self.group_size)
        outer = int(math.prod(shape[:-1]))
        inner = int(shape[-1])
        return outer * math.ceil(inner / self.group_size)

    def memory_bytes_for_shape(self, shape: tuple[int, ...]) -> int:
        """Exact-ish serialized size for a tensor in this format."""
        n_params = int(math.prod(shape)) if len(shape) else 1
        weight_bytes = math.ceil(n_params * self.weight_bits / 8.0)
        scale_bytes = math.ceil(self.scale_count_for_shape(shape) * self.scale_bits / 8.0)
        return weight_bytes + scale_bytes

    def effective_bits_for_shape(self, shape: tuple[int, ...]) -> float:
        n_params = int(math.prod(shape)) if len(shape) else 1
        return 8.0 * self.memory_bytes_for_shape(shape) / max(n_params, 1)


REGISTRY: dict[str, FormatSpec] = {}
FORMAT_ALIASES: dict[str, str] = {
    # Historical artifacts and launchers used the short OCP-MX default name.
    # Keep it accepted at every input boundary, but normalize persisted solver
    # and measurement output to the explicit FP8 variant.
    "MXFP8": "MXFP8_E4M3",
    # User-facing production alias for vLLM FP8 dynamic quantization:
    # per-output-row FP32 weight scales and per-token dynamic activation
    # scales, serialized as compressed-tensors float-quantized FP8_E4M3.
    "FP8": "FP8_E4M3",
    "FP8_DYNAMIC": "FP8_E4M3",
}


def register_format(spec: FormatSpec) -> FormatSpec:
    REGISTRY[spec.name] = spec
    return spec


def canonical_format_name(name: str) -> str:
    raw = str(name).strip()
    if raw in FORMAT_ALIASES:
        return FORMAT_ALIASES[raw]
    if raw in REGISTRY:
        return raw
    upper = raw.upper()
    if upper in FORMAT_ALIASES:
        return FORMAT_ALIASES[upper]
    if upper in REGISTRY:
        return upper
    return raw


def aliases_for(name: str) -> tuple[str, ...]:
    canonical = canonical_format_name(name)
    aliases = [alias for alias, target in FORMAT_ALIASES.items()
               if target == canonical]
    return (canonical, *aliases)


# -----------------------------------------------------------------------
# Reference format implementations
# -----------------------------------------------------------------------
# Helpers for RTN quantization reference impls.

def _rtn_uniform_int(w: torch.Tensor, bits: int, group_size: int,
                     symmetric: bool = True) -> torch.Tensor:
    """Round-to-nearest uniform-integer quantizer with optional group scaling."""
    orig_shape = w.shape
    out_f, in_f = w.shape[-2], w.shape[-1]
    w2 = w.reshape(-1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(-1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(1)
    # Per-group max for scale
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    if symmetric:
        levels = (1 << (bits - 1)) - 1
        scale = max_abs / levels
        q = torch.round(w2 / scale).clamp(-levels - 1, levels)
        w_rec = q * scale
    else:
        levels = (1 << bits) - 1
        w_min = w2.amin(dim=-1, keepdim=True)
        w_max = w2.amax(dim=-1, keepdim=True)
        scale = (w_max - w_min) / levels
        zp = torch.round(-w_min / scale.clamp_min(1e-8))
        q = torch.round(w2 / scale.clamp_min(1e-8) + zp).clamp(0, levels)
        w_rec = (q - zp) * scale
    return w_rec.reshape(orig_shape).to(w.dtype)


def _mx_rounded_amax_power2(amax: torch.Tensor) -> torch.Tensor:
    """Round a block amax to the MX scale power-of-two grid."""
    x = amax.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    raw = x.view(torch.int32).to(torch.int64)
    val_to_add = 1 << (23 - 1 - 1)
    sign_exponent_mask = ((1 << (8 + 1)) - 1) << 23
    rounded = torch.bitwise_and(raw + val_to_add, sign_exponent_mask)
    return rounded.to(torch.int32).view(torch.float32)


def _snap_scale_e8m0(
    scale: torch.Tensor,
    *,
    element_max: torch.Tensor,
    num_bits: int | None = None,
) -> torch.Tensor:
    """Snap a real-valued per-group scale to the served MX E8M0 grid.

    The OCP MX spec encodes the per-block scale as an 8-bit E8M0 value:
    unsigned, exponent-only, range 2^(-127) to 2^127. Representable
    values are exactly powers of two. compressed-tensors derives MXFP4/MXFP8
    weight scales by rounding the block amax to a power of two, then
    subtracting the element-format exponent offset.

    For NV (non-MX) formats, scales are FP8 and effectively continuous;
    no snapping is applied.
    """
    element_max_f = element_max.to(device=scale.device, dtype=torch.float32)
    amax = scale.to(torch.float32) * element_max_f
    if num_bits in {4, 8}:
        e8m0 = generate_mx_scales(amax, num_bits=num_bits).to(torch.uint8)
        return e8m0_to_scale(e8m0, device=scale.device)

    # compressed-tensors only defines MX scale generation for FP4 E2M1 and
    # FP8 E4M3. Keep the local fallback for research-only FP6/E5M2 variants.
    rounded = _mx_rounded_amax_power2(amax)
    element_offset = torch.floor(torch.log2(element_max_f))
    snapped_exp = (
        torch.floor(torch.log2(rounded)) - element_offset
    ).clamp(-127.0, 127.0)
    return torch.pow(2.0, snapped_exp)


def _rtn_fp_codebook(w: torch.Tensor, codebook: torch.Tensor,
                     group_size: int, mx_scale: bool = False,
                     mx_num_bits: int | None = None) -> torch.Tensor:
    """Round to nearest value in a small FP codebook, with per-group scaling.

    Vectorized via torch.bucketize on the sorted codebook. For each scaled
    weight value x, we binary-search the codebook to find the two bracketing
    entries and pick the closer one. O(N log K) instead of the O(N * K)
    pairwise-distance approach, with 0 extra-dim allocations.

    When `mx_scale=True`, the per-group scale is snapped to the nearest
    power of two (E8M0). This matches the OCP MX serving path; without
    it, RTN error for MX formats is slightly under-estimated.
    """
    orig_shape = w.shape
    in_f = w.shape[-1]
    w2 = w.reshape(-1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(-1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(1)

    cb = _codebook_on_device(
        codebook,
        device=w2.device,
        dtype=torch.float32,
    )
    cmax = cb.abs().max()
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / cmax
    if mx_scale:
        scale = _snap_scale_e8m0(
            scale,
            element_max=cmax,
            num_bits=mx_num_bits,
        )
    x = w2 / scale                                    # shape (..., group)

    # Bucketize returns the insertion index: cb[idx-1] <= x < cb[idx].
    idx = torch.bucketize(x.contiguous(), cb)
    idx_lo = (idx - 1).clamp_min(0)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    lo = cb[idx_lo]
    hi = cb[idx_hi]
    choose_hi = (hi - x).abs() < (x - lo).abs()
    q = torch.where(choose_hi, hi, lo)

    w_rec = q * scale
    return w_rec.reshape(orig_shape).to(w.dtype)


# FP codebooks
def _e2m1_codebook() -> torch.Tensor:
    # 4-bit: 1 sign + 2 exp + 1 mantissa.  Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    signed = [0.0] + [+v for v in vals[1:]] + [-v for v in vals[1:]]
    return torch.tensor(sorted(set(signed)), dtype=torch.float32)


def _e3m2_codebook() -> torch.Tensor:
    # 6-bit FP e3m2: 1s + 3 exp + 2 mantissa. 64 codes.
    codes = set([0.0])
    for exp in range(8):
        for m in range(4):
            if exp == 0:
                val = (m / 4) * (2 ** -2)
            else:
                val = (1 + m / 4) * (2 ** (exp - 3))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


def _e2m3_codebook() -> torch.Tensor:
    # 6-bit FP e2m3: 1s + 2 exp + 3 mantissa. 64 codes.
    codes = set([0.0])
    for exp in range(4):
        for m in range(8):
            if exp == 0:
                val = (m / 8) * (2 ** 0)
            else:
                val = (1 + m / 8) * (2 ** (exp - 1))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


def _e4m3_codebook() -> torch.Tensor:
    # 8-bit FP e4m3 (no inf). 256 codes, covering ±448.
    # We compute it analytically from the OCP FP8 spec (nan reserved).
    codes = set([0.0])
    for exp in range(16):
        for m in range(8):
            if exp == 0:
                val = (m / 8) * (2 ** -6)  # subnormals
            else:
                val = (1 + m / 8) * (2 ** (exp - 7))
            codes.add(+val); codes.add(-val)
    # Clip to ±448 per spec (remove overflowing exponents)
    return torch.tensor(sorted(c for c in codes if abs(c) <= 448.0),
                        dtype=torch.float32)


def _e5m2_codebook() -> torch.Tensor:
    # 8-bit FP e5m2. Wider range, less mantissa precision.
    codes = set([0.0])
    for exp in range(31):
        for m in range(4):
            if exp == 0:
                val = (m / 4) * (2 ** -14)
            else:
                val = (1 + m / 4) * (2 ** (exp - 15))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


_CODEBOOKS = {
    "fp4_e2m1": _e2m1_codebook(),
    "fp6_e3m2": _e3m2_codebook(),
    "fp6_e2m3": _e2m3_codebook(),
    "fp8_e4m3": _e4m3_codebook(),
    "fp8_e5m2": _e5m2_codebook(),
}

_CODEBOOK_DEVICE_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _codebook_on_device(
    codebook: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if codebook.device == device and codebook.dtype == dtype:
        return codebook.contiguous()
    key = (id(codebook), str(device), dtype)
    cached = _CODEBOOK_DEVICE_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached
    source = codebook
    if (
        torch.device(device).type == "cuda"
        and codebook.device.type == "cpu"
        and torch.cuda.is_available()
        and not codebook.is_pinned()
    ):
        try:
            source = codebook.pin_memory()
        except RuntimeError:
            source = codebook
    cached = source.to(
        device=device,
        dtype=dtype,
        non_blocking=bool(source.device.type == "cpu" and source.is_pinned()),
    ).contiguous()
    _CODEBOOK_DEVICE_CACHE[key] = cached
    return cached


def _make_rtn(codebook_name: str, group_size: int, mx_scale: bool = False):
    """Return a closure that runs RTN through ``codebook`` at ``group_size``.

    The hot path is wrapped in ``torch.compile`` (mode='reduce-overhead',
    dynamic=False) by default — micro-benchmark on Blackwell + cu130 +
    torch 2.11 shows ~10x speedup on per-Linear activation RTN
    (12 ms eager → 1.2 ms compiled).  Compiled-vs-eager is MSE-identical
    but NOT bit-identical: at exact codebook midpoints Inductor fusion
    can flip the tie (~0.036% of bf16 elements pick the other equidistant
    code), so screens that require bit-reproducibility across the
    compiled/eager boundary must pin one path.  This matters most on the
    polish hot path where the closure is called once per Linear per
    forward, ~497 calls per measurement on 27B, several hundred
    measurements per polish pass.

    Set ``PRISMAQUANT_DISABLE_RTN_COMPILE=1`` to fall back to eager —
    only useful if torch.compile fails on a particular tensor shape
    or kernel mismatch, which has not been observed but might surface
    on older torch / non-Blackwell hardware.

    The codebook device-resolution is kept *outside* the compiled
    function: dynamo cannot trace ``codebook.pin_memory()`` (NYI in
    the Inductor backend), and compiling around it would force a
    graph break on every call.  Instead the closure resolves the
    device-resident codebook eagerly, then passes it as a positional
    argument to the compiled inner.
    """
    cb_cpu = _CODEBOOKS[codebook_name]
    mx_num_bits = {"fp4_e2m1": 4, "fp8_e4m3": 8}.get(codebook_name)

    # Inner function takes a pre-resolved on-device codebook so the
    # compile can trace cleanly.  Functionally equivalent to
    # _rtn_fp_codebook except we skip the device-resolution call
    # (caller already did it).
    def _inner_eager(
        w: torch.Tensor, cb: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = w.shape
        in_f = w.shape[-1]
        w2 = w.reshape(-1, in_f).float()
        if group_size > 0 and group_size < in_f:
            w2 = w2.reshape(-1, in_f // group_size, group_size)
        else:
            w2 = w2.unsqueeze(1)
        cmax = cb.abs().max()
        max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = max_abs / cmax
        if mx_scale:
            scale = _snap_scale_e8m0(
                scale,
                element_max=cmax,
                num_bits=mx_num_bits,
            )
        x = w2 / scale
        idx = torch.bucketize(x.contiguous(), cb)
        idx_lo = (idx - 1).clamp_min(0)
        idx_hi = idx.clamp_max(cb.numel() - 1)
        lo = cb[idx_lo]
        hi = cb[idx_hi]
        choose_hi = (hi - x).abs() < (x - lo).abs()
        q = torch.where(choose_hi, hi, lo)
        w_rec = q * scale
        return w_rec.reshape(orig_shape).to(w.dtype)

    import os as _os
    if group_size == 0:
        # The compiled closure captures group_size.  On torch 2.9/NVIDIA
        # 25.10, compiling grouped RTN first (for NV/MX) and then compiling
        # this per-token/plain-FP8 variant can make Dynamo reuse the wrong
        # specialization and raise division-by-zero inside Inductor.  Keep
        # plain FP8 eager so activation-aware render scores cannot silently
        # degrade to raw weight/output MSE.
        _inner = _inner_eager
    elif _os.environ.get(
        "PRISMAQUANT_DISABLE_RTN_COMPILE", "",
    ).strip().lower() in {"1", "true", "yes", "on"}:
        _inner = _inner_eager
    else:
        # ``dynamic=True`` produces one compiled kernel that handles
        # every (batch, hidden) shape via symbolic shape specialization.
        # ``dynamic=False`` would per-shape-recompile and quickly hit
        # dynamo's default recompile_limit=8, after which every new
        # shape silently falls back to eager.  At polish time we see
        # 30+ unique Linear input shapes on a single 27B model, so the
        # symbolic-shape kernel is the right trade-off (slightly less
        # aggressive optimization in exchange for no recompile thrash).
        # Raise the recompile limit defensively in case dynamic
        # specialization still triggers a recompile path.
        try:
            torch._dynamo.config.recompile_limit = max(
                int(getattr(torch._dynamo.config, "recompile_limit", 8)),
                256,
            )
        except Exception:
            pass
        _inner = torch.compile(_inner_eager, dynamic=True)

    def f(w: torch.Tensor) -> torch.Tensor:
        cb = _codebook_on_device(
            cb_cpu, device=w.device, dtype=torch.float32,
        )
        return _inner(w, cb)

    return f


def _mxfp8_e4m3_activation_vllm_rtn(x: torch.Tensor) -> torch.Tensor:
    """Match vLLM/compressed-tensors dynamic MXFP8 activation quantization."""
    return mxfp8_e4m3_activation_qdq_vllm(x).dequant.to(x.dtype)


def _mxfp8_e4m3_weight_rtn(w: torch.Tensor) -> torch.Tensor:
    """Renderer-side MXFP8_E4M3 weight RTN matching exported metadata."""
    return mxfp8_e4m3_weight_qdq(w).dequant.to(w.dtype)


def _make_plain_fp8_weight_rtn(
    element_dtype: torch.dtype,
    element_max: float,
):
    """Plain FP8 per-output-channel weight QDQ."""
    def f(w: torch.Tensor) -> torch.Tensor:
        return fp8_dynamic_weight_qdq(
            w,
            element_dtype=element_dtype,
            element_max=element_max,
        ).dequant.to(w.dtype)

    return f


def _make_plain_fp8_activation_vllm_rtn(
    element_dtype: torch.dtype,
    element_max: float,
):
    """vLLM dynamic per-token FP8 activation QDQ."""

    def f(x: torch.Tensor) -> torch.Tensor:
        return fp8_dynamic_activation_qdq_vllm(
            x,
            element_dtype=element_dtype,
            element_max=element_max,
        ).dequant.to(x.dtype)

    return f


# -----------------------------------------------------------------------
# Built-in format registrations
# -----------------------------------------------------------------------
# AutoRound layer_config entries match what AutoRound expects for its
# internal QuantizationScheme.  See auto_round.compressors.utils for the
# canonical fields.  Feel free to extend as new formats are added.

def _nv_autoround(bits=4, gsize=16, act_bits=4):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="nv_fp",
        act_bits=act_bits, act_group_size=gsize, act_sym=True,
        act_data_type="nv_fp4_with_static_gs" if bits == 4 else "nv_fp",
        act_dynamic=True,
    )


def _mx_autoround(bits=8, gsize=32, act_bits=8, elt="fp8_e4m3"):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="mx_fp",
        weight_element_dtype=elt,
        act_bits=act_bits, act_group_size=gsize, act_sym=True,
        act_data_type="mx_fp", act_element_dtype=elt, act_dynamic=True,
    )


def _int_autoround(bits, gsize, act_bits=16):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="int",
        act_bits=act_bits, act_group_size=gsize if act_bits <= 8 else 0,
        act_sym=True, act_data_type="int" if act_bits <= 8 else "float",
        act_dynamic=True,
    )


def _plain_fp8_autoround(elt="fp8_e4m3", act_bits=8):
    # Plain per-channel FP8 (no microscaling).  AutoRound's "fp8_e4m3" /
    # "fp8_e5m2" dtypes are represented in PrismaQuant layer_config as
    # group_size=0 (non-grouped) with per-token dynamic activation scaling.
    # The exporter maps this to compressed-tensors' native FP8 scheme.
    return dict(
        bits=8, group_size=0, sym=True, data_type=elt,
        act_bits=act_bits, act_group_size=0, act_sym=True,
        act_data_type=elt if act_bits == 8 else "float",
        act_dynamic=True,
    )


def _nvfp4_export_aligned_rtn(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 WEIGHT RTN matching the export/compressed-tensors convention.

    Routes through the export codec so registry weight emulation and the
    shipped bytes share one rendering (the resident-vs-served mismatch
    class). Last dims that are not a multiple of the group size are
    zero-padded then sliced back — zeros cannot perturb a max-abs group
    scale, so padding is exact for the real columns.

    WEIGHTS ONLY: the export codec derives a per-tensor global scale from
    the tensor it is given; for activations that would make the emulation
    batch-dependent, while serve-time activation quantization uses a
    STATIC input_global_scale fit at calibration. Activation emulation
    stays on the per-group dynamic RTN (see the registration below) until
    a static-scale-aware emulation exists.
    """
    from . import export_native_compressed as enc

    orig_shape = x.shape
    in_features = int(orig_shape[-1])
    flat = x.reshape(-1, in_features).to(torch.float32)
    pad = (-in_features) % 16
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    out = enc._rtn_dequant_nvfp4(flat, group_size=16)
    if pad:
        out = out[:, :in_features]
    return out.reshape(orig_shape).to(x.dtype)


# NVFP4 / NVFP4A16  (NVIDIA, group_size=16, FP8 scales)
register_format(FormatSpec(
    name="NVFP4",
    weight_bits=4, group_size=16, scale_bits=8, scale_dtype_name="fp8_e4m3",
    weight_element_dtype="fp4_e2m1", act_bits=4, act_dtype_name="fp4_e2m1",
    act_group_size=16, family="nv", min_capability_sm=100,
    autoround_config=lambda: _nv_autoround(4, 16, 4),
    quantize_dequantize=_nvfp4_export_aligned_rtn,
    # activations: per-group dynamic RTN, NOT the export codec — see
    # _nvfp4_export_aligned_rtn docstring (batch-dependence).
    activation_quantize_dequantize=_make_rtn("fp4_e2m1", 16),
))
register_format(FormatSpec(
    name="NVFP4A16",
    weight_bits=4, group_size=16, scale_bits=8, scale_dtype_name="fp8_e4m3",
    weight_element_dtype="fp4_e2m1", act_bits=None,
    family="nv", min_capability_sm=100,
    autoround_config=lambda: _nv_autoround(4, 16, 16),
    quantize_dequantize=_nvfp4_export_aligned_rtn,
    activation_quantize_dequantize=lambda x: x,
))

# MXFP4 / MXFP8_E4M3 / MXFP8_E5M2 / MXFP6 variants
# (OCP MX, group_size=32, E8M0 scales)
# All MX formats use mx_scale=True so RTN models the actual E8M0 power-of-two
# per-block scale used by the serving path. Without this the measured RTN
# error would be slightly optimistic vs what the kernel actually produces.
register_format(FormatSpec(
    name="MXFP4",
    weight_bits=4, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp4_e2m1", act_bits=4, act_dtype_name="fp4_e2m1",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(4, 32, 4, "fp4_e2m1"),
    quantize_dequantize=_make_rtn("fp4_e2m1", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp4_e2m1", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP6_E3M2",
    weight_bits=6, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp6_e3m2", act_bits=6, act_dtype_name="fp6_e3m2",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(6, 32, 6, "fp6_e3m2"),
    quantize_dequantize=_make_rtn("fp6_e3m2", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp6_e3m2", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP6_E2M3",
    weight_bits=6, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp6_e2m3", act_bits=6, act_dtype_name="fp6_e2m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(6, 32, 6, "fp6_e2m3"),
    quantize_dequantize=_make_rtn("fp6_e2m3", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp6_e2m3", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP8_E4M3",  # explicit name for the canonical variant
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=8, act_dtype_name="fp8_e4m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e4m3"),
    quantize_dequantize=_mxfp8_e4m3_weight_rtn,
    activation_quantize_dequantize=_mxfp8_e4m3_activation_vllm_rtn,
))
register_format(FormatSpec(
    name="MXFP8_E5M2",  # wider dynamic range, less mantissa precision
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e5m2", act_bits=8, act_dtype_name="fp8_e5m2",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e5m2"),
    quantize_dequantize=_make_rtn("fp8_e5m2", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp8_e5m2", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP8A16",
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=None,
    family="mx", min_capability_sm=80,  # W8A16 works on Marlin
    autoround_config=lambda: _mx_autoround(8, 32, 16, "fp8_e4m3"),
    quantize_dequantize=_mxfp8_e4m3_weight_rtn,
    activation_quantize_dequantize=lambda x: x,
))

# Plain FP8 (per-output-channel FP32 scale on weights, no microscaling).
# vLLM-native serving path; works on Hopper (sm_90) and Blackwell (sm_100+).
register_format(FormatSpec(
    name="FP8_E4M3",
    weight_bits=8, group_size=0, scale_bits=32, scale_dtype_name="fp32",
    weight_element_dtype="fp8_e4m3", act_bits=8, act_dtype_name="fp8_e4m3",
    act_group_size=0, family="fp", min_capability_sm=90,
    autoround_config=lambda: _plain_fp8_autoround("fp8_e4m3", 8),
    quantize_dequantize=_make_plain_fp8_weight_rtn(
        torch.float8_e4m3fn, 448.0,
    ),
    activation_quantize_dequantize=_make_plain_fp8_activation_vllm_rtn(
        torch.float8_e4m3fn, 448.0,
    ),
))
register_format(FormatSpec(
    name="FP8_E5M2",
    weight_bits=8, group_size=0, scale_bits=32, scale_dtype_name="fp32",
    weight_element_dtype="fp8_e5m2", act_bits=8, act_dtype_name="fp8_e5m2",
    act_group_size=0, family="fp", min_capability_sm=90,
    autoround_config=lambda: _plain_fp8_autoround("fp8_e5m2", 8),
    quantize_dequantize=_make_plain_fp8_weight_rtn(
        torch.float8_e5m2, 57344.0,
    ),
    activation_quantize_dequantize=_make_plain_fp8_activation_vllm_rtn(
        torch.float8_e5m2, 57344.0,
    ),
))

# INT8 per-channel / INT4 per-group
register_format(FormatSpec(
    name="INT8_W8A16",
    weight_bits=8, group_size=0, scale_bits=16, scale_dtype_name="fp16",
    weight_element_dtype="int8",
    family="int", min_capability_sm=70,
    autoround_config=lambda: _int_autoround(8, -1, 16),
    quantize_dequantize=lambda w: _rtn_uniform_int(w, 8, 0),
    activation_quantize_dequantize=lambda x: x,
))
register_format(FormatSpec(
    name="INT4_W4A16_g128",
    weight_bits=4, group_size=128, scale_bits=16, scale_dtype_name="fp16",
    weight_element_dtype="int4",
    family="int", min_capability_sm=70,
    autoround_config=lambda: _int_autoround(4, 128, 16),
    quantize_dequantize=lambda w: _rtn_uniform_int(w, 4, 128),
    activation_quantize_dequantize=lambda x: x,
))

# Passthrough for highest-precision layer when budget is loose
register_format(FormatSpec(
    name="BF16",
    weight_bits=16, group_size=0, scale_bits=0, scale_dtype_name="none",
    weight_element_dtype="bfloat16",
    family="fp", min_capability_sm=75,
    autoround_config=lambda: dict(bits=16, group_size=0,
                                   data_type="float", act_bits=16,
                                   act_data_type="float"),
    quantize_dequantize=lambda w: w.clone(),
    activation_quantize_dequantize=lambda x: x.clone(),
))

# Source-FP8 passthrough — for models that ship FP8-quantized (MiniMax
# M2/M2.7, DeepSeek V3, several NVIDIA releases). When the allocator
# picks this format for a Linear, the exporter copies the SOURCE FP8
# tensor (+ its weight_scale_inv block-scale tensor) verbatim into the
# output checkpoint — no dequant/requant round-trip. That preserves
# bit-exact the value the probe's BF16 view was dequanted from.
#
# quantize_dequantize is identity: given the probe's BF16 weight (which
# IS the lossless dequant of the source FP8 — every E4M3 code is
# exactly representable in bfloat16), re-quantizing to FP8_SOURCE gives
# back the SAME BF16 view. Cost is zero Δloss, as it should be.
#
# effective_bits = 8 + 32 / (128*128) ≈ 8.002 bpp (scale_inv is fp32
# at the 128×128 block granularity MiniMax ships; smaller than
# MXFP8_E4M3's 8.25 because the block is 128×128 not group-of-32).
register_format(FormatSpec(
    name="FP8_SOURCE",
    weight_bits=8, group_size=128, scale_bits=32, scale_dtype_name="fp32",
    weight_element_dtype="fp8_e4m3",
    scale_block_shape=(128, 128),
    act_bits=None,
    family="fp", min_capability_sm=89,
    autoround_config=lambda: dict(bits=8, group_size=128,
                                   data_type="fp8_e4m3", sym=True,
                                   act_bits=16, act_data_type="float"),
    quantize_dequantize=lambda w: w.clone(),
    activation_quantize_dequantize=lambda x: x.clone(),
))


# GGUF k-quants (llama.cpp / vLLM-GGUF serving lane) — two-tier superblock
# formats along the input dim: fp16 super-scale(s) per 256 + quantized
# per-sub-block scales (and mins for the asymmetric types). The single-tier
# (weight_bits, group_size, scale_bits) fields below are chosen so
# effective_bits_for_shape reproduces the exact fixed GGUF bpw:
# type_size*8/block_size (Q2_K 2.625, Q3_K 3.4375, Q4_K 4.5, Q5_K 5.5,
# Q6_K 6.5625, Q8_0 8.5) — scale_bits carries ALL non-element bytes of the
# superblock (sub-scales + mins + fp16 d/dmin), so the accounting is exact
# for shapes the legality gate admits (in_features % block == 0).
#
# quantize_dequantize routes through prismaquant.gguf_formats, whose field
# quantizers also feed the export byte packers — emulation and shipped
# bytes share one math path by construction. Activation emulation models
# the ggml MMQ/MMVQ compute path (activations quantized to Q8_1: per-32
# symmetric int8); the dequant fallback path is fp16 and strictly better,
# so the emulation is the conservative bound.
def _gguf_autoround(name: str, bits: int, gsize: int):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="gguf",
        gguf_type=name, act_bits=8, act_group_size=32, act_sym=True,
        act_data_type="int", act_dynamic=True,
    )


def _make_gguf_spec(name: str, weight_bits: int, group_size: int,
                    scale_bits: int) -> FormatSpec:
    return FormatSpec(
        name=name,
        weight_bits=weight_bits, group_size=group_size,
        scale_bits=scale_bits, scale_dtype_name="kquant_two_tier",
        weight_element_dtype=f"gguf_{name.lower()}",
        act_bits=8, act_dtype_name="int8_q8_1", act_group_size=32,
        family="gguf", min_capability_sm=60,
        autoround_config=(
            lambda name=name, weight_bits=weight_bits, group_size=group_size:
            _gguf_autoround(name, weight_bits, group_size)
        ),
        quantize_dequantize=make_gguf_qdq(name),
        activation_quantize_dequantize=(
            lambda x: _rtn_uniform_int(x, 8, 32, symmetric=True)
        ),
    )


register_format(_make_gguf_spec("Q2_K", 2, 256, 160))   # 84 B / 256 = 2.625
register_format(_make_gguf_spec("Q3_K", 3, 256, 112))   # 110 B / 256 = 3.4375
register_format(_make_gguf_spec("Q4_K", 4, 256, 128))   # 144 B / 256 = 4.5
register_format(_make_gguf_spec("Q5_K", 5, 256, 128))   # 176 B / 256 = 5.5
register_format(_make_gguf_spec("Q6_K", 6, 256, 144))   # 210 B / 256 = 6.5625
register_format(_make_gguf_spec("Q8_0", 8, 32, 16))     # 34 B / 32 = 8.5

# GGUF IQ family (sub-Q2_K grid codebooks + non-linear 4-bit). scale_bits
# again carries all non-element superblock bytes so effective_bits reproduces
# the exact ggml bpw (type_size*8/block); see prismaquant/gguf_iq_formats.py
# for the field quantizers / byte layouts. IQ4_NL is the only block-32 rung —
# the one usable when in_features % 256 != 0.
register_format(_make_gguf_spec("IQ2_XXS", 2, 256, 16))   # 66 B / 256 = 2.0625
register_format(_make_gguf_spec("IQ2_XS", 2, 256, 80))    # 74 B / 256 = 2.3125
register_format(_make_gguf_spec("IQ2_S", 2, 256, 144))    # 82 B / 256 = 2.5625
register_format(_make_gguf_spec("IQ3_XXS", 3, 256, 16))   # 98 B / 256 = 3.0625
register_format(_make_gguf_spec("IQ3_S", 3, 256, 112))    # 110 B / 256 = 3.4375
register_format(_make_gguf_spec("IQ4_XS", 4, 256, 64))    # 136 B / 256 = 4.25
register_format(_make_gguf_spec("IQ4_NL", 4, 32, 16))     # 18 B / 32 = 4.5


# NVFP4-CB / FP8-CB vector-quantization codebook family (custom out-of-tree
# vLLM plugin lane; NOT stock compressed-tensors — see docs/nvfp4-cb-plan).
# The k-bit VQ index stream lives in scale_bits (fp4 family, weight_bits=0,
# group_size=256 so effective_bits = k/8 + 0.5 exactly) or in weight_bits as
# k/8 bits/param (fp8 family, per-output-channel fp32 scale, group_size=0 so
# effective_bits_for_shape = k/8 + 32/in_features exactly). quantize_dequantize
# is the weighted-VQ closure that also feeds the (Milestone B) byte packer;
# activations are byte-identical to NVFP4 (fp4) / FP8 dynamic (fp8).
def _make_nvfp4_cb_spec(k: int) -> FormatSpec:
    return FormatSpec(
        name=f"NVFP4_CB_K{k}",
        weight_bits=0, group_size=256, scale_bits=32 * k + 128,
        scale_dtype_name="nvfp4_cb_vq",
        weight_element_dtype=f"nvfp4_cb_k{k}",
        act_bits=4, act_dtype_name="fp4_e2m1", act_group_size=16,
        family="nvfp4_cb", min_capability_sm=100,
        autoround_config=(
            lambda k=k: dict(bits=0, group_size=256, data_type="nvfp4_cb",
                             cb_k=k, sym=True, act_bits=4,
                             act_data_type="nv_fp4_with_static_gs",
                             act_group_size=16, act_dynamic=True)
        ),
        quantize_dequantize=make_nvfp4_cb_qdq(k, "fp4", "product"),
        activation_quantize_dequantize=_make_rtn("fp4_e2m1", 16),
    )


def _make_nvfp4_cb_signed_spec(k: int) -> FormatSpec:
    # Sign-magnitude VQ: 8 explicit sign bits + (k-8)-bit positive-grid
    # magnitude index per 8-dim vector; the 8 sign bits are INSIDE k, so the
    # accounting is identical to the flat rungs (32k + 128 bits / 256).
    return FormatSpec(
        name=f"NVFP4_CB_S{k}",
        weight_bits=0, group_size=256, scale_bits=32 * k + 128,
        scale_dtype_name="nvfp4_cb_vq",
        weight_element_dtype=f"nvfp4_cb_s{k}",
        act_bits=4, act_dtype_name="fp4_e2m1", act_group_size=16,
        family="nvfp4_cb", min_capability_sm=100,
        autoround_config=(
            lambda k=k: dict(bits=0, group_size=256, data_type="nvfp4_cb",
                             cb_k=k, cb_mode="signed", sym=True, act_bits=4,
                             act_data_type="nv_fp4_with_static_gs",
                             act_group_size=16, act_dynamic=True)
        ),
        quantize_dequantize=make_nvfp4_cb_qdq(k, "fp4", "signed"),
        activation_quantize_dequantize=_make_rtn("fp4_e2m1", 16),
    )


def _make_fp8_cb_spec(k: int) -> FormatSpec:
    # Index stream in scale_bits (32k bits / 256-superblock, weight_bits=0,
    # group_size=256) so effective_bits = k/8 exactly, mirroring the GGUF /
    # NVFP4_CB accounting. FP8_CB has NO group-16 scale plane; its
    # per-output-channel fp32 scales are accounted by nvfp4_cb_footprint
    # (the authoritative byte accountant, format-pipeline §1.5), which a
    # single-scale FormatSpec cannot model on top of the superblock stream.
    return FormatSpec(
        name=f"FP8_CB_K{k}",
        weight_bits=0, group_size=256, scale_bits=32 * k,
        scale_dtype_name="fp8_cb_vq",
        weight_element_dtype=f"fp8_cb_k{k}",
        act_bits=8, act_dtype_name="fp8_e4m3", act_group_size=0,
        family="fp8_cb", min_capability_sm=100,
        autoround_config=(
            lambda k=k: dict(bits=0, group_size=0, data_type="fp8_cb",
                             cb_k=k, sym=True, act_bits=8,
                             act_data_type="fp8_e4m3", act_dynamic=True)
        ),
        quantize_dequantize=make_nvfp4_cb_qdq(k, "fp8", "product"),
        activation_quantize_dequantize=_make_plain_fp8_activation_vllm_rtn(
            torch.float8_e4m3fn, 448.0,
        ),
    )


for _k in range(12, 25):                    # 2.000 .. 3.500 bpw in 0.125 steps
    register_format(_make_nvfp4_cb_spec(_k))
for _k in (13, 14, 15, 16):                 # signed: 2.125 .. 2.5 bpw
    register_format(_make_nvfp4_cb_signed_spec(_k))
for _k in range(28, 49):                    # 3.5 .. 6.0 bpw in 0.125 steps
    register_format(_make_fp8_cb_spec(_k))


def list_formats(family: str | None = None) -> list[FormatSpec]:
    if family is None:
        return sorted(REGISTRY.values(), key=lambda s: s.effective_bits)
    return sorted((s for s in REGISTRY.values() if s.family == family),
                  key=lambda s: s.effective_bits)


def get_format(name: str) -> FormatSpec:
    canonical = canonical_format_name(name)
    if canonical not in REGISTRY:
        raise KeyError(f"Unknown format '{name}'. Available: "
                       f"{sorted((*REGISTRY.keys(), *FORMAT_ALIASES.keys()))}")
    return REGISTRY[canonical]


def nvfp4_activation_qdq_served(
    x: torch.Tensor,
    input_global_scale: float,
) -> torch.Tensor:
    """Serve-faithful NVFP4 activation quantize-dequantize.

    Emulates vLLM's runtime activation path for W4A4 NVFP4 with a static
    per-tensor ``input_global_scale`` G (the on-disk value): per 16-group
    along the last dim, the FP8-STORED block scale is
    ``sf = fp8_e4m3(amax/6 * G)`` and elements quantize on the E2M1 grid
    at effective scale ``sf / G``. This models the two effects the plain
    dynamic emulation misses (2026-07-02 audit, M18 residual + C1):
    subnormal/zeroed FP8 block scales when a block's amax is far below
    the calibration amax baked into G, and CLIPPING when a block's amax
    exceeds it. Requires ``x.shape[-1] % 16 == 0``; callers fall back to
    the dynamic path otherwise.
    """
    if x.shape[-1] % 16 != 0:
        raise ValueError(
            f"nvfp4_activation_qdq_served needs last dim % 16 == 0, "
            f"got {tuple(x.shape)}")
    g = float(input_global_scale)
    orig_shape = x.shape
    orig_dtype = x.dtype
    x2 = x.reshape(-1, x.shape[-1] // 16, 16).float()
    amax = x2.abs().amax(dim=-1, keepdim=True)
    s_real = amax / 6.0
    # FP8 snap of the stored scale in LOADED units (clamp before the
    # cast: torch's float8_e4m3fn cast is NOT saturating and mints NaN
    # above ~464).
    sf = (s_real * g).clamp(max=448.0)
    sf = sf.to(torch.float8_e4m3fn).float()
    s_used = sf / g
    # sf == 0 (block scale underflowed FP8): vLLM dequantizes the whole
    # block to exactly 0.
    safe = s_used.clamp_min(torch.finfo(torch.float32).tiny)
    q = (x2 / safe).clamp(-6.0, 6.0)
    cb = _codebook_on_device(
        _CODEBOOKS["fp4_e2m1"], device=q.device, dtype=torch.float32)
    idx = torch.bucketize(q.contiguous().abs(), cb[cb >= 0])
    pos = cb[cb >= 0]
    idx_lo = (idx - 1).clamp_min(0)
    idx_hi = idx.clamp_max(pos.numel() - 1)
    lo, hi = pos[idx_lo], pos[idx_hi]
    qa = q.abs()
    # ties toward zero, matching the export codec's _round_to_codebook
    choose_hi = (hi - qa).abs() < (qa - lo).abs()
    rounded = torch.where(choose_hi, hi, lo) * torch.sign(q)
    out = rounded * s_used
    out = torch.where(sf > 0, out, torch.zeros_like(out))
    return out.reshape(orig_shape).to(orig_dtype)
