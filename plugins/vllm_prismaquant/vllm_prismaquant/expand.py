"""Transient CB->value expander (docs/nvfp4-cb-plan/serving-kernel.md §1a,
prototype ii+ / M-gated prefill dispatch).

``expand_cb_to_value`` is the existing decode-GEMM kernel MINUS the matmul MINUS
the per-channel/group scale: it decodes the codebook VALUE for every ``(n, j)``
into a bounded ``[N, K]`` tile. Because an FP8_CB codebook value already lives on
the e4m3 grid (‖·‖<=448), that tile is a *standard per-output-channel FP8
weight* — the caller casts it to ``float8_e4m3fn`` (lossless) and feeds vLLM's
stock W8A8 fp8 GEMM with the layer's existing ``weight_scale``. That is the whole
trick: an expanded FP8_CB weight IS a plain fp8 checkpoint, so prefill reaches
the native tensor cores instead of re-decoding per M-tile in a bf16-MMA kernel.

**INV-1 (docs §0), honored precisely.** The ``[N, K]`` tile is a per-LAYER
TRANSIENT: the caller expands ONE layer, GEMMs, and frees it before the next.
This is *not* the NVINT2 trap — that died from a RESIDENT, model-wide dense
expansion (92.9 GB artifact -> 115.7 GiB resident, OOM). Here the resident weight
stays the packed k-bit index stream + the tiny flat codebook + the per-channel
fp32 scale; only a single layer's decoded tile is ever live (peak ~9 MiB for the
0.6B MLP rung), and it is released each forward. The bounded transient is the
point, not a compromise.

Self-contained: imports only ``torch`` + ``triton`` (no vLLM, no ``prismaquant``,
no ``.kernels``), so the build-venv correctness gate imports it directly. The
codeword byte-window extraction + product sub-index gather below is copied from
``kernels._cb_decode_gemm_kernel`` (kept in lockstep on purpose — the two must
decode bit-identically). **Even-split product only** (FP8_CB_K44: k=44, n_sub=4,
sub_dim=2), matching the decode prototype's scope.

FP8_CB only. NVFP4_CB stays on the Triton decode path: a *transient* NVFP4 tile
would still need the Blackwell FP4-MMA plumbing (prototype (iii), INV-2) to be
worth expanding, and a decoded fp4 value is not a standalone tensor without its
group-16 scale plane — so this expander refuses ``is_fp4``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cb_expand_value_kernel(
    qw_ptr, cb_ptr, cboff_ptr, w_ptr,
    N, K,
    stride_qn,                 # padded row stride (bytes) of qw
    stride_wn, stride_wk,      # output [N, K] strides
    K_BITS: tl.constexpr,
    SUB_DIM: tl.constexpr,
    SUB_W: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)                    # one 256-weight superblock
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n_i = offs_n.to(tl.int64)

    # --- per-column (within one 256-superblock) decode constants ------------
    # (identical to kernels._cb_decode_gemm_kernel; must stay bit-for-bit).
    kcol = tl.arange(0, 256)
    v_local = kcol // 8                         # which of the 32 codewords
    coord = kcol % 8                            # coord inside the 8-dim vector
    sub = coord // SUB_DIM                       # which sub-codebook
    local = coord % SUB_DIM                      # coord inside the sub-vector
    bitpos = v_local * K_BITS
    byte_base = (bitpos // 8).to(tl.int64)       # first byte of the codeword
    bit_in_byte = bitpos % 8
    mask_k = (1 << K_BITS) - 1
    shift_sub = sub * SUB_W
    mask_sub = (1 << SUB_W) - 1
    cb_base = sub * ((1 << SUB_W) * SUB_DIM)     # flat-codebook block base

    # Per-output-row codebook base offset (0 for a single-codebook Linear; a
    # fused qkv/gate_up module points each shard's rows at that role's block of
    # the concatenated flat codebook — same fusion mechanism as the decode
    # kernel, so the transient path is fusion-correct too).
    cb_off = tl.load(cboff_ptr + offs_n_i, mask=mask_n, other=0).to(tl.int64)

    s = pid_s
    col_byte = s * TYPE_SIZE + byte_base                       # [256] int64
    # 8-byte little-endian window; masked to K_BITS so the extra bytes (next
    # superblock / row pad) fall away. INV-1: no dense weight materialized here,
    # only this one [BLOCK_N, 256] tile in registers.
    code = tl.zeros((BLOCK_N, 256), dtype=tl.int64)
    base_ptr = offs_n_i[:, None] * stride_qn + col_byte[None, :]
    for i in range(0, 8):
        b = tl.load(qw_ptr + base_ptr + i, mask=mask_n[:, None],
                    other=0).to(tl.int64)
        code = code | (b << (8 * i))
    code = (code >> bit_in_byte[None, :]) & mask_k
    sub_idx = (code >> shift_sub[None, :]) & mask_sub          # [BN, 256]
    gather = (cb_off[:, None] + cb_base[None, :]
              + sub_idx * SUB_DIM + local[None, :])
    # The raw codebook VALUE (bf16), NOT * scale — this is the decode kernel
    # minus the `* scale` and minus the `tl.dot`.
    val = tl.load(cb_ptr + gather)                             # [BN, 256] bf16

    xcols = (s * 256 + kcol).to(tl.int64)
    w_out = w_ptr + offs_n_i[:, None] * stride_wn + xcols[None, :] * stride_wk
    tl.store(w_out, val, mask=mask_n[:, None])


def expand_cb_to_value(
    cb_qweight_padded: torch.Tensor,   # (N, row_bytes + 8) uint8, 8-byte pad
    cb_flat: torch.Tensor,             # (cb_total,) bf16 flat codebook(s)
    cb_row_offset: torch.Tensor,       # (N,) int32 per-row base into cb_flat
    N: int, K: int,
    k_bits: int, n_sub: int, type_size: int, is_fp4: bool,
) -> torch.Tensor:
    """Decode the codebook VALUE for every ``(n, j)`` into a fresh ``[N, K]``
    bf16 transient (no per-channel/group scale applied).

    The result is the FP8_CB weight's decoded e4m3-grid values; the caller pairs
    it with the layer's per-output ``weight_scale`` to run a stock fp8 W8A8 GEMM.
    Every value is exactly representable in e4m3 (the codebook is e4m3-valued),
    so ``result.to(torch.float8_e4m3fn)`` is lossless.

    INV-1: the returned tile is a bounded per-layer transient; the caller frees
    it after the GEMM. It is never resident/model-wide.
    """
    if is_fp4:
        raise NotImplementedError(
            "expand_cb_to_value is FP8_CB-only (prototype ii+). NVFP4_CB stays "
            "on the Triton decode path: a transient FP4 tile still needs the "
            "Blackwell FP4-MMA to be worth expanding (prototype iii / INV-2), "
            "and a decoded fp4 value is not a standalone tensor without its "
            "group-16 scale plane. See docs/nvfp4-cb-plan/serving-kernel.md.")
    if k_bits % n_sub != 0:
        raise ValueError("expand supports even bit-splits only "
                         f"(k={k_bits}, n_sub={n_sub})")
    if K % 256 != 0:
        raise ValueError(f"K={K} must be a multiple of the 256-weight superblock")
    sub_dim = 8 // n_sub
    sub_w = k_bits // n_sub
    dev = cb_qweight_padded.device
    W = torch.empty((N, K), dtype=torch.bfloat16, device=dev)
    n_sb = K // 256
    block_n = 64
    grid = (triton.cdiv(N, block_n), n_sb)
    _cb_expand_value_kernel[grid](
        cb_qweight_padded, cb_flat, cb_row_offset, W,
        N, K,
        cb_qweight_padded.stride(0),
        W.stride(0), W.stride(1),
        K_BITS=k_bits, SUB_DIM=sub_dim, SUB_W=sub_w, TYPE_SIZE=type_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return W
