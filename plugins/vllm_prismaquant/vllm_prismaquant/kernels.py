"""Triton decode-GEMM kernels for the NVFP4-CB / FP8-CB codebook formats.

**CORRECTNESS-FIRST PROTOTYPE — NOT PRODUCTION-ELIGIBLE.**

This is prototype (i) of docs/nvfp4-cb-plan/serving-kernel.md. It exists to
measure served KL and get a first speed reading; it is explicitly disqualified
as the production prefill path:

* **INV-1 (honored):** the resident weight is the packed k-bit index stream +
  the tiny flat codebook + the (pre-decoded) E4M3/fp32 scales. The dense [N,K]
  weight is NEVER materialized in HBM — each superblock's [256, BLOCK_N] weight
  tile is expanded inside the kernel, in registers, then consumed immediately by
  the matmul. That is the whole point (the NVINT2 OOM trap was exactly a
  load-time dense expansion).
* **INV-2 (WAIVED for this prototype):** we decode FP4/FP8 codes to bf16 and run
  `tl.dot` (bf16 MMA). Triton cannot emit the Blackwell sm_121 block-scaled FP4
  MMA, so this kernel reaches only bf16 tensor cores. The production prefill
  (CUTLASS/CuTe fused-expand, prototype (iii)) is what routes decoded codes to
  the FP4 MMA; THIS kernel will fail the perf gate by construction and must not
  be promoted. Comments below say so at each relevant point.

Only the **even-split product** mode is implemented (both shipped rungs are even
splits: NVFP4_CB_K16 -> (8,8); FP8_CB_K44 -> (11,11,11,11)). Uneven splits
(e.g. k=13 -> (7,6)) and signed/full modes are out of scope for this prototype.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cb_decode_gemm_kernel(
    x_ptr, qw_ptr, cb_ptr, cboff_ptr, scale_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qn,                 # padded row stride (bytes) of qw
    stride_ym, stride_yn,
    stride_sn,                 # scale-row stride (fp4 only; ignored for fp8)
    K_BITS: tl.constexpr,
    N_SUB: tl.constexpr,
    SUB_DIM: tl.constexpr,
    SUB_W: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    IS_FP4: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_n_i = offs_n.to(tl.int64)

    # --- per-column (within one 256-superblock) decode constants ------------
    kcol = tl.arange(0, 256)
    v_local = kcol // 8                       # which of the 32 codewords
    coord = kcol % 8                          # coord inside the 8-dim vector
    sub = coord // SUB_DIM                     # which sub-codebook
    local = coord % SUB_DIM                    # coord inside the sub-vector
    bitpos = v_local * K_BITS
    byte_base = (bitpos // 8).to(tl.int64)     # first byte of the codeword
    bit_in_byte = bitpos % 8
    mask_k = (1 << K_BITS) - 1
    shift_sub = sub * SUB_W
    mask_sub = (1 << SUB_W) - 1
    cb_base = sub * ((1 << SUB_W) * SUB_DIM)   # flat-codebook block base
    grp16 = kcol // 16                          # group-16 index inside superblock

    # Per-output-row codebook base offset (0 for a single-codebook Linear; for a
    # fused qkv/gate_up module each shard's rows point at that role's block of
    # the concatenated flat codebook — this is how per-role shared codebooks
    # survive vLLM's qkv/gate_up fusion).
    cb_off = tl.load(cboff_ptr + offs_n_i, mask=mask_n, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_sb = K // 256

    if not IS_FP4:
        # fp8: one per-output-channel fp32 scale, hoisted out of the K loop.
        sc_row = tl.load(scale_ptr + offs_n_i, mask=mask_n, other=0.0)  # [BN]

    for s in range(0, n_sb):
        col_byte = s * TYPE_SIZE + byte_base                       # [256] int64
        # --- expand the codeword for every (row, kcol) IN REGISTERS ---------
        # 8-byte little-endian window; masked to K_BITS -> the extra bytes
        # (scale plane / next superblock) fall away. INV-1: no dense weight.
        code = tl.zeros((BLOCK_N, 256), dtype=tl.int64)
        base_ptr = offs_n_i[:, None] * stride_qn + col_byte[None, :]
        for i in range(0, 8):
            b = tl.load(qw_ptr + base_ptr + i, mask=mask_n[:, None],
                        other=0).to(tl.int64)
            code = code | (b << (8 * i))
        code = (code >> bit_in_byte[None, :]) & mask_k
        sub_idx = (code >> shift_sub[None, :]) & mask_sub          # [BN,256]
        gather = (cb_off[:, None] + cb_base[None, :]
                  + sub_idx * SUB_DIM + local[None, :])
        val = tl.load(cb_ptr + gather).to(tl.float32)             # [BN,256]

        if IS_FP4:
            grp = s * 16 + grp16                                    # [256]
            sc = tl.load(scale_ptr + offs_n_i[:, None] * stride_sn + grp[None, :],
                         mask=mask_n[:, None], other=0.0)          # [BN,256]
            w = (val * sc).to(tl.bfloat16)
        else:
            w = (val * sc_row[:, None]).to(tl.bfloat16)            # [BN,256]

        xcols = (s * 256 + kcol).to(tl.int64)
        x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
                    + xcols[None, :] * stride_xk,
                    mask=mask_m[:, None], other=0.0).to(tl.bfloat16)  # [BM,256]
        # bf16 MMA (INV-2 waived): y += x @ w^T.
        acc += tl.dot(x, tl.trans(w))

    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             y, mask=mask_m[:, None] & mask_n[None, :])


def cb_decode_linear(
    x: torch.Tensor,             # (..., K) activations (bf16/fp16)
    qw_padded: torch.Tensor,     # (N, row_bytes+8) uint8, +8 pad for the window
    cb_flat: torch.Tensor,       # (cb_total,) bf16 flat codebook(s), concatenated
    cb_row_offset: torch.Tensor,  # (N,) int32 per-row base into cb_flat
    scale: torch.Tensor,         # fp4: (N, n_sb*16) fp32 ; fp8: (N,) fp32
    *, N: int, K: int,
    k_bits: int, n_sub: int, type_size: int, is_fp4: bool,
) -> torch.Tensor:
    """Launch the decode-GEMM. Returns (..., N). M-gated: a small BLOCK_M for
    the decode regime (M<=16), a larger tile for prefill — mirrors GGUF's
    MMVQ/MMQ split (quantization/linear.py:34-57), one Triton kernel either
    way. The dense weight is never materialized (INV-1)."""
    assert k_bits % n_sub == 0, "prototype supports even bit-splits only"
    sub_dim = 8 // n_sub
    sub_w = k_bits // n_sub
    orig_shape = x.shape
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    block_m = 16 if M <= 16 else 64
    block_n = 64
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    stride_sn = scale.stride(0) if is_fp4 else 0
    _cb_decode_gemm_kernel[grid](
        x2, qw_padded, cb_flat, cb_row_offset, scale, y,
        M, N, K,
        x2.stride(0), x2.stride(1),
        qw_padded.stride(0),
        y.stride(0), y.stride(1),
        stride_sn,
        K_BITS=k_bits, N_SUB=n_sub, SUB_DIM=sub_dim, SUB_W=sub_w,
        TYPE_SIZE=type_size, IS_FP4=is_fp4,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )
    return y.reshape(*orig_shape[:-1], N)
