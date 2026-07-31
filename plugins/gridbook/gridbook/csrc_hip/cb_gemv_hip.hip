// HIP decode-GEMV for the CB (codebook) formats — the decode-regime kernel of
// the gridbook serving lane on RDNA 3.5 / gfx1151 (Strix Halo).
//
// This is the HIP counterpart of gridbook/csrc/cb_gemv.cu.  It is NOT a
// hipified copy: the memory schedule is re-derived for RDNA, and the format
// decode itself is shared with the WMMA prefill GEMM through cb_decode_hip.h so
// the two cannot drift.
//
// ---------------------------------------------------------------------------
// What is different from the CUDA kernel, and why
// ---------------------------------------------------------------------------
// 1. NO LDS STAGING OF THE PACKED STREAM.  The CUDA kernel stages each
//    superblock into smem to get one coalesced 8-byte-per-lane load and aligned
//    32-bit extraction.  Here each lane instead reads its own three ALIGNED
//    32-bit words straight from global (`extract_code`), deriving the aligned
//    base from the pointer, which
//      (a) removes the per-superblock __syncwarp pair from the inner loop —
//          AMD has no cheap warp-scope barrier equivalent and a workgroup
//          barrier here would serialise waves working on different rows;
//      (b) works for BOTH type_size = 4k (fp8, always 4-byte aligned) and
//          type_size = 4k+9 (fp4 v2, arbitrary byte phase) with one code path,
//          where the CUDA kernel needs two;
//      (c) costs redundancy, not bandwidth: the 32 lanes of a wave read
//          overlapping 12-byte windows spanning one superblock (112..192 B =
//          2-3 cache lines), so after the first lane's miss the rest hit the
//          per-CU vector L0.
// 2. THE CODEBOOK LUT IS LDS-RESIDENT (the analogue of the CUDA lane's R6 smem
//    LUT, `sm120_cb_fused_mma.hpp:167-193`).  A VQ decode is a random gather
//    into a table that is 1-32 KiB depending on rung; on gfx1151 the vector L0
//    is small and a k48 table thrashes it.  Staging it once per workgroup makes
//    every gather a `ds_read_u16`.  See the LDS budget table in README.md — no
//    FP8_CB rung needs LUT splitting (K48 is 32 KiB against a 64 KiB budget).
//    That is only affordable because a workgroup covers MANY output rows
//    (`rows_per_wg`), amortising the staging; the CUDA kernel's one-row-per-
//    block shape would pay it per row.
// 3. ONE WAVE OWNS ONE OUTPUT ROW, so the cross-wave reduction of the CUDA
//    kernel disappears: the row's partial sums are reduced with a single
//    wave32 butterfly and written by lane 0.
//
// Wave32 is assumed throughout (32 codewords per superblock == 32 lanes, which
// is what makes the lane<->codeword mapping exact) and asserted by the host
// launcher against hipDeviceProp_t::warpSize.
//
// Numerics: identical decode contract to the CUDA kernel.
//   v1 (default, matches the shipping CUDA default):
//       w_j  = bf16_rn(f32(codebook_j) * scale)     per weight
//       y_mn = f32 accumulation of f32(w_j) * f32(x_mj)
//   v2 (PRISMAQUANT_CB_DECODE_CONTRACT=v2): the per-weight round is removed and
//       the scale multiplies the lane partial / row total once.
// Both are f32 accumulation, so vs any other implementation the difference is
// summation reassociation only.
#include "cb_decode_hip.h"

#include <hip/hip_runtime.h>

#include <stdint.h>
#include <stdio.h>

namespace pq_hip {

// 16-byte activation vector (8 bf16).  Declared locally rather than using HIP's
// `int4` so the alignment contract is explicit at the use site.
struct alignas(16) U4 {
  uint32_t x, y, z, w;
};

// ---------------------------------------------------------------------------
// FMA of one lane's 8 decoded weights against every active token row.
// `xbase` is the element offset of this lane's 8 weights inside the row, so the
// load is 16-byte aligned (xbase is a multiple of 8 bf16 = 16 B) for every
// lane, and x itself is a torch allocation (256 B aligned).
// ---------------------------------------------------------------------------
template <int MT>
PQ_DEVINL void fma_x(const uint16_t* __restrict__ x, int64_t K, int64_t xbase,
                     int M, const float* wv, float* acc) {
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m < M) {
      const U4 xv =
          *reinterpret_cast<const U4*>(x + static_cast<int64_t>(m) * K + xbase);
      const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[m] = fmaf(wv[2 * i],
                      bf16_to_f32(static_cast<uint16_t>(xw[i] & 0xffffu)),
                      acc[m]);
        acc[m] = fmaf(wv[2 * i + 1],
                      bf16_to_f32(static_cast<uint16_t>(xw[i] >> 16)), acc[m]);
      }
    }
  }
}

// Same, but the lane's 8 weights share one scale applied to the partial
// (decode contract v2).
template <int MT>
PQ_DEVINL void fma_x_scaled(const uint16_t* __restrict__ x, int64_t K,
                            int64_t xbase, int M, const float* wv, float sc,
                            float* acc) {
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m < M) {
      const U4 xv =
          *reinterpret_cast<const U4*>(x + static_cast<int64_t>(m) * K + xbase);
      const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
      float p = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        p = fmaf(wv[2 * i], bf16_to_f32(static_cast<uint16_t>(xw[i] & 0xffffu)),
                 p);
        p = fmaf(wv[2 * i + 1], bf16_to_f32(static_cast<uint16_t>(xw[i] >> 16)),
                 p);
      }
      acc[m] = fmaf(sc, p, acc[m]);
    }
  }
}

// ===========================================================================
// FP8_CB decode GEMV (product mode, n_sub = 4, sub_dim = 2 — the shipped
// FP8_CB_K28..K48 ladder).  Per-output-channel fp32 scale; no superblock scale
// plane (LAYOUT.md §0).
// ===========================================================================
template <int MT, int WAVES, bool LDS_LUT, int SRC>
__global__ __launch_bounds__(WAVES * 32) void cb_gemv_fp8_kernel(
    const uint16_t* __restrict__ x,       // [M, K] bf16, already act-QDQ'd
    const uint8_t* __restrict__ qw,       // [N, qw_stride] padded packed rows
    const void* __restrict__ cbp,         // flat codebook: e4m3 bytes OR bf16
    const int32_t* __restrict__ cboff,    // [N] ELEMENT base into cbp per row
    const float* __restrict__ scale,      // [N] per-output-channel fp32
    uint16_t* __restrict__ y,             // [M, N] bf16
    const int M, const int64_t N, const int64_t K, const int64_t qw_stride,
    const int k_bits, const int type_size, const int lut_elems,
    const int rows_per_wg, const int v2) {
  extern __shared__ uint8_t smem_raw[];
  uint16_t* lut = reinterpret_cast<uint16_t*>(smem_raw);
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int n_sb = static_cast<int>(K >> 8);
  const int64_t row0 = static_cast<int64_t>(blockIdx.x) * rows_per_wg;
  const int64_t row_end =
      (row0 + rows_per_wg < N) ? (row0 + rows_per_wg) : N;

  // The LUT staged for this workgroup is the block's FIRST row's codebook.
  // Fused modules (qkv_proj / gate_up_proj) concatenate several roles'
  // codebooks and point each output row at its role's block, so a block that
  // straddles a role boundary has rows whose base differs; those rows fall back
  // to the global table (correct, just slower, and only for the one straddling
  // block per boundary).
  // The LUT is materialised as BF16 whatever the sidecar holds, once per
  // workgroup; the gather below never learns which it was.
  int32_t staged_base = -1;
  if (LDS_LUT) {
    staged_base = cboff[row0];
    const void* src =
        (SRC == CB_SRC_BF16)
            ? static_cast<const void*>(
                  reinterpret_cast<const uint16_t*>(cbp) + staged_base)
            : static_cast<const void*>(
                  reinterpret_cast<const uint8_t*>(cbp) + staged_base);
    stage_lut_bf16(lut, src, lut_elems, SRC, tid,
                   static_cast<int>(blockDim.x));
    __syncthreads();
  }

  const SubSplit<4, 2> sp(k_bits);
  const uint64_t mask = code_mask_of(k_bits);
  const int bitpos = lane * k_bits;     // this lane's codeword, fixed per row

  for (int64_t r = row0 + wave; r < row_end; r += WAVES) {
    const uint8_t* row = qw + r * qw_stride;
    const int32_t cb_base = cboff[r];
    const float sc = scale[r];
    const bool from_lds = LDS_LUT && (cb_base == staged_base);

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    for (int s = 0; s < n_sb; ++s) {
      const uint8_t* sb = row + static_cast<int64_t>(s) * type_size;
      const uint64_t code = extract_code(sb, bitpos, k_bits, mask);
      float wv[8];
      if (from_lds) {
        gather_fp8_vec<CB_SRC_BF16>(code, lut, 0, sp, wv);
      } else {
        gather_fp8_vec<SRC>(code, cbp, cb_base, sp, wv);
      }
      const int64_t xbase = (static_cast<int64_t>(s) << 8) + (lane << 3);
      // Contract v1 rounds bf16(value * scale) per weight; v2 drops that round
      // and multiplies the ROW TOTAL by the per-output-channel scale once in
      // the epilogue (cb_gemv.cu:426).  Note this is NOT the fp4 idiom: there
      // the scale is per-group-16, so it lands on the lane partial instead.
      if (v2) {
        fma_x<MT>(x, K, xbase, M, wv, acc);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          wv[j] = bf16_to_f32(f32_to_bf16_rn(wv[j] * sc));
        }
        fma_x<MT>(x, K, xbase, M, wv, acc);
      }
    }

    float tot[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) tot[m] = wave_allreduce_add(acc[m]);
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        if (m < M) {
          const float t = v2 ? tot[m] * sc : tot[m];
          y[static_cast<int64_t>(m) * N + r] = f32_to_bf16_rn(t);
        }
      }
    }
  }
}

// ===========================================================================
// NVFP4_CB two-tier (layout v2) decode GEMV.  n_sub == 2 is product mode over
// the bf16 codebook; n_sub == 1 is the signed S-rung family.  The group-16
// E4M3 scale is composed in-register from the packed 9-byte plane
// (two-tier-scale-spec.md §1.1) — there is no resident fp32 scale plane, which
// is the spec's G4 residency contract.
//
// Activation QDQ is NOT fused here: the fp4 group-16 RTN runs outside, in
// codec.fp4_group16_act_qdq, exactly as on the CUDA path, so the two stay
// numerically aligned.
// ===========================================================================
template <int MT, int WAVES, bool LDS_LUT>
__global__ __launch_bounds__(WAVES * 32) void cb_gemv_fp4_v2_kernel(
    const uint16_t* __restrict__ x,        // [M, K] bf16, already act-QDQ'd
    const uint8_t* __restrict__ qw,        // [N, qw_stride]
    const uint16_t* __restrict__ cb,       // flat BF16 codebook
    const int32_t* __restrict__ cboff,     // [N] ELEMENT base into cb
    const float* __restrict__ compose,     // [256*16] two-tier compose table
    uint16_t* __restrict__ y,              // [M, N] bf16
    const int M, const int64_t N, const int64_t K, const int64_t qw_stride,
    const int k_bits, const int n_sub, const int type_size, const int lut_elems,
    const int rows_per_wg, const int v2) {
  extern __shared__ uint8_t smem[];
  uint16_t* lut = reinterpret_cast<uint16_t*>(smem);
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int n_sb = static_cast<int>(K >> 8);
  const int64_t row0 = static_cast<int64_t>(blockIdx.x) * rows_per_wg;
  const int64_t row_end = (row0 + rows_per_wg < N) ? (row0 + rows_per_wg) : N;

  int32_t staged_base = -1;
  if (LDS_LUT) {
    // The NVFP4_CB sidecar is bf16 on disk, so this is already the
    // materialised-bf16 form the LUT dtype contract asks for: a straight copy,
    // no per-gather conversion anywhere on the fp4 path either.
    staged_base = cboff[row0];
    stage_lut_bf16(lut, cb + staged_base, lut_elems, CB_SRC_BF16, tid,
                   static_cast<int>(blockDim.x));
    __syncthreads();
  }

  const SubSplit<2, 4> sp(k_bits);       // unused in signed (n_sub == 1) mode
  const uint64_t mask = code_mask_of(k_bits);
  const int bitpos = lane * k_bits;
  const int grp = lane >> 1;             // group-16 index == codeword / 2
  const int super_off = two_tier_super_off(k_bits);
  const int sub_off = two_tier_sub_off(k_bits, grp);

  for (int64_t r = row0 + wave; r < row_end; r += WAVES) {
    const uint8_t* row = qw + r * qw_stride;
    const int32_t cb_base = cboff[r];
    const bool from_lds = LDS_LUT && (cb_base == staged_base);

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    for (int s = 0; s < n_sb; ++s) {
      const uint8_t* sb = row + static_cast<int64_t>(s) * type_size;
      const uint64_t code = extract_code(sb, bitpos, k_bits, mask);
      const float sc = compose_two_tier(compose, sb[super_off], sb[sub_off],
                                        grp);
      float wv[8];
      if (n_sub == 1) {
        if (from_lds) gather_fp4_signed_vec(code, lut, 0, wv);
        else gather_fp4_signed_vec(code, cb, cb_base, wv);
      } else {
        if (from_lds) gather_fp4_vec(code, lut, 0, sp, wv);
        else gather_fp4_vec(code, cb, cb_base, sp, wv);
      }
      const int64_t xbase = (static_cast<int64_t>(s) << 8) + (lane << 3);
      if (v2) {
        fma_x_scaled<MT>(x, K, xbase, M, wv, sc, acc);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          wv[j] = bf16_to_f32(f32_to_bf16_rn(wv[j] * sc));
        }
        fma_x<MT>(x, K, xbase, M, wv, acc);
      }
    }

    float tot[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) tot[m] = wave_allreduce_add(acc[m]);
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        if (m < M) y[static_cast<int64_t>(m) * N + r] = f32_to_bf16_rn(tot[m]);
      }
    }
  }
}

// ===========================================================================
// Fused per-token fp8 dynamic activation QDQ — the bit-exact mirror of
// codec.fp8_dynamic_act_qdq.  One workgroup per token row.
// ===========================================================================
template <int THREADS>
__global__ __launch_bounds__(THREADS) void fp8_act_qdq_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ out, int64_t K) {
  constexpr int kW = THREADS / 32;
  const int64_t m = blockIdx.x;
  const uint16_t* row = x + m * K;
  uint16_t* orow = out + m * K;
  __shared__ float red[kW];
  __shared__ float s_scale;

  float amax = 0.0f;
  for (int64_t i = threadIdx.x; i < K; i += THREADS) {
    amax = fmaxf(amax, fabsf(bf16_to_f32(row[i])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off, kWave));
  }
  const int w = static_cast<int>(threadIdx.x) >> 5;
  if ((threadIdx.x & 31) == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = red[0];
#pragma unroll
    for (int i = 1; i < kW; ++i) a = fmaxf(a, red[i]);
    s_scale = fmaxf(a * kInvFp8Max, kMinActScale);
  }
  __syncthreads();
  const float sc = s_scale;

  for (int64_t i = threadIdx.x; i < K; i += THREADS) {
    const float v = bf16_to_f32(row[i]);
    const float q = fminf(fmaxf(v / sc, -kFp8Max), kFp8Max);
    const float dq = e4m3_to_f32(f32_to_e4m3_c10(q));
    orow[i] = f32_to_bf16_rn(dq * sc);
  }
}

// ===========================================================================
// FP8-direct transient expander: decode the packed weight into a dense [N, K]
// e4m3-byte tile.  Kept for two reasons: it is the fallback prefill path when
// the WMMA GEMM is unavailable (expand -> hipblaslt), and it is an INDEPENDENT
// second implementation of the decode that the self-test cross-checks the GEMV
// against.  INV-1: the tile is a bounded per-layer transient, never resident.
// ===========================================================================
template <int WAVES>
__global__ __launch_bounds__(WAVES * 32) void cb_expand_fp8_kernel(
    const uint8_t* __restrict__ qw, const uint8_t* __restrict__ cb8,
    const int32_t* __restrict__ cboff, uint8_t* __restrict__ w,
    const int64_t N, const int64_t K, const int64_t qw_stride, const int k_bits,
    const int type_size) {
  const int64_t n = blockIdx.x;
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int n_sb = static_cast<int>(K >> 8);
  const uint8_t* row = qw + n * qw_stride;
  const int32_t cb_base = cboff[n];
  const SubSplit<4, 2> sp(k_bits);
  const uint64_t mask = code_mask_of(k_bits);
  const int bitpos = lane * k_bits;

  for (int s = wave; s < n_sb; s += WAVES) {
    const uint8_t* sb = row + static_cast<int64_t>(s) * type_size;
    const uint64_t code = extract_code(sb, bitpos, k_bits, mask);
    uint8_t bytes[8];
    gather_fp8_vec_bytes(code, cb8, cb_base, sp, bytes);
    uint64_t out8 = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) out8 |= static_cast<uint64_t>(bytes[j]) << (8 * j);
    *reinterpret_cast<uint64_t*>(w + n * K + (static_cast<int64_t>(s) << 8) +
                                 (lane << 3)) = out8;
  }
}

// ===========================================================================
// Host launchers (plain C++: no torch, so the self-test binary links them too).
// ===========================================================================

// LDS bytes a rung's FP8_CB codebook needs once materialised as bf16.  Note
// this is 2x the on-disk e4m3 size: the LUT dtype contract (cb_decode_hip.h)
// trades LDS footprint for a conversion-free gather, and for a bf16-grid
// sidecar there is no conversion to trade in the first place.
int cb_gemv_lut_bytes_fp8(int k_bits) {
  return pq_lut_lds_bytes(pq_lut_elems_fp8(k_bits));
}

// MEASURED LDS-LUT POLICY (gfx1151, cb_hip_selftest --bench, 2026-07-30).
//
// Staging the codebook in LDS is not a free win: it is an occupancy trade, and
// the crossover was measured across the whole rung ladder at N=K=4096, M=1:
//
//   rung   LUT     LDS-LUT    global     verdict
//   K36     4 KiB  0.147 ms   0.172 ms   LDS +17%
//   K44    16 KiB  0.171 ms   0.184 ms   LDS  +8%
//   K45    20 KiB  0.269 ms   0.173 ms   global +36%
//   K46    24 KiB  0.281 ms   0.185 ms   global +34%
//   K47    28 KiB  0.280 ms   0.185 ms   global +34%
//   K48    32 KiB  0.270 ms   0.178 ms   global +34%
//
// The cliff sits exactly between 16 and 20 KiB, i.e. where the per-workgroup
// LDS allocation stops allowing enough concurrent workgroups per WGP to cover
// the gather latency.  So the default is "stage if it costs at most 16 KiB",
// which is a measurement, not a guess — and the caller can still force either
// arm for an A/B (PRISMAQUANT_CB_HIP_LUT=lds|global).
constexpr int kLdsLutMaxBytes = 16 * 1024;

bool cb_gemv_prefer_lds(int lut_bytes, int lds_limit) {
  return lut_bytes > 0 && lut_bytes <= kLdsLutMaxBytes && lut_bytes <= lds_limit;
}

// Hard feasibility (as opposed to the perf preference above): a LUT larger than
// the device's per-workgroup LDS simply cannot be staged.
bool cb_gemv_lut_fits(int lut_bytes, int lds_limit) {
  return lut_bytes > 0 && lut_bytes <= lds_limit;
}

#define PQ_GEMV_LAUNCH_FP8(MT_, WAVES_, LDS_, SRC_)                          \
  hipLaunchKernelGGL(                                                        \
      HIP_KERNEL_NAME(cb_gemv_fp8_kernel<MT_, WAVES_, LDS_, SRC_>),          \
      dim3(blocks), dim3((WAVES_) * 32), (LDS_) ? lds_bytes : 0, stream, x,  \
      qw, cbp, cboff, scale, y, M, N, K, qw_stride, k_bits, type_size,       \
      lut_elems, rows_per_wg, v2)

template <int MT>
static void launch_gemv_fp8_mt(const uint16_t* x, const uint8_t* qw,
                               const void* cbp, const int32_t* cboff,
                               const float* scale, uint16_t* y, int M,
                               int64_t N, int64_t K, int64_t qw_stride,
                               int k_bits, int type_size, int lut_elems,
                               int rows_per_wg, bool use_lds, int src_kind,
                               int v2, hipStream_t stream) {
  const unsigned blocks =
      static_cast<unsigned>((N + rows_per_wg - 1) / rows_per_wg);
  const int lds_bytes = pq_lut_lds_bytes(lut_elems);
  if (use_lds) {
    if (src_kind == CB_SRC_BF16) { PQ_GEMV_LAUNCH_FP8(MT, 4, true, CB_SRC_BF16); }
    else { PQ_GEMV_LAUNCH_FP8(MT, 4, true, CB_SRC_E4M3); }
  } else {
    if (src_kind == CB_SRC_BF16) { PQ_GEMV_LAUNCH_FP8(MT, 4, false, CB_SRC_BF16); }
    else { PQ_GEMV_LAUNCH_FP8(MT, 4, false, CB_SRC_E4M3); }
  }
}

// `M` selects the register-tile width MT; MT >= M always, and lanes with
// m >= M are predicated off, so a partial tile is correct, just idle.
void launch_cb_gemv_fp8(const uint16_t* x, const uint8_t* qw,
                        const void* cbp, const int32_t* cboff,
                        const float* scale, uint16_t* y, int M, int64_t N,
                        int64_t K, int64_t qw_stride, int k_bits,
                        int type_size, int rows_per_wg, bool use_lds,
                        int src_kind, int v2, hipStream_t stream) {
  const int lut_elems = pq_lut_elems_fp8(k_bits);
  if (M <= 1) {
    launch_gemv_fp8_mt<1>(x, qw, cbp, cboff, scale, y, M, N, K, qw_stride,
                          k_bits, type_size, lut_elems, rows_per_wg, use_lds,
                          src_kind, v2, stream);
  } else if (M <= 2) {
    launch_gemv_fp8_mt<2>(x, qw, cbp, cboff, scale, y, M, N, K, qw_stride,
                          k_bits, type_size, lut_elems, rows_per_wg, use_lds,
                          src_kind, v2, stream);
  } else if (M <= 4) {
    launch_gemv_fp8_mt<4>(x, qw, cbp, cboff, scale, y, M, N, K, qw_stride,
                          k_bits, type_size, lut_elems, rows_per_wg, use_lds,
                          src_kind, v2, stream);
  } else if (M <= 8) {
    launch_gemv_fp8_mt<8>(x, qw, cbp, cboff, scale, y, M, N, K, qw_stride,
                          k_bits, type_size, lut_elems, rows_per_wg, use_lds,
                          src_kind, v2, stream);
  } else {
    launch_gemv_fp8_mt<16>(x, qw, cbp, cboff, scale, y, M, N, K, qw_stride,
                           k_bits, type_size, lut_elems, rows_per_wg, use_lds,
                           src_kind, v2, stream);
  }
}

#define PQ_GEMV_LAUNCH_FP4(MT_, WAVES_, LDS_)                                \
  hipLaunchKernelGGL(                                                        \
      HIP_KERNEL_NAME(cb_gemv_fp4_v2_kernel<MT_, WAVES_, LDS_>),             \
      dim3(blocks), dim3((WAVES_) * 32), (LDS_) ? (lut_elems * 2) : 0,       \
      stream, x, qw, cb, cboff, compose, y, M, N, K, qw_stride, k_bits,      \
      n_sub, type_size, lut_elems, rows_per_wg, v2)

template <int MT>
static void launch_gemv_fp4_mt(const uint16_t* x, const uint8_t* qw,
                               const uint16_t* cb, const int32_t* cboff,
                               const float* compose, uint16_t* y, int M,
                               int64_t N, int64_t K, int64_t qw_stride,
                               int k_bits, int n_sub, int type_size,
                               int lut_elems, int rows_per_wg, bool use_lds,
                               int v2, hipStream_t stream) {
  const unsigned blocks =
      static_cast<unsigned>((N + rows_per_wg - 1) / rows_per_wg);
  if (use_lds) {
    PQ_GEMV_LAUNCH_FP4(MT, 4, true);
  } else {
    PQ_GEMV_LAUNCH_FP4(MT, 4, false);
  }
}

void launch_cb_gemv_fp4_v2(const uint16_t* x, const uint8_t* qw,
                           const uint16_t* cb, const int32_t* cboff,
                           const float* compose, uint16_t* y, int M, int64_t N,
                           int64_t K, int64_t qw_stride, int k_bits, int n_sub,
                           int type_size, int rows_per_wg, bool use_lds, int v2,
                           hipStream_t stream) {
  // Signed (n_sub == 1) rungs carry ONE table of 8-dim entries indexed by the
  // (k-8)-bit magnitude; product (n_sub == 2) rungs carry two 4-dim tables.
  const int lut_elems = pq_lut_elems_fp4(k_bits, n_sub);
  if (M <= 1) {
    launch_gemv_fp4_mt<1>(x, qw, cb, cboff, compose, y, M, N, K, qw_stride,
                          k_bits, n_sub, type_size, lut_elems, rows_per_wg,
                          use_lds, v2, stream);
  } else if (M <= 2) {
    launch_gemv_fp4_mt<2>(x, qw, cb, cboff, compose, y, M, N, K, qw_stride,
                          k_bits, n_sub, type_size, lut_elems, rows_per_wg,
                          use_lds, v2, stream);
  } else if (M <= 4) {
    launch_gemv_fp4_mt<4>(x, qw, cb, cboff, compose, y, M, N, K, qw_stride,
                          k_bits, n_sub, type_size, lut_elems, rows_per_wg,
                          use_lds, v2, stream);
  } else if (M <= 8) {
    launch_gemv_fp4_mt<8>(x, qw, cb, cboff, compose, y, M, N, K, qw_stride,
                          k_bits, n_sub, type_size, lut_elems, rows_per_wg,
                          use_lds, v2, stream);
  } else {
    launch_gemv_fp4_mt<16>(x, qw, cb, cboff, compose, y, M, N, K, qw_stride,
                           k_bits, n_sub, type_size, lut_elems, rows_per_wg,
                           use_lds, v2, stream);
  }
}

int cb_gemv_fp4_lut_elems(int k_bits, int n_sub) {
  return pq_lut_elems_fp4(k_bits, n_sub);
}

void launch_fp8_act_qdq(const uint16_t* x, uint16_t* out, int64_t M, int64_t K,
                        hipStream_t stream) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(fp8_act_qdq_kernel<256>),
                     dim3(static_cast<unsigned>(M)), dim3(256), 0, stream, x,
                     out, K);
}

void launch_cb_expand_fp8(const uint8_t* qw, const uint8_t* cb8,
                          const int32_t* cboff, uint8_t* w, int64_t N,
                          int64_t K, int64_t qw_stride, int k_bits,
                          int type_size, hipStream_t stream) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(cb_expand_fp8_kernel<4>),
                     dim3(static_cast<unsigned>(N)), dim3(128), 0, stream, qw,
                     cb8, cboff, w, N, K, qw_stride, k_bits, type_size);
}

}  // namespace pq_hip
