// CUDA decode-GEMV for the FP8_CB codebook format (prototype ii — the
// production decode path; docs/nvfp4-cb-plan/serving-kernel.md §1b).
//
// Replaces the Triton `_cb_decode_gemm_kernel` in the decode regime (M<=16).
// The Triton prototype is ~2.4x below the bandwidth bound on GB10 (4.20 tok/s
// vs AURA's 10.26 on the 27B); this kernel is a straight bandwidth-bound
// dequant-GEMV:
//   * one thread block per output row; 8 warps stride the row's 256-weight
//     superblocks; the packed bytes are staged to smem with coalesced uint4
//     loads and each lane extracts one k-bit codeword (32 codewords <-> 32
//     lanes) with aligned 32-bit reads — the packed stream is read from HBM
//     exactly once;
//   * codebook sub-entries are 2 adjacent bf16 values = one 32-bit __ldg
//     gather (L1/L2-resident table);
//   * INV-1: the dense [N,K] weight is never materialized — decode lives in
//     registers, exactly like the Triton kernel it replaces.
//
// Numerics contract (must preserve the served KL of the Triton path):
//   w_j   = bf16_rn(f32(codebook_j) * weight_scale[n])   — identical rounding
//   y_mn  = f32 accumulation of f32(w_j) * f32(xq_mj)    — reassociation-only
//                                                          difference vs tl.dot
//   xq    = fp8 dynamic per-token QDQ of x, bit-exact to
//           codec.fp8_dynamic_act_qdq (fused here as one kernel: f32 amax ->
//           scale = max(amax/448, 1/(448*512)) -> clamp -> e4m3 rn-satfinite
//           -> f32 -> * scale -> bf16_rn).
//
// Scope: fp8 grid, `product` mode, n_sub=4 (sub_dim=2) — the shipped
// FP8_CB_K{36,40,44,48} rungs. Anything else stays on the Triton fallback.
// Compiled by torch.utils.cpp_extension WITHOUT fast-math (division and
// conversion rounding must match torch exactly).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#define DEVINL __device__ __forceinline__

namespace {

constexpr int kThreads = 256;            // 8 warps
constexpr int kWarps = kThreads / 32;
constexpr int kSlotBytes = 208;          // >= max type_size (192) + 16 slack
                                         // for the aligned 3-word extraction
constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1.0f / (448.0f * 512.0f);
// torch computes tensor/scalar as a RECIPROCAL MULTIPLY (a * f32(1/448)), not
// a true division — 1 f32 ULP off correctly-rounded for some amax. The scale
// must match codec.fp8_dynamic_act_qdq bit-for-bit, so replicate that chain.
constexpr float kInvFp8Max = 1.0f / 448.0f;

DEVINL float bf16_to_f32(uint16_t v) {
  __nv_bfloat16_raw r;
  r.x = v;
  return __bfloat162float(__nv_bfloat16(r));
}

DEVINL uint16_t f32_to_bf16_rn(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// Single-rounded f32 -> e4m3 (RN-even, saturating region pre-clamped by the
// caller). Verbatim port of c10::detail::fp8e4m3fn_from_fp32_value — the
// hardware/`__nv_cvt_float_to_fp8` route can double-round via f16 and differs
// from torch at half-ULP boundaries, which would break the bit-exact QDQ
// contract (seen live: x=0.7265625 rounding to the adjacent code).
DEVINL uint8_t f32_to_e4m3_c10(float f) {
  constexpr uint32_t fp8_max = 1087u << 20;      // 480.0f, first non-e4m3fn
  constexpr uint32_t denorm_mask = 141u << 23;   // 2^(-121+127) subnormal magic
  uint32_t f_bits = __float_as_uint(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & 0x80000000u;
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = 0x7f;                               // NaN (unreachable: pre-clamp)
  } else if (f_bits < (121u << 23)) {            // < 2^-6: subnormal result
    f_bits = __float_as_uint(__uint_as_float(f_bits)
                             + __uint_as_float(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1;       // RN-even tie break
    f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFFu;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

// ---------------------------------------------------------------------------
// Fused per-token fp8 dynamic QDQ (bit-exact mirror of
// codec.fp8_dynamic_act_qdq): one block per token row.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void fp8_act_qdq_kernel(
    const uint16_t* __restrict__ x,   // [M, K] bf16 (as u16)
    uint16_t* __restrict__ out,       // [M, K] bf16 (as u16)
    int64_t K) {
  const int64_t m = blockIdx.x;
  const uint16_t* row = x + m * K;
  uint16_t* orow = out + m * K;
  __shared__ float red[kWarps];
  __shared__ float s_scale;

  float amax = 0.0f;
  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(bf16_to_f32(row[i])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  const int warp = threadIdx.x / 32;
  if ((threadIdx.x & 31) == 0) red[warp] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) a = fmaxf(a, red[w]);
    s_scale = fmaxf(a * kInvFp8Max, kMinScale);
  }
  __syncthreads();
  const float scale = s_scale;

  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) {
    float v = bf16_to_f32(row[i]);
    float q = fminf(fmaxf(v / scale, -kFp8Max), kFp8Max);
    __nv_fp8_storage_t f8 = (__nv_fp8_storage_t)f32_to_e4m3_c10(q);
    float dq = __half2float(__nv_cvt_fp8_to_halfraw(f8, __NV_E4M3));
    orow[i] = f32_to_bf16_rn(dq * scale);
  }
}

// ---------------------------------------------------------------------------
// FP8_CB product-mode decode-GEMV. One block per output row n; warp w handles
// superblocks s = w, w+kWarps, ...; lane v owns codeword v (32 codewords per
// superblock, one per 8-weight vector).
// ---------------------------------------------------------------------------
DEVINL float e4m3_to_f32(uint8_t b) {
  return __half2float(
      __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3));
}

template <int MT, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void cb_gemv_fp8_kernel(
    const uint16_t* __restrict__ x,        // [M, K] bf16 (as u16), QDQ'd
    const uint8_t* __restrict__ qw,        // [N, qw_stride] packed rows
    const uint16_t* __restrict__ cb16,     // E4M3-byte codebook as u16 pairs
    const int32_t* __restrict__ cboff,     // [N] element base into cb_flat
    const float* __restrict__ scale,       // [N] per-output-channel fp32
    uint16_t* __restrict__ y,              // [M, N] bf16 (as u16)
    const int M, const int64_t N, const int64_t K,
    const int64_t qw_stride,
    const int k_bits, const int sub_w, const int type_size) {
  const int64_t n = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n_sb = (int)(K >> 8);          // K / 256

  __shared__ __align__(16) uint8_t stage[WARPS][kSlotBytes];
  __shared__ float red[WARPS][MT > 0 ? MT : 1];

  const uint8_t* row = qw + n * qw_stride;
  const float sc_row = __ldg(scale + n);
  const int64_t cb_base = (int64_t)__ldg(cboff + n);
  const uint32_t sub_mask = (1u << sub_w) - 1u;
  const int sub_entries2 = 2 << sub_w;      // elements per sub-table (2^w * 2)
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);

  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

  // Rows are 8-byte aligned (row_bytes is 16-aligned for the fp8 rungs, the
  // pad is 8), so the stage uses 8-byte loads: type_size/8 <= 24 lanes cover
  // one superblock in a single coalesced round.
  const int stage_vecs = type_size >> 3;

  for (int s = warp; s < n_sb; s += WARPS) {
    // --- coalesced stage of this superblock's bytes into the warp slot ---
    // __ldcs (evict-first): the packed stream is read exactly once per token;
    // keep L2 for the codebook / x / the bf16 floor layers instead.
    const uint64_t* gsrc =
        reinterpret_cast<const uint64_t*>(row + (int64_t)s * type_size);
    uint64_t* gdst = reinterpret_cast<uint64_t*>(stage[warp]);
    if (lane < stage_vecs) gdst[lane] = __ldcs(gsrc + lane);
    __syncwarp();

    // --- extract this lane's k-bit codeword (aligned 32-bit smem reads) ---
    const int bitpos = lane * k_bits;
    const int b0 = bitpos >> 3;
    const int rem = ((b0 & 3) << 3) + (bitpos & 7);
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(stage[warp]);
    const int widx = b0 >> 2;
    const uint32_t w0_ = s32[widx];
    const uint32_t w1_ = s32[widx + 1];
    const uint32_t w2_ = s32[widx + 2];
    // All smem reads of this superblock are done; release the slot for the
    // next iteration's stage (independent-thread-scheduling hazard).
    __syncwarp();
    const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
    uint64_t code = lo >> rem;
    if (rem + k_bits > 64) {
      code |= (uint64_t)w2_ << (64 - rem);
    }
    code &= code_mask;

    // --- decode 4 sub-indices -> 8 weights (Triton-bit-exact rounding) ----
    // The codebook is E4M3 BYTES (the values are e4m3-grid by construction,
    // so byte -> f32 equals the Triton path's bf16 -> f32 exactly): 16 KB for
    // k48 instead of 64 KB bf16 — 4x the L1 coverage, half the gather bytes.
    float wv[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t idx = (uint32_t)(code >> (i * sub_w)) & sub_mask;
      const int64_t elt = cb_base + (int64_t)i * sub_entries2 + (int64_t)idx * 2;
      const uint16_t pair = __ldg(cb16 + (elt >> 1));
      // Match the Triton kernel bit-for-bit: w = bf16_rn(val * scale) before
      // the f32 product (tl: (val * sc_row).to(bfloat16) then tl.dot).
      wv[2 * i] = bf16_to_f32(
          f32_to_bf16_rn(e4m3_to_f32((uint8_t)(pair & 0xffu)) * sc_row));
      wv[2 * i + 1] = bf16_to_f32(
          f32_to_bf16_rn(e4m3_to_f32((uint8_t)(pair >> 8)) * sc_row));
    }

    // --- FMA against x: one 16-byte load per (m, lane) --------------------
    const int64_t xbase = ((int64_t)s << 8) + (lane << 3);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < M) {
        const uint4 xv = __ldg(
            reinterpret_cast<const uint4*>(x + (int64_t)m * K + xbase));
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[m] = fmaf(wv[2 * i],
                        bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), acc[m]);
          acc[m] = fmaf(wv[2 * i + 1],
                        bf16_to_f32((uint16_t)(xw[i] >> 16)), acc[m]);
        }
      }
    }
  }

  // --- reduce: 32 lanes -> warp leader -> block --------------------------
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) red[warp][m] = v;
  }
  __syncthreads();
  if (warp == 0 && lane < MT && lane < M) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += red[w][lane];
    y[(int64_t)lane * N + n] = f32_to_bf16_rn(total);
  }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
torch::Tensor fp8_act_qdq(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "fp8_act_qdq wants a CUDA bf16 tensor");
  auto x2 = x.contiguous();
  const int64_t K = x2.size(-1);
  const int64_t M = x2.numel() / K;
  auto out = torch::empty_like(x2);
  if (M == 0 || K == 0) return out;
  const c10::cuda::OptionalCUDAGuard guard(x2.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  fp8_act_qdq_kernel<<<(unsigned)M, kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x2.data_ptr()),
      reinterpret_cast<uint16_t*>(out.data_ptr()), K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int MT>
void launch_gemv(const torch::Tensor& xq, const torch::Tensor& qw,
                 const torch::Tensor& cb, const torch::Tensor& cboff,
                 const torch::Tensor& scale, torch::Tensor& y,
                 int M, int64_t N, int64_t K, int k_bits, int sub_w,
                 int type_size, cudaStream_t stream) {
  // Warp count: superblocks per row are warp-strided, so a row count that is
  // a multiple of 4 but not 8 (e.g. K=5120 -> 20 superblocks) leaves a 20%
  // tail at 8 warps; 4 warps divide it exactly. Large rows amortize the tail
  // and prefer 8 warps for block-level parallelism.
  const int n_sb = (int)(K >> 8);
  const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
#define PQ_LAUNCH(W)                                                       \
  cb_gemv_fp8_kernel<MT, W><<<(unsigned)N, (W)*32, 0, stream>>>(           \
      reinterpret_cast<const uint16_t*>(xq.data_ptr()),                    \
      qw.data_ptr<uint8_t>(),                                              \
      reinterpret_cast<const uint16_t*>(cb.data_ptr()),                    \
      cboff.data_ptr<int32_t>(), scale.data_ptr<float>(),                  \
      reinterpret_cast<uint16_t*>(y.data_ptr()),                           \
      M, N, K, qw.stride(0), k_bits, sub_w, type_size)
  if (use4) {
    PQ_LAUNCH(4);
  } else {
    PQ_LAUNCH(8);
  }
#undef PQ_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor cb_gemv_fp8(torch::Tensor x, torch::Tensor qw_padded,
                          torch::Tensor cb_flat, torch::Tensor cb_row_offset,
                          torch::Tensor scale, int64_t N, int64_t K,
                          int64_t k_bits, int64_t n_sub, int64_t type_size,
                          bool qdq_input) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "cb_gemv_fp8 wants bf16 activations");
  TORCH_CHECK(qw_padded.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cb_flat.scalar_type() == torch::kUInt8,
              "cb_gemv_fp8 wants the E4M3-byte (uint8) codebook");
  TORCH_CHECK(cb_row_offset.scalar_type() == torch::kInt32);
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(n_sub == 4, "CUDA GEMV supports the fp8 n_sub=4 rungs only");
  TORCH_CHECK(k_bits % n_sub == 0, "even bit-split only");
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of the 256 superblock");
  TORCH_CHECK(type_size % 16 == 0 && type_size <= 192,
              "type_size must be 16-aligned and <= 192 (fp8 rungs)");
  TORCH_CHECK(type_size == 4 * k_bits, "fp8 type_size must equal 4*k");
  const int sub_w = (int)(k_bits / n_sub);
  TORCH_CHECK(sub_w <= 12, "sub-table beyond the shipped fp8 rungs");
  TORCH_CHECK(qw_padded.dim() == 2 && qw_padded.size(0) == N);
  TORCH_CHECK(qw_padded.stride(1) == 1, "qw rows must be contiguous");
  TORCH_CHECK(cb_row_offset.numel() == N, "cb_row_offset must cover every row");

  auto sizes = x.sizes().vec();
  auto x2 = x.reshape({-1, K}).contiguous();
  const int64_t M = x2.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "decode GEMV handles M in [1,16]");

  const c10::cuda::OptionalCUDAGuard guard(x2.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor xq = qdq_input ? fp8_act_qdq(x2) : x2;
  auto y = torch::empty({M, N}, x2.options());

  const int m = (int)M;
  if (M <= 1) {
    launch_gemv<1>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, sub_w, (int)type_size, stream);
  } else if (M <= 2) {
    launch_gemv<2>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, sub_w, (int)type_size, stream);
  } else if (M <= 4) {
    launch_gemv<4>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, sub_w, (int)type_size, stream);
  } else if (M <= 8) {
    launch_gemv<8>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, sub_w, (int)type_size, stream);
  } else {
    launch_gemv<16>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                    (int)k_bits, sub_w, (int)type_size, stream);
  }

  sizes.back() = N;
  return y.reshape(sizes);
}

// ---------------------------------------------------------------------------
// FP8-direct transient expand (prefill): decode the whole packed weight into
// a [N, K] e4m3-byte tile. Same stage/extract/LUT structure as the GEMV with
// the FMA replaced by one coalesced 8-byte store per codeword — the Triton
// byte-gather expander ran at 61-86 GB/s and serialized ~half the prefill;
// this one is stream-bandwidth-bound.
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32) void cb_expand_fp8_kernel(
    const uint8_t* __restrict__ qw,        // [N, qw_stride] packed rows
    const uint16_t* __restrict__ cb16,     // E4M3-byte codebook as u16 pairs
    const int32_t* __restrict__ cboff,     // [N] element base into cb_flat
    uint8_t* __restrict__ w,               // [N, K] e4m3 bytes out
    const int64_t N, const int64_t K, const int64_t qw_stride,
    const int k_bits, const int sub_w, const int type_size) {
  const int64_t n = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n_sb = (int)(K >> 8);

  __shared__ __align__(16) uint8_t stage[WARPS][kSlotBytes];

  const uint8_t* row = qw + n * qw_stride;
  const int64_t cb_base = (int64_t)__ldg(cboff + n);
  const uint32_t sub_mask = (1u << sub_w) - 1u;
  const int sub_entries2 = 2 << sub_w;
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
  const int stage_vecs = type_size >> 3;

  for (int s = warp; s < n_sb; s += WARPS) {
    const uint64_t* gsrc =
        reinterpret_cast<const uint64_t*>(row + (int64_t)s * type_size);
    uint64_t* gdst = reinterpret_cast<uint64_t*>(stage[warp]);
    if (lane < stage_vecs) gdst[lane] = __ldcs(gsrc + lane);
    __syncwarp();

    const int bitpos = lane * k_bits;
    const int b0 = bitpos >> 3;
    const int rem = ((b0 & 3) << 3) + (bitpos & 7);
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(stage[warp]);
    const int widx = b0 >> 2;
    const uint32_t w0_ = s32[widx];
    const uint32_t w1_ = s32[widx + 1];
    const uint32_t w2_ = s32[widx + 2];
    __syncwarp();
    const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
    uint64_t code = lo >> rem;
    if (rem + k_bits > 64) {
      code |= (uint64_t)w2_ << (64 - rem);
    }
    code &= code_mask;

    uint64_t out8 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t idx = (uint32_t)(code >> (i * sub_w)) & sub_mask;
      const int64_t elt = cb_base + (int64_t)i * sub_entries2 + (int64_t)idx * 2;
      const uint64_t pair = (uint64_t)__ldg(cb16 + (elt >> 1));
      out8 |= pair << (16 * i);
    }
    // One coalesced 8-byte store per codeword: 256 B per warp-superblock.
    *reinterpret_cast<uint64_t*>(
        w + n * K + ((int64_t)s << 8) + (lane << 3)) = out8;
  }
}

torch::Tensor cb_expand_fp8(torch::Tensor qw_padded, torch::Tensor cb_flat_fp8,
                            torch::Tensor cb_row_offset, int64_t N, int64_t K,
                            int64_t k_bits, int64_t n_sub, int64_t type_size) {
  TORCH_CHECK(qw_padded.is_cuda() && qw_padded.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cb_flat_fp8.scalar_type() == torch::kUInt8,
              "cb_expand_fp8 wants the E4M3-byte (uint8) codebook");
  TORCH_CHECK(cb_row_offset.scalar_type() == torch::kInt32 &&
              cb_row_offset.numel() == N);
  TORCH_CHECK(n_sub == 4 && k_bits % 4 == 0);
  TORCH_CHECK(K % 256 == 0 && type_size == 4 * k_bits && type_size % 16 == 0 &&
              type_size <= 192);
  TORCH_CHECK(qw_padded.dim() == 2 && qw_padded.size(0) == N &&
              qw_padded.stride(1) == 1);
  const c10::cuda::OptionalCUDAGuard guard(qw_padded.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto w = torch::empty({N, K}, qw_padded.options());
  const int sub_w = (int)(k_bits / n_sub);
  const int n_sb = (int)(K >> 8);
  const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
  if (use4) {
    cb_expand_fp8_kernel<4><<<(unsigned)N, 128, 0, stream>>>(
        qw_padded.data_ptr<uint8_t>(),
        reinterpret_cast<const uint16_t*>(cb_flat_fp8.data_ptr()),
        cb_row_offset.data_ptr<int32_t>(), w.data_ptr<uint8_t>(),
        N, K, qw_padded.stride(0), (int)k_bits, sub_w, (int)type_size);
  } else {
    cb_expand_fp8_kernel<8><<<(unsigned)N, 256, 0, stream>>>(
        qw_padded.data_ptr<uint8_t>(),
        reinterpret_cast<const uint16_t*>(cb_flat_fp8.data_ptr()),
        cb_row_offset.data_ptr<int32_t>(), w.data_ptr<uint8_t>(),
        N, K, qw_padded.stride(0), (int)k_bits, sub_w, (int)type_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return w.view(torch::kFloat8_e4m3fn);
}

// ---------------------------------------------------------------------------
// Debug probes (test-only): isolate the conversion and the scale reduction.
// ---------------------------------------------------------------------------
__global__ void e4m3_probe_kernel(const float* __restrict__ q,
                                  uint8_t* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = f32_to_e4m3_c10(q[i]);
}

torch::Tensor e4m3_probe(torch::Tensor q) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat32);
  auto qc = q.contiguous();
  auto out = torch::empty(qc.sizes(), qc.options().dtype(torch::kUInt8));
  const int64_t n = qc.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  e4m3_probe_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
      qc.data_ptr<float>(), out.data_ptr<uint8_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ __launch_bounds__(kThreads) void qdq_scale_kernel(
    const uint16_t* __restrict__ x, float* __restrict__ scales, int64_t K) {
  const int64_t m = blockIdx.x;
  const uint16_t* row = x + m * K;
  __shared__ float red[kWarps];
  float amax = 0.0f;
  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(bf16_to_f32(row[i])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  const int warp = threadIdx.x / 32;
  if ((threadIdx.x & 31) == 0) red[warp] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) a = fmaxf(a, red[w]);
    scales[m] = fmaxf(a * kInvFp8Max, kMinScale);
  }
}

torch::Tensor qdq_scale_probe(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16);
  auto x2 = x.contiguous();
  const int64_t K = x2.size(-1);
  const int64_t M = x2.numel() / K;
  auto out = torch::empty({M}, x2.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  qdq_scale_kernel<<<(unsigned)M, kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x2.data_ptr()),
      out.data_ptr<float>(), K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_act_qdq", &fp8_act_qdq,
        "Fused per-token fp8 dynamic QDQ (bit-exact to codec.fp8_dynamic_act_qdq)");
  m.def("cb_gemv_fp8", &cb_gemv_fp8,
        "FP8_CB product-mode decode GEMV (bandwidth-bound, INV-1)");
  m.def("cb_expand_fp8", &cb_expand_fp8,
        "FP8-direct transient expand (prefill; bounded per-layer tile)");
  m.def("e4m3_probe", &e4m3_probe, "debug: f32 -> e4m3 codes via the c10 port");
  m.def("qdq_scale_probe", &qdq_scale_probe,
        "debug: per-token scale exactly as the QDQ kernel computes it");
}
