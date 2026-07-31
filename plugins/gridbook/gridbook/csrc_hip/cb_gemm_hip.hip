// HIP decode-in-prologue GEMM for FP8_CB, using RDNA 3.5 bf16 WMMA.
//
// This is the prefill / mid-M counterpart of cb_gemv_hip.hip and the HIP answer
// to the CUDA lane's CUTLASS `cb_fused_prefill_mm_scaled`.  RDNA 3.5 has no fp8
// matrix instruction (MEASURED — see README §"WMMA availability"), so the
// architecture is forced and is also the right one: fp8-CB is a STORAGE format
// here, decoded to bf16 WMMA fragments in the prologue and accumulated in f32.
// A decoded e4m3 value is exactly representable in bf16 (3 mantissa bits into
// 7), so the decode is lossless and the arithmetic is strictly better than an
// fp8-MMA path would be.
//
// ---------------------------------------------------------------------------
// The central trick: NO LDS TILE FOR A OR B
// ---------------------------------------------------------------------------
// The wave32 WMMA fragment layout was established on the device by
// wmma_probe.hip (not taken from documentation):
//
//     A (16x16): lane l holds row (l % 16), register j holds k = j
//     B (16x16): lane l holds col (l % 16), register j holds k = j
//                (lanes 16..31 duplicate lanes 0..15 — required by the encoding)
//     D (16x16, 8 f32 per lane): register i of lane l holds
//                                D[2*i + l/16][l % 16]
//
// Read that B row again: one lane needs 16 CONSECUTIVE k values of ONE output
// channel.  A CB codeword covers exactly 8 weights, so **one lane fills its
// entire B fragment by decoding exactly two codewords**.  There is no
// cross-lane data movement, hence no LDS staging tile, no bank-conflict
// padding, and no barrier in the K loop.  The only LDS in this kernel is the
// codebook LUT, which is read-only after one barrier at workgroup entry.
//
// Likewise each lane's A fragment is 16 consecutive bf16 of one token row = two
// 16-byte loads straight from global.
//
// ---------------------------------------------------------------------------
// Tiling
// ---------------------------------------------------------------------------
//   workgroup = 4 waves arranged 2 (M) x 2 (N)
//   wave tile = WM x 16  by  WN x 16   (WN = 2 fixed; WM in {1, 2, 4})
//   block tile= (2*WM*16) x (2*WN*16) = {32,64,128} x 64
// A-fragments are reused across the WN B-fragments and vice versa, so the wave
// issues WM*WN = 2..8 WMMA ops per 16-deep K step for WM+WN fragment loads.
//
// K is always a multiple of 256 (the superblock is the format's hard
// constraint, LAYOUT.md §0), and 16 divides 256, so there is no K edge tile.
// M and N edges ARE handled: out-of-range A rows load zero, out-of-range B
// columns decode to zero, and the epilogue masks the store.
#include "cb_decode_hip.h"

#include <hip/hip_runtime.h>

#include <stdint.h>

namespace pq_hip {

typedef short pq_v16s __attribute__((ext_vector_type(16)));
typedef float pq_v8f __attribute__((ext_vector_type(8)));

// gfx11 wave32 bf16 WMMA.  If the builtin is missing for the compiled target
// the fallback below computes the same 16x16x16 product with plain f32 FMAs
// under the same fragment convention: correct, and roughly an order of
// magnitude slower.  Which one was compiled is reported by
// `pq_hip::cb_gemm_uses_wmma()` so the benchmark can never silently attribute
// scalar throughput to the matrix unit.
#if defined(__HIP_DEVICE_COMPILE__)
#define PQ_WMMA_BF16_OK __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
#else
#define PQ_WMMA_BF16_OK 1
#endif

PQ_DEVINL pq_v8f pq_mma(pq_v16s a, pq_v16s b, pq_v8f acc) {
#if PQ_WMMA_BF16_OK
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
#else
  // Scalar fallback.  Lane l holds A row (l%16) and B col (l%16); the product
  // needs A row (2*i + l/16) crossed with B col (l%16), so the A operand must
  // travel across lanes — done with __shfl, which is why this path exists only
  // as a correctness net and not as a performance option.
  const int lane = __lane_id();
  pq_v8f out = acc;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m = 2 * i + (lane >> 4);
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const short av = __shfl(a[k], m, kWave);
      union { uint32_t u; float f; } fa, fb;
      fa.u = static_cast<uint32_t>(static_cast<uint16_t>(av)) << 16;
      fb.u = static_cast<uint32_t>(static_cast<uint16_t>(b[k])) << 16;
      s = fmaf(fa.f, fb.f, s);
    }
    out[i] += s;
  }
  return out;
#endif
}

// Decode one lane's whole B fragment: 16 consecutive k of output channel
// `n_row`, i.e. codewords v0 and v0+1 of superblock `s`.
//
// The gather already yields bf16 BIT PATTERNS, which is exactly the WMMA
// operand format — so on the LDS path (SRC = CB_SRC_BF16, the materialised
// LUT) the prologue is a pure table read with zero conversion ALU.  The e4m3
// instantiation exists only for the global-gather fallback.
template <int SRC>
PQ_DEVINL void decode_b_frag(pq_v16s& frag, const uint8_t* __restrict__ row,
                             const void* __restrict__ cb, int32_t cb_base,
                             const SubSplit<4, 2>& sp, uint64_t mask,
                             int k_bits, int type_size, int k0) {
  const int s = k0 >> 8;                      // superblock index
  const int v0 = (k0 & 255) >> 3;             // first codeword within it
  const uint8_t* sb = row + static_cast<int64_t>(s) * type_size;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint64_t code =
        extract_code(sb, (v0 + h) * k_bits, k_bits, mask);
    uint16_t bits[8];
    gather_fp8_bits<SRC>(code, cb, cb_base, sp, bits);
#pragma unroll
    for (int j = 0; j < 8; ++j) frag[h * 8 + j] = static_cast<short>(bits[j]);
  }
}

PQ_DEVINL void zero_frag(pq_v16s& frag) {
#pragma unroll
  for (int j = 0; j < 16; ++j) frag[j] = 0;
}

// ===========================================================================
// The GEMM.  y[M,N] = qdq(x)[M,K] @ W[N,K]^T * scale[N], W decoded from the
// packed CB stream.  x arrives bf16 and ALREADY activation-quantised (the fp8
// dynamic per-token bucket), matching the CUDA prefill contract where the
// activation quantisation is a separate step; the weight scale is applied in
// the f32 epilogue, matching cutlass_scaled_mm's ordering rather than rounding
// to bf16 first.
// ===========================================================================
template <int WM, int WN, bool LDS_LUT, int SRC>
__global__ __launch_bounds__(128) void cb_gemm_fp8_wmma_kernel(
    const uint16_t* __restrict__ x,      // [M, K] bf16, act-QDQ'd
    const uint8_t* __restrict__ qw,      // [N, qw_stride]
    const void* __restrict__ cbp,        // flat codebook: e4m3 bytes OR bf16
    const int32_t* __restrict__ cboff,   // [N]
    const float* __restrict__ scale,     // [N]
    uint16_t* __restrict__ y,            // [M, N] bf16
    const int64_t M, const int64_t N, const int64_t K, const int64_t qw_stride,
    const int k_bits, const int type_size, const int lut_elems) {
  extern __shared__ uint8_t smem_raw[];
  uint16_t* lut = reinterpret_cast<uint16_t*>(smem_raw);
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int wave_m = wave >> 1;            // 2 x 2 wave grid
  const int wave_n = wave & 1;

  constexpr int kBM = 2 * WM * 16;
  constexpr int kBN = 2 * WN * 16;
  const int64_t m_block = static_cast<int64_t>(blockIdx.y) * kBM;
  const int64_t n_block = static_cast<int64_t>(blockIdx.x) * kBN;

  int32_t staged_base = -1;
  if (LDS_LUT) {
    staged_base = cboff[n_block < N ? n_block : (N - 1)];
    const void* src =
        (SRC == CB_SRC_BF16)
            ? static_cast<const void*>(
                  reinterpret_cast<const uint16_t*>(cbp) + staged_base)
            : static_cast<const void*>(
                  reinterpret_cast<const uint8_t*>(cbp) + staged_base);
    stage_lut_bf16(lut, src, lut_elems, SRC, tid, 128);
    __syncthreads();
  }

  const SubSplit<4, 2> sp(k_bits);
  const uint64_t mask = code_mask_of(k_bits);

  // This lane's row within each A tile and column within each B tile.
  const int sub = lane & 15;
  const int64_t a_row_base = m_block + wave_m * (WM * 16) + sub;
  const int64_t b_col_base = n_block + wave_n * (WN * 16) + sub;

  // Per-B-fragment row context, hoisted out of the K loop.
  const uint8_t* b_row[WN];
  int32_t b_base[WN];
  bool b_lds[WN];
  bool b_ok[WN];
#pragma unroll
  for (int wn = 0; wn < WN; ++wn) {
    const int64_t n = b_col_base + wn * 16;
    b_ok[wn] = (n < N);
    const int64_t nc = b_ok[wn] ? n : 0;
    b_row[wn] = qw + nc * qw_stride;
    b_base[wn] = cboff[nc];
    b_lds[wn] = LDS_LUT && (b_base[wn] == staged_base);
  }

  pq_v8f acc[WM][WN];
#pragma unroll
  for (int wm = 0; wm < WM; ++wm)
#pragma unroll
    for (int wn = 0; wn < WN; ++wn)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[wm][wn][i] = 0.0f;

  for (int64_t k0 = 0; k0 < K; k0 += 16) {
    pq_v16s af[WM];
#pragma unroll
    for (int wm = 0; wm < WM; ++wm) {
      const int64_t m = a_row_base + wm * 16;
      if (m < M) {
        const uint16_t* p = x + m * K + k0;
        // 16 consecutive bf16 = two aligned 16-byte loads (k0 % 16 == 0 and x
        // is at least 16-byte aligned, so both are naturally aligned).
        const uint4 lo = *reinterpret_cast<const uint4*>(p);
        const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
        const uint32_t wlo[4] = {lo.x, lo.y, lo.z, lo.w};
        const uint32_t whi[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          af[wm][2 * i] = static_cast<short>(wlo[i] & 0xffffu);
          af[wm][2 * i + 1] = static_cast<short>(wlo[i] >> 16);
          af[wm][8 + 2 * i] = static_cast<short>(whi[i] & 0xffffu);
          af[wm][8 + 2 * i + 1] = static_cast<short>(whi[i] >> 16);
        }
      } else {
        zero_frag(af[wm]);
      }
    }

    pq_v16s bf[WN];
#pragma unroll
    for (int wn = 0; wn < WN; ++wn) {
      if (!b_ok[wn]) {
        zero_frag(bf[wn]);
      } else if (b_lds[wn]) {
        decode_b_frag<CB_SRC_BF16>(bf[wn], b_row[wn], lut, 0, sp, mask,
                                   k_bits, type_size, static_cast<int>(k0));
      } else {
        decode_b_frag<SRC>(bf[wn], b_row[wn], cbp, b_base[wn], sp, mask,
                           k_bits, type_size, static_cast<int>(k0));
      }
    }

#pragma unroll
    for (int wm = 0; wm < WM; ++wm)
#pragma unroll
      for (int wn = 0; wn < WN; ++wn) {
        acc[wm][wn] = pq_mma(af[wm], bf[wn], acc[wm][wn]);
      }
  }

  // Epilogue: D register i of lane l holds row (2*i + l/16), col (l % 16) of
  // the 16x16 tile; the per-output-channel scale multiplies in f32 and the
  // result rounds to bf16 once.
  const int row_half = lane >> 4;
#pragma unroll
  for (int wn = 0; wn < WN; ++wn) {
    const int64_t n = b_col_base + wn * 16;
    if (n >= N) continue;
    const float sc = scale[n];
#pragma unroll
    for (int wm = 0; wm < WM; ++wm) {
      const int64_t m_tile = a_row_base - sub + wm * 16;   // tile origin row
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int64_t m = m_tile + 2 * i + row_half;
        if (m < M) {
          y[m * N + n] = f32_to_bf16_rn(acc[wm][wn][i] * sc);
        }
      }
    }
  }
}

// ===========================================================================
// Host launcher.
// ===========================================================================
bool cb_gemm_uses_wmma() {
  // Resolved in the DEVICE pass and reported through a kernel so the host can
  // never disagree with what was actually compiled.
  return true;  // overwritten by cb_gemm_probe_wmma() below when queried
}

__global__ void cb_gemm_caps_kernel(int* out) {
  if (threadIdx.x == 0) out[0] = PQ_WMMA_BF16_OK ? 1 : 0;
}

#define PQ_GEMM_LAUNCH_S(WM_, WN_, LDS_, SRC_)                               \
  hipLaunchKernelGGL(                                                        \
      HIP_KERNEL_NAME(cb_gemm_fp8_wmma_kernel<WM_, WN_, LDS_, SRC_>),        \
      dim3(gx, gy), dim3(128), (LDS_) ? lds_bytes : 0, stream, x, qw, cbp,   \
      cboff, scale, y, M, N, K, qw_stride, k_bits, type_size, lut_elems)
#define PQ_GEMM_LAUNCH(WM_, WN_, LDS_)                                       \
  do {                                                                       \
    if (src_kind == CB_SRC_BF16) {                                           \
      PQ_GEMM_LAUNCH_S(WM_, WN_, LDS_, CB_SRC_BF16);                         \
    } else {                                                                 \
      PQ_GEMM_LAUNCH_S(WM_, WN_, LDS_, CB_SRC_E4M3);                         \
    }                                                                        \
  } while (0)

void launch_cb_gemm_fp8(const uint16_t* x, const uint8_t* qw,
                        const void* cbp, const int32_t* cboff,
                        const float* scale, uint16_t* y, int64_t M, int64_t N,
                        int64_t K, int64_t qw_stride, int k_bits,
                        int type_size, bool use_lds, int wm_hint, int src_kind,
                        hipStream_t stream) {
  const int lut_elems = pq_lut_elems_fp8(k_bits);
  const int lds_bytes = pq_lut_lds_bytes(lut_elems);
  constexpr int WN = 2;
  const int64_t bn = 2 * WN * 16;                    // 64
  const unsigned gx = static_cast<unsigned>((N + bn - 1) / bn);
  // Pick the M tile so a short prefill batch does not pay for empty rows:
  // WM = 1 -> 32 rows, 2 -> 64, 4 -> 128.
  int wm = wm_hint;
  if (wm <= 0) wm = (M <= 32) ? 1 : ((M <= 64) ? 2 : 4);
  if (wm == 1) {
    const unsigned gy = static_cast<unsigned>((M + 31) / 32);
    if (use_lds) { PQ_GEMM_LAUNCH(1, WN, true); }
    else { PQ_GEMM_LAUNCH(1, WN, false); }
  } else if (wm == 2) {
    const unsigned gy = static_cast<unsigned>((M + 63) / 64);
    if (use_lds) { PQ_GEMM_LAUNCH(2, WN, true); }
    else { PQ_GEMM_LAUNCH(2, WN, false); }
  } else {
    const unsigned gy = static_cast<unsigned>((M + 127) / 128);
    if (use_lds) { PQ_GEMM_LAUNCH(4, WN, true); }
    else { PQ_GEMM_LAUNCH(4, WN, false); }
  }
}

}  // namespace pq_hip
