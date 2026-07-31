// Shared CB (codebook) decode primitives for the ROCm / HIP serving path.
//
// This header is the byte-for-byte HIP mirror of the on-disk container spec in
// docs/lanes/nvfp4-cb/LAYOUT.md, and of the CUDA reference implementation in
// gridbook/csrc/cb_gemv.cu.  Nothing here allocates, launches or touches torch:
// it is pure decode arithmetic, so the GEMV and the WMMA GEMM share ONE
// implementation of the format and cannot drift (the same reason cb_gemv.cu
// factors `fp8_decode_fma` / `fp4v2_decode_fma` out of its two schedules).
//
// ---------------------------------------------------------------------------
// Format recap (LAYOUT.md §1, cited per-function below)
// ---------------------------------------------------------------------------
//   * a codeword is a d = 8 vector of grid values; a k-bit index selects it;
//   * 32 codewords = one 256-weight SUPERBLOCK along the input dim;
//   * the 32 codewords are one LSB-first bitstream of 32k bits = 4k bytes;
//   * fp4 superblocks then carry a scale section (16 B v1 / 9 B two-tier v2);
//     fp8 superblocks carry none — fp8 scales are per-output-channel fp32 in a
//     separate tensor.  So type_size = 4k (fp8) / 4k+16 (fp4 v1) / 4k+9 (v2).
//
// ---------------------------------------------------------------------------
// Hardware assumptions (MEASURED on gfx1151 / Strix Halo, see csrc_hip/README)
// ---------------------------------------------------------------------------
//   * wave32.  Every wave-level primitive below assumes 32 lanes and the host
//     launchers assert `props.warpSize == 32` before launching.
//   * 64 KiB LDS per workgroup.  The LUT-residency budget is computed by
//     `pq_lut_elems_*` / `pq_lut_lds_bytes` and checked against the device
//     limit; the LUT is materialised as bf16, so it is 2 B per entry.
//   * No fp8 matrix instruction (RDNA3.5 WMMA covers f16/bf16/iu8/iu4 only) —
//     so fp8-CB here means fp8 STORAGE decoded to bf16 for WMMA compute.
//
#pragma once

#include <hip/hip_runtime.h>

#include <stdint.h>

#define PQ_DEVINL __device__ __forceinline__

namespace pq_hip {

// --- format constants (mirror gridbook/codec.py:16-18) ---------------------
constexpr int kWave = 32;           // wave32; asserted host-side
constexpr int kVecDim = 8;          // d, weights per codeword
constexpr int kSuperblock = 256;    // weights per superblock
constexpr int kFp4Group = 16;       // fp4 group-16 scale granularity
constexpr int kCodewordsPerSb = kSuperblock / kVecDim;   // 32 == wave size

constexpr float kFp8Max = 448.0f;
constexpr float kMinActScale = 1.0f / (448.0f * 512.0f);
// torch computes `tensor / scalar` as a RECIPROCAL MULTIPLY (a * f32(1/448)),
// not a true division, and the two differ by 1 f32 ULP for some amax.  The
// activation QDQ must match codec.fp8_dynamic_act_qdq bit-for-bit, so the
// reciprocal is replicated rather than "corrected" (cb_gemv.cu:67-70).
constexpr float kInvFp8Max = 1.0f / 448.0f;

// ---------------------------------------------------------------------------
// Scalar conversions.  All three are exact-by-construction, not approximations:
// e4m3 -> f32 is a bit reassembly, bf16 <-> f32 is a shift plus RN-even.
// ---------------------------------------------------------------------------

PQ_DEVINL float bf16_to_f32(uint16_t v) {
  union { uint32_t u; float f; } c;
  c.u = static_cast<uint32_t>(v) << 16;
  return c.f;
}

// Round-to-nearest-even f32 -> bf16, identical to CUDA `__float2bfloat16_rn`
// and to torch's `.to(torch.bfloat16)`.  NaN is quieted rather than allowed to
// round into an infinity, matching both.
PQ_DEVINL uint16_t f32_to_bf16_rn(float f) {
  union { float f; uint32_t u; } c;
  c.f = f;
  if ((c.u & 0x7fffffffu) > 0x7f800000u) {          // NaN in -> quiet NaN out
    return static_cast<uint16_t>((c.u >> 16) | 0x0040u);
  }
  const uint32_t lsb = (c.u >> 16) & 1u;
  const uint32_t rounded = c.u + 0x7fffu + lsb;     // RN-even
  return static_cast<uint16_t>(rounded >> 16);
}

// e4m3fn (1/4/3, bias 7, NO infinities, 0x7f/0xff are NaN) -> f32, exact.
//
// CUDA reaches this through `__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)` then
// half->float; every e4m3 value is exactly representable in fp16 (max 448 <
// 65504, min subnormal 2^-9 > 2^-24) so that chain is exact, and so is this
// one — the two agree bit-for-bit on all 256 codes.  Verified exhaustively by
// tests/test_hip_decode_parity.py::test_e4m3_decode_all_256_codes.
PQ_DEVINL float e4m3_to_f32(uint8_t b) {
  const uint32_t s = static_cast<uint32_t>(b & 0x80u) << 24;
  const uint32_t e = (b >> 3) & 0xFu;
  const uint32_t m = b & 0x7u;
  union { uint32_t u; float f; } c;
  if (e == 0u) {                                    // zero or subnormal
    if (m == 0u) { c.u = s; return c.f; }           // +/-0
    // subnormal value = m * 2^-9.  Normalise: m = 1.f * 2^p, p = 31 - clz(m).
    const int p = 31 - __clz(static_cast<int>(m));  // 0, 1 or 2
    const uint32_t exp = static_cast<uint32_t>(p - 9 + 127);
    const uint32_t man = (m << (23 - p)) & 0x7fffffu;
    c.u = s | (exp << 23) | man;
    return c.f;
  }
  if (e == 15u && m == 7u) {                        // the only NaN encodings
    c.u = s | 0x7fc00000u;
    return c.f;
  }
  c.u = s | ((e + 120u) << 23) | (m << 20);         // e - 7 + 127 == e + 120
  return c.f;
}

// Single-rounded f32 -> e4m3 (RN-even), a verbatim port of
// c10::detail::fp8e4m3fn_from_fp32_value.  The vendor conversion intrinsics on
// both platforms can double-round via fp16 and disagree with torch at half-ULP
// boundaries (seen live on CUDA: x = 0.7265625 landing on the adjacent code),
// which would break the bit-exact activation-QDQ contract.  Callers pre-clamp
// to +/-448 so the saturating branch is unreachable.
PQ_DEVINL uint8_t f32_to_e4m3_c10(float f) {
  constexpr uint32_t fp8_max = 1087u << 20;         // 480.0f, first non-e4m3fn
  constexpr uint32_t denorm_mask = 141u << 23;      // 2^(-121+127) magic
  union { float f; uint32_t u; } c;
  c.f = f;
  uint32_t f_bits = c.u;
  uint8_t result = 0u;
  const uint32_t sign = f_bits & 0x80000000u;
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = 0x7fu;                                 // NaN (unreachable)
  } else if (f_bits < (121u << 23)) {               // < 2^-6 -> subnormal
    union { uint32_t u; float f; } a, d;
    a.u = f_bits;
    d.u = denorm_mask;
    union { float f; uint32_t u; } r;
    r.f = a.f + d.f;
    f_bits = r.u;
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    const uint8_t mant_odd = (f_bits >> 20) & 1u;   // RN-even tie break
    f_bits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFFu;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

// ---------------------------------------------------------------------------
// Product-mode sub-index descriptor (LAYOUT.md §1.1 "product mode").
//
// The k index bits split across NSUB sub-tables "as even as possible, larger
// halves first" — the encoder's `_bit_split` (nvfp4_cb_formats.py:167-172) is
// ceil-first: sub i gets w_i = k/NSUB + (i < k%NSUB) bits.  Sub-index i sits at
// codeword bit offset off[i] = sum_{j<i} w_j (sub 0 in the LOW bits) and its
// table starts elt[i] = SUBDIM * sum_{j<i} 2^{w_j} ELEMENTS into the row's
// slice of the flat codebook (the concatenation built by
// codec.build_flat_codebook).  Even k reduces to the uniform split, so this is
// a strict generalisation and the odd rungs (K29, K33, K47, ...) are covered.
//
// All-constant unrolled scalar math: computed once at kernel top, lives in
// registers.  Identical to cb_gemv.cu:170-189.
// ---------------------------------------------------------------------------
template <int NSUB, int SUBDIM>
struct SubSplit {
  int off[NSUB];
  uint32_t mask[NSUB];
  int32_t elt[NSUB];
  PQ_DEVINL explicit SubSplit(int k_bits) {
    const int base = k_bits / NSUB, extra = k_bits % NSUB;
    int o = 0;
    int32_t e = 0;
#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
      const int w = base + (i < extra ? 1 : 0);
      off[i] = o;
      mask[i] = (1u << w) - 1u;
      elt[i] = e;
      o += w;
      e += static_cast<int32_t>(SUBDIM) << w;
    }
  }
};

// Total ELEMENTS in one product codebook block — the host-side twin of the
// loop above, used to size the LDS LUT.
__host__ __device__ inline int pq_codebook_elems(int k_bits, int n_sub,
                                                 int sub_dim) {
  const int base = k_bits / n_sub, extra = k_bits % n_sub;
  int e = 0;
  for (int i = 0; i < n_sub; ++i) {
    e += sub_dim << (base + (i < extra ? 1 : 0));
  }
  return e;
}

// ---------------------------------------------------------------------------
// GRID-SOURCE INDEPENDENCE (the LUT dtype contract)
// ---------------------------------------------------------------------------
// The codebook sidecar may store its entries as e4m3 bytes (the Blackwell
// convention, where a decoded tile must be a bit-standard fp8 tensor) or as
// bf16 (the natural convention on a platform with no fp8 hardware at all,
// where the e4m3 constraint buys nothing and only costs grid quality).  This
// kernel family is agnostic to that choice:
//
//   * the LDS LUT is ALWAYS materialised as bf16 bit patterns, and any e4m3 ->
//     bf16 conversion happens exactly ONCE, at LUT-fill time, never in the
//     gather;
//   * the hot loop gathers bf16 out of LDS and never learns what the sidecar
//     stored.
//
// This is free of accuracy consequences in the e4m3 direction: e4m3 has 3
// mantissa bits and bf16 has 7, so e4m3 -> bf16 is exact and
// e4m3 -> bf16 -> f32 equals e4m3 -> f32 bit-for-bit.  A bf16-grid sidecar is
// then a strict superset of the e4m3 grid at identical bytes and identical
// kernel cost.
//
// The cost that IS real: a bf16 LUT is 2x the bytes of an e4m3 one, which
// moves the LDS budget (see pq_lut_lds_bytes and the README table).  The
// per-gather conversion the CUDA lane's R6 work removed stays removed here —
// when the bf16 LUT does not fit LDS the kernel gathers from GLOBAL instead of
// re-introducing an ALU term, and at those rungs the measurement says global is
// the faster arm anyway.
enum CbGridSrc { CB_SRC_E4M3 = 0, CB_SRC_BF16 = 1 };

// e4m3 byte -> bf16 bit pattern, exact.  Used only at LUT fill (and on the
// global-gather fallback for an e4m3 sidecar).
PQ_DEVINL uint16_t e4m3_to_bf16_bits(uint8_t b) {
  const uint32_t s = static_cast<uint32_t>(b & 0x80u) << 8;   // -> bf16 bit 15
  const uint32_t e = (b >> 3) & 0xFu;
  const uint32_t m = b & 0x7u;
  if (e == 0u) {
    if (m == 0u) return static_cast<uint16_t>(s);             // +/-0
    const int p = 31 - __clz(static_cast<int>(m));            // 0, 1 or 2
    const uint32_t exp = static_cast<uint32_t>(p + 118);      // p - 9 + 127
    const uint32_t man = (m << (7 - p)) & 0x7fu;
    return static_cast<uint16_t>(s | (exp << 7) | man);
  }
  if (e == 15u && m == 7u) return static_cast<uint16_t>(s | 0x7fc0u);  // NaN
  return static_cast<uint16_t>(s | ((e + 120u) << 7) | (m << 4));
}

// Elements in one FP8_CB codebook block (product, n_sub = 4, sub_dim = 2).
__host__ __device__ inline int pq_lut_elems_fp8(int k_bits) {
  return pq_codebook_elems(k_bits, 4, 2);
}

// Elements in one NVFP4_CB block: product (n_sub = 2, sub_dim = 4) or the
// signed S-rungs' single 8-dim magnitude table indexed by k-8 bits.
__host__ __device__ inline int pq_lut_elems_fp4(int k_bits, int n_sub) {
  return (n_sub == 1) ? (8 << (k_bits - 8)) : pq_codebook_elems(k_bits, 2, 4);
}

// LDS bytes for a materialised LUT: bf16 in LDS regardless of sidecar dtype.
__host__ __device__ inline int pq_lut_lds_bytes(int n_elems) {
  return n_elems * 2;
}

// ---------------------------------------------------------------------------
// Codeword extraction (LAYOUT.md §1.1 "index stream").
//
// Codeword v of a superblock occupies stream bits [v*k, v*k+k), LSB first.  We
// read three ALIGNED 32-bit words covering that range and shift, exactly like
// the CUDA kernel — but here the aligned base is derived from the pointer
// itself rather than from a smem stage, so the same code serves fp8
// (type_size = 4k, every superblock 4-byte aligned) AND fp4 v2
// (type_size = 4k+9, superblocks at arbitrary byte phase).
//
// Read slack: the aligned base can sit up to 3 bytes BEFORE the codeword's
// first byte and we touch 12 bytes from it, so the read extends at most 11
// bytes past the codeword start; for the last codeword of the last superblock
// of the last row that is at most 11 bytes past the packed data.  The padded
// buffer supplies 16 (codec.PAD_BYTES) — this is the invariant that pad
// documents, and it must hold or the read is out of bounds.
//
// `bitpos` is the codeword's bit offset within the superblock (= lane * k for
// the GEMV lane mapping).  `sb` points at the superblock's first byte.
// ---------------------------------------------------------------------------
PQ_DEVINL uint64_t extract_code(const uint8_t* __restrict__ sb, int bitpos,
                                int k_bits, uint64_t code_mask) {
  const uint8_t* p = sb + (bitpos >> 3);
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const int phase = static_cast<int>(a & 3u);
  const uint32_t* w32 =
      reinterpret_cast<const uint32_t*>(a - static_cast<uintptr_t>(phase));
  const int rem = (phase << 3) + (bitpos & 7);      // 0..31
  const uint32_t w0 = w32[0];
  const uint32_t w1 = w32[1];
  uint64_t code =
      ((static_cast<uint64_t>(w1) << 32) | static_cast<uint64_t>(w0)) >> rem;
  // rem + k > 64 implies rem >= 17 (k <= 48), so the shift below is in [1,47]
  // and never the UB 64-bit shift.  Third word touched only when needed.
  if (rem + k_bits > 64) {
    code |= static_cast<uint64_t>(w32[2]) << (64 - rem);
  }
  return code & code_mask;
}

PQ_DEVINL uint64_t code_mask_of(int k_bits) {
  return (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
}

// ---------------------------------------------------------------------------
// Codebook gathers.
//
// FP8_CB, product mode, n_sub = 4, sub_dim = 2: sub-entry i is TWO adjacent
// two adjacent grid entries, so one paired read fetches both.  `base` is the
// row's ELEMENT offset (`cb_row_offset[n]`), or 0 when reading the LDS-resident
// copy of that row's block.  Element offsets are always even (elt[i] accumulates
// 2 << w and idx*2 is even), which is what makes the paired read aligned.
// ---------------------------------------------------------------------------
// Gather one 8-wide codeword as BF16 BIT PATTERNS.
//
// `SRC` names what the *table being read* holds, not what the artifact shipped:
// the LDS LUT is always CB_SRC_BF16 (materialised once by stage_lut_bf16), and
// only the global-gather fallback ever instantiates CB_SRC_E4M3.  So the
// per-gather conversion exists on exactly one path and is absent from the one
// that matters.
//
// Element offsets are always even (sp.elt accumulates 2 << w, idx*2 is even,
// and a row's base is a sum of whole blocks), so the bf16 pair is 4-byte
// aligned and loads as a single 32-bit read — one `ds_read_b32` per sub-index
// instead of two 16-bit reads.
template <int SRC>
PQ_DEVINL void gather_fp8_bits(uint64_t code, const void* __restrict__ cb,
                               int32_t base, const SubSplit<4, 2>& sp,
                               uint16_t* bits /* [8] */) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t idx = static_cast<uint32_t>(code >> sp.off[i]) & sp.mask[i];
    const int32_t elt = base + sp.elt[i] + static_cast<int32_t>(idx) * 2;
    if (SRC == CB_SRC_BF16) {
      const uint16_t* t = reinterpret_cast<const uint16_t*>(cb);
      const uint32_t two = *reinterpret_cast<const uint32_t*>(t + elt);
      bits[2 * i] = static_cast<uint16_t>(two & 0xffffu);
      bits[2 * i + 1] = static_cast<uint16_t>(two >> 16);
    } else {
      const uint8_t* t = reinterpret_cast<const uint8_t*>(cb);
      const uint16_t pair = *reinterpret_cast<const uint16_t*>(t + elt);
      bits[2 * i] = e4m3_to_bf16_bits(static_cast<uint8_t>(pair & 0xffu));
      bits[2 * i + 1] = e4m3_to_bf16_bits(static_cast<uint8_t>(pair >> 8));
    }
  }
}

// Float form for the GEMV's FMA chain.  bf16 -> f32 is a shift, and for an
// e4m3-sourced table e4m3 -> bf16 -> f32 equals e4m3 -> f32 exactly, so this
// changes no numerics relative to decoding e4m3 straight to f32.
template <int SRC>
PQ_DEVINL void gather_fp8_vec(uint64_t code, const void* __restrict__ cb,
                              int32_t base, const SubSplit<4, 2>& sp,
                              float* wv /* [8] */) {
  uint16_t bits[8];
  gather_fp8_bits<SRC>(code, cb, base, sp, bits);
#pragma unroll
  for (int j = 0; j < 8; ++j) wv[j] = bf16_to_f32(bits[j]);
}

// Raw e4m3 byte pair — the expander's output IS e4m3 bytes, so it reads the
// e4m3 sidecar directly and is the one consumer that stays byte-typed.
PQ_DEVINL void gather_fp8_vec_bytes(uint64_t code,
                                    const uint8_t* __restrict__ cb,
                                    int32_t base, const SubSplit<4, 2>& sp,
                                    uint8_t* bytes /* [8] */) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t idx = static_cast<uint32_t>(code >> sp.off[i]) & sp.mask[i];
    const int32_t elt = base + sp.elt[i] + static_cast<int32_t>(idx) * 2;
    const uint16_t pair = *reinterpret_cast<const uint16_t*>(cb + elt);
    bytes[2 * i] = static_cast<uint8_t>(pair & 0xffu);
    bytes[2 * i + 1] = static_cast<uint8_t>(pair >> 8);
  }
}

// NVFP4_CB product mode (n_sub = 2, sub_dim = 4) over the BF16 flat codebook.
// `base`/`elt` are ELEMENT (uint16) indices here, not byte offsets.
PQ_DEVINL void gather_fp4_vec(uint64_t code, const uint16_t* __restrict__ cb,
                              int32_t base, const SubSplit<2, 4>& sp,
                              float* wv /* [8] */) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t idx = static_cast<uint32_t>(code >> sp.off[i]) & sp.mask[i];
    const int32_t elt = base + sp.elt[i] + static_cast<int32_t>(idx) * 4;
#pragma unroll
    for (int local = 0; local < 4; ++local) {
      wv[i * 4 + local] = bf16_to_f32(cb[elt + local]);
    }
  }
}

// NVFP4_CB signed mode (S-rungs, n_sub == 1): the low 8 bits of the codeword
// are explicit signs (bit j set <=> coordinate j negative) and the remaining
// k-8 bits index ONE non-negative half-grid table of 8-dim entries
// (LAYOUT.md §1.1 "signed mode").  The sign is applied by flipping the bf16
// sign bit, which is exact.
PQ_DEVINL void gather_fp4_signed_vec(uint64_t code,
                                     const uint16_t* __restrict__ cb,
                                     int32_t base, float* wv /* [8] */) {
  const uint32_t sign8 = static_cast<uint32_t>(code & 0xffu);
  const int32_t elt = base + static_cast<int32_t>(code >> 8) * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint16_t b = cb[elt + j];
    b = static_cast<uint16_t>(b ^ (((sign8 >> j) & 1u) << 15));
    wv[j] = bf16_to_f32(b);
  }
}

// ---------------------------------------------------------------------------
// Two-tier (layout v2) scale composition.
// docs/lanes/nvfp4-cb/two-tier-scale-spec.md §1.1, and the producer/consumer
// pair prismaquant/nvfp4_cb_formats.py:_two_tier_scale_bytes /
// gridbook/codec.py:46-56 (build_compose_table).
//
//   scale_section = [ SUPER 1 B (E8M0, bias 127) | SUB 8 B (16 x 4-bit codes) ]
//   group g lives in sub byte g/2, EVEN g in the LOW nibble (LSB-first,
//   consistent with the index stream);
//   scale_g = T[c_g] * 2^(E-127), taken from the (256,16) fp32 compose table
//   the host builds once (`compose[E*16 + c]`).  Exact E4M3 by construction —
//   the encoder only ever emits legal (E, c) pairs — so this is a plain fp32
//   multiply with no cast and no rounding.
//
// Line-by-line correspondence with codec.build_compose_table:
//   codec:  compose = (T[None,:] * 2**(arange(256)[:,None] - 127)).f32
//   here :  compose[super_e * 16 + code16]           <- same flattening
//   codec:  group g of superblock s, byte g//2, nibble (g%2)*4
//   here :  sub_off = scale_off + 1 + (g >> 1);  (sub_byte >> ((g&1)*4)) & 0xF
// ---------------------------------------------------------------------------
PQ_DEVINL float compose_two_tier(const float* __restrict__ compose,
                                 uint8_t super_e, uint8_t sub_byte, int group) {
  const uint32_t code16 =
      static_cast<uint32_t>((sub_byte >> ((group & 1) * 4)) & 0xFu);
  return compose[static_cast<int>(super_e) * 16 + static_cast<int>(code16)];
}

// Byte offsets of the v2 scale section within a superblock (fp4 only).
PQ_DEVINL int two_tier_super_off(int k_bits) { return 4 * k_bits; }
PQ_DEVINL int two_tier_sub_off(int k_bits, int group) {
  return 4 * k_bits + 1 + (group >> 1);
}

// ---------------------------------------------------------------------------
// Wave-level helpers (wave32).
// ---------------------------------------------------------------------------

// Butterfly all-reduce: every lane ends holding the wave sum.  `__shfl_xor` is
// the HIP spelling; the explicit width argument keeps it wave32 even if the
// module is ever compiled for a wave64 target (where the host guard rejects
// the launch anyway).
PQ_DEVINL float wave_allreduce_add(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor(v, off, kWave);
  }
  return v;
}

// Cooperative LDS staging of a codebook block, MATERIALISED AS BF16 whatever
// the sidecar stored.  This is the single point where an e4m3 sidecar is
// converted — once per workgroup per element, not once per gather — so the hot
// loop is identical for both grid sources and carries no conversion ALU.
//
// The bf16 path copies 32 bits at a time (element counts are multiples of 256,
// so the pairing is always exact); the e4m3 path is element-wise because each
// byte becomes a different 16-bit value.
PQ_DEVINL void stage_lut_bf16(uint16_t* __restrict__ lds,
                              const void* __restrict__ src, int n_elems,
                              int src_kind, int tid, int nthreads) {
  if (src_kind == CB_SRC_BF16) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    uint32_t* d = reinterpret_cast<uint32_t*>(lds);
    const int pairs = n_elems >> 1;
    for (int i = tid; i < pairs; i += nthreads) d[i] = s[i];
    if ((n_elems & 1) && tid == 0) {
      lds[n_elems - 1] =
          reinterpret_cast<const uint16_t*>(src)[n_elems - 1];
    }
  } else {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    for (int i = tid; i < n_elems; i += nthreads) {
      lds[i] = e4m3_to_bf16_bits(s[i]);
    }
  }
}

// Byte-wise staging, still used by the fp4 path's bf16 table (already bf16 on
// disk, so a straight copy) and by any consumer that wants raw bytes.
PQ_DEVINL void stage_lut(uint8_t* __restrict__ lds,
                         const uint8_t* __restrict__ src, int bytes, int tid,
                         int nthreads) {
  const int words = bytes >> 2;
  uint32_t* d = reinterpret_cast<uint32_t*>(lds);
  const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
  for (int i = tid; i < words; i += nthreads) d[i] = s[i];
  for (int i = (words << 2) + tid; i < bytes; i += nthreads) lds[i] = src[i];
}

}  // namespace pq_hip
