// Standalone parity + benchmark harness for the HIP CB kernels.
//
// Deliberately torch-free: it links the plain C++ launchers in cb_gemv_hip.hip
// and cb_gemm_hip.hip directly, so kernel correctness can be established on a
// ROCm box whether or not a HIP-enabled torch is installed, and so a parity
// failure is never confounded by the python extension plumbing.
//
// The reference decoder here is written from LAYOUT.md §1.1 in the most naive
// way possible — one bit at a time, LSB first — precisely so that it shares no
// code with the kernels' aligned-word extraction.  If the two agree, the
// aligned-word trick is right.
//
// Build:
//   hipcc --offload-arch=gfx1151 -O3 -L/usr/lib64 -lamdhip64 \
//         cb_hip_selftest.hip cb_gemv_hip.hip cb_gemm_hip.hip -o cb_hip_selftest
// Run:
//   ./cb_hip_selftest            # parity only
//   ./cb_hip_selftest --bench    # parity + benchmarks
#include "cb_decode_hip.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

namespace pq_hip {
void launch_cb_gemv_fp8(const uint16_t*, const uint8_t*, const void*,
                        const int32_t*, const float*, uint16_t*, int, int64_t,
                        int64_t, int64_t, int, int, int, bool, int, int,
                        hipStream_t);
void launch_cb_gemv_fp4_v2(const uint16_t*, const uint8_t*, const uint16_t*,
                           const int32_t*, const float*, uint16_t*, int,
                           int64_t, int64_t, int64_t, int, int, int, int, bool,
                           int, hipStream_t);
void launch_fp8_act_qdq(const uint16_t*, uint16_t*, int64_t, int64_t,
                        hipStream_t);
void launch_cb_expand_fp8(const uint8_t*, const uint8_t*, const int32_t*,
                          uint8_t*, int64_t, int64_t, int64_t, int, int,
                          hipStream_t);
void launch_cb_gemm_fp8(const uint16_t*, const uint8_t*, const void*,
                        const int32_t*, const float*, uint16_t*, int64_t,
                        int64_t, int64_t, int64_t, int, int, bool, int, int,
                        hipStream_t);
int cb_gemv_fp4_lut_elems(int, int);
}  // namespace pq_hip

using namespace pq_hip;

#define HIP_OK(expr)                                                      \
  do {                                                                    \
    hipError_t _e = (expr);                                               \
    if (_e != hipSuccess) {                                               \
      std::printf("FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, #expr,    \
                  hipGetErrorString(_e));                                 \
      std::exit(2);                                                       \
    }                                                                     \
  } while (0)

// --------------------------------------------------------------------------
// Host-side bit twiddling (independent of the device header on purpose).
// --------------------------------------------------------------------------
static uint16_t h_f32_to_bf16_rn(float f) {
  union { float f; uint32_t u; } c;
  c.f = f;
  if ((c.u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((c.u >> 16) | 0x40u);
  const uint32_t lsb = (c.u >> 16) & 1u;
  return (uint16_t)((c.u + 0x7fffu + lsb) >> 16);
}
static float h_bf16_to_f32(uint16_t v) {
  union { uint32_t u; float f; } c;
  c.u = (uint32_t)v << 16;
  return c.f;
}
// f32 -> e4m3fn (RN-even), then back: used to snap synthetic codebooks onto the
// grid, which is the FP8_CB contract (every codebook entry is grid-valued).
static uint8_t h_f32_to_e4m3(float f) {
  const uint32_t fp8_max = 1087u << 20, denorm_mask = 141u << 23;
  union { float f; uint32_t u; } c;
  c.f = f;
  uint32_t b = c.u;
  const uint32_t sign = b & 0x80000000u;
  b ^= sign;
  uint8_t r;
  if (b >= fp8_max) {
    r = 0x7f;
  } else if (b < (121u << 23)) {
    union { uint32_t u; float f; } a, d, s;
    a.u = b; d.u = denorm_mask;
    s.f = a.f + d.f;
    r = (uint8_t)(s.u - denorm_mask);
  } else {
    const uint8_t odd = (b >> 20) & 1u;
    b += ((uint32_t)(7 - 127) << 23) + 0x7FFFFu;
    b += odd;
    r = (uint8_t)(b >> 20);
  }
  return (uint8_t)(r | (sign >> 24));
}
static float h_e4m3_to_f32(uint8_t b) {
  const uint32_t s = (uint32_t)(b & 0x80u) << 24;
  const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
  union { uint32_t u; float f; } c;
  if (e == 0u) {
    if (m == 0u) { c.u = s; return c.f; }
    int p = (m >= 4u) ? 2 : ((m >= 2u) ? 1 : 0);
    c.u = s | ((uint32_t)(p - 9 + 127) << 23) | ((m << (23 - p)) & 0x7fffffu);
    return c.f;
  }
  if (e == 15u && m == 7u) { c.u = s | 0x7fc00000u; return c.f; }
  c.u = s | ((e + 120u) << 23) | (m << 20);
  return c.f;
}

// LAYOUT.md §1.1, verbatim: codeword v occupies stream bits [v*k, v*k+k), own
// LSB first.  One bit at a time, no word tricks.
static uint64_t ref_code(const uint8_t* sb, int v, int k) {
  uint64_t code = 0;
  const int base = v * k;
  for (int i = 0; i < k; ++i) {
    const int b = base + i;
    code |= (uint64_t)((sb[b >> 3] >> (b & 7)) & 1u) << i;
  }
  return code;
}
static std::vector<int> ref_split(int k, int n_sub) {
  std::vector<int> w(n_sub);
  const int base = k / n_sub, extra = k % n_sub;
  for (int i = 0; i < n_sub; ++i) w[i] = base + (i < extra ? 1 : 0);
  return w;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ull + 1u) {}
  uint32_t next() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(s >> 33);
  }
  float uniform() { return (float)next() / 2147483648.0f - 1.0f; }
};

// --------------------------------------------------------------------------
// Synthetic FP8_CB tensor: random index stream + grid-snapped codebook.
// --------------------------------------------------------------------------
struct Fp8Case {
  int k, type_size, n_sb;
  int64_t N, K;
  std::vector<uint8_t> qw;      // (N, n_sb*ts + PAD)
  int64_t qw_stride;
  std::vector<uint8_t> cb;      // e4m3 bytes
  std::vector<int32_t> cboff;
  std::vector<float> scale;
};

static const int kPad = 16;     // codec.PAD_BYTES

static Fp8Case make_fp8(int k, int64_t N, int64_t K, uint64_t seed,
                        int roles = 1) {
  Rng rng(seed);
  Fp8Case c;
  c.k = k;
  c.type_size = 4 * k;
  c.n_sb = (int)(K / 256);
  c.N = N;
  c.K = K;
  c.qw_stride = (int64_t)c.n_sb * c.type_size + kPad;
  c.qw.assign((size_t)(N * c.qw_stride), 0);
  for (int64_t r = 0; r < N; ++r) {
    for (int64_t b = 0; b < (int64_t)c.n_sb * c.type_size; ++b) {
      c.qw[(size_t)(r * c.qw_stride + b)] = (uint8_t)(rng.next() & 0xff);
    }
  }
  const std::vector<int> w = ref_split(k, 4);
  int per_role = 0;
  for (int i = 0; i < 4; ++i) per_role += 2 << w[i];
  c.cb.assign((size_t)per_role * roles, 0);
  for (size_t i = 0; i < c.cb.size(); ++i) {
    c.cb[i] = h_f32_to_e4m3(rng.uniform() * 4.0f);
  }
  c.cboff.assign((size_t)N, 0);
  c.scale.assign((size_t)N, 0.0f);
  for (int64_t r = 0; r < N; ++r) {
    const int role = (roles == 1) ? 0 : (int)((r * roles) / N);
    c.cboff[(size_t)r] = role * per_role;
    c.scale[(size_t)r] = 0.01f + 0.02f * (float)((rng.next() & 255) / 255.0);
  }
  return c;
}

// Reference decoded weight row.  `apply_scale` selects the v1 contract
// (bf16_rn(value * scale) per weight) vs raw codebook values (v2 / the GEMM,
// which scale in the epilogue).
static void ref_row_fp8_impl(const Fp8Case& c, int64_t n,
                             std::vector<float>& out, bool apply_scale) {
  const std::vector<int> w = ref_split(c.k, 4);
  std::vector<int> off(4, 0), elt(4, 0);
  for (int i = 1; i < 4; ++i) {
    off[i] = off[i - 1] + w[i - 1];
    elt[i] = elt[i - 1] + (2 << w[i - 1]);
  }
  out.assign((size_t)c.K, 0.0f);
  const uint8_t* row = &c.qw[(size_t)(n * c.qw_stride)];
  const int32_t base = c.cboff[(size_t)n];
  const float sc = c.scale[(size_t)n];
  for (int s = 0; s < c.n_sb; ++s) {
    const uint8_t* sb = row + (size_t)s * c.type_size;
    for (int v = 0; v < 32; ++v) {
      const uint64_t code = ref_code(sb, v, c.k);
      for (int i = 0; i < 4; ++i) {
        const uint32_t idx =
            (uint32_t)(code >> off[i]) & (uint32_t)((1u << w[i]) - 1u);
        for (int l = 0; l < 2; ++l) {
          const uint8_t byte = c.cb[(size_t)(base + elt[i] + idx * 2 + l)];
          const float val = h_e4m3_to_f32(byte);
          out[(size_t)(s * 256 + v * 8 + i * 2 + l)] =
              apply_scale ? h_bf16_to_f32(h_f32_to_bf16_rn(val * sc)) : val;
        }
      }
    }
  }
}

// --------------------------------------------------------------------------
static void ref_row_fp8(const Fp8Case& c, int64_t n, std::vector<float>& o) {
  ref_row_fp8_impl(c, n, o, true);
}
static void ref_row_fp8_raw(const Fp8Case& c, int64_t n,
                            std::vector<float>& o) {
  ref_row_fp8_impl(c, n, o, false);
}

// Comparison gate.  The WMMA unit is not bit-exact IEEE (wmma_probe Q5: ~0.5
// f32 ULP relative), and any two implementations differ by summation order
// anyway, so parity is a RELATIVE gate on the bf16 output plus a norm backstop
// — the same discipline as the CUDA lane's Triton-vs-CUDA check.
// --------------------------------------------------------------------------
static bool check(const char* tag, const std::vector<uint16_t>& got,
                  const std::vector<double>& ref, double norm_tol) {
  // Elementwise gate: the kernel output is bf16, so the tightest defensible
  // bound is ONE bf16 output-rounding step away from the bf16 rounding of the
  // fp64 truth (7 stored mantissa bits -> 2^-7 relative), plus a small
  // absolute term for values near zero.  Summation order and the WMMA unit's
  // ~0.5 f32 ULP (wmma_probe Q5) are both far inside that.  A norm backstop
  // catches a systematic bias that stays inside the per-element tolerance.
  double num = 0.0, den = 0.0, worst = 0.0;
  size_t worst_i = 0, nbad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = (double)h_bf16_to_f32(got[i]);
    const double r = ref[i];
    const double rb = (double)h_bf16_to_f32(h_f32_to_bf16_rn((float)r));
    const double d = std::fabs(g - rb);
    const double tol =
        std::fmax(std::fabs(g), std::fabs(rb)) * 0.0078125 + 1e-5;
    if (d > tol) {
      ++nbad;
      const double rel = d / (std::fabs(rb) + 1e-9);
      if (rel > worst) { worst = rel; worst_i = i; }
    }
    const double dn = g - r;
    num += dn * dn;
    den += r * r;
  }
  const double nrm = std::sqrt(num) / (std::sqrt(den) + 1e-12);
  const bool ok = (nbad == 0) && (nrm <= norm_tol);
  std::printf("  %-46s %zu/%zu elems >1 bf16 ULP (worst rel %.2e)  norm-rel "
              "%.3e  %s\n",
              tag, nbad, ref.size(), worst, nrm, ok ? "PASS" : "FAIL");
  (void)worst_i;
  return ok;
}

static int g_fail = 0;
static double g_warm_s = 45.0;  // ~45 s of load to reach the ~2.9 GHz boost

// --------------------------------------------------------------------------
// FP8_CB GEMV parity.
// --------------------------------------------------------------------------
static void test_gemv_fp8(int k, int M, int64_t N, int64_t K, bool use_lds,
                          int roles, int v2,
                          int src_kind = pq_hip::CB_SRC_E4M3) {
  Fp8Case c = make_fp8(k, N, K, (uint64_t)(k * 131 + M * 7 + roles), roles);
  Rng rng(99 + k);
  std::vector<uint16_t> hx((size_t)(M * K));
  for (auto& v : hx) v = h_f32_to_bf16_rn(rng.uniform());

  uint8_t *dqw, *dcb;
  int32_t* dcboff;
  float* dscale;
  uint16_t *dx, *dy;
  HIP_OK(hipMalloc(&dqw, c.qw.size()));
  HIP_OK(hipMalloc(&dcb, c.cb.size()));
  HIP_OK(hipMalloc(&dcboff, c.cboff.size() * 4));
  HIP_OK(hipMalloc(&dscale, c.scale.size() * 4));
  HIP_OK(hipMalloc(&dx, hx.size() * 2));
  HIP_OK(hipMalloc(&dy, (size_t)(M * N) * 2));
  HIP_OK(hipMemcpy(dqw, c.qw.data(), c.qw.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcb, c.cb.data(), c.cb.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcboff, c.cboff.data(), c.cboff.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dscale, c.scale.data(), c.scale.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));

  // Grid-source independence: build a BF16 sidecar holding the same values and
  // hand the kernel that instead.  e4m3 -> bf16 is exact, so the answer must be
  // IDENTICAL, not merely close -- which is what makes this a real test of the
  // LUT dtype contract rather than a smoke test.
  uint16_t* dcb16 = nullptr;
  if (src_kind == pq_hip::CB_SRC_BF16) {
    std::vector<uint16_t> h16(c.cb.size());
    for (size_t i = 0; i < c.cb.size(); ++i) {
      h16[i] = h_f32_to_bf16_rn(h_e4m3_to_f32(c.cb[i]));
    }
    HIP_OK(hipMalloc(&dcb16, h16.size() * 2));
    HIP_OK(hipMemcpy(dcb16, h16.data(), h16.size() * 2,
                     hipMemcpyHostToDevice));
  }
  const void* cb_ptr = (src_kind == pq_hip::CB_SRC_BF16)
                           ? static_cast<const void*>(dcb16)
                           : static_cast<const void*>(dcb);

  launch_cb_gemv_fp8(dx, dqw, cb_ptr, dcboff, dscale, dy, M, N, K, c.qw_stride,
                     k, c.type_size, 32, use_lds, src_kind, v2, 0);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  std::vector<uint16_t> hy((size_t)(M * N));
  HIP_OK(hipMemcpy(hy.data(), dy, hy.size() * 2, hipMemcpyDeviceToHost));

  std::vector<double> ref((size_t)(M * N), 0.0);
  std::vector<float> wrow;
  for (int64_t n = 0; n < N; ++n) {
    // v1: weights are bf16_rn(value * scale) and the scale is already folded
    // in.  v2: raw codebook values, scale applied once to the row total.
    if (v2) ref_row_fp8_raw(c, n, wrow);
    else ref_row_fp8(c, n, wrow);
    for (int m = 0; m < M; ++m) {
      double acc = 0.0;
      for (int64_t j = 0; j < K; ++j) {
        acc += (double)wrow[(size_t)j] *
               (double)h_bf16_to_f32(hx[(size_t)(m * K + j)]);
      }
      ref[(size_t)(m * N + n)] = v2 ? acc * (double)c.scale[(size_t)n] : acc;
    }
  }
  char tag[128];
  std::snprintf(tag, sizeof(tag),
                "GEMV fp8 k=%d M=%d N=%lld K=%lld %s roles=%d %s", k, M,
                (long long)N, (long long)K, use_lds ? "LDS" : "glb", roles,
                src_kind == pq_hip::CB_SRC_BF16 ? "bf16-grid" : "e4m3-grid");
  // v1 is the reference contract; v2 legitimately differs by the removed
  // per-weight rounding, so it gets the looser (still tight) gate.
  if (!check(tag, hy, ref, 5e-3)) ++g_fail;

  HIP_OK(hipFree(dqw)); HIP_OK(hipFree(dcb)); HIP_OK(hipFree(dcboff));
  HIP_OK(hipFree(dscale)); HIP_OK(hipFree(dx)); HIP_OK(hipFree(dy));
  if (dcb16) HIP_OK(hipFree(dcb16));
}

// --------------------------------------------------------------------------
// Expander byte-exactness: an INDEPENDENT decode of the same stream, so it
// pins the codeword extraction without going through the GEMV's arithmetic.
// --------------------------------------------------------------------------
static void test_expand(int k, int64_t N, int64_t K) {
  Fp8Case c = make_fp8(k, N, K, 4242 + k);
  uint8_t *dqw, *dcb, *dw;
  int32_t* dcboff;
  HIP_OK(hipMalloc(&dqw, c.qw.size()));
  HIP_OK(hipMalloc(&dcb, c.cb.size()));
  HIP_OK(hipMalloc(&dcboff, c.cboff.size() * 4));
  HIP_OK(hipMalloc(&dw, (size_t)(N * K)));
  HIP_OK(hipMemcpy(dqw, c.qw.data(), c.qw.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcb, c.cb.data(), c.cb.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcboff, c.cboff.data(), c.cboff.size() * 4,
                   hipMemcpyHostToDevice));
  launch_cb_expand_fp8(dqw, dcb, dcboff, dw, N, K, c.qw_stride, k,
                       c.type_size, 0);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  std::vector<uint8_t> hw((size_t)(N * K));
  HIP_OK(hipMemcpy(hw.data(), dw, hw.size(), hipMemcpyDeviceToHost));

  const std::vector<int> w = ref_split(k, 4);
  std::vector<int> off(4, 0), elt(4, 0);
  for (int i = 1; i < 4; ++i) {
    off[i] = off[i - 1] + w[i - 1];
    elt[i] = elt[i - 1] + (2 << w[i - 1]);
  }
  size_t bad = 0;
  for (int64_t n = 0; n < N; ++n) {
    const uint8_t* row = &c.qw[(size_t)(n * c.qw_stride)];
    for (int s = 0; s < c.n_sb; ++s) {
      const uint8_t* sb = row + (size_t)s * c.type_size;
      for (int v = 0; v < 32; ++v) {
        const uint64_t code = ref_code(sb, v, k);
        for (int i = 0; i < 4; ++i) {
          const uint32_t idx =
              (uint32_t)(code >> off[i]) & (uint32_t)((1u << w[i]) - 1u);
          for (int l = 0; l < 2; ++l) {
            const uint8_t want =
                c.cb[(size_t)(c.cboff[(size_t)n] + elt[i] + idx * 2 + l)];
            if (hw[(size_t)(n * K + s * 256 + v * 8 + i * 2 + l)] != want)
              ++bad;
          }
        }
      }
    }
  }
  std::printf("  %-46s %zu/%lld mismatched bytes  %s\n",
              (std::string("EXPAND fp8 k=") + std::to_string(k)).c_str(), bad,
              (long long)(N * K), bad == 0 ? "PASS" : "FAIL");
  if (bad) ++g_fail;
  HIP_OK(hipFree(dqw)); HIP_OK(hipFree(dcb)); HIP_OK(hipFree(dcboff));
  HIP_OK(hipFree(dw));
}

// --------------------------------------------------------------------------
// WMMA GEMM parity, against the same reference weights as the GEMV.
// --------------------------------------------------------------------------
static void test_gemm_fp8(int k, int64_t M, int64_t N, int64_t K,
                          bool use_lds) {
  Fp8Case c = make_fp8(k, N, K, 777 + k * 3 + M);
  Rng rng(31 + k);
  std::vector<uint16_t> hx((size_t)(M * K));
  for (auto& v : hx) v = h_f32_to_bf16_rn(rng.uniform());

  uint8_t *dqw, *dcb;
  int32_t* dcboff;
  float* dscale;
  uint16_t *dx, *dy;
  HIP_OK(hipMalloc(&dqw, c.qw.size()));
  HIP_OK(hipMalloc(&dcb, c.cb.size()));
  HIP_OK(hipMalloc(&dcboff, c.cboff.size() * 4));
  HIP_OK(hipMalloc(&dscale, c.scale.size() * 4));
  HIP_OK(hipMalloc(&dx, hx.size() * 2));
  HIP_OK(hipMalloc(&dy, (size_t)(M * N) * 2));
  HIP_OK(hipMemcpy(dqw, c.qw.data(), c.qw.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcb, c.cb.data(), c.cb.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcboff, c.cboff.data(), c.cboff.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dscale, c.scale.data(), c.scale.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));

  launch_cb_gemm_fp8(dx, dqw, dcb, dcboff, dscale, dy, M, N, K, c.qw_stride, k,
                     c.type_size, use_lds, 0, pq_hip::CB_SRC_E4M3, 0);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  std::vector<uint16_t> hy((size_t)(M * N));
  HIP_OK(hipMemcpy(hy.data(), dy, hy.size() * 2, hipMemcpyDeviceToHost));

  // The GEMM applies the channel scale in the f32 epilogue (no per-weight
  // round), so its reference is value * x summed, times scale.
  const std::vector<int> w = ref_split(k, 4);
  std::vector<int> off(4, 0), elt(4, 0);
  for (int i = 1; i < 4; ++i) {
    off[i] = off[i - 1] + w[i - 1];
    elt[i] = elt[i - 1] + (2 << w[i - 1]);
  }
  std::vector<double> ref((size_t)(M * N), 0.0);
  std::vector<float> raw((size_t)K);
  for (int64_t n = 0; n < N; ++n) {
    const uint8_t* row = &c.qw[(size_t)(n * c.qw_stride)];
    const int32_t base = c.cboff[(size_t)n];
    for (int s = 0; s < c.n_sb; ++s) {
      const uint8_t* sb = row + (size_t)s * c.type_size;
      for (int v = 0; v < 32; ++v) {
        const uint64_t code = ref_code(sb, v, k);
        for (int i = 0; i < 4; ++i) {
          const uint32_t idx =
              (uint32_t)(code >> off[i]) & (uint32_t)((1u << w[i]) - 1u);
          for (int l = 0; l < 2; ++l) {
            raw[(size_t)(s * 256 + v * 8 + i * 2 + l)] =
                h_e4m3_to_f32(c.cb[(size_t)(base + elt[i] + idx * 2 + l)]);
          }
        }
      }
    }
    for (int64_t m = 0; m < M; ++m) {
      double acc = 0.0;
      for (int64_t j = 0; j < K; ++j) {
        acc += (double)raw[(size_t)j] *
               (double)h_bf16_to_f32(hx[(size_t)(m * K + j)]);
      }
      ref[(size_t)(m * N + n)] = acc * (double)c.scale[(size_t)n];
    }
  }
  char tag[128];
  std::snprintf(tag, sizeof(tag), "GEMM fp8 k=%d M=%lld N=%lld K=%lld %s", k,
                (long long)M, (long long)N, (long long)K,
                use_lds ? "LDS" : "glb");
  if (!check(tag, hy, ref, 5e-3)) ++g_fail;

  HIP_OK(hipFree(dqw)); HIP_OK(hipFree(dcb)); HIP_OK(hipFree(dcboff));
  HIP_OK(hipFree(dscale)); HIP_OK(hipFree(dx)); HIP_OK(hipFree(dy));
}

// --------------------------------------------------------------------------
// Benchmarks.
// --------------------------------------------------------------------------
// SUSTAINED-CLOCK WARMUP.  gfx1151 idles at ~1.2 GHz and needs tens of seconds
// of continuous load to reach its ~2.9 GHz boost.  A conventional
// "5 warmup iterations" costs single-digit MILLISECONDS here and measures the
// idle clock -- which silently understates every kernel by ~2x and, worse,
// understates the MACHINE too, so a kernel can look close to a ceiling that is
// itself an artifact.  This ran once before that was understood; every number
// taken without it has been discarded.  Call before any timing.
static void sustain_clock(void (*fn)(void*), void* ctx, double seconds) {
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a, 0));
  float elapsed = 0.0f;
  while (elapsed < seconds * 1000.0) {
    for (int i = 0; i < 50; ++i) fn(ctx);
    HIP_OK(hipEventRecord(b, 0));
    HIP_OK(hipEventSynchronize(b));
    HIP_OK(hipEventElapsedTime(&elapsed, a, b));
  }
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
}

// Report the clock actually achieved, so a number can never again be quoted
// without knowing which clock produced it.
static int current_sclk_mhz() {
  int mhz = -1;
  FILE* f = popen("rocm-smi --showclocks 2>/dev/null | grep -i sclk", "r");
  if (f) {
    char buf[512];
    while (fgets(buf, sizeof(buf), f)) {
      const char* p = strstr(buf, "(");
      if (p && sscanf(p, "(%dMhz", &mhz) == 1) break;
      p = strstr(buf, ": ");
      if (p && sscanf(p, ": %dMhz", &mhz) == 1) break;
    }
    pclose(f);
  }
  return mhz;
}

static double time_ms(void (*fn)(void*), void* ctx, int iters) {
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  for (int i = 0; i < 20; ++i) fn(ctx);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipEventRecord(a, 0));
  for (int i = 0; i < iters; ++i) fn(ctx);
  HIP_OK(hipEventRecord(b, 0));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0.0f;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return (double)ms / iters;
}

struct BenchCtx {
  const uint16_t* x;
  const uint8_t* qw;
  const uint8_t* cb;
  const int32_t* cboff;
  const float* scale;
  uint16_t* y;
  int M;
  int64_t N, K, stride;
  int k, ts, rows_per_wg, wm;
  bool lds;
};
static void run_gemv(void* p) {
  BenchCtx* c = (BenchCtx*)p;
  launch_cb_gemv_fp8(c->x, c->qw, c->cb, c->cboff, c->scale, c->y, c->M, c->N,
                     c->K, c->stride, c->k, c->ts, c->rows_per_wg, c->lds,
                     pq_hip::CB_SRC_E4M3, 0, 0);
}
static void run_gemm(void* p) {
  BenchCtx* c = (BenchCtx*)p;
  launch_cb_gemm_fp8(c->x, c->qw, c->cb, c->cboff, c->scale, c->y, c->M, c->N,
                     c->K, c->stride, c->k, c->ts, c->lds, c->wm,
                     pq_hip::CB_SRC_E4M3, 0);
}

static void bench(int k, int64_t N, int64_t K) {
  Fp8Case c = make_fp8(k, N, K, 5150 + k);
  const int Mmax = 512;
  Rng rng(5);
  std::vector<uint16_t> hx((size_t)((int64_t)Mmax * K));
  for (auto& v : hx) v = h_f32_to_bf16_rn(rng.uniform());
  uint8_t *dqw, *dcb;
  int32_t* dcboff;
  float* dscale;
  uint16_t *dx, *dy;
  HIP_OK(hipMalloc(&dqw, c.qw.size()));
  HIP_OK(hipMalloc(&dcb, c.cb.size()));
  HIP_OK(hipMalloc(&dcboff, c.cboff.size() * 4));
  HIP_OK(hipMalloc(&dscale, c.scale.size() * 4));
  HIP_OK(hipMalloc(&dx, hx.size() * 2));
  HIP_OK(hipMalloc(&dy, (size_t)Mmax * N * 2));
  HIP_OK(hipMemcpy(dqw, c.qw.data(), c.qw.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcb, c.cb.data(), c.cb.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcboff, c.cboff.data(), c.cboff.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dscale, c.scale.data(), c.scale.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));

  const double packed_gb = (double)(N * c.n_sb * c.type_size) / 1e9;
  std::printf("\n  --- k=%d  N=%lld K=%lld  packed weight %.3f GB ---\n", k,
              (long long)N, (long long)K, packed_gb);
  BenchCtx ctx{dx, dqw, dcb, dcboff, dscale, dy, 1, N, K, c.qw_stride,
               k, c.type_size, 32, 0, true};
  // Drive the clock up BEFORE measuring anything (see sustain_clock).
  ctx.M = 16;
  sustain_clock(run_gemv, &ctx, g_warm_s);
  std::printf("    [clock after %.0f s sustained load: %d MHz]\n", g_warm_s,
              current_sclk_mhz());
  for (int M : {1, 4, 8, 16}) {
    ctx.M = M;
    for (bool lds : {true, false}) {
      ctx.lds = lds;
      const double ms = time_ms(run_gemv, &ctx, 50);
      std::printf("    GEMV M=%-3d %s  %8.3f ms   %6.1f GB/s (packed)   "
                  "%6.2f GFLOP/s\n",
                  M, lds ? "LDS-LUT " : "global  ", ms, packed_gb / (ms / 1e3),
                  2.0 * M * N * K / (ms / 1e3) / 1e9);
    }
  }
  for (int M : {32, 64, 128, 256, 512}) {
    ctx.M = M;
    ctx.lds = true;
    const double ms = time_ms(run_gemm, &ctx, 20);
    std::printf("    GEMM M=%-3d LDS-LUT   %8.3f ms   %6.1f GB/s (packed)   "
                "%6.1f GFLOP/s\n",
                M, ms, packed_gb / (ms / 1e3),
                2.0 * M * N * K / (ms / 1e3) / 1e9);
  }
  HIP_OK(hipFree(dqw)); HIP_OK(hipFree(dcb)); HIP_OK(hipFree(dcboff));
  HIP_OK(hipFree(dscale)); HIP_OK(hipFree(dx)); HIP_OK(hipFree(dy));
}

int main(int argc, char** argv) {
  bool do_bench = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
    if (std::strncmp(argv[i], "--warm=", 7) == 0)
      g_warm_s = atof(argv[i] + 7);
  }

  hipDeviceProp_t p;
  HIP_OK(hipGetDeviceProperties(&p, 0));
  std::printf("=== gridbook HIP CB self-test on %s (%s) ===\n", p.name,
              p.gcnArchName);
  if (p.warpSize != 32) {
    std::printf("FATAL: these kernels assume wave32 (device reports %d)\n",
                p.warpSize);
    return 2;
  }

  std::printf("\n[parity] FP8_CB codeword extraction + GEMV\n");
  // Rung ladder: the two ends, both step-4 rungs and the odd rungs whose
  // ceil-first sub-split makes the sub-tables uneven.
  for (int k : {28, 29, 32, 33, 36, 40, 44, 45, 47, 48}) {
    test_gemv_fp8(k, 1, 128, 1024, true, 1, 0);
  }
  for (int M : {1, 2, 4, 8, 16}) test_gemv_fp8(44, M, 96, 768, true, 1, 0);
  test_gemv_fp8(44, 4, 96, 768, false, 1, 0);        // global-LUT variant
  test_gemv_fp8(44, 4, 96, 768, true, 3, 0);         // fused: 3 roles
  test_gemv_fp8(44, 4, 96, 768, true, 1, 1);         // decode contract v2
  test_gemv_fp8(48, 1, 64, 2048, true, 1, 0);        // largest LUT, deep K

  std::printf("\n[parity] LUT dtype contract: bf16-grid sidecar == e4m3-grid\n");
  for (int k : {28, 36, 44, 48}) {
    test_gemv_fp8(k, 4, 96, 768, true, 1, 0, pq_hip::CB_SRC_BF16);
    test_gemv_fp8(k, 4, 96, 768, false, 1, 0, pq_hip::CB_SRC_BF16);
  }

  std::printf("\n[parity] transient expander (independent decode)\n");
  for (int k : {28, 36, 44, 47, 48}) test_expand(k, 64, 512);

  std::printf("\n[parity] FP8_CB WMMA prefill GEMM\n");
  for (int k : {36, 44, 48}) {
    test_gemm_fp8(k, 32, 128, 1024, true);
    test_gemm_fp8(k, 64, 96, 768, true);
  }
  test_gemm_fp8(44, 17, 96, 768, true);   // ragged M (edge tile)
  test_gemm_fp8(44, 32, 100, 768, true);  // ragged N (edge tile)
  test_gemm_fp8(44, 200, 128, 1024, true);
  test_gemm_fp8(44, 64, 96, 768, false);  // global-LUT variant

  if (do_bench) {
    std::printf("\n[bench] shapes from the shipped artifacts\n");
    bench(44, 5120, 4096);   // 27B-class dense MLP-ish
    bench(36, 4096, 4096);
    // LDS-LUT crossover sweep: the staged table grows 16->32 KiB across these
    // four rungs, and LDS occupancy is what decides whether staging pays.
    bench(45, 4096, 4096);
    bench(46, 4096, 4096);
    bench(47, 4096, 4096);
    bench(48, 4096, 4096);   // largest LUT (32 KiB)
    // Larger than the 32 MiB MALL, so the packed stream really comes from
    // DRAM -- the numbers above are cache-resident and flatter than serving.
    bench(44, 16384, 8192);
  }

  std::printf("\n=== %s (%d failures) ===\n", g_fail ? "FAIL" : "ALL PASS",
              g_fail);
  return g_fail ? 1 : 0;
}
