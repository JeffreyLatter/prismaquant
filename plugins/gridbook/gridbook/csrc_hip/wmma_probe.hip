// gfx1151 (RDNA 3.5) WMMA capability + fragment-layout probe.
//
// Why this exists: the CB prefill GEMM decodes codewords DIRECTLY into WMMA
// operand registers (no LDS round-trip for A or B), which requires knowing
// which matrix element each (lane, register) slot of a fragment holds.  That
// mapping is an ISA property, not something to take on faith from a blog post,
// so this program establishes it on the real device and prints a verdict the
// kernel's static asserts are written against.
//
// It answers four questions:
//   Q1  which WMMA builtins exist for this target (compile-time __has_builtin);
//   Q2  device properties that the launchers assert (wave size, LDS/workgroup);
//   Q3  the *self-consistent* A/B/D fragment layout — see the note below;
//   Q4  whether f32 accumulation of a bf16 WMMA matches an fp64 CPU reference.
//
// On Q3, the important subtlety: a GEMM never needs the ABSOLUTE layout, only
// a self-consistent one.  If we always feed "output row r" through lane r's A
// registers and always read "output row r" back from whatever (lane, reg) slot
// the probe says holds it, any fixed permutation of the hardware's internal row
// labelling cancels out.  So the probe fixes the convention
//     A: lane l (l < 16) holds row l, register j holds k = j
//     B: lane l (l < 16) holds col l, register j holds k = j
//     (lanes 16..31 duplicate lanes 0..15 — required by the wave32 encoding)
// and then SEARCHES for the D mapping that makes the product come out right,
// reporting it.  If no candidate matches it dumps the raw registers next to the
// CPU reference so the mapping can be read off by hand.
//
// Build (Fedora 44 / ROCm 7.1.1 needs the explicit runtime link):
//   hipcc --offload-arch=gfx1151 -O3 -L/usr/lib64 -lamdhip64 \
//         wmma_probe.hip -o wmma_probe && ./wmma_probe
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// --------------------------------------------------------------------------
// Q1: capability table, resolved by the compiler for THIS --offload-arch.
//
// LANDMINE (hit on the first bring-up run): `__has_builtin` for a
// target-specific builtin must be evaluated in the DEVICE compilation pass.
// hipcc preprocesses the file twice; in the host pass the target is x86 and
// every `__builtin_amdgcn_*` reports ABSENT, so a naive file-scope `#if
// __has_builtin(...)` compiles the probe out and reports "no WMMA anywhere"
// on a device that has it.  Everything below is therefore keyed on
// `__HIP_DEVICE_COMPILE__`, and the capability table is produced BY A KERNEL
// so the answer comes from the device pass.
// --------------------------------------------------------------------------
#if defined(__HIP_DEVICE_COMPILE__)
#define PQ_DEV_HAS(x) __has_builtin(x)
#else
#define PQ_DEV_HAS(x) (-1) /* unknown: host pass */
#endif

#define PQ_HAS_WMMA_BF16 PQ_DEV_HAS(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
#define PQ_HAS_WMMA_F16 PQ_DEV_HAS(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
#define PQ_HAS_WMMA_IU8 PQ_DEV_HAS(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
#define PQ_HAS_WMMA_FP8_G12 \
  PQ_DEV_HAS(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
#define PQ_HAS_WMMA_FP8_PLAIN \
  PQ_DEV_HAS(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32)
#define PQ_HAS_SWMMAC_FP8 \
  PQ_DEV_HAS(__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32)

// The WMMA kernels' BODIES are conditional (so an absent builtin still
// compiles), but the symbols are always defined, so the host launch links.
#if !defined(__HIP_DEVICE_COMPILE__) || PQ_HAS_WMMA_BF16
#define PQ_WMMA_BF16_BODY 1
#else
#define PQ_WMMA_BF16_BODY 0
#endif

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

static __host__ __device__ inline uint16_t f32_to_bf16_rn(float f) {
  union { float f; uint32_t u; } c;
  c.f = f;
  if ((c.u & 0x7fffffffu) > 0x7f800000u) {
    return (uint16_t)((c.u >> 16) | 0x0040u);
  }
  const uint32_t lsb = (c.u >> 16) & 1u;
  return (uint16_t)((c.u + 0x7fffu + lsb) >> 16);
}

// --------------------------------------------------------------------------
// Q3/Q4 kernel: one wave computes D = A x B for 16x16x16 under the convention
// documented above, and writes out the raw per-(lane, reg) accumulator so the
// host can search for the element mapping.
// --------------------------------------------------------------------------
// Capability table, written from the DEVICE pass.
__global__ void caps_kernel(int* __restrict__ out) {
  if (threadIdx.x != 0) return;
  out[0] = PQ_HAS_WMMA_BF16;
  out[1] = PQ_HAS_WMMA_F16;
  out[2] = PQ_HAS_WMMA_IU8;
  out[3] = PQ_HAS_WMMA_FP8_PLAIN;
  out[4] = PQ_HAS_WMMA_FP8_G12;
  out[5] = PQ_HAS_SWMMAC_FP8;
#if defined(__gfx1151__)
  out[6] = 1151;
#elif defined(__gfx1150__)
  out[6] = 1150;
#elif defined(__gfx1100__)
  out[6] = 1100;
#else
  out[6] = 0;
#endif
  out[7] = (int)warpSize;
}

__global__ void wmma_probe_kernel(const uint16_t* __restrict__ a_rowmajor,
                                  const uint16_t* __restrict__ b_colmajor,
                                  float* __restrict__ d_raw) {
#if PQ_WMMA_BF16_BODY
  const int lane = threadIdx.x & 31;
  const int r = lane & 15;  // lanes 16..31 duplicate lanes 0..15

  v16s a, b;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    a[j] = (short)a_rowmajor[r * 16 + j];  // A[row=r][k=j]
    b[j] = (short)b_colmajor[r * 16 + j];  // B[k=j][col=r]
  }
  v8f acc;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

  acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);

#pragma unroll
  for (int i = 0; i < 8; ++i) d_raw[lane * 8 + i] = acc[i];
#else
  (void)a_rowmajor; (void)b_colmajor; (void)d_raw;
#endif
}

// Same product computed with plain f32 FMAs (one lane per output element pair)
// so the WMMA unit's arithmetic can be compared against the SM's own f32 path
// on identical inputs -- the honest way to attribute any discrepancy.
__global__ void scalar_ref_kernel(const uint16_t* __restrict__ a_rowmajor,
                                  const uint16_t* __restrict__ b_colmajor,
                                  float* __restrict__ d_raw) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m = 2 * i + (lane >> 4);
    const int n = lane & 15;
    float acc = 0.0f;
    for (int k = 0; k < 16; ++k) {
      union { uint32_t u; float f; } av, bv;
      av.u = (uint32_t)a_rowmajor[m * 16 + k] << 16;
      bv.u = (uint32_t)b_colmajor[n * 16 + k] << 16;
      acc = fmaf(av.f, bv.f, acc);
    }
    d_raw[lane * 8 + i] = acc;
  }
}

// A second kernel proving the accumulator chains correctly across K-steps
// (the GEMM's inner loop feeds acc back in), with a deliberately non-uniform
// operand so a broadcast bug cannot pass.
__global__ void wmma_chain_kernel(const uint16_t* __restrict__ a_rowmajor,
                                  const uint16_t* __restrict__ b_colmajor,
                                  int ksteps, float* __restrict__ d_raw) {
#if PQ_WMMA_BF16_BODY
  const int lane = threadIdx.x & 31;
  const int r = lane & 15;
  v8f acc;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
  for (int s = 0; s < ksteps; ++s) {
    v16s a, b;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      a[j] = (short)a_rowmajor[(s * 16 + r) * 16 + j];
      b[j] = (short)b_colmajor[(s * 16 + r) * 16 + j];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) d_raw[lane * 8 + i] = acc[i];
#else
  (void)a_rowmajor; (void)b_colmajor; (void)ksteps; (void)d_raw;
#endif
}

#define HIP_OK(expr)                                                       \
  do {                                                                     \
    hipError_t _e = (expr);                                                \
    if (_e != hipSuccess) {                                                \
      std::printf("FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, #expr,     \
                  hipGetErrorString(_e));                                  \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

// Candidate D-fragment mappings.  `id` is documented alongside the formula;
// the probe reports which one reproduces the reference product.
struct DMap {
  const char* name;
  int (*row)(int lane, int i);
  int (*col)(int lane, int i);
};
static int row_2i_half(int lane, int i) { return 2 * i + (lane >> 4); }
static int row_8half_i(int lane, int i) { return 8 * (lane >> 4) + i; }
static int row_i_8half(int lane, int i) { return i + 8 * (lane >> 4); }
static int col_lane16(int lane, int /*i*/) { return lane & 15; }

int main() {
  std::printf("=== gridbook HIP WMMA probe ===\n");
  int dev = 0;
  HIP_OK(hipGetDevice(&dev));
  hipDeviceProp_t p;
  HIP_OK(hipGetDeviceProperties(&p, dev));
  std::printf("[Q2] device: %s  arch=%s\n", p.name, p.gcnArchName);
  std::printf("  warpSize=%d  CUs=%d  sharedMemPerBlock=%zu B  "
              "maxThreadsPerBlock=%d\n",
              p.warpSize, p.multiProcessorCount,
              (size_t)p.sharedMemPerBlock, p.maxThreadsPerBlock);
  std::printf("  clock=%d kHz  totalGlobalMem=%.1f GiB\n", p.clockRate,
              (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

  int* dcaps = nullptr;
  int hcaps[8] = {0};
  HIP_OK(hipMalloc(&dcaps, sizeof(hcaps)));
  hipLaunchKernelGGL(caps_kernel, dim3(1), dim3(1), 0, 0, dcaps);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(hcaps, dcaps, sizeof(hcaps), hipMemcpyDeviceToHost));
  static const char* kNames[6] = {
      "wmma_f32_16x16x16_bf16_w32        ",
      "wmma_f32_16x16x16_f16_w32         ",
      "wmma_i32_16x16x16_iu8_w32         ",
      "wmma_f32_16x16x16_fp8_fp8_w32     ",
      "wmma_f32_16x16x16_fp8_fp8_w32_gfx12",
      "swmmac_f32_16x16x32_fp8_fp8_w32   "};
  std::printf("[Q1] builtin availability, evaluated in the DEVICE pass:\n");
  for (int i = 0; i < 6; ++i) {
    std::printf("  %s : %s\n", kNames[i],
                hcaps[i] > 0 ? "PRESENT" : "ABSENT");
  }
  std::printf("  device-pass arch macro: __gfx%d__   device warpSize: %d\n",
              hcaps[6], hcaps[7]);
  const bool have_bf16 = hcaps[0] > 0;
  if (!have_bf16) {
    std::printf("[Q3] bf16 WMMA absent for this target - nothing to map.\n");
    return 1;
  }

  // ---- build operands: small integers, exact in bf16 (|x| <= 256) ---------
  uint16_t ha[256], hb[256];
  double refA[16][16], refB[16][16];
  unsigned seed = 12345u;
  auto rnd = [&seed]() {
    seed = seed * 1103515245u + 12345u;
    return (int)((seed >> 16) & 0xF) - 7;  // -7..8
  };
  for (int r = 0; r < 16; ++r) {
    for (int k = 0; k < 16; ++k) {
      const float av = (float)rnd();
      const float bv = (float)rnd();
      refA[r][k] = av;
      ha[r * 16 + k] = f32_to_bf16_rn(av);
      // hb is indexed [col][k] because the kernel reads b_colmajor[r*16+j]
      // as B[k=j][col=r].
      refB[k][r] = bv;
      hb[r * 16 + k] = f32_to_bf16_rn(bv);
    }
  }
  double refD[16][16];
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      double s = 0.0;
      for (int k = 0; k < 16; ++k) s += refA[m][k] * refB[k][n];
      refD[m][n] = s;
    }

  uint16_t *da, *db;
  float* dd;
  HIP_OK(hipMalloc(&da, sizeof(ha)));
  HIP_OK(hipMalloc(&db, sizeof(hb)));
  HIP_OK(hipMalloc(&dd, 32 * 8 * sizeof(float)));
  HIP_OK(hipMemcpy(da, ha, sizeof(ha), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(db, hb, sizeof(hb), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(wmma_probe_kernel, dim3(1), dim3(32), 0, 0, da, db, dd);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  float raw[32 * 8];
  HIP_OK(hipMemcpy(raw, dd, sizeof(raw), hipMemcpyDeviceToHost));

  {
    int nfrac = 0;
    for (int t = 0; t < 32 * 8; ++t) {
      const float v = raw[t];
      if (v != (float)(long)(v < 0 ? v - 0.5f : v + 0.5f)) {
        if (nfrac < 8)
          std::printf("[Q3] NON-INTEGER device value: lane=%d reg=%d -> %.9g\n",
                      t / 8, t % 8, (double)v);
        ++nfrac;
      }
    }
    std::printf("[Q3] %d/256 device accumulator slots are non-integer "
                "(all operands are exact small integers, so 0 is expected)\n",
                nfrac);
  }
  const DMap cands[] = {
      {"D[2*i + lane/16][lane%16]", row_2i_half, col_lane16},
      {"D[8*(lane/16) + i][lane%16]", row_8half_i, col_lane16},
      {"D[i + 8*(lane/16)][lane%16]", row_i_8half, col_lane16},
  };
  int winner = -1;
  for (int c = 0; c < (int)(sizeof(cands) / sizeof(cands[0])); ++c) {
    double worst = 0.0;
    int wl = -1, wi = -1;
    for (int lane = 0; lane < 32; ++lane)
      for (int i = 0; i < 8; ++i) {
        const int m = cands[c].row(lane, i), n = cands[c].col(lane, i);
        const double d = std::fabs((double)raw[lane * 8 + i] - refD[m][n]);
        if (d > worst) { worst = d; wl = lane; wi = i; }
      }
    // Match criterion is a TOLERANCE, not equality: Q5 below establishes that
    // this WMMA unit is not bit-exact IEEE, so a layout that is correct still
    // shows ~1e-5 absolute error.  A wrong layout is off by O(100) here, so
    // the two are never confusable.
    const bool match = worst < 1e-2;
    std::printf("[Q3] candidate %-32s max|D-ref| = %.6g%s\n", cands[c].name,
                worst, match ? "   <== MATCH" : "");
    if (worst != 0.0 && wl >= 0) {
      const int m = cands[c].row(wl, wi), n = cands[c].col(wl, wi);
      std::printf("       argmax lane=%d reg=%d -> (m=%d,n=%d): dev=%.9g "
                  "ref=%.9g\n", wl, wi, m, n, (double)raw[wl * 8 + wi],
                  refD[m][n]);
    }
    if (match && winner < 0) winner = c;
  }
  if (winner < 0) {
    std::printf("[Q3] NO candidate matched. Raw dump (lane, reg) = value, "
                "then the reference matrix:\n");
    for (int lane = 0; lane < 32; ++lane) {
      std::printf("  lane %2d:", lane);
      for (int i = 0; i < 8; ++i) std::printf(" %8.1f", raw[lane * 8 + i]);
      std::printf("\n");
    }
    for (int m = 0; m < 16; ++m) {
      std::printf("  ref row %2d:", m);
      for (int n = 0; n < 16; ++n) std::printf(" %8.1f", refD[m][n]);
      std::printf("\n");
    }
    return 3;
  }
  std::printf("[Q3] VERDICT: A = lane(row) x reg(k), B = lane(col) x reg(k), "
              "%s\n", cands[winner].name);

  // ---- Q5: arithmetic fidelity of the WMMA unit vs fp64 and vs f32 FMA ----
  // Three operand regimes.  `identity` isolates the multiply, `ones` isolates
  // the 16-term reduction, `normal` is the realistic case (bf16 activations x
  // decoded e4m3 codebook values).  For each we report the WMMA error and the
  // error of the SAME product done with plain f32 FMAs on the same device.
  {
    float* dref = nullptr;
    HIP_OK(hipMalloc(&dref, 32 * 8 * sizeof(float)));
    float sref[32 * 8];
    for (int mode = 0; mode < 3; ++mode) {
      const char* nm = (mode == 0) ? "A=I,      B=int[-7,8]"
                     : (mode == 1) ? "A=1,      B=int[-7,8]"
                                   : "A,B ~ N(0,1) (realistic)";
      double rA[16][16], rB[16][16];
      for (int r = 0; r < 16; ++r)
        for (int k = 0; k < 16; ++k) {
          float av, bv;
          if (mode == 0) { av = (r == k) ? 1.0f : 0.0f; bv = (float)rnd(); }
          else if (mode == 1) { av = 1.0f; bv = (float)rnd(); }
          else {
            // cheap Box-Muller-free gaussian-ish: sum of 4 uniforms
            float u = 0.0f;
            for (int t = 0; t < 4; ++t) u += (float)rnd() / 8.0f;
            av = u * 0.5f;
            float u2 = 0.0f;
            for (int t = 0; t < 4; ++t) u2 += (float)rnd() / 8.0f;
            bv = u2 * 0.5f;
          }
          // round through bf16 so the reference sees exactly what the HW sees
          union { uint32_t u; float f; } ca, cb;
          ca.u = (uint32_t)f32_to_bf16_rn(av) << 16;
          cb.u = (uint32_t)f32_to_bf16_rn(bv) << 16;
          rA[r][k] = ca.f;
          rB[k][r] = cb.f;
          ha[r * 16 + k] = f32_to_bf16_rn(av);
          hb[r * 16 + k] = f32_to_bf16_rn(bv);
        }
      double rD[16][16];
      double scale_ref = 0.0;
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          double acc = 0.0;
          for (int k = 0; k < 16; ++k) acc += rA[m][k] * rB[k][n];
          rD[m][n] = acc;
          if (std::fabs(acc) > scale_ref) scale_ref = std::fabs(acc);
        }
      HIP_OK(hipMemcpy(da, ha, sizeof(ha), hipMemcpyHostToDevice));
      HIP_OK(hipMemcpy(db, hb, sizeof(hb), hipMemcpyHostToDevice));
      hipLaunchKernelGGL(wmma_probe_kernel, dim3(1), dim3(32), 0, 0, da, db, dd);
      HIP_OK(hipGetLastError());
      hipLaunchKernelGGL(scalar_ref_kernel, dim3(1), dim3(32), 0, 0, da, db,
                         dref);
      HIP_OK(hipGetLastError());
      HIP_OK(hipDeviceSynchronize());
      HIP_OK(hipMemcpy(raw, dd, sizeof(raw), hipMemcpyDeviceToHost));
      HIP_OK(hipMemcpy(sref, dref, sizeof(sref), hipMemcpyDeviceToHost));
      double we = 0.0, se = 0.0;
      for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 8; ++i) {
          const int m = 2 * i + (lane >> 4), n = lane & 15;
          const double d1 = std::fabs((double)raw[lane * 8 + i] - rD[m][n]);
          const double d2 = std::fabs((double)sref[lane * 8 + i] - rD[m][n]);
          if (d1 > we) we = d1;
          if (d2 > se) se = d2;
        }
      std::printf("[Q5] %-24s  WMMA max|err| = %-11.4g (rel %.3g)   "
                  "f32-FMA max|err| = %.4g\n", nm, we,
                  scale_ref > 0 ? we / scale_ref : 0.0, se);
    }
    HIP_OK(hipFree(dref));
  }

  // ---- Q4: accumulator chaining across 8 K-steps -------------------------
  const int KS = 8;
  uint16_t* hca = (uint16_t*)std::malloc(sizeof(uint16_t) * 16 * 16 * KS);
  uint16_t* hcb = (uint16_t*)std::malloc(sizeof(uint16_t) * 16 * 16 * KS);
  double* cref = (double*)std::calloc(16 * 16, sizeof(double));
  for (int s = 0; s < KS; ++s) {
    double A2[16][16], B2[16][16];
    for (int r = 0; r < 16; ++r)
      for (int k = 0; k < 16; ++k) {
        const float av = (float)rnd(), bv = (float)rnd();
        A2[r][k] = av;
        B2[k][r] = bv;
        hca[(s * 16 + r) * 16 + k] = f32_to_bf16_rn(av);
        hcb[(s * 16 + r) * 16 + k] = f32_to_bf16_rn(bv);
      }
    for (int m = 0; m < 16; ++m)
      for (int n = 0; n < 16; ++n) {
        double acc = 0.0;
        for (int k = 0; k < 16; ++k) acc += A2[m][k] * B2[k][n];
        cref[m * 16 + n] += acc;
      }
  }
  uint16_t *dca, *dcb;
  HIP_OK(hipMalloc(&dca, sizeof(uint16_t) * 16 * 16 * KS));
  HIP_OK(hipMalloc(&dcb, sizeof(uint16_t) * 16 * 16 * KS));
  HIP_OK(hipMemcpy(dca, hca, sizeof(uint16_t) * 16 * 16 * KS,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dcb, hcb, sizeof(uint16_t) * 16 * 16 * KS,
                   hipMemcpyHostToDevice));
  hipLaunchKernelGGL(wmma_chain_kernel, dim3(1), dim3(32), 0, 0, dca, dcb, KS,
                     dd);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(raw, dd, sizeof(raw), hipMemcpyDeviceToHost));
  double worst = 0.0;
  for (int lane = 0; lane < 32; ++lane)
    for (int i = 0; i < 8; ++i) {
      const int m = cands[winner].row(lane, i), n = cands[winner].col(lane, i);
      const double d = std::fabs((double)raw[lane * 8 + i] - cref[m * 16 + n]);
      if (d > worst) worst = d;
    }
  std::printf("[Q4] %d chained K-steps vs fp64 reference: max|D-ref| = %.6g\n",
              KS, worst);
  std::printf("=== probe %s ===\n", worst < 1e-2 ? "PASS" : "FAIL");
  return worst < 1e-2 ? 0 : 4;
}
