// torch bindings for the HIP CB kernels.
//
// Kept in its own translation unit so that cb_gemv_hip.hip / cb_gemm_hip.hip
// stay torch-free and can be linked into the standalone self-test binary
// (cb_hip_selftest.hip), which is what establishes kernel correctness on a box
// that may not have a HIP-enabled torch.  Only this file knows about ATen.
//
// It is a `.cpp`, NOT a `.hip`, and that is load-bearing: there is no device
// code here, and compiling it with hipcc defines `__HIPCC__`, which makes
// torch's `headeronly/util/complex.h` pull in `<thrust/complex.h>`.  rocThrust
// is a separate package that Fedora 44 does not ship, so the device compile of
// this file fails on a box where the KERNELS compile perfectly.  Building it
// with the host compiler sidesteps that entirely and is the correct split
// anyway — everything below is host API.
//
// NOTE on hipify: torch's `cpp_extension.load()` runs its sources through
// hipify_python.  Everything here is written in HIP spellings already
// (`hipStream_t`, `c10::hip::*`) so there is nothing to rewrite, and hip_ext.py
// stages a copy of the sources into the build directory before compiling so
// that hipify can never write back into an installed package.
#include <torch/extension.h>

// MASQUERADING, not the plain c10::hip API.  A ROCm torch build reports its
// tensors as DeviceType::CUDA and tracks the "current stream" through the
// masquerading guard implementation; the plain HIP guard rejects a CUDA-typed
// device outright ("HIPGuardImpl initialized with non-HIP DeviceType: cuda"),
// and the plain non-masquerading current-stream accessor reads a DIFFERENT
// stream slot than the one torch's own kernels use -- which would be a silent
// ordering bug rather than a loud error.  `c10::OptionalDeviceGuard` is
// device-type agnostic and dispatches to whichever impl is registered, so it is
// correct on a CUDA build too.
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPException.h>

#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <string>

#include <stdint.h>

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
int cb_gemv_lut_bytes_fp8(int);
int cb_gemv_fp4_lut_elems(int, int);
bool cb_gemv_prefer_lds(int, int);
bool cb_gemv_lut_fits(int, int);
enum { SRC_E4M3 = 0, SRC_BF16 = 1 };
}  // namespace pq_hip

namespace {

// The codebook sidecar may be e4m3 bytes (Blackwell convention) or bf16 (the
// natural grid on a platform with no fp8 hardware, where the e4m3 constraint
// costs quality and buys nothing).  Both are accepted and produce identical
// results for an e4m3-valued table; the kernels materialise bf16 in LDS either
// way, converting at most once per element per workgroup.
int grid_src_of(const torch::Tensor& cb) {
  if (cb.scalar_type() == torch::kBFloat16) return pq_hip::SRC_BF16;
  TORCH_CHECK(cb.scalar_type() == torch::kUInt8,
              "CB codebook must be uint8 (E4M3 bytes) or bfloat16; got ",
              cb.scalar_type());
  return pq_hip::SRC_E4M3;
}

hipStream_t current_stream() {
  return at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
}

int lds_limit() {
  static int cached = -1;
  if (cached < 0) {
    hipDeviceProp_t p;
    int dev = 0;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&p, dev);
    cached = static_cast<int>(p.sharedMemPerBlock);
    TORCH_CHECK(p.warpSize == 32,
                "gridbook HIP kernels require wave32; this device reports "
                "warpSize=", p.warpSize,
                ".  The lane<->codeword mapping (32 codewords per 256-weight "
                "superblock) is exact only at 32 lanes.");
  }
  return cached;
}

// Resolve the LDS-vs-global codebook policy: measured default (see the table in
// cb_gemv_hip.hip), overridable for an A/B.
bool resolve_lds(int lut_bytes) {
  const char* e = std::getenv("PRISMAQUANT_CB_HIP_LUT");
  if (e != nullptr) {
    if (std::string(e) == "lds") {
      return pq_hip::cb_gemv_lut_fits(lut_bytes, lds_limit());
    }
    if (std::string(e) == "global") return false;
  }
  return pq_hip::cb_gemv_prefer_lds(lut_bytes, lds_limit());
}

int decode_contract_v2() {
  const char* e = std::getenv("PRISMAQUANT_CB_DECODE_CONTRACT");
  return (e != nullptr && std::string(e) == "v2") ? 1 : 0;
}

void check_common(const torch::Tensor& qw, const torch::Tensor& cboff,
                  int64_t N, int64_t K) {
  TORCH_CHECK(qw.scalar_type() == torch::kUInt8, "cb_qweight must be uint8");
  TORCH_CHECK(qw.dim() == 2 && qw.size(0) == N, "cb_qweight must be (N, bytes)");
  TORCH_CHECK(qw.stride(1) == 1, "cb_qweight rows must be contiguous");
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of the 256-weight superblock");
  TORCH_CHECK(cboff.scalar_type() == torch::kInt32 && cboff.numel() == N,
              "cb_row_offset must be int32 and cover every output row");
}

}  // namespace

torch::Tensor fp8_act_qdq(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "fp8_act_qdq wants a device bf16 tensor");
  auto x2 = x.contiguous();
  const int64_t K = x2.size(-1);
  const int64_t M = x2.numel() / K;
  auto out = torch::empty_like(x2);
  if (M == 0 || K == 0) return out;
  const c10::OptionalDeviceGuard guard(at::device_of(x2));
  pq_hip::launch_fp8_act_qdq(
      reinterpret_cast<const uint16_t*>(x2.data_ptr()),
      reinterpret_cast<uint16_t*>(out.data_ptr()), M, K, current_stream());
  C10_HIP_CHECK(hipGetLastError());
  return out;
}

torch::Tensor cb_gemv_fp8(torch::Tensor x, torch::Tensor qw_padded,
                          torch::Tensor cb_flat_fp8,
                          torch::Tensor cb_row_offset, torch::Tensor scale,
                          int64_t N, int64_t K, int64_t k_bits, int64_t n_sub,
                          int64_t type_size, bool qdq_input) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "cb_gemv_fp8 wants bf16 activations");
  const int src_kind = grid_src_of(cb_flat_fp8);
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(n_sub == 4, "the fp8 HIP GEMV supports the n_sub=4 rungs only");
  TORCH_CHECK(type_size == 4 * k_bits, "fp8 type_size must equal 4*k");
  TORCH_CHECK(k_bits >= 8 && k_bits <= 48, "k outside the FP8_CB ladder");
  check_common(qw_padded, cb_row_offset, N, K);

  auto sizes = x.sizes().vec();
  auto x2 = x.reshape({-1, K}).contiguous();
  const int64_t M = x2.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "the decode GEMV handles M in [1,16]");

  const c10::OptionalDeviceGuard guard(at::device_of(x2));
  torch::Tensor xq = qdq_input ? fp8_act_qdq(x2) : x2;
  auto y = torch::empty({M, N}, x2.options());
  const int lut_bytes = pq_hip::cb_gemv_lut_bytes_fp8(static_cast<int>(k_bits));
  pq_hip::launch_cb_gemv_fp8(
      reinterpret_cast<const uint16_t*>(xq.data_ptr()),
      qw_padded.data_ptr<uint8_t>(), cb_flat_fp8.data_ptr(),
      cb_row_offset.data_ptr<int32_t>(), scale.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(y.data_ptr()), static_cast<int>(M), N, K,
      qw_padded.stride(0), static_cast<int>(k_bits),
      static_cast<int>(type_size), 32, resolve_lds(lut_bytes), src_kind,
      decode_contract_v2(), current_stream());
  C10_HIP_CHECK(hipGetLastError());
  sizes.back() = N;
  return y.reshape(sizes);
}

torch::Tensor cb_gemv_fp4_v2(torch::Tensor x, torch::Tensor qw_padded,
                             torch::Tensor cb_flat, torch::Tensor cb_row_offset,
                             torch::Tensor compose, int64_t N, int64_t K,
                             int64_t k_bits, int64_t n_sub,
                             int64_t type_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "cb_gemv_fp4_v2 wants bf16 activations (act-QDQ'd outside)");
  TORCH_CHECK(cb_flat.scalar_type() == torch::kBFloat16,
              "cb_gemv_fp4_v2 wants the BF16 flat codebook");
  TORCH_CHECK(compose.scalar_type() == torch::kFloat32 &&
                  compose.numel() == 256 * 16,
              "compose must be the (256*16,) fp32 two-tier table");
  TORCH_CHECK(n_sub == 2 || n_sub == 1,
              "fp4-v2 GEMV: n_sub=2 (product) or n_sub=1 (signed S-rungs)");
  TORCH_CHECK(n_sub == 2 || k_bits > 8, "signed mode needs k > 8");
  TORCH_CHECK(type_size == 4 * k_bits + 9,
              "fp4-v2 type_size must be 4k+9 (E8M0 super + 8 sub-nibble bytes)");
  check_common(qw_padded, cb_row_offset, N, K);

  auto sizes = x.sizes().vec();
  auto x2 = x.reshape({-1, K}).contiguous();
  const int64_t M = x2.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "the decode GEMV handles M in [1,16]");

  const c10::OptionalDeviceGuard guard(at::device_of(x2));
  auto y = torch::empty({M, N}, x2.options());
  const int lut_elems = pq_hip::cb_gemv_fp4_lut_elems(
      static_cast<int>(k_bits), static_cast<int>(n_sub));
  pq_hip::launch_cb_gemv_fp4_v2(
      reinterpret_cast<const uint16_t*>(x2.data_ptr()),
      qw_padded.data_ptr<uint8_t>(),
      reinterpret_cast<const uint16_t*>(cb_flat.data_ptr()),
      cb_row_offset.data_ptr<int32_t>(), compose.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(y.data_ptr()), static_cast<int>(M), N, K,
      qw_padded.stride(0), static_cast<int>(k_bits), static_cast<int>(n_sub),
      static_cast<int>(type_size), 32, resolve_lds(lut_elems * 2),
      decode_contract_v2(), current_stream());
  C10_HIP_CHECK(hipGetLastError());
  sizes.back() = N;
  return y.reshape(sizes);
}

torch::Tensor cb_gemm_fp8(torch::Tensor x, torch::Tensor qw_padded,
                          torch::Tensor cb_flat_fp8,
                          torch::Tensor cb_row_offset, torch::Tensor scale,
                          int64_t N, int64_t K, int64_t k_bits, int64_t n_sub,
                          int64_t type_size, bool qdq_input) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "cb_gemm_fp8 wants bf16 activations");
  const int src_kind = grid_src_of(cb_flat_fp8);
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(n_sub == 4, "the fp8 HIP GEMM supports the n_sub=4 rungs only");
  TORCH_CHECK(type_size == 4 * k_bits, "fp8 type_size must equal 4*k");
  check_common(qw_padded, cb_row_offset, N, K);

  auto sizes = x.sizes().vec();
  auto x2 = x.reshape({-1, K}).contiguous();
  const int64_t M = x2.size(0);
  const c10::OptionalDeviceGuard guard(at::device_of(x2));
  torch::Tensor xq = qdq_input ? fp8_act_qdq(x2) : x2;
  auto y = torch::empty({M, N}, x2.options());
  const int lut_bytes = pq_hip::cb_gemv_lut_bytes_fp8(static_cast<int>(k_bits));
  pq_hip::launch_cb_gemm_fp8(
      reinterpret_cast<const uint16_t*>(xq.data_ptr()),
      qw_padded.data_ptr<uint8_t>(), cb_flat_fp8.data_ptr(),
      cb_row_offset.data_ptr<int32_t>(), scale.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(y.data_ptr()), M, N, K, qw_padded.stride(0),
      static_cast<int>(k_bits), static_cast<int>(type_size),
      resolve_lds(lut_bytes), 0, src_kind, current_stream());
  C10_HIP_CHECK(hipGetLastError());
  sizes.back() = N;
  return y.reshape(sizes);
}

torch::Tensor cb_expand_fp8(torch::Tensor qw_padded, torch::Tensor cb_flat_fp8,
                            torch::Tensor cb_row_offset, int64_t N, int64_t K,
                            int64_t k_bits, int64_t n_sub, int64_t type_size) {
  TORCH_CHECK(cb_flat_fp8.scalar_type() == torch::kUInt8);
  TORCH_CHECK(n_sub == 4 && type_size == 4 * k_bits);
  check_common(qw_padded, cb_row_offset, N, K);
  const c10::OptionalDeviceGuard guard(at::device_of(qw_padded));
  auto w = torch::empty({N, K}, qw_padded.options());
  pq_hip::launch_cb_expand_fp8(
      qw_padded.data_ptr<uint8_t>(), cb_flat_fp8.data_ptr<uint8_t>(),
      cb_row_offset.data_ptr<int32_t>(), w.data_ptr<uint8_t>(), N, K,
      qw_padded.stride(0), static_cast<int>(k_bits),
      static_cast<int>(type_size), current_stream());
  C10_HIP_CHECK(hipGetLastError());
  return w.view(torch::kFloat8_e4m3fn);
}

// LDS bytes once materialised as bf16 (2x the on-disk e4m3 size).
int64_t lut_bytes_fp8(int64_t k_bits) {
  return pq_hip::cb_gemv_lut_bytes_fp8(static_cast<int>(k_bits));
}
bool lut_is_lds(int64_t k_bits) {
  return resolve_lds(pq_hip::cb_gemv_lut_bytes_fp8(static_cast<int>(k_bits)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_act_qdq", &fp8_act_qdq,
        "fused per-token fp8 dynamic QDQ (mirrors codec.fp8_dynamic_act_qdq)");
  m.def("cb_gemv_fp8", &cb_gemv_fp8,
        "FP8_CB decode GEMV (wave32, LDS-resident codebook LUT)");
  m.def("cb_gemv_fp4_v2", &cb_gemv_fp4_v2,
        "NVFP4_CB two-tier (v2) decode GEMV with in-register scale compose");
  m.def("cb_gemm_fp8", &cb_gemm_fp8,
        "FP8_CB decode-in-prologue prefill GEMM (RDNA3.5 bf16 WMMA)");
  m.def("cb_expand_fp8", &cb_expand_fp8,
        "FP8-direct transient expand (bounded per-layer tile)");
  m.def("lut_bytes_fp8", &lut_bytes_fp8, "LDS bytes a rung's codebook needs");
  m.def("lut_is_lds", &lut_is_lds,
        "whether this rung will stage its codebook in LDS under the current "
        "policy/env");
}
