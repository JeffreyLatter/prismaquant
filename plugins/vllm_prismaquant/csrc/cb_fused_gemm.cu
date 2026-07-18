// Fused-prefill workstream, step 1: instantiate the SAME sm120 fp8 GEMM as
// sm120_fp8_gemm.cu but through OUR FORKED collective mainloop
// (cutlass_fork/sm120_cb_mma_tma.hpp) — the fork-without-change gate. Must be
// bit-identical to the builder version and speed-equal; then the CB
// decode-in-prologue replaces the B-operand producer inside the fork.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_fork/sm120_cb_mma_tma.hpp"

namespace {

using namespace cute;

using ElementAB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignAB = 16;
constexpr int AlignD = 8;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        void, LayoutD, AlignD,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using BuilderMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementAB, LayoutA, AlignAB,
        ElementAB, LayoutB, AlignAB,
        ElementAcc,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

// Rebind the builder-resolved collective onto our forked dispatch policy —
// same template arguments, our mainloop body.
template <class T>
struct SwapToCb;
template <int S, int SP, class CS, class KS, class... Rest>
struct SwapToCb<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm120TmaWarpSpecialized<S, SP, CS, KS>, Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120CbTmaWarpSpecialized<S, SP, CS, KS>,
      Rest...>;
};

using ForkMainloop = typename SwapToCb<BuilderMainloop>::type;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, ForkMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

torch::Tensor sm120_fp8_mm_fork(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1));
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  TORCH_CHECK(K % AlignAB == 0);
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == K);
  TORCH_CHECK(b.stride(1) == 1 && b.stride(0) == K);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const ElementAB*>(a.data_ptr()), sa,
       reinterpret_cast<const ElementAB*>(b.data_ptr()), sb},
      {{1.0f, 0.0f}, nullptr, StrideD{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws},
                                a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "fork can_implement failed");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) ==
              cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm120_fp8_mm_fork", &sm120_fp8_mm_fork,
        "sm120 fp8 GEMM through the FORKED collective (step-1 gate)");
}
