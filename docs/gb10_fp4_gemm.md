# GB10 hardware NVFP4 GEMM — `gb10` branch working doc

Status: **Research / opt-in.** Gated by `PRISMAQUANT_FP4_GEMM` (default off).
Not promoted; needs the A/B below before it can be default-on.

## Hardware reality (GB10 / sm_121)

GB10 is *consumer/workstation* Blackwell: **no `tcgen05`, no TMEM, no
DSMEM, no TMA/multicast**, 1×1×1 clusters, ~99 KiB opt-in shared memory
per block (`shared_memory_per_block_optin = 101376`). The FP4 path is the
SM120-family block-scaled warp `mma` — *not* the datacenter `tcgen05.mma`.

CUTLASS 4.5.0 is what made this usable: *"Block Scaled MMA for SM120 now
works on Spark"* + 128×32/128×64 tiles (sized to fit ~99 KiB SMEM) for up
to 30% on sm_121. Installed here: CUDA 13.0, torch 2.11+cu130,
flashinfer 0.6.11, CUTLASS 4.5.0.

## Micro-benchmark findings (`scratch/fp4_probe.py`, `scratch/fp4_parity.py`)

GEMM speedup, `flashinfer.mm_fp4(backend='cutlass')` vs bf16:

| shape M·N·K | GEMM speedup | FP4 TFLOP/s |
|---|---|---|
| 256·2048·2048 | ~noise floor (launch-bound) | ~198 |
| 256·5120·5120 | 5.9× | 145 |
| 512·13824·5120 | 4.0× | 162 |
| 4096·5120·5120 | 3.6× | 176 |

Backends: `cutlass` works; `b12x` → DSL ICE (broken `nvidia-cutlass-dsl`
4.5.0, CUTLASS #3227); `auto` → picks cuDNN, no sm_121 FP4 engine. Pin
`backend='cutlass'`.

**Parity (the catch).** PrismaQuant's NVFP4 cost model
(`format_registry._make_rtn("fp4_e2m1",16)`, `mx_scale=False`) uses
continuous **fp32** block scales. The hardware path / exported artifact
use **fp8_e4m3** block scales. Error *magnitude* vs bf16 truth is nearly
equal (13.31% vs 13.44% — PrismaQuant's is marginally optimistic), but the
two quantized outputs differ from **each other by ~10%**. So swapping the
loop GEMM to `mm_fp4` shifts every NVFP4 layer's measured output ~10% —
a numerical-method change, not a transparent speedup. `mm_fp4` is the more
faithful side (it is what vLLM serves).

## Integration

- `prismaquant/kernels/nvfp4_mm_fp4.py` — flashinfer `mm_fp4` adapter:
  `is_available()`, `quantize()`, `gemm()`, `aw_matmul()`, shape gate.
- `prismaquant/perturbed_x_cache.py` — extends the existing fused-forward
  machinery (`_nvfp4_fused_param_plan` / `_try_install_nvfp4_fused_forward`
  / `_nvfp4_fused_linear_forward`). When `PRISMAQUANT_FP4_GEMM` is set,
  NVFP4×NVFP4 Linears dispatch to `_mm_fp4_linear_forward`; below the
  flop threshold or on any failure it falls back to the Triton kernel,
  then the bf16 reference. Weight quantization is cached per layer.
- Refused when a production weight cache is active unless
  `PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE` is set (same gate as the
  Triton fused path — flashinfer re-derives scales).

## A/B before promotion (design_guidelines gate)

Run the cost + KL stages of the pipeline twice, identical except the flag:

```bash
# baseline
PRISMAQUANT_FP4_GEMM=0 ./test-pipeline.sh --repo <model> --auto-accept
# candidate (FP4 GEMM active in the measurement loop)
PRISMAQUANT_FP4_GEMM=1 PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE=1 \
  ./test-pipeline.sh --repo <model> --auto-accept
```

Compare, on the same calibration contract:

- per-layer measured KL and the allocator assignment / achieved bpp;
- final exported-artifact KL (`validate_assignments_kl`);
- measurement-loop wall-time.

**Promote to default-on only if** KL/bpp is preserved or improved (the
hardware path is the faithful one, so KL should not *worsen*) **and**
wall-time drops. Regression or inconclusive → stays opt-in / research.

## A/B result — Qwen/Qwen3.5-0.8B, 2026-05-17

VERDICT: **NOT promoted. Stays opt-in / research.** The measurement-loop
FP4 GEMM regresses measured KL.

`validate_assignments_kl` Pareto-frontier KL, baseline vs candidate
(identical frontier production cache — per-layer `output_mse` matched to
~0.2%, confirming the caches are equivalent and KL is the only moving
part):

| bpp | KL FP4_GEMM=0 | KL FP4_GEMM=1 | Δ |
|---|---|---|---|
| 4.95 | 0.2871 | 0.2846 | −0.9% |
| 5.02 | 0.2151 | 0.3332 | +54.9% |
| 5.07 | 0.1976 | 0.3336 | +68.9% |
| 5.17 | 0.2229 | 0.2903 | +30.2% |
| 5.33/5.42 | 0.1727 | 0.2254 | +30.5% |
| 6.23 | 0.1712 | 0.1742 | +1.8% |
| 7.17 | 0.0828 | 0.0924 | +11.6% |
| 8.25 | 0.0712 | 0.0688 | −3.4% |

KL regresses at 10 of 13 Pareto points, by up to +69% in the
production-relevant 5.0–5.4 bpp band. The noisier candidate curve pushed
the kneedle selection from 5.33 bpp (KL 0.173) to 7.17 bpp (KL 0.092) —
the FP4 GEMM measurement path made the pipeline export a *less*
compressed model (1.49× vs 1.65×) to clear the knee.

This confirms the predicted ~10% per-NVFP4-layer numerical shift: the
hardware fp8_e4m3 block-scale path is not a transparent substitute for
PrismaQuant's continuous-fp32 NVFP4 cost model in the measurement loop.
Wall-time was inconclusive (the baseline reused a pre-built frontier
cache; the FP4 GEMM only touches the measurement loop, not cache
construction), but the KL regression alone fails the promotion gate.

Keep `PRISMAQUANT_FP4_GEMM` default-off. It remains useful as a research
lever and for any future faithful-fp8-scale cost-model variant.
