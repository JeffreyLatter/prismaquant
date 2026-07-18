# CUTLASS serving-kernel — grounding map (2026-07-18)

Concrete starting map for the CB serving kernels, after the 27B served verdict
proved quality (−58% KL at matched bpp) but exposed the speed gap (2.2× prefill,
2.4× decode vs native AURA). This is the working brief for the multi-session
kernel build. Read alongside `serving-kernel.md` (the original plan).

## Why fusion is mandatory (evidence, not assertion)
The transient-expand prefill (`expand_cb_to_value` → `cutlass_scaled_mm`)
**materializes the full [N,K] weight tile to HBM, then the GEMM reads it back**.
That is inherently ≥2× the memory traffic of a resident-weight GEMM — and worse
today because `expand_cb_to_value` writes a **bf16** tile (2 B/elt = 32 GB for
the 27B body) then casts to fp8 (another 16 GB), vs AURA's single 16 GB fp8
read. Measured prefill is 2.2× slower — consistent with the traffic doubling.
No amount of tuning the expand removes this; only **decode-in-prologue fusion
(never materialize the tile)** does. This retires the "transient path may be
enough" hope from the old plan — it is not, at 27B.

Cheap partial win available NOW without CUTLASS (do first, it de-risks + helps):
- Make `expand_cb_to_value` write **fp8 directly** (codebook values are on the
  e4m3 grid), skipping the 32 GB bf16 intermediate → halves expand-side traffic.
- Still ≥2× vs resident; a stopgap, not the fix.

## CUDA-graph is NOT the decode fix (falsified 2026-07-18)
Serving OURS without `--enforce-eager` made decode WORSE (4.20 → 1.21 tok/s):
vLLM pads captured decode batches above `PREFILL_M_THRESHOLD=16`, so every
graphed step takes the expand path per token. Keep `--enforce-eager`. The
decode bottleneck is the `cb_gemm` kernel's own throughput (below even BF16),
not launch overhead. The M-gated dispatch (host branch on M) is cudagraph-
hostile — a production kernel must not branch on a padded batch size.

## Environment (vllm-node:latest, confirmed)
- GB10 **sm_121** (compute_cap 12.1); it is the **sm_120 family** (NOT sm_100a /
  tcgen05 — that's datacenter Blackwell). Target the sm_120 block-scaled `mma`.
- nvcc **13.0**, torch cuda 13.0 → can build a CUDA extension (GGUF-plugin
  `setup.py` CUDAExtension model).
- **CUTLASS 4.5.2** C++ headers vendored in vLLM:
  `.../vllm/third_party/fmha_sm100/cutlass/include` (also under `deep_gemm/`).
- Native reference GEMM AURA uses: `vllm._custom_ops.cutlass_scaled_fp4_mm`
  (+ `scaled_fp4_quant`, `cutlass_scaled_mm_supports_fp4`,
  `flashinfer_quant_nvfp4_8x4_sf_layout` for the SF swizzle,
  `cutlass_fp4_moe_mm` for MoE).

## Fork targets (the files to base the kernel on)
- **Collective mainloop:** `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`
  — its global→shared **A producer** is where we inject: load k-bit CB indices →
  codebook lookup → 8 FP4 codes into the smem staging tile in the exact nibble
  layout the block-scaled MMA consumes. The group-16 E4M3 scale plane is
  **unchanged** (that is the whole point of matching NVFP4's scale envelope) —
  reuse it verbatim.
- **Array/grouped variant (MoE, prototype iv):**
  `sm120_blockscaled_mma_array_tma.hpp`.
- **Kernel driver:** `sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`.
- **Types/layout:** `float_subbyte.h` (FP4/E2M1), `detail/sm100_blockscaled_layout.hpp`.

## Build sequence (revised, evidence-driven)
1. **Baseline parity:** compile a plain sm_120 block-scaled NVFP4 GEMM from the
   CUTLASS collective as a CUDA extension; match `cutlass_scaled_fp4_mm`
   numerically + on speed. Proves the toolchain + our layout understanding
   before touching the mainloop. (This is the de-risking gate.)
2. **Small-k LUT fused prefill (k≤13):** fork the sm120 A-producer to do
   flat-LUT codebook lookup in smem (GB10: 99 KB opt-in smem; k=13 LUT = 32 KB).
   FP8_CB first (the 27B rung set), then NVFP4_CB.
3. **Decode GEMV (parallel, tractable):** a fixed-shape bandwidth-bound dequant-
   GEMV that beats AURA on HBM traffic (fewer bytes). Must not branch on padded
   batch; obey INV-1 (no full-weight materialize).
4. **Structured-codebook variant (k≥14)** + **MoE grouped** (iv) later.

## Open design questions to resolve at step 1
- Does `cutlass_scaled_fp4_mm` expect the flashinfer 8x4 SF swizzle, and does our
  export write the group-16 E4M3 scale plane in that swizzle or a transposable
  one? (Determines whether the fused kernel reuses the scale plane verbatim.)
- Exact FP4 nibble/interleave layout the sm120 MMA smem tile wants (match
  `_pack_fp4_indices`, `nvfp4_fused.py:32`).
