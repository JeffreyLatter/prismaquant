# prismaquant runtime flags

All performance-critical paths can be tuned at runtime via env vars.
Most proven probe/cost/export flags default ON and exist mostly for opt-out /
debugging. CUDA graph flags are different: L3 and coord-descent graph capture
defaults to `auto` because one-shot candidate batches do not amortize capture
cost. Set the env var to `"1"` to force a graph path for benchmarking, or
`"0"` (also `"false"`, `"no"`, etc.) to disable it.

## Probe + cost flags

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_DEFERRED_FISHER_SYNC` | **on** | `_run_body_streaming_shard` accumulates h_trace / h_w2_sum on the device as 0-D tensors and batches the host transfer to one .cpu().tolist() per layer. Without it, every Linear's backward hook does two .item() syncs (~94k stalls per phase-3 sweep). Math identical, only timing differs. |
| `PRISMAQUANT_DEFERRED_FISHER_COMPUTE` | **on** | Defers the per-Linear Fisher matmul itself out of the autograd engine's per-Linear callback path. The bwd hook just queues `(name, x, gy, mod_ref)`; after `out.backward()` returns, a tight Python loop drains the queue. SM utilization rises from ~13% to ~50-80% on MoE-heavy phase-3. Math identical. |
| `PRISMAQUANT_ACT_CACHE_ASYNC` | **on** | Activation-cache writes (per-Linear `.pt` files) submit to a small thread pool instead of blocking the main thread. Drains at end of shard so the cost step sees a fully-flushed cache. |
| `PRISMAQUANT_ACT_CACHE_WORKERS` | `4` | Pool size for the above. Higher = more parallel disk writes, but contends with the CPU readers in cost step. |
| `PRISMAQUANT_DIRECT_CUDA_LOAD` | **on** | Pass `device=cuda:N` to `safetensors.safe_open` so layer tensors land on the GPU directly instead of going through a host stage. ~10-30 ms saved per layer load. Falls back transparently if safetensors complains. |
| `PRISMAQUANT_COST_PREFETCH_ACT` | **on** | `measure_batched_gpu` prefetches chunk N+1's activation files on a thread pool while chunk N runs on the GPU. Hides ~30-40% of the cost step's wall on big models. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR` | **archived** | Historical Fisher row-weighted allocator objective. The production pipeline rejects it; archive context lives under `archive/fisher_2026-05-15/`. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP` | archived companion | Historical cap for Fisher output-MSE allocation; not used by the production pipeline. |

## Export flags

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_BATCHED_NVFP4_EXPORT` | **on** (when act-aware passes fire and an activation cache is supplied) | Routes NVFP4 same-shape Linears through the batched GPTQ / optional scale_sweep path (`export_batched_gptq.py`). Stacks per-layer experts into `(E, out, in)` tensors and runs Cholesky / column update batched across E. |
| `PRISMAQUANT_NVFP4_SCALE_RULE` | `static_6` | NVFP4 local block-scale rule. `static_6` is standard NVFP4 max-to-6 scaling. `four_over_six_mse` tries max-to-6 and max-to-4 per 16-value block and keeps the lower block-MSE scale while preserving the same compressed-tensors NVFP4 schema and vLLM kernel. Experimental until .8B/4B/27B KL and vLLM smokes land. |
| `PRISMAQUANT_RENDER_PROGRESSIVE_GATES` | **on** | Production-cache render gate for local mechanisms. NVFP4 renders score FourOverSix, GPTQ, joint-scale optimization, and optional scale_sweep candidate packages with the shared scorer and keep the prior render when a candidate regresses. MXFP8/FP8 scale_sweep is gated the same way when enabled. Cache metadata records decisions in `render_gates`; FourOverSix has a compact `four_over_six` summary. |
| `PRISMAQUANT_RENDER_GATE_MIN_GAIN` | `0.0` | Optional minimum relative gain required by the progressive render gate. Keep at `0.0` for normal runs so tiny local improvements can accumulate; raise only for ablations. |
| `PRISMAQUANT_FISHER_WEIGHTED_GPTQ` | **archived** | Fisher-weighted GPTQ is archived under `archive/fisher_2026-05-15/` and rejected by the production pipeline. |
| `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP` | archived companion | Historical Fisher-weighted GPTQ row-weight cap; not used by the production pipeline. |
| `PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS` | `0` | Candidate E8M0 exponent shifts for MXFP8 activation-weighted scale search. Nonzero shifts are experimental; .8B and 4B A/Bs on 2026-05-10 regressed KL, so production defaults to RTN-equivalent MXFP8. |

## GB10 hardware NVFP4 GEMM (opt-in, experimental)

Routes NVFP4 measurement-loop GEMMs onto the GB10 / sm_121 hardware FP4
tensor cores via `flashinfer.mm_fp4` (CUTLASS 4.5.0 block-scaled,
`backend='cutlass'`). Lives on the `gb10` branch.

EXPERIMENTAL — not a transparent speedup. PrismaQuant's NVFP4 cost model
uses continuous fp32 block scales; the hardware path uses fp8_e4m3 block
scales (the faithful, vLLM-served form). A micro-benchmark put the
per-NVFP4-layer output shift at ~10%. It is therefore opt-in and must clear
an apples-to-apples KL/bpp A/B (see `docs/design_guidelines.md`) before it
can be promoted to default-on.

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_FP4_GEMM` | `0` (off) | When set, NVFP4-weight × NVFP4-activation Linears in the perturbed-x reference forward run through `flashinfer.mm_fp4` instead of dequant-to-bf16 + `F.linear`. Measured 3.3–5.9× faster on the GEMM for medium/large Linears. Below `PRISMAQUANT_FP4_GEMM_MIN_MNK` it transparently falls back to the bf16 path. Like the Triton fused kernel, it is refused when a production weight cache is active unless `PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE` is also set. |
| `PRISMAQUANT_FP4_GEMM_MIN_MNK` | `2000000000` | Minimum GEMM `M*N*K` below which the FP4 path is skipped — sub-~2e9-flop GEMMs are launch-bound and the micro-benchmark showed no reliable win there. |

## Pipeline production-cache flags

These are `run-pipeline.sh` environment variables rather than
`PRISMAQUANT_*` flags.

Research levers outside the current production recipe live under `archive/`.
The production pipeline fails fast when archived Fisher levers are requested.

| env var | default | what it does |
|---|---|---|
| `PRODUCTION_CACHE` | `1` | Build and use a `ProductionWeightCache` so export packs the same rendered weights that KL/polish measured. |
| `PRODUCTION_RECACHE` | `1` | Replay calibration with production weights installed and re-fit `activation_max_abs` before export. |
| `PRODUCTION_CACHE_LEVERS` | `gptq,joint_scale_opt` | V1 production render levers: GPTQ with damp sweep plus joint NVFP4 scale optimization. `scale_sweep` remains available for explicit ablations but is no longer the default. |
| `FISHER_WEIGHTED_GPTQ` | archived | Any truthy value is rejected; archive context lives under `archive/fisher_2026-05-15/`. |
| `FISHER_OUTPUT_MSE_ALLOCATOR` | archived | Any truthy value is rejected; V1 allocation uses the non-Fisher objective plus measured frontier validation. |
| `H_DETAIL_DIR` | `$WORK_DIR/h_detail` | Historical h-detail location for archived Fisher levers; not required for V1 production defaults. |
| `SELECTION_MODE` | `surrogate` | `surrogate` preserves the normal allocator-selected `TARGET_BITS` assignment. `validated-surrogate` writes allocator Pareto assignments, builds a format-menu production cache, measures real assignment KL for each Pareto point, selects the measured KL/bpp kneedle with `prismaquant.select_validated_frontier`, then recaches and exports the selected assignment. |
| `VALIDATED_FRONTIER_NSAMPLES` / `VALIDATED_FRONTIER_SEQLEN` | `$NSAMPLES` / `$SEQLEN` | Calibration size for measured-frontier KL selection. Keep these at the artifact validation contract for 27B decisions; lower values are smoke-only. |
| `VALIDATED_FRONTIER_PICK` | `kneedle` | Selection rule for `SELECTION_MODE=validated-surrogate`: `kneedle`, `best-kl`, or `lowest-bpp`. Production selection should use `kneedle` unless the run is explicitly an ablation. |
| `PRODUCTION_CACHE_LRU_GB` | `64.0` | Resident tensor budget for disk-backed production-cache use in recache and export. The 27B n=8 recache smoke needed `45.32 GiB` for the selected assignment. |
| `PRODUCTION_CACHE_PREFETCH` | `require` | Standalone recache prefetch policy. `require` fails fast when assignment-required weights cannot fit resident, preventing silent NVMe-bound replay. |
| `PRODUCTION_CACHE_PREFETCH_WORKERS` | `4` | Thread count for eager production-cache prefetch. |

## CUDA / system flags

| env var | recommended | what it does |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Required on UMA hardware (DGX Spark) to keep the CUDA caching allocator from hoarding freed blocks. Set automatically by `incremental_probe.py` at module load. |
| `PRISMAQUANT_L3_CUDA_GRAPHS` | `auto` | Graphs decoder-tail L3 propagation only when the same graph key has enough repeated calibration calls. Default threshold: `8`, override with `PRISMAQUANT_L3_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_COORD_LANE_CUDA_GRAPHS` | `auto` | Graphs lane-batched coord flip evaluation only when repeated calls can amortize capture. Replay-cache coord batches are one-shot, so auto leaves them eager. Default thresholds: `8` for replay mode, `16` for full-forward mode; override with `PRISMAQUANT_COORD_LANE_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_KL_CUDA_GRAPHS` | `auto` | Graphs assignment-KL validation only for larger calibration batches. Default threshold: `16`, override with `PRISMAQUANT_KL_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_COORD_REPLAY_CACHE` | `off` | Opt-in LayerHiddenStateCache for coord descent. It reduces tail layer forwards but currently copies too much baseline model state on large Qwen runs, so the fast default is lane-batched eager evaluation. |
| `PRISMAQUANT_L3_MIN_HOST_MEM_GB` | unset | Optional host-memory floor for L3 pair/scout diagnostics. When set, L3 raises `GPUMemoryBudgetExceeded` between paired-override chunks if `/proc/meminfo` `MemAvailable` drops below this many GiB, giving long runs a chance to stop before system OOM. |

## Disabling for debugging

To revert a single flag for A/B comparison:

```bash
PRISMAQUANT_DEFERRED_FISHER_COMPUTE=0 \
  python -m prismaquant.incremental_probe ...
```

To revert ALL perf flags (legacy v20 behavior):

```bash
for f in DEFERRED_FISHER_SYNC DEFERRED_FISHER_COMPUTE ACT_CACHE_ASYNC \
         DIRECT_CUDA_LOAD COST_PREFETCH_ACT BATCHED_NVFP4_EXPORT; do
    export "PRISMAQUANT_$f=0"
done
```
