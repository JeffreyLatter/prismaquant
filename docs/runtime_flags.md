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
| `PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK` | **on** | (set automatically by `multi_chunk_probe --retain-cross-chunk-cache`, default true.) Keep LayerCache contents across chunk boundaries — layer weights are model-invariant, so an entry that fit the budget at end of chunk N is still valid for chunk N+1. Disable on small boxes. |
| `PRISMAQUANT_PROBE_CTX_CACHE` | (set by `multi_chunk_probe`) | Lets `incremental_probe.main()` reuse a cached `StreamingContext` across N invocations in the same Python process. Don't set manually unless you know what you're doing. |
| `PRISMAQUANT_PROBE_DOMAIN` | (per-chunk) | Tag each chunk's probe pickle with a domain label (used by adaptive-sampling per-domain saliency). Set automatically by `multi_chunk_probe` based on `chunk_<domain>_<idx>.jsonl` filename. |
| `PRISMAQUANT_COST_PREFETCH_ACT` | **on** | `measure_batched_gpu` prefetches chunk N+1's activation files on a thread pool while chunk N runs on the GPU. Hides ~30-40% of the cost step's wall on big models. |

Probe CLI: `kl_sensitivity_probe.py --prismaclip-candidates` adds
`NVFP4_CLIPPED` as a same-bpp candidate beside baseline NVFP4. It uses the
existing production cache, stores the clipped rendering under a distinct cache
key, and still exports through the normal NVFP4 vLLM runtime format. This is
the preferred way to use PrismaClip when MXFP8/BF16 promotions are available.

## Export flags

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_BATCHED_NVFP4_EXPORT` | **on** (when act-aware passes fire and an activation cache is supplied) | Routes NVFP4 same-shape Linears through the batched GPTQ + scale_sweep path (`export_batched_gptq.py`). Stacks per-layer experts into `(E, out, in)` tensors and runs Cholesky / column update batched across E. ~5-10% faster on MiniMax, more on bigger MoE models. |
| `PRISMAQUANT_NVFP4_SCALE_RULE` | `static_6` | NVFP4 local block-scale rule. `static_6` is standard NVFP4 max-to-6 scaling. `four_over_six_mse` tries max-to-6 and max-to-4 per 16-value block and keeps the lower block-MSE scale while preserving the same compressed-tensors NVFP4 schema and vLLM kernel. Experimental until .8B/4B/27B KL and vLLM smokes land. |
| `PRISMAQUANT_ACT_CLIP_SOLVER` | **off** | Enables **PrismaClip**, the production-cache NVFP4 activation-clipping solver. PrismaClip picks one scalar render-time activation clamp per Linear/fused-sibling group, scores candidates on original unclipped activations, and stores only selected thresholds in cache metadata. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS` | `6` | Maximum threshold evaluations per eligible group for PrismaClip's log-space scalar solver. Higher values can improve the threshold but multiply NVFP4 render cost. Cache metadata records each group's evaluation trace so convergence can be audited without rerunning the solver. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN` | `0.0` | Optional minimum relative output-MSE gain over the existing render path before PrismaClip selects a solved threshold. A nonzero floor is useful for ablations, but is not the default because tiny per-group gains can be collectively useful. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT` | **off** | Experimental stability gate requiring each PrismaClip threshold selected by the full local activation score to also improve held-out activation rows before it is stored. Kept off by default: an .8B smoke on 2026-05-11 measured KL `0.34022548` with the holdout veto versus `0.12600227` for the no-prewrite PrismaClip baseline. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT_FRACTION` | `0.25` | Fraction of activation rows reserved for PrismaClip holdout scoring, implemented as deterministic row striding so the split stays GPU-resident and reproducible. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT_MIN_GAIN` | `0.0` | Optional minimum relative holdout gain required in addition to a positive holdout improvement. Keep at `0.0` unless running an explicit stability ablation. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE` | **off** | Opt-in baseline NVFP4 prewrite while PrismaClip scores candidates. Kept off by default: .8B validation on 2026-05-11 regressed from KL `0.12600227` with prewrite off to `0.21575844` with prewrite on. A 4B no-prewrite rerun still regressed to KL `0.16388227`, so PrismaClip itself remains experimental until its threshold acceptance gate is strengthened. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_BATCHED` | **off** | Opt-in use of the existing same-shape batched NVFP4 GPTQ/scale-sweep path for PrismaClip threshold evaluations when all group members support it. Falls back to scalar rendering for AWQ-scaled groups or incompatible shapes. Kept off by default until 4B KL parity with scalar rendering is proven. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_TOP_FRACTION` | `1.0` | Restrict PrismaClip threshold search to the highest-baseline-error fraction of eligible fused groups. All groups still get baseline-scored; skipped groups keep baseline behavior. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_TOP_K` | `0` | Optional hard cap on PrismaClip threshold-solved groups after baseline-error ranking. `0` means no cap. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_VERBOSE` | **off** | Prints per-group PrismaClip decisions while filling the production cache. |
| `PRISMAQUANT_PRISMAFISHERCLIP` | **off** | Enables **PrismaFisherClip**, an experimental PrismaClip diagnostic that reuses h-detail `g2_per_token` vectors to score clip candidates with Fisher weights. It requires an h-detail directory and does not enable Fisher-weighted GPTQ unless `fisher_gptq` is also enabled. |
| `PRISMAQUANT_ACT_CLIP_SOLVER_FISHER` | **off** | Alias for `PRISMAQUANT_PRISMAFISHERCLIP`; kept for explicit activation-clip naming in ad hoc smokes. |
| `PRISMAQUANT_PRISMAFISHERCLIP_MODE` / `PRISMAFISHERCLIP_MODE` | `audit` | `audit` records Fisher-weighted clip scores but keeps normal PrismaClip decisions. `veto` additionally requires Fisher-weighted improvement, and `score` makes Fisher-weighted score the primary objective; both are ablation modes after the 2026-05-11 .8B regressions. |
| `PRISMAQUANT_PRISMAFISHERCLIP_MIN_GAIN` | `0.0` | Optional minimum relative Fisher-weighted gain required in `veto` mode. Keep at `0.0` unless running an explicit stability ablation. |
| `PRISMAQUANT_FISHER_WEIGHTED_GPTQ` | **off** | Enables Fisher/output-weighted local objectives when an h-detail cache is supplied. NVFP4 GPTQ/scale-sweep and MXFP8 scale-sweep weight activation rows by normalized `g2_per_token`. |
| `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP` | `64` | Caps normalized per-token Fisher weights before re-normalizing, preventing a single calibration token from dominating the local solve. |
| `PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS` | `0` | Candidate E8M0 exponent shifts for MXFP8 activation-weighted scale search. Nonzero shifts are experimental; .8B and 4B A/Bs on 2026-05-10 regressed KL, so production defaults to RTN-equivalent MXFP8. |

## Pipeline production-cache flags

These are `run-pipeline.sh` environment variables rather than
`PRISMAQUANT_*` flags.

| env var | default | what it does |
|---|---|---|
| `PRODUCTION_CACHE` | `1` | Build and use a `ProductionWeightCache` so export packs the same rendered weights that KL/polish measured. |
| `PRODUCTION_RECACHE` | `1` | Replay calibration with production weights installed and re-fit `activation_max_abs` before export. |
| `PRODUCTION_CACHE_LEVERS` | `gptq,scale_sweep` | Render-time quality levers for the production cache. `FISHER_WEIGHTED_GPTQ=1` appends `fisher_gptq`; `PRISMAQUANT_ACT_CLIP_SOLVER=1` enables PrismaClip through the cache code. |
| `FISHER_WEIGHTED_GPTQ` | `0` | Pipeline switch that writes h-detail during probe, passes it into production cache fill, and uses a distinct `_fisher` cache path. |
| `PRISMAFISHERCLIP` | `0` | Pipeline switch that writes h-detail during probe, enables `act_clip_solver,fisher_clip`, and tags the production cache with `_fisherclip`. This scores clip thresholds with Fisher weights but leaves Fisher-weighted GPTQ off unless `FISHER_WEIGHTED_GPTQ=1`. |
| `H_DETAIL_DIR` | `$WORK_DIR/h_detail` | h-detail directory used when `FISHER_WEIGHTED_GPTQ=1` or `PRISMAFISHERCLIP=1`. |
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
  python -m prismaquant.multi_chunk_probe ...
```

To revert ALL perf flags (legacy v20 behavior):

```bash
for f in DEFERRED_FISHER_SYNC DEFERRED_FISHER_COMPUTE ACT_CACHE_ASYNC \
         DIRECT_CUDA_LOAD COST_PREFETCH_ACT BATCHED_NVFP4_EXPORT; do
    export "PRISMAQUANT_$f=0"
done
```
