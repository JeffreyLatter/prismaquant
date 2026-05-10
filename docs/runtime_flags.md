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

## Export flags

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_BATCHED_NVFP4_EXPORT` | **on** (when act-aware passes fire and an activation cache is supplied) | Routes NVFP4 same-shape Linears through the batched GPTQ + scale_sweep path (`export_batched_gptq.py`). Stacks per-layer experts into `(E, out, in)` tensors and runs Cholesky / column update batched across E. ~5-10% faster on MiniMax, more on bigger MoE models. |

## Pipeline production-cache flags

These are `run-pipeline.sh` environment variables rather than
`PRISMAQUANT_*` flags.

| env var | default | what it does |
|---|---|---|
| `PRODUCTION_CACHE` | `1` | Build and use a `ProductionWeightCache` so export packs the same rendered weights that KL/polish measured. |
| `PRODUCTION_RECACHE` | `1` | Replay calibration with production weights installed and re-fit `activation_max_abs` before export. |
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
