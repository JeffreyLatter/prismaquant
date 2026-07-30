# prismaquant runtime flags

*Reconciled against code 2026-07-30 (branch `claude/docs-consolidation`).*
Method: AST + literal sweep of `os.environ` / `os.getenv` / `_env_flag` /
`_env_int` / `_env_flag_enabled` / registry `*_env=` parameters / `pq_env_*`
across `prismaquant/`, `plugins/gridbook/`, `tools/`, `scripts/`, `pipeline.py`
(excluding `archive/`, `fp8/`, `scratch/`, `tests/`, worktrees). Every row cites
its reading `file:line`; when a flag has several readers the row cites the one
that decides behaviour and notes the others.

All performance-critical paths can be tuned at runtime via env vars.
Most proven probe/cost/export flags default ON and exist mostly for opt-out /
debugging. CUDA graph flags are different: L3 and coord-descent graph capture
defaults to `auto` because one-shot candidate batches do not amortize capture
cost. Set the env var to `"1"` to force a graph path for benchmarking, or
`"0"` (also `"false"`, `"no"`, etc.) to disable it.

Three consumer families share the `PRISMAQUANT_*` namespace and are separated
below: the **build pipeline** (probe/cost/render/export, §1–§4), the **CB build
lane** (§5), and the **gridbook serving plugin** (§7), which runs inside vLLM
and never sees a build flag.

## 1. Probe + cost flags

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_DEFERRED_FISHER_SYNC` | **on** | `incremental_probe.py:2044` | `_run_body_streaming_shard` accumulates h_trace / h_w2_sum on the device as 0-D tensors and batches the host transfer to one `.cpu().tolist()` per layer. Without it, every Linear's backward hook does two `.item()` syncs (~94k stalls per phase-3 sweep). Math identical, only timing differs. |
| `PRISMAQUANT_DEFERRED_FISHER_COMPUTE` | **on** | `incremental_probe.py:2073` | Defers the per-Linear Fisher matmul itself out of the autograd engine's per-Linear callback path. The bwd hook queues `(name, x, gy, mod_ref)`; after `out.backward()` returns, a tight Python loop drains the queue. SM utilization rises from ~13% to ~50-80% on MoE-heavy phase-3. Math identical. |
| `PRISMAQUANT_ACT_CACHE_ASYNC` | **on** | `incremental_probe.py:1738` | Activation-cache writes (per-Linear `.pt` files) submit to a small thread pool instead of blocking the main thread. Drains at end of shard so the cost step sees a fully-flushed cache. |
| `PRISMAQUANT_ACT_CACHE_WORKERS` | `4` | `incremental_probe.py:1744` | Pool size for the above. Higher = more parallel disk writes, but contends with the CPU readers in cost step. |
| `PRISMAQUANT_ACT_CACHE_FP32` | `1` | `incremental_probe.py:1770` / `:2561` | Store cached activations in fp32 rather than the model dtype. |
| `PRISMAQUANT_DIRECT_CUDA_LOAD` | **on** | `layer_streaming.py:103` | Pass `device=cuda:N` to `safetensors.safe_open` so layer tensors land on the GPU directly instead of going through a host stage. ~10-30 ms saved per layer load. Falls back transparently if safetensors complains. |
| `PRISMAQUANT_COST_PREFETCH_ACT` | **on** | `measure_quant_cost.py:1518` | `measure_batched_gpu` prefetches chunk N+1's activation files on a thread pool while chunk N runs on the GPU. Hides ~30-40% of the cost step's wall on big models. |
| `PRISMAQUANT_PROBE_DOMAIN` | unset | `incremental_probe.py:970` | Calibration-domain tag stamped into probe provenance. |
| `PRISMAQUANT_PROBE_CTX_CACHE` | unset | `incremental_probe.py:3126` | Reuse the cross-chunk probe context cache. |
| `PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK` | unset | `incremental_probe.py:3143` | Retain cross-chunk probe state instead of dropping it between chunks. |
| `PRISMAQUANT_ALLOW_KV_SHARED_FISHER` | `0` | `incremental_probe.py:1022` | Probe guard override for KV-sharing architectures (MINOR-M33). Only reachable with `PRISMAQUANT_KV_COTANGENT=0`: severing the shared-consumer cotangent *under*-counts the storing layer's `k_proj`/`v_proj` `h_trace`, so the probe fails fast; set `1` to probe anyway, accepting the under-count. (Earlier revisions called this an aliased-Fisher *double*-count — wrong direction; the missing edge only ever removes gradient.) |
| `PRISMAQUANT_KV_COTANGENT` | on | `sensitivity_probe.py:1254` | The KV-cotangent path. On cross-layer KV-sharing architectures (Gemma4 `num_kv_shared_layers>0`) the phase-3 reverse sweep grafts grad-enabled leaves over borrowed K/V, sums each consumer's `leaf.grad` per source, and drives the storing layer's backward with that sum alongside its own output cotangent. Without it a sharing layer's backward stops at the detached capture and the storing layer's `h_trace` is under-counted. `0` restores the pre-fix severed cotangent for an A/B and re-arms `PRISMAQUANT_ALLOW_KV_SHARED_FISHER`. Verified against an end-to-end backward in `tests/test_kv_cotangent_path.py`. |
| `PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER` | `0` (off) | `incremental_probe.py` | Restores the v22 "Fix E1" phase-1 activation transfer: hold all L+1 layer activations device-resident, then stack for a single device→host copy. Default (off) streams each layer's activation to host inside the forward loop, bounding device residency to one activation — a doubling DSv4's multi-stream hidden can't afford. Exists to A/B the unmeasured transfer-time cost; both paths report true copy time as `host transfer`. Batched mode requires uniform activation shapes. |
| `PRISMAQUANT_SOLVER_TRACE` | `0` (off) | `allocator_solver.py` | Per-evaluation trace for `solve_with_promotion`: each tightened-target DP eval with achieved bits, predicted Δloss, wall time, plus DP-infeasible probes. Read once at module import — set before launching the allocator. Observability only. |
| `PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER` | `0` | `sensitivity_probe.py:673` (const `:451`), `measure_quant_cost.py:1824` | Probe guard override for packed-MoE experts whose compute is NOT a per-expert `F.linear(x, packed[e])` (e.g. bmm/grouped-mm): the per-token Fisher interception cannot capture them, and by default the probe fails fast rather than fall back to squaring the token-summed weight gradient (the sum-then-square estimator, audit M3: 5-50× cross-token-covariance inflation). Set `1` to accept the biased legacy estimator. Also accepted by `prepare_cost_context` to reuse a PRE-FIX probe.pkl whose packed-expert entries lack the `packed_fisher_estimator=per_token_v2` meta stamp (stale pickles are otherwise refused). |
| `PRISMAQUANT_COST_UCB_Z` | `0` — **RESEARCH-ONLY** | `allocator_candidates.py:450-453` (`_cost_ucb_z`) | Risk-aware allocation: charge `z·predicted_dloss_stderr` on top of each AURA cost row (upper-confidence-bound). `0` = bit-identical legacy behavior, and it only bites on the `predicted_dloss` branch (AURA / expert-empirical), never `output_mse`/`weight_mse`. **Research-only, and no driver sets it** — `run-pipeline.sh` never exports it; the only setters in the tree are tests. Its one measured win is confined to the **thin-calibration regime**: on the 27B old-vs-new AURA A/B at thin calib, `z=2` won −8.0%. At *production* calibration the stderr collapses and the hedge buys nothing — 6/252 rows of assignment churn, served parity — so the production-calib decision is **keep at `0`**. Turn it on only when deliberately allocating off a thin/noisy cost run, and re-measure on the serving metric before shipping anything it picked. |
| `--kl-ucb-z` (CLI, not env) | `0` — **RESEARCH-ONLY** | `validate_assignments_kl.py:859` → `:1359`, stamped at `:680` | Selection-side twin of the above: reports `kl_ucb = mean + z·stderr` over `--calib-repeats` alongside `kl_mean/std/stderr`, for `select_validated_frontier --metric ucb` to select on. Same status — **no driver passes it**, `run-pipeline.sh`'s validated-surrogate arm selects on the mean. Same regime caveat: a UCB frontier point is only meaningfully different from the mean point when the repeat stderr is large, i.e. thin calib. |
| `PRISMAQUANT_FISHER_CAP_MULTIPLIER` | unset (off) — **RESEARCH lever** | `allocator.py:1187-1280` (`clip_probe_fisher_outliers`), called from `main` at `:1579`, right after `renormalize_probe_fisher` | Robust Fisher clip: cap each row's finalized `h_trace` at `K × median(h_trace)` over its **role** bucket, rescaling `h_w2_sum` by the same ratio so the derived cost stays consistent (raw accumulators untouched). Unset or empty = byte-identical no-op; a non-finite or `≤ 0` value is a hard error, not a silent skip. Motivation: `predicted_dloss = ½·h_trace·MSE` is *linear* in `h_trace`, so a few heavy-tailed rows can capture the whole DP budget. Role buckets are deliberately the reference tool's grouping (`/home/rob/dq-runs/robust_fisher_clip.py`) — the regex `layers\.<N>\.<one container>\.<role>$`, i.e. dense attention/MLP leaves only; packed/unpacked MoE experts, shared experts and sidecars are skipped, since that is the grouping the result was measured under. **Status: research.** `K=3` measured ~5% better WikiText PPL at 6.0 bpp on Qwen3-4B (2026-05-19); never carried to a served A/B, so promote nothing on it without one. Tests: `tests/test_fisher_normalization.py`. |
| `PRISMAQUANT_FISHER_COL_WEIGHTS` | `0` | `aura_cost.py:853` | Opt-in: `aura_cost` also emits a per-Linear per-column KL-Fisher energy vector (`stats[name]['fisher_col']`, length `in_features`, sums to `h_trace`) alongside the scalar cost. Strictly additive — the rest of the cost payload is bit-identical when off. Feeds `fisher_col_weights.py`. Equivalent to `aura_cost --collect-col-energy`. |
| `PRISMAQUANT_EXPERT_COST_SAMPLE` | falls back to `PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE`, then `0` | `measure_quant_cost.py:988`, fallback `measure_quant_cost.py:989` | Stratified expert subsample per packed-expert unit in the cost stage; `0` = full stacks. The fallback chain exists so the GGUF lane's older name keeps working. |
| `PRISMAQUANT_SKIP_PACKED_EXPERT_COST` | `0` | `measure_quant_cost.py:876` | `1` skips the local packed-expert cost measurement entirely — the single most expensive part of the local cost stage. **The pipeline sets it itself** (`run-pipeline.sh:333`) whenever `EXPORT_CONTAINER=nvfp4_cb` and `CB_EXPERT_EMPIRICAL=1`, because stage `[2d-CB]` replaces every packed-expert row wholesale. Do not set by hand unless that replacement is guaranteed to run. |
| `PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE` | `0` | `measure_quant_cost.py:989` | GGUF-lane name for the above subsample; consulted only as the fallback. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR` | **archived** | `allocator_candidates.py:253` | Historical Fisher row-weighted allocator objective. The production pipeline rejects it; archive context lives under `archive/fisher_2026-05-15/`. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP` | falls back to `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP`, then `64` | `measure_quant_cost.py:248`, fallback `measure_quant_cost.py:249` | Historical cap for Fisher output-MSE allocation; not used by the production pipeline. |

## 2. Export flags

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_BATCHED_NVFP4_EXPORT` | **on** (when act-aware passes fire and an activation cache is supplied) | `export_native_compressed.py:5605`; fingerprint `:5472` | Routes NVFP4 same-shape Linears through the batched GPTQ / optional scale_sweep path (`export_batched_gptq.py`). Stacks per-layer experts into `(E, out, in)` tensors and runs Cholesky / column update batched across E. |
| `PRISMAQUANT_BLOCK_OUTPUT_MATCH` | **ARCHIVED 2026-07-30** — setting it truthy is a hard `SystemExit` | gate `export_native_compressed.py::_refuse_archived_block_output_match`; fingerprint records the constant `"archived_2026-07-30"` | Quality lever #12 (`block_output_match.py`), walled under `archive/block_output_match_2026-07-30/` by re-vet **R25** (closes D16 as *unreachable*, not unmeasured). It ran a greedy `{0.95, 1.0, 1.05}` per-Linear gain search against an FP32 reference **block** output. Three reasons: (1) **it never executed** — the production-cache pack `continue`s first, so with `PRODUCTION_CACHE=1` no dense NVFP4 Linear reached the branch (0 hits in two real production export logs); (2) had it run it would have re-derived NVFP4 group scales outside `_export_match_render_scale_rule`, discarding the render's `joint_mse` scales — the −6.6% KL defect **M19** fixed everywhere else; (3) a per-tensor gain re-search *after* JSO already solved the scale, wrapped in `except Exception → WARN` so failures were invisible. Its "~0.05–0.10 PPL" docstring was a pre-JSO expectation, never a measurement. **`0` and unset both pass** (they asked for what now always happens); any other value refuses so an old launcher fails loudly rather than exporting differently in silence. |
| `PRISMAQUANT_NVFP4_SCALE_RULE` | `static_6` | `export_native_compressed.py:190` (const `:107`); `incremental_measure_quant_cost.py:287` | NVFP4 local block-scale rule. `static_6` is standard NVFP4 max-to-6 scaling. `four_over_six_mse` tries max-to-6 and max-to-4 per 16-value block. `joint_mse` is the production JSO scale rule selected by the `joint_scale_opt` lever: it chooses from `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` under the served FP8-snapped scale objective. All preserve the compressed-tensors NVFP4 schema and vLLM kernel. |
| `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` | `6,4` | `export_native_compressed.py:298` | Candidate max-to-levels for NVFP4 joint scale optimization. Extend only for explicit JSO ablations; production defaults use the validated `{6,4}` grid. |
| `PRISMAQUANT_NVFP4_JOINT_SCALE_OPT` | `0` | `export_native_compressed.py`; `production_weight_cache.py:2162` | Direct JSO switch for callers that bypass `PRODUCTION_CACHE_LEVERS`. |
| `PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID` / `_SPAN_LO` / `_SPAN_HI` | `—` / `0.75` / `1.25` | `export_native_compressed.py:574` / `:582` / `:586` | Per-tensor global-scale search grid for the JSO fused joint pre-pass. |
| `PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING` | `0` | `export_native_compressed.py:359`; fingerprint `:5466` | Research lever: score NVFP4 scale candidates under the FP8-snapped effective scale with a per-tensor global fixed point. More serve-faithful in principle but changes shipped NVFP4 bytes for `joint_mse`/`four_over_six` — default OFF pending a served gold-metric A/B. Recorded in the export fingerprint. |
| `PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE` | **on** | `export_native_compressed.py:1445` | M19: NVFP4 export re-derives block scales from the cached production render using the SAME scale rule the render chose (`joint_mse` under JSO), instead of re-quantizing the bf16 dequant with `static_6`. Served-validated −6.6% KL / −3.3% PPL on the 4B paired A/B. Since the 2026-07-02 audit (M2) this also covers packed-expert re-pack and the fused joint-global pre-passes (rule = the cache's recorded `nvfp4_scale_rule` lever; env default when nothing is recorded). `0` reproduces pre-M19 artifacts. |
| `PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES` | `0` | `perturbed_x_cache.py:108` | Perturbed-X emulation hooks: `1` quantizes NVFP4 activations with the SERVE-faithful two-level semantics (static per-tensor `input_global_scale` derived from the calibrated max_abs — honoring `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` — plus FP8 snap of each 16-group block scale, including block-zeroing and above-calibration-amax clipping) instead of the dynamic exact-fp32-scale RTN. Closes the audit M18-residual/C1 measurement gap; default off pending a served correlation study. |
| `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` | `0` | `export_native_compressed.py:888` | C1 (2026-07-02 audit): `1` switches NVFP4 `input_global_scale` to the compressed-tensors/vLLM `generate_gparam` convention `448·6/amax`, placing serve-time FP8-stored activation block scales in (0,448] instead of the legacy (0,1] — rescues blocks ≫64× below calibration amax from FP8 subnormals, but CLIPS any serve block whose amax exceeds calibration amax. Served A/Bs (byte-identical weights, only this scalar ×448, 2 window draws each): 35B MoE frontier **−14.1% KL (win)**; 27B regen dense **+37.5% (loss)**; LFM2.5 thin-calib smoke +5.8% (loss). Strongly artifact-dependent → default stays legacy; opt in per artifact behind a served A/B. The scale is a free post-export knob (in-place patch, no re-render) — see `/home/rob/dq-runs/c1-igs-ab-20260702/patch_igs.py`. |
| `PRISMAQUANT_GPTQ_BLOCK_SIZE` | `128`, via `PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE` fallback | `export_native_compressed.py:1793`, fallback `export_native_compressed.py:1794` | Column block size for the FP-Quant-style GPTQ OBS update across NVFP4, FP8_DYNAMIC/FP8_E4M3, and explicit MX research formats. Quantizer scales are fixed before the solve; each column is quantized and its error is propagated through the current GPTQ block and later blocks. |
| `PRISMAQUANT_GPTQ_DAMP` | `""` (fixed 1.0) | `export_native_compressed.py:1872`; fingerprint `:5464` | Overrides the fixed GPTQ damping constant. |
| `PRISMAQUANT_GPTQ_DAMP_SWEEP` | **`0`** (one reader) | `export_native_compressed.py:1857`, fingerprint | Damp sweep is OFF for production render/export (fixed damp 1.0, 2026-06-12); `1` reproduces historical artifacts. **D5 is fully closed:** the second reader with the opposite default lived in `kl_sensitivity_probe`, was made a delegation to `production_weight_cache._resolve_production_render_levers` on 2026-07-30, and the file itself was walled the same day with the L3 cascade (`archive/l3_propagated_2026-07-30/`, re-vet R4). One reader, one default; the contract is pinned by `tests/test_production_weight_cache.py`. |
| `PRISMAQUANT_GPTQ_DAMP_ROLES` | unset | `export_native_compressed.py:1939` | Per-role GPTQ damp override, e.g. `qkv=1.0,o_proj=1.0,gate_up=0.3,down=3.0`. Default-off research lever (the 2026-06-22 per-role served A/B was NULL; fixed damp 1.0 is final). Unlisted roles keep the fixed damp. |
| `PRISMAQUANT_GPTQ_STATIC_ACT_ORDER` | `0` | `export_native_compressed.py`; `production_weight_cache.py:2158` | Direct static-act-order switch for callers that bypass `PRODUCTION_CACHE_LEVERS`. |
| `PRISMAQUANT_DAMP_ANALYTICAL` / `_C` | `""` / `1.784e-5` | `export_native_compressed.py:2201` / `:2205` | Archived closed-form damp (refuted: +100–161% KL vs the discrete sweep). Kept for reproduction only. |
| `PRISMAQUANT_DAMP_SWEEP_LOG` | unset | `export_native_compressed.py:2178` | Per-Linear damp-sweep decision log. |
| `PRISMAQUANT_ACT_CLIP_QUANTILE` | `0.999` | `export_native_compressed.py:681`; fingerprint `:5468` | Calibrated activation-max clip quantile. |
| `PRISMAQUANT_FP8_SCALE_SWEEP_FACTORS` | `0.25 … 2.0` (9 log-spaced) | `export_native_compressed.py:3543` | Candidate FP8 scale multipliers for the explicit `scale_sweep` ablation. |
| `PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN` | `0` | `export_native_compressed.py:1663`, enforced `:6113-6127` | Research/A-B escape hatch: allows non-BF16 packed-MoE experts to skip the production-cache GPTQ render and export RTN bytes. Never use for a production artifact. |
| `PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ` | `0` | `export_native_compressed.py:1674` | 295B-class alternative when no dequant cache can exist: run expert GPTQ inline during export. |
| `PRISMAQUANT_ALLOW_UNSCALED_FP8` | `0` | `layer_streaming.py:417` | Streaming-load guard override: by default a float8-dtyped checkpoint tensor with no entry in the fp8 scale-inv map fails fast (loading raw FP8 codes as if they were weights is the historical ±448-range corruption). Set `1` to permit the raw cast anyway (debug only). |
| `PRISMAQUANT_EXPERT_LAZY_FILL` | **on** | `validate_assignments_kl.py:337` | M4 frontier expert selection: `validate_assignments_kl` lazily renders a Pareto point's missing packed-expert entries (e.g. FP8) into the shared frontier cache just before scoring, on the BUILD/render calib split, then re-pickles the cache so recache/export ship the same bytes real KL selected. The format-menu build eager-renders only the NVFP4 rung. `0` restores the legacy hard-fail on expert cache misses. |
| `PRISMAQUANT_STRICT_ASSIGNMENT_COVERAGE` | **conditional** — defaults ON when a production weight cache is supplied *or* `PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT` is set | `kl_measurement.py:5491`, default computed `kl_measurement.py:5483-5489` | Coverage guard for assignment-required cache entries in the **KL hook path** (not, despite the older wording here, the exporter). Missing non-BF16 renders fail early instead of falling through to RTN. |
| `PRISMAQUANT_STRICT_PRODUCTION_CACHE` | **on** | `kl_measurement.py:2586` (helper default `True`, `memory_management.py:20`); `validate_assignments_kl.py:325` | KL/activation-cache residency guard. Missing required production-cache weights fail fast by default; set `0` only for explicit legacy/non-production fallback runs. |
| `PRISMAQUANT_DO_NO_HARM` | **on** | `export_native_compressed.py:4209`, `:4350`, `:4492`, `:4797`; fingerprint `:5460` | Enables export-time GPTQ-vs-RTN do-no-harm gates where supported. Failures and reverts are counted in export provenance. |
| `PRISMAQUANT_DO_NO_HARM_VERBOSE` | unset | `export_native_compressed.py:4231`, `:4372`, `:4514`, `:4831` | Per-Linear do-no-harm decision log. |
| `PRISMAQUANT_RENDER_PROGRESSIVE_GATES` | **on** | `production_weight_cache.py:1831` | Production-cache render gate for local mechanisms. All formats use the same progressive order, while unsupported mechanisms are format-gated off. NVFP4 can score FourOverSix, static activation ordering, GPTQ, joint-scale optimization, and optional scale_sweep candidate packages; FP8_DYNAMIC/FP8_E4M3 can score GPTQ with damp sweep, and can additionally score explicit scale_sweep when enabled. MXFP8 remains explicit opt-in and can score static activation ordering plus GPTQ using the canonical E8M0 scale rule. Regressive candidates keep the prior accepted render. Cache metadata records decisions in `render_gates`; FourOverSix has a compact `four_over_six` summary. |
| `PRISMAQUANT_RENDER_GATE_MIN_GAIN` | `0.0` | `production_weight_cache.py:1495` / `:1905` | Minimum relative gain required by the progressive render gate (reason string `below_min_gain`, `render_score.py:122`). Keep at `0.0` for normal runs so tiny local improvements can accumulate; raise only for ablations. |
| `PRISMAQUANT_TARGET_PROFILE` | unset | `export_native_compressed.py:1280` | Serving-profile override for direct exporter invocations; `run-pipeline.sh` passes `TARGET_PROFILE` on the CLI instead. |
| `PQ_EXPORT_VECTOR_CHUNK` | `auto` (cap 128) | `export_native_compressed.py:4878` | Upper bound on the grouped-export vectorization chunk. **Note the `PQ_` prefix** — the only non-`PRISMAQUANT_` flag in the exporter. |
| `PRISMAQUANT_FISHER_WEIGHTED_GPTQ` | **archived** | `production_weight_cache.py:2169` | Fisher-weighted GPTQ is archived under `archive/fisher_2026-05-15/` and rejected by the production pipeline. |
| `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP` | `64` | `export_native_compressed.py:784`; `render_score.py:38` / `:55` | Historical Fisher-weighted GPTQ row-weight cap; not used by the production pipeline. |
| `PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS` | `-1,0` | `export_native_compressed.py:3081` | Candidate E8M0 exponent shifts for the MXFP8 JSO block search. Live code, but only reached when MXFP8 is rendered with `joint_scale_opt=True`; `joint_scale_opt` is registered as an `nvfp4_scale_optimizer` mechanism (`render_score.py:349-358`) and is not offered to MXFP8 under any production lever set, so in practice it never fires. |
| `PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS` | `0` | `export_native_compressed.py:2917` | Explicit-ablation candidate E8M0 exponent shifts for MXFP8_E4M3 activation-weighted scale search. The default is a no-op; nonzero shifts are experimental and refine the current accepted render under the same progressive gate. |

## 3. Pipeline production-cache flags

These are `prismaquant/run-pipeline.sh` environment variables rather than
`PRISMAQUANT_*` flags. Research levers outside the current production recipe
live under `archive/`; the pipeline fails fast (`exit 2`) when archived Fisher
levers or cost modes are requested.

| env var | default | what it does |
|---|---|---|
| `PRODUCTION_CACHE` | `1` | Build and use a `ProductionWeightCache` so export packs the same rendered weights that KL/polish measured. |
| `PRODUCTION_RECACHE` | `1` | Replay calibration with production weights installed and re-fit `activation_max_abs` before export. |
| `PRODUCTION_CACHE_LEVERS` | `gptq,static_act_order,joint_scale_opt` | V1 production render levers. FP8_DYNAMIC/FP8_E4M3 uses GPTQ without static ordering or JSO because the served representation is per-row scaled FP8 dynamic. `static_act_order` applies to production microscaling GPTQ formats: NVFP4, MXFP4, and explicit MXFP8. `joint_scale_opt` applies only to NVFP4. MXFP4/MXFP8 use the canonical E8M0 scale rule when explicitly requested. `scale_sweep` remains available for explicit ablations but is not a default. Runtime activation scores use their served activation quantizers; NVFP4 is the only current score path that applies the calibrated activation-max clip. |
| `FORMATS` | `NVFP4,FP8_DYNAMIC,BF16` (`run-pipeline.sh:45`) | Allocator format menu. MXFP8 is de-menued for inference — exact-scale FP8 Pareto-dominates it. |
| `TARGET_BITS` | `4.75` | Allocator bit budget over quantizable parameters. |
| `TARGET_PROFILE` | `vllm_packed_moe` (`run-pipeline.sh:91`) | Serving profile. Note it **overrides** the model spec's `default_serving_profile` (`serving_profiles.py:426-427`) — a known contract leak. |
| `FISHER_WEIGHTED_GPTQ` | archived | Any truthy value is rejected; archive context lives under `archive/fisher_2026-05-15/`. |
| `FISHER_OUTPUT_MSE_ALLOCATOR` | archived | Any truthy value is rejected; V1 allocation uses the non-Fisher objective plus measured frontier validation. |
| `COST_MODE` | `production-render-score` (`run-pipeline.sh:187`) | `production-render-score` (default) renders the full `FORMATS` menu for every Linear through `ProductionWeightCache` and writes an allocator-compatible cost.pkl from the recorded production render scores. `local` keeps the legacy `h_trace × output_mse` objective (and is **required** for the GGUF and nvfp4_cb containers — see the gates at `:99` and `:121`). `aura` runs the AURA downstream-KL-adjoint cost (`aura_cost.py`, served −38% KL @4B / −17.9% @27B vs the h_trace×output_mse baseline) against a production-rendered dW cache; on packed-MoE models the route-flip-blind smooth cost is replaced for experts by measured empirical unit-KL (`expert_empirical_cost.py`, FP8 kept in the menu) merged into one hybrid payload, with MTP/visual sidecar rows backfilled from the baseline cost. AURA is fully wired but **opt-in** (sub-stages at `:314-336`, `:825-956`). `grouped-kl`, `production-render-staged`, `fisher`, `hdq` and `multi-shot` are **archived** and `exit 2`. `production-render-staged` was walled 2026-07-30 (`archive/production_render_staged_2026-07-30/`, re-vet R17): it rendered NVFP4 first and offered promotion formats only to the top-30% error tail, so on 27B its last-token-KL screen improved (0.0232 vs 0.0280) while direct WikiText PPL regressed (10.83 vs 8.33) — "Do not ship". |
| `AURA_COST_NPROBES` / `AURA_COST_NSAMPLES` / `AURA_COST_SEQLEN` / `AURA_COST_CALIB_SEED` | `32` / `8` / `128` / `42` | `COST_MODE=aura` probe/calibration volume (defaults = the regen-27b recipe). Also `AURA_COST_LINEAR_CHUNKS` (8), `AURA_COST_PROBE_MICROBATCH` (8), `AURA_COST_MIN_FREE_GIB` (18). |
| `AURA_COST_DTYPE` | `auto` | Resident model dtype for the aura_cost stage. `auto` sizes the checkpoint (fp8 sidecars counted at 1 byte/param) and picks `float32` (additivity-preferred, the 27B regen regime) only when params×4 bytes + `AURA_COST_MIN_FREE_GIB` fits in MemAvailable, else `bfloat16` (35B-class — fp32 is ~140 GiB against the 121 GiB pool and OOM-kills the box). |
| `AURA_EXPERT_NSAMPLES` / `AURA_EXPERT_SEQLEN` | `16` / `512` | `COST_MODE=aura` empirical packed-expert unit-KL stage calibration volume (the 35B arm-E recipe). |
| `VALIDATED_FRONTIER_MATERIALIZATION` | `hooks` | How the validated frontier materializes each Pareto point. `hooks` = all points in one process (fast; needs model + full render set co-resident — OOMs 35B-class MoE on the 128 GB pool). `inplace` = one `validate_assignments_kl` process per point, per-point JSONs merged for selection (the memory-fit 35B path). |
| `PRODUCTION_RENDER_COST_NSAMPLES` / `_SEQLEN` / `_SEED` | `8` / `1024` / `42` | Calibration contract for `COST_MODE=production-render-score`, using the production cache scorer. |
| `PRODUCTION_RENDER_COST_SCORE_FIELD` | `weight_mse` | Render-score field used as allocator cost. `weight_mse` (default since 2026-07-02, audit M6) routes through the dimensionally-consistent `h_trace × weight_mse` path — the legacy `output_mse` product double-counted activation energy E‖x‖² (h_trace already contains it), a per-Linear bias ∝ in_features·x_rms². Served two-arm A/B at matched 4.75 bpp: Qwen3-4B KL −50.8% / 32k-PPL −15.1%; Qwen3-0.6B KL −58.5% / 32k-PPL −24.4% (5 window draws each, same pipeline seeds). `output_mse` reproduces the historical objective; `score_sum`/`score` are ablation fields. 27B-class confirmation = ladder debt. |
| `SELECTION_MODE` | `surrogate` (`run-pipeline.sh:250`) | `surrogate` preserves the normal allocator-selected `TARGET_BITS` assignment. `validated-surrogate` writes allocator Pareto assignments, builds a format-menu production cache, measures real assignment KL for each Pareto point, selects the measured KL/bpp kneedle with `prismaquant.select_validated_frontier`, then recaches and exports the selected assignment. The flagship artifacts used `validated-surrogate`; it is not the default. |
| `VALIDATED_FRONTIER_NSAMPLES` / `_SEQLEN` | `$NSAMPLES` / `$SEQLEN` | Calibration size for measured-frontier KL selection. Keep these at the artifact validation contract for 27B decisions; lower values are smoke-only. |
| `VALIDATED_FRONTIER_PICK` | `kneedle` | Selection rule for `SELECTION_MODE=validated-surrogate`: `kneedle`, `best-kl`, or `lowest-bpp`. Production selection should use `kneedle` unless the run is explicitly an ablation. |
| `VALIDATED_SOURCE_PREFETCH` | `require` (`run-pipeline.sh:282`) | Source-checkpoint residency gate for the validated-frontier stages (`source_prefetch.py`). `require` fails fast rather than silently becoming NVMe-bound. |
| `PRODUCTION_CACHE_LRU_GB` | `64.0` | Resident tensor budget for disk-backed production-cache use in recache and export. The 27B n=8 recache smoke needed `45.32 GiB` for the selected assignment. |
| `PRODUCTION_CACHE_PREFETCH` | `require` (`run-pipeline.sh:170`) | Standalone recache prefetch policy. `require` fails fast when assignment-required weights cannot fit resident. Note the **exporter has no such argument** — its prefetch helper (`export_native_compressed.py:1392-1412`) has no require mode, so a cache miss at export silently yields 0 prefetched keys. |
| `PRODUCTION_CACHE_PREFETCH_WORKERS` | `4` | Thread count for eager production-cache prefetch. |
| `EXPORT_CONTAINER` | `compressed-tensors` | `gguf` and `nvfp4_cb` switch stage 4 to their own exporters and impose gates (see §5, §8). |

## 4. CUDA / system flags

| env var | recommended | read at | what it does |
|---|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | set by `incremental_probe.py` at module load | Required on UMA hardware (DGX Spark) to keep the CUDA caching allocator from hoarding freed blocks. |
| `PRISMAQUANT_L3_CUDA_GRAPHS` | `auto` | `kl_measurement.py:4688` / `:5122` | Graphs decoder-tail L3 propagation only when the same graph key has enough repeated calibration calls. Default threshold `8`; override with `PRISMAQUANT_L3_CUDA_GRAPHS_MIN_CALLS` (composed as `f"{name}_MIN_CALLS"`, `kl_measurement.py:165`). |
| `PRISMAQUANT_COORD_LANE_CUDA_GRAPHS` | `auto` | `kl_measurement.py:3591` / `:3770` / `:4172` | Graphs lane-batched coord flip evaluation only when repeated calls can amortize capture. Replay-cache coord batches are one-shot, so auto leaves them eager. Default thresholds `8` (replay) / `16` (full-forward); override with `PRISMAQUANT_COORD_LANE_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_KL_CUDA_GRAPHS` | `auto` | `kl_measurement.py:5516` | Graphs assignment-KL validation only for larger calibration batches. Default threshold `16`; override with `PRISMAQUANT_KL_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_VALIDATION_CUDA_GRAPHS` | **on** | `validation_harness.py:52` | Graph capture in the validation harness. |
| `PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE` / `PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE` / `PRISMAQUANT_VALIDATION_CUDA_GRAPH_CACHE_SIZE` | `4` each | `kl_measurement.py:5348` / `:2006`; `validation_harness.py:44` | Per-registry graph-cache capacity (registry `max_entries_env` parameter). |
| `PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH` | `4` | `kl_measurement.py:2054` | Global fallback capacity when a registry has no dedicated cache-size env set. |
| `PRISMAQUANT_KL_CUDA_GRAPHS_VERBOSE` | unset | `kl_measurement.py:5349` | Per-capture logging for the assignment-KL registry. |
| `PRISMAQUANT_GRAPH_SHARED_POOL` / `PRISMAQUANT_GRAPH_OUTPUT_CLONE` / `PRISMAQUANT_GRAPH_AUDIT` | unset | `kl_measurement.py:421` / `:1546`; `memory_management.py:242` | Graph memory-pool sharing, output cloning, and capture auditing. |
| `PRISMAQUANT_COORD_REPLAY_CACHE` | `off` | `kl_measurement.py:3576` / `:4156` | Opt-in `LayerHiddenStateCache` for coord descent. It reduces tail layer forwards but currently copies too much baseline model state on large Qwen runs, so the fast default is lane-batched eager evaluation. |
| `PRISMAQUANT_L3_PREQUANT_CACHE` / `_RESERVE_GB` / `_RESERVE_FRACTION` / `_PEAK_MULTIPLIER` | unset / — | `kl_measurement.py:3485` / `:320` / `:324` / `:330` | L3 pre-quantized weight cache and its memory reserve policy. |
| `PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE` | unset | `kl_measurement.py:3491` / `:4061` / `:4549` | Freeze the perturbed-activation cache across L3 chunks. |
| `PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB` / `_FRACTION` | unset | `kl_measurement.py:3329` / `:3333` / `:3373` / `:3377` | Lane-count ceiling derived from free memory. |
| `PRISMAQUANT_L3_MIN_HOST_MEM_GB` | unset | `kl_measurement.py:363` / `:385` | Host-memory floor for L3 pair/scout diagnostics. When set, L3 raises `GPUMemoryBudgetExceeded` between paired-override chunks if `/proc/meminfo` `MemAvailable` drops below this many GiB, giving long runs a chance to stop before system OOM. |
| `PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES` / `_MIN_FREE_GB` / `_MIN_FREE_FRACTION` | — / — / `0.05` | `perturbed_x_cache.py:1016`; `kl_measurement.py:190` / `:192` | Frozen-weight cache capacity and free-memory floors. |
| `PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE` | unset | `kl_measurement.py:5463` (written to `0` by `validate_assignments_kl.py:938` under `--disable-frozen-weight-cache`) | Enable the frozen-weight cache in assignment-KL measurement. |
| `PRISMAQUANT_GPU_MEM_RESERVE_GB` / `_FRACTION`, `PRISMAQUANT_HOST_MEM_RESERVE_GB` / `_FRACTION`, `PRISMAQUANT_MAX_GPU_MEM_GB` | unset | `memory_management.py:144` / `:147` / `:156` / `:159` / `:180` | Memory-budget reserves used by the resident-fit calculations. |
| `PRISMAQUANT_UMA_MEMORY_INFO` | `auto` | `memory_management.py:98` | Treat GPU+host as one physical pool (DGX Spark). |
| `PRISMAQUANT_EMPTY_CACHE_EACH_REPLAY_BATCH` | unset | `kl_measurement.py:112` | `torch.cuda.empty_cache()` between replay batches (memory-pressure debugging). |
| `PRISMAQUANT_FULL_SEQUENCE_KL` | unset | `kl_measurement.py:133` | Score all positions instead of the last-token hook screen. |
| `PRISMAQUANT_DETERMINISTIC` | `0` | `build_production_cache.py:505` | Deterministic algorithms + `CUBLAS_WORKSPACE_CONFIG` for the render path. |
| `PRISMAQUANT_MASK_CUDA_DURING_META_INIT` | `1` | `streaming_model.py:114` | Hide CUDA during meta-device model construction. |
| `PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT` | unset | `perturbed_x_cache.py:799` / `:1099` | Declares that weights are staged by an external owner (`weight_session`); also flips the `STRICT_ASSIGNMENT_COVERAGE` default to ON. (Its other reader, `kl_sensitivity_probe`, was walled 2026-07-30 — `archive/l3_propagated_2026-07-30/`.) |
| `PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE` | unset | `perturbed_x_cache.py:470` / `:821` | Share one rendered-format cache across perturbed-X passes. |
| `PRISMAQUANT_PROD_ACT_SCALES` | unset | `perturbed_x_cache.py:91`; `production_weight_cache.py:1286` | Use production activation scales in the emulation hooks. |
| `PRISMAQUANT_FUSED_KERNEL_NVFP4` / `_OVER_PROD_CACHE` | unset | `perturbed_x_cache.py:865` / `:875` | Route emulation through the fused NVFP4 kernel, optionally in preference to the production cache. |
| `PRISMAQUANT_NVFP4_FUSED_JIT_WARMUP` | unset | `kernels/nvfp4_fused.py:365` | Pre-JIT the fused NVFP4 kernel. |
| `PRISMAQUANT_DISABLE_RTN_COMPILE` | `""` | `format_registry.py:479` | Pin the RTN quantize/dequantize hot path to eager instead of `torch.compile`. |
| `PRISMAQUANT_ALLOW_PYTORCH_FALLBACK` | **no live reader** | `archive/orphans_2026-07-30/prismaquant/_fast_kernel_guard.py:42` | Permitted the slow PyTorch path where a fast kernel is expected. The guard that read it (`require_fast_kernels`) lost its only caller when `polish_from_assignment` was archived on 2026-05-15 and was walled 2026-07-30 (re-vet R19). **Principle 9's kernel-performance gate is manual today** — see `docs/ARCHITECTURE.md` §12 (LOW). |
| `PRISMAQUANT_IQ_COMPILE_SWEEP` | `1` | `gguf_iq_formats.py:181` | `torch.compile` the GGUF IQ-quant sweep kernels. |
| `PRISMAQUANT_TMPDIR` | unset | `sensitivity_probe.py:88` (then `TMPDIR`, `sensitivity_probe.py:89`) | Scratch directory for probe temporaries. **Never point either at `/tmp`** — see the operational landmines. |
| `PRISMAQUANT_VALIDATION_PROD_CACHE` / `_DIR` / `_LRU_GB` | unset / unset / `16` | `validation_harness.py:303` / `:308` / `:311` | Production cache wiring for the validation harness. |
| `PRISMAQUANT_VALIDATION_SKIP_END_KL` | `0` | `validation_harness.py:248` | Skip end-KL in the harness (smoke only). |
| `PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE` | unset | `validation_harness.py:392` | WikiText stride for the harness PPL screen. |
| `PRISMAQUANT_VALIDATION_FAKE_METRICS` | unset | `validation_harness.py:166` | Test-only metric injection. Never set in a run whose numbers will be cited. |
| `PRISMAQUANT_SMOKE_MODEL` / `_SAMPLES` / `_SEQLEN` / `_SEED` / `_DETERMINISM` | unset / `2` / `32` / required / `0` | `tools/smoke_graph_memory.py:101` / `:142` / `:140` / `:215` / `:25` | Graph-memory smoke harness knobs. |

## 5. NVFP4-CB / FP8-CB build lane

Enabled by `EXPORT_CONTAINER=nvfp4_cb`, which the pipeline gates to
`COST_MODE=local`, `TARGET_PROFILE=nvfp4_cb`, `PRODUCTION_CACHE=0` and
`PRODUCTION_RECACHE=0` (`run-pipeline.sh:121-129`). Lane record:
`docs/lanes/nvfp4-cb/PLAN.md`.

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_CB_ENCODE_TIER` | `balanced` | `nvfp4_cb_formats.py:145` (`_resolve_encode_tier`; env const `:128`, default const `:130`); provenance stamp `expert_empirical_cost.py:907` | Encoder speed-accuracy tier: `fast` / `balanced` / `max`. `max` is the original exhaustive scale sweep, bit-identical (regression-pinned); `balanced`/`fast` use the analytic s0 init + moment-scored micro-sweep + hill climb (measured ×3.9/×5.9 mean, `docs/lanes/nvfp4-cb/encode_tiers.md`). |
| `PRISMAQUANT_CB_ENCODE_COMPILE` | **on** (`"1"`) | `nvfp4_cb_formats.py:614` (env const `:140`) | `torch.compile` the CB moment-scoring inner kernels (fast/balanced tiers only; max never compiles). Set `0` to pin eager — the compiled-vs-eager tie-flip caveat applies within a tier. |
| `PRISMAQUANT_CB_LADDER_INTERP` | `0` | `measure_quant_cost.py:1593` (dense path); `expert_empirical_cost.py:1090` (expert path) | **Live in both cost paths** (the older "declared wiring point, not read by any code" note here was wrong). `1` enables per-`(family,mode)` RD-law ladder interpolation: anchors + holdout are measured normally, predicted rungs are fitted per TENSOR and holdout-gated with a measured fallback, so a tensor that defies the law never receives an interpolated cost (`encode_tiers.md` §B/§C). Since R20 (2026-07-30) **both paths run ONE law** (`expert_empirical_cost._cb_ladder_law:482`: floored-linear in the exact ceil-first rate factor `R(k)` → smooth floor law → log-linear), and both log a holdout accept/reject rate when the interp runs. One shell knob drives both wirings: `CB_LADDER_INTERP=1` (`run-pipeline.sh:323-324` and `run-pipeline.sh:1014-1015`). |
| `PRISMAQUANT_CB_LADDER_TOL` | `0.10` | `measure_quant_cost.py:1563` | Holdout-gate relative-error tolerance for the dense ladder; matches the expert stage's `--ladder-holdout-tol` default. **This is the FALLBACK value, not the rule** (R20): per `encode_tiers.md` §B the gate must trust a fit only where the holdout error clears the *between-seed cost noise*, so `_cb_ladder_holdout_tol` (`expert_empirical_cost.py:559`) derives the tolerance from the paired per-calibration-window spread of the measured rungs — free on the expert path, which already measures every unit KL window by window. The constant stands only where that datum is absent or degenerate: the **dense** path measures each `(tensor, format)` exactly once (accumulator `_count == 1`), so it has no between-draw spread and always uses this value. |
| `PRISMAQUANT_CB_COL_WEIGHTS` | unset | `measure_quant_cost.py:1350` | Path to the shared CB col-weights (imatrix) pickle; exported by the pipeline at `run-pipeline.sh:642`. This is the lockstep contract: measured CB cost and the exporter's weighted-VQ render must use the same weights, including the synthesized per-expert down_proj replay entries the inline module-input pool cannot provide. |
| `PRISMAQUANT_EXPERT_CALIB_BATCH` | `1` | `expert_empirical_cost.py:92` (`_CALIB_BATCH_ENV`), used in `_calib_batch()` | Calibration sequences per forward in the empirical expert unit-KL stage — the dominant-wall knob of `[2d]` / `[2d-CB]`. `1` preserves the historical per-sequence numerics exactly; `>1` batches independent windows (both arms always use the same batching, so the KL comparison stays internally consistent). |
| `PRISMAQUANT_EXPORT_REUSE_PRIOR` | unset | `export_nvfp4_cb_streaming.py:1360` | Env alias for `--reuse-prior`: delta-export byte-copies CB/stock targets whose `(format, scheme, codebook)` are unchanged from a prior artifact instead of re-encoding. Surfaced in the shell as `EXPORT_REUSE_PRIOR` (`run-pipeline.sh:1635`). Default off — every target encodes fresh. |
| `PRISMAQUANT_EXPORT_REUSE_VERIFY` | `3` | `export_nvfp4_cb_streaming.py:1363` | Number of reused tensors re-encoded fresh and byte-checked; any mismatch aborts. |

Shell-side CB knobs (`run-pipeline.sh`): `CB_LADDER_INTERP` (`0`),
`CB_EXPERT_EMPIRICAL` (`1`, `:332`), `CB_EXPERT_NSAMPLES` / `CB_EXPERT_SEQLEN`
(`16` / `512`, `:969-970`), `CB_EXPERT_SAMPLE` (`0`, `:1021`), `CB_COL_WEIGHTS`
(`$WORK_DIR/artifacts/cb_col_weights.pkl`), `CB_OUT`
(`$WORK_DIR/exported_nvfp4_cb`), `CB_CODEBOOK_SOURCE` (`lattice`),
`CB_CODEBOOK_ITERS` (`4`), `CB_CODEBOOK_SEED` (`0`), `CB_SCALE_SWEEP` (`1`;
`0` is the one-shot amax/grid-max ablation only), `CB_SCALE_CODING` (`v1`;
`two_tier` layout-v2 serve gates are pending — do NOT ship),
`EXPORT_REUSE_PRIOR` / `EXPORT_REUSE_VERIFY`.

## 6. Non-`PRISMAQUANT_` shell vars read directly by Python

A coupling worth knowing: several `run-pipeline.sh` variables are read straight
out of `os.environ` by library code, so they act on library defaults even when
the corresponding CLI flag is absent.

| env var | default | read at |
|---|---|---|
| `NSAMPLES` | `32` | `autoscale.py:403`, `streaming_model.py:881` |
| `SEQLEN` | `1024` | `autoscale.py:404`, `streaming_model.py:882` |
| `LAYERS_PER_SHARD` | `1` | `autoscale.py:406`, `streaming_model.py:879-880` |
| `CACHE_HEADROOM_GB` | — | `autoscale.py:407`, `streaming_model.py:871` |
| `PREFETCH_WORKERS` | `auto` | `incremental_probe.py:2959`; `streaming_model.py:284` |
| `PREFETCH_MIN_AVAILABLE_GB` | `auto` | `incremental_probe.py:2963`, `streaming_model.py:301` |
| `PREFETCH_LOOKAHEAD` | `auto` | `incremental_probe.py:2954` |
| `ACTIVATION_ROWS_LIMIT` | `256` | `incremental_probe.py:2979` |
| `TMPDIR` | — | `sensitivity_probe.py:89`, `validate_assignments_kl.py:963` |
| `VLLM_URL` | `http://localhost:8000` | `validate_quantized_model.py:507` |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | set by the allocator | `allocator.py:1215-1216` |
| `CUBLAS_WORKSPACE_CONFIG` | set under `PRISMAQUANT_DETERMINISTIC` | `build_production_cache.py:519` |

## 7. gridbook serving-plugin flags

These are read **inside the vLLM process** by `plugins/gridbook/gridbook/`
(PyPI `gridbook`), not by the build pipeline. They select kernels and schedules;
several change numerics, and those are marked. Registry key is `gridbook` with
legacy alias `prismaquant` (`plugin.py:133-139`).

### 7a. Dispatch and dense Linear (`linear.py`, `ops.py`, `cuda_ext.py`)

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_CB_DECODE` | `cuda` | `linear.py:299`; `moe.py:1927` | Selects the CUDA GEMV decode path; any other value falls back to Triton. |
| `PRISMAQUANT_CB_CUDA_M_MAX` | `8` | `linear.py:53` | Within the decode regime, the CUDA GEMV handles `M ≤ this`; above it Triton's `tl.dot` wins (measured 0.66× at M=16, 3.2× loss going the other way at M=1-2). |
| `PRISMAQUANT_PREFILL_M_THRESHOLD` | `16` | `linear.py:46` | Decode/prefill regime boundary. Set huge to force the Triton decode path at prefill (isolates the transient-expansion lever). |
| `PRISMAQUANT_CB_FUSED_MIDM` | `1` | `linear.py:439` | Fused mid-M kernel for `16 < M ≤ 128` at `k ∈ {28,32,36,40,44,48}` (1.04-1.45× on GB10). The `_scaled` entry applies both scales inside its fp32 EVT epilogue and rounds once to bf16, matching `cutlass_scaled_mm`'s rounding order. |
| `PRISMAQUANT_CB_PREFILL_DENSE` | unset | `linear.py:455` | `persistent` enables the transient-free persistent-tile dense prefill kernel for `M > 128`. **Any constraint miss falls through silently** to the shipping expand+cutlass path. Rounds to bf16 before scaling — a rounding-ORDER difference vs the shipping path. |
| `PRISMAQUANT_PTC_VARIANT` | `1` | `linear.py:462` | Persistent-tile-compute kernel variant selector. |
| `PRISMAQUANT_ENABLE_PTC` | unset | `cuda_ext.py:165` | The PTC extension builds only on explicit `=1` — **quarantined** after the 2026-07-23 wedge crisis. |
| `PRISMAQUANT_CB_EXT_DIR` | `~/.cache/prismaquant-cb-ext` | `cuda_ext.py:100` / `:173` / `:226` | Override for the CUDA-extension build directory (base ext, `ptc/`, `fused/`). |
| `PRISMAQUANT_CB_DISPATCH` | `op` | `ops.py:271` | `inline` bypasses the custom-op dispatch wrapper. |
| `PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE` | unset | `ops.py:36` | `1` tags the CB ops `cudagraph_unsafe`, restoring the old partition boundary. Reproduces the compile+piecewise corruption for an A/B; do not ship. |
| `PRISMAQUANT_PRELOAD_FUSED` | unset | `plugin.py:126` | `1` force-builds and loads the fused ext even when its dispatch env is off, so both arms of a served logprob comparison carry identical CUDA-extension residency. Required whenever an A/B could otherwise be confounded by the session-arithmetic-drift mechanism. |
| `PRISMAQUANT_DEBUG_PREFIXES` | unset | `config.py:340` (weight-prefix → scheme resolution); `linear.py:334` (persistent-TC ineligibility) | `1` logs prefix/eligibility resolution to stderr. First thing to set when CB tensors appear not to load. |

### 7b. MoE paths (`moe.py`)

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_CB_PREFILL` | `auto` (fp8) / `loop` (fp4) | `moe.py:404` | MoE prefill strategy override. `auto` is the measured per-layer selection promoted in `3062fbf`. `l2_pipeline` selects the diagnostic path directly. |
| `PRISMAQUANT_CB_AUTOTUNE_MIN_M` | `1024` | `moe.py:1395` | Minimum M before prefill autotuning engages. |
| `PRISMAQUANT_CB_PREFILL_AUTO_FORCE` | unset | `moe.py:1397` | Pins the autotuner to one named candidate. |
| `PRISMAQUANT_CB_L2_AUTOTUNE` | unset | `moe.py:1375` | `1` adds the L2-pipeline variant to the autotune candidates. **DIAGNOSTIC-ONLY** (`afc64ec`): gated off by default after three live-serve wedges; its parity / ragged / buffer-rotation race tests have never executed on hardware. |
| `PRISMAQUANT_CB_L2_WINDOW_MB` / `_GROUP` / `_MIN_M` / `_OVERLAP` | unset / unset / `128` (`moe_l2.py:138`) / unset | `moe.py:920` / `:923` / `:1115` / `:1171` | L2-pipeline per-half window cap (can only lower the device-derived cap), forced expert group size, tiny-M floor, and cross-stream overlap. `_OVERLAP=1` refuses to run inside a graph capture rather than risk the driver hang. |
| `PRISMAQUANT_CB_PREFILL_GROUPED_MM` | `0` | `moe.py:1471` | Opt-in grouped-MM prefill. **Reassociation-class numerics change** (GEMM accumulation and cross-expert combine only; the per-expert QDQ rows are reproduced exactly). |
| `PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK` | `64` (loop path) / `256` (fp8 chunked) | `moe.py:1474` / `:1687` | Expert-chunk size. Note the two paths carry different defaults. |
| `PRISMAQUANT_CB_PREFILL_OVERLAP` | off (`=1` enables) | `moe.py:1757` | FP8 prefill side-stream overlap. Measured NULL on 35B-A3B (17 ms/layer, both arms identical, 2026-07-26); stays opt-in until a positive exists at any scale. |
| `PRISMAQUANT_CB_PREFILL_TIMING` | unset | `moe.py:1717` | Per-stage prefill timing instrumentation (qdq/align/expand/gemm/act/combine). |
| `PRISMAQUANT_CB_GROUPED_TRIM` | `1` | `moe.py:773` | Trims `expert_ids` to the live block count — costs the one device sync. |
| `PRISMAQUANT_CB_EXPAND` | unset | `moe.py:1874` | `triton` restores the pre-optimization expand path (bisection lever; the old path cost ~26 ms/layer of pure memcpy on Laguna-256E prefill). |

### 7c. CUDA-kernel schedule switches (`csrc/cb_gemv.cu`)

All read host-side in the launcher, so they are CUDA-graph-capture-safe (no
device reads, no new syncs).

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_CB_FP8_SCHED` | double-buffer | `cb_gemv.cu:470` | `legacy` selects the single-buffer dense fp8 decode schedule. The default software-pipelined double buffer is **bit-identical** and +3-6%. |
| `PRISMAQUANT_CB_FP4V2_SCHED` | single-buffer | `cb_gemv.cu:759` | `db` opts the fp4-v2 dense kernel into double-buffering, which **regressed** (its two-tier decode is compute-bound). Kept as the switch that measured the loss. |
| `PRISMAQUANT_CB_DECODE_CONTRACT` | `v1` | `cb_gemv.cu:471` / `:760` / `:1357` / `:1437` | `v2` selects the per-weight decode contract v2. |
| `PRISMAQUANT_CB_W2_SCHED` | round-2 warp schedule | `cb_gemv.cu:1436` / `:1438` | fp4-v2 grouped down-projection (w2) schedule. Default drops the idle-warp 8-warp launch to 2 warps at `n_sb=6` (+50% on the Hy3 w2 shape); it **reassociates** the fp32 partial sum vs `legacy` and is served-KL-validated. `w13` (`n_sb=16`) stays 8-warp and bit-identical. `legacy` = the numerics-preserving 8/4-warp baseline. `rowpack` = round-3 experiment (further reassociated, **pending its own served check**). |
| `PRISMAQUANT_CB_W2_ROWS` | `8` | `cb_gemv.cu:1442` | `rowpack` rows-per-block, `RPB ∈ {4,8,16}`. |
| `PRISMAQUANT_CB_W2_WARPS` | `0` (derive) | `cb_gemv.cu:1486` | Overrides the derived warp count for the default w2 schedule. |

## 8. GGUF lane (`docs/lanes/gguf.md`)

| env var | default | what it does |
|---|---|---|
| `EXPORT_CONTAINER` | `compressed-tensors` | `gguf` switches stage 4 to skeleton-build + `export_gguf`; the pipeline requires `TARGET_PROFILE=gguf`, `COST_MODE=local`, `PRODUCTION_CACHE=0` (`run-pipeline.sh:97-107`) |
| `PRISMAQUANT_GGUF_IMATRIX` | `1` (`measure_quant_cost.py:1211`) | Activation-weighted (imatrix) k-quant scale selection in BOTH the batched cost path and the pipeline's export call — keep the two in lockstep or the A/B has a rendering confound |
| `LLAMA_CPP_DIR` | `/home/rob/dq-runs/llama.cpp` | Source of `convert_hf_to_gguf.py` for the skeleton |
| `GGUF_SKELETON` | `WORK_DIR/artifacts/skeleton.gguf` | bf16 skeleton path (built if missing) |
| `GGUF_TOKEN_EMBEDDING_FORMAT` / `GGUF_OUTPUT_FORMAT` | keep skeleton precision | Quantize `token_embd` / `output` (llama.cpp presets use Q2_K / Q6_K) |

## 9. Dead entries and non-flags

Previously listed here, now removed from the index. Kept as a record so they are
not re-added by a future scrape.

| token | verdict |
|---|---|
| `PRISMAQUANT_L2_CUDA_GRAPHS` | **DEAD** — never read. Sole occurrence is the comment at `perturbed_x_cache.py:1225` ("intentionally not applied here"). Cleanup pending. |
| `PRISMAQUANT_DO_NO_HARM_MIN_GAIN` | **DEAD** — no occurrence anywhere in the tree; the documented default `0.0` was fiction. The live analogue is `PRISMAQUANT_RENDER_GATE_MIN_GAIN` (a different gate). Cleanup pending. |
| `PRISMAQUANT_PATCH_SENTINEL` / `PRISMAQUANT_CHANNEL_SENTINEL` / `PRISMAQUANT_FULL_SENTINEL` | **Not env vars.** Python module-attribute name constants `_PRISMAQUANT_*_SENTINEL` (`sensitivity_probe.py:800-802`), used with `setattr`/`getattr`. |
| `PRISMAQUANT_GRAPH_POOL` | **Not an env var.** Module global `_PRISMAQUANT_GRAPH_POOL` (`kl_measurement.py:407`). |
| `PRISMAQUANT_COORD_LANE_BATCH` | **Set but never read** — written by `tools/smoke_graph_memory.py:72`, consumed by nothing. |

## 10. Live flag index

The complete live `PRISMAQUANT_*` vocabulary (157 flags), including low-level
debug, cache, validation, serving and archived-research switches. Anything not
on this list is not read by current code. `PQ_EXPORT_VECTOR_CHUNK` is the one
live flag outside the namespace.

```text
PRISMAQUANT_ACT_CACHE_ASYNC
PRISMAQUANT_ACT_CACHE_FP32
PRISMAQUANT_ACT_CACHE_WORKERS
PRISMAQUANT_ACT_CLIP_QUANTILE
PRISMAQUANT_ALLOW_KV_SHARED_FISHER
PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN
PRISMAQUANT_ALLOW_PYTORCH_FALLBACK
PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER
PRISMAQUANT_ALLOW_UNSCALED_FP8
PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE
PRISMAQUANT_BATCHED_NVFP4_EXPORT
PRISMAQUANT_BLOCK_OUTPUT_MATCH
PRISMAQUANT_CB_AUTOTUNE_MIN_M
PRISMAQUANT_CB_COL_WEIGHTS
PRISMAQUANT_CB_CUDA_M_MAX
PRISMAQUANT_CB_DECODE
PRISMAQUANT_CB_DECODE_CONTRACT
PRISMAQUANT_CB_DISPATCH
PRISMAQUANT_CB_ENCODE_COMPILE
PRISMAQUANT_CB_ENCODE_TIER
PRISMAQUANT_CB_EXPAND
PRISMAQUANT_CB_EXT_DIR
PRISMAQUANT_CB_FP4V2_SCHED
PRISMAQUANT_CB_FP8_SCHED
PRISMAQUANT_CB_FUSED_MIDM
PRISMAQUANT_CB_GROUPED_TRIM
PRISMAQUANT_CB_L2_AUTOTUNE
PRISMAQUANT_CB_L2_GROUP
PRISMAQUANT_CB_L2_MIN_M
PRISMAQUANT_CB_L2_OVERLAP
PRISMAQUANT_CB_L2_WINDOW_MB
PRISMAQUANT_CB_LADDER_INTERP
PRISMAQUANT_CB_LADDER_TOL
PRISMAQUANT_CB_PREFILL
PRISMAQUANT_CB_PREFILL_AUTO_FORCE
PRISMAQUANT_CB_PREFILL_DENSE
PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK
PRISMAQUANT_CB_PREFILL_GROUPED_MM
PRISMAQUANT_CB_PREFILL_OVERLAP
PRISMAQUANT_CB_PREFILL_TIMING
PRISMAQUANT_CB_W2_ROWS
PRISMAQUANT_CB_W2_SCHED
PRISMAQUANT_CB_W2_WARPS
PRISMAQUANT_COORD_LANE_CUDA_GRAPHS
PRISMAQUANT_COORD_LANE_CUDA_GRAPHS_MIN_CALLS
PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE
PRISMAQUANT_COORD_REPLAY_CACHE
PRISMAQUANT_COST_PREFETCH_ACT
PRISMAQUANT_COST_UCB_Z
PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH
PRISMAQUANT_DAMP_ANALYTICAL
PRISMAQUANT_DAMP_ANALYTICAL_C
PRISMAQUANT_DAMP_SWEEP_LOG
PRISMAQUANT_DEBUG_PREFIXES
PRISMAQUANT_DEFERRED_FISHER_COMPUTE
PRISMAQUANT_DEFERRED_FISHER_SYNC
PRISMAQUANT_DETERMINISTIC
PRISMAQUANT_DIRECT_CUDA_LOAD
PRISMAQUANT_DISABLE_RTN_COMPILE
PRISMAQUANT_DO_NO_HARM
PRISMAQUANT_DO_NO_HARM_VERBOSE
PRISMAQUANT_EMPTY_CACHE_EACH_REPLAY_BATCH
PRISMAQUANT_ENABLE_PTC
PRISMAQUANT_EXPERT_CALIB_BATCH
PRISMAQUANT_EXPERT_COST_SAMPLE
PRISMAQUANT_EXPERT_LAZY_FILL
PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ
PRISMAQUANT_EXPORT_REUSE_PRIOR
PRISMAQUANT_EXPORT_REUSE_VERIFY
PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT
PRISMAQUANT_FISHER_CAP_MULTIPLIER
PRISMAQUANT_FISHER_COL_WEIGHTS
PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP
PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR
PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP
PRISMAQUANT_FISHER_WEIGHTED_GPTQ
PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE
PRISMAQUANT_FP8_SCALE_SWEEP_FACTORS
PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES
PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_FRACTION
PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_GB
PRISMAQUANT_FULL_SEQUENCE_KL
PRISMAQUANT_FUSED_KERNEL_NVFP4
PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE
PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE
PRISMAQUANT_GGUF_IMATRIX
PRISMAQUANT_GPTQ_BLOCK_SIZE
PRISMAQUANT_GPTQ_DAMP
PRISMAQUANT_GPTQ_DAMP_ROLES
PRISMAQUANT_GPTQ_DAMP_SWEEP
PRISMAQUANT_GPTQ_STATIC_ACT_ORDER
PRISMAQUANT_GPU_MEM_RESERVE_FRACTION
PRISMAQUANT_GPU_MEM_RESERVE_GB
PRISMAQUANT_GRAPH_AUDIT
PRISMAQUANT_GRAPH_OUTPUT_CLONE
PRISMAQUANT_GRAPH_SHARED_POOL
PRISMAQUANT_HOST_MEM_RESERVE_FRACTION
PRISMAQUANT_HOST_MEM_RESERVE_GB
PRISMAQUANT_IQ_COMPILE_SWEEP
PRISMAQUANT_KL_CUDA_GRAPHS
PRISMAQUANT_KL_CUDA_GRAPHS_MIN_CALLS
PRISMAQUANT_KL_CUDA_GRAPHS_VERBOSE
PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE
PRISMAQUANT_KV_COTANGENT
PRISMAQUANT_L3_CUDA_GRAPHS
PRISMAQUANT_L3_CUDA_GRAPHS_MIN_CALLS
PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE
PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_FRACTION
PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB
PRISMAQUANT_L3_MIN_HOST_MEM_GB
PRISMAQUANT_L3_PREQUANT_CACHE
PRISMAQUANT_L3_PREQUANT_CACHE_PEAK_MULTIPLIER
PRISMAQUANT_L3_PREQUANT_CACHE_RESERVE_FRACTION
PRISMAQUANT_L3_PREQUANT_CACHE_RESERVE_GB
PRISMAQUANT_MASK_CUDA_DURING_META_INIT
PRISMAQUANT_MAX_GPU_MEM_GB
PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS
PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS
PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES
PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE
PRISMAQUANT_NVFP4_FUSED_JIT_WARMUP
PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE
PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID
PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_HI
PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_LO
PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS
PRISMAQUANT_NVFP4_JOINT_SCALE_OPT
PRISMAQUANT_NVFP4_SCALE_RULE
PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING
PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE
PRISMAQUANT_PREFILL_M_THRESHOLD
PRISMAQUANT_PRELOAD_FUSED
PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER
PRISMAQUANT_PROBE_CTX_CACHE
PRISMAQUANT_PROBE_DOMAIN
PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK
PRISMAQUANT_PROD_ACT_SCALES
PRISMAQUANT_PTC_VARIANT
PRISMAQUANT_RENDER_GATE_MIN_GAIN
PRISMAQUANT_RENDER_PROGRESSIVE_GATES
PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE
PRISMAQUANT_SKIP_PACKED_EXPERT_COST
PRISMAQUANT_SMOKE_DETERMINISM
PRISMAQUANT_SMOKE_MODEL
PRISMAQUANT_SMOKE_SAMPLES
PRISMAQUANT_SMOKE_SEED
PRISMAQUANT_SMOKE_SEQLEN
PRISMAQUANT_SOLVER_TRACE
PRISMAQUANT_STRICT_ASSIGNMENT_COVERAGE
PRISMAQUANT_STRICT_PRODUCTION_CACHE
PRISMAQUANT_TARGET_PROFILE
PRISMAQUANT_TMPDIR
PRISMAQUANT_UMA_MEMORY_INFO
PRISMAQUANT_VALIDATION_CUDA_GRAPHS
PRISMAQUANT_VALIDATION_CUDA_GRAPH_CACHE_SIZE
PRISMAQUANT_VALIDATION_FAKE_METRICS
PRISMAQUANT_VALIDATION_PROD_CACHE
PRISMAQUANT_VALIDATION_PROD_CACHE_DIR
PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB
PRISMAQUANT_VALIDATION_SKIP_END_KL
PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE
```

## 11. Disabling for debugging

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

Two traps when doing this:

- `PRISMAQUANT_GPTQ_DAMP_SWEEP` used to have **different defaults in the
  exporter and in `kl_sensitivity_probe`**; D5 closed 2026-07-30 and the probe
  is walled (§2). There is now one reader and one default (`0`), but pinning it
  explicitly in an A/B is still the cheap habit.
- Extension residency shifts allocator addresses and moves confident-KL by up
  to ±17% between arms. Set `PRISMAQUANT_PRELOAD_FUSED=1` on both arms of any
  served comparison where one arm would otherwise not load the fused ext.
