# PrismaQuant / PrismaSCOUT Handover — 2026-05-02 morning

## TL;DR

The PrismaSCOUT L3-redesign work landed end-to-end on `feat/propagated-cost` (which IS `origin/main` since the merge today). Algorithmic part is **proven** — v5 produced **34% KL improvement over L2 baseline at 4.5 bpp on Qwen 4B** (KL 0.371 → 0.245). The performance optimization stack (fused NVFP4 Triton kernel + CUDA graphs across 5 paths + replay cache + memory mgmt) is **landed but not validated together**. Every attempt to run the kernel + graphs + replay combo OOM-killed the box on GB10's 121 GB UMA.

**Recommended immediate move**: re-run with v5's proven config (no kernel, default eager paths) to produce the rest of the Pareto curve and ship a re-exported Qwen 4B / 27B artifact.

## State at handover

- Branch: `feat/propagated-cost` ≡ `origin/main`
- Worktree: `/home/rob/prismaquant-propagated-cost`
- Main repo (user's WIP on different branch): `/home/rob/prismaquant`
- Last clean commit: `7f7521a` (Spec O — coord-lane CUDA graph + replay-cache KV bug fixes)
- All containers stopped; system recovered (2 GB used / 119 GB free)
- Memory watchdog Monitor armed; v5 monitor stale
- One pending Monitor (`bx1k9q9b3`) for memory; no other in-flight work

## The headline result (v5 anchor 4.50)

```
L2 baseline KL = 0.3711
L3 polish KL  = 0.4609 (rejected, regressed 24% — known non-additivity bug)
Coord descent → KL = 0.2453 (34% better than L2, 6 flips committed of 101 tried)
```

This is the empirical proof PrismaSCOUT beats its predecessor:
- 47% KL reduction at 6.0 bpp on Qwen 0.6B (earlier session run)
- 34% KL reduction at 4.5 bpp on Qwen 4B (v5 today)

Saved at `/home/rob/dq-runs/qwen3-4b-perturbed-x/iterate_v5_out/anchor_bpp_4.50/`:
- `final_assignment_bpp_4.50.json` — the layer config to export
- `final_layer_config_bpp_4.50.json` — same in exporter format
- `l3_propagated_costs.pkl` — reusable via `--resume-l3-costs-dir` (saves 22 min/anchor)
- `costs_iter_03.pkl` — final L2 costs

## What works (validated empirically)

| Component | Status | Evidence |
|---|---|---|
| L2 perturbed-X iteration | ✓ Solid | Converges in 3 iters consistently |
| L3 measurement (global mode) | ✓ Solid | 22 min/anchor on Qwen 4B |
| L3 polish (DP at budget) | ⚠ Regresses | Non-additivity bug; rollback fires correctly |
| Iterated L3 trust-region | ✓ Solid | Hamming cap + EMA + cycle hash |
| Coord descent fallback | ✓ Solid | Provably non-regressive on real KL |
| Lane-batched coord descent | ✓ Solid | 30-60× over sequential |
| L3 resume from saved costs | ✓ Solid | `--resume-l3-costs-dir`, validated v9-v13 |
| KL graph capture (Spec M fix) | ✓ Solid | 67/67 + 6 skipped tests pass |
| Memory budget enforcement | ✓ Solid | Catches OOM cleanly when it has time |
| Validation harness | ✓ Solid | Spec B; PPL on WikiText + MMLU |
| Artifact registry | ✓ Solid | Spec B; JSON-backed gate for releases |

## What's NOT validated (overnight blockers)

| Component | Status | Failure mode |
|---|---|---|
| Fused NVFP4 Triton kernel | ⚠ Loaded, never finished a run | OOM at coord descent start (kernel + caches stack to >121 GB) |
| Replay cache (G2 wired by H) | ✗ Buggy | KV cache batch-dim mismatch; **Spec O claims fix at `7f7521a`** but not tested live yet |
| Coord-lane CUDA graphs | ✗ Buggy | Same KV bug; **Spec O claims fix** but not tested live yet |
| Per-batch coord logging (Spec I) | ⊘ Untested live | Should fire under v6+ but every run died before it could emit |

**Critical unknown**: Spec O's fix MAY actually solve both the replay cache and coord-lane graph bugs. We never got to test it live because v13 OOM'd before coord descent started for unrelated reasons (the kernel JIT + cache fill stack).

## The OOM landscape on GB10

GB10 has 121 GB unified memory. GPU and system share it. Confirmed memory peaks:

| Component | Approx peak |
|---|---|
| Model weights (Qwen 4B BF16) | ~16 GB |
| L2 activation caches (3 iters × 8 GB each) | up to 24 GB |
| Frozen-weight cache (Spec D, capped 400 entries) | ~4 GB |
| LayerHiddenStateCache (G2) | ~200 MB |
| Lane-batched activations (32 lanes) | ~3-5 GB |
| 4-path CUDA graph capture pools | unbounded, observed ~10-20 GB total |
| Triton kernel JIT compilation transient | ~1-2 GB spike |
| CUDA stream + allocator overhead | ~2-3 GB |
| Python + transformers baseline | ~3 GB |

**Single-component runs fit fine.** All-on stacks to 99-121 GB peak, depending on exact phase. v10 hit budget enforcement at 99.85 GB; v13 OOM'd silently before mem_get_info could intervene.

## Code state (commits today, in landing order)

```
7f7521a Spec O — coord-lane graph + replay KV bug fixes (LATEST)
8b502ec Spec N — bounded LRU caches + budget backpressure + JIT warmup
71f771f Spec M — KL graph capture root-cause fix
3433497 Spec L — pin CPU tensors + --resume-l3-costs-dir flag
bc6d81f kernel num_stages 3→2 (Blackwell shared mem fit)
d5fc0ef Spec J — fused NVFP4 Triton kernel (proof-of-concept)
bb9ca58 Spec K — CUDA graphs across coord lane / KL eval / L2 / validation
01d9f4f Spec I — per-batch coord descent logging
5142581 Spec H — wire LayerHiddenStateCache into coord descent
2a35875 G1 — lane-batched coord descent (30-60× speedup)
093b986 G2 — LayerHiddenStateCache infrastructure
c910926 Spec F — actually wire frozen-weight cache into coord-descent eval
87790db Fork 1 — cache-aware coord descent + L3 ranking + early stop + CRN eval
768c750 Spec D — pre-quant cache dispatch fix + frozen-weight cache for L2 context
4ebfd0f Spec C — pre-quant cache + CUDA graphs + bigger lane batches
b3d8445 Spec B — validation harness + artifact registry
98715f4 Spec A — trust-region iterated L3 + coord-descent fallback
a8b4de6 logging — L2 KL alongside L3 KL in multi-budget emit
e78e7bf logging — L3 depth-group progress
+ all the morning's PrismaSCOUT rounds 1-19 (foundational)
```

Total LOC added today: ~5000+ across the redesign.

## Recommended immediate path (next 1-2 hours)

**Option A: Ship the proven algorithm (lowest risk, fastest result)**

1. Launch v14 in v5 config (kernel OFF, default eager paths, multi-budget Pareto):
   ```bash
   docker run ... -e PRISMAQUANT_FUSED_KERNEL_NVFP4=0 \
                  -e PRISMAQUANT_COORD_LANE_CUDA_GRAPHS=0 \
                  -e PRISMAQUANT_KL_CUDA_GRAPHS=0 \
                  -e PRISMAQUANT_COORD_REPLAY_CACHE=0 \
                  -e PRISMAQUANT_MAX_GPU_MEM_GB=110 \
                  ... --target-bits-list 4.5,5.0,5.5,6.0,6.5,7.0 ...
   ```
   - 6 anchors × ~75 min = ~7.5 h on Qwen 4B
   - Reuse v5's anchor 4.50 L3 costs via `--resume-l3-costs-dir`
   - Should produce full Pareto curve

2. Run validation harness on each layer_config:
   ```bash
   python -m prismaquant.validation_harness validate \
     --model /hfcache/Qwen3-4B \
     --layer-config /work/iterate_v14_out/final_layer_config_bpp_X.XX.json \
     --register --notes "v14 Pareto sweep, no kernel/graphs"
   ```

3. Compare PPL/MMLU vs the existing shipped artifact. If candidates beat shipped, re-export.

**Option B: Validate Spec O's fixes work (medium risk, learns about kernel viability)**

1. Launch v14b: kernel OFF but coord-lane graphs ON + replay ON (test Spec O without kernel JIT spike). Single anchor 4.50.
   - If it works without OOM → Spec O fix is real, kernel is the SOLE remaining blocker
   - If it OOMs → Spec O fix isn't enough; need more work on graph memory pool

2. If A passes, try v14c: kernel ON + coord-lane graphs OFF + replay OFF (kernel alone). Single anchor.
   - Tests if kernel can fit at all
   - If it fits, the issue is the SUM of graphs + replay + kernel, not any single component

**Option C: Aggressive — reduce per-component memory and try all-on again**

1. Drop max_lanes_per_batch from 64 → 16 (1/4 the activation memory in coord descent)
2. Drop frozen_weight_cache_max_entries from 400 → 100
3. Drop CUDA graph cache from 4 entries to 2
4. Re-launch v14d with everything ON

This should fit but loses some throughput. Worth trying once.

## Longer-term unblocking

The kernel + graphs + replay combo at Qwen 4B scale needs ~30 GB more headroom than GB10 has. Options:
1. **Use a bigger host** (the user has Spark with 80 GB but it's smaller; mention if a 256 GB+ host is accessible)
2. **Stream weights from disk** — only keep currently-needed Linears in memory (slow but bounded)
3. **Split L2/L3/coord-descent into separate processes** with explicit memory handoff
4. **Disable graphs entirely** — just lean on the kernel (graphs save 2-3× per launch but consume the most memory)

For 27B specifically: even without the kernel, at current speeds the Pareto sweep is ~7-8 hours WITH speed kit, ~50 hours WITHOUT. Need to pick a single anchor (probe-based kneedle prediction) for 27B, not full Pareto.

## Files to know

```
/home/rob/prismaquant-propagated-cost/                — main worktree (origin/main = HEAD)
  prismaquant/
    iterate_perturbed_allocation.py     (3155+ LOC)   — top-level driver, multi-budget
    propagated_cost.py                  (~2200 LOC)   — L3 measurement, lane batching
    perturbed_x_cache.py                (~700 LOC)    — L2 perturbed-X cache + frozen weight cache
    layer_state_cache.py                (505 LOC)     — G2's replay infrastructure
    memory_management.py                (174 LOC)     — Spec N bounded caches + budget
    validation_harness.py               (812 LOC)     — Spec B PPL/MMLU eval
    artifact_registry.py                (259 LOC)     — Spec B JSON registry
    kernels/nvfp4_fused.py              (305 LOC)     — Spec J Triton kernel
    format_registry.py                  (~300 LOC)    — format definitions, RTN codebooks

/home/rob/dq-runs/qwen3-4b-perturbed-x/                — Qwen 4B run dir
  artifacts/probe.pkl                   (76 KB)       — sensitivity probe
  artifacts/cost.pkl                    (?)           — initial L2 cost table
  iterate_v5_out/anchor_bpp_4.50/                     — THE PROVEN RESULT
    l3_propagated_costs.pkl                            — reusable via --resume-l3-costs-dir
    final_assignment_bpp_4.50.json                     — layer_config to export
    final_layer_config_bpp_4.50.json
    iteration_trace.jsonl                              — L2 iter trace

/home/rob/prismaquant/.claude/
  prismaquant-handover-2026-04-29.md                  — earlier handover (HALO landing, etc)
  prismaquant-handover-2026-05-01.md                  — yesterday's handover (allocator Δloss)
  prismaquant-handover-2026-05-02.md                  — THIS FILE
```

Memory entries (auto-loaded into context):
```
~/.claude/projects/-home-rob-prismaquant/memory/
  session_2026_05_02_overnight.md                     — overnight session note
  prismascout_l3_non_additivity.md                    — the L3 polish bug we built around
  prismaquant_traction_context.md                     — 60k downloads context, paper ambition
  feedback_cuda_graphs_everywhere.md                  — standing instruction
  feedback_autonomy_authorized_paths.md               — don't pause once authorized
  ... (older session memories)
```

## Open issues for follow-up

1. **Spec O's fix needs live validation** — should fire on v14b above. If it doesn't OOM, we know the KV bug is fixed.
2. **Kernel memory diet** — Triton compilation could reduce shared memory further; or use `torch.compile` instead of raw Triton.
3. **CUDA graph allocator pool** — currently each path has its own pool; consolidating to one shared pool would cut memory ~4×.
4. **Resume metadata schema** — old pickles (pre-Spec L) require `--no-resume-on-mismatch`. Could add a one-time upgrader to write the metadata into existing pickles.
5. **27B run plan** — needs the probe-based kneedle prediction approach (single anchor, not full Pareto) given compute budget realities. Probe data isn't yet generated for 27B.
6. **Pending tasks** in tracker:
   - #25 Export both layer_configs via export_native_compressed
   - #26 PPL eval: with-MXFP4 vs no-MXFP4 on wikitext

## Wrapper-bug pattern to know

Every codex agent dispatched today used a wrapper that exits prematurely while codex itself is still running. Pattern: the wrapper sees codex's first output and reports "completed" before codex finishes. **Always verify by checking the actual git state and the codex log file directly** — don't trust the wrapper's "complete" notification at face value. If the wrapper reports done but `git log origin/main..HEAD` is non-empty (or staged work exists), commit + push manually.

## What I would do first if I were waking up to this

1. Read this file + read `MEMORY.md` index for context
2. `cd /home/rob/prismaquant-propagated-cost && git log --oneline -8 origin/main` to see commits
3. `docker ps -a --filter name=pq-qwen` to see prior runs
4. Decide: ship Option A (proven config) or test Option B (validate Spec O)
5. If Option A: launch v14, set Monitor for events + memory watchdog, sleep 2-hour intervals
6. While Pareto runs: prep validation harness invocation script, build comparison table generator
7. When v14 lands: run validation harness, compare to shipped 4B artifact, decide on re-export
