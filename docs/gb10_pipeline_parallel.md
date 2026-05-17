# GB10 Pipeline-Parallel Probe — design & continuation note

Status as of 2026-05-17:

- **Part 1 — LANDED** on the `gb10` branch (this commit). Phase-3
  reverse-sweep shard→shard gradient carry. Single-box, no NCCL,
  math-identical, verified.
- **Part 2 — NOT STARTED.** Multi-box (2+ GB10) pipeline parallelism
  over NCCL/RoCE. Design below; full approved plan at
  `~/.claude/plans/cheerful-rolling-marble.md`.

This file is the continuation note: read it (and the plan) before
resuming Part 2.

## Why this exists

PrismaQuant's probe streams decoder layers one at a time because large
models exceed a single GB10's ~120 GB UMA. When a model doesn't fit
resident every layer is a cold checkpoint read (~40 s/layer measured on
MiniMax-M2.7) — phase-1 forward and phase-3 reverse sweep go disk-bound,
violating the GPU-first rule in `AGENTS.md`. Pipeline parallelism across
2+ boxes lets each box hold a contiguous **slice** of decoder layers
resident, so the model fits in aggregate memory and cold reads vanish.

## Part 1 — what landed (single-box, no NCCL)

**Problem it fixes:** phase-3 ran once *per shard*, each shard
re-sweeping `num_layers-1 → sweep_floor` from `grad_at_tail`. At
`layers_per_shard=1` that is ~N× redundant layer fwd/bwd.

**Change:** body shards now process in **descending layer order**,
carrying each shard's bottom-edge `grad_out` into the next-lower shard
as its `seed_grad`. Each shard sweeps only its own layers. The reverse
sweep is the chain rule — splitting it at a layer boundary and resuming
with the saved intermediate gradient is mathematically identical to the
full sweep.

**Files / symbols (`prismaquant/incremental_probe.py`):**
- `_run_body_streaming_shard` — new params `seed_grad`, `seed_ceil`;
  returns `(bottom_edge_grad, sweep_floor)`. The sweep upper bound is
  `seed_ceil` (carry) or `num_layers-1` (full-tail fallback). A
  `use_seed` guard re-validates `seed_ceil == max(in_scope_layers)` and
  falls back to a full-tail sweep on any mismatch — correctness is never
  at risk for any model.
- `main()` shard loop — iterates `exec_order` = body shards descending,
  then non-body ascending. Closures `_seed_for` / `_record_boundary` /
  `_load_boundary` thread the carry and persist boundaries to
  `work/grad_boundary_{idx:03d}.pt` (`{"grad", "top"}`). Reused /
  synthesized shards reload the boundary from disk; if absent the next
  shard falls back to a full-tail sweep (self-healing).
- Builds on the earlier `sweep_floor = min(in_scope_layers)` truncation.

Also in this commit, `prismaquant/streaming_model.py`: the prefetch
memory floor was decoupled from the LayerCache headroom reserve (new
`prefetch_floor_bytes()` vs `memory_pressure_floor_bytes()`;
`_auto_prefetch_min_available_bytes` is now worker-count aware). This
unblocks phase-1/phase-3 prefetch on large streamed models where the
cache's phase-3-sized reserve previously disabled it.

**Verified:** `tests/test_incremental_probe.py::TestPhase3GradCarry`
runs a real probe on a tiny 6-layer Llama at `lps=1` (full carry),
`lps=3` (mixed) and `lps=6` (one shard, no carry = baseline). Merged
`probe.pkl` Fisher stats are **bit-identical** across all three
(0.00e+00 relative diff). Identical probe → identical allocator →
identical loss/bpp Pareto.

## Part 2 — remaining work (multi-box, NCCL/RoCE)

Scope confirmed with user: probe streaming loop only (phase-1 forward,
phase-2 head, phase-3 sweep). Cost / allocator / production cache /
export stay single-box. Each box reads its layer slice from its own
local checkpoint copy. Single-box behavior stays the default
(`--pp-ranks 1`).

The shard→shard `grad_out` carry from Part 1 is the seam: across a
**rank boundary** the in-process carry becomes a NCCL point-to-point
send. Likewise phase-1's `hidden` is sent forward rank→rank.

### New module: `prismaquant/pp_runtime.py`
- `pp_init()` / `PPGroup` — `init_process_group` (`nccl` over RoCE, or
  `gloo` for single-box testing via `PRISMAQUANT_PP_BACKEND`); reads
  torchrun env (`RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`) or
  `--pp-*` args.
- `split_layers_by_bytes(per_layer_bytes, world_size)` — contiguous
  layer ranges balanced by **weight bytes** (reuse the per-layer sizes
  from `_estimate_layer_cache_bytes`, `streaming_model.py:201`).
- `send_boundary` / `recv_boundary` — `[B,T,H]` point-to-point with a
  shape/dtype handshake.
- `gather_pickle_bytes(local_paths, dst=0)` — `gather_object` of shard
  pickle bytes to rank 0 for the final merge (work-dirs are local).

### `prismaquant/streaming_model.py`
- `_build_streaming_context(..., layer_range=(lo,hi))` — build
  resolvers / subset `weight_shard`+`weight_ckpt` / install only layers
  `[lo,hi)`; rest stay on meta.
- `install_resolvers` list → dict keyed by absolute layer index (the
  only 0-based assumption; `install_resolvers[L]` at ~`:474`). Boundary
  checks in `schedule_prefetch` (~`:427`) become `lo ≤ L < hi`.
  `LayerCache` is already layer-index-agnostic.

### `prismaquant/incremental_probe.py`
- argparse (after `--layers-per-shard`): `--pp-ranks` (default 1),
  `--pp-rank`, `--pp-master-addr`, `--pp-master-port`. `--pp-ranks 1`
  skips all PP code.
- Phase-1 (`_compute_global_precompute`) PP-chained: rank 0 embeds +
  forwards its layers → `send_boundary(hidden)` → rank r recv/forward/
  send. Position embeddings computed independently per rank (shape-only
  dependency — `_compute_position_embeddings`, `layer_streaming.py`).
  Each rank caches `activations_cpu` for its layers only.
- Phase-2 (final norm + lm_head + CE) on the **last rank only** —
  produces `grad_at_tail` + resident Fisher. No broadcast needed.
- Per-rank precompute cache `precomputed.rank{r}.pt`; fingerprint
  (`_compute_precompute_key`) gains `pp_world`, `pp_rank`,
  `layer_range`.
- Phase-3: each rank owns the shards in its layer range and runs
  Part 1's descending carry. The carry across a rank boundary is a
  NCCL send (last rank seeds from `grad_at_tail`, sends `grad_out`
  down; rank r recvs, sweeps, sends down). v1 ranks **serialize** — the
  win is residency, not throughput; micro-batched overlap is future
  work.
- mtp / lm_head / visual shards → last rank.
- Merge: `gather_pickle_bytes` → rank 0 → existing
  `merge_probe_pickles` (already order-independent, disjoint-stats).

### `test-pipeline.sh`
Optional `PP_RANKS` env: `>1` launches the probe via `torchrun`.

### Exception note (AGENTS.md / design_guidelines.md)
Multi-node is a new mechanism. Rule bypassed: "single-box streaming."
Justification: no single GB10 holds a >120 GB model resident, so PP is
the only way to keep the hot path GPU-bound (it *restores* the
GPU-first principle by eliminating cold reads). Isolated behind
`--pp-ranks>1`. Promotion gated on the 2-rank equivalence test +
a resident-residency / wall-time measurement on the real 2-box system.

## Key decisions & gotchas for whoever resumes Part 2

- Body shards from `build_shard_schedule` are contiguous disjoint layer
  ranges → descending `shard_idx` == descending layer order, and
  consecutive shards are adjacent. The carry depends on this.
- The carry is self-validating (`seed_ceil` vs actual
  `max(in_scope_layers)`); a schedule/scope mismatch degrades to a
  correct full-tail sweep, never to wrong math.
- `merge_probe_pickles` errors if a Linear appears in two shards — PP
  layer ranges are disjoint so this holds; keep it that way.
- `grad_at_tail` does NOT need broadcasting — it seeds only the last
  rank; lower ranks get their seed via the rank-boundary NCCL carry.
- Resident Fisher dicts (lm_head etc.) live on the last rank only;
  the lm_head shard is owned there — no broadcast.
- NCCL code is **unverifiable in the dev environment** (single box, no
  fabric). Use the `gloo` backend + 2 processes on one box for a logic
  equivalence test (`test_probe_two_rank_equivalence`); real RoCE is the
  user's deployment test.

## Verification plan for Part 2

- Single-box gloo: 2 processes, `--pp-ranks 2`, each owns half a tiny
  model; merged `probe.pkl` must match a single-rank run within fp
  tolerance.
- Real 2-box (user): `torchrun --nnodes=2` over RoCE on a model
  exceeding one box. Pass: each rank's LayerCache shows high hit-rate /
  no cold reads, merged `probe.pkl` drives an unchanged allocator
  result vs a single-box reference, phase-1/3 wall-time GPU-bound.
