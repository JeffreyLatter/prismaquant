# v20: memory policy + scheduling instrumentation

Status: design, pre-implementation.
Target: replaces v19 multipass run on MiniMax-M2.7 / 121 GB Spark / 1.8 TB disk.

## Why v20 exists

v19 is running but leaves visible hardware on the floor. Symptom from the
chunk-0 phase-3 log:

```
LayerCache: 0 layers, 0.0 GB / 75.4 GB, residency=empty
            hits=434 misses=398 hit_rate=52%
phase-3 reverse sweep: 389.1s  load=105.0s  bwd=208.4s
```

Three things wrong with that line:

1. Cache is at 0 of 75 GB. We have a budget; we're not using it.
2. Hit rate is 52% but residency is empty — every "hit" is a layer that
   was prefetched-and-immediately-discarded. The cache is acting as a
   one-shot prefetch buffer, not a cache.
3. 105 s / 389 s = 27% of every shard is GPU-idle waiting for disk.

Meanwhile host memory shows 77 GB free and disk shows 206 GB free. We are
budget-bound by policy, not by hardware.

## What v20 changes

Two themes: (A) make the cache value-aware and deterministic, (B) make
the pipeline measurable so the next round of optimization is data-driven
not intuition-driven.

### A. Cache policy

Three rules, applied in order. The first rule that fires wins.

1. **Deterministic mark-done eviction.** When the scheduler knows for
   certain a layer will not be touched again, evict it. Sources of
   certainty:
   - End of phase-1: any layer not in any phase-3 shard's scope.
   - End of shard S in phase-3: layers that appear in S but no later
     shard. With contiguous shards (lps=4) that's *all* of S's
     in-scope layers, every shard exit. (Reverse-sweep traversal still
     touches all 62 layers per shard, but stat accumulation is only for
     in-scope; the layers outside any remaining shard are pure
     read-once after their owning shard finishes.)
   - End of chunk: everything per-chunk.

2. **Pressure-triggered shrink.** If `MemAvailable < threshold`, evict
   the lowest-value retained entry until back above threshold. Already
   exists in `LayerCache._maybe_pressure_shrink`; v20 wires the
   priority signal so it knows what's low-value.

3. **Value-aware retention on install.** Today `install()` discards on
   exit. v20 retains the layer if `cache_used + layer_bytes <
   dynamic_budget`. Dynamic budget = `min(static_max_bytes,
   MemAvailable - reserve)`, recomputed per call.

Combined, the cache fills opportunistically when memory is plentiful,
shrinks when memory tightens, and dumps known-dead entries the moment
they're known dead.

### B. Predeclared shard schedule

Build the full `[(shard_idx, layer_set)]` mapping at startup. mark_done
events fall out of the schedule by static computation rather than
runtime detection. Makes #A trivial to test and removes a class of
"forgot to evict" bugs.

### C. Instrumentation (cheap-Gantt)

Per op, log `(start_wall, end_wall, resource, op_name, layer_idx)` to
a JSONL alongside the probe.log. Resource ∈ {gpu, cpu, disk_read,
disk_write}. After a chunk completes, emit a summary:

```
chunk 0:  GPU busy 64%  CPU busy 11%  disk_read 28%  disk_write 4%
          all-idle 7%  chunk wall 4980s
          biggest gap: gpu idle while disk_read on layer 47 (34s)
```

This is the data we need to argue for or against the full static
scheduler in v22. ~100 LoC; hot path overhead is one wall-clock read
per op.

## Smaller items folded in

- **Pinned-memory staging buffer** for layer load. Cuts the pageable
  copy on the disk → CPU → GPU path. ~30 LoC.
- **Async activation save** during phase-1. Background thread + bounded
  queue so next layer's forward overlaps with previous layer's write.
  ~50 LoC.
- **BF16 h_trace / h_w2_sum CPU accumulators.** Halves the per-shard
  CPU memory floor (~52 GB → ~26 GB at lps=4). Lets us push lps back
  to 8 or grow cache budget. Use chunk-wise sum to bound rounding
  error. ~20 LoC.
- **Skip-empty-scope short-circuit.** Assertion that we don't load a
  layer for a shard that has no in-scope params in it. ~10 LoC.

## Larger items still in v20

- **VmHWM peak per phase** — `/proc/self/status` read at phase
  boundaries; printed alongside the cache summary line.
- **gradient_checkpointing audit** — assert it's off during phase-1
  on every layer; loud failure if any layer enables it.
- **Per-channel Fisher summaries** (`h_per_in_channel`,
  `h_per_out_channel`) — replaces the killed h_detail. Per-Linear
  cost: `~(in + out) * 4 bytes` instead of `in * out * 4 bytes`. For
  a 3072 × 1408 expert weight that's ~18 KB instead of ~17 MB. Total
  for the model: ~1 GB instead of 940 GB. Suitable for direct write
  to disk per chunk + sum at merge time. The cost step's GPTQ Hessian
  surrogate gets per-channel scale; cleaner than the scalar fallback.

## Implementation order (dependency-respecting)

1. Predeclared shard schedule (B). Pure refactor, no behavior change.
2. mark_done API on LayerCache + scheduler-driven calls (A.1). Kills
   `residency=empty` for the layers we know are dead.
3. Value-aware install retention (A.3). Fills the cache.
4. Dynamic budget (folds into A.3). Lets the cache breathe.
5. Pressure shrink wiring to priority signal (A.2). Was already
   built; just connect the new priority data.
6. Gantt instrumentation (C). Data for v22 decisions.
7. BF16 accumulators. Independent; lands when convenient.
8. Pinned staging + async activation save. Independent; land
   together.
9. Per-channel Fisher summaries. Independent; lands as its own change
   because it touches the cost-step contract.

Steps 1–5 are the critical path. The expected outcome from steps 1–5
alone, based on the v19 numbers: phase-3 sweep load time drops from
~100 s to roughly the cost of the misses-not-cacheable (out-of-budget
layers), which on a budget that fits 17 of 62 layers is bounded by
`~(62 - 17)/62 * 100 s ≈ 73 s` worst case and likely much less because
mark_done removes layers from the candidate pool in a useful order.

## What v20 explicitly does NOT do

- **Hierarchical CPU/GPU cache (#9).** Skipped — UMA Spark has CPU and
  GPU sharing the same physical RAM, so the tier doesn't exist.
- **Full static scheduler.** That's v22+. v20 collects the data to
  decide whether it's worth building.
- **g²_per_token reinstate.** Stays disabled; ~38 GB total is fine on
  disk but the merge logic isn't written. File for v21 or later.
- **Touch the cost-step contract for the existing run.** Per-channel
  Fisher (#9) lands but defaults to off so v19's cost step still works
  on its probe.pkl shape.
