---
Followup to:
  - .claude/prismaquant-handover-2026-05-02.md      (overnight autonomous)
  - .claude/prismaquant-handover-2026-05-02-pm.md  (mid-day pickup)
This file: end-of-day state for 2026-05-02.
---

# PrismaQuant / PrismaSCOUT Handover — 2026-05-02 evening

## TL;DR

Big day. CUDA graph memory work landed end-to-end and is on `origin/main`
(now `a24342a`). Vendored Qwen3 with precomputed RoPE fixes a
determinism-mode NaN. Draft PR upstreamed to transformers
(huggingface/transformers#45748). Profiler tooling pushed on a side branch.
Real PrismaSCOUT run at Qwen 4B / 4.75 bpp was killed mid-L3: L3 is
running 5× slower than the morning estimate (~190 s/layer × 37 layers
≈ 117 min). Codex investigation into a `_initialize_weights` no-op bug
left uncommitted edits in the worktree.

## What landed today (commits ahead of `7f7521a` baseline)

```
a24342a qwen3: vendor RoPE precompute determinism fix
d9d67a9 graphs: re-enable shared pool default ON; smoke det flag at import time
e1c6ee9 graphs: workspace config at smoke top + opt-in determinism
2ba832f Merge fix/cuda-graph-shared-pool: shared pool + audit + safety overrides
d830a01 graphs: default shared pool OFF after smoke NaN risk           ← later reverted
20ce781 graphs: drop use_deterministic_algorithms from smoke RNG pin
76354ea graphs: pin smoke RNG for pool comparison
5acb941 graphs: safety override no-clone after smoke NaN
3f320c7 graphs: share pool and audit graph memory
```

All on `origin/main` and `origin/feat/propagated-cost` at `a24342a`.

Profiler branch separately:
```
8104e99 tools: detect callable graph pool ids
f9d14b4 tools: add cuda graph memory profiler
```
Pushed at `origin/tool/cuda-graph-memory-profiler` — unmerged, complementary.

## What ships by default now

- ✅ **F1 shared CUDA graph memory pool** — `PRISMAQUANT_GRAPH_SHARED_POOL=1` is default. Proven bit-identical to private via 12-run matrix on Qwen 0.6B and 3-run matrix on Qwen 4B with det mode on.
- ✅ **F3 slice-and-clone in 4 captured forwards** — kl/end-kl/coord-replay/coord-full all `.clone()` the small slice inside the captured fn. Pool footprint stays tiny.
- ✅ **F4 graph memory audit** — `PRISMAQUANT_GRAPH_AUDIT=1` to enable. Wired at ~10 phase boundaries.
- ✅ **F2 output-clone safety override** — when shared pool is on, `PRISMAQUANT_GRAPH_OUTPUT_CLONE=0` is silently overridden to clone (one-time stderr warning).
- ✅ **Vendored Qwen3** with precomputed RoPE cos/sin cache. AutoModel-registered via `prismaquant/__init__.py` → `prismaquant.vendored.register_qwen3()`. Math bit-identical to upstream (`assert_close(rtol=0, atol=0)`).
- ✅ **Smoke harness hardened** — `CUBLAS_WORKSPACE_CONFIG=:4096:8` set at module top before any prismaquant import; `torch.use_deterministic_algorithms` opt-in via `PRISMAQUANT_SMOKE_DETERMINISM=1` at module top (matters because the call must precede CUDA init).

Test counts on `feat/propagated-cost`: **407 passed, 11 skipped, 0 failures** in `tests/`.

## Major findings (chronological, with the corrections)

1. **Shared pool divergence at smoke scale** — initially looked like a 5%-NaN-rate, paired-mismatch shared-pool bug. Wrong. Within-process back-to-back `measure_assignment_kl` produces bit-identical KL across shared/private (3.339811e-03 every time). Across separate processes, smoke is non-deterministic for unrelated reasons. The shared pool changes nothing.

2. **Determinism flag NaN root cause** — codex found it via the PyTorch issue investigation:
   - `torch.use_deterministic_algorithms(True)` enables `torch.utils.deterministic.fill_uninitialized_memory` (NaN-fills uninitialized tensors as a debug aid).
   - `prismaquant/__init__.py:49` no-ops `transformers.modeling_utils.PreTrainedModel._initialize_weights` to skip wasted random init.
   - Some HF model classes (Qwen3RotaryEmbedding) rely on `_initialize_weights` to populate non-parameter buffers like `inv_freq`. With our no-op, those buffers stay uninitialized → NaN under det mode.
   - The vendored Qwen3 sidesteps this for Qwen3 only (precomputes inv_freq + cos/sin in `__init__` directly). Other model families (Llama, Mistral, Qwen2.5, DeepSeek) still hit it under det mode.

3. **Performance regression in L3 propagated cost** — the 4.75 bpp run was killed after the first 2 of 37 depth groups. Layer 0: 180.7 s. Layer 1: 194.6 s. Projected L3 total ≈ 117 min (5× the morning handover's claimed 22 min/anchor). Suspected cause: graph cache thrashing — each layer has 7 Linears × 3 formats = 21 distinct shapes, and `PRISMAQUANT_*_CUDA_GRAPH_CACHE_SIZE` defaults to 4. Cache evicts/recaptures repeatedly. Not verified; needs a controlled experiment.

## Upstream artifacts

| | URL | Status |
|---|---|---|
| Transformers PR (Qwen3 RoPE precompute) | https://github.com/huggingface/transformers/pull/45748 | Draft. Framed as perf optimization. |
| PyTorch issue (cuBLAS det-mode NaN) | not filed | Codex correctly determined the bug is local, not PyTorch |
| Repro script for the (originally) suspected PyTorch bug | `/tmp/torch_det_nan_repro.py` | Reduces NaN to a `_initialize_weights` no-op interaction |

Note: the transformers PR description currently mentions the determinism-fix as a side benefit. With the new understanding (the bug is in our `_initialize_weights` no-op), the determinism framing is partially wrong. Consider editing the PR body to remove the determinism claim and lean entirely on the perf framing.

## In-flight work (uncommitted, in worktree)

When the user said "stop", codex `init-fix` was running and had made local edits. They are uncommitted:

- `prismaquant/__init__.py` (modified) — codex implemented Option B from the brief: gate the `_initialize_weights` no-op on `torch.are_deterministic_algorithms_enabled()`. When det mode is on, run the original initializer; otherwise no-op as before.
- `tests/test_transformers_init_patch.py` (new, 93 lines)

The diff is small and clean. **Status: needs review, not yet committed or pushed.** Decide:
- Commit as-is and push to `feat/propagated-cost` + `main`, or
- Modify (e.g. switch to Option A — selective init that always runs buffer init), or
- Discard (keep the existing no-op, accept that det mode requires the vendored Qwen3 for now).

Codex's brief is at `/tmp/codex-brief-init-fix.md`; codex's log is at `/tmp/codex-init-fix.log` (it didn't print a final report because it was killed mid-run).

## The Qwen 4B / 4.75 bpp run

- Work dir: `/home/rob/dq-runs/qwen3-4b-perturbed-x/iterate_4p75/`
- Probe and cost reused from `/home/rob/dq-runs/qwen3-4b-perturbed-x/artifacts/`
- L2 converged in 2 iters: format histogram `{'BF16': 13, 'NVFP4': 240}`, achieved bpp 4.7511.
- L3 killed mid-flight at depth group 2/37. Partial outputs exist at `iterate_4p75/out/` but no L3 costs persisted.

If you want to resume: the L2 result is durable; you could re-launch L3 from there with `--resume-l3-costs-dir` once the perf issue is understood.

## Recommended next moves

1. **Decide on the codex init-fix patch** (review and commit, or modify, or discard).

2. **Investigate the L3 perf regression**. Hypothesis to test first: graph cache size. Bump
   `PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE`, `PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE`, and `PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH` to ≥ 32 and re-run a single layer. If layer 0 drops from 180 s to 30 s, cache thrash is confirmed.

   Failing that: rerun with `PRISMAQUANT_KL_CUDA_GRAPHS=0 PRISMAQUANT_L3_CUDA_GRAPHS=0 PRISMAQUANT_COORD_LANE_CUDA_GRAPHS=0` (the morning's "Option A" config) — this is the config that allegedly hit 22 min/anchor. If that's faster, graphs are net-negative for L3 and we should disable them in that path.

   Failing that: profile with `nsys profile` or `torch.profiler` to find the hot path.

3. **If perf is acceptable**, complete the 4.75 anchor: capture L3 costs, run coord descent, write final assignment + layer config, run validation harness. Compare against the v5 anchor 4.50 (KL 0.245, the morning's headline result).

4. **Update the transformers PR body** to remove the determinism-fix framing (now that we know the bug is local).

5. **Optional**: schedule a /loop or remote agent to monitor the transformers PR for review activity.

## Files / paths to know

```
/home/rob/prismaquant-propagated-cost/             — main worktree, on feat/propagated-cost
/home/rob/prismaquant-graph-profiler/              — profiler worktree, on tool/cuda-graph-memory-profiler
/home/rob/prismaquant-mistral-medium/              — older worktree
/home/rob/prismaquant-perturbed-x/                 — older worktree
/home/rob/prismaquant-quality-wins/                — older worktree

/home/rob/dq-runs/qwen3-4b-perturbed-x/
  artifacts/probe.pkl, cost.pkl                    — usable, no need to re-probe Qwen 4B
  iterate_v5_out/anchor_bpp_4.50/                  — morning's PROVEN result (KL 0.245)
  iterate_4p75/                                    — today's killed run (L2 done, L3 partial)

/home/rob/dq-runs/graph-smoke-2026-05-02/
  qwen4b_full_v2/                                  — 3-run bit-identical proof at 4B
  stat/                                            — earlier seed sweeps
  logs/                                            — all per-experiment logs
```

## Wrapper-bug reminder (still active)

Codex agents under bash wrappers occasionally report "completed" before the codex child exits. Always verify via `git log` + log file contents before trusting a wrapper signal. Today's runs were OK on this front, but watch for it.

## Open follow-ups for tomorrow

1. Init-fix patch decision (above).
2. L3 perf root-cause + fix.
3. Complete 4.75 anchor end-to-end if perf is acceptable; compare to v5 anchor 4.50.
4. Mirror the RoPE precompute fix to other model families if the transformers PR gets traction.
5. Update transformers PR body framing.
6. Decide whether to merge `tool/cuda-graph-memory-profiler` into main or leave as side tool.
