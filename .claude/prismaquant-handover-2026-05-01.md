# PrismaQuant handover — 2026-05-01

## User's standing demand (read first, hold to it)

**Prismaquant is a sensitivity-driven quantization optimizer.** That is
the entire conceptual core. The platform's job is:

1. Measure the sensitivity of every weight (Fisher / Hessian / activation
   curvature — all of it).
2. Measure the cost of each quantization choice in those sensitivity
   units.
3. Let the optimizer pick the format mix that minimizes expected loss
   under a bpp budget.

Anything that bypasses this — hardcoded format bans, streak limits,
"this format is dangerous so demote it", arbitrary clamps — is
**out of scope and not acceptable as a real fix**. They may exist as
debug aids during diagnosis but must not survive in the shipped
platform.

If the optimizer picks something that breaks at runtime, the
measurement is wrong, not the optimizer. Fix the measurement so the
optimizer sees the real cost; let it converge to the right answer
on its own. That is the spirit of prismaquant; preserve it.

The user has been explicit about this multiple times in this session.
Do not relitigate it. Do not propose another heuristic constraint.

## TL;DR

Latent allocator bug since 4/19 (commit `59004ca`) fully exposed by 4/30
Qwen 3.6 27B re-export with full wins stack — produced an artifact
that NaN-cascades to uniform-random output. Bug: `predicted_dloss =
0.5 * h_trace * weight_mse` ignores activation-quant error in W*A*
formats; with the wins reducing MXFP8 weight_mse disproportionately,
allocator picked MXFP8 at layers where W8A8 runtime error is much
worse than NVFP4. Fixed mathematically (`bc62104`) but exposed a
secondary issue: 5+ consecutive layers of all-MXFP8 MLPs cumulative-
drift to NaN at runtime because `block-output-match` skips refinement
when block-input Linear isn't NVFP4-pending. Two stopgap commits
landed (`530d1d2` streak-limit, `5e821dd` cal_input fallback) but
**they are debug aids, not production fixes**. User vetoed shipping
them. The principled path is **perturbed-X cost measurement**:
upstream quant noise should appear in cost-step X so the optimizer
sees true runtime cost; clusters are then either chosen because cost
endorses them or rejected because cost penalizes them. No magic
streak limits. No format bans.

## Where to start (next instance)

1. **Read the prior handover** at
   `/home/rob/prismaquant/.claude/prismaquant-handover-2026-04-29.md`
   for the previous session's state — overnight Qwen 3.5 4B smoke,
   27B re-export prep, etc. The 4/30 27B artifact mentioned there is
   the BROKEN one this session diagnosed.
2. **Read this doc top to bottom**.
3. **Don't re-derive the bug.** It's nailed down. The math is in
   `bc62104` commit message and in this file's "Root cause" section.
4. **Do NOT ship FIXED2** (`/home/rob/dq-runs/qwen3p6-27b-rerun/fisher_on_FIXED2/exported`). It is a stopgap with the
   streak-limit applied. User explicitly said "don't ship if it's
   shitty." It's working but heuristic-supported; not production.
5. **Plan the perturbed-X cost step** before writing code (see
   "Principled fix" section). Coordinate with user on design before
   long expensive runs.

## Branch / commits

Branch: `feat/quality-wins-batch1` (worktree at
`/home/rob/prismaquant-quality-wins`).

Commits landed this session (most recent first):

| sha | nature | ship? |
|---|---|---|
| `5e821dd` | export: block-output-match cal_input fallback | **debug only** — bundled some pre-existing in-worktree edits unrelated to the fix; will need clean re-commit if kept |
| `530d1d2` | allocator: cap consecutive same-module MXFP8 at 3 | **debug only** — heuristic with hardcoded MAX_STREAK |
| `bc62104` | allocator: use output_mse instead of weight_mse | **KEEP** — mathematically principled. This is the one true fix from this session. |

Remote: `origin/feat/quality-wins-batch1` is up-to-date with all three.

**My opinion on the commit hygiene**: `5e821dd` accidentally bundled
unrelated WIP from the worktree (200+ lines of pre-existing changes
to `export_native_compressed.py`). The actual fix is ~13 lines. If the
user decides to keep the cal_input fallback as a real fix, redo it as
a clean isolated commit. If not, just revert.

## Root cause (canonical narrative)

The allocator's per-Linear Δloss model:

```python
# allocator_solver.py:33
def predicted_dloss(h_trace, weight_mse, gain=1.0):
    return 0.5 * h_trace * weight_mse * gain
```

This is the diagonal-Fisher Δloss for **weight-only** quantization.
For W*A* formats (NVFP4 / MXFP8 — both quant activations at runtime
in vLLM's CUTLASS path) the runtime error is dominated by ACTIVATION
quantization, not weight. The cost step measures this correctly via
`output_mse = mean((y_ref - X̂ @ Ŵ.T)²)` (line 313 of
`measure_quant_cost.py`) — captures joint W·X perturbation. The
allocator just wasn't reading it.

Concrete evidence from the broken Qwen 27B run:

- Layer 13 `linear_attn.in_proj_z`:
  - NVFP4 weight_mse 2.09e-6, output_mse 4.08e-3
  - MXFP8 weight_mse 1.30e-6 (better), output_mse 7.57e-2 (**18.5×
    worse**)
  - Allocator picked MXFP8 because lower weight_mse → catastrophic
    runtime error.

11 of 19 MXFP8 picks in the 4/30 broken artifact had MXFP8 output_mse
> NVFP4 output_mse, i.e. were genuinely wrong choices invisible to
the allocator's weight-only model.

### Why the bug only became catastrophic on 4/30

The bug has been in place since `59004ca` (2026-04-19) — same day
the original Qwen 3.6 27B artifact shipped. Two things changed
between then and 4/30:

1. **Wins stack** (Fisher-weighted GPTQ, GPTQ damp sweep, etc.) —
   shrinks weight_mse for both NVFP4 and MXFP8, but disproportionately
   for MXFP8 (8 bits has more headroom to optimize). This pushes the
   weight-only ranking toward MXFP8 at more layers.
2. **Empirical Fisher correction** (`88a65c9`, 4/26) — fixed a math
   bug where Fisher was inflated 5–50× and layer-non-uniform. With
   correct Fisher, the allocator's cross-layer ranking shifted.
   Layers that previously demanded BF16 (because their inflated h_trace
   made any quant cost large) now allowed MXFP8 picks.

Neither change is wrong — both are correct improvements. They just
gave the latent bug enough rope to make it catastrophic.

## Pervasiveness audit

Allocator's MXFP8 picks under the BUGGY weight_mse-based ranking, vs
under the FIXED output_mse-based ranking, on every shipped model:

| Ship | Buggy picks | Fixed picks | Status |
|---|---|---|---|
| Qwen 3.6 27B 4/19 SHIPPED (HF) | 39 modules @ 3 layers (29/44/56) | not re-allocated this session — but ratios suggest ~1-2 picks survive | quality slightly suboptimal, runs |
| Qwen 3.6 27B 4/30 (broken) | 19 modules @ 10 layers | 10 modules @ 5 consecutive layers (24-28 mlp) | catastrophic NaN at runtime — but the FIXED picks ALSO failed (5-streak) until streak-limit demoted 27/28 to NVFP4 |
| **Gemma 4 31B SHIPPED** (HF) | **43 modules** | **1 module** (only layer 18 `self_attn.v_proj`) | quality measurably suboptimal — 95% miss rate. Should re-ship after principled fix lands. |
| MiniMax M2.7 | 0 (no MXFP8 in menu) | 0 | unaffected |
| Qwen 3.5/4 0.6B/4B | 0 (target_bits ≤ 4.5) | 0 | unaffected |

**Affected ships needing re-export: Qwen 3.6 27B + Gemma 4 31B.**

## Secondary issue: cumulative MXFP8 drift

After applying just the output_mse fix, the re-allocated Qwen 27B
produced 10 MXFP8 modules at layers 24-28 mlp.gate+up_proj — 5
consecutive layers. Runtime NLL: 12.42 (uniform random output).

What's happening: vLLM's MXFP8 CUTLASS path is W8A8. Per-MLP block
runtime error compounds across consecutive blocks. Normally
`block-output-match` (the export-side scale-sweep refiner) bounds
this drift. But it skips when the block-input Linear (gate_proj for
MLP, q_proj for attn) isn't in the NVFP4-pending refinement set —
so when gate_proj is MXFP8, `cal_input_mlp` lookup fails → `mlp=no_cal`
→ no drift bounding → 5 layers of unbounded drift → NaN.

Empirical boundary observed: 3-streaks survive (4/19 shipped + test_a
both have 24/25/26 mlp 3-streak and work); 5-streak breaks. The
boundary is heuristic.

The two stopgap commits address this two ways:
- `530d1d2` caps streak length to 3 in the allocator output
- `5e821dd` makes block-output-match's cal_input lookup work when
  gate_proj/q_proj is non-NVFP4

User's correct framing: **mixed quantization is fine, MXFP8 is fine,
errors should be bounded by the platform itself**. Constraints on
what the allocator can choose are debug aids, not fixes.

## Principled fix: perturbed-X cost measurement

**The signal we're missing is upstream quant noise.** Each layer's
need for precision is a function of (a) the loss-direction sensitivity
of its weights (Fisher) AND (b) the noise distribution it sees on its
input. Today we measure (a) and we measure (b) — but only under the
fiction that upstream layers are at full precision. At runtime,
upstream layers are quantized too, so input X is noisy. A layer
downstream of noisy upstream needs MORE precision (or to be careful
about formats that amplify noise), and a layer upstream of insensitive
downstream needs LESS. That whole picture is invisible to the current
cost step.

So the cost step's `X` is the unperturbed BF16 calibration activation
captured by the probe. Runtime `X` is the cumulatively-quantized
output of upstream layers. `output_mse` correctly measures "this
Linear in isolation under wizard-provided clean X" but is blind to
upstream quant noise feeding into its input distribution. The
optimizer then solves the wrong problem: it picks formats that
minimize per-layer-isolated cost, not formats that minimize total
loss under the actual coupled-noise dynamics of runtime.

The fix — explicitly model upstream quant noise via perturbed X.
**The optimization is bidirectional**: each layer's noise contribution
matters for downstream cost, and each layer's input noise matters
for its own cost. A correct platform must capture both. Level your
ambition appropriately:

### Level 1 — single-pass perturbed-X (fastest, partial)

1. Probe runs as today (clean activations cached).
2. Pick a baseline allocation A_0 (uniform NVFP4 + necessary BF16).
3. Apply RTN quant at A_0; forward-pass on cal data; cache perturbed
   activations at each Linear input.
4. Re-run cost step using perturbed activations. `output_mse(L, F)`
   now reflects realistic input noise.
5. Allocator picks A_1 against this fixed perturbed X.

Catches the downstream-suffers-from-upstream-noise direction but
NOT the closed feedback loop where layer L's choice affects layer
M's cost. ~½ day. Local optimum only. Probably good enough for the
specific Qwen 27B / Gemma 4 31B re-ships we have queued, but does
not deliver the platform's full principled spec.

### Level 2 — iterative to fixed point (recommended) [target]

Do Level 1, then iterate. Generate new perturbed X under A_1,
re-cost, re-allocate → A_2. Repeat until allocation stops changing.
At convergence, A* is a Nash equilibrium under the noise
distribution everyone is jointly producing — no single-layer format
flip improves loss given the others' picks. This explicitly
captures upstream→downstream propagation: when L tightens
precision, downstream costs shrink in the next iteration's
measurement, which the optimizer responds to.

Converges in 2–4 passes for typical models. ~1–2 days.

### Level 3 — direct sensitivity propagation (research, ideal)

For each (L, F) candidate, push the synthetic noise that F would
inject at L through the remainder of the network and measure
integrated downstream loss change. Per-(L, F) cost = local Δloss
+ propagated downstream amplification. Doesn't need iteration; the
joint-cost answer is computed directly. Costs O(n × |formats|)
extra forward work over the current cost step. ~3–5 days.

### Recommendation

Build Level 2 as the production target. It's principled, captures
both directions of the coupling, stays inside the measure-then-
optimize loop. Reuse Level 1 as the inner iteration step. Keep
Level 3 as a future enhancement if Level 2's local optima turn out
to be insufficient on some model.

Implementation entry points to study before coding:

- `prismaquant/calibrate_allocator.py` — there's already a
  calibration concept. Read first; might be a starting scaffold or
  might be unrelated. Confirm before extending.
- `prismaquant/measure_quant_cost.py` — the cost step. Currently
  loads `X` from `_CACHED_ACTIVATIONS` (probe-time cache). For
  perturbed-X mode, swap in a different cache populated by a
  pre-cost forward pass with allocation `A_0` applied.
- `prismaquant/incremental_probe.py` — see how probe forward pass
  is structured. The perturbed-X forward needs the same hooks but
  with weights being quant-dequant'd in-place per `A_0`.
- `prismaquant/export_native_compressed.py` line 309 — RTN quantize-
  dequantize per format. Already factored, reusable for the
  perturbed-X forward pass.

Cost: one extra forward pass per iteration. For Qwen 27B, ~5-10 min
per pass. Two iterations = ~20 min added to pipeline. Cheap.

This makes the platform actually do its job: measure realistic cost,
optimize over realistic cost. No format bans, no streak limits.

## Pending work (priority order)

1. **Revert `530d1d2` and `5e821dd`** once perturbed-X cost is
   implemented and validated. Until then leave them in for debug.
2. **Implement perturbed-X cost** as described above. Start by
   reading `calibrate_allocator.py`. Coordinate design with user
   before long runs.
3. **Re-allocate + re-export Qwen 3.6 27B** with perturbed-X cost.
   Validate vs shipped via `/home/rob/dq-runs/qwen3p6-27b-rerun/compare.py`.
4. **Re-allocate + re-export Gemma 4 31B**. Probe + cost data already
   on disk at `/home/rob/dq-runs/gemma4-31b/work/artifacts/`. Same
   workflow.
5. If both beat their shipped counterparts on raw + chat PPL,
   re-ship.
6. **Audit other recently-shipped artifacts** if any I missed.
   MiniMax variants likely safe (no MXFP8) but worth a glance.

## State of artifacts on disk

```
/home/rob/dq-runs/qwen3p6-27b-rerun/
  shipped_download/        — 4/19 shipped HF artifact (reference)
  fisher_on/exported/      — 4/30 broken artifact (DO NOT SERVE)
  fisher_on_FIXED/         — output_mse fix only, 5-streak, broken
  fisher_on_FIXED2/        — output_mse + manual 27/28 demote, 3-streak, works (stopgap)
  fisher_on_FIXED3/        — output_mse + streak-limit constraint, picks identical to FIXED2
  test_a / test_e / etc    — bisection artifacts from earlier debugging

/home/rob/dq-runs/gemma4-31b/
  work/artifacts/          — original probe + cost + buggy layer_config (43 MXFP8 picks)
  work_FIXED/artifacts/    — re-allocated layer_config under output_mse fix (1 MXFP8 pick)
```

## Useful commands / paths

- Compare harness: `/home/rob/dq-runs/qwen3p6-27b-rerun/compare.py`
  (edit `CONFIGS` list to point at the artifact you want vs `shipped`)
- Source BF16: `/home/rob/.cache/huggingface/qwen36-27b-bf16` (for
  Qwen 27B) and `/models/gemma-4-31b-it-bf16` (for Gemma)
- Container image: `vllm-fresh-b12x:latest`
- Wins stack env vars (set when re-exporting):
  - `PRISMAQUANT_GPTQ_DAMP_SWEEP=1`
  - `PRISMAQUANT_ACT_CLIP_QUANTILE=0.999`
  - `PRISMAQUANT_ACT_CACHE_FP32=1`
  - `PRISMAQUANT_BLOCK_OUTPUT_MATCH=1`
  - `PRISMAQUANT_DO_NO_HARM=1`
  - `PRISMAQUANT_GPTQ_FISHER_WEIGHT=1`
  - `PRISMAQUANT_BATCHED_NVFP4_EXPORT=0`
- HALO: leave `--halo-mode off` for Qwen 3.6 — `block_specs_for_layer`
  in `halo.py:328` does NOT include linear_attn, so HALO + DeltaNet
  hybrid attention will skip rotation on linear_attn modules and
  produce a topology-mismatched rotation. Future work: extend HALO
  block specs to handle linear_attn properly.

## Opinion (asked for)

- The output_mse fix (`bc62104`) is a clear win and should stay even
  if everything else is reverted. It's the one true principled fix
  from this session. Re-shipping Gemma 4 31B alone with this fix
  would meaningfully improve quality.
- The streak-limit (`530d1d2`) is the worst commit of the three.
  It's a magic-number heuristic in the optimizer that papers over a
  real measurement gap. Revert as soon as perturbed-X cost works.
- The cal_input fallback (`5e821dd`) is more defensible — it fixes a
  legitimate gap where block-output-match was silently skipping work
  it should have done. But its scope is asymmetric: it lets refinement
  see the block-input but doesn't extend refinement to MXFP8 modules
  themselves. So it doesn't actually bound runtime drift in
  all-MXFP8 blocks; it just helps slightly when at least one module
  in the block is NVFP4. With perturbed-X cost in place this becomes
  unnecessary because the allocator won't pick all-MXFP8 blocks
  unless cost endorses them.
- The biggest risk in the perturbed-X path is that the iteration
  doesn't converge or oscillates between allocations. Standard fix:
  damp the iteration (mix new + old allocation 50/50) or just take
  one iteration and ship. Start simple.
- HALO + DeltaNet (linear_attn) is a separate latent issue. HALO is
  off by default so not biting us, but if someone turns it on in the
  qwen3_5_dense profile they'll get a topology mismatch. Worth a
  TODO comment in the profile or `halo.py`.
- Don't re-introduce the streak-limit even after perturbed-X cost
  ships. It's a measurement-bypass, and prismaquant's invariant is
  "the platform measures and optimizes."
- If the perturbed-X work is going to take more than a week, consider
  shipping Gemma 4 31B re-allocate (using just output_mse fix, since
  Gemma's fixed picks have no streaks and don't need stopgaps) as an
  intermediate win. Qwen 27B can wait.

## Save this commit / state

The FIXED2 export under `fisher_on_FIXED2/` is the only validated-
working Qwen 27B artifact in this whole session that beats shipped
on raw PPL. Compare result for it never finished (killed mid-run).
If you want a working artifact to compare against during perturbed-X
development, it's there. Just don't ship.
