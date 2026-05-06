# Block-CLADO pipeline

This is the complete Block-CLADO pipeline as it lives on the `block-clado`
branch.  All claims are validated on Qwen3-0.6B; scaling to 4B / 27B is
projected from runtime estimates only.

## Architectural overview

```
┌─────────────────────────────────────────────────────────────────────┐
│   prismaquant.measure_block_clado                                   │
│   ─────────────────────────────────────                             │
│   For each fused-sibling decision unit U and each format f:         │
│     Ω_ii(U, f) = KL(teacher ‖ x_c ⊕ U→f) − KL(x_c)                  │
│   For each intra-block pair (U_a, U_b) and (f_a, f_b):              │
│     Ω_ij = KL(x_c ⊕ a→f_a, b→f_b) − Ω_ii(a,f_a) − Ω_ii(b,f_b) − KL(x_c) │
│   x_c is BF16-everywhere by default, or any per-Linear assignment   │
│   when invoked with --center-assignment (sandwich recalibration).   │
│                                                                     │
│   Output: prismaquant.block_clado.v1 JSON payload                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│   prismaquant.block_clado (solver)                                  │
│   ─────────────────────────────────                                 │
│   Per-block exact enumeration of fused-group format combos with     │
│   Pareto filtering by (bits_total, cost).                           │
│                                                                     │
│   Two solve paths:                                                  │
│   • Lagrangian λ-sweep: per-block independent argmin(cost+λ·bits);  │
│     log-scale sweep over λ recovers the convex-hull frontier.       │
│   • Multi-choice knapsack DP: exact budget-constrained solve when   │
│     a specific bpp target matters more than the frontier shape.     │
│                                                                     │
│   Output: sweep JSON (bpp, surrogate_cost, per-unit assignment)     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│   prismaquant.block_clado kneedle / validate_block_clado            │
│   ─────────────────────────────────────────────────────              │
│   Pick max-perpendicular-distance elbow on the positive-cost        │
│   region of the frontier, expand per-unit assignments back to       │
│   per-Linear members, validate the kneedle ± neighbors with real    │
│   teacher-student KL.                                               │
│                                                                     │
│   Output: validated kneedle assignment JSON + per-candidate KL     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│   prismaquant.coord_descent_polish                                  │
│   ─────────────────────────────────                                 │
│   Real-KL-gated single-flip search.  For each unit and each non-    │
│   current format option, measure trial KL; accept the move with     │
│   the largest improvement (greedy-best) or first improvement that   │
│   beats the noise floor (steepest-first when surrogate priority is  │
│   supplied).                                                        │
│                                                                     │
│   Constraints:                                                      │
│   • Fused-sibling members move together.                            │
│   • Optional bits_budget cap with creep tolerance (default 5% in    │
│     iterate, 0% in standalone polish).                              │
│                                                                     │
│   Output: polished assignment JSON + step-by-step KL trace          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ (sandwich)
┌─────────────────────────────────────────────────────────────────────┐
│   prismaquant.iterate_block_clado                                   │
│   ─────────────────────────────────                                 │
│   Wraps measure → sweep → validate → polish in a loop, re-centering │
│   the sandwich payload on the polished assignment each iteration    │
│   until the polish output stops changing.  Runs in-process so the   │
│   model + ref-logprob cache + tokenized calibration are paid once.  │
│                                                                     │
│   Output: per-iteration JSONs + summary.json + best_assignment.json │
└─────────────────────────────────────────────────────────────────────┘
```

## Design decisions

### Loss = teacher-student KL, not task CE

Centering the Taylor expansion at the BF16 reference (KL = 0 by definition)
makes the linear term vanish exactly: `KL(w + Δw) ≈ (1/2) Δw^T F Δw` where
F is the categorical Fisher of the teacher's output distribution.  Two
practical consequences:

1. The four-term identity collapses to three measured terms, since
   `KL(teacher ‖ teacher) = 0`.  ~33% speedup on pair measurements.
2. The Fisher is PSD by construction.  CLADO's eigendecomposition + clip-
   negative-eigenvalues PSD projection step is unnecessary; sample noise
   can make F low-rank but never indefinite.

CoopQ (2509.15455) reaches the same conclusion via Shapley sampling but
needs an `α`-shrinkage hyperparameter to denoise off-diagonals.  Block-
restricted four-term identity sidesteps that by only measuring the pairs
that actually matter.

### Block restriction over treewidth-2 cliques

Full-coverage CLADO requires `O(|𝔹|² · I²)` measurements (~1.7M forwards
on Qwen 27B, ~700 GPU-hours).  Block restriction to within-architectural-
block edges drops this to `O(L · K²)` where K is the fused-group count
inside one transformer block (~4 for q/k/v + o + gate/up + down).

Empirically, transformer LayerNorms reset error magnitude between blocks,
so cross-block `Ω_ij` is small relative to within-block.  The per-block
clique has small treewidth (the V→O edge is 2 nodes, the Gate→Up→Down
edge is 3 nodes), so per-block exact DP is cheap even without the IQP
machinery CLADO uses.

### Sandwich recalibration via the same collector

A single `--center-assignment` flag on `measure_block_clado` switches
from BF16-centered measurement to any-point-centered measurement.  No
parallel code path: the four-term identity is the same, only the base
state changes.  This makes the iterate runner straightforward.

### Real-KL gating at every selection step

Surrogate-real Spearman correlation on Qwen 0.6B is 0.23 (Block-CLADO)
vs 0.63 (unary-only).  The Block-CLADO surrogate gets the macroscopic
frontier shape right but is unreliable at point-level discrimination —
exactly because the cross-layer cancellation it captures is also the
source of higher-order noise.

Consequence: every selection step is gated by measured KL, never by
surrogate cost alone:

* The kneedle is picked from validated points, not surrogate points.
* Polish accepts a move only if the trial real KL beats current real KL
  by more than the noise floor.
* Iterate stops when the polished assignment is unchanged across rounds.

The surrogate is for *generating* candidates; reality decides which to
ship.

## Smoke results on Qwen3-0.6B

Initial Block-CLADO + frontier validation (no polish):

| bpp ceiling | Block-CLADO best | Unary-only baseline | Δ |
|---|---|---|---|
| ≤ 4.6 | **0.124** @ 4.57 bpp | 0.202 @ 4.57 bpp | **39% better** |
| ≤ 5.5 | **0.124** @ 4.57 bpp | 0.178 @ 5.19 bpp | **30% better** |
| ≤ 9.0 | **0.124** @ 4.57 bpp | 0.155 @ 7.87 bpp | **20% better** |

Coord-descent polish on the best validated point (4.57 bpp / KL 0.124):

| metric | starting | polished | Δ |
|---|---|---|---|
| KL | 0.1238 | **0.0871** | **−30%** |
| bpp | 4.5726 | ~4.61 | +0.8% |
| accepted moves | — | 2 | — |
| total measurements | — | 679 | — |
| elapsed | — | 605 s | — |

**Net pipeline result on Qwen3-0.6B:** ~4.61 bpp / real KL 0.087 from a
~10 min full pipeline (76 s measurement + ~5 min validation + ~10 min
polish), vs surrogate-kneedle-alone 4.86 bpp / KL 0.217 (worse on both
axes).

Iterate (sandwich-recalibrated) results: see `iterate-block-clado-iter-*`
run dirs and the `summary.json` therein.

## Cost projections

| Model | Linears | Per-iteration cost (approx) |
|---|---|---|
| Qwen3-0.6B | 197 | 76 s measure + ~10 min polish ≈ 12 min |
| Qwen3-4B | ~250 | ~5 min measure + ~25 min polish ≈ 30 min |
| Qwen3.6-27B | ~614 | ~20 min measure + ~3 hr polish ≈ 3.5 hr |

Polish cost dominates because each candidate trial does a full forward
pass.  `--polish-steepest-first` reduces measurements per pass roughly
5–10× when the surrogate ranks moves accurately around the current
assignment, bringing 27B polish to ~30 min/pass / ~1 hr per iteration.

## Test coverage

`tests/test_block_clado.py`: 15 tests — types, payload IO, per-block
Pareto enumeration, λ-sweep, budget DP, kneedle picking, expansion.

`tests/test_coord_descent_polish.py`: 9 tests — strict-improvement
acceptance, greedy-best ordering, noise floor, fused-sibling movement,
budget enforcement, unconstrained mode, surrogate-priority ordering.

All 24 tests pass.  Sandwich recalibration is integration-tested via
the iterate runner; centered-measurement math is validated by the
identical-output property when `center_assignment` is BF16-everywhere.

## Open questions worth empirical answers

1. **Does sandwich recalibration help?**  Iterate output answers this
   directly: if iter_1 polished KL < iter_0 polished KL, sandwich is
   doing real work.  If they're within noise, the BF16-centered surrogate
   is already good enough and we can drop the sandwich step.
2. **Does the pipeline survive scale?**  The 0.6B numbers are encouraging
   but small-model behavior is not always predictive — at 4B and 27B the
   relative weight of cross-layer interactions can shift.
3. **Does it help downstream task quality?**  KL is a calibration-set
   token-distribution metric; the 27B PrismaSCOUT history (TC-42
   regression at lower KL) is a reminder that mean KL is not a sufficient
   shipping criterion.
