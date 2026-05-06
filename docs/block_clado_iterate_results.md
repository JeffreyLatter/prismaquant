# Block-CLADO iterate-pipeline smoke results (Qwen3-0.6B)

Branch: `block-clado` (commits through `7bf1889`)

## Pipeline composition

```
iter_0:  measure (BF16-centered)   → λ-sweep → validate cone → pick best validated
         → coord-descent polish (5% bits-budget creep)

iter_1:  measure (centered at iter_0 polished)  → λ-sweep → validate cone → pick best
         → coord-descent polish

iter_2+: same; stops when polished assignment is unchanged across iterations.
```

Single-process driver: `prismaquant.iterate_block_clado`.  Model + reference
log-probabilities are loaded once and shared across all iterations.

## Per-iteration result (TBD after run completes)

(Numbers filled in from `summary.json` once the iterate run lands.)

| iter | centered_at | kneedle bpp | best validated KL | polished KL | polish steps | elapsed |
|---|---|---|---|---|---|---|
| 0 | BF16 | _t.b.d._ | _t.b.d._ | _t.b.d._ | _t.b.d._ | _t.b.d._ |
| 1 | iter_0_polish | _t.b.d._ | _t.b.d._ | _t.b.d._ | _t.b.d._ | _t.b.d._ |
| 2 | iter_1_polish | _t.b.d._ | _t.b.d._ | _t.b.d._ | _t.b.d._ | _t.b.d._ |

## What we already know from the standalone runs

| Stage | bpp | real KL | notes |
|---|---|---|---|
| Block-CLADO surrogate kneedle | 4.86 | 0.217 | surrogate-only selection |
| Block-CLADO best validated frontier | 4.57 | 0.124 | real-KL-gated cone scan |
| + Coord-descent polish | ~4.65 | 0.087 | 30% relative KL improvement |

The polish landed two NVFP4→BF16 upgrades on `model.layers.24.self_attn.o_proj`
and `model.layers.27.mlp.down_proj`.  These two layers are downstream
attention/MLP outputs — the precision bottleneck for cross-layer
propagation.  The Block-CLADO surrogate did not flag them strongly enough on
its own.

## What sandwich recalibration is supposed to add

The 14-point validated frontier had real KL fluctuating 0.124–0.389 across
points 0.01 bpp apart (specifically: bpp 4.5528 → real KL 0.389, bpp 4.5726 →
0.124).  This is the second-order Taylor approximation breaking down for
specific format configurations.

Sandwich recalibration re-measures Ω_ii / Ω_ij centered at the polished
assignment instead of at BF16, so the linear surrogate sits in the basin
where the polished assignment lives.  Hypothesis: the iter-1 frontier
should be smoother (less point-to-point variance) and the iter-1 polished
KL should be ≤ iter-0 polished KL.

## Engineering notes

- Each iteration on Qwen3-0.6B costs ~7-8 minutes (76 s measure + 30 s validate cone + 5-7 min polish).
- Sandwich measurement is expensive in absolute terms but the COMPLEXITY is the same as the initial measurement (just re-runs the four-term identity from the new center).  No incremental "delta" optimisation is implemented yet.
- The polish budget creep tolerance (default 5%) lets polish take a small number of Pareto-beneficial precision upgrades without runaway BF16 drift.  Strict budget (`--polish-budget-creep 0.0`) is also supported for fixed-bpp shipping.
- Steepest-first polish (`--polish-steepest-first`) orders candidates by surrogate ΔΩ and accepts the first measured improvement; can be 5-10× faster per pass when the surrogate ranks moves accurately near the current point.

## Open questions

1. How much does sandwich actually help on this model?  (Pending iterate result.)
2. Does the iterate pipeline converge in ≤3 iterations, or does it oscillate?
3. Does the steepest-first polish converge to the same final point as greedy-best?  (Need both flags on the same starting point.)
4. Does the measured frontier "smoothness" improve under sandwich centering?  (Variance of real KL across adjacent bpp points.)
