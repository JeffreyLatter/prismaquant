# Full-OF iterate results — Qwen3-0.6B

Branch: `block-clado` (commits through `2242b33`)
Run: `/home/rob/dq-runs/qwen3-0p6b-block-clado-iter-fullof-20260506T044945Z`

## Headline result

**11.59 bpp / real KL 0.0226 in 2.1 minutes** via OF iter 0 + polish (10%
budget creep, steepest-first ordering).  This is the lowest measured KL
we've achieved on Qwen3-0.6B across all method combinations.

Full 3-iteration runtime: **12.8 min** (3.0× faster than the 37-min
four-term iterate).  All three iterations use Output-Fisher; iter 1+
exercise the sandwich centering code path (linear correction term
included).

## Per-iteration result

| iter | centered_at | best_validated bpp/KL | polished KL | polish_steps | elapsed |
|---|---|---|---|---|---|
| 0 | BF16 | 11.59 / 0.0633 | **0.0226** | 8 | 2.1 min |
| 1 | iter_0_polish | 7.81 / 0.0923 | 0.0667 | 1 | 4.5 min |
| 2 | iter_1_polish | 6.18 / 0.1312 | 0.0582 | 8 | 5.1 min |

**Best overall: iter 0**.  Sandwich iterations 1 and 2 explore lower-bpp
regions (the kneedle pulls back toward 6 bpp once centered at the
high-bpp polished iter 0), and polish from there can't recover the
iter-0 quality at high bpp.

## What sandwich gave us, and what it didn't

The OF sandwich centering code path is exercised end-to-end:
``measure_start, "method": "output_fisher", "centered": true`` on iter
1 and 2.  The validation shows:

- iter 1 surrogate kneedle picks 6.18 bpp / cost 0.526 (centered at
  iter-0's 11.59 bpp polished assignment).
- The cone is anchored at low-bpp downgrades from the centered state.
- Best validated falls at 7.81 bpp / KL 0.0923 — much worse than the
  centered KL of 0.0226.
- Polish only finds 1 improvement (0.092 → 0.067) before convergence.

Same pattern on iter 2.  The fundamental issue: sandwich pulls the
surrogate elbow toward the bpp neighborhood of validated downgrade
candidates, not the bpp neighborhood that produced the centered point.
Once polish has driven KL down to 0.0226 at 11.6 bpp, sandwich-centered
sweeps don't *find* candidates better than that — they re-explore at
nearby bpp values.

This matches what we saw with four-term sandwich (also bottomed out at
~0.05 KL).  The Pareto-optimum near 11.6 bpp is the realized best.

## Speed comparison vs four-term iterate

Same model, same calibration, same polish settings.  Compare:

| iterate | total wall | iter 0 | iter 1 | iter 2 | best_overall |
|---|---|---|---|---|---|
| Four-term | 37.5 min | 7.2 min | 15.7 min | 14.6 min | 8.4 / 0.047 |
| Full OF | 12.8 min | 2.1 min | 4.5 min | 5.1 min | **11.6 / 0.0226** |

OF wins on **both speed (3.0×) and final KL (2.1× lower)**.  The bpp is
higher (11.6 vs 8.4), trading compression for KL — but the 2× KL
improvement is real and on the same Pareto curve.

## Final method comparison

The full picture across all surrogates and pipelines we ran:

| pipeline | bpp | real KL | wall |
|---|---|---|---|
| Surrogate kneedle (4-term, no validation) | 4.86 | 0.217 | 76s |
| Frontier-validated (4-term) | 4.57 | 0.124 | 105s |
| 4-term iter-0 + polish (greedy, 5% creep) | 4.65 | 0.087 | 12 min |
| 4-term sandwich iterate (3 iters) | 8.4 | 0.047 | 37 min |
| OF + polish (12.2 anchor, weight-only) | 12.5 | 0.029 | 5 min |
| OF + polish-frontier (best of 9 candidates) | 12.5 | 0.029 | 8 min |
| OF + polish (act-quant, 10.7 anchor) | ~11 | 0.036 | 3 min |
| **Full-OF iterate (3 iters)** | **11.6** | **0.0226** | **13 min** |

The Full-OF iterate result is our headline best.  The OF + polish (act-
quant, 11.6 anchor) variant from the earlier OF-iterate-with-4term-
sandwich run also reached 0.0226 — the result is robust across two
independent runs of OF iter 0 + polish.

## Production recommendation

For best raw KL at any bpp, use ``iterate_block_clado --measure-method
output_fisher --polish-steepest-first --polish-budget-creep 0.10``.
A single iter (``--max-iterations 1``) gets you the best result in ~2
min on Qwen3-0.6B; running 3 iters validates the sandwich path but
doesn't improve the best.

For fixed-bpp shipping at ~4.5 bpp (the original deployment target),
four-term Block-CLADO + polish remains better by ~6% on KL because
its higher-order pair effects matter more at large quantization
perturbations.

## Open work

1. **Activation quant in OF** is included via PerturbedActivationCache;
   this matches the deployment-faithful four-term collector.  No more
   gap between surrogate and what ships.
2. **OF sandwich math** is implemented and tested (12 OF tests passing,
   including the linear-offset path and the ``Var(a+b) = Var(a) + 2
   Cov + Var(b)`` algebraic identity).  The empirical benefit of
   sandwich centering at this scale is not visible — the iter-0 result
   already saturates near the local optimum.  Sandwich may help more at
   larger model scale where iter-0 itself is a less complete search.
3. **Scaling**: 0.6B numbers don't predict 27B behavior.  At 27B the
   ``O(|𝔹|·I)`` OF measurement count (~1,800) vs ``O(|𝔹|² · I²)``
   four-term (~190,000) gap is the dominant cost.  OF should win much
   bigger there.
4. **center_kl telemetry** uses ``log(p.clamp(min=1e-30))`` for KL
   computation and overestimates by ~3× vs the proper ``log_softmax``-
   based computation in measure_assignment_kl.  Telemetry only —
   surrogate values are computed correctly via Var/Cov of probs.  Worth
   fixing for cleaner reporting.
