# Block-CLADO iterate (sandwich + polish) results — Qwen3-0.6B

Branch: `block-clado`
Run: `/home/rob/dq-runs/qwen3-0p6b-block-clado-iter-20260506T023854Z`

Three-iteration `iterate_block_clado` with sandwich recalibration:

```
iter_0:  measure (BF16-centered)        → λ-sweep → validate cone → polish
iter_1:  measure (centered at iter_0)   → λ-sweep → validate cone → polish
iter_2:  measure (centered at iter_1)   → λ-sweep → validate cone → polish
```

Each iteration uses 5% bits-budget creep on polish, frozen weight cache
enabled (`--use-frozen-weight-cache`), greedy-best polish ordering.
Total wall time: **37.5 min** for all three iterations.

## Headline numbers

| iter | centered_at | best_validated bpp / KL | polished KL | polish_steps | elapsed |
|---|---|---|---|---|---|
| 0 | BF16 | 4.5726 / 0.124 | **0.0871** | 2 | 7.2 min |
| 1 | iter_0_polish | 8.0008 / 0.121 | **0.0470** | 4 | 15.7 min |
| 2 | iter_1_polish | 7.0375 / 0.125 | 0.0711 | 4 | 14.6 min |

**Best overall: iter 1, polished KL 0.0470 at ~8.4 bpp** (`best_assignment.json`).

Compared to the pre-iterate baselines on the same model + calibration:

| Method | bpp | real KL |
|---|---|---|
| Surrogate kneedle alone (no validation, no polish) | 4.86 | 0.217 |
| Frontier-validated best (no polish) | 4.57 | 0.124 |
| Iter 0 polish (BF16-centered + polish, 5% creep) | ~4.65 | 0.087 |
| **Iter 1 polish (sandwich-centered + polish, 5% creep)** | **~8.4** | **0.0470** |

The pipeline reduces real KL by **78%** over the surrogate kneedle alone
and **46%** over the BF16-centered + polish baseline, but at the cost of
nearly doubling bpp.  This is the classic Pareto trade-off being explored.

## What sandwich actually does (and doesn't)

**The sandwich-centered surrogate is not a trustworthy kneedle picker at
this scale.** Iter 1's surrogate frontier predicts costs ranging from
−0.99 to −10.60 (negative = improvement over center), but the smallest
real KL on the validation cone is 0.121 — *worse than* the centered KL
of 0.087.  Iter 2 amplifies this: surrogate costs go to −9.16, real KL
ranges 0.125–0.414.  The Taylor expansion linearises around the centered
point, but the actual quantization perturbations are large enough to
push the surrogate well outside the trust region.

What sandwich *does* do is shift the validation cone toward higher-bpp
regions of the frontier, where polish can subsequently extract real KL
wins by upgrading specific high-impact units to BF16.  Iter 1 polish
accepted four `MXFP8 → BF16` and `NVFP4 → MXFP8` moves on attention QKV
and MLP-down layers.  These were not visible to the BF16-centered iter 0
polish because at that center the same units were already at NVFP4 and
the relevant perturbation Ω_ii(NVFP4 → BF16) was the wrong cost-of-
improvement direction.

## Per-iteration polish trajectories

```
iter 0  (start 0.124 → final 0.087, 2 moves, ~4.65 bpp)
  pass 0  layers.24.self_attn.o_proj  NVFP4 → BF16   KL 0.124 → 0.095
  pass 1  layers.27.mlp.down_proj     NVFP4 → BF16   KL 0.095 → 0.087

iter 1  (start 0.121 → final 0.047, 4 moves, ~8.4 bpp)
  pass 0  layers.5.self_attn.qkv_proj  MXFP8 → BF16  KL 0.121 → 0.073
  pass 1  layers.3.self_attn.qkv_proj  MXFP8 → BF16  KL 0.073 → 0.056
  pass 2  layers.26.mlp.down_proj      MXFP8 → BF16  KL 0.056 → 0.053
  pass 3  layers.27.self_attn.o_proj   NVFP4 → MXFP8 KL 0.053 → 0.047

iter 2  (start 0.125 → final 0.071, 4 moves, mixed-direction)
  pass 0  layers.20.self_attn.o_proj   MXFP8 → NVFP4   ← downgrade!
  pass 1  layers.6.self_attn.o_proj    NVFP4 → BF16
  pass 2  layers.16.mlp.gate_up_proj   NVFP4 → BF16
  pass 3  layers.27.self_attn.o_proj   MXFP8 → BF16
```

Iter 2 polish opens with a *downgrade* (MXFP8 → NVFP4): it inherited a
mostly-BF16 base from iter 1 and the budget creep gives it room to swap
precision around — the algorithm is genuinely exploring the local Pareto
basin around iter 1 polish, not just monotonically upgrading.

## Validation cone details

### Iter 0 (BF16-centered, 9-candidate cone around bpp 4.86 kneedle)
```
4.5528  surrogate +0.224  real_kl 0.389  {MXFP8: 5,  NVFP4: 192}
4.5726  surrogate +0.209  real_kl 0.124  {MXFP8: 6,  NVFP4: 191}  ← best
4.6781  surrogate +0.159  real_kl 0.214  {MXFP8: 15, NVFP4: 182}
4.7507  surrogate +0.132  real_kl 0.197  {MXFP8: 21, NVFP4: 176}
4.8629  surrogate +0.102  real_kl 0.217  {MXFP8: 31, NVFP4: 166}  ← kneedle
5.0212  surrogate +0.074  real_kl 0.224  {MXFP8: 42, NVFP4: 155}
5.2007  surrogate +0.052  real_kl 0.135  {MXFP8: 50, NVFP4: 146, BF16: 1}
5.2601  surrogate +0.047  real_kl 0.176  {MXFP8: 55, NVFP4: 141, BF16: 1}
5.3815  surrogate +0.039  real_kl 0.197  {MXFP8: 53, NVFP4: 141, BF16: 3}
```

Surrogate-vs-real Spearman ρ ≈ 0.23 — surrogate gets the macroscopic
shape but fails point-by-point ranking.

### Iter 1 (sandwich-centered at iter 0 polish, KL 0.087)
```
4.6320  surrogate -0.987   real_kl 0.182  {MXFP8: 10, NVFP4: 187}
4.6452  surrogate -1.094   real_kl 0.269  {MXFP8: 11, NVFP4: 186}
5.0560  surrogate -3.417   real_kl 0.179  {MXFP8: 35, NVFP4: 160, BF16: 2}
5.1549  surrogate -3.799   real_kl 0.233  {MXFP8: 42, NVFP4: 153, BF16: 2}
6.0641  surrogate -6.421   real_kl 0.127  {MXFP8: 66, NVFP4: 117, BF16: 14} ← kneedle
7.1215  surrogate -8.469   real_kl 0.171  {MXFP8: 90, NVFP4: 74,  BF16: 33}
8.0008  surrogate -9.728   real_kl 0.121  {MXFP8: 105,NVFP4: 45,  BF16: 47} ← best
8.8158  surrogate -10.530  real_kl 0.279  {MXFP8: 108,NVFP4: 24,  BF16: 65}
8.9249  surrogate -10.605  real_kl 0.168  {MXFP8: 105,NVFP4: 24,  BF16: 68}
```

The surrogate is wildly over-confident here.  None of the validation
points actually beat the centered KL of 0.087 — the surrogate's predicted
−9.73 improvement at 8.0 bpp evaporates to a +0.034 *regression* in
reality.  Polish then claws back to 0.047 from a higher-bpp starting
point.

## Headline interpretation

- The **measure → λ-sweep → frontier-validate → polish** pipeline is the
  load-bearing path.  Each stage demonstrably contributes value on real
  KL.
- **Sandwich recalibration** acts more as a Pareto-exploration mechanism
  than a kneedle-quality improver.  It doesn't make the surrogate more
  predictive at this scale, but it does push the system into bpp regions
  the BF16-centered surrogate didn't propose.
- **Polish budget creep** drives most of the apparent per-iteration KL
  improvement.  Tightening creep to ~1% would force apples-to-apples bpp
  comparison across iterations and isolate the sandwich effect cleanly;
  loosening it lets polish chase Pareto improvements at higher cost.

## Recommendations

1. **For fixed-bpp shipping** at the original ~4.5 bpp target: use a
   *single* iter (BF16-centered) with strict polish budget (`--polish-
   budget-creep 0.0`).  Result: 4.57 bpp / KL 0.124 (frontier-validated)
   or 4.65 bpp / KL 0.087 (with 5% creep allowing two BF16 upgrades).
2. **For Pareto-exploration**: keep iterate; report all iterations'
   (bpp, KL) pairs and let the operator pick.  The `best_overall`
   mechanism correctly tracks the lowest-KL point across rounds.  In this
   run that's iter 1 at 8.4 bpp / KL 0.047.
3. **For surrogate fidelity at non-trivial centers**: the next research
   direction is the Output-Fisher form — cache δz_i^m once per (i, m)
   and compute pair interactions analytically through the closed-form
   per-token Fisher.  This replaces rank-R Gaussian probing with full-
   rank-(T·V) precision and should fix the sandwich's surrogate-cost
   over-prediction.

## Cost on Qwen3-0.6B

Total iterate runtime: **37.5 min** for 3 iterations with
`--use-frozen-weight-cache` enabled.  Without frozen cache the iter-1
sandwich measurement alone would have taken hours — every measurement
re-quantizes the centered base when caching is off.

Per-iteration breakdown:
- iter 0: 7.2 min (no center cost; ~7 min polish dominant)
- iter 1: 15.7 min (76 s center_kl + 7 min measure + 8 min polish)
- iter 2: 14.6 min (similar; pol`ish converged after 4 accepted moves)

For 4B / 27B class scaling, polish is the bottleneck and the next levers
are:
- `--polish-steepest-first` (already implemented): order trials by
  surrogate ΔΩ and accept the first improvement, 5-10× speedup per pass.
- A larger (unit, format) → quantized-weight cache pool, swapping in/out
  per trial.  Defers re-quantization across polish trials.  ~50 GB at
  27B; OOM-risky on 121 GB UMA.
