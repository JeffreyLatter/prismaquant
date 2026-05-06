# Block-CLADO + Output-Fisher pipeline — final Qwen3-0.6B results

Branch: `block-clado` (commits through `5ff5934`)

## Best result

**~12.2 bpp / real KL 0.0287** via Output-Fisher measurement → λ-sweep
→ frontier validate → polish (steepest-first, 5% creep) starting from
the validated 12.2 bpp candidate.

Total wall time: ~5 minutes.

## Method comparison (all on Qwen3-0.6B, same calibration)

| pipeline | bpp | real KL | wall time | notes |
|---|---|---|---|---|
| Surrogate kneedle alone (4-term) | 4.86 | 0.217 | 76s | trust surrogate, no validation |
| Frontier-validated best (4-term) | 4.57 | 0.124 | ~2 min | gate on real KL but no polish |
| 4-term iter 0 + polish (greedy) | ~4.65 | 0.087 | 12 min | original baseline |
| 4-term sandwich-iter best (3 iters) | ~8.4 | 0.047 | 37 min | Pareto-explored via sandwich |
| OF + polish (low-bpp anchor, 5% creep) | ~5 | 0.092 | 2.4 min | OF can't beat 4-term at low-bpp |
| OF + polish (mid-bpp 7.8 anchor) | ~8 | 0.064 | ~4 min | |
| OF + polish (10.7 anchor) | ~11 | 0.036 | ~3 min | |
| **OF + polish (12.2 anchor, 5% creep)** | **~12.5** | **0.0287** | **~5 min** | **best so far** |
| OF iterate + sandwich (iter 0 OF, iter 1+ 4-term) | TBD | TBD | TBD | running |

## What we learned

### 1. The Pareto curve is broad, not a single elbow

Real-KL-validated assignments at every measured bpp:

```
bpp     real_KL   format mix
4.50    0.218     all NVFP4
4.57    0.124     191 NVFP4 + 6 MXFP8         ← 4-term frontier best
4.65   ~0.087     +2 BF16                     ← 4-term polish
5.20    0.135     ~146 NVFP4 + 50 MXFP8 + 1 BF16
6.06    0.127
7.04    0.125
7.77    0.162     surrogate kneedle (OF)
8.00    0.121
8.40   ~0.047     iter-1 sandwich + polish
9.34    0.112
10.69   0.062
10.7   ~0.036     OF polish from 10.69
11.4   ~0.029     OF + 12.2 anchor polish      ← BEST
12.19   0.061     OF validated
12.50  ~0.029     polish output (this run)
16.00   0.000     all-BF16 (trivial)
```

The Pareto curve is monotone in real KL only on average — local
fluctuations from cross-layer interactions create ~0.05 KL noise at
similar bpp.

### 2. Output-Fisher vs four-term: complementary surrogates

Side-by-side payload comparison on Qwen3-0.6B:

| component | n entries | Spearman ρ |
|---|---|---|
| Unary Ω_ii | 339 | **+0.895** (strong agreement) |
| Pair Ω_ij | 1,512 | **−0.10** (essentially uncorrelated) |

The four-term identity captures higher-order ``O(δw³+)`` finite-
difference effects on cross-layer interactions that the analytic
second-order Fisher form misses.  These higher-order effects
materially flip the sign of pair Ω_ij values.

**Practical consequence**: at low bpp (~4.5) where individual
quantization perturbations are large, four-term wins by ~6% on real KL.
At high bpp (~10+) where δw is small, OF's analytic Fisher is closer
to truth and faster.  Polish (real-KL-gated) always rescues both.

### 3. Polish is the workhorse, not the surrogate

Across every starting point we tested, polish accounts for the
majority of the realized KL improvement:

| starting (validated) | polished | Δ |
|---|---|---|
| 4.57 / 0.124 | ~4.65 / 0.087 | -29% |
| 4.78 / 0.139 | ~5.0 / 0.092 | -34% |
| 7.77 / 0.162 | ~8 / 0.065 | -60% |
| 10.69 / 0.062 | ~11 / 0.036 | -42% |
| 12.19 / 0.061 | ~12.5 / 0.029 | -53% |

Steepest-first ordering (sort candidates by surrogate ΔΩ, accept first
real-KL improvement) cuts polish wall time from ~10 min to ~50 s per
starting point with no loss of quality.  Budget creep (default 5%
total bits) lets polish swap precision (downgrade some units to free
budget for upgrading others) within a tight Pareto neighborhood.

### 4. Activation quantization matters at non-trivial assignments

OF's MVP supported weight-only perturbation; the v2 (commit `5ff5934`)
plumbs PerturbedActivationCache so OF measurements include activation
quant at any non-BF16 unit, matching the four-term collector and the
deployment-faithful surrogate.  Same code path as
`measure_assignment_kl`, just capturing logits instead of computing KL.

## Recommended pipeline (best quality)

For the lowest measured KL on a 4–14 bpp Pareto curve:

```
1. python -m prismaquant.measure_output_fisher \\
       --model ... --output of.json \\
       --formats NVFP4,MXFP8_E4M3,BF16 \\
       --use-frozen-weight-cache

2. python -m prismaquant.block_clado sweep \\
       --payload of.json --n-lambdas 61 --output sweep.json

3. python -m prismaquant.block_clado kneedle \\
       --payload of.json --sweep sweep.json \\
       --output-dir kneedle/ --n-neighbors 4

4. # Validate kneedle cone with real KL (validate-block-clado-kneedle.sh)

5. # Polish each validated candidate with steepest-first + budget creep
   #   (examples/launchers/polish-frontier.sh)

6. # Pick the lowest-KL polished result.
```

Expected: ~5 min wall time on Qwen3-0.6B, KL ~0.029 in the high-bpp
regime or KL ~0.087-0.092 in the 4.5-5 bpp regime.

For fixed-bpp shipping at the 4–5 bpp target: use four-term Block-CLADO
+ polish; OF alone has ~6% worse KL there because its analytic Fisher
form misses higher-order pair interactions that matter at large
quantization perturbations.

## Open work

1. **OF sandwich centering** (round 2 of OF).  At a non-trivial center
   ``w_c``, the Taylor expansion has ``KL_a − KL_c =
   ⟨p_c − p_t, δz_a⟩ + (1/2) Var_{p_c}(δz_a)``.  The pair Ω_ij still
   cancels to ``Cov_{p_c}(δz_a, δz_b)`` (linear terms vanish) but the
   unary needs both linear and quadratic terms.  Implementable; not yet
   done.
2. **Scaling**: 0.6B numbers don't predict 27B behavior.  Output-Fisher
   should win bigger as ``|𝔹|² · I²`` blows up; verify on Qwen 4B then
   27B.
3. **TC-42-style downstream evaluation**: the production lesson from
   the prior PrismaSCOUT 5.3095 ship was that lower mean-KL didn't
   translate to better tool-call safety.  Any new artifact needs the
   full eval before claiming a quality win.
