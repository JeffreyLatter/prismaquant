# Multi-shot Recalibration Archive

This archive preserves the implementation, tests, design doc, and validation
artifacts for the multi-shot per-Linear cost recalibration loop, an attempt
to capture cross-Linear interaction in the PrismaQuant allocator using the
LLM-Surgeon §3.5 (arXiv:2312.17244) cheap variant.

**The technique did not pay off on Qwen3-4B and is not in the production
path.** Both hypotheses we tested came back negative:

1. *Does multi-shot improve over the production-calibration baseline?*
   No. 5/5 budgets converged at shot 2 with ΔKL = exactly 0 — the
   allocator's shot-1 pick under production-cal cost numbers is already a
   local optimum that recache does not move.
2. *Does small-cal + 2-shot recover production-cal quality?*
   No. Cross-evaluation at production-grade KL showed mean -42.5% gap
   closed; one budget regressed by -154% (multi-shot made things worse
   than just running small-cal once).

The implementation is preserved here for reference; the production
pipeline (`prismaquant/run-pipeline.sh`) errors fast if
`MULTI_SHOT_PASSES>1` is set, pointing back to this directory.

## Contents

```
prismaquant/multi_shot.py        — the recalibration module + CLI
tests/test_multi_shot.py         — unit tests (7 tests, all pass on the archived code)
docs/multi_shot_recalibration.md — design doc with the full negative-result writeup
validation/orchestrator.sh       — reusable Phase-0/1 validation harness
validation/analyze.py            — paired-Δ analyzer for kl_comparison.json files
validation/{small,production,low_budget}_cal_analysis.txt
                                 — the actual measurement outputs
validation/calefficiency_summary.json
                                 — small-cal-vs-production-cal cross-eval summary
```

## Measurement summary

### Phase 0/1 — initial small-cal validation (N=8 T=512 × 4 repeats)

3/3 budgets showed directional KL improvement (mean -8.3%, best -16.6% at
6.0 bpp, z=-1.86). This *looked* like real signal at the time, was
documented as such, and motivated the production-cal retest. In retrospect
it was noise correction at the surrogate cost level — see below.

### Production-cal validation (N=32 T=1024 × 4 repeats)

| Budget | Baseline KL ± σ | Multi-shot KL ± σ | ΔKL |
|--------|------------------|--------------------|------|
| 4.70   | 0.13577 ± 0.036  | 0.13577 ± 0.036    | 0    |
| 4.85   | 0.13750 ± 0.023  | 0.13750 ± 0.023    | 0    |
| 5.00   | 0.09764 ± 0.005  | 0.09764 ± 0.005    | 0    |
| 5.50   | 0.10062 ± 0.024  | 0.10062 ± 0.024    | 0    |
| 6.00   | 0.09819 ± 0.014  | 0.09819 ± 0.014    | 0    |

All 5 budgets converge at shot 2 — `layer_config.sha256(shot_2) ==
layer_config.sha256(shot_1)`. The recalibrated cost numbers shift only
~0.6–1.2% on the surrogate Δloss, which is below the DP's tolerance, so
the optimal assignment is unchanged. ΔKL is *exactly* 0 (byte-identical
layer_configs), not "indistinguishable from 0 within noise."

### Calibration-efficiency cross-evaluation (small-cal assignments at production-cal eval)

| Budget | Prod-cal base | Small-cal base | Small-cal 2-shot | Gap-to-prod closed |
|--------|----------------|------------------|--------------------|---------------------|
| 5.00   | 0.0976         | 0.1423           | 0.1314             | +24% |
| 5.50   | 0.1006         | 0.1191           | 0.1474             | **-154%** (multi-shot regressed below baseline) |
| 6.00   | 0.0982         | 0.1239           | 0.1235             | +2% |

Mean gap-closed: **-42.5%**. The 95% assignment-similarity observation
that motivated this cross-eval was misleading; the 5% of Linears that
differ between small-cal 2-shot and production-cal baseline shift KL in
unpredictable directions, and the 5.5 case shows multi-shot can actively
hurt at production-grade evaluation.

## Why it failed

The LLM-Surgeon-derived hypothesis was that per-Linear costs become stale
once many Linears change, and that iterative recalibration captures
cross-Linear interaction. Empirically, on PrismaQuant's Qwen3-4B allocator
with {NVFP4, MXFP8_E4M3, BF16} at production calibration:

- The per-Linear cost surrogate already captures the dominant signal at
  production-cal accuracy. The allocator's shot-1 pick is a local optimum
  that survives recalibration.
- The cross-Linear-coupling penalty the surrogate "misses" is below the
  DP's resolution at this format menu × budget × model scale.
- The L3-polish failure mode (`prismaclade_l3_non_additivity`) was about
  incremental refinement of an already-optimal base, not about whether
  the base allocator itself benefits from recalibration. Multi-shot
  attacks a different problem.

## Where multi-shot *might* still help (untested)

These regimes were not tested and would each need their own validation:

1. Larger models (27B, 35B+) where cost-noise behavior may differ.
2. Richer format menus (NVINT3, MXFP6, …) — more allocator decisions per
   Linear could expose interactions invisible to the 3-format menu.
3. Different objectives (perplexity, downstream task accuracy) — KL may
   be insensitive to interactions perplexity catches.
4. Different model families with more correlated Linear sensitivities.

## Important measurement-discipline lesson

Single-rep KL on Qwen3-4B at our calibration sizes is dangerously noisy.
The Phase 0/1 run's first attempt at N=8 T=512 × 1 rep showed a +10%
*regression* at 5.5 bpp that completely flipped to a -5.2% *win* with 4
calibration repeats. The "wins" we saw at small-cal eval were also a
small-cal-eval artifact — the production-grade cross-eval contradicts the
small-cal-eval direction at 5.5 (where it was a -5.2% small-cal-eval
"win" but a +23.8% production-eval regression).

**Always run `--calib-repeats ≥ 4`** for differential KL on this model
scale. Single-rep differential measurements at any calibration size are
not trustworthy.

## To revive

Copy the contents out of this archive into the production tree and remove
the `MULTI_SHOT_PASSES` guard from `prismaquant/run-pipeline.sh`. Then
re-run the validation harness with whatever new regime motivates the
revival (larger model, richer format menu, different objective). Do not
re-promote without measured KL evidence that clears the seed-noise floor
at production calibration AND survives held-out evaluation.
