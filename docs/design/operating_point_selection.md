# Operating-point selection on a log-linear frontier

**Status: PROPOSAL (2026-08-18).** Nothing in the pipeline changes defaults on
this document. The analysis half is implemented (`tools/operating_point.py`);
the anchor half is a decision recorded here for adoption or amendment.

## The finding

The measured held-out KL frontier is log-linear across the entire useful
range. Qwen3.8-27B dense, 11 validated points, 4.50 -> 8.25 bpp
(`qwen38-27b-arm-b/artifacts/validated_frontier_selection.json`):

    log10(KL) = -0.1133 - 0.2357 * bpp      R^2 = 0.9948
    per-bpp exchange rate: x1.72 KL per +1 bpp (x1.31 per +0.5)
    tail: kl_max slope -0.2394/bpp -- tracks the mean on this model
    saturation: NONE in range; the 8.25 point still sits ON the line

Residuals never exceed +-8%. The surrogate's fit on the same sweep
(log10(dloss) = 4.533 - 0.294*bpp, R^2 0.9943) told the same story in a
different currency.

## What a log-linear frontier means

An exponential D(b) = D0 * e^(-lambda*b) is **scale-free**: the relative
return -dD/D per bpp is the constant lambda. Every bpp in the band buys the
same x1.72 KL improvement. Three consequences:

1. **No interior optimum exists on the curve.** "Bang for buck" is constant;
   there is no point where returns diminish *relatively*.
2. **Every knee is an axis artifact.** On this very sweep, kneedle's
   log-error spelling picks 4.75 and its raw-linear spelling picks 6.00 --
   a 1.25 bpp gap from the same tool on the same data. This closes the
   kneedle demotion permanently: it is not noisy, it is *ill-posed*.
3. **Any pick encodes an external anchor.** The only honest procedure is to
   name the anchor explicitly and measure it.

## Anchors considered

**A. Task-parity saturation (RECOMMENDED as the default when no card is
named).** KL is unbounded-resolution and never saturates; the metric
ladder's top -- bounded task metrics (ToolEvalBench hardmode, GSM8K, IFEval,
p99-NLL) -- does. "Bang" measured at the top of the ladder genuinely stops
at BF16 parity: bits past parity buy quality no sanctioned metric can see.
The pick rule: **ship the cheapest bpp whose task suite is statistically at
BF16 parity**, with parity defined against the suite's own measured
replicate noise (BF16-vs-BF16 reruns), not a chosen epsilon.

The fitted frontier makes this a 2-3 measurement procedure instead of a
sweep: the KL fit is the interpolator, task runs are the oracle.

  1. Run the validated sweep as today; fit log10(KL) = alpha - beta*bpp
     (`tools/operating_point.py`). Require R^2 >= 0.99; below that the
     regime assumption failed -- stop and look.
  2. Measure the suite's replicate noise once (BF16 vs BF16).
  3. Start from the family prior (Qwen3.6-27B evidence: every artifact at
     KL <= 0.03 landed inside TEB noise of BF16 -- 0.0292 -> TEB 88 vs 90,
     0.0151 -> 85 vs 86, ~0.009 -> 91 vs 86; the KL 0.0475 artifact landed
     at the bottom, TEB 84. Prior bracket: KL_parity in [0.03, 0.05]).
     Convert with the fit: on Qwen3.8 that bracket is **bpp 5.04 -> 5.98**.
  4. Bisect with real suite runs on rendered artifacts (2-3 total; the
     already-published point is the free first probe). Smallest parity bpp,
     to +-0.25 bpp, is **b_parity**.
  5. Ship `min(b_parity, card budget)`. Never above b_parity. Below it,
     print the priced tradeoff on the card: each -0.5 bpp = x1.31 KL =
     -1.5 GB (this model). KL_parity is a *measured, per-model* threshold;
     the 0.03 folklore is a prior, never the answer.

  Tail discipline (principle 4): the parity suite includes p99-NLL and the
  adversarial tool-call set. On Qwen3.8 the tail tracks the mean
  (kl_max slope -0.239 vs mean -0.236); where it does not, the tail curve
  gets its own fit and the anchor tests both.

**B. Card budget** (existing rule, unchanged): when a card is named,
`TARGET_DISK_GB` fits it exactly -- but A caps it: past b_parity, spend the
bytes on KV/context instead.

**C. Measurement-envelope saturation B\*** (valid, rarely binding): smallest
bpp indistinguishable from the 8-bit reference within the publication
reproducibility envelope. On Qwen3.8 the curve never flattens by 8.25, so
B* > 8 -- not binding inside 4.75-6.5. It binds on easy models; keep it as
the ceiling check the sweep gets for free.

**D. Serving economics** (per-deployment refinement, not model-agnostic):
on a fixed-memory box, +1 bpp displaces its bytes of KV cache; the card
should print the exchange chain (bpp <-> GB <-> KL-multiplier <-> context
tokens) so a deployer prices their own utility. This is a reporting duty,
not a pick rule.

**E. Iso-byte family exchange** (the cross-model question): at fixed file
bytes, is 27B@6.0 better than 35B-A3B@4.6? This is the user's actual
download decision and the literature's "most bang per byte" (Dettmers &
Zettlemoyer 2023 land it near 4-5 bpw for naive quantizers, drifting lower
as the quantizer improves). It needs a cross-model metric (tasks/PPL, never
KL-vs-own-BF16) and is a separate study, not a per-model pick rule.

**F. Allocation-gain maximum** (REJECTED as a pick rule): b maximizing the
DP's advantage over the naive mixing chord. Interior and constant-free, but
axis-dependent in the same way kneedle is (absolute gain peaks ~5.9, ratio
gain ~7.0 on the same data) and it measures *our* value-add, not the user's
value. Diagnostic at most.

## Bounds of the band

The 4.75 floor and ~6.5 ceiling are not derived optima and should not be
dressed as such. The floor is where the menu closes (the assignment
degenerates toward uniform NVFP4 and the allocator has nothing left to
choose); the ceiling is where FP8-uniform becomes the honest artifact. The
anchor above picks *within* the open band; on current evidence it lands at
**~5.0-6.0 on Qwen3.8-27B**, pending the two task-suite runs that turn the
prior into a measurement.
