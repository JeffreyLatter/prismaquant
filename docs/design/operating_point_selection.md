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

**A. Offline parity certificate (RECOMMENDED as the default when no card is
named).** The anchor is "statistically the same as BF16", and the oracle must
be numerical, deterministic, and analyzable offline -- task suites are out
(TEB is non-deterministic and needs a serving loop). Rev 2 (2026-08-18)
replaces the task-suite oracle with statistics computed from the full-vocab
teacher/student distributions the KL evaluator already materializes on the
canonical held-out contract. No model runs at analysis time; the only live
work is the validated sweep the pipeline performs anyway, plus a ONE-TIME
calibration of the measurement system itself.

The statistical core: with paired per-draw measurements, a null-hypothesis
test will call ANY bpp "different from BF16" given enough tokens -- excess
loss is strictly positive. Equivalence therefore needs a margin, and the only
non-arbitrary margins are measured properties of the shop's own measurement
system:

  * **The publication envelope** (primary): the cross-session / cross-stack
    reproducibility of the gold numbers themselves (the class of drift the
    house already documents: KL readings moving 4-8x across sessions until
    provenance-pinned, +-0.7 % on the pinned gridbook gold lane, conf-KL
    +-17 % under a CUDA-extension change). A quality difference smaller than
    what the shop could reproducibly CLAIM is, by its own epistemics, not a
    claimable difference. Measured once per model+stack by re-running the
    gold evaluator on one fixed artifact across sessions/serving modes.
  * **The serving stack's own BF16 nondeterminism** (for the behavioral
    leg): eager-vs-graph and session-to-session argmax flips of the BF16
    model at near-tie positions -- real, accepted by every deployment, and
    the exact meaning of "a different seed of the same model".

The certificate: parity(b) holds when, on the canonical contract
(n=8x512, --calib-repeats >= 4, held-out draws), ALL of:

  1. **Mean leg**: paired excess NLL (student minus teacher on the same
     draws) is inside the publication envelope of the BF16 PPL reading.
  2. **Tail leg** (principle 4): p99 per-prompt excess NLL and kl_max inside
     the same envelope scaled to their own reference readings.
  3. **Behavior leg**: greedy top-1 flip rate vs the teacher <= the BF16
     self-flip rate across sanctioned schedules (eager/graph, cross-session),
     per-sequence paired sign test.

Pick `b_parity` = the smallest bpp on the fitted line whose measured point
passes; the log-linear fit interpolates between sweep points, so localizing
to +-0.25 bpp costs at most one extra rendered point, and usually zero.

Plumbing this needs (small, evaluator-side): persist `teacher_nll_*`
alongside the existing `nll_*` fields; log per-position argmax agreement
(the evaluator already holds both full-vocab distributions); measure the
one-time envelope + BF16 self-flip reference per model+stack. All three land
in `validate_assignments_kl` output, so `tools/operating_point.py` can test
the certificate offline from JSON alone.

Task suites stay where the metric ladder already puts them: the promotion
backstop for materialized artifacts (a certificate pass that regresses
ToolEvalBench still demotes), never the pick oracle.

Current placement on Qwen3.8-27B, prior pending the envelope measurement:
the Qwen3.6-family evidence (KL <= 0.03 landing inside task noise of BF16)
converts through the fit to **bpp 5.04-5.98**; the certificate's own
threshold replaces that prior the first time the envelope is measured.

For the model card, the detection-budget reading of KL is worth printing
regardless of the anchor: a likelihood-ratio distinguisher needs roughly
3/KL tokens of output to separate the artifact from BF16 at 95 %
(KL 0.03 -> ~100 tokens; KL 0.008 -> ~370). It prices what a KL number
MEANS without another measurement.

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
**~5.0-6.0 on Qwen3.8-27B**, pending the one-time envelope measurement that
turns the family prior into this model's own threshold.
