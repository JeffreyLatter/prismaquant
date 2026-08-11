# Requested: CBL results on K44–K47 (locate the crossover)

**Robert Tand, 2026-08-10:** *"I'd like to see cbl results on k44-k47."*

This is a **second work item**, additional to the export-path fix in your
original brief. Read the constraint in §3 before planning any of it.

## 1. Why these rungs

`cost-ldlq/transfer-study-fable-verify/F1_GENERALIZATION.md` measured the CBL
ladder and found a crossover it did not localize:

| rung | poolb vs lattice | verdict |
|---|---|---|
| K28 | −18.7 .. −25.8% | GO |
| K33 | −40 .. −41% | GO |
| K38 | −47.2 .. −57.0% | GO |
| **K43** | **−39 .. −41%** | **GO** |
| **K44–K47** | **UNMEASURED — this request** | unknown |
| K48 | **+54 .. +98%** | **NO-GO** |

The doc states plainly: *"My crossover prediction was wrong: the boundary sits
between K43 and K48."* So the boundary is somewhere in K44–K47 and nobody has
looked.

**Mechanism to test, not assume:** below palette saturation, codebook
*placement* dominates and pooled Lloyd wins at matched bytes; above it, per-row
*scale precision* (the sweep's job) dominates and no book placement compensates.
K48 is 4×4096-entry. The question is where the two curves cross.

## 2. Measure both pooling arms — they behave differently at the boundary

Do **not** measure `poolb` alone. At K48 the doc records:

- `poolb`: +54..+98% median — fails outright.
- `poolr`: *"rescues median (+10..+21%, down_proj −5..−8%) but tails reach
  +37%..+145% — fails every adoption criterion."*

So the two arms cross at *different* places, and `poolr` degrades via its
**tails** while its median still looks acceptable. **Report per-expert tail
quantiles, not just medians** — a median-only readout would have called K48
`poolr` a marginal GO, which the tails refute. This is the single most likely
way to get the answer wrong.

## 3. HARD CONSTRAINT — this is GPU work and the GPU is not available

A production export (`pq-dsv4-export`) is running and holds the GPU at ~93%
with 8 GB CUDA allocations on the expert stacks. On this box (one DGX Spark,
128 GB *unified* memory shared between GPU and host) a second CUDA consumer
OOMs the machine, and that has historically required a hard reboot.

**Therefore:**
- **Do NOT run any part of this sweep now.** Prepare it only.
- Deliver a **ready-to-run driver script** plus a written cost estimate, and
  stop. The coordinator will launch it once the export finishes.
- The script must acquire `dq-runs/dsv4-quality-hybrid/sfd-analysis/gpu_run.sh`'s
  flock mutex (or an equivalent atomic lock) rather than polling `nvidia-smi` —
  polling is check-then-act and two workers have raced into an OOM before.
- It must be resumable/checkpointed per cell, so a kill does not lose the sweep.

## 4. Design it to match the existing study, or the numbers are not comparable

Reuse the banked harness rather than writing a new evaluator. Match
`F1_GENERALIZATION.md`'s design exactly:

- **Cells:** layers {14, 21} × {gate, up, down} = 6 families, × rungs
  {K44, K45, K46, K47} = **24 cells**, × arms {poolb, poolr}.
- **All 256 experts** per cell (the study's design; do not sample).
- Certify against the **production evaluator** to the same ≤1.2e-08 bar the
  study used, and report the achieved certification residual.
- **LDLQ out of the measurement loop** — per Rob's 2026-08-05 decision and
  `LDLQ_CBL_VERDICT.md`: the table metric is a diagonal weight-domain MSE and is
  structurally blind to LDLQ's off-diagonal benefit, so including it makes the
  comparison uninterpretable. Keep the arm fixed.

## 5. Cost estimate to produce (do not guess — derive it)

Known timings from the study: Lloyd per book 0.33 s (K28) / 1.85 s (K38) /
9.9 s (K48-class); CBL cell (Lloyd + assign + measure) 6.4 s (K28), 33 s (K38),
~10 s (K33), ~85 s (K43), scaling with entry count. K44–K47 sit between K43 and
K48-class, so derive from entry-count scaling and state the total GPU-minutes
for all 24 cells × 2 arms. Report it **before** anything runs.

## 6. What to report

1. The driver script path and how to launch it.
2. Derived GPU cost for the full sweep.
3. The adoption criteria you will apply (median **and** tail bar), stated
   *before* seeing data, so this is pre-registered rather than fitted.
4. Anything in `F1_GENERALIZATION.md` that turns out not to match the code when
   you check it.

**Do not run it. Prepare and report.**
