# L3 Propagated Cost Polish

L3 is an opt-in final pass on top of the converged perturbed-X/L2 allocation.
It does not replace the L2 fixed-point loop. In the default selective mode it
measures a small neighborhood around the final L2 assignment, then re-solves
only that measured neighborhood while freezing every other Linear at its L2
pick. In global mode it measures every eligible Linear candidate and runs one
joint DP over the full measured set.

## Cost Semantics

For each selected `(Linear, format)` candidate, L3 computes a paired local
propagation measurement:

1. Run the model with all non-target modules under the converged L2 assignment.
2. For the target Linear's paired baseline lane, leave that target in BF16.
3. For the candidate lane, apply the candidate format to the target input and
   target weights using the same RTN and activation-quantization semantics as
   the L2 perturbed-X hooks.
4. Compare candidate logits against that target-specific BF16 lane with end-KL.
5. Optionally record downstream output MSE as a diagnostic.

The allocator uses only `propagated_end_kl` inside the L3 neighborhood. It does
not mix L3 end-KL with L2 `predicted_dloss` values, and downstream output-MSE
is opt-in because the allocator does not consume it. Unmeasured Linears are
frozen at their L2 choices.

The paired baseline is deliberately not a global BF16 model run. For target
`L`, the baseline is "target `L` BF16, all other modules at L2." This subtracts
the per-depth context and keeps the candidate costs local enough for the final
DP pass.

## Candidate Scope

`--l3-polish --l3-mode selective` selects a bounded Linear neighborhood:

- uncertain allocator-neighborhood choices, using `--l3-uncertainty-rel-tol`;
- current L2 picks that are not the cheapest available measured format,
  ranked by expected L2 flip benefit;
- a small safety set of high L2 predicted-cost Linears;
- fill to `--l3-min-fraction` when too few Linears qualify;
- `--l3-max-fraction` is a hard cap on the combined neighborhood. L3 fills
  that cap with ranked non-cheapest current picks first, then uncertain choices,
  then safety picks.

For each selected Linear, L3 measures the current L2 pick, one cheaper format,
one more accurate format, and BF16 when available. The final DP is run only
over candidates with measured L3 costs and a measured current-format candidate.

`--l3-mode global` uses the same per-Linear format filter but skips selection
entirely: every eligible Linear is measured under the converged L2 baseline, and
a single global DP optimizes all measured Linears using propagated end-KL. This
is the cleaner algorithmic path because it removes selection coverage gaps and
multi-flip interaction from the selective frozen-DP polish. It is also more
expensive, so selective remains the default until larger smoke data justifies a
default switch.

`--l3-measure-all-formats` widens every selected Linear to the full format menu
instead of trusting the L2-neighbor filter. This costs more lanes, but is the
right mode when L2 may have chosen a poor starting format and L3 needs to search
the full marginal bit frontier.

## Sparse Interaction Refinement

`--l3-interaction-refine` adds a bounded interaction-aware stage after the
additive L3 DP proposal and before the normal validation/rollback gate. It keeps
the measured propagated end-KL values as unary costs, selects a small set of
high-impact Linear units near the DP proposal, measures paired override KL for
format combinations among those units, and feeds the residuals into the sparse
quadratic local refiner.

The residual for a pair is:

```text
paired_KL(i=a, j=b) - unary_KL(i=a) - unary_KL(j=b)
```

where `paired_KL` uses the same target-BF16 paired baseline as L3 unary
measurement: both targets are BF16 in the paired baseline lane, while all other
modules remain at the current assignment. This makes the stage a local
CLADO/CoopQ-style correction to additive DP rather than a separate global KL
objective.

Useful bounds:

- `--l3-interaction-top-units` default `8`;
- `--l3-interaction-neighbor-radius` default `1`;
- `--l3-interaction-max-pairs` default `0`, meaning uncapped within the selected
  frontier;
- `--l3-interaction-max-passes` default `4`;
- `--l3-interaction-exact` default enabled, with
  `--l3-interaction-exact-max-states` default `2000000`.

When the selected option product is below the exact-state cap, the refiner
enumerates the measured local quadratic problem exactly under the bit budget.
Larger frontiers fall back to bounded greedy single/pair moves. This gives
medium-size diagnostic runs a clear interpretation: they are locally optimal for
the measured unary and pairwise residual model, even though they are not a
global proof over every layer or higher-order interaction.

The final assignment still goes through full validation. If the interaction
refiner proposes a worse assignment, the existing L3 regression handling rolls
it back.

## Hook Ordering

L3 installs the normal perturbed-X context hooks first, but excludes the target
modules in the current depth microbatch. It then installs L3 target hooks for
those excluded modules.

The target hooks apply lane-specific behavior:

- the target's paired baseline lane stays BF16;
- the target's candidate lane applies the candidate format;
- lanes belonging to other targets in the same depth group apply that module's
  original L2 format.

This avoids double-quantizing the target input while preserving "all other
modules at the L2 assignment" for every lane.

## Outputs

When enabled, `iterate_perturbed_allocation.py` writes:

- `l3_propagated_costs.pkl`;
- `l3_polish_summary.json`;
- the usual final assignment and final layer config, using the accepted
  assignment after validation rollback.

The summary reports L3 mode, selected or total Linears, measured count, KL
before/after, flip count, per-Linear flips, and whether `kl_after > kl_before`.

Multi-budget runs add:

- `pareto_curve.csv`;
- `pareto_curve.json`;
- `final_layer_config_bpp_X.XX.json` for each requested target;
- a `pareto` field in `summary.json`.

Knee-search runs also write:

- `knee_assignment.json`;
- `final_layer_config_knee.json`;
- a `knee` field in `summary.json` describing the selected point.

## Flags

```bash
python3 -m prismaquant.iterate_perturbed_allocation \
  ... \
  --l3-polish
```

Useful tuning flags:

- `--l3-mode {selective,global}` default `selective`;
- `--l3-uncertainty-rel-tol` default `0.10`;
- `--l3-min-fraction` default `0.05`;
- `--l3-max-fraction` default `0.30`;
- `--l3-safety-fraction` default `0.02`;
- `--l3-max-lanes-per-batch` default `64`;
- `--l3-tail-only` default on, with `--no-l3-tail-only` for full-forward
  fallback;
- `--l3-measure-all-formats` default auto for wide multi-budget spans and off
  for single-budget runs;
- `--l3-output-mse` default off; enables downstream output-MSE diagnostics that
  are not used by the allocator;
- `--l3-n-calib-samples` default `4`;
- `--l3-calib-seqlen` default `256`;
- `--l3-regression-tolerance` default `0.0`, rejecting polish output when
  validation KL regresses beyond the configured fraction of `kl_before`;
- `--frozen-dp-budget-tolerance` default `0.05`, allowing the L3 greedy
  fallback to use up to 5% of total target bits as hard budget slack.

CUDA graph capture is opportunistic by default. `PRISMAQUANT_L3_CUDA_GRAPHS`,
`PRISMAQUANT_COORD_LANE_CUDA_GRAPHS`, and `PRISMAQUANT_KL_CUDA_GRAPHS` accept
`auto`/unset, `1`, or `0`. Auto mode avoids graphing one-shot flip batches
because capture warmup dominates on the Qwen 4B PrismaSCOUT L3 workload.
`PRISMAQUANT_COORD_REPLAY_CACHE` is opt-in for the same reason: it can reduce
tail forwards, but the current cache path copies too much baseline model state
for large dense checkpoints.

On UMA systems, paired L3 interaction measurement can consume host-visible
GPU memory that is not reflected in container RSS. Set
`PRISMAQUANT_L3_MIN_HOST_MEM_GB` to make L3 check `/proc/meminfo`
`MemAvailable` between paired-override chunks and raise
`GPUMemoryBudgetExceeded` before the host reaches OOM pressure. The
paired-override loop also reduces the next chunk's lane count as it approaches
that floor. The memory-aware 4B-class cap currently uses up to 6 interaction
lanes after empirical tests showed 4 lanes safe, 8 lanes unsafe, and 6 lanes
viable on the GB10 diagnostic machine.

L3 is one final pass by design; per-iteration L3 is not exposed yet.

## Multi-Budget And Knee Search

`--target-bits-list` evaluates several budgets in one process:

```bash
python3 -m prismaquant.iterate_perturbed_allocation \
  ... \
  --l3-polish \
  --target-bits-list 4.5,5.0,5.5,6.0,6.5
```

The list mode clusters targets by `--target-bits-share-tolerance` (default
`0.25` bpp). Each cluster anchor runs a full L2 convergence and one L3
measurement; targets inside the cluster reuse that anchor's L3 costs for DP and
then run their own validation KL. By default, the anchor uses the bounded
`--l3-mode selective` neighborhood and the local format filter, so multi-budget
and knee runs remain time-bounded. Pass `--l3-mode global` and/or
`--l3-measure-all-formats` explicitly when you want the older exhaustive
behavior.

`--knee-search` adaptively samples the Pareto curve:

```bash
python3 -m prismaquant.iterate_perturbed_allocation \
  ... \
  --l3-polish \
  --knee-search \
  --knee-bpp-min 4.0 \
  --knee-bpp-max 8.0
```

The default `--knee-mode kneedle` starts with endpoints plus midpoint, scores
the normalized validation-KL curve by distance to the endpoint chord, and
refines the largest interval adjacent to the current best knee until
`--knee-tolerance` or `--knee-max-evaluations` stops it. Threshold mode instead
uses bisection to find the lowest bpp satisfying
`--knee-threshold-kl`.

The hard runtime bound is the product of evaluated Pareto points, L3 selected
linears, selected formats per linear, L3 iterations, validation-scout limits,
and any explicit interaction-refine caps. The output metadata records these
scope controls alongside the Pareto curve so a knee result is auditable.

`--quality-equivalent-bits` makes the threshold explicit in "bit-equivalent"
terms. It first evaluates the requested reference bpp, then sets:

```text
threshold_kl = KL(reference_bpp) + quality_equivalent_kl_slack
```

It then searches for the lowest bpp in
`[--quality-equivalent-bpp-min, --quality-equivalent-bits]` whose validated KL
meets that threshold:

```bash
python3 -m prismaquant.iterate_perturbed_allocation \
  ... \
  --l3-polish \
  --quality-equivalent-bits 8.0 \
  --quality-equivalent-bpp-min 4.0 \
  --quality-equivalent-kl-slack 0.01
```

The output still writes `pareto_curve.csv`, `pareto_curve.json`, and
`summary.json`. The summary includes the reference KL, threshold KL, chosen
quality-equivalent bpp, and a best-effort knee candidate computed from the
evaluated points.

The Qwen 4B smoke launcher exposes the final-pass path through:

```bash
L3_POLISH=1 examples/launchers/run-perturbed-x-smoke.sh
```

## Compute Notes

Depth grouping batches candidate lanes from the same decoder layer and compares
each candidate with its own BF16 target baseline lane in the same forward. This
improves launch overhead and numerical pairing, but FLOPs still scale with the
number of selected candidate lanes. Keep the uncertainty filter tight for large
models.
