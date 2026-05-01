# L3 Propagated Cost Polish

L3 is an opt-in final pass on top of the converged perturbed-X/L2 allocation.
It does not replace the L2 fixed-point loop. It measures a small neighborhood
around the final L2 assignment, then re-solves only that measured neighborhood
while freezing every other Linear at its L2 pick.

## Cost Semantics

For each selected `(Linear, format)` candidate, L3 computes a paired local
propagation measurement:

1. Run the model with all non-target modules under the converged L2 assignment.
2. For the target Linear's paired baseline lane, leave that target in BF16.
3. For the candidate lane, apply the candidate format to the target input and
   target weights using the same RTN and activation-quantization semantics as
   the L2 perturbed-X hooks.
4. Compare candidate logits against that target-specific BF16 lane with end-KL.
5. Record downstream output MSE as a diagnostic.

The allocator uses only `propagated_end_kl` inside the L3 neighborhood. It does
not mix L3 end-KL with L2 `predicted_dloss` values. Unmeasured Linears are
frozen at their L2 choices.

The paired baseline is deliberately not a global BF16 model run. For target
`L`, the baseline is "target `L` BF16, all other modules at L2." This subtracts
the per-depth context and keeps the candidate costs local enough for the final
DP pass.

## Candidate Scope

`--l3-polish` selects a bounded Linear neighborhood:

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
- the usual final assignment and final layer config, using the polished
  assignment.

The summary reports selected Linears, measured count, KL before/after, flip
count, per-Linear flips, and whether `kl_after > kl_before`.

## Flags

```bash
python3 -m prismaquant.iterate_perturbed_allocation \
  ... \
  --l3-polish
```

Useful tuning flags:

- `--l3-uncertainty-rel-tol` default `0.10`;
- `--l3-min-fraction` default `0.05`;
- `--l3-max-fraction` default `0.30`;
- `--l3-safety-fraction` default `0.02`;
- `--l3-max-lanes-per-batch` default `16`;
- `--l3-tail-only` default on, with `--no-l3-tail-only` for full-forward
  fallback;
- `--l3-n-calib-samples` default `4`;
- `--l3-calib-seqlen` default `256`;
- `--l3-regression-tolerance` default `0.0`, rejecting polish output when
  validation KL regresses beyond the configured fraction of `kl_before`;
- `--frozen-dp-budget-tolerance` default `0.05`, allowing the L3 greedy
  fallback to use up to 5% of total target bits as hard budget slack.

L3 is one final pass by design; per-iteration L3 is not exposed yet.

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
