# MSE-driven post-frontier promotion — archived 2026-07-30

**Kill order.** Architecture re-vet 2026-07-30, proposal **R18** (Lens 6 P4),
accepted by Robert as part of the wave-2 mandate: *"implement ALL proposals."*
See `docs/audits/architecture_re-vet_2026-07-30.md` §R18 and
`docs/ARCHITECTURE.md` §11.

**Ledger check (the one condition R18 attached to this wall):** no shipped run
directory carries a `layer_config_before_mse_promotion.json`. Verified before
the move — CLEAR. No shipped artifact rode this lever.

## What it was

`MSE_PROMOTION=1` was a **post-frontier assignment rewrite**: after
`select_validated_frontier` picked a measured-KL point, it re-read the local
`output_mse` costs, grouped them by serving unit, ranked by
`output_mse_per_bit`, and promoted the worst attention/linear-attention groups
to `BF16` (or another target format) until a bpp target or delta was consumed.
It ran only under `SELECTION_MODE=validated-surrogate` **and**
`PRODUCTION_CACHE=1` **and** an explicit target/delta — three conditions, with a
documented silent no-op outside them (CODEX M31).

Its one real evaluation is `docs/results/qwen36_35b_mse_promotion_phase1_2026-05-25.md`:
on the 35B it removed 86% of stored local MSE and landed KL 0.0898 / PPL 9.81 —
it beat the strategic baseline but **lost to both the shipped 4.75 artifact and
the 5.16 kneedle**.

## Durable lesson

**A post-allocator rewrite that re-ranks by a local surrogate cannot beat the
allocator that already solved the global problem against a faithful cost.** The
promotion ran *after* the measured-KL frontier point was selected, so every
flip it made was chosen by `output_mse_per_bit` — a local screen — and then
shipped without a real-KL re-decision. This is the "post-allocator rewrite"
pattern principle 1 vetoes, and it is the same failure the graveyard records
for `scale_sweep` (re-picking scales after the loop that compensated for them)
and for L3-polish-of-many-DP (a batch of individually-measured flips applied
together).

The correct answer to "structurally sensitive regions need protection" turned
out to be **a better cost, not a rewrite**: AURA's KL-adjoint × rendered-dW
cost (−38% KL @4B, −17.9% @27B) plus empirical packed-expert unit-KL puts the
bits where they belong *inside* the DP, so there is nothing left to promote
afterwards. `docs/README.md` has filed the Phase-1 result HISTORICAL —
superseded by AURA-on-MoE — since that lane shipped.

Secondary lesson, on test mass: 564 lines of test behind a default-off,
superseded lever is not evidence of importance. Test mass tracks how hard
something was to get right, not whether it should still exist.

## Contents

- `prismaquant/mse_promotion.py` (633 L) — `build_mse_promotion_assignment`
  (the rewrite) and `build_promotion_candidate_report` (the candidate-ranking
  report consumed by the propagated-sensitivity tooling, itself walled the same
  day under `archive/l3_propagated_2026-07-30/`).
- `tools/build_mse_promotion_assignment.py` (144 L) — the CLI the pipeline
  invoked.
- `tests/test_mse_promotion.py` (564 L) — the full behavioural suite, kept as a
  record of the semantics (bpp budgeting, serving-unit grouping, packed-expert
  skipping, `_bits_delta`). It imports the walled
  `prismaquant.propagated_sensitivity_costs` and is **not** runnable from here
  without restoring both walls.

## Live-tree state after the wall

`run-pipeline.sh` no longer defines the seven `MSE_PROMOTION*` variables, the
`SELECTION_MODE`/`PRODUCTION_CACHE` precondition gate, or the stage-4/4-C′
rewrite block. Setting `MSE_PROMOTION` to a non-zero value now **fails fast
with `exit 2`** pointing here (the §3.5 archived-lever gate convention).
`tests/test_run_pipeline_defaults.py::test_mse_promotion_is_archived` pins the
gate.

## To resurrect

Move `prismaquant/mse_promotion.py`, `tools/build_mse_promotion_assignment.py`
and `tests/test_mse_promotion.py` back, restore the `MSE_PROMOTION*` defaults,
the precondition `case` block and the stage-4/4-C′ rewrite in
`run-pipeline.sh`, and delete the `exit 2` gate. Note that the test file also
requires `archive/l3_propagated_2026-07-30/prismaquant/propagated_sensitivity_costs.py`
to be restored.
