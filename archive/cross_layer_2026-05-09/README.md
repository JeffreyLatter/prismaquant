# Cross-Layer Archive

Archived on 2026-05-09 during the PrismaQuant next-phase cleanup branch.

This directory preserves cross-layer allocation and interaction machinery that
is useful for historical comparison and artifact replay, but is no longer part
of the current recommended production path.

Rationale:

- Reproducible production wins came from per-Linear knapsack, production GPTQ,
  scale sweep, activation clipping, sibling-global scales, and measured
  polish-of-1.
- Block-CLADO, adjoint L3, output Fisher, propagated cost, dense cone,
  PrismaSCOUT iteration, QUBO refinement, and polish-of-many did not provide
  reproducible non-regressive wins on the modern production stack.
- The next phase should measure targeted per-Linear improvements without these
  cross-layer entrypoints competing for attention.

Live replacements:

- Shared calibration helpers are in `prismaquant/calibration_data.py`.
- Shared decision-unit payload helpers are in `prismaquant/decision_units.py`.
- Production KL measurement helpers are in `prismaquant/kl_measurement.py`.
- The proven production polish entrypoint remains
  `prismaquant/polish_from_assignment.py`, backed by `prismaquant/polish.py`.

Contents:

- `adjoint_l3.py`, `adjoint_l3_frontier.py`, `adjoint_l3_screen.py`
- `measure_adjoint_l3.py`
- `block_clado.py`, `measure_block_clado.py`, `iterate_block_clado.py`,
  `validate_block_clado.py`
- `analyze_block_clado_iter.py`, `analyze_iterate.py`
- `dense_cone.py`
- `measure_interactions.py`, `interaction_refine.py`
- `measure_output_fisher.py`
- `propagated_cost.py`
- `iterate_perturbed_allocation.py`
- `quadratic_refine_allocator.py`
- `validate_polish_flips.py`
- `compare_payloads.py`, `merge_payloads.py`

Note: the production-safe single-flip implementation from the old
`coord_descent_polish.py` path was renamed to `prismaquant/polish.py` because
`polish_from_assignment.py` still depends on it. The inactive
`tests/test_coord_descent_polish.py` suite was removed with the archived
cross-layer tests.

The full phase rationale is in `docs/ultraplan-context-2026-05-09.md`.
