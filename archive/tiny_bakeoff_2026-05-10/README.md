# Tiny Bakeoff Archive

Archived on 2026-05-10 during the PrismaQuant surface-area cleanup.

This directory preserves the old tiny-model bakeoff, local reconstruction,
oracle-search, and empirical allocator-calibration CLIs. They were useful for
early method development, but they are no longer part of the live production
path.

Rationale:

- The orchestrator in `tiny_bakeoff.py` still invoked cross-layer entrypoints
  (`measure_interactions.py`, `quadratic_refine_allocator.py`) that were
  already archived on 2026-05-09.
- The current production path is per-Linear and GPU-faithful:
  `incremental_probe -> incremental_measure_quant_cost -> allocator ->
  kl_sensitivity_probe -> production_weight_cache/production_recache ->
  polish_from_assignment -> export_native_compressed`.
- Direct KL validation now lives in `prismaquant/validate_assignments_kl.py`
  and shared KL measurement helpers live in `prismaquant/kl_measurement.py`.
- Leaving broken research CLIs importable from `prismaquant/` made the live
  surface look larger and more supported than it is.

Contents:

- `tiny_bakeoff.py` — old one-command small-model regression bakeoff.
- `bakeoff.py` — JSON decision summary for tiny bakeoff outputs.
- `local_reconstruct.py` — local per-layer reconstruction/refinement tool.
- `calibrate_allocator.py` — empirical per-format gain fitter used by the
  old bakeoff frontier.
- `oracle_search.py` — exact local search for tiny interaction neighborhoods.
- `refinement_units.py` — grouping helpers used by the archived local
  reconstruction/oracle tools.

The active tests for these modules were removed from `tests/` because archive
contents are preserved for reference and replay, not maintained as live
package APIs.
