# PrismaQuant Codebase Audit — 2026-05-10

Scope: reduce live repository surface after the cross-layer archive while
preserving the current production path.

## Method

- Built an import graph for live `prismaquant/*.py` modules, excluding
  `archive/` and vendored transformer code.
- Scanned exact duplicate function bodies with `ast` normalization.
- Searched every candidate module for non-test references before archiving.
- Added or extended focused tests before consolidating helpers.
- Treated CLI entrypoints with no internal importers as live unless they are
  broken by previously archived dependencies or clearly research-only.

## Archived

Moved to `archive/tiny_bakeoff_2026-05-10/`:

- `tiny_bakeoff.py`
- `bakeoff.py`
- `local_reconstruct.py`
- `calibrate_allocator.py`
- `oracle_search.py`
- `refinement_units.py`

Rationale: `tiny_bakeoff.py` still invoked cross-layer modules already
archived on 2026-05-09 (`measure_interactions.py`,
`quadratic_refine_allocator.py`). The cluster was no longer a runnable live
workflow, and the current production loop validates assignments directly with
KL measurement rather than tiny frontier/oracle bakeoffs.

Removed the active tests for those archived APIs:

- `tests/test_prismaquant_bakeoff.py`
- `tests/test_prismaquant_local_reconstruct.py`
- `tests/test_prismaquant_tiny_bakeoff.py`

Trimmed `tests/test_prismaquant_native_math.py` so it no longer imports the
archived calibration CLI while keeping allocator and format-registry coverage.

## Consolidated

- Added `prismaquant/layer_config.py` as the shared layer-config parser.
  Export, production re-cache, and assignment-KL validation now use the same
  canonicalization path.
- Added `prismaquant/incremental_shards.py` for shared shard pickle annotation
  used by incremental probe and incremental cost measurement.
- Added `env_truthy` and `model_device` to `prismaquant/memory_management.py`
  and removed local copies from cache/re-cache/session/kernel-guard modules.
- Reused `memory_management` env parsing in `kl_measurement.py` instead of a
  second local implementation.
- Removed the duplicated local production-cache safe-path wrappers.

## Tests Added

- `tests/test_layer_config.py`
- `tests/test_incremental_shards.py`
- New `memory_management` tests for strict truthy parsing and meta-parameter
  device detection.

## Kept Live

Modules with no internal importers after cleanup are CLI entrypoints or
intentional standalone tools:

- `allocator`
- `build_production_cache`
- `collapse_config_groups`
- `incremental_measure_quant_cost`
- `kl_fisher`
- `kl_sensitivity_probe`
- `model_profiles.validate`
- `multi_chunk_probe`
- `polish_from_assignment`
- `validate_assignments_kl`
- `validate_native_export`
- `validate_quantized_model`
- `validation_harness`

These should stay live for now because they are production pipeline stages,
validators, or documented one-off maintenance tools.

## Remaining Duplication

The remaining exact duplicates are mostly intentional:

- `init_empty_weights` fallback stubs in loader/export modules.
- no-op model-profile methods.
- repeated tiny closure methods generated inside graph/cache classes.
- model-profile stubs whose repetition keeps each architecture readable.

Potential future cleanup, if it becomes worth the churn:

- Factor the two `device` CLI parsers in polish/KL validation.
- Move `_assignment_digest` into a tiny assignment utility module.
- Review whether `collapse_config_groups.py` is still needed once every
  shipped artifact has the exporter-side collapsed target generation.
