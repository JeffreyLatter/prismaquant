# Fisher Archive (Fisher-weighted GPTQ + Fisher output-MSE allocator)

Archived 2026-05-15 per user direction: "kill the fisher-weighted gptq and
fisher output-mse allocator. Archive."

## What was archived

- **Fisher-weighted GPTQ** (`FISHER_WEIGHTED_GPTQ=1`,
  `--enable ...,fisher_gptq` lever): per-token Fisher weights from
  incremental_probe's h-detail directory weighting GPTQ row residuals.
- **Fisher output-MSE allocator** (`FISHER_OUTPUT_MSE_ALLOCATOR=1`,
  recent commits b44332e + 4836ecf): allocator objective using
  Fisher-weighted output-space MSE per Linear instead of the KL
  surrogate. Required `fisher_output_mse` entries in cost.pkl from
  incremental_measure_quant_cost.

## Where the code lives now

The Python implementations (~50 `fisher_row_weights` parameter sites,
`measure_quant_cost.py`'s Fisher cost computation,
`allocator_candidates.py`'s Fisher mode, `kl_sensitivity_probe.py`'s
h-detail wiring, `production_weight_cache.py`'s row-weighting plumbing,
etc.) remain in the live tree as dead code paths. The production
pipeline cannot reach them: `run-pipeline.sh` errors out fast if
`FISHER_WEIGHTED_GPTQ` or `FISHER_OUTPUT_MSE_ALLOCATOR` is set, and
rejects `fisher_gptq` in `PRODUCTION_CACHE_LEVERS`.

A subsequent cleanup sweep can excise the dead surface — keeping it
out of the live tree was deferred because it is woven through too many
call signatures to remove safely in one pass.

## To resurrect

Re-enable the env-var paths in `run-pipeline.sh` (remove the archive
guard at the top of the file and restore the legacy `case` blocks for
`FISHER_OUTPUT_MSE_ALLOCATOR` and `FISHER_WEIGHTED_GPTQ`).
