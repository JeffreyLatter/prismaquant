# Codebase consolidation plan (2026-05-15)

Target stack after consolidation: **gptq** (with damp_sweep sub-feature) +
**jso** + **scale_sweep** + the main **PrismaQuant solver** (probe → cost →
allocator → production cache build → native compressed-tensors export →
KL validate). Everything else moves to `archive/`.

Per user direction:
- Docs stay in place ("they may be useful").
- CLADO already lives in `archive/cross_layer_2026-05-09/`; do not disturb.
  The residual `decision_units.py` API is load-bearing — keep it.
- Polish is **not** in the run-pipeline.sh path (only mentioned in a comment).
  Archive it.

## Archive (full file moves)

| Path | New home |
|---|---|
| `prismaquant/halo.py` | `archive/halo_2026-05-15/prismaquant/halo.py` |
| `tests/test_halo.py` | `archive/halo_2026-05-15/tests/test_halo.py` |
| `prismaquant/polish.py` | `archive/polish_2026-05-15/prismaquant/polish.py` |
| `prismaquant/polish_from_assignment.py` | `archive/polish_2026-05-15/prismaquant/polish_from_assignment.py` |

## Strip from production code

### HALO (9 production files)

`production_recache.py`, `production_weight_cache.py`, `build_production_cache.py`,
`validate_assignments_kl.py`, `tools/stage_untied_lm_head.py`, `incremental_probe.py`,
`export_native_compressed.py`, `render_score.py`, `run-pipeline.sh`.

Strip: `--halo-mode`, `--halo-seed` CLI args; HALO_MODE/HALO_SEED env vars;
`halo` register_render_mechanism call; `halo` lever wiring.

### Fisher-weighted GPTQ + Fisher output-MSE allocator (8 production files + 2 tests)

`validate_assignments_kl.py`, `incremental_measure_quant_cost.py`,
`production_weight_cache.py`, `allocator_candidates.py`, `kl_sensitivity_probe.py`,
`build_production_cache.py`, `render_score.py`, `measure_quant_cost.py`,
`tests/test_validate_assignments_kl.py`, `tests/test_production_weight_cache.py`.
Also `run-pipeline.sh` FISHER_WEIGHTED_GPTQ + FISHER_OUTPUT_MSE_ALLOCATOR.

Strip: `fisher_gptq` lever, `fisher_row_weights` parameter plumbing,
`PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR` env var, h-detail-dir wiring,
related register_render_mechanism calls.

### SAO (validated useless this session, 6 production files + 3 tests)

Same files as JSO: strip `static_act_order` lever + `_select_nvfp4_joint_gptq_eff_scale`
the SAO branch + tests + `tools/lift_gptq_ablation.py` SAO arms.

### awq_round (4 production files + 2 tests)

Strip `awq_round` lever + `_activation_weighted_round_nvfp4` function +
register_render_mechanism call + tests.

### PrismaClip dead helpers (already walled off, finish cleanup)

`production_weight_cache.py` (~50 hits): `_is_prismaclip_format`,
`_prismaclip_rescale_candidates`, `_prismaclip_activation_distribution_stats`,
`_prismaclip_activation_split`, `_prismaclip_row_weight_split`.

`kl_sensitivity_probe.py` (~48 hits): `_measure_prismaclip_variant_gate`
and its call site at line 3022, `PRISMACLIP_FORMAT` imports/uses,
`_prismaclip_candidate_enabled` arg threading.

`export_native_compressed.py`: dead `_rescale_clipped_activation_matrix`,
`_normalize_act_clip_rescale`, `_resolve_act_clip_quantile` (keep
`PRISMAQUANT_ACT_CLIP_QUANTILE` env default if anything still reads it for
plain GPTQ; otherwise delete). `act_clip_threshold`, `act_clip_rescale`
parameter plumbing.

`tests/test_production_weight_cache.py`: 13 PrismaClip-specific test
functions identified earlier.

## Stage order (commit between stages)

1. **Wait for scale_sweep ablation, decide its fate.** If dropped, also strip.
2. Archive full-file moves (HALO, polish).
3. Strip non-core levers from production files (HALO + Fisher refs, SAO, awq_round).
4. Clean up dead PrismaClip helpers.
5. Run test suite; fix any breaks.
6. Final commit.

Each stage: small commit + syntax check + targeted test run.

## Borderline items left in place

None now — user has cleared HALO, Fisher, Polish, SAO, PrismaClip,
awq_round. The remainder of `prismaquant/` is core: probe / cost /
allocator / cache / exporter / kl-validate / model_profiles / format
plumbing.
