# Dead launcher archive (walled 2026-07-30)

These 14 launchers under `examples/launchers/` are **unrunnable on any current
checkout**: every `python -m prismaquant.<module>` they invoke names a module
that no longer exists in the tree. They are preserved here as the record of how
the block-CLADO / adjoint-L3 / coordinate-polish research lanes were actually
driven — not as anything a reader should run.

Verified 2026-07-30 (`scratch/doc-consolidation-2026-07-30/cleanup/orphans.md`
§3.1–§3.2): none of `iterate_block_clado`, `measure_block_clado`, `block_clado`,
`validate_block_clado`, `measure_output_fisher`, `dense_cone`,
`polish_from_assignment`, `coord_descent_polish`, `measure_adjoint_l3`,
`adjoint_l3_frontier` exists under `prismaquant/`.

## §3.1 — every python invocation is a removed block-CLADO-family module (8)

| File | Dead invocation |
|---|---|
| `best-quality-pipeline.sh` | `iterate_block_clado` |
| `iterate-block-clado.sh` | `iterate_block_clado` |
| `iterate-block-clado-full-of.sh` | `iterate_block_clado` |
| `iterate-qwen3-4b-low-bpp.sh` | `iterate_block_clado` |
| `launch-qwen3-0p6b-block-clado.sh` | `measure_block_clado`, `block_clado` |
| `launch-qwen3-0p6b-output-fisher.sh` | `measure_output_fisher`, `block_clado` |
| `sandwich-block-clado.sh` | `measure_block_clado`, `block_clado` |
| `validate-block-clado-kneedle.sh` | `validate_block_clado` |

## §3.2 — six more, equally dead (adjoint-L3 / dense-cone / polish lanes)

| File | Dead invocation |
|---|---|
| `dense-knee-search-4b.sh` | `dense_cone`, `validate_block_clado` |
| `dense-pareto-pinned-4b.sh` | `dense_cone`, `validate_block_clado`, `polish_from_assignment` |
| `polish-4p9-prodfaithful-4b.sh` | `polish_from_assignment` |
| `polish-block-clado-best.sh` | `coord_descent_polish` |
| `launch-qwen3p6-35b-a3b-adjoint-l3.sh` | `measure_adjoint_l3` |
| `launch-qwen3p6-35b-a3b-adjoint-l3-frontier.sh` | `adjoint_l3_frontier` |

## Deliberately NOT moved

`proto-production-prismascout.sh`, `dense-knee-search-prodfaithful-0p6b.sh` and
`dense-knee-search-prodfaithful-4b.sh` are **partially** dead — each still
invokes live modules (`build_production_cache`, `allocator`,
`kl_sensitivity_probe`) alongside removed ones — so they stay live pending a
per-stage decision (orphans.md §3.3).

The block-CLADO method write-ups live at `docs/archive/block_clado/`; the
prior launcher wall is `archive/legacy_frontiers_2026-05-06/launchers/`.
