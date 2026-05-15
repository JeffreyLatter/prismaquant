# Documentation Map

Current docs in this directory describe the live PrismaQuant implementation:

- `design_guidelines.md`: mandatory design rules for new functionality,
  including GPU-first execution, cache reuse, vLLM gates, and measurement
  discipline.
- `runtime_flags.md`: environment controls used by production runs.
- `v1_milestone_validation.md`: pre-tag V1 validation checklist with exact
  commands and required log artifacts.
- `progressive_render_pipeline.md`: shared local render scoring, mechanism
  ordering, and extension contract for new numerical methods.
- `calibration_diverse_v1.md`: fixed mixed-domain calibration recipe.
- `validation_harness.md`: validation harness behavior and CLI notes.
- `v20_memory_and_scheduling.md`: memory scheduling notes.
- `vectorization_refactor.md`: vectorization implementation notes.

Historical frontier experiments, cross-layer allocators, ReSpinQuant residual
basis work, archived render methods, handovers, and Block-CLADO result logs live in
`archive/legacy_frontiers_2026-05-06/`,
`archive/cross_layer_2026-05-09/`, and
`archive/respinquant_2026-05-13/`.
