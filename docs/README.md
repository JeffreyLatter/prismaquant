# Documentation Map

Current docs in this directory describe the live PrismaQuant implementation:

- `design_guidelines.md`: mandatory design rules for new functionality,
  including GPU-first execution, cache reuse, vLLM gates, and measurement
  discipline.
- `runtime_flags.md`: environment controls used by production runs.
- `progressive_render_pipeline.md`: shared local render scoring, mechanism
  ordering, and extension contract for new numerical methods.
- `qwen36_27b_fp8_frontier_2026-05-13.md`: BF16-inclusive 27B frontier
  rerun adding plain `FP8_E4M3` for skinny linear-attention shapes.
- `calibration_diverse_v1.md`: fixed mixed-domain calibration recipe.
- `halo_27b_attempt_2026-05-09.md`: invalidated HALO 27B attempt and
  production-cache guardrail.
- `gpu_validation_2026-05-09.md`: GPU serving smoke results for recache,
  MTP, and graph-mode validation.
- `validation_harness.md`: validation harness behavior and CLI notes.
- `v20_memory_and_scheduling.md`: memory scheduling notes.
- `vectorization_refactor.md`: vectorization implementation notes.
- `halo/design.md`: HALO design notes.

Historical frontier experiments, cross-layer allocators, ReSpinQuant residual
basis work, handovers, and Block-CLADO result logs live in
`archive/legacy_frontiers_2026-05-06/`,
`archive/cross_layer_2026-05-09/`, and
`archive/respinquant_2026-05-13/`.
