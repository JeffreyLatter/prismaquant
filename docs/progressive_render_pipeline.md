# Progressive Render Pipeline

PrismaQuant local numerical methods should be accepted by the same local
render score used for allocator-style output error:

1. Render the current baseline weight for a Linear or fused-sibling group.
2. Render the candidate after one mechanism, or a candidate package when a
   mechanism can only be useful as an initializer for downstream refinement.
3. Score both on the same activation rows.
4. If h-detail/Fisher rows are available, use Fisher-weighted output MSE.
   Otherwise use activation output MSE. Weight MSE is only a fallback when
   activations are unavailable.
5. Accept the candidate only when the score improves by the configured
   minimum gain. If it regresses or ties, keep the baseline and continue to
   the next mechanism.

The shared implementation lives in `prismaquant/render_score.py`.
Production-cache renders store decisions under
`cache.metadata["render_gates"]`; FourOverSix also has a compact
`cache.metadata["four_over_six"]` summary because it is a first-class plugin.

## Ordering

Mechanisms declare what kind of operation they perform. The production cache
resolves the order from those declarations rather than from the text order in
an environment variable.

Current local order:

```text
activation candidate:      PrismaClip or PrismaFisherClip
NVFP4 scale rule:          FourOverSix
rounding objective:        Fisher-weighted GPTQ
rounding solver:           GPTQ
codebook scale refine:     scale_sweep
```

AWQ, SmoothQuant, and BlockOrtho-G were archived on 2026-05-13 under
`archive/foldscale_orthog_2026-05-13/` and are no longer registered render
mechanisms. The production cache and pipeline reject their levers so they do
not enter the shipping path by accident.

HALO is excluded from this local sequence because it is a global basis
transform; evaluate it as a separate full-recipe arm.

FourOverSix is a first-class NVFP4 scale-rule plugin. The production cache
tests `static_6` against `four_over_six_mse` directly, then also lets
FourOverSix participate in GPTQ/scale-sweep packages. This catches the case
where FourOverSix alone is neutral or negative, but FourOverSix plus a
downstream rounding/scale refine improves the active score.

MXFP8 scale-sweep uses the same progressive gate: if the activation-aware
candidate regresses the active score, the MXFP8 baseline render is kept.

The gate can be disabled only for debugging with
`PRISMAQUANT_RENDER_PROGRESSIVE_GATES=0`. The minimum relative gain is
`PRISMAQUANT_RENDER_GATE_MIN_GAIN` and defaults to `0.0`.

## Extension Contract

New local mechanisms should register a `RenderMechanismSpec` with:

- a stable `name`;
- an `operation` class;
- a `scope` such as `linear`, `fused_sibling_group`, or `nvfp4_block`;
- a numeric `phase`;
- the `gate_metric`;
- optional `after` / `before` dependencies;
- optional `exclusive_group`.

Then the mechanism's candidate renderer should call `score_render_error()` and
`gate_render_candidate()` for accept/reject. This keeps new numeric methods
easy to include, toss, or reorder without adding one-off scoring logic.
