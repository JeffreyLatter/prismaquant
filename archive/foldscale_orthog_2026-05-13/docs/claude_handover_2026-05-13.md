# Claude Handover - PrismaQuant - 2026-05-13T18:15Z

This document is the current working handover for `/home/rob/prismaquant`.
Read `AGENTS.md` and `docs/design_guidelines.md` before changing code.

## Non-Negotiable User Preferences

- GPU-bound by default. Production probes, cache fills, recache, polish,
  export, KL validation, and serving smokes should bottleneck on GPU, not CPU
  or NVMe.
- Anything hot-path NVMe-bound is probably a failure to use the existing
  pre-cache/prefetch system.
- Do not add a parallel cache/residency mechanism. Extend
  `ProductionWeightCache`, `PerturbedActivationCache`, streaming prefetch, or
  current pipeline wiring.
- Keep PrismaQuant's promise: the right quantization for the right layer,
  selected empirically per Linear.
- Only production-promote formats and transforms that vLLM loads, generates
  correctly, and routes to performant kernels on representative shapes.
- Bits-per-parameter reports must exclude immutable/non-quantizable BF16
  regions such as `lm_head` and profile-pinned model components.
- Keep archived cross-layer machinery archived unless explicitly requested.

## Current User Request

The latest substantive engineering request before this handover was:

> "let's reset. Archive the work we've done on DQ++."

I started inspecting the DQ/DuQuant hooks but have not yet performed the
archive edits. If continuing from here, archive DQ++/DQ-fold cleanly:

1. Move the DQ-specific implementation and results into an archive directory,
   for example `archive/duquant_dqpp_2026-05-13/`.
2. Remove DQ/DuQuant from live pipeline flags, production cache levers,
   exporter help/detection text, docs, and active tests.
3. Keep shared infrastructure that is not DQ-specific, especially
   `prismaquant/render_score.py`, progressive render gates, AWQ,
   SmoothQuant, FourOverSix, Fisher-GPTQ, FP8/MXFP8 support, and vLLM smoke
   helpers.
4. Run targeted compile/tests after cleanup.

Do not interpret this as a request to delete AWQ, SmoothQuant, FourOverSix,
PrismaClip, Fisher-GPTQ, or the general render-scoring plugin framework.

## Dirty Worktree Snapshot

At handover time:

```text
 M docs/README.md
 M docs/design_guidelines.md
 M docs/runtime_flags.md
 M prismaquant/allocator.py
 M prismaquant/allocator_candidates.py
 M prismaquant/awq.py
 M prismaquant/build_production_cache.py
 M prismaquant/export_native_compressed.py
 M prismaquant/kl_sensitivity_probe.py
 M prismaquant/production_weight_cache.py
 M prismaquant/run-pipeline.sh
 M tests/test_allocator_shape_mask.py
 M tests/test_awq_v2.py
 M tests/test_prismaquant_export_native_compressed.py
 M tests/test_production_weight_cache.py
?? archive/respinquant_2026-05-13/
?? docs/duquant_fold_smoke_2026-05-13.md
?? docs/halo_4b_noclip_smoke_2026-05-13.md
?? docs/progressive_render_pipeline.md
?? docs/qwen36_27b_fp8_frontier_2026-05-13.md
?? prismaquant/duquant.py
?? prismaquant/render_score.py
?? tests/test_duquant.py
?? tests/test_render_score.py
?? tools/render_method_attribution.py
?? tools/vllm_prompt_smoke.py
```

Assume unrelated user/agent edits may exist. Do not revert broad changes.

## GPU Container

Host Python currently does not expose CUDA. Use the CUDA/vLLM container for
GPU work:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/rob:/home/rob \
  -w /home/rob/prismaquant \
  vllm-fresh-b12x-fla:latest \
  bash -lc "<command>"
```

Important: do not set `HOME=/home/rob` inside this container. That caused
Python to pick up host user-site packages and lose CUDA. Running as the normal
container user/root worked; if root-owned outputs are created, `chown -R
1000:1000 "$RUN"` afterward.

## Current Architecture Direction

The current live direction before the DQ reset:

- Production cache is the canonical render store.
- Progressive local render mechanisms are scored/gated by output MSE or
  Fisher-weighted output MSE through `prismaquant/render_score.py`.
- Local mechanisms are supposed to be accretive: if a candidate worsens the
  active local score, keep the previous rendered baseline and continue.
- HALO remains opt-in/research; global basis transforms are full-recipe arms,
  not local progressive steps.
- Layer-wise residual-basis methods that need runtime transition operators
  are not production-compatible unless a vLLM/plugin/kernel support decision
  is made.

## Important Implemented Pieces To Keep

### Progressive Render Scoring

`prismaquant/render_score.py` is general infrastructure and should remain live.
It provides:

- `RenderMechanismSpec`
- `register_render_mechanism`
- `resolve_render_mechanism_order`
- `score_render_error`
- `gate_render_candidate`

It supports deterministic ordering by operation phase and explicit
dependencies. Tests live in `tests/test_render_score.py`.

If DQ is archived, remove only the DQ builtin registration and DQ references
from tests/docs; keep the framework.

### AWQ

AWQ was rebuilt as a first-class fold-scale preconditioner. It is opt-in,
mutually exclusive with SmoothQuant, and integrated through
`ProductionWeightCache.awq_scales` for export-time predecessor folding.

Current user read: AWQ is useful and gave signal, but should remain gated/off
by default until full production validation clears.

### SmoothQuant

SmoothQuant is present as an opt-in fold-scale method. Results were mixed:

- `.8B` no-PrismaClip smoke improved KL:
  baseline `0.201933401`, SmoothQuant `0.149363996`.
- `4B` no-PrismaClip smoke regressed:
  baseline `0.069056227`, SmoothQuant `0.076682127`.
- `4B` layer isolation showed one attention group helped and another hurt:
  layer 29 q/k/v improved to `0.063870477`, layer 28 q/k/v worsened.
- A later full-stack result looked very bad, but attribution suggested the
  main regression came from unrelated unstable NVFP4 PrismaClip rerenders.

Conclusion: keep opt-in/research with stricter full downstream gates. It
should probably be evaluated as transform-aware allocation candidates, not as
post-hoc mutation of a fixed assignment.

### FourOverSix

FourOverSix is a first-class NVFP4 scale-rule plugin. It is valuable and cheap
relative to PrismaClip. Keep it live. It participates in local gates and in
package acceptance with downstream GPTQ/scale-sweep.

### PrismaClip / PrismaFisherClip

PrismaClip exists but is production-disabled in `run-pipeline.sh` unless
`PRISMACLIP_RESEARCH_OVERRIDE=1`. Reason: 27B top-32 run stayed GPU-bound but
was too slow to be useful. RBC mode is disabled pending investigation.

The current user direction before DQ was to leave PrismaClip disabled for
major runs unless explicitly researching it.

### Fisher-GPTQ / Fisher Output MSE Allocator

Fisher-weighted GPTQ and Fisher-weighted allocator objective use h-detail
row weights (`g2_per_token`). The user wants quality metrics that are less
calibration-fragile than direct KL but still closer to final-logit impact than
raw weight MSE.

Keep these live. Do not rename them "GuidedQuant allocator"; describe the
metric by what it is.

### FP8/MXFP8 Format Work

Recent work expanded/adjusted format support and shape gating. User wants BF16
included as fallback, but wants MXFP8/FP8 available where vLLM supports them
and shape constraints are known up front in the optimizer, not discovered as
export-time coercions.

## DQ++ / DQ-Fold State To Archive

DQ-fold is not full DuQuant++. It is a no-runtime, fold-only microscale
preconditioner inspired by DuQuant++ that searches per-channel scales and
folds them through predecessor normalization. It deliberately does not
implement runtime block rotations.

Live DQ-specific files/references found by search:

- `prismaquant/duquant.py`
- `tests/test_duquant.py`
- `docs/duquant_fold_smoke_2026-05-13.md`
- DQ hooks in `prismaquant/production_weight_cache.py`
- DQ option/defaults in `prismaquant/run-pipeline.sh`
- DQ lever defaults in `prismaquant/kl_sensitivity_probe.py`
- DQ text in `prismaquant/build_production_cache.py`
- DQ fold-scale detection text in `prismaquant/export_native_compressed.py`
- DQ registration and dependencies in `prismaquant/render_score.py`
- DQ docs rows in `docs/runtime_flags.md`
- DQ references in `docs/progressive_render_pipeline.md`
- DQ-specific assertions in `tests/test_render_score.py`,
  `tests/test_awq_v2.py`, and
  `tests/test_prismaquant_export_native_compressed.py`

Recommended archive contents:

- `prismaquant/duquant.py`
- `tests/test_duquant.py`
- `docs/duquant_fold_smoke_2026-05-13.md`
- A README explaining:
  - DQ-fold was a partial/no-runtime approximation, not full DuQuant++.
  - Cheap pre-GPTQ proxy accepted too many folds and regressed KL.
  - Full downstream gating improved `.8B` but added complexity/cost.
  - User chose to reset/archive rather than keep it live.

After moving those files, remove the live hooks listed above.

## DQ-Fold Measurements For Archive README

`.8B` model:

- source: `/home/rob/.cache/huggingface/qwen35-0p8b-bf16`
- assignment:
  `/home/rob/dq-runs/qwen35-0p8b-progressive-gates-v2-20260512T224854Z/artifacts/layer_config.json`
- baseline cache run:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-baseline-20260513T171330Z`
- bad cheap-proxy run:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-smoke-20260513T170840Z`
- full-gate DQ run:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-fullgate-20260513T171847Z`

Bad cheap-proxy result:

- accepted `47/48` groups, `100` linears
- local proxy MSE improved, but KL regressed
- n=16/seqlen=512 KL: baseline `0.2012080634`, DQ `0.2273565184`
  (`+13%` regression)

Full downstream gate result:

- accepted `25/48` groups, `54` linears
- local score improvement `0.61%`
- cache fill cost: baseline `165.1s`, DQ `352.7s` (`~2.1x` on `.8B`)
- n=16/seqlen=512 KL: baseline `0.2012080634`, DQ `0.1355714227`
  (`-32.62%`)
- n=64/seqlen=512 KL: baseline `0.2182590449`, DQ `0.1916450168`
  (`-12.19%`)
- same bpp: `4.966982922201138`
- exported artifact:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-fullgate-20260513T171847Z/exported`
- vLLM loaded with `FlashInferCutlassNvFp4LinearKernel`
- prompt smoke generated coherent Paris text

Critical DQ bug that was fixed before archive:

- `_render_awq_scaled_for_cache` had been scoring NVFP4 `compute_only`
  pre-pack tensors.
- Identity candidates could score as exact BF16 when GPTQ was disabled.
- The fix was to score the actual packed/dequantized NVFP4 tensor via
  `enc._rtn_dequant_nvfp4(result["_w_dq"], group_size=16,
  global_real_override=joint_global_real)`.

If DQ is later revisited, preserve that lesson.

## ReSpinQuant State

ReSpinQuant work was already archived under:

```text
archive/respinquant_2026-05-13/
```

Reason: layer-wise residual-basis rotation is not viable for vanilla
deployment without runtime transition support. Do not revive it in the live
pipeline unless the user explicitly asks for a vLLM/plugin/kernel path.

There may be a stale vLLM plugin import error for `prismaquant_residual_adapter`
from that attempt. It was nonfatal during a vLLM smoke, but should be cleaned
later if convenient.

## HALO State

HALO is still workable but paused. It is a global/no-runtime rotation path
when conditions are met, not a local progressive plugin. The user asked to
pause HALO several times while exploring other mechanisms.

Observed `.8B` run suggested HALO did most of the work and reduced other
local mechanisms to smaller perturbations, but this was not promoted. Keep it
opt-in/research unless a full recipe arm validates on the target calibration
contract.

## 27B / Pareto Context

Recent 27B work focused on shape-aware FP8/MXFP8/BF16 candidate selection and
export at target bpp. User specifically wanted:

- BF16 included as fallback.
- MXFP8/FP8 included only where dimensions and vLLM kernels support them.
- Constraints applied in allocator/optimizer, not caught by exporter coercion.
- More Pareto points around 4.85 to 5.25, including 5.05 export.
- MTP/toolevalbench after materialization.

Do not assume those long runs completed unless logs prove it. Search
`docs/qwen36_27b_fp8_frontier_2026-05-13.md` and `dq-runs/` if needed.

## Current Verification Known Good

After DQ work, this passed:

```bash
python3 -m py_compile prismaquant/*.py
python3 -m pytest \
  tests/test_duquant.py \
  tests/test_awq_v2.py \
  tests/test_production_weight_cache.py \
  tests/test_prismaquant_export_native_compressed.py \
  -q
```

Result at that time: `120 passed, 15 warnings`.

After archiving DQ, replace `tests/test_duquant.py` with the relevant
non-DQ targeted tests:

```bash
python3 -m py_compile prismaquant/*.py
python3 -m pytest \
  tests/test_render_score.py \
  tests/test_awq_v2.py \
  tests/test_production_weight_cache.py \
  tests/test_prismaquant_export_native_compressed.py \
  -q
```

If production-cache code is touched, also run at least one tiny GPU smoke in
the vLLM container before claiming runtime correctness.

## Suggested DQ Archive Edit Plan

Use `apply_patch` for manual edits. Prefer `git mv` for archiving files.

1. Create `archive/duquant_dqpp_2026-05-13/README.md`.
2. Move:
   - `prismaquant/duquant.py`
   - `tests/test_duquant.py`
   - `docs/duquant_fold_smoke_2026-05-13.md`
3. In `production_weight_cache.py`:
   - remove `from prismaquant.duquant import ...`
   - remove `_DUQUANT_FOLD_FORMATS`
   - remove `_duquant_fold_format_supported`
   - remove `_solve_duquant_scales`
   - keep `_render_fold_scaled_for_cache` only if AWQ/SmoothQuant still need
     it; if kept, remove DQ wording from its docstring.
   - remove `levers.setdefault("duquant", False)`
   - fold-scale mutual exclusion should be only `awq` and `smoothquant`
   - remove DQ from `enabled_mechanisms`
   - remove DQ activation-aware format additions and qname activation logic
   - remove DQ metadata and solving block
   - remove DQ from `awq_scale` condition; AWQ/SmoothQuant formats should
     still work.
4. In `render_score.py`:
   - remove DQ builtin registration
   - remove DQ from `after=(...)` dependencies for PrismaClip/FisherClip
5. In `run-pipeline.sh`:
   - remove `DUQUANT` env default and case block
   - remove `_duquant` cache tag
   - remove DQ from validated-surrogate error text
   - remove `DUQUANT=$DUQUANT` from config echo
6. In `kl_sensitivity_probe.py`, remove the DQ lever default.
7. In `build_production_cache.py`, remove DQ from `--enable` help.
8. In `export_native_compressed.py`, remove DQ from fold-scale help/detection
   text and `_production_cache_fold_scale_enabled`.
9. In tests:
   - delete/move `tests/test_duquant.py`
   - update render-score order test to use `awq` or `smoothquant` only
   - update mutual-exclusion test to assert AWQ/SmoothQuant only
   - update export cache fold-scale detection test to AWQ/SmoothQuant only
   - update AWQ mutual-exclusion test levers to only AWQ/SmoothQuant
10. In docs:
   - `docs/runtime_flags.md`: remove DQ rows.
   - `docs/progressive_render_pipeline.md`: remove DQ from current order and
     replace the DQ section with a short archive pointer.

## Caution Points

- `ProductionWeightCache.awq_scales` is a historical name used for both AWQ
  and SmoothQuant fold scales. Do not rename casually; export depends on it.
- `render_score.py` is not DQ-specific. Keep it.
- Do not introduce CPU fallback for quantization. If CUDA is missing in host
  Python, run the container.
- Do not use exporter coercion as a format support strategy. Optimizer and
  allocator should know format legality before selecting candidates.
- Avoid broad `rg /home/rob/dq-runs` without scoping; the tree is large and
  has permission-noisy generated outputs.

## Disk Notes

`/home/rob` was about 95% used but still had roughly 90GB free during this
work. DQ runs that can be cleaned later if the user approves:

- Bad DQ smoke:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-smoke-20260513T170840Z`
- Baseline/DQ fullgate runs are useful for archive provenance; avoid deleting
  until the archive README is written.

## Recommended Next Engineering Direction After DQ Archive

After DQ is archived, the clean path is:

1. Stabilize the live progressive pipeline around AWQ, FourOverSix,
   Fisher-GPTQ, scale-sweep, and shape-aware FP8/MXFP8/BF16 candidates.
2. Keep PrismaClip research-only unless a cheaper candidate strategy is
   implemented.
3. If revisiting SmoothQuant, make it transform-aware in candidate generation:
   evaluate identity+format and SmoothQuant+format packages before allocation,
   not after a fixed assignment.
4. For attention q/k/v, use group-level validation/gating because per-Linear
   output MSE can miss Q/K softmax interaction.
5. For full DuQuant++ or similar runtime rotations, build an optional vLLM
   plugin repo first and measure pure PyTorch correctness/performance before
   considering kernels.

