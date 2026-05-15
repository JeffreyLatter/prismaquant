# Pluggable Refactor Contract

This refactor separates four concerns that were previously easy to couple:

1. **Model structure** lives in `prismaquant/model_profiles/specs/*.json`.
   These specs describe live-to-recipe naming, source/vLLM naming,
   fused-sibling groups, packed experts, passthrough prefixes, and pinned
   tensors. They may also name the default serving profile for the model
   family. `ModelProfile` still owns executable behavior, but low-risk profile
   answers now read these specs.

2. **Serving/runtime constraints** live in
   `prismaquant/serving_profile_specs/*.json`. These specs describe backend
   format menus and kernel shape rules such as MXFP8/NVFP4 alignment. The
   allocator asks `serving_profiles.py`; it should not branch on a hardcoded
   vLLM profile id.

3. **Pipeline artifacts and gates** live in `prismaquant/pipeline.py`.
   `PipelineSpec` describes stages, typed artifacts, metric gates, and cache
   ownership. The current production flow is exposed by
   `default_production_pipeline_spec()` as a contract first. `run-pipeline.sh`
   writes the configured contract to
   `${WORK_DIR}/artifacts/pipeline_spec.json` before launching the GPU-bound
   stages, so each run records the render mechanisms, target profile, formats,
   and cache policy it executed.

4. **Optimization units** come from `ModelGraph.optimization_units()`.
   This is the typed replacement for scattered coupling rules: individual
   tensors, fused-sibling groups, and packed expert groups are explicit units
   with constraints.

## Cache Rule

Rendered weights remain owned by `ProductionWeightCache`. Activation replay
remains owned by `PerturbedActivationCache` or the established streaming
activation cache. Pipeline validation rejects alternate owners for those
resources so new plugins cannot accidentally introduce a parallel cache.

## Adding a Component

Add model-specific naming/decomposition in a model-structure spec. Add
backend legality in a serving-profile spec. Add local numerical transforms as
render mechanisms using `render_score.py` and gate them with `MetricGateSpec`.
Only then wire the component into production execution, reusing
`ProductionWeightCache` / `PerturbedActivationCache`.

When adding a multimodal or MoE model, keep source-checkpoint naming and export
emission naming separate. `ModelProfile.source_tensor_name()` is used for
source lookup and shape audits; `ModelProfile.export_tensor_name()` is used for
keys PrismaQuant writes. They are usually identical, but Gemma 4 keeps export
expert keys in recipe form because vLLM performs its own `.experts` to
`.moe.experts` remap during loading.
