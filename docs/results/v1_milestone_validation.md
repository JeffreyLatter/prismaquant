> **HISTORICAL (2026-07-30) — the V1 release gate, kept as a record of what
> was run for the V1 tag.** The live release-gate summary is
> `docs/ARCHITECTURE.md` §7. Do not run the recipe below verbatim; the body is
> left untouched, and these are its known-stale points:
>
> | Claim in body | Current truth |
> |---|---|
> | `FORMATS=NVFP4,MXFP8_E4M3,FP8_E4M3,BF16` | default is `NVFP4,FP8_DYNAMIC,BF16` (`run-pipeline.sh:45`). MXFP8 is de-menued — exact-scale FP8 Pareto-dominates it. `FP8_DYNAMIC` is an alias of `FP8_E4M3` (`format_registry.py:142`), so that rung is unchanged in substance. |
> | `PRODUCTION_CACHE_LEVERS=gptq,joint_scale_opt` | default adds `static_act_order`: `gptq,static_act_order,joint_scale_opt` (`run-pipeline.sh:178`). |
> | no `COST_MODE` in the environment block | `COST_MODE` exists and defaults to `production-render-score` (`run-pipeline.sh:187`); accepted values `local\|production-render-score\|production-render-staged\|aura`. The V1 run predates the variable, so it is implicitly the `local` era. |
> | `SELECTION_MODE=validated-surrogate` exported | still valid but no longer the default — `surrogate` is (`run-pipeline.sh:250`); validated-surrogate is opt-in. |
> | serving smokes are the whole gate | the numeric ship gate is `validate_quantized_model.py:116-120` (PPL 25 / mean-NLL 3 / worst-NLL 6 / MTP p0 0.60), which this checklist never invokes; the gold-lane n=8×512 full-vocab KL and 8192-token PPL contracts live in `tools/measure_vllm_full_kl.py:461-462` and `tools/measure_vllm_wikitext_ppl.py:78-79`. |
>
> Checked and still true: `artifacts/format_applicability.json` is produced on
> every allocator run (`allocator.py:1624-1628,1693`, written beside
> `--pareto-csv`), as are `pareto_assignments/manifest.json` and the
> `validated_frontier_*.json` pair under `SELECTION_MODE=validated-surrogate`
> (`run-pipeline.sh:1057,1192,1287`).

# V1 Milestone Validation Gate

Use this checklist before tagging the current V1 state or branching V2. The
goal is to prove the live production recipe, not archived research paths.

## Scope

V1 production recipe:

- calibration: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- local render levers: `gptq,joint_scale_opt`
- allocator selection: per-Linear candidate costs plus measured frontier
  validation
- production cache: `ProductionWeightCache` with resident prefetch required
- export: native compressed-tensors
- serving gate: vLLM eager and graph-mode prompt smokes

Archived research levers must stay off. The V1 gate validates only the live
production recipe listed above.

## Local Checks

Run from the repository root:

```bash
mkdir -p /home/rob/dq-runs/v1-milestone/logs
python3 -m compileall -q prismaquant tests tools
python3 -m pytest -q 2>&1 | tee /home/rob/dq-runs/v1-milestone/logs/pytest_cpu.log
```

The CPU suite must pass. Skips are acceptable for GPU/vLLM-only tests, but
failures are not.

## Full Pipeline

Use a fresh work directory for the milestone candidate:

```bash
export MODEL_PATH=/path/to/source-model
export WORK_DIR=/home/rob/dq-runs/v1-milestone
export DATASET=/home/rob/dq-runs/calibration/diverse-v1.jsonl
export NSAMPLES=32
export SEQLEN=1024
export TARGET_BITS=4.75
export FORMATS=NVFP4,MXFP8_E4M3,FP8_E4M3,BF16
export TARGET_PROFILE=vllm_packed_moe
export SELECTION_MODE=validated-surrogate
export VALIDATED_FRONTIER_PICK=kneedle
export PRODUCTION_CACHE=1
export PRODUCTION_RECACHE=1
export PRODUCTION_CACHE_PREFETCH=require
export PRODUCTION_CACHE_LEVERS=gptq,joint_scale_opt
export VISUAL_FORMAT=BF16
export MTP_FORMAT=BF16

./prismaquant/run-pipeline.sh 2>&1 | tee "$WORK_DIR/logs/pipeline.log"
```

Expected artifacts:

- `$WORK_DIR/artifacts/probe.pkl`
- `$WORK_DIR/artifacts/cost.pkl`
- `$WORK_DIR/artifacts/pareto.csv`
- `$WORK_DIR/artifacts/format_applicability.json`
- `$WORK_DIR/artifacts/pareto_assignments/manifest.json`
- `$WORK_DIR/artifacts/validated_frontier_kl.json`
- `$WORK_DIR/artifacts/validated_frontier_selection.json`
- `$WORK_DIR/artifacts/layer_config.json`
- `$WORK_DIR/artifacts/production_weight_cache*_recached.pkl`
- `$WORK_DIR/exported/config.json`

The logs must show production-cache prefetch policy `require` and no archived
research path.

## Export And Serving Smokes

Validate the exported compressed-tensors artifact in eager mode:

```bash
python3 -m prismaquant.validate_native_export \
  --model "$WORK_DIR/exported" \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048 \
  2>&1 | tee "$WORK_DIR/logs/validate_native_export_eager.log"
```

Then validate graph/compiled mode:

```bash
python3 -m prismaquant.validate_native_export \
  --model "$WORK_DIR/exported" \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048 \
  --no-enforce-eager \
  2>&1 | tee "$WORK_DIR/logs/validate_native_export_graph.log"
```

For a compact vLLM prompt smoke with JSON output:

```bash
python3 tools/vllm_prompt_smoke.py \
  --model "$WORK_DIR/exported" \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048 \
  --output-json "$WORK_DIR/logs/vllm_smoke_eager.json" \
  2>&1 | tee "$WORK_DIR/logs/vllm_smoke_eager.log"

python3 tools/vllm_prompt_smoke.py \
  --model "$WORK_DIR/exported" \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048 \
  --no-enforce-eager \
  --output-json "$WORK_DIR/logs/vllm_smoke_graph.json" \
  2>&1 | tee "$WORK_DIR/logs/vllm_smoke_graph.log"
```

## Milestone Record

Before tagging, record these in the release note or commit message:

- git commit SHA
- source model path and model config hash
- calibration dataset path and `NSAMPLES` / `SEQLEN`
- `TARGET_BITS`, achieved bpp, selected frontier label, and measured KL
- production-cache settings, especially prefetch mode and resident budget
- export path
- all log paths above
- vLLM eager and graph-mode smoke result summaries
