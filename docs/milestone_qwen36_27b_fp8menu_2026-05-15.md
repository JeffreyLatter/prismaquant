# Qwen3.6-27B FP8-Menu Milestone (2026-05-15)

## Scope

Milestone snapshot before starting the next upgrade phase. This records the
current Qwen3.6-27B BF16-source quantization state, the reusable calibration
inputs, and the live materialization run.

## Source And Calibration Contract

- Source model: `/home/rob/.cache/huggingface/qwen36-27b-bf16`
- Model profile: `qwen3_5_dense`
- Calibration dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Calibration shape: `8 x 1024`, text-only
- Reused probe:
  `/home/rob/dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/probe.pkl`
- Reused FP8-menu cost:
  `/home/rob/dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/cost_fp8menu.pkl`

## Current Quantization Run

- Run directory:
  `/home/rob/dq-runs/qwen36-27b-fp8menu-5p05-quantize-20260515T195513Z`
- Container:
  `pq-qwen36-27b-quantize-20260515T195513Z`
- Pipeline log:
  `/home/rob/dq-runs/qwen36-27b-fp8menu-5p05-quantize-20260515T195513Z/logs/pipeline.log`
- Source menu:
  `NVFP4,MXFP8_E4M3,FP8_E4M3,BF16`
- Target bits:
  `5.05`
- Selection mode:
  `surrogate`
- Production render:
  `assignment` scope through `ProductionWeightCache`
- Enabled production levers:
  `gptq,joint_scale_opt`
- Production recache:
  enabled

## Allocator Result

Allocator completed successfully and wrote:

- Layer config:
  `/home/rob/dq-runs/qwen36-27b-fp8menu-5p05-quantize-20260515T195513Z/artifacts/layer_config.json`
- Pareto curve:
  `/home/rob/dq-runs/qwen36-27b-fp8menu-5p05-quantize-20260515T195513Z/artifacts/pareto.csv`

Selected target:

| target bpp | achieved bpp | predicted loss | NVFP4 | FP8_E4M3 | MXFP8_E4M3 | BF16 |
|---:|---:|---:|---:|---:|---:|---:|
| 5.05 | 5.049 | 799.5 | 228 | 74 | 0 | 7 |

Allocator notes:

- `MXFP8_E4M3` was dropped for 96 Linears because of kernel shape limits.
- Visual Linears are uniformly `BF16` in this text-only calibration run.
- MTP Linears are uniformly `BF16`.
- Source-dtype manifest reported `0 fp8`, `755 bf16`.

## Live Materialization Status

As of `2026-05-15T20:09:48Z`, the production cache build is still running.

Observed status:

- Container status: running
- Last pipeline marker: `[prod-cache] 25/487`
- Production-cache files present: `48`
- Run directory size: `4.3G`
- Cache directory:
  `/home/rob/dq-runs/qwen36-27b-fp8menu-5p05-quantize-20260515T195513Z/artifacts/production_weight_cache`

Production cache setup already completed:

- Loaded BF16 source weights.
- Found `496` quantizable Linears.
- Render scope is `487` non-BF16 assignment entries.
- Activation capture stored `336/487` Linears on `cuda:0`.
- Resident activation bytes: `5,163,188,224`.
- Computed joint NVFP4 globals for `360` fused-sibling members.
- Computed activation `max_abs` for `487` Linears across `299` fused groups.

## Pending Gates

This is not a completed shipping artifact yet. Remaining gates:

- Finish production cache materialization.
- Finish production recache.
- Export compressed-tensors checkpoint to `exported/`.
- Run vLLM eager load/generation smoke.
- Run vLLM graph/compiled load/generation smoke.
- Record final bpp over quantizable parameters only.
- Run KL validation before making quality claims.

## Baseline Context

Most recent completed clean runtime export:

- Run:
  `/home/rob/dq-runs/qwen36-27b-kneedle-serving-constrained-export-20260512T115610Z`
- Export:
  `/home/rob/dq-runs/qwen36-27b-kneedle-serving-constrained-export-20260512T115610Z/exported`
- Reported quantizable-body bpp:
  `4.587260469372376`
- Format counts:
  `401 NVFP4`, `11 MXFP8_E4M3`, `84 BF16`
- vLLM eager:
  passed
- vLLM graph:
  passed
- KL:
  not run; prior note says the HF KL path became block-I/O bound and was
  stopped under the GPU-first policy.

## Upgrade Starting Point

The next upgrade phase should treat the in-progress FP8-menu 5.05 run as the
current materialization branch, and the completed 4.587 bpp serving-constrained
export as the last fully validated load/generation baseline.
