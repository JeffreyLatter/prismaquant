# GPU Validation, 2026-05-09

This records the GPU-serving checks from the next-phase refactor branch.
All runtime checks used `vllm-fresh-b12x-fla:latest` on CUDA.

## Qwen3.5-0.8B

Run:

`/home/rob/dq-runs/qwen35-0p8b-pipeline-recache-smoke-20260509T213900Z`

Configuration:

- `NSAMPLES=1`, `SEQLEN=128`
- `FORMATS=NVFP4,BF16`
- `PRODUCTION_CACHE=1`
- `PRODUCTION_RECACHE=1`
- `MTP_FORMAT=BF16`

Result:

- Pipeline completed end-to-end.
- Production cache rendered 186 entries and recache measured 186 activation ranges.
- vLLM eager load/generation passed.

## Qwen3-4B

Run:

`/home/rob/dq-runs/qwen3-4b-pipeline-recache-smoke-20260509T233407Z`

Configuration:

- `NSAMPLES=1`, `SEQLEN=128`
- `FORMATS=NVFP4,BF16`
- `PRODUCTION_CACHE=1`
- `PRODUCTION_RECACHE=1`
- `MTP_FORMAT=BF16`

Result:

- Pipeline completed end-to-end.
- Production cache rendered 252 entries and recache measured 252 activation ranges.
- vLLM eager load/generation passed with `FlashInferCutlassNvFp4LinearKernel`.
- vLLM graph-mode load/generation passed with torch.compile and CUDA graph capture.

Logs:

- Eager: `/home/rob/dq-runs/qwen3-4b-pipeline-recache-smoke-20260509T233407Z/logs/validate_native_spawn.log`
- Graph: `/home/rob/dq-runs/qwen3-4b-pipeline-recache-smoke-20260509T233407Z/logs/validate_native_graph.log`

## Shipped Qwen3.6-27B Artifact

Artifact:

`/home/rob/dq-runs/qwen36-27b-step00-production-cache-vllmlegal-export-main-20260509T073125Z/exported`

Result:

- MTP speculative eager smoke passed with
  `{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}`.
- vLLM resolved the draft architecture as `Qwen3_5MTP`.
- vLLM shared target embedding and `lm_head` weights with the draft model.
- Non-MTP graph-mode serving passed with torch.compile and CUDA graph capture.
- The body used `FlashInferCutlassNvFp4LinearKernel`; 8-bit Linears used
  `FlashInferCutlassMxfp8LinearKernel`; linear-attention prefill used FLA.

Logs:

- MTP eager: `/home/rob/dq-runs/qwen36-27b-step00-production-cache-vllmlegal-export-main-20260509T073125Z/validate_native_mtp_codex.log`
- Graph: `/home/rob/dq-runs/qwen36-27b-step00-production-cache-vllmlegal-export-main-20260509T073125Z/validate_native_graph_codex.log`

## Qwen3.6-27B Re-cache Replay Smoke

Run:

`/home/rob/dq-runs/qwen36-27b-recache-smoke-20260509T235955Z`

Inputs:

- source model: `/home/rob/.cache/huggingface/qwen36-27b-bf16`
- layer config: `/home/rob/dq-runs/qwen36-27b-center-polish-validation-main-20260509T063717Z/selected_step_00/layer_config_mtp_bf16.json`
- production cache manifest: `/home/rob/dq-runs/qwen3p6-27b-kl-probe-triad-n64-production-20260508T032958Z-directpy/production_weight_cache_nvfp4_mxfp8.pkl`
- production cache directory: `/home/rob/dq-runs/qwen3p6-27b-kl-probe-triad-n64-production-20260508T032958Z-directpy/production_weight_cache`
- calibration: `diverse-v1.jsonl`, `n=1`, `seqlen=128`

Result:

- Production-weight replay ran on CUDA and measured 497 activation ranges in
  65.3 seconds.
- The source production cache sidecars were left untouched via
  `--no-write-sidecar`.
- Re-cache materially changed activation ranges even on this tiny smoke:
  median after/before max-abs ratio `0.9156`, p05 `0.2154`, max `1.1688`,
  and `351/496` common ranges moved by more than 5%.
- The recached manifest compacted back to path references before pickle:
  `109K`, `992` weight entries, `0` resident tensors.

Logs:

- Replay: `/home/rob/dq-runs/qwen36-27b-recache-smoke-20260509T235955Z/production_recache_compact.log`
- Delta JSON: `/home/rob/dq-runs/qwen36-27b-recache-smoke-20260509T235955Z/recache_delta_summary.json`

An initial `n=8`, `seqlen=256` replay against the same 91G disk-backed cache
was stopped because it became NVMe-bound: about 1.3GB/s read throughput and
low GPU utilization. The rerun used assignment-required cache prefetch with
`--production-cache-prefetch require`, `--production-cache-lru-gb 64`, and
8 preload workers.

Prefetch-required rerun:

`/home/rob/dq-runs/qwen36-27b-recache-prefetch-n8s256-20260510T001700Z`

Result:

- Preloaded `422` assignment-required rendered weights, `45.32 GiB`, before
  replay.
- Replay observed GPU utilization in the `51-75%` range and no NVMe hot-path
  traffic after preload.
- Production-weight replay measured `497` activation ranges in `68.5` seconds.
- Activation range movement versus the source cache: median after/before
  max-abs ratio `0.9924`, p05 `0.4225`, max `1.8203`, and `198/496` common
  ranges moved by more than 5%.
- The recached manifest compacted back to path references before pickle:
  `110940` bytes, `992` weight entries, `0` resident tensors.

Logs:

- Replay: `/home/rob/dq-runs/qwen36-27b-recache-prefetch-n8s256-20260510T001700Z/production_recache.log`
- Delta JSON: `/home/rob/dq-runs/qwen36-27b-recache-prefetch-n8s256-20260510T001700Z/recache_delta_summary.json`

## Notes

- The validation harness now sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` before
  importing vLLM to avoid CUDA fork reinitialization.
- `--no-enforce-eager` enables graph-mode validation after the eager smoke.
- MTP-family speculative configs no longer inject `model=<same checkpoint>`;
  vLLM expects the model field absent/null so it can use the embedded-MTP path.
- `run-pipeline.sh` now defaults `PRODUCTION_RECACHE=1`. Use
  `PRODUCTION_RECACHE=0` for explicit no-recache ablations.
- `run-pipeline.sh` defaults production recache prefetch to fail-fast
  residency mode: `PRODUCTION_CACHE_PREFETCH=require` and
  `PRODUCTION_CACHE_LRU_GB=64.0`.
- Disk-backed `ProductionWeightCache` manifests now compact resident tensors
  back to path references before pickle; this prevents re-cache replay from
  turning a small manifest into a multi-GB tensor pickle.
