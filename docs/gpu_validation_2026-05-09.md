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

## Notes

- The validation harness now sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` before
  importing vLLM to avoid CUDA fork reinitialization.
- `--no-enforce-eager` enables graph-mode validation after the eager smoke.
- MTP-family speculative configs no longer inject `model=<same checkpoint>`;
  vLLM expects the model field absent/null so it can use the embedded-MTP path.
- `run-pipeline.sh` now defaults `PRODUCTION_RECACHE=1`. Use
  `PRODUCTION_RECACHE=0` for explicit no-recache ablations.
