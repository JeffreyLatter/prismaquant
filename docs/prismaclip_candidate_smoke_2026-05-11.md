# PrismaClip Candidate-Mode Smoke - 2026-05-11

Purpose: verify that PrismaClip can be measured as a same-bpp candidate
without globally rewriting baseline NVFP4, and that the pseudo-format does not
leak into vLLM metadata.

## Setup

- Model: `/home/rob/.cache/huggingface/qwen35-0p8b-bf16`
- Runtime: Docker `vllm-fresh-b12x-fla:latest`, CUDA, `NVIDIA GB10`
- Calibration: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`,
  `n=2`, `seqlen=256`
- NVFP4 scale rule: `PRISMAQUANT_NVFP4_SCALE_RULE=four_over_six_mse`
- Cache levers: `gptq,scale_sweep`
- Smoke root:
  `/home/rob/dq-runs/prismaclip-candidate-smoke-20260511/qwen35-0p8b`

The container was run with `PYTHONNOUSERSITE=1`; otherwise Python picked up
the host user-site CPU PyTorch and PrismaQuant correctly refused to run.

## Targeted Recipe

The smoke config assigned:

- `model.layers.1.mlp.gate_proj`: `NVFP4`
- `model.layers.1.mlp.up_proj`: `NVFP4_CLIPPED`
- `model.layers.1.linear_attn.in_proj_qkv`: `MXFP8`
- `model.layers.1.linear_attn.in_proj_z`: `MXFP8`
- MTP Linears: `BF16`

The first attempt intentionally exposed a vLLM fused-sibling rule: assigning
`in_proj_qkv` to MXFP8 while leaving `in_proj_z` BF16 fails because vLLM
requires one scheme for `linear_attn.in_proj_qkvz`. The corrected recipe keeps
that fused unit homogeneous.

## Results

Production cache build:

- Device: `cuda`
- Activation capture: `4/4` Linears, `524,288` resident bytes on `cuda:0`
- PrismaClip solver: `status=applied`, `candidate_format=NVFP4_CLIPPED`,
  `cache_formats=['NVFP4_CLIPPED']`
- Rendered entries: `4`, failures: `0`
- Coverage check: passed
- Wall time: `5.2s`

Export:

- Recipe mix: `{'MXFP8': 2, 'NVFP4': 1, 'NVFP4_CLIPPED': 1, 'BF16': 5}`
- Materialization hist included `2` MXFP8 production-cache Linears and `2`
  NVFP4 production-cache Linears.
- Exported `config.json` contains no `NVFP4_CLIPPED` string; the clipped
  candidate is exported as normal NVFP4 compressed-tensors metadata.

vLLM eager validation:

- Quantization config loaded as compressed-tensors mixed precision.
- vLLM selected `FlashInferCutlassMxfp8LinearKernel` for MXFP8 GEMM.
- vLLM selected `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM.
- Prompt: `The capital of France is`
- Output: ` Paris.\nThe capital of France is`

This is a functional smoke, not a KL or throughput benchmark. The tiny mostly
BF16 artifact spent normal one-time startup/autotune time in vLLM, so the
generation timing should not be interpreted as performance data.
