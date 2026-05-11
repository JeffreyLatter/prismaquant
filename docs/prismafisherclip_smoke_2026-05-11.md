# PrismaFisherClip .8B Smoke - 2026-05-11

## Setup

- Model: `/home/rob/.cache/huggingface/qwen35-0p8b-bf16`
- Assignment: `/home/rob/dq-runs/fouroversix-smoke-20260510T225344Z/qwen35-0p8b/artifacts/layer_config.json`
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Calibration: `n=16`, `seqlen=1024`
- Formats: `NVFP4,MXFP8_E4M3,BF16`
- Base levers: FourOverSix NVFP4 scale rule, GPTQ, damp sweep, scale sweep,
  PrismaClip
- Run root: `/home/rob/dq-runs/prismafisherclip-smoke-20260511T210153Z/qwen35-0p8b`
- Execution: Docker GPU path, `vllm-fresh-b12x-fla:latest`, CUDA device
  `NVIDIA GB10`

## Results

| Variant | Mode | KL | bpp | Cache prefetch |
| --- | --- | ---: | ---: | --- |
| Fixed FourOverSix + PrismaClip reference | none | `0.14918706` | `5.08083491` | prior cache |
| PrismaFisherClip | `score` | `0.19577897` | `5.08083491` | `150/150` |
| PrismaFisherClip | `veto` | `0.19948923` | `5.08083491` | `150/150` |

The `score` run used Fisher-weighted local output MSE as the primary clipping
objective. The `veto` run kept unweighted PrismaClip as the primary objective
but required the chosen threshold to improve the Fisher-weighted score too.

Metadata showed the Fisher signal rejected useful unweighted clip decisions:

- `score`: `77/94` groups solved, `123` qnames selected.
- `veto`: `53/94` groups solved, `91` qnames selected, with `31` Fisher
  rejections.

## Decision

Do not run the 4B PrismaFisherClip modifier smoke yet. The .8B result is far
enough behind the fixed PrismaClip reference that a 4B run would mostly spend
compute confirming the same failure mode.

Keep PrismaFisherClip as an opt-in diagnostic/ablation:

- Default mode: `audit`, which records Fisher-weighted clip scores but leaves
  normal PrismaClip decisions unchanged.
- Explicit ablations: `veto` and `score`.

This preserves the h-detail reuse path for future analysis without allowing
the Fisher diagonal to steer production clipping until it proves itself.
