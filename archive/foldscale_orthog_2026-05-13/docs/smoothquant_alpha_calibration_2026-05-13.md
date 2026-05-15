# SmoothQuant Alpha Calibration - 2026-05-13

## Setup

- Model: `/home/rob/.cache/huggingface/qwen35-0p8b-bf16`
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Calibration: `n=8`, `seqlen=512`, `max_act_rows=512`
- Run dir: `/home/rob/dq-runs/smoothquant-alpha-kl-20260513`
- Validator: `prismaquant.validate_assignments_kl`
- Decode smoke: `tools/vllm_prompt_smoke.py`, eager vLLM,
  `--quantization compressed-tensors`, prompt
  `Write one concise paragraph explaining why the sky appears blue.`

The key production-shaped assignment was:

`/home/rob/dq-runs/qwen35-0p8b-joint-sq-format-pipeline-20260513T203351Z/artifacts/layer_config.json`

It contains `152` NVFP4, `2` MXFP8, and `90` BF16 entries. The SmoothQuant
cache was rendered through `ProductionWeightCache`, then exported through
`export_native_compressed.py`, so the vLLM smoke exercised the actual
compressed-tensors fold path.

## Local KL Sweeps

Monoculture body assignments showed local replay KL keeps improving as alpha
increases:

| Format | Arm | Last-token KL | Full-sequence KL |
|---|---:|---:|---:|
| NVFP4 | RTN | 0.38687898 | 0.24629571 |
| NVFP4 | SQ cap 0.50 | 0.18146304 | 0.22652549 |
| NVFP4 | SQ cap 1.00 | 0.15791093 | 0.22017514 |
| MXFP8 | RTN | 0.16364049 | 0.19556141 |
| MXFP8 | SQ cap 0.50 | 0.12994960 | 0.18163466 |
| MXFP8 | SQ cap 1.00 | 0.07500855 | 0.15041270 |
| FP8 | RTN | 0.01052353 | 0.01508784 |
| FP8 | SQ cap 0.50 | 0.01301842 | 0.01526242 |
| FP8 | SQ cap 1.00 | 0.00925197 | 0.01479522 |

Interpretation: KL replay alone is not enough to choose NVFP4 alpha. It would
prefer cap `1.0`, which fails decode after export.

## vLLM Decode Bracket

On the realistic joint assignment:

| Arm | Selected alpha median | vLLM output |
|---|---:|---|
| no SmoothQuant | none | coherent |
| cap 0.25 | 0.2361 | coherent |
| cap 0.35 | 0.3305 | catastrophic `sky sky ...` repetition |
| temporary cap 0.5 | 0.4615 | repetitive/paraphrase loop |
| cap 1.0 | 0.7163 | catastrophic special-token repetition |

Representative artifacts:

- cap 0.25 smoke:
  `/home/rob/dq-runs/smoothquant-alpha-kl-20260513/artifacts/vllm_smoke_joint_cap0p25.json`
- cap 0.35 smoke:
  `/home/rob/dq-runs/smoothquant-alpha-kl-20260513/artifacts/vllm_smoke_joint_cap0p35.json`
- cap 1.0 smoke:
  `/home/rob/dq-runs/smoothquant-alpha-kl-20260513/artifacts/vllm_smoke_joint_cap1p0.json`

## Promoted-Format Export Bracket

Body-monoculture export/vLLM smokes for promoted formats changed the earlier
local-KL interpretation:

| Format | Arm | vLLM output |
|---|---|---|
| MXFP8 | RTN | coherent |
| MXFP8 | cap 0.50 | coherent |
| MXFP8 | cap 0.75 | borderline/repetitive |
| MXFP8 | cap 1.00 | empty output |
| FP8_E4M3 | RTN | coherent |
| FP8_E4M3 | cap 0.25 | topic drift / Disneyland repetition |
| FP8_E4M3 | cap 0.35 | malformed short output |
| FP8_E4M3 | cap 0.50 | `Stepwise thinking` repetition |
| FP8_E4M3 | cap 1.00 | degenerate `(` output |

Representative artifacts:

- MXFP8 cap 0.50:
  `/home/rob/dq-runs/smoothquant-alpha-kl-20260513/artifacts/vllm_smoke_body_mxfp8_sq_cap0p50.json`
- MXFP8 cap 0.75:
  `/home/rob/dq-runs/smoothquant-alpha-kl-20260513/artifacts/vllm_smoke_body_mxfp8_sq_cap0p75.json`
- FP8 cap 0.25:
  `/home/rob/dq-runs/smoothquant-alpha-kl-20260513/artifacts/vllm_smoke_body_fp8_sq_cap0p25.json`

## Decision

The decode bracket above was run before the production solver scored the
folded runtime activation/weight path. Those results showed that local replay
KL can prefer alpha values that fail vLLM generation, so static caps were a
reasonable temporary guard.

The current policy removes default per-format caps and lets
`PRISMAQUANT_SMOOTHQUANT_ALPHA_HI` define the search range. The format-specific
`*_ALPHA_MAX` variables remain as explicit rollback/ablation controls.

The production solver now scores SmoothQuant candidates with folded runtime
W/A output MSE:

```text
Y_ref = X @ W.T
Y_hat = Q_A(X/s) @ Q_W(W*s).T
```

This is the analytical score that exposes activation/weight dynamic-range
collapse from alpha before export. The allocator should consume joint
`(format, transform)` costs so a format is naturally skipped when SmoothQuant
makes its measured error too high. Export/vLLM smoke remains the final gate for
promoting the no-cap policy.
