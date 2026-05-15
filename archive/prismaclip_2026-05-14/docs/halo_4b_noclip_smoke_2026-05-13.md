# Qwen3-4B No-Clip HALO Smoke, 2026-05-13

Purpose: compare the current progressive render pipeline with PrismaClip and
PrismaFisherClip disabled, then test whether HALO composes with the same 4B
path.

## Common Setup

- Source: `/home/rob/dq-runs/qwen3-4b-untied-bf16`
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Calibration: `NSAMPLES=8`, `SEQLEN=512`
- Formats: `NVFP4,MXFP8_E4M3,BF16`
- Target: `4.75` quantizable-body bpp
- Enabled render levers: FourOverSix, Fisher-GPTQ/GPTQ, scale sweep,
  production recache
- Disabled render levers: PrismaClip, PrismaFisherClip, AWQ, SmoothQuant
- Validation: vLLM eager and CUDA-graph load/generation, plus
  `validate_assignments_kl` last-token KL on the same smoke calibration shape

## Artifacts

- HALO off:
  `/home/rob/dq-runs/qwen3-4b-noclip-halooff-20260512T234806Z/exported`
- HALO random seed 0:
  `/home/rob/dq-runs/qwen3-4b-noclip-halo-random-seed0-20260513T001336Z/exported`

## Results

| Arm | KL n=8/s512 | bpp | Layer config mix | vLLM eager | vLLM graph |
|---|---:|---:|---|---|---|
| HALO off | 0.08506471 | 4.746032 | 240 NVFP4, 5 MXFP8, 7 BF16 | pass | pass |
| HALO random seed 0 | 0.21888617 | 4.744408 | 239 NVFP4, 3 MXFP8, 10 BF16 | pass | pass |

Local cost summaries from `validate_assignments_kl`:

- HALO off: `output_mse=5.17998`, `pred_dloss=4908.48`
- HALO random seed 0: `output_mse=4.32968`, `pred_dloss=6374.47`

Production recache coverage:

- HALO off: measured 252 Linears, moved >5% for 37/245 cached entries.
- HALO random seed 0: measured 252 Linears, moved >5% for 61/242 cached
  entries.

Render gates:

- HALO off:
  - FourOverSix accepted 227, rejected 13.
  - Fisher-GPTQ accepted 239, rejected 1.
  - Scale sweep accepted 32, rejected 213.
- HALO random seed 0:
  - FourOverSix accepted 239, rejected 0.
  - Fisher-GPTQ accepted 237, rejected 2.
  - Scale sweep accepted 12, rejected 230.

The two layer configs differ in 13 Linears. HALO moves a few late attention
projections up to MXFP8/BF16, but also moves some layer-35 projections down
from MXFP8 to NVFP4.

## Interpretation

HALO is mechanically compatible with this 4B path: production cache rendering,
recache, export, eager vLLM, and CUDA-graph vLLM all pass. It is not
quality-positive in this smoke. KL worsens from `0.08506471` to `0.21888617`
despite a lower local Fisher-output-MSE number, so the local render metric is
not sufficient to justify HALO.

Recommendation: keep `HALO_MODE=off` by default. Do not spend 27B compute on
HALO until either a HALO-aware probe/cost path is implemented and validated, or
a small-model run shows a clear KL win under the same calibration contract.
