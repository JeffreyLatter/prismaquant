# SmoothQuant No-PrismaClip Smoke - 2026-05-11

Purpose: test SmoothQuant as an outlier-handling replacement candidate after
PrismaClip threshold selection proved unstable.

## Contract

- PrismaClip disabled: `PRISMAQUANT_ACT_CLIP_SOLVER=0`.
- Stack: GPTQ + damp sweep + scale_sweep + FourOverSix.
- SmoothQuant alphas: `0.25,0.5,0.75`.
- SmoothQuant clamp: `10`.
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`.
- Calibration: `n=16`, `seqlen=1024`, seed `42`.
- Production-cache prefetch: required, GPU-resident.

## Results

| Model | Arm | Run root | KL | Notes |
|---|---|---|---:|---|
| Qwen3.5-0.8B | no-clip baseline | `/home/rob/dq-runs/qwen35-0p8b-smoothquant-no-clip-20260511T201339Z` | `0.201933401` | 150/150 cache entries prefetched |
| Qwen3.5-0.8B | SmoothQuant | same | `0.149363996` | selected 2 groups / 4 Linears |
| Qwen3-4B | no-clip baseline | `/home/rob/dq-runs/qwen3-4b-smoothquant-no-clip-20260511T201752Z` | `0.069056227` | 232/232 cache entries prefetched |
| Qwen3-4B | SmoothQuant | same | `0.076682127` | selected 2 groups / 6 Linears |

4B targeted isolation using hardlinked cache variants:

| Variant | KL |
|---|---:|
| baseline | `0.069056227` |
| SmoothQuant layer 28 `q/k/v` only | `0.076200686` |
| SmoothQuant layer 29 `q/k/v` only | `0.063870477` |
| SmoothQuant layer 28 + 29 `q/k/v` | `0.076682127` |

## Interpretation

SmoothQuant is cheaper than PrismaClip in the current implementation. On .8B,
the no-clip baseline cache rendered in `100.8s`; the SmoothQuant cache rendered
in `82.0s`. Recent PrismaClip cache fills on the same .8B setup took roughly
`580-630s` because each NVFP4 fused group is rendered repeatedly across
threshold candidates.

SmoothQuant and AWQ should remain mutually exclusive by default. Both are
per-channel diagonal fold-scale transforms applied through the same predecessor
normalization coordinates; multiplying them together is a new compound
parameterization, not an independent composition.

SmoothQuant is not ready as a broad default. The .8B result is positive, and
the 4B layer-29 isolation is positive, but the 4B local gate also selected a
harmful layer-28 attention `q/k/v` group. That means the current local
output-MSE SmoothQuant gate is not reliable enough for attention q/k/v.

## Next Gate

Keep SmoothQuant opt-in. The next useful implementation is a per-selected-group
validation gate or a better attention-aware gate for `q/k/v`, so one harmful
attention group cannot dominate the benefit from another. Do not re-enable
PrismaClip as part of this decision; the point of this smoke was to measure
SmoothQuant without unstable NVFP4 rerenders.
