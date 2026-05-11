# Clip / Fisher / MXFP8 Ablation - 2026-05-10

Purpose: decide whether the new local production-cache levers should move
toward the production recipe before HALO/rotation work resumes.

## Contract

- Models:
  - `/home/rob/.cache/huggingface/qwen35-0p8b-bf16`
  - `/home/rob/.cache/huggingface/Qwen3-4B`
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Assignment scope: concrete allocator assignment only; no full format-menu
  production-cache render.
- Production-cache prefetch: required resident preload for KL validation.
- Formats: `NVFP4,MXFP8_E4M3,BF16`
- HALO/rotations: disabled.

## 0.8B Result

Run root:
`/home/rob/dq-runs/qwen35-0p8b-lever-ablation-20260510T164352Z`

Settings: `nsamples=32`, `seqlen=1024`, target around 5 bpp.

| Arm | KL |
|---|---:|
| baseline, MXFP8 shifts disabled | 0.171615734 |
| clip solver | 0.174581131 |
| Fisher-weighted GPTQ | 0.185641893 |
| clip + Fisher | 0.195912688 |
| MXFP8 exponent sweep | 0.215005701 |

0.8B conclusion: no lever promoted from this model alone.

## 4B Result

Run root:
`/home/rob/dq-runs/qwen3-4b-lever-ablation-20260510T170245Z`

Settings: `nsamples=16`, `seqlen=1024`, measured at `4.999639` bpp.
Assignment counts were constant across arms: `225 NVFP4`, `7 MXFP8`,
`20 BF16`.

| Arm | KL | Relative to baseline |
|---|---:|---:|
| baseline, MXFP8 shifts disabled | 0.105940132 | - |
| MXFP8 exponent sweep | 0.127217386 | +20.08% worse |
| clip solver, all 133 fused groups | 0.077850933 | -26.51% better |
| clip solver, top 64 groups | 0.088425541 | -16.53% better |
| clip solver, top 32 groups | 0.213804499 | +101.82% worse |
| Fisher-weighted GPTQ | 0.210982392 | +99.15% worse |
| clip + Fisher | 0.180224068 | +70.12% worse |

Fisher rerun used regenerated h-detail files containing `g2_per_token`;
production-cache metadata confirmed `232` loaded and `0` misses.

## Decisions

- Keep `PRISMAQUANT_ACT_CLIP_SOLVER` opt-in, but treat it as a real 4B
  candidate. Full all-groups solve is the quality winner; top-64 is a
  possible runtime/quality compromise. Top-32 is unsafe.
- Keep Fisher-weighted GPTQ experimental and out of recipes. The corrected
  path regressed on 4B, both alone and with clipping.
- Default `PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS` to `0`. Nonzero MXFP8
  exponent shifts regressed on both 0.8B and 4B.
- Do not compose these with HALO yet. The next production-facing candidate is
  the activation clipper, measured on 27B with the same validation contract.

## FourOverSix Additivity Smoke

Run root:
`/home/rob/dq-runs/fouroversix-smoke-20260510T225344Z`

FourOverSix was tested as an NVFP4 block-scale rule, not a new runtime
format. `PRISMAQUANT_NVFP4_SCALE_RULE=four_over_six_mse` tries max-to-6
and max-to-4 per 16-weight block and keeps the lower block-MSE scale while
preserving the same compressed-tensors NVFP4 schema and vLLM kernel.

Settings were assignment-scoped production-cache renders, required resident
prefetch for KL, rotations disabled, Fisher disabled, MXFP8 exponent shifts
disabled, GPTQ + damp sweep + scale_sweep enabled. Clip arms used
`PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS=6` over all eligible fused groups.

### 0.8B FourOverSix

Run root:
`/home/rob/dq-runs/fouroversix-smoke-20260510T225344Z/qwen35-0p8b`

Settings: `nsamples=32`, `seqlen=1024`, measured at `5.080835` bpp.
Assignment counts were constant across arms: `146 NVFP4`, `4 MXFP8`,
`94 BF16`.

| Arm | KL | Relative to static |
|---|---:|---:|
| static | 0.204225378 | - |
| FourOverSix | 0.212856986 | +4.23% worse |
| clip solver | 0.227983240 | +11.63% worse |
| FourOverSix + clip solver | 0.149187058 | -26.95% better |

Clip metadata: static+clip accepted thresholds for `64/94` fused groups
(`103` qnames). FourOverSix+clip accepted `68/94` fused groups (`111`
qnames). The `.8B` signal is noisy for individual levers but strongly favors
the composition.

### 4B FourOverSix

Run root:
`/home/rob/dq-runs/fouroversix-smoke-20260510T225344Z/qwen3-4b`

Settings: `nsamples=16`, `seqlen=1024`, measured at `4.999639` bpp.
Assignment counts were constant across arms: `225 NVFP4`, `7 MXFP8`,
`20 BF16`.

| Arm | KL | Relative to static |
|---|---:|---:|
| static | 0.158064165 | - |
| FourOverSix | 0.069056227 | -56.31% better |
| clip solver | 0.065697306 | -58.44% better |
| FourOverSix + clip solver | 0.056417932 | -64.31% better |

The 4B composition improved `18.30%` over FourOverSix alone and `14.12%`
over clipping alone. Clip metadata: static+clip accepted thresholds for
`87/133` fused groups (`164` qnames). FourOverSix+clip accepted `79/133`
fused groups (`157` qnames), so FourOverSix reduced the number of selected
clip groups while still improving the combined KL.

Decision: keep FourOverSix opt-in, but promote it to the same candidate tier
as the clip solver. The next production-facing measurement should test
`four_over_six_mse + act_clip_solver` on 27B before HALO/rotation work
resumes.

## BF16 / FP8 Audit

The 2026-05-10 current-recipe smokes validated dense `FP8_E4M3` in vLLM eager
and graph mode:

- 0.8B current recipe:
  `/home/rob/dq-runs/qwen35-0p8b-current-recipe-smoke-docker-20260510T012126Z`
  exported `36` dense `FP8_E4M3` Linears and passed vLLM eager + graph smoke.
- 4B current recipe:
  `/home/rob/dq-runs/qwen3-4b-current-recipe-smoke-docker-20260510T012846Z`
  exported `21` dense `FP8_E4M3` Linears and passed vLLM eager + graph smoke.

BF16 audit metadata from the clip/Fisher smoke exports shows unresolved
upgrade opportunities:

| Model | BF16 audit count | Legal 8-bit alternatives |
|---|---:|---|
| 0.8B | 86 | `60` legal for `MXFP8_E4M3`, `86` legal for `FP8_E4M3` |
| 4B | 18 | `18` legal for `MXFP8_E4M3`, `18` legal for `FP8_E4M3` |

These were allocator-selected BF16 entries, not immutable/pinned BF16. The
next allocator/cost pass should keep `FP8_E4M3` in the production menu and
measure whether these BF16 entries can move down without KL loss. E5M2 remains
research-only: `tests/test_format_menu_expansion.py` and `layer_config.py`
reject `MXFP8_E5M2` / `FP8_E5M2` for the production profile until vLLM weight
dispatch is validated.
