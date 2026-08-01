# AURA 4B dense frontier check, 2026-06-05

## Goal

Check whether the AURA rate-distortion curve on Qwen3-4B has a stable
kneedle when the AURA cost table is fp32 and the measured KL frontier is dense.

This is a research measurement, not a production-promotion result.

## Inputs

- Model:
  `/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`
- Probe: `/home/rob/dq-runs/xlayer_rung0/probe.pkl`
- AURA costs: `/home/rob/dq-runs/xlayer_rung0/cost_aura_rep2.pkl`
- Cost provenance: `/home/rob/dq-runs/xlayer_rung0/aura_cost_rep2.log`
  records `dtype=float32`, `n_probes=32`, and a loaded production cache.
- Production weight cache:
  `/home/rob/dq-runs/xlayer_rung0/prod4b_nodamp/prodcache_nvfp4_damp0p01.pkl`

## Assignment Grid

Generated a two-format `NVFP4,BF16` dense grid from `4.5` to `8.0` bpp in
`0.25` bpp steps:

```bash
PYTHONPATH=/home/rob/prismaquant \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m prismaquant.allocator \
  --probe /home/rob/dq-runs/xlayer_rung0/probe.pkl \
  --costs /home/rob/dq-runs/xlayer_rung0/cost_aura_rep2.pkl \
  --model-override /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --formats NVFP4,BF16 \
  --target-bits 6.0 \
  --pareto-targets 4.5,4.75,5.0,5.25,5.5,5.75,6.0,6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0 \
  --layer-config /home/rob/dq-runs/aura-4b-dense-fp32/artifacts/lc_aura_2fmt_6p0.json \
  --pareto-csv /home/rob/dq-runs/aura-4b-dense-fp32/artifacts/pareto_aura_2fmt_dense.csv \
  --pareto-output-dir /home/rob/dq-runs/aura-4b-dense-fp32/pareto
```

Allocator log:
`/home/rob/dq-runs/aura-4b-dense-fp32/logs/allocate.log`

## Validation

Both validation passes used:

- full-sequence KL
- `n_calib_samples=4`
- `calib_seqlen=256`
- `calib_repeats=8`
- production-cache replay through the existing `ProductionWeightCache`
- source and production-cache prefetch set to `require`

The fp32 pass wrote:

- JSON: `/home/rob/dq-runs/aura-4b-dense-fp32/artifacts/vak_dense_fp32.json`
- Log: `/home/rob/dq-runs/aura-4b-dense-fp32/logs/validate_dense_fp32.log`

The bf16 pass wrote:

- JSON: `/home/rob/dq-runs/aura-4b-dense-fp32/artifacts/vak_dense_bf16.json`
- Log: `/home/rob/dq-runs/aura-4b-dense-fp32/logs/validate_dense_bf16.log`

Combined analysis:

- JSON: `/home/rob/dq-runs/aura-4b-dense-fp32/artifacts/analysis_dense_fp32_bf16.json`
- CSV: `/home/rob/dq-runs/aura-4b-dense-fp32/artifacts/measured_dense_fp32_bf16.csv`
- Plot: `/home/rob/dq-runs/aura-4b-dense-fp32/artifacts/dense_frontier_fp32_bf16.png`

## Results

The measured fp32 frontier was monotone:

| target bpp | achieved bpp | fp32 KL | fp32 stderr |
| ---: | ---: | ---: | ---: |
| 4.50 | 4.500 | 0.222204 | 0.014148 |
| 4.75 | 4.736 | 0.208196 | 0.017666 |
| 5.00 | 4.998 | 0.197576 | 0.015494 |
| 5.25 | 5.243 | 0.192230 | 0.013462 |
| 5.50 | 5.500 | 0.187631 | 0.015622 |
| 5.75 | 5.745 | 0.182955 | 0.013930 |
| 6.00 | 5.998 | 0.177950 | 0.014127 |
| 6.25 | 6.238 | 0.170409 | 0.010795 |
| 6.50 | 6.495 | 0.168289 | 0.013348 |
| 6.75 | 6.749 | 0.158842 | 0.013698 |
| 7.00 | 6.989 | 0.153362 | 0.011051 |
| 7.25 | 7.246 | 0.147504 | 0.012680 |
| 7.50 | 7.499 | 0.139384 | 0.010958 |
| 7.75 | 7.748 | 0.137063 | 0.012929 |
| 8.00 | 7.993 | 0.131830 | 0.011971 |

Fit diagnostics:

| dtype | raw-linear R2 | log10-linear R2 | raw kneedle | log kneedle |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 0.98755 | 0.99187 | 5.00 | 5.00 |
| bf16 | 0.97332 | 0.98498 | 4.75 | 4.75 |

Bootstrap kneedle stability over repeat resamples:

- fp32 raw kneedle: 5.00 in 454/1000, 5.25 in 266/1000, 4.75 in 118/1000.
- fp32 log kneedle: 5.00 in 453/1000, 4.75 in 213/1000, 5.25 in 183/1000,
  with a secondary high-bpp tail at 7.50 in 117/1000.
- bf16 raw kneedle: 4.75 in 698/1000, then diffuse alternatives.
- bf16 log kneedle: 4.75 in 786/1000, then diffuse alternatives.

Paired adjacent differences show that the cumulative curve is real, but many
individual 0.25-bpp adjacent gains are not significant at this sample size.
For fp32, cumulative improvement from 4.5 to 8.0 bpp was `0.090374` with paired
stderr `0.006069` (`z=14.89`). Adjacent-step paired z-scores were mixed, with
several below `z=1`.

## Interpretation

The 4B dense frontier does not support the idea that AURA failed to produce a
measurable loss-vs-bpp curve. It produced a clean, mostly log-linear measured
frontier with significant cumulative improvement.

The unstable part is local elbow selection. Kneedle moves because adjacent
0.25-bpp steps are small relative to calibration variance, especially in bf16.
The fp32 curve is monotone and cleaner, while the bf16 curve has small local
reversals at the low and high ends.

For this 4B two-format run, a diagnostic kneedle around `5.0` bpp is plausible
in fp32, but it is not stable enough to use as the production ship rule. A
target-size or marginal-KL-per-byte policy remains the cleaner deployment
selector.
