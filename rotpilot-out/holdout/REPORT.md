# Fixed-CB LDLQ held-out validation

Date: 2026-08-03

Branch: `feat/rotation-ldlq-pilot`

GPU: NVIDIA GB10, torch 2.11.0+cu130

Raw results: [`results.json`](results.json)

## Verdict

**VALIDATED-PENDING-MODEL-LEVEL.** Across the 27 CAL32 decisions, mean in-sample reduction is 95.37% and mean held-out reduction is 80.27%, retaining 84.16% of the in-sample gain. Cell verdicts: 27 validated, 0 partial, and 0 overfit-artifact.

The decision rule is fixed in advance: held-out reduction greater than 50% of the in-sample reduction is `VALIDATED-PENDING-MODEL-LEVEL`; held-out reduction below 10% is `OVERFIT-ARTIFACT`; intermediate outcomes are `PARTIAL` with retained gain reported.

## Contract

Each Linear has exactly 64 activation rows. For each of seeds 0, 1, and 2, a deterministic CPU random permutation defines disjoint CAL32 (first 32 rows) and HOLDOUT32 (last 32 rows) sets. There is no stratification. The CAL16 mechanics probe uses the first 16 rows of the same CAL32 permutation and is evaluated on the same HOLDOUT32.

For every split and rung, the plain production encoder fits its codebook, two-tier scales, and assignments using only the CAL-derived `mean(X^2)` column weights. LDLQ freezes those fields, uses the same CAL-only assignment metric, and builds its damped Hessian from CAL only. Both reconstructed weights are then evaluated separately on CAL and HOLDOUT. Ratios below 1 and positive reductions favor LDLQ.

## CAL32 decision table

| Linear | K | split | CAL ratio | CAL reduction | HOLDOUT ratio | HOLDOUT reduction | gap (pp) | retained | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `layers.40.attn.wq_b` | 12 | 0 | 0.0895 | 91.05% | 0.1399 | 86.01% | 5.03 | 94.47% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 15 | 0 | 0.0848 | 91.52% | 0.1388 | 86.12% | 5.40 | 94.10% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 18 | 0 | 0.0865 | 91.35% | 0.1399 | 86.01% | 5.33 | 94.16% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 12 | 1 | 0.0877 | 91.23% | 0.1838 | 81.62% | 9.62 | 89.46% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 15 | 1 | 0.0823 | 91.77% | 0.1849 | 81.51% | 10.26 | 88.82% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 18 | 1 | 0.0863 | 91.37% | 0.1902 | 80.98% | 10.39 | 88.63% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 12 | 2 | 0.0887 | 91.13% | 0.2042 | 79.58% | 11.55 | 87.32% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 15 | 2 | 0.0836 | 91.64% | 0.2062 | 79.38% | 12.26 | 86.62% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.attn.wq_b` | 18 | 2 | 0.0861 | 91.39% | 0.2076 | 79.24% | 12.15 | 86.70% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 12 | 0 | 0.0320 | 96.80% | 0.4144 | 58.56% | 38.24 | 60.50% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 15 | 0 | 0.0242 | 97.58% | 0.4059 | 59.41% | 38.17 | 60.88% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 18 | 0 | 0.0234 | 97.66% | 0.4133 | 58.67% | 38.99 | 60.08% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 12 | 1 | 0.0320 | 96.80% | 0.3441 | 65.59% | 31.21 | 67.76% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 15 | 1 | 0.0262 | 97.38% | 0.3456 | 65.44% | 31.94 | 67.20% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 18 | 1 | 0.0250 | 97.50% | 0.3461 | 65.39% | 32.11 | 67.07% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 12 | 2 | 0.0315 | 96.85% | 0.4104 | 58.96% | 37.89 | 60.88% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 15 | 2 | 0.0255 | 97.45% | 0.3987 | 60.13% | 37.32 | 61.70% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.40.experts.81.up_proj` | 18 | 2 | 0.0241 | 97.59% | 0.4184 | 58.16% | 39.43 | 59.59% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 12 | 0 | 0.0254 | 97.46% | 0.0260 | 97.40% | 0.06 | 99.94% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 15 | 0 | 0.0239 | 97.61% | 0.0247 | 97.53% | 0.08 | 99.92% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 18 | 0 | 0.0259 | 97.41% | 0.0268 | 97.32% | 0.09 | 99.90% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 12 | 1 | 0.0254 | 97.46% | 0.0259 | 97.41% | 0.05 | 99.95% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 15 | 1 | 0.0251 | 97.49% | 0.0257 | 97.43% | 0.06 | 99.93% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 18 | 1 | 0.0263 | 97.37% | 0.0271 | 97.29% | 0.08 | 99.92% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 12 | 2 | 0.0264 | 97.36% | 0.0269 | 97.31% | 0.05 | 99.95% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 15 | 2 | 0.0245 | 97.55% | 0.0252 | 97.48% | 0.07 | 99.93% | VALIDATED-PENDING-MODEL-LEVEL |
| `layers.20.experts.63.up_proj` | 18 | 2 | 0.0269 | 97.31% | 0.0277 | 97.23% | 0.08 | 99.92% | VALIDATED-PENDING-MODEL-LEVEL |

## CAL-size overfit-mechanics probe

The comparison below holds the 32-row evaluation set fixed. A positive delta means held-out gain grew when calibration shrank from 32 to 16 rows, the classic overfit signature named in the experiment design.

| Linear | K | split | HOLDOUT reduction CAL16 | HOLDOUT reduction CAL32 | CAL16-CAL32 (pp) | trend |
|---|---:|---:|---:|---:|---:|---|
| `layers.40.attn.wq_b` | 12 | 0 | 73.06% | 86.01% | -12.95 | stable/better at CAL32 |
| `layers.40.attn.wq_b` | 15 | 0 | 72.74% | 86.12% | -13.38 | stable/better at CAL32 |
| `layers.40.attn.wq_b` | 18 | 0 | 72.43% | 86.01% | -13.59 | stable/better at CAL32 |
| `layers.40.attn.wq_b` | 12 | 1 | 77.47% | 81.62% | -4.15 | stable/better at CAL32 |
| `layers.40.attn.wq_b` | 15 | 1 | 77.38% | 81.51% | -4.14 | stable/better at CAL32 |
| `layers.40.attn.wq_b` | 18 | 1 | 77.01% | 80.98% | -3.97 | stable/better at CAL32 |
| `layers.40.attn.wq_b` | 12 | 2 | 80.21% | 79.58% | 0.63 | grows as CAL shrinks |
| `layers.40.attn.wq_b` | 15 | 2 | 80.09% | 79.38% | 0.71 | grows as CAL shrinks |
| `layers.40.attn.wq_b` | 18 | 2 | 80.00% | 79.24% | 0.76 | grows as CAL shrinks |
| `layers.40.experts.81.up_proj` | 12 | 0 | 50.80% | 58.56% | -7.76 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 15 | 0 | 50.86% | 59.41% | -8.55 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 18 | 0 | 49.56% | 58.67% | -9.12 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 12 | 1 | 58.03% | 65.59% | -7.57 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 15 | 1 | 59.44% | 65.44% | -6.00 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 18 | 1 | 57.47% | 65.39% | -7.92 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 12 | 2 | 54.54% | 58.96% | -4.42 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 15 | 2 | 54.48% | 60.13% | -5.65 | stable/better at CAL32 |
| `layers.40.experts.81.up_proj` | 18 | 2 | 52.78% | 58.16% | -5.37 | stable/better at CAL32 |
| `layers.20.experts.63.up_proj` | 12 | 0 | 96.63% | 97.40% | -0.77 | stable/better at CAL32 |
| `layers.20.experts.63.up_proj` | 15 | 0 | 96.59% | 97.53% | -0.94 | stable/better at CAL32 |
| `layers.20.experts.63.up_proj` | 18 | 0 | 96.19% | 97.32% | -1.13 | stable/better at CAL32 |
| `layers.20.experts.63.up_proj` | 12 | 1 | 97.33% | 97.41% | -0.08 | stable/better at CAL32 |
| `layers.20.experts.63.up_proj` | 15 | 1 | 97.49% | 97.43% | 0.06 | grows as CAL shrinks |
| `layers.20.experts.63.up_proj` | 18 | 1 | 97.19% | 97.29% | -0.10 | stable/better at CAL32 |
| `layers.20.experts.63.up_proj` | 12 | 2 | 97.32% | 97.31% | 0.01 | grows as CAL shrinks |
| `layers.20.experts.63.up_proj` | 15 | 2 | 97.52% | 97.48% | 0.05 | grows as CAL shrinks |
| `layers.20.experts.63.up_proj` | 18 | 2 | 97.30% | 97.23% | 0.07 | grows as CAL shrinks |

Mean held-out reduction is 76.00% at CAL16 versus 80.27% at CAL32; gain grows under smaller CAL in 7/27 matched cases. **This argues that feedback is capturing structure rather than benefiting from a smaller fit set.**

## Limitation and next gate

This 32-row holdout comes from the same calibration distribution. It tests estimation overfit, not distribution shift. The stronger test is a future GPU probe that runs fresh text through the model and captures new activations; this experiment deliberately performs no new model forwards.

Command:

```bash
export PYTHONPATH=/w PRISMAQUANT_CB_EXT_DIR=/w/.ext
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  scripts/rotation_ldlq_holdout.py --out rotpilot-out/holdout \
  --sample-root /home/rob/dq-runs/dsv4-flash-0731/tier3-sample \
  --ext-dir /w/.ext --block-size 64 --damping-fraction 0.01
```
