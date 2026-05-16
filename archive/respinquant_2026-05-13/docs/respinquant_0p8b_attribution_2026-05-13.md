# Random-Givens ReSpin Substrate 0.8B Local Attribution — 2026-05-13

## Paper-Fidelity Correction

After re-reading the ReSpinQuant paper, this run should be interpreted as a
runtime-substrate and attribution smoke, not as a ReSpinQuant quality result.
The tested artifact used an untrained rank-16 alternating disjoint-Givens basis.
The paper instead trains full layer-wise rotations initialized from Hadamard
matrices with a Cayley optimizer, then compresses the residual-basis transition
using an SVD/polar subspace approximation with default rank 32.

Therefore the negative static-NVFP4 result below only rejects this random-Givens
smoke configuration. It does not reject paper-faithful ReSpinQuant.

## Artifacts

- Baseline data run:
  `/home/rob/dq-runs/qwen35-0p8b-progressive-gates-v2-20260512T224854Z`
- Source BF16:
  `/home/rob/.cache/huggingface/qwen35-0p8b-bf16`
- Fold-only control:
  `/home/rob/dq-runs/qwen35-0p8b-respin-foldonly-gemmafix-a0-20260513T151057Z`
- Random-Givens residual-basis arm:
  `/home/rob/dq-runs/qwen35-0p8b-respin-equivalent-gemmafix-r16-a0p05-20260513T151343Z`

Command:

```bash
python3 tools/respin_render_attribution.py \
  --run-dir /home/rob/dq-runs/qwen35-0p8b-progressive-gates-v2-20260512T224854Z \
  --model /home/rob/.cache/huggingface/qwen35-0p8b-bf16 \
  --fold-model /home/rob/dq-runs/qwen35-0p8b-respin-foldonly-gemmafix-a0-20260513T151057Z \
  --respin-model /home/rob/dq-runs/qwen35-0p8b-respin-equivalent-gemmafix-r16-a0p05-20260513T151343Z
```

Outputs:

- `artifacts/respin_render_attribution.json`
- `artifacts/respin_render_attribution.csv`
- `artifacts/respin_render_attribution_by_layer.csv`
- `artifacts/respin_render_attribution.log`

Metric: Fisher-weighted output MSE when h-detail rows are available, otherwise
output MSE. Lower is better. The run scored 150 body NVFP4 Linears. PrismaClip
was not replayed because existing thresholds were solved in the non-ReSpin
activation basis and must be re-solved under the ReSpin basis.

## Aggregate Result

| Arm | Score | Delta vs baseline |
|---|---:|---:|
| Baseline static NVFP4 | 0.362675269 | - |
| Fold-only static NVFP4 | 0.370724078 | +0.008048808 worse |
| Random-Givens static NVFP4 | 0.370699780 | +0.008024511 worse |
| Random-Givens + progressive methods | 0.026653499 | -0.336021770 better |

The low-rank rotation itself recovered only `0.000024297` versus fold-only.
That is effectively noise relative to the fold-induced regression and the
local-method gains.

## Post-Rotation Local Methods

| Method | Attempts | Accepted | Score reduction | Relative reduction |
|---|---:|---:|---:|---:|
| FourOverSix | 150 | 143 | 0.050725289 | 13.68% |
| Fisher-GPTQ | 150 | 148 | 0.293269105 | 91.66% |
| Scale sweep | 150 | 4 | 0.000040654 | 0.15% |

Compared with the non-rotation replay from the same run, the random-Givens arm
worsened the final no-clip local score (`0.026653499` vs `0.020076996`). This
does not prove KL will regress, but it is a negative local signal for this
untrained basis.

## Layer Readout

Largest total reductions after random-Givens rotation + local methods:

| Layer | Qnames | Baseline | Rotation static | Final | Total reduction |
|---:|---:|---:|---:|---:|---:|
| 18 | 6 | 0.028832 | 0.029358 | 0.001796 | 0.027036 |
| 20 | 6 | 0.027252 | 0.027472 | 0.002100 | 0.025152 |
| 21 | 6 | 0.026874 | 0.027856 | 0.002316 | 0.024558 |
| 17 | 6 | 0.025045 | 0.025253 | 0.001413 | 0.023632 |
| 22 | 6 | 0.020974 | 0.021444 | 0.001566 | 0.019408 |

Best rotation-static layers were small and sparse:

| Layer | Baseline | Rotation static | Rotation delta |
|---:|---:|---:|---:|
| 7 | 0.007320 | 0.007214 | +0.000106 |
| 3 | 0.015793 | 0.015780 | +0.000013 |

Worst rotation-static layers:

| Layer | Baseline | Rotation static | Rotation delta |
|---:|---:|---:|---:|
| 21 | 0.026874 | 0.027856 | -0.000983 |
| 23 | 0.016781 | 0.017656 | -0.000874 |
| 8 | 0.009973 | 0.010573 | -0.000601 |

## Interpretation

This smoke validates the runtime/plugin and attribution machinery, but the
measured numeric effect is not attractive. The tested rank-16
alternating-Givens basis does not improve the static NVFP4 local objective on
0.8B. The useful work still comes from FourOverSix and Fisher-GPTQ. Treat this
configuration as research-only. A paper-faithful ReSpinQuant attempt must first
train the rotations, then write the SVD/polar residual adapters, then re-run the
Pareto probe and KL.
