# AWQ / HALO / SmoothQuant Full-Stack 4B Read - 2026-05-11

Purpose: test whether fold-scale or rotation methods improve the current
Qwen3-4B production numerical stack before spending 27B time.

## Contract

- Model: `/home/rob/.cache/huggingface/Qwen3-4B`
- HALO model: `/home/rob/dq-runs/qwen3-4b-untied-bf16`
- Baseline run root:
  `/home/rob/dq-runs/fouroversix-smoke-20260510T225344Z/qwen3-4b`
- Assignment: fixed `225 NVFP4`, `7 MXFP8`, `20 BF16`
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Calibration: `n=16`, `seqlen=1024`, seed `42`
- Production cache: assignment-scoped, required resident prefetch
- Numerical stack: GPTQ + damp sweep + scale sweep + FourOverSix NVFP4 scale
  rule + full activation clip solver
- Rotations: off except the HALO arm
- bpp: `4.99963924963925`

## Results

| Arm | Run root | KL | Relative to baseline |
|---|---|---:|---:|
| FourOverSix + clip baseline | `/home/rob/dq-runs/fouroversix-smoke-20260510T225344Z/qwen3-4b` | `0.056417932` | - |
| AWQ-v2 + full stack | `/home/rob/dq-runs/qwen3-4b-four6-clip-awq-pseudosearch-20260511T051111Z` | `0.225788253` | `+300.2%` worse |
| HALO + full stack | `/home/rob/dq-runs/qwen3-4b-four6-clip-halo-gpuact-20260511T063430Z` | `0.177460096` | `+214.5%` worse |
| SmoothQuant + full stack | `/home/rob/dq-runs/qwen3-4b-four6-clip-smoothquant-20260511T075939Z` | `0.167588803` | `+197.0%` worse |

All KL validations prefetched `232/232` production-cache entries, `6.50 GiB`
resident, with `0` missing.

## Notes

- AWQ candidate search was changed to avoid rerunning GPTQ for every candidate:
  `PRISMAQUANT_AWQ_SEARCH_GPTQ=0`,
  `PRISMAQUANT_AWQ_SEARCH_SCALE_SWEEP=0`. The selected scale is still rendered
  once through the full production stack.
- HALO used an untied source checkpoint and a block-Hadamard rotation with
  `dim=2560`, `block_sizes=[2048, 512]`, rotation hash
  `6766c8b5bc5c5bdd`.
- Production cache activation capture now stays CUDA-resident for all
  activation-aware levers, not just AWQ. The HALO rerun confirmed
  `activation_capture store_device=cuda:0`.
- AWQ selected only one group, layer 29 `q/k/v` (`3` Linears), with local
  pseudo-rendered MSE gain `3.50%`.
- SmoothQuant selected only one group, layer 28 `q/k/v` (`3` Linears), with
  local pseudo-rendered MSE gain `3.64%`.
- Those selected groups are in the MXFP8 attention promotion region, so these
  tests did apply fold-scale methods where they make sense for MXFP8.

## SmoothQuant Follow-Up Attribution

Run root:
`/home/rob/dq-runs/qwen3-4b-smoothquant-targeted-20260511T113334Z`.

The full SmoothQuant cache differed from the baseline cache in exactly six
files: the intended layer 28 MXFP8 `q/k/v` files plus three unrelated NVFP4
PrismaClip rerender files.

| Isolated cache variant | KL |
|---|---:|
| Baseline manifest | `0.056417932` |
| SmoothQuant layer 28 `q/k/v` only | `0.055455598` |
| Three unrelated NVFP4 rerender files only | `0.163882268` |
| All six differing files, no SmoothQuant metadata | `0.167588803` |
| All six differing files, with SmoothQuant scales/metadata | `0.167588803` |

Single-file NVFP4 isolation:

| NVFP4 file replaced from second render | KL |
|---|---:|
| `model.layers.8.self_attn.o_proj` | `0.133624067` |
| `model.layers.14.mlp.gate_proj` | `0.081835703` |
| `model.layers.14.mlp.up_proj` | `0.067776848` |
| `model.layers.14.mlp.gate_proj` + `up_proj` | `0.061268054` |

Conclusion: the `0.167588803` SmoothQuant full-cache result is not evidence
that the selected SmoothQuant MXFP8 `q/k/v` fold regressed. The selected fold
slightly improved KL in isolation. The regression was caused by unstable
PrismaClip NVFP4 rerender choices with sub-0.2% local-MSE gains. A later 4B
isolation run showed that making `PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN=0.002`
the default removed many collectively useful clips, so the production default
returned to `0.0`; nonzero floors are ablation knobs only.

## Decision

Keep AWQ, HALO, and full-cache SmoothQuant rerenders out of production recipes
for this stack until they pass the follow-up gates. Do not combine HALO+AWQ:
both individual arms regressed heavily. SmoothQuant is still a research lever:
the isolated MXFP8 layer 28 `q/k/v` replacement was slightly positive, but it
needs a rerun under the stricter PrismaClip gate and then an allocation-aware
validation before promotion.

The current production-facing winner remains:

```text
GPTQ + damp sweep + scale sweep + FourOverSix + full activation clip solver
```
