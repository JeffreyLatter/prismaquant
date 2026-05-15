# Qwen3.6-27B FP8 Frontier Check (2026-05-13)

## Question

The `MXFP8_E4M3` kernel cannot serve the skinny linear-attention
`in_proj_a/b` matrices with shape `[48, 5120]`. We cannot change model
dimensions, so the production question was whether a BF8/plain-FP8 fallback
should be available while keeping BF16 in the allocator menu.

Local vLLM inspection showed the compressed-tensors W8A8 FP8 path is
`torch.float8_e4m3fn` based. It does not expose an E5M2/BF8 weight path in the
installed serving image. Therefore the production-backed candidate is
`FP8_E4M3`, not `FP8_E5M2`.

## Run

Base run:

`/home/rob/dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z`

The original probe, activation cache, and h-detail cache were reused. Only the
cost table was regenerated to add `FP8_E4M3`:

```bash
python3 -m prismaquant.incremental_measure_quant_cost \
  --model /home/rob/.cache/huggingface/qwen36-27b-bf16 \
  --probe .../artifacts/probe.pkl \
  --activation-cache-dir .../act \
  --formats NVFP4,MXFP8_E4M3,FP8_E4M3,BF16 \
  --output .../artifacts/cost_fp8menu.pkl \
  --work-dir .../work_fp8menu \
  --device cuda --dtype bf16 --mode batched --chunk-size 256 \
  --layers-per-shard 4 --swap-grow-limit-mb 2048 \
  --min-mem-available-mb 32768 \
  --skip-missing-activations --include-mtp --include-visual \
  --no-include-lm-head --h-detail-dir .../h_detail
```

Allocator command used the same BF16-inclusive menu:

```bash
PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR=1 \
python3 -m prismaquant.allocator \
  --probe .../artifacts/probe.pkl \
  --costs .../artifacts/cost_fp8menu.pkl \
  --formats NVFP4,MXFP8_E4M3,FP8_E4M3,BF16 \
  --target-bits 4.75 \
  --pareto-targets 4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25 \
  --target-profile vllm_qwen3_5_packed_moe \
  --visual-format BF16 --visual-sensitivity uniform --mtp-format BF16 \
  --layer-config .../artifacts/layer_config_fp8menu.json \
  --pareto-csv .../artifacts/pareto_fp8menu.csv \
  --pareto-output-dir .../artifacts/pareto_fp8menu_assignments \
  --applicability-report .../artifacts/format_applicability_fp8menu.json
```

Logs:

- `.../logs/cost_fp8menu.log`
- `.../logs/allocator_fp8menu.log`

## Result

| target bpp | achieved bpp | predicted loss | NVFP4 | FP8_E4M3 | MXFP8_E4M3 | BF16 |
|---:|---:|---:|---:|---:|---:|---:|
| 4.75 | 4.749 | 1232.8 | 243 | 54 | 0 | 12 |
| 5.00 | 4.999 | 892.7 | 239 | 66 | 0 | 4 |

Previous BF16-inclusive menu without plain FP8:

| target bpp | achieved bpp | predicted loss | NVFP4 | MXFP8_E4M3 | BF16 |
|---:|---:|---:|---:|---:|---:|
| 4.75 | 4.749 | 1647.2 | 255 | 0 | 54 |
| 5.00 | 5.001 | 1261.0 | 254 | 0 | 55 |

At 4.75 bpp, adding `FP8_E4M3` lowers the allocator objective by about 25%
while preserving BF16 fallback. At 5.00 bpp, it lowers the objective by about
29%.

Shape applicability:

- pre-aggregation: `BF16=504`, `FP8_E4M3=504`, `MXFP8_E4M3=408`, `NVFP4=504`
- post-aggregation: `BF16=309`, `FP8_E4M3=309`, `MXFP8_E4M3=261`, `NVFP4=309`
- `MXFP8_E4M3` masks: `96` Linears, all due to `kernel_shape`

The expanded 4.75 layer config contains `70` `FP8_E4M3` selections on the
skinny `[48, 5120]` linear-attention `in_proj_a/b` matrices. These are exactly
the shapes `MXFP8_E4M3` cannot cover.

## Decision

Keep `FP8_E4M3` in the production menu for Qwen3.6-27B. Do not enable
`FP8_E5M2` / BF8 naming yet: the installed vLLM image exposes E4M3 for the
compressed-tensors FP8 path, and E5M2 remains research-only until a vLLM load
and kernel-path smoke proves it is real.
