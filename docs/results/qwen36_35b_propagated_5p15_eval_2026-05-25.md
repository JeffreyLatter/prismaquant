# Qwen3.6 35B Propagated 5.15 Evaluation

Date: 2026-05-25

Purpose: materialize the propagated-sensitivity 5.15-bpp candidate and compare
it against the shipped 4.75 artifact and recent 5.15/5.16 candidates on the
same vLLM KL and WikiText PPL checks.

## Candidate

- Run root: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z`
- Source sensitivity report: `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_group_sensitivity_all_n4s512.json`
- Base assignment: `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/layer_config.json`
- Candidate layer config: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/artifacts/layer_config.json`
- Exported model: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/exported`
- Actual bpp: 5.149831
- Selected groups: 62 groups, 137 Linear members
- Selected group categories: 35 shared_expert, 19 linear_attn, 8 self_attn
- Final format mix: BF16 393, MXFP8_E4M3 26, NVFP4 92

## Materialization

Recache:

```bash
PYTHONPATH=. \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m prismaquant.production_recache \
  --model /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409 \
  --layer-config /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/artifacts/layer_config.json \
  --production-weight-cache /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/production_weight_cache_4p7526_recached.pkl \
  --output /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/artifacts/production_weight_cache_propagated_5p15_recached.pkl \
  --work-root /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/artifacts/recache_work \
  --production-cache-prefetch auto \
  --production-cache-lru-gb 16
```

Export:

```bash
PYTHONPATH=. \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m prismaquant.export_native_compressed \
  --model /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409 \
  --layer-config /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/artifacts/layer_config.json \
  --output /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/exported \
  --production-weight-cache /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/artifacts/production_weight_cache_propagated_5p15_recached.pkl \
  --production-cache-lru-gb 64 \
  --production-cache-prefetch-workers 4 \
  --device cuda
```

Native vLLM smoke loaded the artifact with the expected kernels:

- `FlashInferCutlassMxfp8LinearKernel` for MXFP8 GEMM
- `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM

The smoke prompt `The capital of France is` completed successfully.

## Evaluation Commands

Full-vocab KL:

```bash
PYTHONPATH=/home/rob/prismaquant HF_HUB_ENABLE_HF_TRANSFER=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn FLASHINFER_DISABLE_VERSION_CHECK=1 \
  TRITON_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/metrics/triton_cache \
  TORCHINDUCTOR_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/metrics/torchinductor_cache \
  /home/rob/dq-runs/venvs/prismaquant-vllm-kl-20260521/bin/python \
  tools/measure_vllm_full_kl.py \
  --mode student \
  --model /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/exported \
  --teacher-payload /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/metrics/bf16_teacher_wikitext_n8_s512.pt \
  --output /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/metrics/propagated_5p15_vllm_kl_vs_bf16_wikitext_n8_s512.json \
  --dataset-cache-dir /hfcache/datasets \
  --n-samples 8 \
  --seqlen 512 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.58 \
  --max-logprobs 248320 \
  --enforce-eager
```

WikiText PPL:

```bash
PYTHONPATH=/home/rob/prismaquant HF_HUB_ENABLE_HF_TRANSFER=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn FLASHINFER_DISABLE_VERSION_CHECK=1 \
  TRITON_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/metrics/triton_cache \
  TORCHINDUCTOR_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/metrics/torchinductor_cache \
  /home/rob/dq-runs/venvs/prismaquant-vllm-kl-20260521/bin/python \
  tools/measure_vllm_wikitext_ppl.py \
  --model /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/exported \
  --output /home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/metrics/propagated_5p15_vllm_wikitext_ppl_8192_s512.json \
  --dataset-cache-dir /home/rob/.cache/huggingface/datasets \
  --split test \
  --n-tokens 8192 \
  --seqlen 512 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.58 \
  --enforce-eager
```

## Results

| Artifact | bpp | KL mean | PPL | Mean NLL |
|---|---:|---:|---:|---:|
| Shipped 4.75 | 4.75 | 0.0671039 | 9.63952 | 2.26587 |
| Current kneedle 5.1569 | 5.1569 | 0.0642787 | 9.51186 | 2.25254 |
| Phase 1 local-MSE 5.15 | 5.15 | 0.0897760 | 9.81468 | 2.28388 |
| Propagated-sensitivity 5.15 | 5.149831 | 0.0487880 | 9.37146 | 2.23767 |

Against shipped 4.75, the propagated 5.15 candidate is 27.3% lower KL and
0.268 lower PPL.  It also beats the previous current 5.1569 candidate on both
metrics, despite being slightly lower bpp.

Against the Phase 1 local-MSE 5.15 candidate, the propagated version reverses
the regression: KL drops from 0.0897760 to 0.0487880, and PPL drops from 9.81468
to 9.37146.

## Logs

- Recache: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/logs/production_recache.log`
- Export: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/logs/export.log`
- vLLM smoke: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/logs/validate_native_export.log`
- KL: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/logs/measure_kl_propagated_5p15.log`
- PPL: `/home/rob/dq-runs/qwen36-35b-propagated-5p15-20260525T030334Z/logs/measure_ppl_propagated_5p15.log`

## Conclusion

The propagated-sensitivity allocation is the first current 35B candidate in
this pass that clearly beats the shipped artifact on both vLLM KL and WikiText
PPL.  It is not an equal-bpp replacement for shipped 4.75; it spends about 0.40
additional bpp.  At the 5.15-bpp target, however, it is materially better than
both the shipped artifact and the earlier local-MSE promotion attempt.
