# Qwen3.6-35B MSE Promotion Phase 1, 2026-05-25

Work dir:

`/home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z`

Source model:

`/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409`

Candidate assignment:

`/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/attn_shared_target_5p15_layer_config.json`

## Candidate

The assignment was produced by MSE promotion from the current strategic 4.75
assignment, promoting `linear_attn`, `self_attn`, and `shared_expert` groups to
BF16 until the tool-reported quantizable bpp reached 5.15.

- Base bpp: 5.028082036266279
- Promoted bpp: 5.149785774993715
- Delta bpp: 0.12170373872743519
- Selected groups: 47
- Selected Linears: 135
- Stored local output MSE removed: 86.1406035459196%
- Format counts before: BF16 146, MXFP8_E4M3 112, NVFP4 143
- Format counts after: BF16 281, MXFP8_E4M3 14, NVFP4 106

Category contribution:

| category | groups | linears | output MSE removed |
|---|---:|---:|---:|
| linear_attn | 11 | 33 | 55.57333803356115% |
| shared_expert | 35 | 101 | 30.454945296768187% |
| self_attn | 1 | 1 | 0.11232021559026611% |

## Commands

Production recache:

```bash
PYTHONPATH=/home/rob/prismaquant PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  -m prismaquant.production_recache \
  --model /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409 \
  --layer-config /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/artifacts/layer_config.json \
  --production-weight-cache /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/production_weight_cache_frontier_raw.pkl \
  --output /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/artifacts/production_weight_cache_attn_shared_5p15_recached.pkl \
  --cache-dir-override /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/production_weight_cache_frontier \
  --production-cache-lru-gb 64 \
  --work-root /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/artifacts/recache_work \
  --dtype bf16 --device cuda \
  --production-cache-prefetch require \
  --production-cache-prefetch-workers 4
```

Export:

```bash
PYTHONPATH=/home/rob/prismaquant PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  -m prismaquant.export_native_compressed \
  --model /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409 \
  --layer-config /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/artifacts/layer_config.json \
  --output /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/exported \
  --device cuda \
  --production-weight-cache /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/artifacts/production_weight_cache_attn_shared_5p15_recached.pkl \
  --production-cache-dir-override /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/production_weight_cache_frontier \
  --production-cache-lru-gb 64 \
  --production-cache-prefetch-workers 4 \
  --gptq --scale-sweep --no-gptq-static-act-order --no-gptq-joint-scale-opt
```

vLLM validation:

```bash
PYTHONPATH=/home/rob/prismaquant HF_HUB_ENABLE_HF_TRANSFER=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  /home/rob/dq-runs/venvs/prismaquant-vllm-kl-20260521/bin/python \
  -m prismaquant.validate_native_export \
  --model /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/exported \
  --gpu-memory-utilization 0.58 \
  --max-model-len 1024 \
  --max-new-tokens 16
```

Full-vocab KL:

```bash
PYTHONPATH=/home/rob/prismaquant HF_HUB_ENABLE_HF_TRANSFER=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn FLASHINFER_DISABLE_VERSION_CHECK=1 \
  TRITON_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/metrics/triton_cache \
  TORCHINDUCTOR_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/metrics/torchinductor_cache \
  /home/rob/dq-runs/venvs/prismaquant-vllm-kl-20260521/bin/python \
  tools/measure_vllm_full_kl.py \
  --mode student \
  --model /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/exported \
  --teacher-payload /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/metrics/bf16_teacher_wikitext_n8_s512.pt \
  --output /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/metrics/attn_shared_5p15_vllm_kl_vs_bf16_wikitext_n8_s512.json \
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
  TRITON_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/metrics/triton_cache \
  TORCHINDUCTOR_CACHE_DIR=/home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/metrics/torchinductor_cache \
  /home/rob/dq-runs/venvs/prismaquant-vllm-kl-20260521/bin/python \
  tools/measure_vllm_wikitext_ppl.py \
  --model /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/exported \
  --output /home/rob/dq-runs/qwen36-35b-mse-promoted-5p15-phase1-20260525T013511Z/metrics/attn_shared_5p15_vllm_wikitext_ppl_8192_s512.json \
  --dataset-cache-dir /home/rob/.cache/huggingface/datasets \
  --split test \
  --n-tokens 8192 \
  --seqlen 512 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.58 \
  --enforce-eager
```

PPL used the local WikiText cache at `/home/rob/.cache/huggingface/datasets`;
`/hfcache/datasets` is not present on this host.

Logs:

- Recache: `logs/production_recache.log`
- Export: `logs/export.log`
- vLLM validation: `logs/validate_native_export.log`
- KL: `logs/measure_kl_attn_shared_5p15.log`
- PPL: `logs/measure_ppl_attn_shared_5p15.log`

## Results

The export loaded in vLLM, routed MXFP8 to FlashInfer CUTLASS MXFP8 GEMM, routed
NVFP4 to FlashInfer CUTLASS NVFP4, and generated a valid smoke continuation.

| artifact | bpp | KL vs BF16 | WikiText PPL | mean NLL |
|---|---:|---:|---:|---:|
| shipped 4.75 | n/a | 0.06710393726825714 | 9.639520482014522 | 2.2658713648556397 |
| current strategic 4.7526 | 4.7526 | 0.12602472305297852 | 9.87271968028455 | 2.2897753656692172 |
| MSE-promoted attention/shared 5.1498 | 5.1498 | 0.08977600932121277 | 9.814681347925863 | 2.2838793613762185 |
| current kneedle 5.1569 | 5.1569 | 0.06427867710590363 | 9.511862119982359 | 2.252539663907303 |

Metric files:

- Candidate KL: `metrics/attn_shared_5p15_vllm_kl_vs_bf16_wikitext_n8_s512.json`
- Candidate PPL: `metrics/attn_shared_5p15_vllm_wikitext_ppl_8192_s512.json`
- Current 5.1569 KL/PPL:
  `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/metrics/`
- Shipped 4.75 KL/PPL:
  `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/metrics/`
- Current strategic 4.7526 KL/PPL:
  `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/metrics/`

## Takeaway

The local-MSE promotion is directionally useful compared with the current
strategic 4.7526 assignment, reducing KL from 0.1260 to 0.0898 and PPL from
9.8727 to 9.8147. It does not beat the shipped 4.75 artifact or the current
5.1569 kneedle. This supports using local MSE as a cheap screen, but not as the
only sensitivity objective.

For phase 2, the next diagnostic should measure paired propagated error per
semantic group and rank by propagated error removed per bit, while reporting
amplification separately.
