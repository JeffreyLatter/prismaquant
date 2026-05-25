# Qwen3.6 35B Propagated Group Sensitivity

Date: 2026-05-25

Purpose: measure whether MSE-promotion groups are also sensitive under paired
propagated KL.  Each candidate group is measured as current production-rendered
format versus the same group promoted to BF16, with all surrounding modules at
the fixed 4.7526 strategic assignment.

## Inputs

- Model: `/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409`
- Base assignment: `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/layer_config.json`
- Cost: `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/cost.pkl`
- Probe: `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/probe.pkl`
- Production cache: `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/production_weight_cache_4p7526_recached.pkl`

## Command

```bash
PYTHONPATH=. PRISMAQUANT_L3_CUDA_GRAPHS=0 \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  tools/sensitivity_propagated_group_report.py \
  --model /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409 \
  --base-assignment /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/layer_config.json \
  --costs /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/cost.pkl \
  --probe /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/probe.pkl \
  --production-weight-cache /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/production_weight_cache_4p7526_recached.pkl \
  --output-report /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_group_sensitivity_all_n4s512.json \
  --work-root /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_group_work \
  --n-calib-samples 4 \
  --calib-seqlen 512 \
  --max-lanes-per-batch 4 \
  --production-cache-prefetch file-pages \
  --production-cache-lru-gb 8
```

## Results

- Full report: `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_group_sensitivity_all_n4s512.json`
- Measured groups: 75/75
- Calibration: 4 samples, 512 tokens
- Runtime after calibration/model setup: 687.6 seconds
- Base bpp: 5.028082
- Candidate categories: 35 shared_expert, 30 linear_attn, 10 self_attn
- Top-20 propagated categories: 19 shared_expert, 1 linear_attn
- Top-20 local-MSE categories: 20 shared_expert

Top propagated KL per added bit:

| Rank | Group | Category | Local Rank | KL | KL/bit |
|---:|---|---|---:|---:|---:|
| 1 | shared_expert.layer_33 | shared_expert | 32 | 0.0147829 | 1.819e-09 |
| 2 | shared_expert.layer_36 | shared_expert | 28 | 0.0069671 | 8.573e-10 |
| 3 | shared_expert.layer_2 | shared_expert | 42 | 0.0161959 | 5.721e-10 |
| 4 | linear_attn.layer_9 | linear_attn | 72 | 0.0353402 | 3.663e-10 |
| 5 | shared_expert.layer_16 | shared_expert | 17 | 0.0063471 | 2.242e-10 |
| 6 | shared_expert.layer_21 | shared_expert | 13 | 0.0050470 | 1.783e-10 |
| 7 | shared_expert.layer_11 | shared_expert | 24 | 0.0046399 | 1.639e-10 |
| 8 | shared_expert.layer_8 | shared_expert | 21 | 0.0045673 | 1.613e-10 |
| 9 | shared_expert.layer_17 | shared_expert | 19 | 0.0045162 | 1.595e-10 |
| 10 | shared_expert.layer_1 | shared_expert | 40 | 0.0044382 | 1.568e-10 |

The clearest cross-layer signal is that local MSE alone misses some sensitive
attention groups.  `linear_attn.layer_9` is local-MSE rank 72 but propagated
rank 4.  `self_attn.layer_11` is local-MSE rank 66 but propagated rank 22.

A greedy 5.15-bpp selection by propagated KL per added bit would spend
4,196,139,008 added bits, reaching 5.149831 bpp.  It selects 62 groups:
35 shared_expert, 19 linear_attn, and 8 self_attn.  That is materially broader
than the local-MSE-only Phase 1 selection and gives attention a quantitative
path into the allocation.

## Notes

The initial smoke with the default frozen context cache tripped the CUDA memory
guard at 116.92 GB used against a 114.95 GB budget.  The report tool therefore
defaults that cache off and uses the production cache plus LRU/file-page
prefetch path.  The completed runs used strict production-cache mode; no RTN
fallback was allowed.

## Verification

```bash
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m py_compile \
  prismaquant/kl_measurement.py \
  prismaquant/mse_promotion.py \
  tools/sensitivity_propagated_group_report.py

/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q \
  tests/test_mse_promotion.py \
  tests/test_kl_measurement_override_cache.py
```

Result: 6 passed.
