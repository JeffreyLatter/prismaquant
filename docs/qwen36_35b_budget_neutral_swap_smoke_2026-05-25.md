# Qwen3.6 35B Budget-Neutral Swap Smoke - 2026-05-25

Purpose: smoke-test the budget-neutral swap measurement path on the current
35B 4.75 serving-unit propagated assignment. This is a small `n=4`,
`seqlen=512` last-token KL run, intended to validate the measurement plumbing
and ranking signal before wider candidate measurement.

Artifacts:

- Swap candidates:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/budget_neutral_swaps_35b_4p75.json`
- Measurement report:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/metrics/budget_neutral_swaps_35b_4p75_smoke_n4_s512.json`
- Log:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/logs/measure_budget_neutral_swaps_smoke_n4_s512.log`

Command:

```bash
RUN=/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z
PYTHONPATH=. PRISMAQUANT_L3_CUDA_GRAPHS=0 \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  tools/measure_budget_neutral_swaps.py \
  --model /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409 \
  --base-assignment /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/layer_config.json \
  --swaps "$RUN/budget_neutral_swaps_35b_4p75.json" \
  --production-weight-cache /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/production_weight_cache_4p7526_recached.pkl \
  --output-report "$RUN/metrics/budget_neutral_swaps_35b_4p75_smoke_n4_s512.json" \
  --work-root "$RUN/work_smoke" \
  --n-calib-samples 4 \
  --calib-seqlen 512 \
  --max-swaps 4 \
  --max-lanes-per-batch 4 \
  --production-cache-prefetch file-pages \
  --production-cache-lru-gb 8
```

Results:

| Rank | Swap key | Delta KL vs BF16 teacher | Swap KL | Base-assignment drift | Net bits |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `tensor:model.layers.5.mlp.shared_expert.down_proj::paid_by::tensor:model.layers.0.linear_attn.out_proj` | -0.006797803 | 0.005556153 | 0.010543754 | -27,525,120 |
| 2 | `tensor:model.layers.5.mlp.shared_expert.down_proj::paid_by::fused:model.layers.15.self_attn.qkv_proj` | -0.001935375 | 0.010418581 | 0.004165379 | -142,344,192 |
| 3 | `tensor:model.layers.5.mlp.shared_expert.down_proj::paid_by::tensor:model.layers.22.mlp.shared_expert.down_proj` | 0.004102279 | 0.016456235 | 0.006649909 | 0 |
| 4 | `tensor:model.layers.5.mlp.shared_expert.down_proj::paid_by::tensor:model.layers.36.linear_attn.out_proj` | 0.012742055 | 0.025096010 | 0.006721876 | -27,525,120 |

Base assignment KL vs BF16 teacher: `0.012353956`.

Interpretation:

- `swap_delta_kl_vs_bf16` is the optimization signal. Negative values improved
  the candidate assignment relative to the base assignment under the same BF16
  teacher.
- `swap_kl_vs_base_assignment` is a drift/safety metric, not the optimization
  objective.
- Two of the four first surrogate-ranked candidates improved the KL; two
  worsened it. The next phase should empirically measure a larger candidate set
  and select by BF16-relative delta, not by surrogate order alone.

Selection output:

```bash
PYTHONPATH=. python3 tools/select_measured_budget_swaps.py \
  --base-assignment /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/layer_config.json \
  --measurement-report /home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/metrics/budget_neutral_swaps_35b_4p75_smoke_n4_s512.json \
  --output-layer-config /home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/artifacts/layer_config_budget_swaps_smoke_n4_s512.json \
  --output-report /home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/metrics/budget_swaps_selection_smoke_n4_s512.json
```

The smoke selector chose one non-conflicting improving swap:
`tensor:model.layers.5.mlp.shared_expert.down_proj::paid_by::tensor:model.layers.0.linear_attn.out_proj`.
It touches two qnames, has net bit delta `-27,525,120`, and estimated
single-swap KL delta `-0.006797803`. The second improving row was skipped
because it promotes the same target qname and therefore conflicts with the
selected row.

Top32 follow-up:

```bash
RUN=/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z
PYTHONPATH=. PRISMAQUANT_L3_CUDA_GRAPHS=0 \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  tools/measure_budget_neutral_swaps.py \
  --model /home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409 \
  --base-assignment /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/layer_config.json \
  --swaps "$RUN/budget_neutral_swaps_35b_4p75.json" \
  --production-weight-cache /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/artifacts/production_weight_cache_4p7526_recached.pkl \
  --output-report "$RUN/metrics/budget_neutral_swaps_35b_4p75_top32_n8_s512.json" \
  --work-root "$RUN/work_top32_n8" \
  --n-calib-samples 8 \
  --calib-seqlen 512 \
  --max-swaps 32 \
  --max-lanes-per-batch 4 \
  --production-cache-prefetch file-pages \
  --production-cache-lru-gb 8
```

Top32 artifacts:

- Measurement report:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/metrics/budget_neutral_swaps_35b_4p75_top32_n8_s512.json`
- Log:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/logs/measure_budget_neutral_swaps_top32_n8_s512.log`
- Selector report:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/metrics/budget_swaps_selection_top32_n8_s512.json`

Top32 results:

- Base assignment KL vs BF16 teacher: `0.015214228`.
- Negative-delta swaps: `0 / 32`.
- Best measured row:
  `tensor:model.layers.20.mlp.shared_expert.down_proj::paid_by::tensor:model.layers.0.linear_attn.out_proj`,
  delta KL `+0.001252776`, swap KL `0.016467003`, base drift
  `0.015027539`, net bits `-27,525,120`.
- Selector output: `selected=0`, `net_bits=0`, estimated selected KL delta
  `0`.

The top32 result supersedes the n=4 smoke for policy decisions. The smoke
proved the measurement path; the n=8 pass says the current budget-neutral
surrogate candidates should not be applied automatically yet.

Top32 n=8 rerun:

- Measurement report:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/metrics/budget_neutral_swaps_35b_4p75_top32_n8_s512_rerun.json`
- Log:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/logs/measure_budget_neutral_swaps_top32_n8_s512_rerun.log`
- Selector report:
  `/home/rob/dq-runs/empirical-budget-swaps-20260525T000000Z/metrics/budget_swaps_selection_top32_n8_s512_rerun.json`

The rerun matched the prior n=8 ranked deltas exactly. Base KL was
`0.015214228`; negative-delta swaps were again `0 / 32`; the best row was again
`tensor:model.layers.20.mlp.shared_expert.down_proj::paid_by::tensor:model.layers.0.linear_attn.out_proj`
with delta KL `+0.001252776`, swap KL `0.016467003`, base drift
`0.015027539`, and net bits `-27,525,120`. Selector output was again
`selected=0`, `net_bits=0`.
