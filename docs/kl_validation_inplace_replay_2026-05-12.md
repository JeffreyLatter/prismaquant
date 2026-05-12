# KL Validation In-Place Replay Smoke - 2026-05-12

## Why

The 27B assignment KL replay path previously became host-memory/NVMe bound
when `validate_assignments_kl` preloaded every rendered production-cache
tensor and then used perturbation hooks that clone/restore weights. A n=16
attempt oversubscribed host memory and swap.

The replacement path keeps the existing `ProductionWeightCache` as the source
of rendered weights, but for a single assignment:

1. Prefetch BF16 source safetensors into the OS page cache.
2. Cache BF16 reference logits.
3. Prefetch production-cache files into the OS page cache without `torch.load`.
4. Destructively copy rendered assignment weights into the live CUDA model.
5. Reuse `PerturbedActivationCache` only for activation quantization, with
   capture disabled and external weight management enabled.

This avoids a second rendered-weight cache and avoids resident tensor preload
of the full production cache.

## 27B Smoke

Model:
`/home/rob/.cache/huggingface/qwen36-27b-bf16`

Assignment:
`/home/rob/dq-runs/qwen36-27b-kneedle-export-gated-20260512T044940Z/artifacts/layer_config_kneedle_runtime_assignment.json`

Production cache:
`/home/rob/dq-runs/qwen3p6-27b-kl-probe-triad-n64-production-20260508T032958Z-directpy/production_weight_cache_nvfp4_mxfp8.pkl`

Cache dir:
`/home/rob/dq-runs/qwen3p6-27b-kl-probe-triad-n64-production-20260508T032958Z-directpy/production_weight_cache`

Calibration:
`/home/rob/dq-runs/calibration/diverse-v1.jsonl`, n=16, seqlen=2048,
last-token KL.

Command shape:

```bash
python3 -m prismaquant.validate_assignments_kl \
  --model /home/rob/.cache/huggingface/qwen36-27b-bf16 \
  --probe /home/rob/dq-runs/qwen36-27b-kneedle-export-gated-20260512T044940Z/artifacts/probe_stats_from_seedonly.pkl \
  --base-assignment /home/rob/dq-runs/qwen36-27b-kneedle-export-gated-20260512T044940Z/artifacts/layer_config_kneedle_runtime_assignment.json \
  --assignment runtime=/home/rob/dq-runs/qwen36-27b-kneedle-export-gated-20260512T044940Z/artifacts/layer_config_kneedle_runtime_assignment.json \
  --dataset /home/rob/dq-runs/calibration/diverse-v1.jsonl \
  --n-calib-samples 16 --calib-seqlen 2048 \
  --kl-scope last_token --kl-cuda-graphs off \
  --assignment-materialization inplace \
  --source-prefetch require --source-prefetch-workers 4 \
  --source-prefetch-headroom-gb 24 \
  --production-weight-cache <cache.pkl> \
  --production-cache-dir-override <cache-dir> \
  --production-cache-lru-gb 4 \
  --production-cache-prefetch require \
  --production-cache-file-prefetch-headroom-gb 12
```

Output:
`/home/rob/dq-runs/qwen36-27b-kl-inplace-diverse-n16-20260512T133916Z/artifacts/validate_kl_n16_inplace.json`

Log:
`/home/rob/dq-runs/qwen36-27b-kl-inplace-diverse-n16-20260512T133916Z/logs/validate_kl_n16_inplace.log`

Results:

- KL: `0.0272074366`
- quantizable-body bpp: `4.5872604694`
- format counts: `401 NVFP4`, `11 MXFP8`, `93 BF16`
- source prefetch: `51.75 GiB` in `7.1s`
- production-cache file prefetch: `45.32 GiB` in `19.2s`
- in-place materialization: `45.32 GiB` copied in `44.3s`
- swap stayed flat at roughly `240 MiB` used; no production-cache tensor
  preload was used.

## Notes

This fixes the memory-residency failure mode. The remaining setup cost is
`torch.load` deserialization of `.pt` cache shards during materialization.
If this becomes the bottleneck for repeated 27B validation, the next
infrastructure improvement should be storing production-cache shards in a
faster tensor format that preserves `ProductionWeightCache` ownership.

Final-code n=4 smoke after narrowing activation hooks to quantized Linears
only:

- Output:
  `/home/rob/dq-runs/qwen36-27b-kl-inplace-final-smoke-20260512T134542Z/artifacts/validate_kl_n4_inplace.json`
- Log:
  `/home/rob/dq-runs/qwen36-27b-kl-inplace-final-smoke-20260512T134542Z/logs/validate_kl_n4_inplace.log`
- KL: `0.0007658131`
- activation-hook plans: `412`
- materialization: `45.32 GiB` copied in `35.8s`
