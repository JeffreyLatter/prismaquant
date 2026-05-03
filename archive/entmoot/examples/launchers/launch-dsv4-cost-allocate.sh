#!/usr/bin/env bash
# DSv4-Flash-Base: cost step + allocator on the partial-merged probe
# (19/32 chunks). Picks up from /work/artifacts/probe.pkl produced by
# merge_partial.py.
set -euo pipefail

if docker ps --format '{{.Names}}' | grep -E '^pq-' | head -1; then
    echo "ABORT: another pq-* container running."
    exit 1
fi

HOST_SNAP=$(ls -d /home/rob/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Base/snapshots/*/ 2>/dev/null | head -1)
[ -z "$HOST_SNAP" ] && { echo "DSv4-Flash-Base snapshot missing"; exit 1; }
CONTAINER_SNAP=${HOST_SNAP/\/home\/rob\/.cache\/huggingface/\/hfcache}
CONTAINER_SNAP=${CONTAINER_SNAP%/}

HOST_WORK=/home/rob/dq-runs/dsv4-flash-base
[ -f "$HOST_WORK/artifacts/probe.pkl" ] || { echo "merged probe.pkl missing"; exit 1; }

mkdir -p "$HOST_WORK"/{cost_work,logs}
docker rm -f pq-deepseek-v4-cost 2>/dev/null || true

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-deepseek-v4-cost \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v /models:/models \
  -v "$HOST_WORK":/work \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e PYTHONPATH=/prismaquant \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  -e PRISMAQUANT_DIRECT_CUDA_LOAD=1 \
  -e PRISMAQUANT_COST_PREFETCH_ACT=1 \
  -w /prismaquant \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 2>&1 | tail -1

    MODEL='$CONTAINER_SNAP'
    PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl

    echo '[1/2] cost step ...'
    python3 -m prismaquant.incremental_measure_quant_cost \\
      --model \"\$MODEL\" \\
      --probe \"\$PROBE\" \\
      --activation-cache-dir /work/act \\
      --output \"\$COST\" \\
      --work-dir /work/cost_work \\
      --device cuda --dtype bf16 \\
      --formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \\
      --mode batched --chunk-size 64 \\
      --layers-per-shard 2 \\
      --swap-grow-limit-mb 8192 \\
      --min-mem-available-mb 1024 \\
      --no-include-mtp --no-include-visual --no-include-lm-head \\
      --skip-missing-activations \\
      2>&1 | tee /work/logs/cost.log

    echo '[2/2] allocator (3-5 bpp pareto, NVFP4 + aggressive expert prune) ...'
    python3 -m prismaquant.allocator \\
      --model \"\$MODEL\" \\
      --probe \"\$PROBE\" \\
      --costs \"\$COST\" \\
      --formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \\
      --target-bits 4.00 \\
      --pareto-targets 3.00,3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00 \\
      --enable-expert-prune \\
      --prune-ratios 0.0,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5 \\
      --prune-alpha 0.15 \\
      --prune-domain-policy union \\
      --layer-config /work/artifacts/layer_config.json \\
      --pareto-csv /work/artifacts/pareto.csv \\
      2>&1 | tee /work/logs/allocator.log

    echo '[done] cost=' \$COST ' layer_config=/work/artifacts/layer_config.json'
    echo '       pareto=/work/artifacts/pareto.csv'
"

echo "[launch] container: pq-deepseek-v4-cost"
echo "[launch] tail:      docker logs -f pq-deepseek-v4-cost"
echo "[launch]   or:      tail -f $HOST_WORK/logs/cost.log"
