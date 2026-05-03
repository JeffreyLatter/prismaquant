#!/usr/bin/env bash
# DSv4-Flash-Base: allocator-only re-run with extended prune ratios up to
# 0.75 (keep 64/256 experts) to push toward the 90 GB on-disk target
# without reintroducing NVINT2/3. Pareto sweep also extended down to
# 2.5 bpp.
#
# Reuses /work/artifacts/probe.pkl + /work/artifacts/cost.pkl from prior
# runs. Saves the prior R=0.5 pareto/layer_config as *.r05.* in advance.
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
[ -f "$HOST_WORK/artifacts/probe.pkl" ] || { echo "probe.pkl missing"; exit 1; }
[ -f "$HOST_WORK/artifacts/cost.pkl"  ] || { echo "cost.pkl missing";  exit 1; }

mkdir -p "$HOST_WORK"/logs
docker rm -f pq-deepseek-v4-alloc 2>/dev/null || true

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-deepseek-v4-alloc \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v /models:/models \
  -v "$HOST_WORK":/work \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e PYTHONPATH=/prismaquant \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  -w /prismaquant \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 2>&1 | tail -1

    MODEL='$CONTAINER_SNAP'
    PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl

    echo '[allocator] final pick: target=2.60 bpp (~91.5 GB on disk, R=0.5625) ...'
    python3 -m prismaquant.allocator \\
      --model \"\$MODEL\" \\
      --probe \"\$PROBE\" \\
      --costs \"\$COST\" \\
      --formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \\
      --target-bits 2.60 \\
      --enable-expert-prune \\
      --prune-ratios 0.0,0.125,0.25,0.375,0.4375,0.5,0.5625,0.625,0.6875,0.75 \\
      --prune-alpha 0.15 \\
      --prune-domain-policy union \\
      --layer-config /work/artifacts/layer_config.json \\
      --pareto-csv /work/artifacts/pareto_t26.csv \\
      2>&1 | tee /work/logs/allocator_t26.log

    echo '[done] layer_config=/work/artifacts/layer_config.json'
    echo '       pareto=/work/artifacts/pareto.csv'
"

echo "[launch] container: pq-deepseek-v4-alloc"
echo "[launch] tail:      docker logs -f pq-deepseek-v4-alloc"
echo "[launch]   or:      tail -f $HOST_WORK/logs/allocator_t26.log"
