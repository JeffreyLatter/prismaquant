#!/usr/bin/env bash
# DSv4-Flash-Base export at target=2.60 bpp / R=0.5625 / ~91.5 GB on disk.
# Consumes the layer_config.json + prune manifest produced by the
# allocator. Uses --export-cache-dir for resumable export per
# v23_v24_export_speedups memory.
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
[ -f "$HOST_WORK/artifacts/layer_config.json"            ] || { echo "layer_config.json missing"; exit 1; }
[ -f "$HOST_WORK/artifacts/layer_config.json.prune.json" ] || { echo "prune manifest missing"; exit 1; }
[ -d "$HOST_WORK/act" ] || { echo "activation cache dir missing"; exit 1; }

mkdir -p "$HOST_WORK"/{exported,export-cache,logs}
docker rm -f pq-deepseek-v4-export 2>/dev/null || true

AVAIL_GB=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
echo "[launch] host MemAvailable = ${AVAIL_GB} GB"

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-deepseek-v4-export \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v /models:/models \
  -v "$HOST_WORK":/work \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e PYTHONPATH=/prismaquant \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  -e PRISMAQUANT_BATCHED_NVFP4_EXPORT=1 \
  -e PRISMAQUANT_DIRECT_CUDA_LOAD=1 \
  -w /prismaquant \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 2>&1 | tail -1

    MODEL='$CONTAINER_SNAP'
    LAYER_CFG=/work/artifacts/layer_config.json
    EXPORT_DIR=/work/exported

    echo '[export] target=2.60 bpp / R=0.5625 / ~91.5 GB on disk ...'
    python3 -m prismaquant.export_native_compressed \\
      --model \"\$MODEL\" \\
      --layer-config \"\$LAYER_CFG\" \\
      --prune-manifest \"\$LAYER_CFG.prune.json\" \\
      --output \"\$EXPORT_DIR\" \\
      --activation-cache-dir /work/act \\
      --export-cache-dir /work/export-cache \\
      --drop-mtp \\
      --device cuda \\
      2>&1 | tee /work/logs/export_t260.log

    echo '[done] artifact at' \"\$EXPORT_DIR\"
"

echo "[launch] container: pq-deepseek-v4-export"
echo "[launch] tail:      docker logs -f pq-deepseek-v4-export"
echo "[launch]   or:      tail -f $HOST_WORK/logs/export_t260.log"
