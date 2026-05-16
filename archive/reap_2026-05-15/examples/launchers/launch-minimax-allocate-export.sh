#!/usr/bin/env bash
# Re-run allocator at target-bits=3.20 (the in-band knee for 90-95 GB)
# then export the artifact for vLLM serving.
set -euo pipefail

HOST_SNAP=$(ls -d /home/rob/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*/ | head -1)
CONTAINER_SNAP=${HOST_SNAP/\/home\/rob\/.cache\/huggingface/\/hfcache}
CONTAINER_SNAP=${CONTAINER_SNAP%/}

HOST_WORK=/home/rob/dq-runs/minimax-m2p7-v21

docker rm -f pq-minimax-v21-export 2>/dev/null || true

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-minimax-v21-export \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v "$HOST_WORK":/work \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  -e PRISMAQUANT_BATCHED_NVFP4_EXPORT=1 \
  -e PRISMAQUANT_COST_PREFETCH_ACT=1 \
  -w /prismaquant \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 'transformers==4.57.5' 2>&1 | tail -1

    MODEL='$CONTAINER_SNAP'
    LAYER_CFG=/work/artifacts/layer_config_prune_t320.json
    EXPORT_DIR=/work/exported

    echo '[1/2] allocator at target-bits=3.20 ...'
    python3 -m prismaquant.allocator \
      --model \"\$MODEL\" \
      --probe /work/artifacts/probe.pkl \
      --costs /work/artifacts/cost.pkl \
      --formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \
      --target-bits 3.20 \
      --pareto-targets 3.10,3.16,3.20,3.25,3.30,3.40,3.50,3.60 \
      --enable-expert-prune \
      --prune-ratios 0.0,0.125,0.1875,0.25,0.3125,0.375 \
      --prune-alpha 0.15 \
      --prune-domain-policy union \
      --layer-config \"\$LAYER_CFG\" \
      --pareto-csv /work/artifacts/pareto_prune_t320.csv \
      2>&1 | tee /work/logs/allocator_t320.log

    echo '[2/2] export ...'
    mkdir -p \"\$EXPORT_DIR\"
    python3 -m prismaquant.export_native_compressed \
      --model \"\$MODEL\" \
      --layer-config \"\$LAYER_CFG\" \
      --prune-manifest \"\$LAYER_CFG.prune.json\" \
      --output \"\$EXPORT_DIR\" \
      --activation-cache-dir /work/act \
      --device cuda \
      2>&1 | tee /work/logs/export_t320.log

    echo '[done] artifact at' \"\$EXPORT_DIR\"
"

echo "[launch] container: pq-minimax-v21-export"
echo "[launch] tail allocator: tail -f $HOST_WORK/logs/allocator_t320.log"
echo "[launch] tail export:    tail -f $HOST_WORK/logs/export_t320.log"
