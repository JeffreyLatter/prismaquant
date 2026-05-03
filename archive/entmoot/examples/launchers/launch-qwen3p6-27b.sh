#!/usr/bin/env bash
# Serve the published Qwen3.6-27B-PrismaQuant-5.5bit-vllm artifact for
# correctness benchmarking. Community is reporting issues; we want to
# reproduce locally and bisect.
#
# Per memory `container_transformers_pin.md`: do NOT pin transformers
# inside the container — the Qwen3.5/3.6 model_types only exist in
# transformers 5.5.4 (the image default).
set -euo pipefail

# Use the HF cache directly (artifact downloaded via `hf download`).
ARTIFACT_REPO=rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm
HF_SNAP=$(ls -d /home/rob/.cache/huggingface/hub/models--rdtand--Qwen3.6-27B-PrismaQuant-5.5bit-vllm/snapshots/*/ 2>/dev/null | head -1)
[ -z "$HF_SNAP" ] && { echo "Qwen3.6-27B-5.5bit snapshot missing — run hf download first"; exit 1; }
[ -f "$HF_SNAP/config.json" ] || { echo "snapshot incomplete"; exit 1; }

LOG_DIR=/home/rob/dq-runs/qwen3p6-27b-5p5bit/logs
mkdir -p "$LOG_DIR"

FREE_GB=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$FREE_GB" -lt 60 ]; then
    echo "[serve] WARNING: only ${FREE_GB} GB available." >&2
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -E '^pq-(vllm|minimax|deepseek|qwen)' | head -1; then
    echo "ABORT: another pq-* container running."
    exit 1
fi

docker rm -f pq-vllm-qwen3p6-27b 2>/dev/null || true

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-vllm-qwen3p6-27b \
  -p 8000:8000 \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v "$LOG_DIR":/serve_logs \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/serve_logs/hf_modules \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /work \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 2>&1 | tail -1
    # Qwen 3.6 recipe (per upstream README): reasoning-parser qwen3,
    # enable-auto-tool-choice + tool-call-parser qwen3_coder. tp=1 on
    # this single-GB10 box (recipe assumes tp=8); max-model-len capped
    # at 32768 (recipe is 262144).
    vllm serve $ARTIFACT_REPO \
      --served-model-name qwen3p6-27b-5p5bit \
      --quantization compressed-tensors \
      --trust-remote-code \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 32768 \
      --max-num-seqs 8 \
      --gpu-memory-utilization 0.85 \
      --kv-cache-dtype fp8 \
      --enable-prefix-caching \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      2>&1 | tee /serve_logs/vllm_serve.log
"

echo "[serve] container: pq-vllm-qwen3p6-27b  (port 8000, bound 0.0.0.0)"
echo "[serve] tail:      docker logs -f pq-vllm-qwen3p6-27b"
echo "[serve]   or:      tail -f $LOG_DIR/vllm_serve.log"
echo
echo "[serve] once 'Application startup complete' appears, smoke-test:"
echo "  curl -s http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"qwen3p6-27b-5p5bit\", \"messages\": [{\"role\":\"user\",\"content\":\"What is 17 * 23?\"}], \"max_tokens\": 64}'"
