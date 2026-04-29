#!/usr/bin/env bash
# Serve the v22/v23 MiniMax M2.7 artifact via vLLM.
# Mirrors the proven launch-minimax-prune-serve.sh param set used for
# the prior low-bit custom-kernel run that hit 3.78 tok/s (per
# session_2026_04_24_3stream_win memory).
#
# Per memory `feedback_vllm_serve_binding`: bind 0.0.0.0:8000 with
# docker -p 8000:8000 so opencode (or any other client) can reach it.
set -euo pipefail

ARTIFACT=/home/rob/dq-runs/minimax-m2p7-v21/exported
LOG_DIR=/home/rob/dq-runs/minimax-m2p7-v21/logs
mkdir -p "$LOG_DIR"

if [ ! -f "$ARTIFACT/config.json" ]; then
    echo "ABORT: $ARTIFACT/config.json missing — export not complete?"
    exit 1
fi

# Pre-flight: refuse to launch if free mem is already low.
FREE_GB=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$FREE_GB" -lt 105 ]; then
    echo "[serve] WARNING: only ${FREE_GB} GB available; need ~105 GB headroom." >&2
    echo "[serve] kill stragglers (docker ps; pkill python; etc.) before retrying." >&2
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -E '^(pq-minimax|pq-vllm)' | head -1; then
    echo "ABORT: another pq-minimax / pq-vllm container running."
    exit 1
fi

docker rm -f pq-vllm-minimax-v21 2>/dev/null || true

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-vllm-minimax-v21 \
  -p 8000:8000 \
  -v "$ARTIFACT":/model:ro \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v "$LOG_DIR":/serve_logs \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/serve_logs/hf_modules \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /work \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c '
    set -euo pipefail
    vllm serve /model \
      --served-model-name minimax-m2.7-prismaquant \
      --quantization compressed-tensors \
      --trust-remote-code \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 32768 \
      --max-num-seqs 4 \
      --gpu-memory-utilization 0.85 \
      --kv-cache-dtype fp8 \
      --enable-prefix-caching \
      --enable-auto-tool-choice \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2 \
      2>&1 | tee /serve_logs/vllm_serve.log
'

echo "[serve] container: pq-vllm-minimax-v21  (port 8000, bound 0.0.0.0)"
echo "[serve] tail:      docker logs -f pq-vllm-minimax-v21"
echo "[serve]   or:      tail -f $LOG_DIR/vllm_serve.log"
echo "[serve] watch:     watch -n 2 'free -h; echo; docker stats --no-stream pq-vllm-minimax-v21'"
echo
echo "[serve] once 'Application startup complete' appears, smoke-test:"
echo "  curl -s http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"minimax-m2.7-prismaquant\", \"messages\": [{\"role\":\"user\",\"content\":\"say hi\"}], \"max_tokens\": 32}'"
