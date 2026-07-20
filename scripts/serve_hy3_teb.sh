#!/usr/bin/env bash
# ============================================================================
# serve_hy3_teb.sh — serve the Hy3 295B NVFP4-CB artifact under the EXACT
# ToolEvalBench protocol the shipped GGUF artifacts used (July-08/13 runs,
# hy3/bench/serving/serve.sh): 12288 ctx, kv fp8, max-num-seqs 2, hy_v3
# tool+reasoning parsers, eager. Only the artifact + plugin differ, so the
# TEB score is protocol-comparable to GGUF IQ 87 / k-quant 86.
#
# VLLM_TORCH_PROFILER_DIR enables POST /start_profile → /stop_profile for the
# decode time-budget trace (no overhead unless started).
# EXTRA_ARGS lets perf experiments (cudagraph configs) reuse this script.
# ============================================================================
set -u
NAME=pq_hy3_cb
MODEL=/dqruns/prod-hy3-nvfp4cb-2p9/exported_nvfp4_cb
MAXLEN="${MAXLEN:-12288}"
UTIL="${UTIL:-0.95}"
EXTRA_ARGS="${EXTRA_ARGS:---enforce-eager}"
LOG=/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9/logs/serve_teb.log
PROF=/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9/profiles
mkdir -p "$PROF"

docker rm -f "$NAME" >/dev/null 2>&1
echo "[serve] launching $NAME (TEB protocol: len $MAXLEN, util $UTIL, extra: $EXTRA_ARGS) $(date '+%H:%M:%S')"
# EXTRA_ARGS reaches the container via env + a SINGLE-quoted -c script: the
# container shell expands it with word-splitting but WITHOUT quote removal, so
# embedded JSON (--compilation-config '{"…"}') survives intact. Host-side
# interpolation into a double-quoted -c string strips the JSON's quotes.
docker run -d --rm --gpus all --ipc=host -p 8000:8000 --name "$NAME" \
  -v /home/rob/prismaquant:/repo \
  -v /home/rob/dq-runs:/dqruns \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_TORCH_PROFILER_DIR=/dqruns/prod-hy3-nvfp4cb-2p9/profiles \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PQ_MODEL="$MODEL" -e PQ_MAXLEN="$MAXLEN" -e PQ_UTIL="$UTIL" \
  -e PQ_EXTRA="$EXTRA_ARGS" \
  --entrypoint bash vllm-node:latest -c '
    pip install -e /repo/plugins/vllm_prismaquant --no-deps -q 2>/dev/null
    exec vllm serve "$PQ_MODEL" --host 0.0.0.0 --port 8000 \
      --served-model-name hy3 \
      --max-model-len "$PQ_MAXLEN" --max-num-seqs 2 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization "$PQ_UTIL" \
      --enable-auto-tool-choice --tool-call-parser hy_v3 \
      --reasoning-parser hy_v3 \
      $PQ_EXTRA' > "$LOG" 2>&1

for i in $(seq 1 240); do   # up to 20 min (110 GB load + plugin JIT)
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "[serve] READY after $((i*5))s $(date '+%H:%M:%S')"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[serve] FAILED (container exited)"; docker logs "$NAME" 2>&1 | tail -40; exit 1; fi
  sleep 5
done
echo "[serve] TIMEOUT after 20min"; docker logs "$NAME" 2>&1 | tail -40; exit 1
