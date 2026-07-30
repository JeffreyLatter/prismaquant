#!/usr/bin/env bash
# PrismaQuant Gemma 4 31B IT (heaviest dense Gemma 4 variant).
#
# Source: google/gemma-4-31b-it (multimodal Gemma4Config; text branch
# is hidden=5376, 60 layers, tied embeddings). Vision/audio towers
# stay BF16 passthrough per the Gemma4Profile.
#
# Default flags exercise the validated wins stack:
#   norm FP32 (always-on), act_clip 0.999, GPTQ damp sweep, block-output
#   match, do-no-harm, FP32 activation cache. Override any with env vars.
#
# Tied embeddings → cost step runs --no-include-lm-head to avoid the
# meta-tensor crash we hit on Qwen 3.5 4B.
set -euo pipefail

WORKTREE="${WORKTREE:-/home/rob/prismaquant-quality-wins}"
HOST_MODEL="${MODEL_PATH:-/home/rob/dq-runs/gemma4-31b/source}"
HOST_WORK="${WORK_ROOT:-/home/rob/dq-runs/gemma4-31b/work}"
TARGET_BITS="${TARGET_BITS:-4.75}"
# Spark UMA = 121 GB shared between host + GPU. 31B BF16 weights are
# ~62 GB resident; phase-1's batched device→host transfer at the end
# briefly holds device_acts AND a host stack copy (2× the activation
# block), so peak is ~weights + 2× nsamples * seqlen * num_layers *
# hidden * 2 bytes. With 60 layers × hidden 5376, nsamples=32 /
# seqlen=2048 hit ~160 GB and OOM-killed at exit-137 right after L59.
# nsamples=16 halves it (~80 GB peak + 62 GB weights ≈ 142 GB) — still
# tight; combined with seqlen=1536 we land near 100 GB which clears
# the budget on a 119 GB-available host.
NSAMPLES="${NSAMPLES:-16}"
SEQLEN="${SEQLEN:-1536}"

[ -d "$HOST_MODEL" ] || { echo "model missing: $HOST_MODEL"; exit 1; }
[ -d "$WORKTREE" ]   || { echo "worktree missing: $WORKTREE"; exit 1; }

CAL=/home/rob/dq-runs/cal-mix-v1/cal_mix_shuf.jsonl
[ -f "$CAL" ] || { echo "cal mix missing: $CAL"; exit 1; }

mkdir -p "$HOST_WORK"/{artifacts,act,work,logs,exported,export-cache}

if docker ps --format '{{.Names}}' | grep -qE '^pq-gemma4-31b$'; then
    echo "ABORT: pq-gemma4-31b already running"; exit 1
fi
docker rm -f pq-gemma4-31b 2>/dev/null || true

AVAIL_GB=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
echo "[launch] host MemAvailable = ${AVAIL_GB} GB"
echo "[launch] target_bits = $TARGET_BITS"

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-gemma4-31b \
  -v "$WORKTREE":/prismaquant \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v "$HOST_MODEL":/source:ro \
  -v "$HOST_WORK":/work \
  -v /home/rob/dq-runs/cal-mix-v1:/cal:ro \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  -w /prismaquant \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 2>&1 | tail -1

    MODEL=/source
    PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl
    LAYER_CFG=/work/artifacts/layer_config.json
    EXPORT_DIR=/work/exported

    if [[ ! -f \$PROBE ]]; then
      echo '[1/4] probe ...'
      python3 -m prismaquant.incremental_probe \\
        --model \"\$MODEL\" \\
        --dataset /cal/cal_mix_shuf.jsonl \\
        --nsamples $NSAMPLES --seqlen $SEQLEN \\
        --device cuda --dtype bf16 \\
        --output \"\$PROBE\" \\
        --activation-cache-dir /work/act \\
        --work-dir /work/work \\
        --layers-per-shard auto \\
        --calibration-modality text-only \\
        2>&1 | tee /work/logs/probe.log
    else
      echo '[1/4] probe.pkl exists, reusing'
    fi

    if [[ ! -f \$COST ]]; then
      echo '[2/4] cost ...'
      python3 -m prismaquant.incremental_measure_quant_cost \\
        --model \"\$MODEL\" \\
        --probe \"\$PROBE\" \\
        --activation-cache-dir /work/act \\
        --formats NVFP4,MXFP8_E4M3,BF16 \\
        --output \"\$COST\" \\
        --work-dir /work/work \\
        --device cuda --dtype bf16 \\
        --mode batched --chunk-size 256 \\
        --skip-missing-activations \\
        --no-include-lm-head \\
        2>&1 | tee /work/logs/cost.log
    else
      echo '[2/4] cost.pkl exists, reusing'
    fi

    if [[ ! -f \$LAYER_CFG ]]; then
      echo '[3/4] allocator ... target=$TARGET_BITS'
      python3 -m prismaquant.allocator \\
        --model \"\$MODEL\" \\
        --probe \"\$PROBE\" \\
        --costs \"\$COST\" \\
        --formats NVFP4,MXFP8_E4M3,BF16 \\
        --target-bits $TARGET_BITS \\
        --layer-config \"\$LAYER_CFG\" \\
        --pareto-csv /work/artifacts/pareto.csv \\
        2>&1 | tee /work/logs/allocator.log
    else
      echo '[3/4] layer_config.json exists, reusing'
    fi

    echo '[4/4] export ...'
    python3 -m prismaquant.export_native_compressed \\
      --model \"\$MODEL\" \\
      --layer-config \"\$LAYER_CFG\" \\
      --output \"\$EXPORT_DIR\" \\
      --activation-cache-dir /work/act \\
      --export-cache-dir /work/export-cache \\
      --device cuda \\
      2>&1 | tee /work/logs/export.log

    echo '[done] artifact at' \"\$EXPORT_DIR\"
"

echo "[launch] container: pq-gemma4-31b"
echo "[launch] tail:      docker logs -f pq-gemma4-31b"
echo "[launch]   or:      tail -f $HOST_WORK/logs/probe.log"
