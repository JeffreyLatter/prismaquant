#!/usr/bin/env bash
# Qwen3.5-4B smoke test for the v1 quality wins. Runs probe + cost +
# allocator + export for ONE configuration. Driver script
# (run-qwen4b-smoke-all.sh) iterates over the configurations.
#
# Usage: CONFIG=baseline bash launch-qwen4b-smoke.sh
#
# Configurations (CONFIG env var):
#   baseline       — main, no wins
#   +damp          — PRISMAQUANT_GPTQ_DAMP_SWEEP=1
#   +clip          — +PRISMAQUANT_ACT_CLIP_QUANTILE=0.999
#   +cache_fp32    — +PRISMAQUANT_ACT_CACHE_FP32=1 (needs fresh probe)
#   +halo          — + --halo-mode=random
#   +block_match   — +PRISMAQUANT_BLOCK_OUTPUT_MATCH=1
#   full           — all of the above
#   solo_halo      — only --halo-mode=random
#   solo_damp      — only PRISMAQUANT_GPTQ_DAMP_SWEEP=1
#   solo_block     — only PRISMAQUANT_BLOCK_OUTPUT_MATCH=1
#
# WORKTREE env var picks the prismaquant source — defaults to the
# wins worktree so all the new flags are available.
set -euo pipefail

CONFIG="${CONFIG:?must set CONFIG env var}"
WORKTREE="${WORKTREE:-/home/rob/prismaquant-quality-wins}"
HOST_MODEL="${MODEL_PATH:-/models/Qwen3.5-4B-bf16}"
WORK_ROOT="${WORK_ROOT:-/home/rob/dq-runs/qwen4b-smoke}"
TARGET_BITS="${TARGET_BITS:-4.50}"
# Probe footprint: each layer's activation block is
# nsamples * seqlen * hidden_size * 2 bytes (bf16); the post-probe
# host transfer momentarily holds (device_acts + host stack), so peak
# scales as 2 * nsamples * seqlen * num_layers * hidden_size. For
# small-to-mid models (4B–8B), nsamples=32 / seqlen=2048 is safe;
# for 27B+ on 119 GB UMA it OOMs. Override via env for big models.
NSAMPLES="${NSAMPLES:-32}"
SEQLEN="${SEQLEN:-2048}"

[ -d "$HOST_MODEL" ] || { echo "model missing: $HOST_MODEL"; exit 1; }
[ -d "$WORKTREE" ] || { echo "worktree missing: $WORKTREE"; exit 1; }

# Container path translation: docker run mounts /models→/models and
# /home/rob/.cache/huggingface→/hfcache. Translate HOST_MODEL into the
# corresponding container-side path.
case "$HOST_MODEL" in
    /home/rob/.cache/huggingface/*)
        CONTAINER_MODEL="${HOST_MODEL/#\/home\/rob\/.cache\/huggingface/\/hfcache}"
        ;;
    /models/*)
        CONTAINER_MODEL="$HOST_MODEL"
        ;;
    *)
        echo "MODEL_PATH must live under /models or /home/rob/.cache/huggingface"
        exit 1
        ;;
esac
echo "[smoke] host model: $HOST_MODEL"
echo "[smoke] container model: $CONTAINER_MODEL"

CAL=/home/rob/dq-runs/cal-mix-v1/cal_mix_shuf.jsonl
[ -f "$CAL" ] || { echo "cal mix missing: $CAL"; exit 1; }

HOST_WORK=$WORK_ROOT/$CONFIG
mkdir -p "$HOST_WORK"/{artifacts,act,work,logs,exported,export-cache}

# Configuration-specific env flags
BATCH_OFF="-e PRISMAQUANT_BATCHED_NVFP4_EXPORT=0"
declare -A ENV_FLAGS=(
    [baseline]="$BATCH_OFF -e PRISMAQUANT_DO_NO_HARM=0"
    [+damp]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1"
    [+clip]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999"
    [+cache_fp32]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_ACT_CACHE_FP32=1"
    [+halo]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_ACT_CACHE_FP32=1"
    [+block_match]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_ACT_CACHE_FP32=1 -e PRISMAQUANT_BLOCK_OUTPUT_MATCH=1"
    [full]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_ACT_CACHE_FP32=1 -e PRISMAQUANT_BLOCK_OUTPUT_MATCH=1"
    [solo_halo]="$BATCH_OFF -e PRISMAQUANT_DO_NO_HARM=0"
    [solo_damp]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_DO_NO_HARM=0"
    [solo_clip]="$BATCH_OFF -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_DO_NO_HARM=0"
    [solo_block]="$BATCH_OFF -e PRISMAQUANT_BLOCK_OUTPUT_MATCH=1 -e PRISMAQUANT_DO_NO_HARM=0"
    [solo_dnh]="$BATCH_OFF -e PRISMAQUANT_DO_NO_HARM=1"
    [full_v2]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_ACT_CACHE_FP32=1 -e PRISMAQUANT_BLOCK_OUTPUT_MATCH=1 -e PRISMAQUANT_DO_NO_HARM=1"
    # #38: Fisher-weighted GPTQ ablation. Both inherit the validated
    # defaults-on stack (matches `full_v2`); only differ on the Fisher
    # env flag. Apples-to-apples comparison.
    [fisher_off]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_ACT_CACHE_FP32=1 -e PRISMAQUANT_BLOCK_OUTPUT_MATCH=1 -e PRISMAQUANT_DO_NO_HARM=1 -e PRISMAQUANT_GPTQ_FISHER_WEIGHT=0"
    [fisher_on]="$BATCH_OFF -e PRISMAQUANT_GPTQ_DAMP_SWEEP=1 -e PRISMAQUANT_ACT_CLIP_QUANTILE=0.999 -e PRISMAQUANT_ACT_CACHE_FP32=1 -e PRISMAQUANT_BLOCK_OUTPUT_MATCH=1 -e PRISMAQUANT_DO_NO_HARM=1 -e PRISMAQUANT_GPTQ_FISHER_WEIGHT=1"
)

declare -A HALO_MODE=(
    [baseline]="off"
    [+damp]="off"
    [+clip]="off"
    [+cache_fp32]="off"
    [+halo]="off"
    [+block_match]="off"
    [full]="off"
    [solo_halo]="random"
    [solo_damp]="off"
    [solo_clip]="off"
    [solo_block]="off"
    [solo_dnh]="off"
    [full_v2]="off"
    [fisher_off]="off"
    [fisher_on]="off"
)

if [[ -z "${ENV_FLAGS[$CONFIG]+x}" ]]; then
    echo "unknown CONFIG=$CONFIG"
    echo "valid: ${!ENV_FLAGS[*]}"
    exit 1
fi

EFLAGS="${ENV_FLAGS[$CONFIG]}"
HMODE="${HALO_MODE[$CONFIG]}"

echo "[smoke] CONFIG=$CONFIG"
echo "[smoke] env flags: $EFLAGS"
echo "[smoke] halo mode: $HMODE"
echo "[smoke] worktree: $WORKTREE"

if docker ps --format '{{.Names}}' | grep -E '^pq-qwen4b-smoke$' | head -1; then
    echo "ABORT: pq-qwen4b-smoke already running"
    exit 1
fi
docker rm -f pq-qwen4b-smoke 2>/dev/null || true

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-qwen4b-smoke \
  -v "$WORKTREE":/prismaquant \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v /models:/models:ro \
  -v "$HOST_WORK":/work \
  -v /home/rob/dq-runs/cal-mix-v1:/cal:ro \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  $EFLAGS \
  -w /prismaquant \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 2>&1 | tail -1

    MODEL='$CONTAINER_MODEL'
    PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl
    LAYER_CFG=/work/artifacts/layer_config.json
    EXPORT_DIR=/work/exported

    if [[ ! -f \$PROBE ]]; then
      echo '[1/4] probe ...'
      # --h-detail-dir writes per-Linear [out, in] Fisher diagonals
      # which task #38 (Fisher-weighted GPTQ) consumes via
      # PRISMAQUANT_GPTQ_FISHER_WEIGHT=1 at export time. Always
      # written here; cheap on small/dense models, opt-out for huge MoE.
      python3 -m prismaquant.incremental_probe \\
        --model \"\$MODEL\" \\
        --dataset /cal/cal_mix_shuf.jsonl \\
        --nsamples $NSAMPLES --seqlen $SEQLEN \\
        --device cuda --dtype bf16 \\
        --output \"\$PROBE\" \\
        --activation-cache-dir /work/act \\
        --h-detail-dir /work/h_detail \\
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

    echo '[4/4] export ... (halo_mode=$HMODE)'
    python3 -m prismaquant.export_native_compressed \\
      --model \"\$MODEL\" \\
      --layer-config \"\$LAYER_CFG\" \\
      --output \"\$EXPORT_DIR\" \\
      --activation-cache-dir /work/act \\
      --export-cache-dir /work/export-cache \\
      --halo-mode $HMODE \\
      --halo-seed 0 \\
      --device cuda \\
      2>&1 | tee /work/logs/export.log

    echo '[done] $CONFIG artifact at \$EXPORT_DIR'
"

echo "[launch] container: pq-qwen4b-smoke"
echo "[launch] tail: docker logs -f pq-qwen4b-smoke"
echo "[launch]  or: tail -f $HOST_WORK/logs/export.log"
