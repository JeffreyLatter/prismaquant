#!/usr/bin/env bash
# Densify Pareto in 5.625-5.95 with lm_head pinned, then polish from
# the 5.70 candidate to break into the low-bpp band.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

SOURCE_RUN="${SOURCE_RUN:-/home/rob/dq-runs/qwen3-4b-prodfaithful-20260506T114753Z}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-4b-pareto-densify-${TS}}"
CNAME="${CNAME:-pq-densify-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

# Dense in the feasible-floor band (5.625 → 5.95 at 0.025 bpp resolution).
BPPS="${BPPS:-5.625,5.650,5.675,5.700,5.725,5.750,5.775,5.800,5.825,5.850,5.875,5.900,5.925,5.950}"

POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP:-0.05}"
POLISH_MAX_PASSES="${POLISH_MAX_PASSES:-12}"

mkdir -p "${RUN_ROOT}"/{logs,home,hf_modules,hf_datasets,tf_cache,cone}

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  --name "${CNAME}" \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/home/rob/.cache/huggingface:ro \
  -v "${SOURCE_RUN}":/source:ro \
  -v "${RUN_ROOT}":/work \
  -e HOME=/work/home \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e HF_DATASETS_CACHE=/work/hf_datasets \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e BPPS="${BPPS}" \
  -e POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP}" \
  -e POLISH_MAX_PASSES="${POLISH_MAX_PASSES}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 | tee /work/logs/pip.log

    echo "[1/3] dense cone with lm_head pinned"
    python3 -m prismaquant.dense_cone \
      --payload /source/iter_0/block_clado.json \
      --output-dir /work/cone \
      --bpps "${BPPS}" \
      --pin lm_head \
      2>&1 | tee /work/logs/dense_cone.log

    echo "[2/3] real-KL validate dense band"
    python3 -m prismaquant.validate_block_clado \
      --model "'"${MODEL_PATH}"'" \
      --kneedle-dir /work/cone \
      --output /work/dense_validation.json \
      --n-calib-samples 2 --calib-seqlen 128 \
      --calib-split train --calib-seed 42 \
      --dtype bf16 \
      --production-weight-cache /source/production_cache.pkl \
      2>&1 | tee /work/logs/validate.log

    echo "[3/3] polish from 5.70 candidate (PARETO point with KL=0.165)"
    python3 -m prismaquant.polish_from_assignment \
      --model "'"${MODEL_PATH}"'" \
      --payload /source/iter_0/block_clado.json \
      --assignment /work/cone/budget_bpp_5p7025.json \
      --output /work/polish_from_5p70.json \
      --production-weight-cache /source/production_cache.pkl \
      --polish-budget-creep "${POLISH_BUDGET_CREEP}" \
      --polish-max-passes "${POLISH_MAX_PASSES}" \
      --polish-steepest-first \
      --pin lm_head \
      2>&1 | tee /work/logs/polish.log
  '

echo "container=${CNAME}"
echo "workdir=${RUN_ROOT}"
