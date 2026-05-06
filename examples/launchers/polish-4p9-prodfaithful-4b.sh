#!/usr/bin/env bash
# Polish the 4.90 bpp cone candidate (225 NVFP4 + 28 MXFP8) from the
# prod-faithful run with production cache active and 5% budget creep.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

SOURCE_RUN="${SOURCE_RUN:-/home/rob/dq-runs/qwen3-4b-prodfaithful-20260506T114753Z}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-4b-polish-4p9-${TS}}"
CNAME="${CNAME:-pq-pol49-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP:-0.05}"
POLISH_MAX_PASSES="${POLISH_MAX_PASSES:-12}"

mkdir -p "${RUN_ROOT}"/{logs,home,hf_modules,hf_datasets,tf_cache}

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
  -e MODEL_PATH="${MODEL_PATH}" \
  -e POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP}" \
  -e POLISH_MAX_PASSES="${POLISH_MAX_PASSES}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/pip.log"

    echo "[polish-4.9] running coord_descent_polish from 4.9045 starting state"
    python3 -m prismaquant.polish_from_assignment \
      --model "${MODEL_PATH}" \
      --payload /source/iter_0/block_clado.json \
      --assignment /source/cone/budget_bpp_4p9045.json \
      --output /work/polish_4p9.json \
      --production-weight-cache /source/production_cache.pkl \
      --n-calib-samples 2 \
      --calib-seqlen 128 \
      --polish-budget-creep "${POLISH_BUDGET_CREEP}" \
      --polish-max-passes "${POLISH_MAX_PASSES}" \
      --polish-steepest-first \
      2>&1 | tee "/work/logs/polish.log"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] tail:      docker logs -f ${CNAME}"
echo "[launch] result:    ${RUN_ROOT}/polish_4p9.json"
