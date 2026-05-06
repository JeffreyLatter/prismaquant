#!/usr/bin/env bash
# Coordinate-descent polish on a Block-CLADO assignment, gated by real KL.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the block-clado run dir}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CNAME="${CNAME:-pq-bc-polish-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

PAYLOAD="${PAYLOAD:-${RUN_ROOT}/artifacts/block_clado.json}"
STARTING_ASSIGNMENT="${STARTING_ASSIGNMENT:?set STARTING_ASSIGNMENT to a kneedle JSON}"
OUTPUT="${OUTPUT:-${RUN_ROOT}/polish/result.json}"
LOG_NAME="${LOG_NAME:-polish.log}"

N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
MAX_PASSES="${MAX_PASSES:-8}"
NOISE_FLOOR="${NOISE_FLOOR:-1e-5}"
BITS_BUDGET_MODE="${BITS_BUDGET_MODE:-starting}"
BITS_TOLERANCE="${BITS_TOLERANCE:-0}"
USE_FROZEN_WEIGHT_CACHE="${USE_FROZEN_WEIGHT_CACHE:-1}"
STEEPEST_FIRST="${STEEPEST_FIRST:-0}"

mkdir -p "${RUN_ROOT}"/{polish,logs}

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  --name "${CNAME}" \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/home/rob/.cache/huggingface:ro \
  -v "${RUN_ROOT}":/work \
  -e HOME=/work/home \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e HF_DATASETS_CACHE=/work/hf_datasets \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e PAYLOAD="${PAYLOAD/$RUN_ROOT/\/work}" \
  -e STARTING_ASSIGNMENT="${STARTING_ASSIGNMENT/$RUN_ROOT/\/work}" \
  -e OUTPUT="${OUTPUT/$RUN_ROOT/\/work}" \
  -e N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
  -e CALIB_SEQLEN="${CALIB_SEQLEN}" \
  -e CALIB_SPLIT="${CALIB_SPLIT}" \
  -e CALIB_SEED="${CALIB_SEED}" \
  -e MAX_PASSES="${MAX_PASSES}" \
  -e NOISE_FLOOR="${NOISE_FLOOR}" \
  -e BITS_BUDGET_MODE="${BITS_BUDGET_MODE}" \
  -e BITS_TOLERANCE="${BITS_TOLERANCE}" \
  -e USE_FROZEN_WEIGHT_CACHE="${USE_FROZEN_WEIGHT_CACHE}" \
  -e STEEPEST_FIRST="${STEEPEST_FIRST}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/polish_pip.log"
    EXTRA=()
    if [[ "${USE_FROZEN_WEIGHT_CACHE:-0}" == "1" ]]; then
      EXTRA+=(--use-frozen-weight-cache)
    fi
    if [[ "${STEEPEST_FIRST:-0}" == "1" ]]; then
      EXTRA+=(--steepest-first)
    fi
    python3 -m prismaquant.coord_descent_polish \
      --model "${MODEL_PATH}" \
      --payload "${PAYLOAD}" \
      --starting-assignment "${STARTING_ASSIGNMENT}" \
      --output "${OUTPUT}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      --device cuda \
      --max-passes "${MAX_PASSES}" \
      --noise-floor "${NOISE_FLOOR}" \
      --bits-budget-mode "${BITS_BUDGET_MODE}" \
      --bits-tolerance "${BITS_TOLERANCE}" \
      "${EXTRA[@]}" \
      2>&1 | tee "/work/logs/${LOG_NAME}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] result:    ${OUTPUT}"
