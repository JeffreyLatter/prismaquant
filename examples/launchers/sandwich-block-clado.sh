#!/usr/bin/env bash
# Sandwich-recalibrate Block-CLADO: re-measure the cost payload centered
# at a proposed assignment, then re-solve.  This is one step of the
# round-02 deliberation idea (proximal trust-region around x_proposed).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the block-clado run dir}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CNAME="${CNAME:-pq-bc-sandwich-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

CENTER_ASSIGNMENT="${CENTER_ASSIGNMENT:?set CENTER_ASSIGNMENT to a per-Linear assignment JSON}"
ITERATION_LABEL="${ITERATION_LABEL:-iter1}"
PAYLOAD_OUTPUT="${PAYLOAD_OUTPUT:-${RUN_ROOT}/sandwich/${ITERATION_LABEL}/block_clado.json}"
SWEEP_OUTPUT="${SWEEP_OUTPUT:-${RUN_ROOT}/sandwich/${ITERATION_LABEL}/lambda_sweep.json}"
KNEEDLE_DIR="${KNEEDLE_DIR:-${RUN_ROOT}/sandwich/${ITERATION_LABEL}/kneedle}"
LOG_NAME="${LOG_NAME:-sandwich_${ITERATION_LABEL}.log}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
LAMBDA_MIN="${LAMBDA_MIN:-1e-12}"
LAMBDA_MAX="${LAMBDA_MAX:-1e-3}"
N_LAMBDAS="${N_LAMBDAS:-61}"

mkdir -p "${RUN_ROOT}/sandwich/${ITERATION_LABEL}"/{kneedle} "${RUN_ROOT}/logs"

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
  -e CENTER_ASSIGNMENT="${CENTER_ASSIGNMENT/$RUN_ROOT/\/work}" \
  -e PAYLOAD_OUTPUT="${PAYLOAD_OUTPUT/$RUN_ROOT/\/work}" \
  -e SWEEP_OUTPUT="${SWEEP_OUTPUT/$RUN_ROOT/\/work}" \
  -e KNEEDLE_DIR="${KNEEDLE_DIR/$RUN_ROOT/\/work}" \
  -e FORMATS="${FORMATS}" \
  -e N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
  -e CALIB_SEQLEN="${CALIB_SEQLEN}" \
  -e CALIB_SPLIT="${CALIB_SPLIT}" \
  -e CALIB_SEED="${CALIB_SEED}" \
  -e LAMBDA_MIN="${LAMBDA_MIN}" \
  -e LAMBDA_MAX="${LAMBDA_MAX}" \
  -e N_LAMBDAS="${N_LAMBDAS}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/sandwich_pip.log"

    echo "[sandwich] phase 1: re-measure centered at proposed assignment"
    python3 -m prismaquant.measure_block_clado \
      --model "${MODEL_PATH}" \
      --output "${PAYLOAD_OUTPUT}" \
      --formats "${FORMATS}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      --device cuda \
      --work-dir /work/work \
      --center-assignment "${CENTER_ASSIGNMENT}" \
      2>&1 | tee "/work/logs/${LOG_NAME}"

    echo "[sandwich] phase 2: λ-sweep on centered payload"
    python3 -m prismaquant.block_clado sweep \
      --payload "${PAYLOAD_OUTPUT}" \
      --lambda-min "${LAMBDA_MIN}" \
      --lambda-max "${LAMBDA_MAX}" \
      --n-lambdas "${N_LAMBDAS}" \
      --output "${SWEEP_OUTPUT}" \
      2>&1 | tee -a "/work/logs/${LOG_NAME}"

    echo "[sandwich] phase 3: kneedle extraction"
    python3 -m prismaquant.block_clado kneedle \
      --payload "${PAYLOAD_OUTPUT}" \
      --sweep "${SWEEP_OUTPUT}" \
      --output-dir "${KNEEDLE_DIR}" \
      --n-neighbors 2 \
      2>&1 | tee -a "/work/logs/${LOG_NAME}"

    echo "[sandwich] done ${ITERATION_LABEL}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] payload:   ${PAYLOAD_OUTPUT}"
echo "[launch] kneedle:   ${KNEEDLE_DIR}"
