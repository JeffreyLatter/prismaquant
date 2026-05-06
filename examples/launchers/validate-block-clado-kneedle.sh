#!/usr/bin/env bash
# Validate a Block-CLADO kneedle (and optional neighbour) assignments
# with measured teacher-student KL on the same calibration mix used for
# measurement.  Runs in the same vLLM-fresh Docker image.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the block-clado run dir}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CNAME="${CNAME:-pq-bc-validate-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

KNEEDLE_DIR="${KNEEDLE_DIR:-${RUN_ROOT}/kneedle}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
LOG_NAME="${LOG_NAME:-validate.log}"

if [[ ! -d "${KNEEDLE_DIR}" ]]; then
  echo "[validate] missing kneedle dir: ${KNEEDLE_DIR}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"/{validate,logs}

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
  -e KNEEDLE_DIR="/work/$(basename "${KNEEDLE_DIR}")" \
  -e N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
  -e CALIB_SEQLEN="${CALIB_SEQLEN}" \
  -e CALIB_SPLIT="${CALIB_SPLIT}" \
  -e CALIB_SEED="${CALIB_SEED}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/validate_pip.log"
    python3 -m prismaquant.validate_block_clado \
      --model "${MODEL_PATH}" \
      --kneedle-dir "${KNEEDLE_DIR}" \
      --output /work/validate/results.json \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      2>&1 | tee "/work/logs/${LOG_NAME}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] result:    ${RUN_ROOT}/validate/results.json"
