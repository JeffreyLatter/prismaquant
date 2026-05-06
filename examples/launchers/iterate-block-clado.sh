#!/usr/bin/env bash
# Iterated Block-CLADO refinement: measure → sweep → validate → polish,
# then re-center on the polished assignment and repeat until convergence.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-0p6b-block-clado-iter-${TS}}"
CNAME="${CNAME:-pq-bc-iter-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
N_NEIGHBORS_VALIDATE="${N_NEIGHBORS_VALIDATE:-4}"
POLISH_MAX_PASSES="${POLISH_MAX_PASSES:-6}"
POLISH_NOISE_FLOOR="${POLISH_NOISE_FLOOR:-1e-5}"
LOG_NAME="${LOG_NAME:-iterate.log}"

mkdir -p "${RUN_ROOT}"/{logs,home,hf_modules,hf_datasets,tf_cache}

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
  -e FORMATS="${FORMATS}" \
  -e N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
  -e CALIB_SEQLEN="${CALIB_SEQLEN}" \
  -e CALIB_SPLIT="${CALIB_SPLIT}" \
  -e CALIB_SEED="${CALIB_SEED}" \
  -e MAX_ITERATIONS="${MAX_ITERATIONS}" \
  -e N_NEIGHBORS_VALIDATE="${N_NEIGHBORS_VALIDATE}" \
  -e POLISH_MAX_PASSES="${POLISH_MAX_PASSES}" \
  -e POLISH_NOISE_FLOOR="${POLISH_NOISE_FLOOR}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/iter_pip.log"

    python3 -m prismaquant.iterate_block_clado \
      --model "${MODEL_PATH}" \
      --output-root /work \
      --max-iterations "${MAX_ITERATIONS}" \
      --formats "${FORMATS}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      --device cuda \
      --n-neighbors-validate "${N_NEIGHBORS_VALIDATE}" \
      --polish-max-passes "${POLISH_MAX_PASSES}" \
      --polish-noise-floor "${POLISH_NOISE_FLOOR}" \
      2>&1 | tee "/work/logs/${LOG_NAME}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] summary:   ${RUN_ROOT}/summary.json"
