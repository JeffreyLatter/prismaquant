#!/usr/bin/env bash
# Output-Fisher Block-CLADO measurement on Qwen3-0.6B.  Drop-in payload
# for the rest of the pipeline (sweep, kneedle, validate, polish).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-0p6b-output-fisher-${TS}}"
CNAME="${CNAME:-pq-of-qwen3-0p6b-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
DELTA_Z_DTYPE="${DELTA_Z_DTYPE:-fp16}"
LAMBDA_MIN="${LAMBDA_MIN:-1e-12}"
LAMBDA_MAX="${LAMBDA_MAX:-1e-3}"
N_LAMBDAS="${N_LAMBDAS:-61}"

PAYLOAD_JSON="${PAYLOAD_JSON:-/work/artifacts/output_fisher.json}"
SWEEP_JSON="${SWEEP_JSON:-/work/sweep/lambda_sweep.json}"
LOG_NAME="${LOG_NAME:-output_fisher.log}"

mkdir -p "${RUN_ROOT}"/{artifacts,sweep,kneedle,validate,work,logs,home,hf_modules,hf_datasets,tf_cache}

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
  -e DELTA_Z_DTYPE="${DELTA_Z_DTYPE}" \
  -e LAMBDA_MIN="${LAMBDA_MIN}" \
  -e LAMBDA_MAX="${LAMBDA_MAX}" \
  -e N_LAMBDAS="${N_LAMBDAS}" \
  -e PAYLOAD_JSON="${PAYLOAD_JSON}" \
  -e SWEEP_JSON="${SWEEP_JSON}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/of_pip.log"

    echo "[output-fisher] phase 1: collect surrogate"
    python3 -m prismaquant.measure_output_fisher \
      --model "${MODEL_PATH}" \
      --output "${PAYLOAD_JSON}" \
      --formats "${FORMATS}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      --device cuda \
      --delta-z-dtype "${DELTA_Z_DTYPE}" \
      2>&1 | tee "/work/logs/${LOG_NAME}"

    echo "[output-fisher] phase 2: λ-sweep"
    python3 -m prismaquant.block_clado sweep \
      --payload "${PAYLOAD_JSON}" \
      --lambda-min "${LAMBDA_MIN}" \
      --lambda-max "${LAMBDA_MAX}" \
      --n-lambdas "${N_LAMBDAS}" \
      --output "${SWEEP_JSON}" \
      2>&1 | tee -a "/work/logs/${LOG_NAME}"

    echo "[output-fisher] phase 3: kneedle expansion"
    python3 -m prismaquant.block_clado kneedle \
      --payload "${PAYLOAD_JSON}" \
      --sweep "${SWEEP_JSON}" \
      --output-dir /work/kneedle \
      --n-neighbors 4 \
      2>&1 | tee -a "/work/logs/${LOG_NAME}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] payload:   ${RUN_ROOT}/artifacts/output_fisher.json"
echo "[launch] sweep:     ${RUN_ROOT}/sweep/lambda_sweep.json"
echo "[launch] kneedle:   ${RUN_ROOT}/kneedle/"
