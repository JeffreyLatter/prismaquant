#!/usr/bin/env bash
# Block-CLADO smoke run on Qwen3-0.6B.
#
# Phase 1: measure block-CLADO costs (unary Ω_ii + intra-block Ω_ij).
# Phase 2: λ-sweep over the cost payload to recover a Pareto frontier.
# Phase 3: validate the kneedle assignment with real KL.
#
# Output lands at $RUN_ROOT/{artifacts,sweep,validate}/.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-0p6b-block-clado-${TS}}"
CNAME="${CNAME:-pq-qwen3-0p6b-block-clado-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"

LAMBDA_MIN="${LAMBDA_MIN:-1e-12}"
LAMBDA_MAX="${LAMBDA_MAX:-1e-3}"
N_LAMBDAS="${N_LAMBDAS:-61}"

PAYLOAD_JSON="${PAYLOAD_JSON:-/work/artifacts/block_clado.json}"
SWEEP_JSON="${SWEEP_JSON:-/work/sweep/lambda_sweep.json}"
LOG_NAME="${LOG_NAME:-block_clado.log}"

mkdir -p "${RUN_ROOT}"/{artifacts,sweep,validate,work,logs,home,hf_modules,hf_datasets,tf_cache}

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
      | tee "/work/logs/pip_install.log"

    echo "[block-clado] phase 1: measure costs"
    python3 -m prismaquant.measure_block_clado \
      --model "${MODEL_PATH}" \
      --output "${PAYLOAD_JSON}" \
      --formats "${FORMATS}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      --device cuda \
      --work-dir /work/work \
      2>&1 | tee "/work/logs/${LOG_NAME}"

    echo "[block-clado] phase 2: λ-sweep"
    python3 -m prismaquant.block_clado sweep \
      --payload "${PAYLOAD_JSON}" \
      --lambda-min "${LAMBDA_MIN}" \
      --lambda-max "${LAMBDA_MAX}" \
      --n-lambdas "${N_LAMBDAS}" \
      --output "${SWEEP_JSON}" \
      2>&1 | tee -a "/work/logs/${LOG_NAME}"

    echo "[block-clado] sweep summary:"
    python3 -c "
import json, sys
data = json.load(open(\"${SWEEP_JSON}\"))
rows = data[\"rows\"]
print(f\"frontier points: {len(rows)}\")
for r in rows:
    print(f\"  λ={r[\\\"lambda\\\"]:.3e}  bpp={r[\\\"bpp\\\"]:.4f}  cost={r[\\\"cost_total\\\"]:.6g}\")
" 2>&1 | tee -a "/work/logs/${LOG_NAME}"

    echo "[block-clado] done"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] logs:      docker logs -f ${CNAME}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] payload:   ${RUN_ROOT}/artifacts/block_clado.json"
echo "[launch] sweep:     ${RUN_ROOT}/sweep/lambda_sweep.json"
