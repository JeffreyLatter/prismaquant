#!/usr/bin/env bash
# Solve an adjoint-sketch L3 frontier for BF16-source Qwen3.6-35B-A3B.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
RUN_ROOT="${RUN_ROOT:-$(ls -td /home/rob/dq-runs/qwen3p6-35b-a3b-prismascout-bf16-* 2>/dev/null | head -1)}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CNAME="${CNAME:-pq-qwen36-35b-a3b-adjoint-frontier-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
ADJOINT_COSTS="${ADJOINT_COSTS:-$(ls -t "${RUN_ROOT}"/artifacts/adjoint_l3*.json 2>/dev/null | head -1)}"
PROBE="${PROBE:-${RUN_ROOT}/artifacts/probe.pkl}"
BASE_ASSIGNMENT="${BASE_ASSIGNMENT:-${RUN_ROOT}/artifacts/layer_config_l2_5p5_visual_nvfp4.json}"
TARGET_FULL_BPPS="${TARGET_FULL_BPPS:-4.5,4.6,4.75,4.9,5.05,5.25,5.5,6.0}"
DIAGONAL_FLOOR_FRAC="${DIAGONAL_FLOOR_FRAC:-1.0}"
MSE_DIAGONAL_FLOOR_FRAC="${MSE_DIAGONAL_FLOOR_FRAC:-2.0}"
LAMBDAS="${LAMBDAS:-0,1e-12,3e-12,1e-11,3e-11,1e-10,3e-10,1e-9}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/out/adjoint_l3_frontier}"
LOG_NAME="${LOG_NAME:-adjoint_l3_frontier.log}"

if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
  echo "[launch] RUN_ROOT not found. Set RUN_ROOT to the probe/cost run dir." >&2
  exit 2
fi
for path in "${ADJOINT_COSTS}" "${PROBE}" "${BASE_ASSIGNMENT}"; do
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    echo "[launch] missing required file: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}"/{logs,out,home,hf_modules,hf_datasets,tf_cache}

docker run -d --ipc=host --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  --name "${CNAME}" \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/dq-runs:/home/rob/dq-runs \
  -v /home/rob/.cache/huggingface:/home/rob/.cache/huggingface:ro \
  -v "${RUN_ROOT}":/work \
  -e HOME=/work/home \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e HF_DATASETS_CACHE=/work/hf_datasets \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e FORMATS="${FORMATS}" \
  -e ADJOINT_COSTS="${ADJOINT_COSTS}" \
  -e PROBE="${PROBE}" \
  -e BASE_ASSIGNMENT="${BASE_ASSIGNMENT}" \
  -e TARGET_FULL_BPPS="${TARGET_FULL_BPPS}" \
  -e DIAGONAL_FLOOR_FRAC="${DIAGONAL_FLOOR_FRAC}" \
  -e MSE_DIAGONAL_FLOOR_FRAC="${MSE_DIAGONAL_FLOOR_FRAC}" \
  -e LAMBDAS="${LAMBDAS}" \
  -e OUTPUT_DIR="${OUTPUT_DIR}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m prismaquant.adjoint_l3_frontier \
      --adjoint-costs "${ADJOINT_COSTS}" \
      --probe "${PROBE}" \
      --base-assignment "${BASE_ASSIGNMENT}" \
      --model "${MODEL_PATH}" \
      --formats "${FORMATS}" \
      --fused-groups \
      --target-full-bpps "${TARGET_FULL_BPPS}" \
      --diagonal-floor-frac "${DIAGONAL_FLOOR_FRAC}" \
      --mse-diagonal-floor-frac "${MSE_DIAGONAL_FLOOR_FRAC}" \
      --lambdas "${LAMBDAS}" \
      --output-dir "${OUTPUT_DIR}" \
      2>&1 | tee "/work/logs/${LOG_NAME}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] logs:      docker logs -f ${CNAME}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] output:    ${RUN_ROOT}/out/adjoint_l3_frontier"
