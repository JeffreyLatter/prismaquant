#!/usr/bin/env bash
# Measure adjoint-sketch L3 costs for BF16-source Qwen3.6-35B-A3B.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
RUN_ROOT="${RUN_ROOT:-$(ls -td /home/rob/dq-runs/qwen3p6-35b-a3b-prismascout-bf16-* 2>/dev/null | head -1)}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CNAME="${CNAME:-pq-qwen36-35b-a3b-adjoint-l3-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
DIRECTION_MODE="${DIRECTION_MODE:-kl-fisher}"
FISHER_PROBES_PER_SAMPLE="${FISHER_PROBES_PER_SAMPLE:-4}"
FISHER_SEED="${FISHER_SEED:-17}"
FISHER_TEMPERATURE="${FISHER_TEMPERATURE:-1.0}"
FISHER_TOKEN_SCOPE="${FISHER_TOKEN_SCOPE:-last}"
FISHER_PROBE_DISTRIBUTION="${FISHER_PROBE_DISTRIBUTION:-gaussian}"
DIAGONAL_FLOOR_FRAC="${DIAGONAL_FLOOR_FRAC:-1.0}"
MSE_DIAGONAL_FLOOR_FRAC="${MSE_DIAGONAL_FLOOR_FRAC:-2.0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_TARGETS="${MAX_TARGETS:-}"
TARGETS_JSON="${TARGETS_JSON:-${RUN_ROOT}/artifacts/adjoint_l3_targets_from_l3.json}"
OUTPUT_JSON="${OUTPUT_JSON:-${RUN_ROOT}/artifacts/adjoint_l3_samples${N_CALIB_SAMPLES}_fisher${FISHER_PROBES_PER_SAMPLE}_${DIRECTION_MODE}_mse${MSE_DIAGONAL_FLOOR_FRAC}.json}"
LOG_NAME="${LOG_NAME:-adjoint_l3_samples${N_CALIB_SAMPLES}_fisher${FISHER_PROBES_PER_SAMPLE}_${DIRECTION_MODE}_mse${MSE_DIAGONAL_FLOOR_FRAC}.log}"

if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
  echo "[launch] RUN_ROOT not found. Set RUN_ROOT to the probe/cost run dir." >&2
  exit 2
fi
if [[ ! -f "${TARGETS_JSON}" ]]; then
  echo "[launch] missing target list: ${TARGETS_JSON}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"/{logs,artifacts,home,hf_modules,hf_datasets,tf_cache}

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
  -e DIRECTION_MODE="${DIRECTION_MODE}" \
  -e FISHER_PROBES_PER_SAMPLE="${FISHER_PROBES_PER_SAMPLE}" \
  -e FISHER_SEED="${FISHER_SEED}" \
  -e FISHER_TEMPERATURE="${FISHER_TEMPERATURE}" \
  -e FISHER_TOKEN_SCOPE="${FISHER_TOKEN_SCOPE}" \
  -e FISHER_PROBE_DISTRIBUTION="${FISHER_PROBE_DISTRIBUTION}" \
  -e DIAGONAL_FLOOR_FRAC="${DIAGONAL_FLOOR_FRAC}" \
  -e MSE_DIAGONAL_FLOOR_FRAC="${MSE_DIAGONAL_FLOOR_FRAC}" \
  -e DEVICE_MAP="${DEVICE_MAP}" \
  -e MAX_TARGETS="${MAX_TARGETS}" \
  -e TARGETS_JSON=/work/artifacts/$(basename "${TARGETS_JSON}") \
  -e OUTPUT_JSON=/work/artifacts/$(basename "${OUTPUT_JSON}") \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/adjoint_l3_pip_install.log"
    EXTRA=()
    if [[ -n "${MAX_TARGETS:-}" ]]; then
      EXTRA+=(--max-targets "${MAX_TARGETS}")
    fi
    python3 -m prismaquant.measure_adjoint_l3 \
      --model "${MODEL_PATH}" \
      --output "${OUTPUT_JSON}" \
      --formats "${FORMATS}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      --device cuda \
      --device-map "${DEVICE_MAP}" \
      --target-names-json "${TARGETS_JSON}" \
      --direction-mode "${DIRECTION_MODE}" \
      --fisher-probes-per-sample "${FISHER_PROBES_PER_SAMPLE}" \
      --fisher-seed "${FISHER_SEED}" \
      --fisher-temperature "${FISHER_TEMPERATURE}" \
      --fisher-token-scope "${FISHER_TOKEN_SCOPE}" \
      --fisher-probe-distribution "${FISHER_PROBE_DISTRIBUTION}" \
      --diagonal-floor-frac "${DIAGONAL_FLOOR_FRAC}" \
      --mse-diagonal-floor-frac "${MSE_DIAGONAL_FLOOR_FRAC}" \
      --error-device cpu \
      "${EXTRA[@]}" \
      2>&1 | tee "/work/logs/${LOG_NAME}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] logs:      docker logs -f ${CNAME}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] output:    ${OUTPUT_JSON}"
