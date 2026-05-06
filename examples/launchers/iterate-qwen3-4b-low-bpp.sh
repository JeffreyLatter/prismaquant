#!/usr/bin/env bash
# Low-bpp focused Block-CLADO + polish pipeline on Qwen3-4B.
#
# Uses the four-term identity surrogate (better than OF at low bpp where
# higher-order pair effects matter), tight polish budget creep so polish
# doesn't drift to high bpp, and a single iteration (sandwich iters
# didn't help on 0.6B; revisit if first iter is unsatisfactory).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-4b-block-clado-low-bpp-${TS}}"
CNAME="${CNAME:-pq-bc-4b-low-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
N_NEIGHBORS_VALIDATE="${N_NEIGHBORS_VALIDATE:-4}"
POLISH_MAX_PASSES="${POLISH_MAX_PASSES:-12}"
POLISH_NOISE_FLOOR="${POLISH_NOISE_FLOOR:-1e-5}"
# Tight creep so polish stays in the deployment-relevant low-bpp range
POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP:-0.02}"
MEASURE_METHOD="${MEASURE_METHOD:-four_term}"
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
  -e POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP}" \
  -e MEASURE_METHOD="${MEASURE_METHOD}" \
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
      --polish-budget-creep "${POLISH_BUDGET_CREEP}" \
      --polish-steepest-first \
      --use-frozen-weight-cache \
      --measure-method "${MEASURE_METHOD}" \
      2>&1 | tee "/work/logs/${LOG_NAME}"

    echo "[low-bpp] result:"
    python3 -c "
import json
s = json.load(open(\"/work/summary.json\"))
b = s[\"best_overall\"]
print(f\"  bpp={b[\\\"best_validated_bpp\\\"]:.4f}  KL={b[\\\"polished_kl\\\"]:.6f}\")
"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
echo "[launch] summary:   ${RUN_ROOT}/summary.json"
