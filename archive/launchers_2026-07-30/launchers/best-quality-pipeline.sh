#!/usr/bin/env bash
# Headline best-quality pipeline: single-iteration OF + polish.
# Reaches the lowest measured KL we've achieved on Qwen3-0.6B
# (~11.6 bpp / KL 0.0226) in ~2-3 minutes.  Use this as the
# default for "best raw KL at any bpp"-class targets.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-0p6b-best-quality-${TS}}"
CNAME="${CNAME:-pq-bc-best-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP:-0.10}"
POLISH_MAX_PASSES="${POLISH_MAX_PASSES:-12}"
LOG_NAME="${LOG_NAME:-best_quality.log}"

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
  -e POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP}" \
  -e POLISH_MAX_PASSES="${POLISH_MAX_PASSES}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/best_pip.log"

    # Single iteration only — sandwich iters dont improve at this scale.
    python3 -m prismaquant.iterate_block_clado \
      --model "${MODEL_PATH}" \
      --output-root /work \
      --max-iterations 1 \
      --formats "${FORMATS}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      --device cuda \
      --polish-max-passes "${POLISH_MAX_PASSES}" \
      --polish-noise-floor 1e-5 \
      --polish-budget-creep "${POLISH_BUDGET_CREEP}" \
      --polish-steepest-first \
      --use-frozen-weight-cache \
      --measure-method output_fisher \
      --n-neighbors-validate 4 \
      2>&1 | tee "/work/logs/${LOG_NAME}"

    echo "[best-quality] result:"
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
echo "[launch] best:      ${RUN_ROOT}/best_assignment.json"
