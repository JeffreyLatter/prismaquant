#!/usr/bin/env bash
# PrismaSCOUT/kneedle phase for BF16-source Qwen3.6-35B-A3B.
#
# Run this after launch-qwen3p6-35b-a3b-prismascout-probe-cost.sh has produced
# artifacts/probe.pkl and artifacts/cost.pkl. The seed layer_config is created
# with allocator.py so visual Linears are source-discovered and stamped NVFP4,
# matching the 27B PrismaSCOUT artifact's visual policy.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
RUN_ROOT="${RUN_ROOT:-$(ls -td /home/rob/dq-runs/qwen3p6-35b-a3b-prismascout-bf16-* 2>/dev/null | head -1)}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CNAME="${CNAME:-pq-qwen36-35b-a3b-knee-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
ANCHOR_BITS="${ANCHOR_BITS:-5.5}"
VISUAL_FORMAT="${VISUAL_FORMAT:-NVFP4}"
VISUAL_SENSITIVITY="${VISUAL_SENSITIVITY:-uniform}"
PARETO_TARGETS="${PARETO_TARGETS:-4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25}"

KNEE_BPP_MIN="${KNEE_BPP_MIN:-5.05}"
KNEE_BPP_MAX="${KNEE_BPP_MAX:-5.55}"
KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.02}"
KNEE_MAX_EVALUATIONS="${KNEE_MAX_EVALUATIONS:-3}"
KNEE_INITIAL_POINTS="${KNEE_INITIAL_POINTS:-3}"
KNEE_LAMBDA_EVALUATIONS="${KNEE_LAMBDA_EVALUATIONS:-41}"
KNEE_ARCHIVE_VALIDATION_CANDIDATES="${KNEE_ARCHIVE_VALIDATION_CANDIDATES:-31}"
KNEE_ARCHIVE_BEAM_PER_BIN="${KNEE_ARCHIVE_BEAM_PER_BIN:-4}"
KNEE_ARCHIVE_REFINE_CANDIDATES="${KNEE_ARCHIVE_REFINE_CANDIDATES:-8}"
RESUME_L3_COSTS="${RESUME_L3_COSTS:-}"
RESUME_L3_COSTS_DIR="${RESUME_L3_COSTS_DIR:-}"
KNEE_OUTPUT_DIR="${KNEE_OUTPUT_DIR:-/work/out}"
KNEE_WORK_DIR="${KNEE_WORK_DIR:-/work/work/knee}"
KNEE_LOG_NAME="${KNEE_LOG_NAME:-archive_knee_visual_nvfp4.log}"

if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
  echo "[launch] RUN_ROOT not found. Set RUN_ROOT to the probe/cost run dir." >&2
  exit 2
fi
if [[ ! -f "${RUN_ROOT}/artifacts/probe.pkl" ]]; then
  echo "[launch] missing probe: ${RUN_ROOT}/artifacts/probe.pkl" >&2
  exit 2
fi
if [[ ! -f "${RUN_ROOT}/artifacts/cost.pkl" ]]; then
  echo "[launch] missing cost: ${RUN_ROOT}/artifacts/cost.pkl" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"/{logs,out,work,home,hf_modules,hf_datasets,tf_cache}

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
  -e ANCHOR_BITS="${ANCHOR_BITS}" \
  -e VISUAL_FORMAT="${VISUAL_FORMAT}" \
  -e VISUAL_SENSITIVITY="${VISUAL_SENSITIVITY}" \
  -e PARETO_TARGETS="${PARETO_TARGETS}" \
  -e KNEE_BPP_MIN="${KNEE_BPP_MIN}" \
  -e KNEE_BPP_MAX="${KNEE_BPP_MAX}" \
  -e KNEE_TOLERANCE="${KNEE_TOLERANCE}" \
  -e KNEE_MAX_EVALUATIONS="${KNEE_MAX_EVALUATIONS}" \
  -e KNEE_INITIAL_POINTS="${KNEE_INITIAL_POINTS}" \
  -e KNEE_LAMBDA_EVALUATIONS="${KNEE_LAMBDA_EVALUATIONS}" \
  -e KNEE_ARCHIVE_VALIDATION_CANDIDATES="${KNEE_ARCHIVE_VALIDATION_CANDIDATES}" \
  -e KNEE_ARCHIVE_BEAM_PER_BIN="${KNEE_ARCHIVE_BEAM_PER_BIN}" \
  -e KNEE_ARCHIVE_REFINE_CANDIDATES="${KNEE_ARCHIVE_REFINE_CANDIDATES}" \
  -e RESUME_L3_COSTS="${RESUME_L3_COSTS}" \
  -e RESUME_L3_COSTS_DIR="${RESUME_L3_COSTS_DIR}" \
  -e KNEE_OUTPUT_DIR="${KNEE_OUTPUT_DIR}" \
  -e KNEE_WORK_DIR="${KNEE_WORK_DIR}" \
  -e KNEE_LOG_NAME="${KNEE_LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl
    SEED=/work/artifacts/layer_config_l2_5p5_visual_nvfp4.json

    echo "[launch] model=${MODEL_PATH}"
    echo "[launch] work=/work"
    echo "[launch] formats=${FORMATS}"
    echo "[launch] visual=${VISUAL_FORMAT} sensitivity=${VISUAL_SENSITIVITY}"
    echo "[launch] seed=${SEED}"

    python3 -m pip install --user --quiet accelerate datasets \
      2>&1 | tee /work/logs/knee_pip_install.log

    if [[ ! -f "${SEED}" ]]; then
      echo "[0/1] allocator seed at anchor=${ANCHOR_BITS} with visual=${VISUAL_FORMAT} ..."
      python3 -m prismaquant.allocator \
        --probe "${PROBE}" \
        --costs "${COST}" \
        --model-override "${MODEL_PATH}" \
        --target-bits "${ANCHOR_BITS}" \
        --formats "${FORMATS}" \
        --target-profile vllm_packed_moe \
        --pareto-targets "${PARETO_TARGETS}" \
        --visual-format "${VISUAL_FORMAT}" \
        --visual-sensitivity "${VISUAL_SENSITIVITY}" \
        --layer-config "${SEED}" \
        --pareto-csv /work/artifacts/pareto_l2_5p5_visual_nvfp4.csv \
        --bit-precision 0.001 \
        2>&1 | tee /work/logs/allocator_seed_visual_nvfp4.log
    else
      echo "[0/1] seed layer_config exists, skipping allocator seed"
    fi

    echo "[1/1] PrismaSCOUT archive kneedle ..."
    RESUME_L3_ARGS=()
    if [[ -n "${RESUME_L3_COSTS:-}" ]]; then
      RESUME_L3_ARGS+=(--resume-l3-costs "${RESUME_L3_COSTS}" --force-resume-l3-costs)
      echo "[launch] resume_l3_costs=${RESUME_L3_COSTS}"
    fi
    if [[ -n "${RESUME_L3_COSTS_DIR:-}" ]]; then
      RESUME_L3_ARGS+=(--resume-l3-costs-dir "${RESUME_L3_COSTS_DIR}" --force-resume-l3-costs)
      echo "[launch] resume_l3_costs_dir=${RESUME_L3_COSTS_DIR}"
    fi
    python3 -m prismaquant.iterate_perturbed_allocation \
      --model "${MODEL_PATH}" \
      --probe "${PROBE}" \
      --initial-costs "${COST}" \
      --initial-config "${SEED}" \
      --formats "${FORMATS}" \
      --knee-search \
      --knee-archive-search \
      --knee-mode kneedle \
      --knee-bpp-min "${KNEE_BPP_MIN}" \
      --knee-bpp-max "${KNEE_BPP_MAX}" \
      --knee-tolerance "${KNEE_TOLERANCE}" \
      --knee-max-evaluations "${KNEE_MAX_EVALUATIONS}" \
      --knee-initial-points "${KNEE_INITIAL_POINTS}" \
      --knee-lambda-evaluations "${KNEE_LAMBDA_EVALUATIONS}" \
      --knee-archive-validation-candidates "${KNEE_ARCHIVE_VALIDATION_CANDIDATES}" \
      --knee-archive-beam-per-bin "${KNEE_ARCHIVE_BEAM_PER_BIN}" \
      --knee-archive-refine-candidates "${KNEE_ARCHIVE_REFINE_CANDIDATES}" \
      --no-knee-seed-remeasure \
      --target-bits-anchor "${ANCHOR_BITS}" \
      --target-bits-share-tolerance 0.5 \
      --max-iters 0 \
      --work-dir "${KNEE_WORK_DIR}" \
      --output-dir "${KNEE_OUTPUT_DIR}" \
      --device cuda \
      --dtype bf16 \
      --n-calib-samples 2 \
      --calib-seqlen 128 \
      --calib-split train \
      --calib-seed 42 \
      --l3-polish \
      --l3-mode global \
      --l3-measure-all-formats \
      --l3-n-calib-samples 2 \
      --l3-calib-seqlen 128 \
      --l3-calib-split train \
      --l3-calib-seed 42 \
      --l3-max-lanes-per-batch 6 \
      "${RESUME_L3_ARGS[@]}" \
      --bit-precision 0.001 \
      2>&1 | tee "/work/logs/${KNEE_LOG_NAME}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] logs:      docker logs -f ${CNAME}"
echo "[launch] knee:      tail -f ${RUN_ROOT}/logs/${KNEE_LOG_NAME}"
