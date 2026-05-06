#!/usr/bin/env bash
# Smoke test: production-faithful δw cache → re-validate dense cone on
# Qwen3-0.6B.  Validates the plumbing end-to-end.  If 0.6B works,
# scale up to 4B with `dense-knee-search-prodfaithful-4b.sh`.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-0p6b-prodfaithful-${TS}}"
CNAME="${CNAME:-pq-pf-0p6b-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

# 1. Build production cache (NVFP4 only — MXFP8/BF16 are RTN/passthrough)
PROD_FORMATS="${PROD_FORMATS:-NVFP4}"
PROD_LEVERS="${PROD_LEVERS:-gptq,scale_sweep}"
PROD_N_CALIB="${PROD_N_CALIB:-8}"
PROD_CALIB_SEQ="${PROD_CALIB_SEQ:-256}"
PROD_MAX_ACT_ROWS="${PROD_MAX_ACT_ROWS:-512}"

# 2. Build a small dense cone from the four-term payload we already have
#    on 0.6B (fall back to a fresh measurement if absent).
SOURCE_RUN="${SOURCE_RUN:-/home/rob/dq-runs/qwen3-0p6b-block-clado-iter-fullof-20260506T044945Z/iter_0}"
PAYLOAD_PATH="${PAYLOAD_PATH:-${SOURCE_RUN}/block_clado.json}"

# Cone bpps — focus on the surrogate-knee region.
BPPS="${BPPS:-4.50,4.75,5.00,5.50,6.00,7.00,8.00,10.00,12.00}"

mkdir -p "${RUN_ROOT}"/{logs,home,hf_modules,hf_datasets,tf_cache,cone}
if [[ -f "${PAYLOAD_PATH}" ]]; then
  cp "${PAYLOAD_PATH}" "${RUN_ROOT}/block_clado.json"
else
  echo "[error] no four-term payload at ${PAYLOAD_PATH}; skipping cone re-validate"
  exit 1
fi

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
  -e PROD_FORMATS="${PROD_FORMATS}" \
  -e PROD_LEVERS="${PROD_LEVERS}" \
  -e PROD_N_CALIB="${PROD_N_CALIB}" \
  -e PROD_CALIB_SEQ="${PROD_CALIB_SEQ}" \
  -e PROD_MAX_ACT_ROWS="${PROD_MAX_ACT_ROWS}" \
  -e BPPS="${BPPS}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/pip.log"

    echo "[step 1/3] build production cache (NVFP4 + GPTQ + scale_sweep)"
    python3 -m prismaquant.build_production_cache \
      --model "${MODEL_PATH}" \
      --output /work/production_cache.pkl \
      --formats "${PROD_FORMATS}" \
      --enable "${PROD_LEVERS}" \
      --n-calib-samples "${PROD_N_CALIB}" \
      --calib-seqlen "${PROD_CALIB_SEQ}" \
      --max-act-rows "${PROD_MAX_ACT_ROWS}" \
      --dtype bf16 \
      2>&1 | tee "/work/logs/build_cache.log"

    echo "[step 2/3] dense exact-budget cone"
    python3 -m prismaquant.dense_cone \
      --payload /work/block_clado.json \
      --output-dir /work/cone \
      --bpps "${BPPS}" \
      2>&1 | tee "/work/logs/dense_cone.log"

    echo "[step 3a/3] real-KL validate cone WITHOUT prod cache (RTN baseline)"
    python3 -m prismaquant.validate_block_clado \
      --model "${MODEL_PATH}" \
      --kneedle-dir /work/cone \
      --output /work/dense_validation_rtn.json \
      --n-calib-samples 2 \
      --calib-seqlen 128 \
      --calib-split train \
      --calib-seed 42 \
      --dtype bf16 \
      2>&1 | tee "/work/logs/validate_rtn.log"

    echo "[step 3b/3] real-KL validate cone WITH prod cache"
    python3 -m prismaquant.validate_block_clado \
      --model "${MODEL_PATH}" \
      --kneedle-dir /work/cone \
      --output /work/dense_validation_prod.json \
      --n-calib-samples 2 \
      --calib-seqlen 128 \
      --calib-split train \
      --calib-seed 42 \
      --dtype bf16 \
      --production-weight-cache /work/production_cache.pkl \
      2>&1 | tee "/work/logs/validate_prod.log"

    echo
    echo "[summary] RTN vs production KL (sorted by bpp):"
    python3 - <<PYEOF
import json
rtn  = json.load(open("/work/dense_validation_rtn.json"))["results"]
prod = json.load(open("/work/dense_validation_prod.json"))["results"]
by_label = {r["label"]: r for r in rtn}
print(f"  {chr(0x20)*4}{\"bpp\":>8s}  {\"rtn_kl\":>8s}  {\"prod_kl\":>8s}  {\"delta\":>8s}")
for p in sorted(prod, key=lambda r: r["bpp"]):
    r = by_label.get(p["label"])
    if r is None:
        continue
    delta = p["real_kl"] - r["real_kl"]
    print(f"      {p[\"bpp\"]:>8.4f}  {r[\"real_kl\"]:>8.4f}  {p[\"real_kl\"]:>8.4f}  {delta:>+8.4f}")
PYEOF
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] tail:      docker logs -f ${CNAME}"
