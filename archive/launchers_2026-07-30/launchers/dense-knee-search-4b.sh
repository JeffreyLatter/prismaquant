#!/usr/bin/env bash
# Dense knee search on Qwen3-4B: reuse existing four-term payload, run
# exact-budget knapsack at 22 target bpps in 4.50-6.00, real-KL validate
# all in one container.  No re-measurement needed (~15 min total).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Reuse the four-term payload from the prior iterate run.
SOURCE_RUN="${SOURCE_RUN:-/home/rob/dq-runs/qwen3-4b-block-clado-low-bpp-20260506T082141Z}"
PAYLOAD_PATH="${PAYLOAD_PATH:-${SOURCE_RUN}/iter_0/block_clado.json}"

RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-4b-dense-knee-${TS}}"
CNAME="${CNAME:-pq-bc-knee-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

# Dense in 4.50-5.50 (the suspected knee region per user), sparser at edges.
BPPS="${BPPS:-4.50,4.55,4.60,4.65,4.70,4.75,4.80,4.85,4.90,4.95,5.00,5.05,5.10,5.15,5.20,5.25,5.30,5.40,5.50,5.75,6.00,6.50,7.00,8.00}"

N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"

mkdir -p "${RUN_ROOT}"/{logs,home,hf_modules,hf_datasets,tf_cache,cone}
# Stage payload into container-visible workdir
cp "${PAYLOAD_PATH}" "${RUN_ROOT}/block_clado.json"

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
  -e BPPS="${BPPS}" \
  -e N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
  -e CALIB_SEQLEN="${CALIB_SEQLEN}" \
  -e CALIB_SPLIT="${CALIB_SPLIT}" \
  -e CALIB_SEED="${CALIB_SEED}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/pip.log"

    echo "[step 1/2] dense exact-budget cone"
    python3 -m prismaquant.dense_cone \
      --payload /work/block_clado.json \
      --output-dir /work/cone \
      --bpps "${BPPS}" \
      2>&1 | tee "/work/logs/dense_cone.log"

    echo "[step 2/2] real-KL validate cone"
    python3 -m prismaquant.validate_block_clado \
      --model "${MODEL_PATH}" \
      --kneedle-dir /work/cone \
      --output /work/dense_validation.json \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split "${CALIB_SPLIT}" \
      --calib-seed "${CALIB_SEED}" \
      --dtype bf16 \
      2>&1 | tee "/work/logs/validate.log"

    echo
    echo "[summary] bpp vs real KL (sorted by bpp):"
    python3 -c "
import json
v = json.load(open(\"/work/dense_validation.json\"))
rows = sorted(v[\"results\"], key=lambda r: r[\"bpp\"])
print(f\"  {\"bpp\":>8s}  {\"surrogate\":>10s}  {\"real_kl\":>8s}  counts\")
for r in rows:
    counts = r[\"format_counts\"]
    counts_str = \" \".join(f\"{k}:{v}\" for k, v in sorted(counts.items()))
    print(f\"  {r[\\\"bpp\\\"]:>8.4f}  {r[\\\"surrogate_cost\\\"]:>+10.4f}  {r[\\\"real_kl\\\"]:>8.4f}  {counts_str}\")
"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] tail:      docker logs -f ${CNAME}"
echo "[launch] result:    ${RUN_ROOT}/dense_validation.json"
