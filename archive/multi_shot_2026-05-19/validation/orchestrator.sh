#!/bin/bash
# Multi-shot recalibration validation harness on Qwen3-4B.
#
# Layout:
#   step 1: baseline pipeline (1-shot, validated-surrogate, PARETO_TARGETS=3.5,4.0,4.5).
#           Produces probe.pkl, cost.pkl, format-menu production cache, and KL-measured
#           per-budget pareto assignments.
#   step 2: for each B in 3.5,4.0,4.5, run a 2-shot pipeline at TARGET_BITS=B,
#           reusing baseline's probe.pkl + cost.pkl (skips probe + cost stages).
#   step 3: validate_assignments_kl on the 3 multi-shot layer_configs against the
#           same calibration the baseline used (and the same format-menu production cache).
#   step 4: emit a comparison JSON + a printed summary table.
#
# Designed to run autonomously inside the vllm-fresh-b12x:latest container.
# Set TS, then call this script (already inside the container). The script
# does NOT manage docker — wrap externally.
set -euo pipefail

TS="${TS:?TS env var must be set, e.g. 20260518T180000}"
BASE_DIR="/work/multi-shot-validate-${TS}"
BASELINE_DIR="${BASE_DIR}/baseline"
COMPARISON_DIR="${BASE_DIR}/comparison"

MODEL_PATH="${MODEL_PATH:-/hfcache/Qwen3-4B}"
DATASET="${DATASET:-/work/calibration/diverse-v1.jsonl}"
NSAMPLES="${NSAMPLES:-8}"
SEQLEN="${SEQLEN:-512}"
FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
# Budgets above the NVFP4 floor of 4.5 bpp, spanning enough range that the
# allocator has to make real per-Linear NVFP4-vs-MXFP8 decisions (which is
# where multi-shot recalibration has any chance to matter).
PARETO_TARGETS="${PARETO_TARGETS:-5.0,5.5,6.0}"
# BUDGETS_CSV defaults to the same set as PARETO_TARGETS but can be overridden
# (e.g. focus a 4-shot run on a single best-signal budget).
BUDGETS_CSV="${BUDGETS_CSV:-${PARETO_TARGETS}}"
IFS=',' read -r -a BUDGETS <<< "$BUDGETS_CSV"
ACTIVATION_ROWS_LIMIT="${ACTIVATION_ROWS_LIMIT:-256}"
LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-auto}"
# 2 = cheap variant per the validated design. Allow overriding for the
# 4-shot diminishing-returns experiment.
MULTI_SHOT_PASSES_ARM="${MULTI_SHOT_PASSES_ARM:-2}"

mkdir -p "${BASE_DIR}" "${COMPARISON_DIR}"

echo "[validate] TS=${TS}"
echo "[validate] BASE_DIR=${BASE_DIR}"
echo "[validate] MODEL_PATH=${MODEL_PATH}"
echo "[validate] FORMATS=${FORMATS}  PARETO=${PARETO_TARGETS}"
echo "[validate] NSAMPLES=${NSAMPLES} SEQLEN=${SEQLEN}"
echo

# ---------------------------------------------------------------------------
# Step 1: baseline (1-shot, validated-surrogate)
# ---------------------------------------------------------------------------
echo "[validate] === step 1: baseline 1-shot pipeline ==="
MODEL_PATH="${MODEL_PATH}" \
WORK_DIR="${BASELINE_DIR}" \
DATASET="${DATASET}" \
FORMATS="${FORMATS}" \
TARGET_BITS="${BASELINE_TARGET_BITS:-5.5}" \
PARETO_TARGETS="${PARETO_TARGETS}" \
NSAMPLES="${NSAMPLES}" SEQLEN="${SEQLEN}" \
ACTIVATION_ROWS_LIMIT="${ACTIVATION_ROWS_LIMIT}" \
LAYERS_PER_SHARD="${LAYERS_PER_SHARD}" \
MULTI_SHOT_PASSES=1 \
SELECTION_MODE=validated-surrogate \
PRODUCTION_CACHE=1 PRODUCTION_RECACHE=0 \
PRODUCTION_CACHE_LRU_GB=32.0 \
TARGET_PROFILE=vllm_packed_moe \
CALIBRATION_MODALITY=text-only \
bash /prismaquant/prismaquant/run-pipeline.sh
echo
echo "[validate] baseline finished. Artifacts: ${BASELINE_DIR}"
ls "${BASELINE_DIR}/artifacts/pareto_assignments/" 2>/dev/null | head -20

# ---------------------------------------------------------------------------
# Step 2: multi-shot at each budget
# ---------------------------------------------------------------------------
for B in "${BUDGETS[@]}"; do
  MS_DIR="${BASE_DIR}/multishot-${B}"
  echo
  echo "[validate] === step 2.${B}: 2-shot pipeline at TARGET_BITS=${B} ==="
  mkdir -p "${MS_DIR}/artifacts"
  # Reuse baseline's probe.pkl + cost.pkl so we skip stages 1+2.
  cp "${BASELINE_DIR}/artifacts/probe.pkl" "${MS_DIR}/artifacts/probe.pkl"
  cp "${BASELINE_DIR}/artifacts/cost.pkl"  "${MS_DIR}/artifacts/cost.pkl"

  MODEL_PATH="${MODEL_PATH}" \
  WORK_DIR="${MS_DIR}" \
  DATASET="${DATASET}" \
  FORMATS="${FORMATS}" \
  TARGET_BITS="${B}" \
  PARETO_TARGETS="${PARETO_TARGETS}" \
  NSAMPLES="${NSAMPLES}" SEQLEN="${SEQLEN}" \
  ACTIVATION_ROWS_LIMIT="${ACTIVATION_ROWS_LIMIT}" \
  LAYERS_PER_SHARD="${LAYERS_PER_SHARD}" \
  MULTI_SHOT_PASSES="${MULTI_SHOT_PASSES_ARM}" \
  SELECTION_MODE=surrogate \
  PRODUCTION_CACHE=1 PRODUCTION_RECACHE=0 \
  PRODUCTION_CACHE_LRU_GB=32.0 \
  TARGET_PROFILE=vllm_packed_moe \
  CALIBRATION_MODALITY=text-only \
  bash /prismaquant/prismaquant/run-pipeline.sh
  echo "[validate] multi-shot ${B} done; final layer_config = ${MS_DIR}/artifacts/layer_config.json"
done

# ---------------------------------------------------------------------------
# Step 3: validate_assignments_kl on all 6 layer_configs (3 baseline + 3 multi-shot)
# Reuses baseline's format-menu production weight cache so the apples-to-apples
# rendered weights are the same across both arms.
# ---------------------------------------------------------------------------
echo
echo "[validate] === step 3: measured KL across all 6 assignments ==="
BASELINE_CACHE_DIR="${BASELINE_DIR}/artifacts/production_weight_cache_frontier"
BASELINE_CACHE_PKL="${BASELINE_DIR}/artifacts/production_weight_cache_frontier_raw.pkl"
[ -f "${BASELINE_CACHE_PKL}" ] || BASELINE_CACHE_PKL="${BASELINE_DIR}/artifacts/production_weight_cache_raw.pkl"
[ -d "${BASELINE_CACHE_DIR}" ] || BASELINE_CACHE_DIR="${BASELINE_DIR}/artifacts/production_weight_cache"

# Sanity: baseline's KL validation already wrote a JSON; we'll merge our results into our own JSON.
echo "[validate] baseline production cache: ${BASELINE_CACHE_PKL} (dir: ${BASELINE_CACHE_DIR})"

ASSIGNMENT_ARGS=()
# Baseline frontier points: read the validated-surrogate manifest authoritatively
# rather than globbing *.json (which would sweep up manifest.json itself).
while IFS=$'\t' read -r label path; do
  [[ -n "$label" && -n "$path" ]] || continue
  ASSIGNMENT_ARGS+=("--assignment" "baseline__${label}=${path}")
done < <(python3 - "${BASELINE_DIR}/artifacts/pareto_assignments/manifest.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for row in payload.get("candidates", []):
    print(f"{row['label']}\t{row['path']}")
PY
)
# Multi-shot endpoints
for B in "${BUDGETS[@]}"; do
  ms_layer_cfg="${BASE_DIR}/multishot-${B}/artifacts/layer_config.json"
  [[ -f "$ms_layer_cfg" ]] || { echo "[validate] missing ${ms_layer_cfg}; skipping"; continue; }
  ASSIGNMENT_ARGS+=("--assignment" "multishot_${B}=${ms_layer_cfg}")
done

VALIDATION_JSON="${COMPARISON_DIR}/kl_comparison.json"
BASE_FOR_KL="${BASELINE_DIR}/artifacts/layer_config.json"

python3 -m prismaquant.validate_assignments_kl \
  --model "${MODEL_PATH}" \
  --probe "${BASELINE_DIR}/artifacts/probe.pkl" \
  --costs "${BASELINE_DIR}/artifacts/cost.pkl" \
  --base-assignment "${BASE_FOR_KL}" \
  "${ASSIGNMENT_ARGS[@]}" \
  --output "${VALIDATION_JSON}" \
  --formats "${FORMATS}" \
  --dataset "${DATASET}" \
  --n-calib-samples "${NSAMPLES}" \
  --calib-seqlen "${SEQLEN}" \
  --calib-repeats "${KL_REPEATS:-4}" \
  --calib-repeat-seed-stride 100 \
  --dtype bf16 \
  --device cuda \
  --kl-scope last_token \
  --assignment-materialization hooks \
  --production-weight-cache "${BASELINE_CACHE_PKL}" \
  --production-cache-dir-override "${BASELINE_CACHE_DIR}" \
  --production-cache-lru-gb 32.0 \
  --production-cache-prefetch require \
  --production-cache-prefetch-workers 4

# ---------------------------------------------------------------------------
# Step 4: summary table
# ---------------------------------------------------------------------------
echo
echo "[validate] === step 4: summary ==="
python3 - "${VALIDATION_JSON}" "${COMPARISON_DIR}/summary.json" <<'PY'
import json
import re
import sys
from pathlib import Path

vp = Path(sys.argv[1])
data = json.loads(vp.read_text())

per = {}
for entry in data.get("assignments", []):
    label = entry.get("label", "")
    bpp = entry.get("achieved_bits") or entry.get("bpp")
    kl  = entry.get("kl") or entry.get("end_kl") or entry.get("mean_kl")
    if label.startswith("baseline__"):
        m = re.search(r"target_([0-9p]+)", label)
        if m:
            tb = float(m.group(1).replace("p", "."))
            per.setdefault(tb, {})["baseline"] = {"label": label, "bpp": bpp, "kl": kl}
    elif label.startswith("multishot_"):
        tb = float(label.split("_", 1)[1])
        per.setdefault(tb, {})["multishot"] = {"label": label, "bpp": bpp, "kl": kl}

rows = []
print("\n=== KL comparison (baseline vs 2-shot) ===")
print(f"{'budget':>8} {'baseline_kl':>14} {'multishot_kl':>14} {'delta':>10} {'pct':>8}")
print("-" * 60)
for tb in sorted(per.keys()):
    b = per[tb].get("baseline", {})
    m = per[tb].get("multishot", {})
    bk = b.get("kl")
    mk = m.get("kl")
    if bk is None or mk is None:
        print(f"{tb:>8.2f} {'?':>14} {'?':>14}")
        continue
    delta = mk - bk
    pct   = 100.0 * delta / bk if bk else 0.0
    print(f"{tb:>8.2f} {bk:>14.4e} {mk:>14.4e} {delta:>+10.3e} {pct:>+7.2f}%")
    rows.append({"budget": tb, "baseline_kl": bk, "multishot_kl": mk, "delta_kl": delta, "delta_pct": pct})

Path(sys.argv[2]).write_text(json.dumps({
    "schema": "prismaquant.multi_shot.validation_summary.v1",
    "comparison": rows,
}, indent=2, sort_keys=True) + "\n")
print(f"\n[validate] summary → {sys.argv[2]}")
PY

echo
echo "[validate] === ALL DONE ==="
echo "[validate]   base dir       : ${BASE_DIR}"
echo "[validate]   kl_comparison  : ${VALIDATION_JSON}"
echo "[validate]   summary        : ${COMPARISON_DIR}/summary.json"
