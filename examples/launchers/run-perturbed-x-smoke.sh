#!/usr/bin/env bash
set -euo pipefail

# User-gated Qwen 4B perturbed-X smoke. This script intentionally does not
# choose local artifact paths for you; set MODEL, PROBE, and INITIAL_COSTS.

: "${MODEL:?Set MODEL to the Qwen 4B model directory}"
: "${PROBE:?Set PROBE to the Qwen 4B sensitivity probe pickle}"
: "${INITIAL_COSTS:?Set INITIAL_COSTS to the initial cost pickle}"

OUT_DIR="${OUT_DIR:-$PWD/runs/qwen4b-perturbed-x-smoke}"
WORK_DIR="${WORK_DIR:-$OUT_DIR/work}"
EXPORT_DIR="${EXPORT_DIR:-$OUT_DIR/export-native}"
FORMATS="${FORMATS:-NVFP4,MXFP8,BF16}"
TARGET_BITS="${TARGET_BITS:-6.0}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-8}"
CALIB_SEQLEN="${CALIB_SEQLEN:-512}"
INPUT_ROWS="${INPUT_ROWS:-256}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
SHARD_BYTES="${SHARD_BYTES:-5368709120}"

mkdir -p "$OUT_DIR" "$WORK_DIR"

initial_config_args=()
if [[ -n "${INITIAL_CONFIG:-}" ]]; then
  initial_config_args=(--initial-config "$INITIAL_CONFIG")
fi

python3 -m prismaquant.iterate_perturbed_allocation \
  --model "$MODEL" \
  --probe "$PROBE" \
  --initial-costs "$INITIAL_COSTS" \
  --formats "$FORMATS" \
  --target-bits "$TARGET_BITS" \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUT_DIR" \
  --n-calib-samples "$N_CALIB_SAMPLES" \
  --calib-seqlen "$CALIB_SEQLEN" \
  --input-rows "$INPUT_ROWS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  "${initial_config_args[@]}"

python3 -m prismaquant.export_native_compressed \
  --model "$MODEL" \
  --perturbed-x-dir "$OUT_DIR" \
  --output "$EXPORT_DIR" \
  --device "$DEVICE" \
  --shard-bytes "$SHARD_BYTES"
