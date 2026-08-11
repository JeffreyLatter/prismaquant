#!/usr/bin/env bash
# AURA-on-CB cost-only driver.  Inventory/preflight are CPU-only.  Launch
# obtains the one host-wide GPU mutex before any CUDA/container process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT="${REPO_ROOT}/tools/aura_cb_reprice_preflight.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_LOCK="${GPU_LOCK:-/home/rob/dq-runs/gpu.lock}"
DATASET="${DATASET:-/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
IMAGE="${IMAGE:-gridbook:test}"

usage() {
  cat <<'EOF'
Usage:
  tools/run_aura_cb_reprice.sh inventory dsv4
  tools/run_aura_cb_reprice.sh preflight dsv4
  tools/run_aura_cb_reprice.sh launch dsv4

  MODEL_PATH=/abs/Qwen3.8-27B-FP8 \
  WORK_DIR=/home/rob/dq-runs/qwen38-27b-aura-cb \
  CB_COL_WEIGHTS=/abs/cb_col_weights.pkl \
  CB_CODEBOOK_BUNDLE=/abs/cb_learned_bundle.pqcb \
    tools/run_aura_cb_reprice.sh preflight dense

Actions:
  inventory  Report existing/build/block items and return success. CPU only.
  preflight  Require every launch gate. CPU only.
  launch     Re-run preflight, take flock -x on gpu.lock, re-check, then run.

Preflight/launch also require AURA_CB_LAUNCH_RECEIPT to name an external,
HEAD-bound implementation/test attestation. Inventory does not.

The current branch is expected to FAIL launch preflight until the streamed,
identity-resumable AURA/CB capabilities named by the report are implemented.
EOF
}

ACTION="${1:-}"
TARGET="${2:-}"
if [[ ! "$ACTION" =~ ^(inventory|preflight|launch)$ ]] \
   || [[ ! "$TARGET" =~ ^(dsv4|dense)$ ]]; then
  usage >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/dsv4-flash-0731}"
if [[ "$TARGET" == "dsv4" ]]; then
  MODEL_PATH="${MODEL_PATH:-${RUN_ROOT}/source}"
  WORK_DIR="${WORK_DIR:-${RUN_ROOT}/aura-cb-reprice}"
  CB_COL_WEIGHTS="${CB_COL_WEIGHTS:-${RUN_ROOT}/prod-cal-0p7/artifacts/cb_col_weights.pkl}"
else
  MODEL_PATH="${MODEL_PATH:-}"
  WORK_DIR="${WORK_DIR:-}"
  CB_COL_WEIGHTS="${CB_COL_WEIGHTS:-}"
fi
CB_CODEBOOK_BUNDLE="${CB_CODEBOOK_BUNDLE:-}"
CB_ROUTED_MOE_BOOK_SELECTION="${CB_ROUTED_MOE_BOOK_SELECTION:-}"
AURA_CB_LAUNCH_RECEIPT="${AURA_CB_LAUNCH_RECEIPT:-}"

fp8_menu() {
  local first="$1"
  local last="$2"
  local result=""
  local rung
  for ((rung = first; rung <= last; rung++)); do
    if [[ -n "$result" ]]; then
      result+=","
    fi
    result+="FP8_CB_K${rung}"
  done
  printf '%s\n' "$result"
}

# Menus are env-overridable (same form as CALIB_* below) so a caller can pin the
# priced menu to exactly the rungs its learned bundle trained. Two reasons this
# must NOT stay hardcoded:
#   1. Under CB_CODEBOOK_SOURCE_SCOPE=fp8 there is no per-rung lattice fallback —
#      codebook_for() raises ValueError on an untrained rung, mid-render.
#   2. A cached-menu rung costs 2.002 bytes/qparam of disk, K-independent, so a
#      21-rung menu on a 27B is ~619 GiB against ~281 GB free.
# Defaults are the DSv4 values: unexported env reproduces the prior behaviour
# byte-for-byte.
EXPERT_FORMATS="${EXPERT_FORMATS:-$(fp8_menu 28 33)}"
NONEXPERT_FORMATS="${NONEXPERT_FORMATS:-$(fp8_menu 28 48)}"
DENSE_FORMATS="${DENSE_FORMATS:-$NONEXPERT_FORMATS}"
CALIB_NSAMPLES="${CALIB_NSAMPLES:-16}"
CALIB_SEQLEN="${CALIB_SEQLEN:-512}"
CALIB_SEED="${CALIB_SEED:-42}"
AURA_NPROBES="${AURA_NPROBES:-32}"

preflight_args=(
  "$TARGET"
  --repo "$REPO_ROOT"
  --run-root "$RUN_ROOT"
  --dataset "$DATASET"
  --gpu-lock "$GPU_LOCK"
)
if [[ -n "$MODEL_PATH" ]]; then
  preflight_args+=(--model "$MODEL_PATH")
fi
if [[ -n "$WORK_DIR" ]]; then
  preflight_args+=(--work-dir "$WORK_DIR")
fi
if [[ -n "$CB_CODEBOOK_BUNDLE" ]]; then
  preflight_args+=(--cb-codebook-bundle "$CB_CODEBOOK_BUNDLE")
fi
if [[ -n "$CB_COL_WEIGHTS" ]]; then
  preflight_args+=(--cb-col-weights "$CB_COL_WEIGHTS")
fi
if [[ -n "$CB_ROUTED_MOE_BOOK_SELECTION" ]]; then
  preflight_args+=(
    --routed-book-selection "$CB_ROUTED_MOE_BOOK_SELECTION"
  )
fi
if [[ -n "$AURA_CB_LAUNCH_RECEIPT" ]]; then
  preflight_args+=(--implementation-receipt "$AURA_CB_LAUNCH_RECEIPT")
fi

run_preflight() {
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON_BIN" "$PREFLIGHT" "${preflight_args[@]}" "$@"
}

if [[ "$ACTION" == "inventory" ]]; then
  run_preflight --allow-blocked
  exit 0
fi

if [[ "$ACTION" == "preflight" ]]; then
  run_preflight
  exit $?
fi

# Static checks happen before waiting for the mutex.  They cannot authorize a
# CUDA launch by themselves; the same checks run again while the lock is held.
if [[ "$TARGET" == "dsv4" ]]; then
  run_preflight --verify-hashes
else
  run_preflight
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[aura-cb] BLOCK: docker is not installed" >&2
  exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[aura-cb] BLOCK: known-good image is absent: $IMAGE" >&2
  exit 2
fi

case "$WORK_DIR" in
  ""|/tmp|/tmp/*|/home/rob/prismaquant|/home/rob/prismaquant/*)
    echo "[aura-cb] BLOCK: unsafe/forbidden WORK_DIR=$WORK_DIR" >&2
    exit 2
    ;;
esac
mkdir -p "$WORK_DIR" "$WORK_DIR/logs" "$WORK_DIR/artifacts" \
  "$WORK_DIR/cache/pairs" "$WORK_DIR/checkpoints" "$WORK_DIR/tmp" \
  "$WORK_DIR/torchinductor" "$WORK_DIR/ext"

# Atomic exclusivity.  There is deliberately no check-then-act GPU probe.
exec 9<"$GPU_LOCK"
flock -x 9

if [[ "$TARGET" == "dsv4" ]]; then
  run_preflight --verify-hashes
else
  run_preflight
fi

CONTAINER_NAME="pq-aura-cb-${TARGET}-$$"
container_started=0
cleanup() {
  local status=$?
  if [[ "$container_started" == "1" ]]; then
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  return "$status"
}
trap cleanup EXIT INT TERM

docker_args=(
  run --rm --name "$CONTAINER_NAME" --gpus all --ipc=host
  -v "$REPO_ROOT:/pq:ro"
  -v "$MODEL_PATH:$MODEL_PATH:ro"
  -v "$WORK_DIR:$WORK_DIR"
  -v "$DATASET:$DATASET:ro"
  -e "PYTHONPATH=/pq"
  -e "PYTHONDONTWRITEBYTECODE=1"
  -e "MODEL_PATH=$MODEL_PATH"
  -e "WORK_DIR=$WORK_DIR"
  -e "DATASET=$DATASET"
  -e "TMPDIR=$WORK_DIR/tmp"
  -e "TORCHINDUCTOR_CACHE_DIR=$WORK_DIR/torchinductor"
  -e "PRISMAQUANT_CB_EXT_DIR=$WORK_DIR/ext"
  -e "PRISMAQUANT_CB_ENCODE_COMPILE=${PRISMAQUANT_CB_ENCODE_COMPILE:-1}"
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  -e "CB_CODEBOOK_SOURCE=learned"
  -e "CB_CODEBOOK_SOURCE_SCOPE=fp8"
  -e "CB_CODEBOOK_BUNDLE=$CB_CODEBOOK_BUNDLE"
  -e "CB_SCALE_CODING=two_tier"
  -e "CB_SCALE_SWEEP=1"
  -e "CB_SCALE_SWEEP_SCOPE=all"
  -e "PRISMAQUANT_CB_LDLQ=0"
  -e "PRISMAQUANT_CB_LDLQ_SCOPE=none"
  -e "PRISMAQUANT_CB_MINCHAIN=0"
  -e "PRISMAQUANT_CB_ENCODE_TIER=balanced"
  -e "CB_COL_WEIGHTS=$CB_COL_WEIGHTS"
  -e "EXPERT_FORMATS=$EXPERT_FORMATS"
  -e "NONEXPERT_FORMATS=$NONEXPERT_FORMATS"
  -e "DENSE_FORMATS=$DENSE_FORMATS"
  -e "CALIB_NSAMPLES=$CALIB_NSAMPLES"
  -e "CALIB_SEQLEN=$CALIB_SEQLEN"
  -e "CALIB_SEED=$CALIB_SEED"
  -e "AURA_NPROBES=$AURA_NPROBES"
  -e "COST_MODE=aura"
  -e "COST_RENDER=cached-menu"
  -e "COST_OBJECTIVE=aura-adjoint"
)

# Mount immutable value inputs read-only when they are not already under the
# writable work tree.  Docker accepts a file bind at the same absolute path.
case "$CB_CODEBOOK_BUNDLE" in
  "$WORK_DIR"/*) ;;
  *) docker_args+=(-v "$CB_CODEBOOK_BUNDLE:$CB_CODEBOOK_BUNDLE:ro") ;;
esac
case "$CB_COL_WEIGHTS" in
  "$WORK_DIR"/*|"$MODEL_PATH"/*) ;;
  *) docker_args+=(-v "$CB_COL_WEIGHTS:$CB_COL_WEIGHTS:ro") ;;
esac

if [[ "$TARGET" == "dsv4" ]]; then
  docker_args+=(
    -v "$RUN_ROOT/prod-cal-0p7:$RUN_ROOT/prod-cal-0p7:ro"
    -e "RUN_ROOT=$RUN_ROOT"
    -e "CB_ROUTED_MOE_BOOK_SELECTION=$CB_ROUTED_MOE_BOOK_SELECTION"
  )
  case "$CB_ROUTED_MOE_BOOK_SELECTION" in
    "$WORK_DIR"/*|"$RUN_ROOT"/*) ;;
    *)
      docker_args+=(
        -v "$CB_ROUTED_MOE_BOOK_SELECTION:$CB_ROUTED_MOE_BOOK_SELECTION:ro"
      )
      ;;
  esac
fi

container_started=1
if [[ "$TARGET" == "dense" ]]; then
  docker "${docker_args[@]}" --entrypoint bash "$IMAGE" -lc '
set -euo pipefail
CACHE_MANIFEST="$WORK_DIR/cache/production_weight_cache.pkl"
COST_OUTPUT="$WORK_DIR/artifacts/cost_aura.pkl"
# Never skip merely because a nonempty output exists.  The future resume
# interfaces named here must validate model/menu/imatrix/bundle/calibration
# identity per unit, then either resume or prove the result complete.
python3 -m prismaquant.build_production_cache \
  --model "$MODEL_PATH" \
  --output "$CACHE_MANIFEST" \
  --formats "$DENSE_FORMATS" \
  --render-scope format-menu \
  --dataset "$DATASET" \
  --n-calib-samples "$CALIB_NSAMPLES" \
  --calib-seqlen "$CALIB_SEQLEN" \
  --calib-seed "$CALIB_SEED" \
  --dtype bf16 \
  --max-act-rows "${CACHE_MAX_ACT_ROWS:-1024}" \
  --enable gptq,static_act_order,joint_scale_opt \
  --cache-dir "$WORK_DIR/cache/pairs" \
  --checkpoint-dir "$WORK_DIR/checkpoints/cache" \
  --resume \
  --col-weights "$CB_COL_WEIGHTS" \
  2>&1 | tee -a "$WORK_DIR/logs/cached_menu.log"
python3 -m prismaquant.aura_cost \
  --model "$MODEL_PATH" \
  --cost-mode aura \
  --output "$COST_OUTPUT" \
  --formats "$DENSE_FORMATS,FP8_SOURCE" \
  --production-cache "$CACHE_MANIFEST" \
  --require-production-cache \
  --n-probes "$AURA_NPROBES" \
  --n-calib-samples "$CALIB_NSAMPLES" \
  --calib-seqlen "$CALIB_SEQLEN" \
  --calib-seed "$CALIB_SEED" \
  --dtype auto \
  --dataset "$DATASET" \
  --hook-harvest \
  --gradient-checkpointing \
  --n-linear-chunks "${AURA_LINEAR_CHUNKS:-8}" \
  --probe-microbatch "${AURA_PROBE_MICROBATCH:-8}" \
  --min-free-gib "${AURA_MIN_FREE_GIB:-18}" \
  --accurate-chunk-bytes \
  --checkpoint-dir "$WORK_DIR/checkpoints/aura" \
  --resume \
  2>&1 | tee -a "$WORK_DIR/logs/aura_cost.log"
echo "[aura-cb] dense cost ready: $COST_OUTPUT"
'
else
  # The module named here is the bounded implementation seam: it must consume
  # the split source-rate plan, stream DSv4, checkpoint each unit, and write the
  # hybrid payload.  Preflight refuses before the lock while it is absent.
  docker "${docker_args[@]}" --entrypoint bash "$IMAGE" -lc '
set -euo pipefail
python3 -m prismaquant.dsv4_aura_cb_reprice \
  --model "$MODEL_PATH" \
  --probe "$RUN_ROOT/prod-cal-0p7/artifacts/probe.pkl" \
  --activation-cache-dir "$RUN_ROOT/prod-cal-0p7/act" \
  --col-weights "$CB_COL_WEIGHTS" \
  --dataset "$DATASET" \
  --work-dir "$WORK_DIR" \
  --expert-formats "$EXPERT_FORMATS" \
  --nonexpert-formats "$NONEXPERT_FORMATS" \
  --routed-book-selection "$CB_ROUTED_MOE_BOOK_SELECTION" \
  --checkpoint-dir "$WORK_DIR/checkpoints" \
  --n-calib-samples "$CALIB_NSAMPLES" \
  --calib-seqlen "$CALIB_SEQLEN" \
  --calib-seed "$CALIB_SEED" \
  --n-probes "$AURA_NPROBES" \
  --resume \
  --cost-mode aura \
  --require-production-cache \
  2>&1 | tee -a "$WORK_DIR/logs/dsv4_aura_cb_reprice.log"
'
fi
container_started=0
trap - EXIT INT TERM
echo "[aura-cb] complete under exclusive lock: target=$TARGET work=$WORK_DIR"
