#!/usr/bin/env bash
# AURA-on-CB cost-only driver.  Inventory/preflight are CPU-only.  Launch
# obtains the one host-wide GPU mutex before any CUDA/container process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT="${REPO_ROOT}/tools/aura_cb_reprice_preflight.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_LOCK="${GPU_LOCK:-/home/rob/dq-runs/gpu.lock}"
DATASET="${DATASET:-/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
RUNTIME_IDENTITY_TOOL="${REPO_ROOT}/tools/container_runtime_identity.py"
# The image name is only a human-facing repository label.  The digest is the
# execution identity: local tags on the Spark have repeatedly been repointed
# between materially different Gridbook/Transformers environments.
PINNED_PRODUCER_IMAGE="gridbook@sha256:f7dad9260fea6f4207bd894acc9ebc034d91c599a70489a89ab1938a75db9c47"
IMAGE="${IMAGE:-$PINNED_PRODUCER_IMAGE}"

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
  # The worst routed layer retains 25.62 GiB of BF16 anchor deltas alongside
  # 12.18 GiB of source weights and its activation/cotangent state. Parameter
  # gradients are harvested inside backward, but a multi-layer source cache
  # would still erase the single-Spark safety margin. A 100 GiB reserve leaves
  # at most the forced current-layer slot at the observed 110 GiB free.  Keep
  # this explicit and overridable for a future box with a different memory
  # envelope; the launch log reports cache_slots and must stay <= 1 here.
  CACHE_HEADROOM_GB="${CACHE_HEADROOM_GB:-100}"
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
# The DSv4 defaults are the ON-LAW rungs, matching prismaquant.dsv4_aura_cb_reprice's
# FP8_EXPERT_FORMATS / FP8_NONEXPERT_FORMATS. That module raises
# "frozen CLI menus differ from the DSv4 on-law contract" on anything else, so the
# old contiguous K28..K33 / K28..K48 defaults could not run a dsv4 campaign at all --
# they aborted it at prepare time and every invocation had to override them. The
# contract lives in the module; these defaults now agree with it instead of
# contradicting it. (k % 4 == 0 is gridbook K1.2's fused mid-M prefill law; routed
# experts are additionally capped at K33 by the byte-exact source-payload ceiling,
# leaving exactly K28/K32.)
if [[ "$TARGET" == "dsv4" ]]; then
  EXPERT_FORMATS="${EXPERT_FORMATS:-FP8_CB_K28,FP8_CB_K32}"
  NONEXPERT_FORMATS="${NONEXPERT_FORMATS:-FP8_CB_K28,FP8_CB_K32,FP8_CB_K36,FP8_CB_K40,FP8_CB_K44,FP8_CB_K48}"
else
  EXPERT_FORMATS="${EXPERT_FORMATS:-$(fp8_menu 28 33)}"
  NONEXPERT_FORMATS="${NONEXPERT_FORMATS:-$(fp8_menu 28 48)}"
fi
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
# Tags are mutable even with --pull=never. Resolve only a full content digest
# or full local image ID, then launch the already-inspected ID so a concurrent
# retag cannot change the bytes between inspection and docker run.
if [[ ! "$IMAGE" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ \
      && ! "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "[aura-cb] BLOCK: IMAGE must be an immutable @sha256 digest or full image ID: $IMAGE" >&2
  exit 2
fi
IMAGE_ID="$(
  docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true
)"
if [[ ! "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]; then
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

# Producer identity for resumable pair shards.  The image ships no git binary
# and /pq is an immutable read-only mount, so the in-container producer cannot
# resolve its own commit -- which is the case PRISMAQUANT_IDENTITY_GIT_COMMIT
# documents ("for immutable/container source mounts whose checkout metadata is
# unavailable").  Resolve it on the host, where the preflight has just bound
# this exact commit to the launch receipt and asserted a clean tree over
# prismaquant/ tools/ tests/, and hand the value in.  Fail closed: an
# unresolvable commit is not a useful identity.
IDENTITY_GIT_COMMIT="$(
  git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}' 2>/dev/null || true
)"
if [[ ! "$IDENTITY_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[aura-cb] BLOCK: cannot resolve $REPO_ROOT HEAD for producer identity" >&2
  exit 2
fi
IDENTITY_GIT_DIRTY="$(
  git -C "$REPO_ROOT" status --porcelain --untracked-files=all -- \
    prismaquant tools tests docs/ARCHITECTURE.md
)"
if [[ -n "$IDENTITY_GIT_DIRTY" ]]; then
  echo "[aura-cb] BLOCK: producer identity scope became dirty after preflight" >&2
  exit 2
fi

# Bind resumes to the actual base image, reviewed source bytes, and external
# implementation receipt.  A missing identity is accepted only for an empty
# checkpoint tree; old checkpoints are never retroactively blessed.  The
# in-container check below re-hashes the read-only mount before any producer
# module is imported, eliminating stale site-package/PYTHONPATH ambiguity.
RUNTIME_IDENTITY_PATH="$WORK_DIR/checkpoints/container_runtime_identity.json"
runtime_identity_args=(
  write-or-verify
  --identity "$RUNTIME_IDENTITY_PATH"
  --checkpoint-root "$WORK_DIR/checkpoints"
  --source-root "$REPO_ROOT"
  --target "$TARGET"
  --image-ref "$IMAGE"
  --image-id "$IMAGE_ID"
  --git-commit "$IDENTITY_GIT_COMMIT"
  --implementation-receipt "$AURA_CB_LAUNCH_RECEIPT"
)
if [[ "$TARGET" == "dsv4" ]]; then
  runtime_identity_args+=(--require-receipt-image)
fi
"$PYTHON_BIN" "$RUNTIME_IDENTITY_TOOL" "${runtime_identity_args[@]}"

docker_args=(
  run --rm --name "$CONTAINER_NAME" --gpus all --ipc=host
  -v "$REPO_ROOT:/pq:ro"
  -v "$MODEL_PATH:$MODEL_PATH:ro"
  -v "$WORK_DIR:$WORK_DIR"
  -v "$DATASET:$DATASET:ro"
  -e "PYTHONPATH=/pq"
  -e "PYTHONDONTWRITEBYTECODE=1"
  -e "PRISMAQUANT_IDENTITY_GIT_COMMIT=$IDENTITY_GIT_COMMIT"
  -e "PQ_RUNTIME_IDENTITY_PATH=$RUNTIME_IDENTITY_PATH"
  -e "PQ_RUNTIME_IMAGE_REF=$IMAGE"
  -e "PQ_RUNTIME_IMAGE_ID=$IMAGE_ID"
  -e "PQ_RUNTIME_PRISMAQUANT_ROOT=/pq"
  -e "PQ_RUNTIME_PRISMAQUANT_GIT_COMMIT=$IDENTITY_GIT_COMMIT"
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

# Memory lever.  Dense campaigns retain the old opt-in behavior.  DSv4 sets a
# measured single-Spark default above so the container never falls back to the
# unsafe six-slot autoscale observed in launch #3.
#
# CACHE_HEADROOM_GB overrides the autoscaler's derived headroom AND sets the
# dynamic reserve (streaming_model.py `configure_dynamic_budget`).  The AURA
# renderer now consumes one canonical anchor directly into its BF16 dW, but the
# complete dW plane and current source/gradient state still require the forced
# one-layer ceiling.  The layer cache only shrinks inside `put()` -- which does
# not happen during that window.  The access pattern here is strictly
# sequential (one install/unload per layer), so a multi-slot cache buys nothing
# and capping it is close to free.
# CACHE_HEADROOM_GB is the ONLY real knob here -- `streaming_model.py:1108` is
# its single reader, and there is no env override for the cache's layer count.
if [[ -n "${CACHE_HEADROOM_GB:-}" ]]; then
  docker_args+=(-e "CACHE_HEADROOM_GB=$CACHE_HEADROOM_GB")
fi

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
    # The base image carries NO_CUDA_MEMORY_CACHING=1 for a historical
    # diagnostic. That turns every temporary VQ matmul allocation into a
    # driver cudaMalloc and leaves the production renderer allocator-bound.
    # Keep the normal caching allocator hot inside a layer; the streamed path
    # synchronizes and empty_cache()s after the last transient anchor and
    # before backward, which is the actual lifetime boundary.
    -e "PYTORCH_NO_CUDA_MEMORY_CACHING=0"
    -v "$RUN_ROOT/prod-cal-0p7:$RUN_ROOT/prod-cal-0p7:ro"
    -e "RUN_ROOT=$RUN_ROOT"
    -e "CB_ROUTED_MOE_BOOK_SELECTION=$CB_ROUTED_MOE_BOOK_SELECTION"
  )
  # A routed selection is not one file: load_routed_moe_cbl_selection requires
  # `book_root` to be a real directory, and load_banked_cbl_book opens each
  # named burn shard. Those live under $RUN_ROOT/cost-ldlq, which nothing
  # mounts -- only $RUN_ROOT/prod-cal-0p7 is bound. The old case statement made
  # that invisible: it treated any path under $RUN_ROOT as already mounted and
  # skipped the bind, so a selection in the natural place resolved on the host
  # during preflight and then vanished inside the container.
  #
  # Derive the mounts from the selection itself rather than hardcoding a tree,
  # so moving the bank or the shards cannot silently unmount them.
  if [[ -n "$CB_ROUTED_MOE_BOOK_SELECTION" ]]; then
    while IFS= read -r mount_path; do
      [[ -n "$mount_path" ]] || continue
      case "$mount_path" in
        "$WORK_DIR"|"$WORK_DIR"/*) continue ;;
      esac
      docker_args+=(-v "$mount_path:$mount_path:ro")
    done < <("$PYTHON_BIN" - "$CB_ROUTED_MOE_BOOK_SELECTION" <<'PY'
import json, os, sys
path = os.path.realpath(sys.argv[1])
payload = json.load(open(path))
roots = {os.path.realpath(payload["book_root"]), os.path.dirname(path)}
roots.update(
    os.path.dirname(os.path.realpath(cell["burn_shard"]))
    for cell in payload.get("cells", ())
)
# Drop any root already covered by an ancestor so docker gets no redundant binds.
keep = [r for r in sorted(roots)
        if not any(r != o and r.startswith(o + os.sep) for o in roots)]
print("\n".join(keep))
PY
    )
  fi
fi

container_started=1
if [[ "$TARGET" == "dense" ]]; then
  docker "${docker_args[@]}" --entrypoint bash "$IMAGE_ID" -lc '
set -euo pipefail
python3 /pq/tools/container_runtime_identity.py verify-mounted \
  --identity "$PQ_RUNTIME_IDENTITY_PATH" \
  --expected-root "$PQ_RUNTIME_PRISMAQUANT_ROOT" \
  --expected-image-ref "$PQ_RUNTIME_IMAGE_REF" \
  --expected-image-id "$PQ_RUNTIME_IMAGE_ID" \
  --expected-git-commit "$PQ_RUNTIME_PRISMAQUANT_GIT_COMMIT"
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
  docker "${docker_args[@]}" --entrypoint bash "$IMAGE_ID" -lc '
set -euo pipefail
python3 /pq/tools/container_runtime_identity.py verify-mounted \
  --identity "$PQ_RUNTIME_IDENTITY_PATH" \
  --expected-root "$PQ_RUNTIME_PRISMAQUANT_ROOT" \
  --expected-image-ref "$PQ_RUNTIME_IMAGE_REF" \
  --expected-image-id "$PQ_RUNTIME_IMAGE_ID" \
  --expected-git-commit "$PQ_RUNTIME_PRISMAQUANT_GIT_COMMIT"
if [[ "${PYTORCH_NO_CUDA_MEMORY_CACHING:-}" != "0" ]]; then
  echo "[aura-cb] BLOCK: DSv4 requires the layer-local caching allocator" >&2
  exit 2
fi
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
