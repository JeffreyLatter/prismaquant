#!/usr/bin/env bash
# Exact-pin, one-Spark DSv4 CB load/generation gate.
#
# Run each arm separately; each invocation creates a fresh container and
# installs Gridbook from the immutable PrismaQuant runtime pin inside it:
#
#   MODEL=/abs/path/to/artifact scripts/serve_dsv4_cb_validate.sh eager
#   MODEL=/abs/path/to/artifact scripts/serve_dsv4_cb_validate.sh graph
#
# This closes only native_export.{eager,graph}.  It does not claim quality or
# throughput and deliberately exposes no speculative-decode option.
set -euo pipefail

ARM=${1:-}
case "$ARM" in
  eager|graph) ;;
  *) echo "usage: MODEL=/absolute/artifact $0 {eager|graph}" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
. "$REPO/prismaquant/gridbook_runtime/gridbook_runtime.sh"
gridbook_runtime_prepare

MODEL=${MODEL:-}
if [[ -z "$MODEL" || "$MODEL" != /* || ! -d "$MODEL" ]]; then
  echo "REFUSE: MODEL must be an existing absolute artifact directory" >&2
  exit 2
fi
MODEL=$(cd -- "$MODEL" && pwd -P)
SHIPCARD=${SHIPCARD:-$MODEL/shipcard.json}
if [[ ! -f "$MODEL/config.json" || ! -f "$MODEL/quant_config.json" \
      || ! -f "$SHIPCARD" ]]; then
  echo "REFUSE: MODEL must contain config.json, quant_config.json, and the requested shipcard" >&2
  exit 2
fi
if ! compgen -G "$MODEL/*.safetensors" >/dev/null; then
  echo "REFUSE: MODEL has no safetensors checkpoint" >&2
  exit 2
fi

# DSv4 evidence pin: eugr Spark vLLM 0.26.1rc1.dev515 at g653ebb52d.
BASE_IMAGE=eugr/spark-vllm@sha256:7bf752a9fa225b528b27c6a1118cb1727cddd7c383096d83281010c4f8b407bc
VLLM_VERSION=0.26.1rc1.dev515+g653ebb52d.d20260808
VLLM_COMMIT=653ebb52d
GRAPH_COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
if [[ -n "${SERVED_MODEL:-}" ]]; then
  echo "REFUSE: exact DSv4 gate owns the per-run served-model nonce" >&2
  exit 2
fi
if command -v openssl >/dev/null 2>&1; then
  SERVE_NONCE=$(openssl rand -hex 16)
else
  SERVE_NONCE=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
fi
if [[ ! "$SERVE_NONCE" =~ ^[0-9a-f]{32}$ ]]; then
  echo "REFUSE: could not generate a 128-bit served-model session nonce" >&2
  exit 2
fi
SERVED_MODEL=dsv4-flash-gridbook-${SERVE_NONCE}
PORT=${PORT:-8000}
EXPECTED_GPU_LOCK=/home/rob/dq-runs/gpu.lock
LOCK=${GPU_LOCK:-$EXPECTED_GPU_LOCK}
START_FLOOR_GIB=${START_FLOOR_GIB:-110}
READY_FLOOR_GIB=${READY_FLOOR_GIB:-8}
WATCHDOG_GIB=${WATCHDOG_GIB:-4}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-1800}
RUN_ROOT=${RUN_ROOT:-$(dirname -- "$MODEL")/cb-serve-validation}
EVIDENCE=${EVIDENCE:-$RUN_ROOT/$ARM}
EXT_CACHE_ROOT=${EXT_CACHE_ROOT:-/home/rob/dq-runs/gridbook-ext-cache}
EXT_CACHE=$EXT_CACHE_ROOT/${GRIDBOOK_RUNTIME_COMMIT}-eugr-${VLLM_COMMIT}-${ARM}
NAME=${NAME:-pq-dsv4-cb-${ARM}}

mkdir -p -- "$EVIDENCE" "$EXT_CACHE" "$(dirname -- "$LOCK")"
EVIDENCE=$(cd -- "$EVIDENCE" && pwd -P)
EXT_CACHE=$(cd -- "$EXT_CACHE" && pwd -P)
case "$EVIDENCE/" in
  "$MODEL/"*) echo "REFUSE: EVIDENCE must be outside the immutable MODEL tree" >&2; exit 2 ;;
esac
case "$EXT_CACHE/" in
  "$MODEL/"*) echo "REFUSE: EXT_CACHE must be outside the immutable MODEL tree" >&2; exit 2 ;;
esac
WATCHDOG_LOG=$EVIDENCE/memory-watchdog.log
SERVE_LOG=$EVIDENCE/serve.log
CAPTURE_LOG=$EVIDENCE/capture-evidence.log
MANIFEST=$EVIDENCE/serve_manifest.json
RESULT=$EVIDENCE/validation.json

for value in "$START_FLOOR_GIB" "$READY_FLOOR_GIB" "$WATCHDOG_GIB"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "REFUSE: memory floors must be whole GiB values" >&2
    exit 2
  fi
done
if (( START_FLOOR_GIB < 110 || READY_FLOOR_GIB < 8 || WATCHDOG_GIB < 4 \
      || READY_FLOOR_GIB <= WATCHDOG_GIB )); then
  echo "REFUSE: safety floors require start>=110, ready>=8, watchdog>=4, ready>watchdog GiB" >&2
  exit 2
fi
if [[ "$LOCK" != "$EXPECTED_GPU_LOCK" ]]; then
  echo "REFUSE: exact DSv4 gate requires GPU_LOCK=$EXPECTED_GPU_LOCK" >&2
  exit 2
fi

# Header-only and sidecar-only preflight.  Refuse an incomplete decoder map or
# a missing/mismatched released DSpark overlay before reserving the single GPU.
(cd -- "$REPO" && python3 - "$MODEL" <<'PY'
import json
import sys

from prismaquant.validate_cb_endpoint import (
    validate_cb_artifact,
    validate_cb_artifact_decode_contract,
)

artifact = sys.argv[1]
quant_config = validate_cb_artifact(artifact)
evidence = validate_cb_artifact_decode_contract(artifact, quant_config)
print(json.dumps(evidence, sort_keys=True))
PY
)

exec 9>"$LOCK"
if ! flock -n -x 9; then
  echo "REFUSE: GPU lock is held: $LOCK" >&2
  exit 75
fi
: > "$WATCHDOG_LOG"
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "REFUSE: container name already exists: $NAME" >&2
  exit 2
fi
if ss -H -ltn "sport = :$PORT" | grep -q .; then
  echo "REFUSE: TCP port $PORT is already listening" >&2
  exit 2
fi
docker image inspect "$BASE_IMAGE" >/dev/null

available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
start_floor_kib=$((START_FLOOR_GIB * 1048576))
if (( available_kib < start_floor_kib )); then
  echo "REFUSE: MemAvailable ${available_kib} KiB < start floor ${start_floor_kib} KiB" >&2
  exit 3
fi

{
  date --iso-8601=seconds
  echo "ARM=$ARM"
  echo "MODEL=$MODEL"
  echo "BASE_IMAGE=$BASE_IMAGE"
  echo "VLLM_VERSION=$VLLM_VERSION"
  echo "VLLM_COMMIT=$VLLM_COMMIT"
  echo "GRIDBOOK_RUNTIME_COMMIT=$GRIDBOOK_RUNTIME_COMMIT"
  echo "GRIDBOOK_RUNTIME_VERSION=$GRIDBOOK_RUNTIME_VERSION"
  echo "MEM_AVAILABLE_KIB=$available_kib"
  docker image inspect "$BASE_IMAGE"
  nvidia-smi
  free -h
} > "$EVIDENCE/prelaunch.log" 2>&1

# A fresh ephemeral container is load-bearing here.  Reusing one container for
# both arms can make graph pass on extensions or allocator state loaded by the
# eager arm, and no longer proves either exact install independently.
CID=$(docker create --pull=never --name "$NAME" --gpus all --ipc=host \
  --user 0:0 --oom-score-adj=1000 \
  --log-driver local --log-opt max-size=100m --log-opt max-file=3 \
  -p "127.0.0.1:$PORT:8000" \
  -v "$REPO:/repo:ro" \
  -v "$MODEL:/model:ro" \
  -v "$EVIDENCE:/evidence:rw" \
  -v "$EXT_CACHE:/opt/gridbook/ext-cache:rw" \
  "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" \
  -e "PQ_ARM=$ARM" \
  -e "PQ_SERVED_MODEL=$SERVED_MODEL" \
  -e "PQ_VLLM_VERSION=$VLLM_VERSION" \
  -e "PQ_GRAPH_COMPILATION_CONFIG=$GRAPH_COMPILATION_CONFIG" \
  -e XDG_CACHE_HOME=/opt/gridbook/ext-cache/xdg \
  -e PRISMAQUANT_CB_EXT_DIR=/opt/gridbook/ext-cache \
  -e PRISMAQUANT_CB_GEMV=inherited \
  -e PRISMAQUANT_CB_BF16_SM120=0 \
  -e PRISMAQUANT_CB_FP4_FUSED_MIDM=0 \
  -e PRISMAQUANT_CB_MOE_PERSISTENT_B=0 \
  -e PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG=0 \
  -e PRISMAQUANT_CB_FUSED_MIDM=1 \
  -e PRISMAQUANT_CB_GROUPED_TRIM=1 \
  -e PRISMAQUANT_CB_PREFILL_CHUNK_BYTES=1073741824 \
  -e PRISMAQUANT_CB_DECODE_CONTRACT=v1 \
  -e PRISMAQUANT_SKIP_CB_CAST_CHECK=0 \
  -e PRISMAQUANT_PRELOAD_FUSED=1 \
  -e GRIDBOOK_MXFP8_DENSE=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$BASE_IMAGE" -lc '
    set -euo pipefail
    # Absence is part of the released environment contract.  In particular,
    # literal 0 is invalid for both fused-FP4 selectors and expert chunking.
    unset CUDACXX CXX \
      PRISMAQUANT_CB_DECODE PRISMAQUANT_CB_EXPAND PRISMAQUANT_CB_PREFILL \
      PRISMAQUANT_CB_FUSED_FP4 PRISMAQUANT_CB_FUSED_FP4_MOE \
      PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK \
      PRISMAQUANT_CB_FP8_SCHED PRISMAQUANT_CB_FP4V2_SCHED \
      PRISMAQUANT_CB_W2_SCHED PRISMAQUANT_CB_W2_ROWS \
      PRISMAQUANT_CB_W2_WARPS PRISMAQUANT_CUTLASS_INCLUDE \
      PRISMAQUANT_DEBUG_PREFIXES
    bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container
    test "$(python3 -c '\''from importlib.metadata import version; print(version("gridbook"))'\'')" = "$PQ_GRIDBOOK_RUNTIME_VERSION"
    test "$(python3 -c '\''from importlib.metadata import version; print(version("vllm"))'\'')" = "$PQ_VLLM_VERSION"
    arm_args=()
    if [[ "$PQ_ARM" == eager ]]; then
      arm_args+=(--enforce-eager)
    else
      arm_args+=(--compilation-config "$PQ_GRAPH_COMPILATION_CONFIG")
    fi
    extra_route_args=()
    if python3 - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("/model/quant_config.json").read_text())

def contains_mxfp4_wire(value):
    if isinstance(value, dict):
        return any(contains_mxfp4_wire(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_mxfp4_wire(item) for item in value)
    return value == "mxfp4_e2m1_ue8m0_g32"

raise SystemExit(0 if contains_mxfp4_wire(config) else 1)
PY
    then
      extra_route_args+=(--moe-backend marlin)
    fi
    exec /usr/local/bin/vllm serve /model \
      --served-model-name "$PQ_SERVED_MODEL" \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --tokenizer-mode deepseek_v4 \
      --generation-config vllm \
      --quantization gridbook \
      --tensor-parallel-size 1 \
      --kv-cache-dtype fp8 \
      --kv-cache-memory-bytes 1073741824 \
      --max-model-len 8192 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 512 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.90 \
      "${arm_args[@]}" "${extra_route_args[@]}"
  ')
echo "$CID" > "$EVIDENCE/container-id.txt"

watchdog_pid=
cleanup() {
  set +e
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" >/dev/null 2>&1 || true
    wait "$watchdog_pid" 2>/dev/null || true
  fi
  docker logs "$CID" > "$SERVE_LOG" 2>&1 || true
  docker inspect "$CID" > "$EVIDENCE/container-final-inspect.json" 2>&1 || true
  if [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) == true ]]; then
    docker stop -t 30 "$CID" >/dev/null 2>&1 || docker kill "$CID" >/dev/null 2>&1 || true
  fi
  docker rm -f "$CID" >/dev/null 2>&1 || true
  free -h > "$EVIDENCE/memory-after-exit.txt" 2>&1 || true
}
on_signal() {
  code=$1
  trap - EXIT INT TERM HUP
  cleanup
  exit "$code"
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

docker start "$CID" >/dev/null
(
  while [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) == true ]]; do
    timestamp=$(date --iso-8601=seconds)
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    printf '%s mem_available_kib=%s\n' "$timestamp" "$available_kib" >> "$WATCHDOG_LOG"
    if (( available_kib < WATCHDOG_GIB * 1048576 )); then
      printf '%s WATCHDOG: MemAvailable %s KiB below %s GiB; stopping %s\n' \
        "$timestamp" "$available_kib" "$WATCHDOG_GIB" "$CID" >> "$WATCHDOG_LOG"
      : > "$EVIDENCE/watchdog-tripped"
      docker stop -t 5 "$CID" >/dev/null 2>&1 || docker kill "$CID" >/dev/null 2>&1 || true
      break
    fi
    sleep 2
  done
) &
watchdog_pid=$!

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if curl --fail --silent --show-error "http://127.0.0.1:$PORT/v1/models" \
      > "$EVIDENCE/models.json"; then
    break
  fi
  if [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) != true ]]; then
    echo "REFUSE: $ARM serving container exited before READY" >&2
    exit 1
  fi
  sleep 5
done
if ! curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "REFUSE: $ARM serving container did not become ready" >&2
  exit 1
fi

sleep 10
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < READY_FLOOR_GIB * 1048576 )); then
  echo "REFUSE: post-READY MemAvailable is below ${READY_FLOOR_GIB} GiB" >&2
  exit 3
fi
if [[ -e "$EVIDENCE/watchdog-tripped" ]]; then
  echo "REFUSE: memory watchdog tripped" >&2
  exit 3
fi

# Exercise one inference before fingerprinting so lazily loaded Gridbook/JIT
# extensions are part of the server-side residency scan.
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$SERVED_MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0.0,\"top_p\":1.0,\"seed\":0,\"n\":1,\"stream\":false}" \
  "http://127.0.0.1:$PORT/v1/completions" > "$EVIDENCE/warmup.json"
python3 - "$EVIDENCE/warmup.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
text = payload["choices"][0]["text"]
if not isinstance(text, str) or not text.strip():
    raise SystemExit("warmup completion was empty")
PY

# Capture evidence is an immutable pre-validation snapshot. cleanup writes a
# separate final log, so the digest recorded on the graph receipt cannot be
# invalidated by the validator's own requests or container shutdown messages.
docker logs "$CID" > "$CAPTURE_LOG" 2>&1

# Fingerprinting is fatal for this gate.  The older serve helper is advisory;
# this ship receipt cannot be, because the stack identity is part of the proof.
docker exec -e PYTHONPATH=/repo "$CID" \
  python3 /repo/tools/serve_fingerprint.py write \
  --out /evidence/serve_manifest.json --image "$BASE_IMAGE" \
  --artifact-dir /model --base-url http://127.0.0.1:8000
test -s "$MANIFEST"

graph_evidence_args=()
if [[ "$ARM" == graph ]]; then
  graph_evidence_args+=(--serve-log "$CAPTURE_LOG")
fi
(cd -- "$REPO" && python3 -m prismaquant.validate_cb_endpoint \
  --arm "$ARM" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model-dir "$MODEL" \
  --model-name "$SERVED_MODEL" \
  --serve-manifest "$MANIFEST" \
  --shipcard "$SHIPCARD" \
  --output-json "$RESULT" \
  --defer-shipcard-fill \
  "${graph_evidence_args[@]}")

if [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) != true \
      || -e "$EVIDENCE/watchdog-tripped" ]]; then
  echo "REFUSE: container or memory watchdog failed during validation" >&2
  exit 3
fi
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < READY_FLOOR_GIB * 1048576 )); then
  echo "REFUSE: final MemAvailable is below ${READY_FLOOR_GIB} GiB" >&2
  exit 3
fi

# End the measured serve before mutating the receipt. This removes the final
# check/write race: no watchdog or server transition can occur after the
# terminal proof and leave a stale PASS behind.
if ! kill -0 "$watchdog_pid" 2>/dev/null; then
  echo "REFUSE: memory watchdog exited before planned server shutdown" >&2
  exit 3
fi
docker stop -t 30 "$CID" >/dev/null
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=
if [[ -e "$EVIDENCE/watchdog-tripped" ]]; then
  echo "REFUSE: memory watchdog tripped before clean server shutdown" >&2
  exit 3
fi
container_exit=$(docker inspect -f '{{.State.ExitCode}}' "$CID")
container_oom=$(docker inspect -f '{{.State.OOMKilled}}' "$CID")
if [[ "$container_oom" != false ]]; then
  echo "REFUSE: serving container was OOM-killed" >&2
  exit 3
fi
if [[ "$container_exit" != 0 && "$container_exit" != 137 && "$container_exit" != 143 ]]; then
  echo "REFUSE: serving container exited unexpectedly with code $container_exit" >&2
  exit 3
fi
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < READY_FLOOR_GIB * 1048576 )); then
  echo "REFUSE: post-shutdown MemAvailable is below ${READY_FLOOR_GIB} GiB" >&2
  exit 3
fi

# This is intentionally the terminal mutation. A PASS cannot reach the card
# until endpoint, clean server shutdown, watchdog, and memory checks all pass.
(cd -- "$REPO" && python3 - "$RESULT" "$SHIPCARD" "$MODEL" <<'PY'
import sys
from prismaquant.validate_cb_endpoint import commit_deferred_result

commit_deferred_result(sys.argv[1], sys.argv[2], sys.argv[3])
PY
)
echo "PASS: native_export.$ARM filled from exact-pinned CB serve"
