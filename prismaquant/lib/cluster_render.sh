# cluster_render.sh — multi-box production-cache rendering.
#
# Sourced from run-pipeline.sh. Everything here is a no-op when
# CLUSTER_NODES is unset/empty: the single call site that matters
# (build_prod_cache_maybe_clustered) falls straight through to the exact
# local invocation the caller would have made anyway, byte-for-byte.
#
# When CLUSTER_NODES is set (space-separated SSH-reachable addresses — use
# the fast-NIC address, not a management/VPN hostname, since this rsyncs the
# model and dispatches the render over it), the production-cache render is
# split across (1 + N-nodes) shards via PRISMAQUANT_UNIT_SHARD=i/N
# (prismaquant/unit_sharding.py), one per box, merged back with
# tools/merge_unit_shards.py. See docs/design/runtime_flags.md and the
# CLUSTER_NODES section of run-pipeline.sh's own usage comment.
#
# Assumes: every listed node is reachable over SSH with the key already
# configured (no interactive auth), and has the SAME absolute repo path and
# a working .venv already provisioned (`uv sync`) — this file checks for
# that and fails with the exact one-line fix rather than provisioning it.

: "${CLUSTER_SSH_OPTS:=-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"

_cluster_node_count() {
  # 1 (this box) + however many addresses are in CLUSTER_NODES.
  local n=1
  local node
  for node in $CLUSTER_NODES; do
    n=$((n + 1))
  done
  echo "$n"
}

_cluster_ssh() {
  local node="$1"; shift
  # shellcheck disable=SC2086
  ssh $CLUSTER_SSH_OPTS "$node" "$@"
}

_cluster_check_prereqs() {
  # Fail fast, before touching any data, if a node isn't ready. Prints the
  # exact one-time setup command per the user's earlier decision: this
  # script provisions nothing on a remote box on its own.
  local node ok=1
  for node in $CLUSTER_NODES; do
    if ! _cluster_ssh "$node" true 2>/dev/null; then
      echo "[cluster] ERROR: cannot SSH to '$node' (CLUSTER_NODES). Check the address and that the shared key is authorized there." >&2
      ok=0
      continue
    fi
    if ! _cluster_ssh "$node" "test -x '${REPO_ROOT}/.venv/bin/python3'" 2>/dev/null; then
      echo "[cluster] ERROR: '$node' has no prismaquant venv at ${REPO_ROOT}/.venv yet. One-time setup:" >&2
      echo "  ssh $node 'cd ${REPO_ROOT} && (which uv || curl -LsSf https://astral.sh/uv/install.sh | sh) && uv sync'" >&2
      echo "  (repo tree gets rsynced there automatically first; this just needs the venv to already exist.)" >&2
      ok=0
    fi
  done
  [[ "$ok" == "1" ]]
}

_cluster_sync_to_node() {
  local node="$1"
  echo "[cluster] syncing repo + model + calibration data to $node ..."
  _cluster_ssh "$node" "mkdir -p '${REPO_ROOT}' '${WORK_DIR}/artifacts' '${WORK_DIR}/logs' '$(dirname -- "$MODEL_PATH")'"
  rsync -az --delete \
    --exclude='.git/' --exclude='dq-runs/' --exclude='runs/' \
    --exclude='scratch/' --exclude='archive/' --exclude='.venv/' \
    --exclude='__pycache__/' --exclude='.pytest_cache/' \
    "${REPO_ROOT}/" "${node}:${REPO_ROOT}/"
  # -L: MODEL_PATH is frequently an HF hub cache snapshot dir of symlinks
  # into ../../blobs/; a plain --archive sync would ship broken links since
  # blobs/ isn't synced. -L dereferences to the real bytes on both plain
  # directories (no-op) and HF cache layouts (the fix).
  rsync -azL --size-only "${MODEL_PATH}/" "${node}:${MODEL_PATH}/"
  if [[ -n "${DATASET:-}" && -f "$DATASET" ]]; then
    _cluster_ssh "$node" "mkdir -p '$(dirname -- "$DATASET")'"
    rsync -az "$DATASET" "${node}:${DATASET}"
  fi
  if [[ -n "${EXPERT_GATE_DATASET:-}" && -f "$EXPERT_GATE_DATASET" ]]; then
    _cluster_ssh "$node" "mkdir -p '$(dirname -- "$EXPERT_GATE_DATASET")'"
    rsync -az "$EXPERT_GATE_DATASET" "${node}:${EXPERT_GATE_DATASET}"
  fi
  # The render needs the allocator's assignment (layer_config.json) and
  # whatever else stage [4/4] reads from WORK_DIR/artifacts (e.g. CB col
  # weights) — all already finalized locally by the time this stage runs,
  # since probe/cost/allocator are single-box only in v1. Excludes the
  # production-cache output dirs themselves: those don't exist yet at sync
  # time (this stage is about to create them, per-shard) and would just be
  # large no-op transfers on a later re-sync otherwise.
  rsync -az \
    --exclude='production_weight_cache*/' \
    --exclude='production_render_score_weight_cache/' \
    --exclude='production_weight_cache_frontier*/' \
    --exclude='*.pkl.shard*.pkl' \
    "${WORK_DIR}/artifacts/" "${node}:${WORK_DIR}/artifacts/"
}

_cluster_sync_all() {
  local node
  local -a pids=()
  for node in $CLUSTER_NODES; do
    _cluster_sync_to_node "$node" &
    pids+=("$!")
  done
  local pid failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == "0" ]] || { echo "[cluster] ERROR: sync to one or more nodes failed" >&2; return 1; }
}

_cluster_sentinel_preflight() {
  # Cross-box determinism preflight (tools/cluster_render_sentinel.py):
  # both/all boxes render the same K small units and must produce byte-
  # identical manifests before any box commits to the real sharded render.
  local formats="$1"
  local sentinel_dir="${WORK_DIR}/artifacts/cluster_sentinel"
  mkdir -p "$sentinel_dir"
  echo "[cluster] sentinel preflight: rendering $formats on every node ..."

  local -a manifests=("${sentinel_dir}/local.json")
  python3 "${REPO_ROOT}/tools/cluster_render_sentinel.py" render \
    --model "$MODEL_PATH" --k 8 --formats "$formats" \
    --output "${sentinel_dir}/local.json" &
  local -a pids=("$!")

  local node i=0
  for node in $CLUSTER_NODES; do
    i=$((i + 1))
    local remote_manifest="${sentinel_dir}/node${i}.json"
    _cluster_ssh "$node" \
      "cd '${REPO_ROOT}' && PRISMAQUANT_DETERMINISTIC=1 .venv/bin/python3 -m tools.cluster_render_sentinel render --model '$MODEL_PATH' --k 8 --formats '$formats' --output '$remote_manifest'" &
    pids+=("$!")
    manifests+=("$remote_manifest")
  done

  local pid failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == "0" ]] || { echo "[cluster] ERROR: sentinel render failed on one or more nodes" >&2; return 1; }

  i=0
  for node in $CLUSTER_NODES; do
    i=$((i + 1))
    rsync -az "${node}:${sentinel_dir}/node${i}.json" "${sentinel_dir}/node${i}.json"
  done

  local -a compare_args=()
  local m
  for m in "${manifests[@]}"; do
    compare_args+=(--manifest "$m")
  done
  echo "[cluster] sentinel: comparing manifests across $(( ${#manifests[@]} )) node(s) ..."
  python3 "${REPO_ROOT}/tools/cluster_render_sentinel.py" compare "${compare_args[@]}"
}

# build_prod_cache_maybe_clustered OUTPUT_PATH CACHE_DIR LOG_PATH SENTINEL_FORMATS -- <build_production_cache args...>
#
# CLUSTER_NODES empty: runs the exact local invocation the caller would have
# made directly, unchanged (same args, same tee'd log, same exit-code
# propagation under `set -o pipefail`).
#
# CLUSTER_NODES set: syncs, runs the sentinel preflight, dispatches
# PRISMAQUANT_UNIT_SHARD=i/N to each node in parallel (this box is always
# shard 0), collects the shard outputs, and merges them into OUTPUT_PATH /
# CACHE_DIR with tools/merge_unit_shards.py — same two paths the caller
# expects populated either way.
build_prod_cache_maybe_clustered() {
  local output_path="$1" cache_dir="$2" log_path="$3" sentinel_formats="$4"
  shift 4
  [[ "$1" == "--" ]] || { echo "[cluster] internal error: expected -- separator" >&2; return 1; }
  shift
  local -a bpc_args=("$@")

  if [[ -z "${CLUSTER_NODES:-}" ]]; then
    python3 -m prismaquant.build_production_cache \
      "${bpc_args[@]}" --output "$output_path" --cache-dir "$cache_dir" \
      2>&1 | tee "$log_path"
    return "${PIPESTATUS[0]}"
  fi

  # Cross-box correctness needs a reproducible GPTQ Cholesky/U-update
  # reduction order — the sentinel itself refuses to certify a match
  # without this (see tools/cluster_render_sentinel.py), and the same
  # reasoning applies to the real sharded render, not just the preflight.
  export PRISMAQUANT_DETERMINISTIC=1

  _cluster_check_prereqs || return 1
  _cluster_sync_all || return 1
  _cluster_sentinel_preflight "$sentinel_formats" || return 1

  local n
  n=$(_cluster_node_count)
  echo "[cluster] rendering across $n shard(s) (this box + ${CLUSTER_NODES}) ..."

  local -a shard_outputs=()
  local -a pids=()
  local -a pid_labels=()

  # Shard 0: local, backgrounded like every other shard so all N run
  # concurrently.
  local shard0_out="${output_path}.shard0.pkl"
  local shard0_dir="${cache_dir}_shard0"
  shard_outputs+=("$shard0_out")
  (
    PRISMAQUANT_UNIT_SHARD="0/${n}" \
      python3 -m prismaquant.build_production_cache \
      "${bpc_args[@]}" --output "$shard0_out" --cache-dir "$shard0_dir" \
      > "${log_path}.shard0" 2>&1
  ) &
  pids+=("$!")
  pid_labels+=("shard 0 (local)")

  local node i=0
  for node in $CLUSTER_NODES; do
    i=$((i + 1))
    local shard_out="${output_path}.shard${i}.pkl"
    local shard_dir="${cache_dir}_shard${i}"
    shard_outputs+=("$shard_out")
    # Only the actual command gets %q-quoted; `&&` is spliced in as a real
    # shell operator afterward — %q would otherwise escape it to a literal
    # `\&\&` token instead of a command separator, breaking the remote
    # command entirely.
    local -a remote_cmd=(env "PRISMAQUANT_UNIT_SHARD=${i}/${n}" \
      "PRISMAQUANT_DETERMINISTIC=1" \
      .venv/bin/python3 -m prismaquant.build_production_cache \
      "${bpc_args[@]}" --output "$shard_out" --cache-dir "$shard_dir")
    local quoted
    quoted=$(printf '%q ' "${remote_cmd[@]}")
    _cluster_ssh "$node" "cd '${REPO_ROOT}' && ${quoted}" > "${log_path}.shard${i}" 2>&1 &
    pids+=("$!")
    pid_labels+=("shard ${i} (${node})")
  done

  local failed=0
  local idx
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[cluster] ERROR: ${pid_labels[$idx]} failed — see ${log_path}.shard${idx}" >&2
      failed=1
    fi
  done
  [[ "$failed" == "0" ]] || { echo "[cluster] one or more shard renders failed" >&2; return 1; }

  # Pull remote shard outputs (+ their cache dirs, which merge_unit_shards
  # also reads) back to the control node.
  i=0
  for node in $CLUSTER_NODES; do
    i=$((i + 1))
    local shard_dir="${cache_dir}_shard${i}"
    rsync -az "${node}:${output_path}.shard${i}.pkl" "${output_path}.shard${i}.pkl"
    rsync -az "${node}:${shard_dir}/" "${shard_dir}/"
  done

  echo "[cluster] merging ${n} shard(s) into ${output_path} ..."
  local -a merge_args=()
  local so
  for so in "${shard_outputs[@]}"; do
    merge_args+=(--shard "$so")
  done
  python3 "${REPO_ROOT}/tools/merge_unit_shards.py" merge \
    "${merge_args[@]}" --output "$output_path" --output-cache-dir "$cache_dir" \
    2>&1 | tee "$log_path"
}

# run_aura_cost_maybe_clustered OUTPUT_PATH LOG_PATH PROD_CACHE_PATH PROD_CACHE_DIR -- <aura_cost.py args, no --output/--unit-filter/--n-linear-chunks>
#
# CLUSTER_NODES empty: the exact local invocation (AURA_COST_LINEAR_CHUNKS
# chunks, full scope, no unit-filter), unchanged.
#
# CLUSTER_NODES set: splits the model's decoder layers into 1+N contiguous
# layer-index ranges (tools/aura_cost_shard_ranges.py) and dispatches each
# box with --unit-filter scoped to its range and --n-linear-chunks 0 (auto
# — aura_cost sizes chunk count off the SCOPED weight footprint, so a
# smaller unit-filter genuinely shrinks compute, not just bookkeeping).
# Each box's cost pickle is disjoint by qname; merged with
# tools/merge_aura_cost_shards.py (a plain dict-union with an overlap/gap
# refusal, not the render-byte-balanced unit_sharding.py machinery — AURA
# cost has no rendered-bytes output to reconcile).
run_aura_cost_maybe_clustered() {
  local output_path="$1" log_path="$2" prod_cache_path="$3" prod_cache_dir="$4"
  shift 4
  [[ "$1" == "--" ]] || { echo "[cluster] internal error: expected -- separator" >&2; return 1; }
  shift
  local -a ac_args=("$@")

  if [[ -z "${CLUSTER_NODES:-}" ]]; then
    python3 -m prismaquant.aura_cost \
      "${ac_args[@]}" --n-linear-chunks "$AURA_COST_LINEAR_CHUNKS" \
      --output "$output_path" \
      2>&1 | tee "$log_path"
    return "${PIPESTATUS[0]}"
  fi

  _cluster_check_prereqs || return 1
  _cluster_sync_all || return 1

  # The AURA dW cache is an INPUT here (built by stage [2b/4], possibly
  # itself cluster-rendered and already merged locally by the time this
  # runs) — sync it explicitly. _cluster_sync_to_node's general artifacts
  # sync excludes production-cache dirs on purpose (they're an OUTPUT,
  # not yet built, at the sync time stage [2b/4]'s own cluster call uses).
  echo "[cluster] syncing AURA dW cache to $(_cluster_node_count_minus_one) node(s) ..."
  local node
  local -a sync_pids=()
  for node in $CLUSTER_NODES; do
    (
      _cluster_ssh "$node" "mkdir -p '$(dirname -- "$prod_cache_path")' '$prod_cache_dir'"
      rsync -az "$prod_cache_path" "${node}:${prod_cache_path}"
      rsync -az "${prod_cache_dir}/" "${node}:${prod_cache_dir}/"
    ) &
    sync_pids+=("$!")
  done
  local pid failed=0
  for pid in "${sync_pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == "0" ]] || { echo "[cluster] ERROR: syncing AURA dW cache to one or more nodes failed" >&2; return 1; }

  local n
  n=$(_cluster_node_count)
  echo "[cluster] AURA cost: splitting $n-way by decoder-layer range ..."
  local -a filters=()
  while IFS= read -r line; do
    filters+=("$line")
  done < <(python3 "${REPO_ROOT}/tools/aura_cost_shard_ranges.py" "$MODEL_PATH" "$n")
  if [[ "${#filters[@]}" != "$n" ]]; then
    echo "[cluster] ERROR: aura_cost_shard_ranges.py returned ${#filters[@]} filter(s), expected $n" >&2
    return 1
  fi

  # _auto_n_chunks (--n-linear-chunks 0) sizes chunks from *current* free
  # memory at decision time, which measured looser than reality on a real
  # 27B run (chunks=2 for a 124-Linear shard — a BIGGER per-chunk footprint
  # than the pipeline's own tuned AURA_COST_LINEAR_CHUNKS=8 for the full
  # 248-Linear model, ~62 vs ~31 Linears/chunk) and blew the watchdog floor
  # mid-chunk. Scale the known-safe full-model chunk count by this shard's
  # share of the model instead of trusting auto-sizing on a subset.
  local shard_chunks=$(( (AURA_COST_LINEAR_CHUNKS + n - 1) / n ))
  [[ "$shard_chunks" -lt 1 ]] && shard_chunks=1

  local -a shard_outputs=()
  local -a pids=()
  local -a pid_labels=()

  local shard0_out="${output_path}.shard0.pkl"
  shard_outputs+=("$shard0_out")
  (
    python3 -m prismaquant.aura_cost \
      "${ac_args[@]}" --n-linear-chunks "$shard_chunks" \
      --unit-filter "${filters[0]}" --output "$shard0_out" \
      > "${log_path}.shard0" 2>&1
  ) &
  pids+=("$!")
  pid_labels+=("shard 0 (local)")

  local i=0
  for node in $CLUSTER_NODES; do
    i=$((i + 1))
    local shard_out="${output_path}.shard${i}.pkl"
    shard_outputs+=("$shard_out")
    local -a remote_cmd=(.venv/bin/python3 -m prismaquant.aura_cost \
      "${ac_args[@]}" --n-linear-chunks "$shard_chunks" \
      --unit-filter "${filters[$i]}" --output "$shard_out")
    local quoted
    quoted=$(printf '%q ' "${remote_cmd[@]}")
    _cluster_ssh "$node" "cd '${REPO_ROOT}' && ${quoted}" > "${log_path}.shard${i}" 2>&1 &
    pids+=("$!")
    pid_labels+=("shard ${i} (${node})")
  done

  local idx
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[cluster] ERROR: ${pid_labels[$idx]} failed — see ${log_path}.shard${idx}" >&2
      failed=1
    fi
  done
  [[ "$failed" == "0" ]] || { echo "[cluster] one or more AURA cost shards failed" >&2; return 1; }

  i=0
  for node in $CLUSTER_NODES; do
    i=$((i + 1))
    rsync -az "${node}:${output_path}.shard${i}.pkl" "${output_path}.shard${i}.pkl"
  done

  echo "[cluster] merging ${n} AURA cost shard(s) into ${output_path} ..."
  local -a merge_args=()
  local so
  for so in "${shard_outputs[@]}"; do
    merge_args+=(--shard "$so")
  done
  python3 "${REPO_ROOT}/tools/merge_aura_cost_shards.py" \
    "${merge_args[@]}" --output "$output_path" \
    2>&1 | tee "$log_path"
}

_cluster_node_count_minus_one() {
  local n=0 node
  for node in $CLUSTER_NODES; do n=$((n + 1)); done
  echo "$n"
}

# run_cost_maybe_clustered OUTPUT_PATH LOG_PATH -- <incremental_measure_quant_cost.py args, no --output/--shard-range>
#
# Assumes the caller's args include `--work-dir "${WORK_DIR}/work"` (the
# pipeline's own convention) so this box's shard dir is
# "${WORK_DIR}/work/shards" on every node.
#
# CLUSTER_NODES empty: the exact local invocation, unchanged.
#
# CLUSTER_NODES set: dispatches 1+N boxes each with --shard-range i/N (an
# incremental_measure_quant_cost.py flag — see its --help — that computes
# only that box's slice of shard INDICES over the full, unrestricted shard
# schedule and exits before the merge; the schedule itself isn't restricted
# per box, so indices/filenames never collide across boxes, unlike
# --start-layer/--end-layer). Each box measures independently (no cross-
# shard carry, unlike the probe's reverse-mode sweep — this stage reads the
# already-complete probe.pkl + activation cache and measures per-Linear RTN
# error). Shard pickles are synced into one shard dir, then this box
# reruns the SAME command once more with --shard-range unset so the
# existing skip-if-exists path performs the merge + coverage gate.
run_cost_maybe_clustered() {
  local output_path="$1" log_path="$2"
  shift 2
  [[ "$1" == "--" ]] || { echo "[cluster] internal error: expected -- separator" >&2; return 1; }
  shift
  local -a mc_args=("$@")

  if [[ -z "${CLUSTER_NODES:-}" ]]; then
    python3 -m prismaquant.incremental_measure_quant_cost \
      "${mc_args[@]}" --output "$output_path" \
      2>&1 | tee "$log_path"
    return "${PIPESTATUS[0]}"
  fi

  _cluster_check_prereqs || return 1
  _cluster_sync_all || return 1

  # The activation cache (probe stage output) is an INPUT here and isn't
  # covered by _cluster_sync_to_node's general artifacts sync.
  echo "[cluster] syncing activation cache to $(_cluster_node_count_minus_one) node(s) ..."
  local node
  local -a sync_pids=()
  for node in $CLUSTER_NODES; do
    (
      _cluster_ssh "$node" "mkdir -p '${WORK_DIR}/act' '${WORK_DIR}/work/shards'"
      rsync -az "${WORK_DIR}/act/" "${node}:${WORK_DIR}/act/"
    ) &
    sync_pids+=("$!")
  done
  local pid failed=0
  for pid in "${sync_pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == "0" ]] || { echo "[cluster] ERROR: syncing activation cache to one or more nodes failed" >&2; return 1; }

  local n
  n=$(_cluster_node_count)
  echo "[cluster] base cost: splitting $n-way by shard index ..."

  local -a pids=()
  local -a pid_labels=()

  (
    python3 -m prismaquant.incremental_measure_quant_cost \
      "${mc_args[@]}" --shard-range "0/${n}" --output "$output_path" \
      > "${log_path}.shard0" 2>&1
  ) &
  pids+=("$!")
  pid_labels+=("shard 0 (local)")

  local i=0
  for node in $CLUSTER_NODES; do
    i=$((i + 1))
    local -a remote_cmd=(.venv/bin/python3 -m prismaquant.incremental_measure_quant_cost \
      "${mc_args[@]}" --shard-range "${i}/${n}" --output "$output_path")
    local quoted
    quoted=$(printf '%q ' "${remote_cmd[@]}")
    _cluster_ssh "$node" "cd '${REPO_ROOT}' && ${quoted}" > "${log_path}.shard${i}" 2>&1 &
    pids+=("$!")
    pid_labels+=("shard ${i} (${node})")
  done

  local idx
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[cluster] ERROR: ${pid_labels[$idx]} failed — see ${log_path}.shard${idx}" >&2
      failed=1
    fi
  done
  [[ "$failed" == "0" ]] || { echo "[cluster] one or more base-cost shards failed" >&2; return 1; }

  echo "[cluster] collecting cost shards from $(_cluster_node_count_minus_one) node(s) ..."
  for node in $CLUSTER_NODES; do
    rsync -az "${node}:${WORK_DIR}/work/shards/" "${WORK_DIR}/work/shards/"
  done

  echo "[cluster] merging base-cost shards into ${output_path} ..."
  python3 -m prismaquant.incremental_measure_quant_cost \
    "${mc_args[@]}" --output "$output_path" \
    2>&1 | tee "$log_path"
}
