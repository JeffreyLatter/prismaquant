#!/usr/bin/env bash
# Run the Qwen 4B audit: per-win attribution against the validator
# methodology (apples-to-apples vs the shipped 27B's 4.16 reference).
#
# Sequencing:
#   1. Wait for the in-flight validator-baseline-full eval to complete
#      (only one vLLM serve at a time on this box).
#   2. For each solo config (solo_damp, solo_clip, solo_block, solo_dnh,
#      full_v2): symlink baseline's probe.pkl/cost.pkl/layer_config.json
#      and activation cache into the config workspace, then run the
#      launcher — it'll detect the existing artifacts and skip straight
#      to export. ~5 min per config instead of ~25.
#   3. Run validator on every artifact in one shot.
#
# Sequential by necessity (single GPU). No timers; just polls for
# container liveness.
set -euo pipefail

CONFIGS=(solo_damp solo_clip solo_block solo_dnh full_v2)
SMOKE="${WORK_ROOT:-/home/rob/dq-runs/qwen4b-smoke}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.5-4B-bf16}"
LAUNCHER=/home/rob/prismaquant/examples/launchers/launch-qwen4b-smoke.sh

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# Wait for any active eval to clear.
log "waiting for any pq-qwen4b-eval-* container to clear ..."
while docker ps --format '{{.Names}}' | grep -qE '^pq-qwen4b-eval-'; do
    sleep 30
done
log "eval area clear"

# Pre-stage each solo config's workspace by COPYING baseline artifacts.
# Symlinks don't work because the docker container mounts each config's
# /work but symlink targets escape the mount. Activation cache is only
# 456 MB so 5× copies = ~2.3 GB; cheap relative to fresh probes.
for cfg in "${CONFIGS[@]}"; do
    mkdir -p "$SMOKE/$cfg"/{artifacts,logs,exported,export-cache,work}
    for f in probe.pkl cost.pkl layer_config.json; do
        if [ -f "$SMOKE/baseline/artifacts/$f" ] \
                && [ ! -e "$SMOKE/$cfg/artifacts/$f" ]; then
            cp "$SMOKE/baseline/artifacts/$f" \
                   "$SMOKE/$cfg/artifacts/$f"
        fi
    done
    if [ -d "$SMOKE/baseline/act" ] && [ ! -e "$SMOKE/$cfg/act" ]; then
        cp -r "$SMOKE/baseline/act" "$SMOKE/$cfg/act"
    fi
    log "pre-staged $cfg with baseline artifacts (copied)"
done

# Run each export sequentially.
for cfg in "${CONFIGS[@]}"; do
    if [ -f "$SMOKE/$cfg/exported/config.json" ]; then
        log "$cfg already has exported artifact, skipping"
        continue
    fi
    log "=== launching $cfg ==="
    docker rm -f pq-qwen4b-smoke 2>/dev/null || true
    rm -rf "$SMOKE/$cfg/export-cache"   # always fresh; fingerprint would catch stale anyway
    CONFIG=$cfg WORK_ROOT="$SMOKE" MODEL_PATH="$MODEL_PATH" bash "$LAUNCHER"

    # Poll for container completion.
    while docker ps --format '{{.Names}}' | grep -q '^pq-qwen4b-smoke$'; do
        sleep 30
    done

    if [ -f "$SMOKE/$cfg/exported/config.json" ]; then
        size=$(du -sh "$SMOKE/$cfg/exported" | cut -f1)
        log "$cfg DONE ($size)"
    else
        log "$cfg FAILED — check $SMOKE/$cfg/logs/export.log"
        tail -10 "$SMOKE/$cfg/logs/export.log" || true
    fi
done

# Final validator pass on every artifact.
log "=== running validator on all configs ==="
ALL_CONFIGS="baseline,full,$(IFS=,; echo "${CONFIGS[*]}")"
log "validator configs: $ALL_CONFIGS"
python3 -u /home/rob/prismaquant/examples/launchers/qwen4b-validator-eval.py \
    --configs "$ALL_CONFIGS" \
    --output "$SMOKE/audit_eval.csv" \
    2>&1 | tee "$SMOKE/audit_eval.log"

log "=== AUDIT DONE ==="
log "results in $SMOKE/audit_eval.csv"
