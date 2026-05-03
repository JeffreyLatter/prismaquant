#!/usr/bin/env bash
# Run all Qwen 4B smoke configurations sequentially. Each config is a
# fresh probe + cost + allocator + export. Container `pq-qwen4b-smoke`
# is reused (removed before each run); workspaces under
# /home/rob/dq-runs/qwen4b-smoke/<config>/ are kept independent.
#
# CONFIGS env var lets a caller override the order or subset.
# Default: baseline → cumulative wins ladder → solos. Total ~3-4 hours.
set -euo pipefail

CONFIGS="${CONFIGS:-baseline +damp +clip +cache_fp32 +halo +block_match full solo_halo solo_damp solo_block}"
WORKTREE="${WORKTREE:-/home/rob/prismaquant-quality-wins}"
LAUNCHER=/home/rob/prismaquant/examples/launchers/launch-qwen4b-smoke.sh

[ -x "$LAUNCHER" ] || { echo "missing launcher: $LAUNCHER"; exit 1; }

echo "[smoke-all] configs: $CONFIGS"
echo "[smoke-all] worktree: $WORKTREE"

for cfg in $CONFIGS; do
    artifact_dir=/home/rob/dq-runs/qwen4b-smoke/$cfg/exported
    if [ -d "$artifact_dir" ] && [ -f "$artifact_dir/config.json" ]; then
        echo "[smoke-all] $cfg already has artifact, skipping"
        continue
    fi

    echo ""
    echo "=== [$(date '+%H:%M:%S')] $cfg ==="
    CONFIG=$cfg WORKTREE=$WORKTREE bash "$LAUNCHER"

    # Wait for container to finish.
    while docker ps --format '{{.Names}}' | grep -q '^pq-qwen4b-smoke$'; do
        sleep 30
    done

    # Check for success — log should end with "[done]"
    if ! tail -5 /home/rob/dq-runs/qwen4b-smoke/$cfg/logs/export.log 2>/dev/null \
            | grep -q '\[done\]'; then
        echo "[smoke-all] WARN $cfg did not complete cleanly"
        echo "  tail:"
        tail -10 /home/rob/dq-runs/qwen4b-smoke/$cfg/logs/export.log 2>/dev/null \
            | sed 's/^/    /'
    else
        echo "[smoke-all] $cfg done"
    fi
done

echo ""
echo "[smoke-all] all configs processed. Artifacts:"
for cfg in $CONFIGS; do
    a=/home/rob/dq-runs/qwen4b-smoke/$cfg/exported
    if [ -d "$a" ]; then
        size=$(du -sh "$a" 2>/dev/null | cut -f1)
        echo "  $cfg: $size"
    else
        echo "  $cfg: MISSING"
    fi
done
