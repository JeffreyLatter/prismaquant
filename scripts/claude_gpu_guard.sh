#!/usr/bin/env bash
# PreToolUse guard: ONE GPU workload at a time on the 128 GB unified-memory
# Spark. Blocks launching a NEW GPU docker container while a serving container
# (pq_*, publishing :8000) is running — the exact pattern behind the 2026-07-21
# box OOM + power cycle (test containers run next to a ~110 GiB-resident
# server). Allowed: docker exec into the serve, non-GPU docker, and the serve's
# own relaunch (--name pq_* — serve scripts docker rm -f their own name first).
# Reads the Claude Code hook JSON on stdin; emits a deny decision JSON to block.
set -u
payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null)
[ -z "$cmd" ] && exit 0

# Only interested in commands that launch a new GPU container.
case "$cmd" in
  *"docker run"*"--gpus"*|*"--gpus"*"docker run"*) ;;
  *) exit 0 ;;
esac

# The serve's own relaunch names a pq_ container; it self-replaces safely.
case "$cmd" in
  *"--name pq_"*|*'--name "pq_'*) exit 0 ;;
esac

serving=$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null \
          | grep -E '^pq_.*8000' | head -1)
if [ -n "${GPU_GUARD_TEST_FORCE_SERVING:-}" ]; then
  serving="pq_test (forced by GPU_GUARD_TEST_FORCE_SERVING)"
fi
[ -z "$serving" ] && exit 0

reason="GPU-exclusivity guard: serving container [$serving] is running and holds most of the 128 GB unified pool. Launching another GPU container beside it is the box-OOM pattern that forced a power cycle (2026-07-21). Stop the serve first (docker stop/rm the pq_ container), run the GPU work, then relaunch the serve."
jq -n --arg r "$reason" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
