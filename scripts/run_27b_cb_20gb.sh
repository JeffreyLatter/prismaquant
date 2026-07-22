#!/usr/bin/env bash
# ============================================================================
# run_27b_cb_20gb.sh — Qwen3.6-27B on the FULL gridbook stack, 20 GB target
# ============================================================================
# Robert 2026-07-21: "create a Qwen 3.6 27B using the new stack and serve with
# the new kernels; target 20 GB so it runs on a 24 GB 4090 or 5090."
#
# MENU: the full all-integer CB ladder (fp4 K12-K24 two-tier + fp8 K28-K48,
# 0.125-bpw steps — landed 2026-07-21, 116-test gate) + FP8_DYNAMIC + BF16.
# NO vanilla NVFP4, deliberately: an RTX 4090 (sm_89, Ada) has no NVFP4
# hardware, while every CB rung serves via bf16-decode GEMV / fp8-expand
# GEMMs and FP8_DYNAMIC has native sm_89 tensor cores — this one artifact can
# serve on both 4090 (sm_89) and 5090 (sm_120). gridbook's min capability is
# 80 and its JIT build targets the local arch. (Blackwell-only NVFP4 units
# won ZERO units on the Hy3 joint menu; at this bpp the CB rungs dominate it
# on error/byte anyway.)
#
# SIZE: 20 GB total artifact. Prior 5.5-bpp CB export = 23 GB with ~5.1 GB
# BF16 embed+lm_head (248k vocab, untied) => body ~25B params. 20 GB total
# - 5.1 embeds - ~0.7 visual/norms - ~0.7 CB MTP => body ~13.5 GB ~ 4.3-4.6
# bpp. TARGET_BITS=4.5 with a 4.25/4.75 Pareto bracket; the ship pick is by
# EXACT footprint <= 19.3 GB pre-MTP (nvfp4_cb_footprint is authoritative),
# NOT by the bpp label.
#
# REUSE from prod-27b-nvfp4cb-5p5 (menu-independent): probe.pkl (+settings),
# cb_col_weights.pkl. Cost is measured FRESH across the new menu (ladder
# interpolation keeps it anchor-priced). Act cache regenerates (prior was
# cleaned).
#
# MTP: mtp.* (15 keys) is encoded post-body via the canon throughput selector
# (prismaquant/mtp_rung_selection.py) and merged — "always include MTP".
# Visual tower: BF16 passthrough -> ignore list (unchanged from prior run).
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/qwen36-27b-bf16}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/prod-27b-cb-20gb}"
export TMPDIR="${WORK_DIR}/tmp"
PRIOR="/home/rob/dq-runs/prod-27b-nvfp4cb-5p5/artifacts"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs" "$WORK_DIR/artifacts"

# --- probe + col-weights reuse (menu-independent; settings travel along) ---
for f in cb_col_weights.pkl; do  # probe NOT reusable without its act/ cache (the probe stage writes activations)
  if [ -f "$PRIOR/$f" ] && [ ! -f "$WORK_DIR/artifacts/$f" ]; then
    cp -v "$PRIOR/$f" "$WORK_DIR/artifacts/$f"
  fi
done

# --- CB lane contract ---
export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
export CB_LADDER_INTERP=1
export EXPORT_STREAMING=auto

# --- FULL ladder, no vanilla NVFP4 (see header) ---
export FORMATS="$(python3 - <<'PYF'
fp4 = ",".join(f"NVFP4_CB_K{k}" for k in range(12, 25))
fp8 = ",".join(f"FP8_CB_K{k}" for k in range(28, 49))
print(f"{fp4},{fp8},FP8_DYNAMIC,BF16")
PYF
)"
export CB_SCALE_CODING=two_tier
export TARGET_BITS=4.5
export PARETO_TARGETS="4.25,4.5,4.75"

# --- calibration (box-forced 8x1024, same as the prior 27B CB run) ---
export NSAMPLES=8
export SEQLEN=1024
export CACHE_HEADROOM_GB=45
export ACTIVATION_ROWS_LIMIT=1024
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export VISUAL_FORMAT=BF16
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-auto}"

echo "============================================================================"
echo "Qwen3.6-27B FULL-LADDER CB @ ${TARGET_BITS} bpp (ship pick: footprint <= 19.3 GB)"
echo "  WORK_DIR=$WORK_DIR (probe/col-weights reused from prod-27b-nvfp4cb-5p5)"
echo "  FORMATS=$FORMATS"
echo "  4090/5090-portable: NO vanilla NVFP4; CB + FP8_DYNAMIC + BF16 only."
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"

echo
echo "##  BODY EXPORT DONE — next: MTP canon encode (mtp.*) + merge, footprint"
echo "##  check vs 20 GB, serve smoke (Spark), gold-lane KL-vs-BF16 + TEB."
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
