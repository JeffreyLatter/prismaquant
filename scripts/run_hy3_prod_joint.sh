#!/usr/bin/env bash
# ============================================================================
# run_hy3_prod_joint.sh — Hy3 295B JOINT-MENU regeneration (CB + vanilla)
# ============================================================================
# Robert 2026-07-20: regenerate the shipped 2.9bpp CB artifact with vanilla
# NVFP4 + FP8_DYNAMIC added to the menu, so the allocator can exploit the
# measured joint-menu case (dense-tier A/B: K36 wins 87-100% of units, but 42
# outlier-row attention units favor NVFP4's group-16 scales act-weighted —
# and stock units also get native zero-expand prefill). The prior HF repo was
# pulled down pending this regeneration.
#
# Reuses the proven CB run's probe (menu-independent) + act cache + col
# weights, COPIED into WORK_DIR beforehand; cost is measured FRESH across the
# full joint menu (no cross-run cost mixing). MTP: the existing CB MTP sidecar
# (mtp_cb) is reused verbatim at merge time — its rungs are unchanged.
#
# REQUIRES: the streaming exporter's stock-CT extension (mixed-container
# packing for NVFP4/FP8_DYNAMIC) — hard-fails cleanly if absent.
# NO QUALITY CLAIMS (295B rule). Validation = load + coherent gen + bit-exact
# packing + speed, per the prior ship.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/dq-runs/hy3-prod/source}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/prod-hy3-joint-2p9}"
export TMPDIR="${WORK_DIR}/tmp"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs"

export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
export CB_EXPERT_EMPIRICAL=0
export PRISMAQUANT_EXPERT_COST_SAMPLE=16
export CB_LADDER_INTERP=1
export EXPORT_STREAMING=auto

# --- JOINT menu: the CB ladder + vanilla NVFP4 (4.5) + FP8_DYNAMIC (8) ---
export FORMATS="NVFP4_CB_K14,NVFP4_CB_K16,NVFP4_CB_K18,NVFP4_CB_K20,FP8_CB_K28,FP8_CB_K32,FP8_CB_K36,FP8_CB_K44,NVFP4,FP8_DYNAMIC,BF16"
export CB_SCALE_CODING=two_tier
export TARGET_BITS=2.9
export PARETO_TARGETS="2.7,2.9,3.1"

export NSAMPLES=8
export SEQLEN=1024
export CACHE_HEADROOM_GB=45
export ACTIVATION_ROWS_LIMIT=1024
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PRISMAQUANT_PROBE_PREFETCH_LOOKAHEAD=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRISMAQUANT_PROBE_MIN_AVAILABLE_GB=40
export LAYERS_PER_SHARD=8

echo "============================================================================"
echo "Hy3 295B JOINT MENU — CB ladder + vanilla NVFP4/FP8_DYNAMIC @${TARGET_BITS} bpp"
echo "  WORK_DIR=$WORK_DIR (probe/act/col-weights pre-copied from the CB run)"
echo "  FORMATS=$FORMATS"
echo "  NO QUALITY CLAIMS. Validation = load + coherent-gen + bit-exact packing."
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"

echo
echo "##  JOINT EXPORT DONE — next: merge MTP sidecar (merge_hy3_mtp_reshard.py"
echo "##  --main-dir ${WORK_DIR}/exported_nvfp4_cb --mtp-dir /home/rob/dq-runs/prod-hy3-nvfp4cb-2p9/mtp_cb),"
echo "##  serve smoke + speed, TEB, re-upload."
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
