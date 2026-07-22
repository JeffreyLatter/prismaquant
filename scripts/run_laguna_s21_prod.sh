#!/usr/bin/env bash
# ============================================================================
# run_laguna_s21_prod.sh — poolside Laguna-S-2.1 on the full gridbook standard
# ============================================================================
# Robert 2026-07-22: "go forward with Laguna. Leave enough room for 256k
# cache." The KV budget SETS the artifact size: GQA 8 kv-heads x 128 dim x
# 48 layers = ~96 KiB/token at fp8 KV -> 256k ctx ~= 24 GiB. Spark serving
# pool ~110 GiB (util ~0.88 + slack-gate discipline) - 24 KV - ~3 act/graphs
# - ~1-2 DFlash drafter => ~83 GB weight ceiling. Body ~116.7B params =>
# ~5.5 bpp — a near-lossless-class budget (fp8-CB K40-K48 band expected).
#
# Menu: the full all-integer standard (STANDARDS.md) — 34 CB rungs + NVFP4 +
# FP8_DYNAMIC + BF16. 256-expert top-10 MoE, per-expert on disk (the
# Qwen3.5/Ornith bridge), shared expert per layer, sigmoid routing.
# Embeddings/lm_head BF16 (untied, 100k vocab = only ~1.2 GB tax).
#
# Drafter: poolside's DFlash (separate checkpoint, vLLM-native class) — NOT
# an in-body MTP; attach at serve time; rung/precision via the canon
# selector once the body footprint is known. NO QUALITY CLAIMS at 117B
# without a served teacher A/B protocol — gates are load + coherent gen +
# packing checks + speed + the uniform-quant comparisons.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/dq-runs/laguna-s21/source}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/laguna-s21/prod}"
export TMPDIR="${WORK_DIR}/tmp"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs"

export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
export CB_LADDER_INTERP=1
export EXPORT_STREAMING=auto
export CB_EXPERT_EMPIRICAL=0
export PRISMAQUANT_EXPERT_COST_SAMPLE=16

export FORMATS="$(python3 - <<'PYF'
fp4 = ",".join(f"NVFP4_CB_K{k}" for k in range(12, 25))
fp8 = ",".join(f"FP8_CB_K{k}" for k in range(28, 49))
print(f"{fp4},{fp8},NVFP4,FP8_DYNAMIC,BF16")
PYF
)"
export CB_SCALE_CODING=two_tier
export TARGET_BITS="${TARGET_BITS:-5.5}"
export PARETO_TARGETS="${PARETO_TARGETS:-5.25,5.5,5.75}"

export NSAMPLES=8
export SEQLEN=1024
export CACHE_HEADROOM_GB=45
export ACTIVATION_ROWS_LIMIT=1024
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRISMAQUANT_PROBE_MIN_AVAILABLE_GB=40
export LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-auto}"

echo "============================================================================"
echo "Laguna-S-2.1 FULL-STANDARD CB @ ${TARGET_BITS} bpp (ship pick: footprint <= 83 GB"
echo "  so 256k fp8 KV [~24 GiB] + DFlash drafter fit the Spark pool)"
echo "  FORMATS=$FORMATS"
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"
echo "## LAGUNA EXPORT DONE — next: DFlash drafter (canon selector), serve smoke"
echo "## at 256k ctx (util per slack gate), speed, uniform-quant A/B."
