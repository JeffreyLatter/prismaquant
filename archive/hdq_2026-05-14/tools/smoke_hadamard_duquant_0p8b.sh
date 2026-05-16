#!/usr/bin/env bash
# Smoke launcher: Qwen3.5-0.8B with Hadamard-DuQuant.
#
# Per the Phase 7 first-ship criterion (HANDOVER.md):
#   - All 4 insertion points (residual, V→O, attn out_proj, down_proj)
#   - Learned rotations + calibrated zigzag permutation
#   - Format menu = {NVFP4, MXFP8_E4M3, FP8_E4M3, BF16}
#   - Target 4.5 bpp, NVFP4-dominant
#   - Loads in vLLM 0.20 (eager + CUDA-graph)
#   - Non-regressing calibration KL, WikiText-2 PPL, C4 PPL, BF16-argmax
#     vs current production stack (HALO off) at matched bpp.
#
# Invocation (from inside the spark-vllm-docker container OR any env that
# has CUDA torch + transformers + safetensors installed):
#
#   ./tools/smoke_hadamard_duquant_0p8b.sh
#
# Override defaults via env vars before invoking. Common ones:
#   MODEL_PATH=...       (default: local Qwen3.5-0.8B-untied)
#   WORK_DIR=...         (default: /home/rob/dq-runs/qwen35-0p8b-hdq-<timestamp>)
#   NSAMPLES=8           (calibration samples; 4 for faster smoke)
#   SEQLEN=512           (calibration sequence length; 256 for faster)
#   TARGET_BITS=4.5      (target bpp; 4.75 for production parity)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults — override via env vars before invoking.
: "${MODEL_PATH:=/home/rob/.cache/huggingface/qwen35-0p8b-bf16-untied}"
: "${WORK_DIR:=/home/rob/dq-runs/qwen35-0p8b-hdq-$(date +%Y%m%dT%H%M%SZ)}"
: "${FORMATS:=NVFP4,MXFP8_E4M3,FP8_E4M3,BF16}"
: "${TARGET_BITS:=4.5}"
: "${NSAMPLES:=8}"
: "${SEQLEN:=512}"
: "${DATASET:=/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
: "${TARGET_PROFILE:=vllm_qwen3_5_packed_moe}"

# Hadamard-DuQuant flags (Phase 6 pipeline gates).
: "${HADAMARD_DUQUANT:=1}"
: "${HADAMARD_DUQUANT_GROUP_SIZE:=16}"

# Production-cache levers: stay on the production default (gptq + scale_sweep)
# plus four_over_six's NVFP4 scale rule. We avoid PrismaClip / PrismaFisherClip
# (research-only after the 2026-05-13 27B top-32 regression) and the archived
# AWQ / SmoothQuant / BlockOrtho-G transforms.
: "${PRODUCTION_CACHE_LEVERS:=gptq,scale_sweep}"
: "${PRISMAQUANT_NVFP4_SCALE_RULE:=four_over_six_mse}"
: "${FISHER_OUTPUT_MSE_ALLOCATOR:=1}"

# HALO must be OFF for the first-ship criterion (Hadamard-DuQuant replaces
# HALO on the rotation axis; comparing against a HALO-off baseline is the
# locked design — see HANDOVER.md decision 2).
: "${HALO_MODE:=off}"

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/artifacts"

cat <<EOF
=================================================================
Hadamard-DuQuant smoke — Qwen3.5-0.8B
-----------------------------------------------------------------
MODEL_PATH:             $MODEL_PATH
WORK_DIR:               $WORK_DIR
FORMATS:                $FORMATS
TARGET_BITS:            $TARGET_BITS
NSAMPLES x SEQLEN:      $NSAMPLES x $SEQLEN
DATASET:                $DATASET
HADAMARD_DUQUANT:       $HADAMARD_DUQUANT
HADAMARD_DUQUANT_GROUP_SIZE: $HADAMARD_DUQUANT_GROUP_SIZE
PRODUCTION_CACHE_LEVERS: $PRODUCTION_CACHE_LEVERS
PRISMAQUANT_NVFP4_SCALE_RULE: $PRISMAQUANT_NVFP4_SCALE_RULE
FISHER_OUTPUT_MSE_ALLOCATOR: $FISHER_OUTPUT_MSE_ALLOCATOR
HALO_MODE:              $HALO_MODE  (must be 'off' for first-ship gate)
=================================================================
EOF

export MODEL_PATH WORK_DIR FORMATS TARGET_BITS NSAMPLES SEQLEN DATASET
export TARGET_PROFILE HADAMARD_DUQUANT HADAMARD_DUQUANT_GROUP_SIZE
export PRODUCTION_CACHE_LEVERS PRISMAQUANT_NVFP4_SCALE_RULE
export FISHER_OUTPUT_MSE_ALLOCATOR HALO_MODE

exec bash "$REPO_ROOT/prismaquant/run-pipeline.sh" "$@"
