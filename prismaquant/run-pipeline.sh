#!/usr/bin/env bash
# run-pipeline.sh — end-to-end PrismaQuant pipeline: probe → cost →
# allocator → native compressed-tensors export → vLLM validate.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.6-35B-A3B \
#   WORK_DIR=./dq-runs/qwen36 \
#   FORMATS=NVFP4,FP8_DYNAMIC,BF16 \
#   TARGET_BITS=4.75 \
#   VISUAL_FORMAT=BF16 \
#   CALIBRATION_MODALITY=text-only \
#   ./prismaquant/run-pipeline.sh
#
# VISUAL_FORMAT accepts any format registry name allowed by the target
# serving profile and applies to visual-encoder Linears on multimodal models.
# In text-only calibration mode it's the Phase 1 uniform override for every
# visual Linear. In multimodal mode it's the fallback applied to un-probed
# visual Linears the allocator's Fisher-driven DP didn't touch (plus the
# graceful OOM fallback on 122B-scale VLMs that can't fit the whole model in
# RAM for the multimodal pass).
#
# CALIBRATION_MODALITY (text-only | multimodal):
#   - text-only (default): body-only streaming probe + cost. Visual
#     shards emit empty pickles; allocator stamps all visual Linears
#     with --visual-format uniformly.
#   - multimodal: also runs a non-streaming second probe pass with
#     image+text calibration (synthetic stub by default; set MM_DATASET
#     to a HuggingFace dataset id to use real images). The allocator
#     treats visual Linears as regular DP candidates when real Fisher
#     stats are present. Requires ~full-model RAM; falls back to
#     text-only behavior automatically on OOM.
#
# Memory note: probe + cost peak around 90 GB on a 35B model under
# BF16 calibration. The watchdog in incremental_measure_quant_cost
# aborts cleanly on swap pressure rather than OOM-killing the host.
#
# MTP is folded into the incremental probe + cost as a built-in shard;
# mtp.* tensors are measured in the same pass as the body and land in
# the same probe/cost pickles. No separate MTP stages.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the source HF model directory}"
: "${WORK_DIR:?Set WORK_DIR to a writable directory for artifacts}"
: "${FORMATS:=NVFP4,FP8_DYNAMIC,BF16}"
: "${TARGET_BITS:=4.75}"
: "${PARETO_TARGETS:=4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25}"
# Calibration defaults. 4x256 was the historical minimum for correctness
# validation; 32x1024 (N=32, T=1024 = 32768 tokens/sample, 32 samples)
# produces ~7% lower PPL on the resulting quantized artifact at a
# linear time cost in probe wall-time. Override for faster iteration.
: "${NSAMPLES:=32}"
: "${SEQLEN:=1024}"
# LAYERS_PER_SHARD controls how many decoder layers share one reverse
# sweep for Fisher accumulation. Larger = fewer sweeps = faster probe,
# but each shard needs more gradient + retained-activation memory.
# Default `auto` asks prismaquant.autoscale to pick from available RAM +
# model size. Set to an int (e.g. `2`) to override.
: "${LAYERS_PER_SHARD:=auto}"
# CACHE_HEADROOM_GB controls the streaming layer-cache budget
# (cache = free_RAM - headroom). Default `auto` picks from model +
# LAYERS_PER_SHARD. Set to a float to override.
: "${CACHE_HEADROOM_GB:=auto}"
: "${PREFETCH_LOOKAHEAD:=auto}"
: "${PREFETCH_WORKERS:=auto}"
: "${PREFETCH_MIN_AVAILABLE_GB:=auto}"
# GGUF lane defaults to 4x the activation rows: the GPTQ-into-k-quant
# Hessian is rank-starved at 256 rows on wide layers (rank 2.6% of a
# 9728-dim H) and degenerates toward RTN — measured on Qwen3-4B, 1024
# rows closed the render gap vs llama.cpp's imatrix quantizer from +20%
# to +7.7% KLD (top-1 at parity). See docs/gguf_lane.md.
# The nvfp4_cb lane inherits the same default: the weighted VQ codeword
# search wants a higher-rank imatrix than 256 rows (format-pipeline.md §6);
# 1024 is the analogy-to-GGUF starting point pending a CB-specific
# measurement (docs/nvfp4-cb-plan/format-pipeline.md open-Q 6).
if [[ "${EXPORT_CONTAINER:-compressed-tensors}" == "gguf" \
   || "${EXPORT_CONTAINER:-compressed-tensors}" == "nvfp4_cb" ]]; then
  : "${ACTIVATION_ROWS_LIMIT:=1024}"
else
  : "${ACTIVATION_ROWS_LIMIT:=256}"
fi
: "${DATASET:=/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
# MINOR-M2: packed-MoE experts use a cross-domain held-out corpus for the
# GPTQ-vs-RTN do-no-harm gate (the served-validated recipe — arm E in
# moe_expert_gptq_vs_rtn — beat same-corpus/in-sample gating). Must be DISJOINT
# from DATASET. Ignored for dense models (no packed experts). Set empty to
# reproduce the historical same-corpus in-sample gate.
: "${EXPERT_GATE_DATASET:=/home/rob/dq-runs/calibration/xdom-gate-v1.jsonl}"
: "${DEVICE:=cuda}"
: "${EXPORT_DEVICE:=cuda}"   # CUDA ~10× faster than CPU on NVFP4 packing
: "${TARGET_PROFILE:=vllm_packed_moe}"

# GGUF lane consistency gates (see docs/gguf_lane.md). The imatrix lockstep
# contract (measured cost == shipped bytes) only holds under COST_MODE=local
# today: production-render-score renders gguf formats via the unweighted
# registry qdq, and the production cache is never read by export_gguf.
if [[ "${EXPORT_CONTAINER:-compressed-tensors}" == "gguf" ]]; then
  if [[ "${COST_MODE:-production-render-score}" != "local" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=gguf requires COST_MODE=local — production-render-score scores UNWEIGHTED registry renders while the GGUF exporter ships imatrix-weighted bytes (rendering confound). Set COST_MODE=local." >&2
    exit 2
  fi
  if [[ "${PRODUCTION_CACHE:-1}" != "0" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=gguf requires PRODUCTION_CACHE=0 — export_gguf requantizes the bf16 skeleton and never reads the production cache; building one burns hours rendering bytes that never ship. Set PRODUCTION_CACHE=0 PRODUCTION_RECACHE=0." >&2
    exit 2
  fi
  if [[ "${TARGET_PROFILE:-vllm_packed_moe}" != "gguf" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=gguf requires TARGET_PROFILE=gguf (the exporter hard-fails on non-GGUF formats in the assignment)." >&2
    exit 2
  fi
fi

# NVFP4-CB / FP8-CB codebook lane consistency gates (docs/nvfp4-cb-plan/
# format-pipeline.md §6, LAYOUT.md). Same rendering-confound contract as the
# GGUF lane: the CB exporter (export_nvfp4_cb) ships imatrix-weighted VQ bytes
# and requantizes the bf16 skeleton, so the allocator cost MUST be the
# skeleton-requantize weighted render (COST_MODE=local), never
# production-render-score (which scores UNWEIGHTED registry renders and never
# reads the production cache the CB exporter also ignores).
if [[ "${EXPORT_CONTAINER:-compressed-tensors}" == "nvfp4_cb" ]]; then
  if [[ "${COST_MODE:-production-render-score}" != "local" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=nvfp4_cb requires COST_MODE=local — production-render-score scores UNWEIGHTED registry renders while the CB exporter ships imatrix-weighted VQ bytes (the identical rendering confound the GGUF gate exists for). Set COST_MODE=local." >&2
    exit 2
  fi
  if [[ "${TARGET_PROFILE:-vllm_packed_moe}" != "nvfp4_cb" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=nvfp4_cb requires TARGET_PROFILE=nvfp4_cb — the allocator must gate every candidate through the nvfp4_cb serving profile (allow_formats = CB rungs + NVFP4/FP8_DYNAMIC/BF16, in_features%256 shape rule) and the exporter hard-fails on non-CB formats in the assignment. Set TARGET_PROFILE=nvfp4_cb." >&2
    exit 2
  fi
  if [[ "${PRODUCTION_CACHE:-1}" != "0" || "${PRODUCTION_RECACHE:-1}" != "0" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=nvfp4_cb requires PRODUCTION_CACHE=0 PRODUCTION_RECACHE=0 — export_nvfp4_cb requantizes the bf16 skeleton and never reads the production cache; building one burns hours rendering bytes that never ship. Set PRODUCTION_CACHE=0 PRODUCTION_RECACHE=0." >&2
    exit 2
  fi
fi

if [[ "$DEVICE" != cuda* || "$EXPORT_DEVICE" != cuda* ]]; then
  echo "[pipeline] ERROR: PrismaQuant production pipeline is GPU-or-bust; DEVICE and EXPORT_DEVICE must be cuda*" >&2
  exit 2
fi
python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[pipeline] ERROR: CUDA is not available; refusing CPU quantization", file=sys.stderr)
    raise SystemExit(2)
PY
# Visual encoder format: fallback for visual Linears. See header docstring
# for the full text-only vs multimodal semantics. BF16 (default) is
# passthrough; non-BF16 registry formats uniformly quantize
# non-Fisher-allocated visual Linears via the existing RTN math at export time.
: "${VISUAL_FORMAT:=BF16}"
# Calibration modality. `text-only` (default) is the streaming body probe
# alone; `multimodal` adds a second non-streaming pass over the full
# model with image+text calibration so visual Linears get real Fisher
# stats. See header docstring.
: "${CALIBRATION_MODALITY:=text-only}"
# Multimodal dataset. `synthetic` is the offline stub that exercises the
# code path without network. Set to an HF dataset id (e.g.
# `HuggingFaceM4/COCO`) for real image calibration when
# CALIBRATION_MODALITY=multimodal.
: "${MM_DATASET:=synthetic}"
: "${MTP_FORMAT:=BF16}"
# Production-cache export path. Enabled by default so export packs the same
# rendered weights that KL/polish paths measure. Re-cache is enabled by default
# after the Qwen3.5-0.8B and Qwen3-4B smoke ladder cleared vLLM eager/graph
# serving; set PRODUCTION_RECACHE=0 for an explicit no-recache ablation.
: "${PRODUCTION_CACHE:=1}"
: "${PRODUCTION_RECACHE:=1}"
: "${PRODUCTION_CACHE_MAX_ACT_ROWS:=512}"
: "${PRODUCTION_CACHE_LRU_GB:=64.0}"
: "${PRODUCTION_CACHE_PREFETCH:=require}"
: "${PRODUCTION_CACHE_PREFETCH_WORKERS:=4}"
: "${PRODUCTION_RECACHE_MICROBATCH:=1}"
: "${PRODUCTION_CACHE_FORMATS:=auto}"
# assignment: render only concrete non-BF16 entries from layer_config.json.
# format-menu: render every requested format for every quantizable Linear,
# useful when intentionally building a reusable cache for reallocations.
: "${PRODUCTION_CACHE_RENDER_SCOPE:=assignment}"
: "${PRODUCTION_CACHE_LEVERS:=gptq,static_act_order,joint_scale_opt}"
: "${PRODUCTION_CACHE_DISABLE_LEVERS:=}"
# PRODUCTION_CACHE_UNION=1 switches the validated-surrogate frontier path
# from full format-menu rendering (every Linear × every quantized format)
# to a smart-union render that only adds FP8_DYNAMIC fallback entries for
# Linears whose NVFP4 output_mse exceeds the band threshold. It deliberately
# does not render MXFP8 or E5M2 fallbacks in the production path.
: "${PRODUCTION_CACHE_UNION:=0}"
: "${PRODUCTION_CACHE_UNION_P_FP8:=0.75}"
: "${COST_MODE:=production-render-score}"
: "${PRODUCTION_RENDER_COST_NSAMPLES:=8}"
: "${PRODUCTION_RENDER_COST_SEQLEN:=1024}"
: "${PRODUCTION_RENDER_COST_SEED:=42}"
# weight_mse default since 2026-07-02 (audit M6): the legacy
# h_trace*output_mse product carries the activation energy E||x||^2 twice
# (h_trace is the weight-space Fisher trace and already contains it), a
# per-Linear multiplicative bias ~ in_features*x_rms^2. Served two-arm A/B
# at matched 4.75 bpp, 5 window draws + 32k-token PPL per model:
#   Qwen3-4B:   KL -50.8% (31/40 windows), PPL 21.82 -> 18.52 (-15.1%)
#   Qwen3-0.6B: KL -58.5% (35/40 windows), PPL 45.20 -> 34.16 (-24.4%)
# output_mse reproduces the historical objective. Target-scale (27B-class)
# confirmation is ladder debt, mirroring the damp-1.0 promotion.
: "${PRODUCTION_RENDER_COST_SCORE_FIELD:=weight_mse}"
: "${PRODUCTION_RENDER_COST_REQUIRE_SCORES:=0}"
: "${PRODUCTION_RENDER_COST_REQUIRE_OUTPUT:=1}"
: "${PRODUCTION_RENDER_COST_PROMOTE_FRACTION:=0.30}"
: "${PRODUCTION_RENDER_COST_MIN_PROMOTE_SCORE:=}"
: "${PRODUCTION_RENDER_COST_MAX_PROMOTIONS:=}"
# Optional post-frontier MSE promotion rewrite. This is disabled by default
# and only promotes existing quantized assignments to a higher target format
# (BF16 by default), so it does not require additional production-cache
# renders. It is intended as an explicit recipe arm for testing whether local
# output-MSE can protect structurally sensitive regions without KL replay.
: "${MSE_PROMOTION:=0}"
: "${MSE_PROMOTION_CATEGORIES:=linear_attn,self_attn}"
: "${MSE_PROMOTION_TARGET_FORMAT:=BF16}"
: "${MSE_PROMOTION_MAX_BPP_DELTA:=}"
: "${MSE_PROMOTION_TARGET_BPP:=}"
: "${MSE_PROMOTION_GROUP_BY:=serving_unit}"
: "${MSE_PROMOTION_METRIC:=output_mse_per_bit}"
: "${ALLOC_PROPAGATED_SENSITIVITY_REPORT:=}"
: "${ALLOC_PROPAGATED_SENSITIVITY_SCALE:=1.0}"
: "${ALLOC_PROPAGATED_SENSITIVITY_SCORE_FIELD:=propagated_kl}"
: "${ALLOC_PROPAGATED_SENSITIVITY_FORMAT_EXTRAPOLATION:=local_mse_ratio}"
: "${ALLOC_PROPAGATED_SENSITIVITY_TARGET_FORMAT:=}"
: "${EXPORT_GPTQ:=auto}"
: "${EXPORT_SCALE_SWEEP:=auto}"
: "${PIPELINE_SPEC_PATH:=${WORK_DIR}/artifacts/pipeline_spec.json}"
: "${PIPELINE_SPEC_VALIDATE:=1}"
# Fisher-weighted GPTQ + Fisher output-MSE allocator are archived under
# archive/fisher_2026-05-15/ (the row-weighting + allocator-objective code
# paths remain in the live tree but are unreachable from the production
# pipeline). Any explicit override here fails fast.
: "${FISHER_WEIGHTED_GPTQ:=0}"
: "${FISHER_OUTPUT_MSE_ALLOCATOR:=0}"
for _legacy_fisher_var in FISHER_WEIGHTED_GPTQ FISHER_OUTPUT_MSE_ALLOCATOR; do
  case "${!_legacy_fisher_var}" in
    0|false|False|FALSE|no|No|NO|"") ;;
    *)
      echo "[pipeline] ERROR: $_legacy_fisher_var=${!_legacy_fisher_var} — Fisher levers are archived under archive/fisher_2026-05-15." >&2
      exit 2
      ;;
  esac
done
unset _legacy_fisher_var
case ",$PRODUCTION_CACHE_LEVERS," in
  *,fisher_gptq,*)
    echo "[pipeline] ERROR: PRODUCTION_CACHE_LEVERS includes fisher_gptq — Fisher levers are archived under archive/fisher_2026-05-15." >&2
    exit 2
    ;;
esac
export PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR=0
: "${SELECTION_MODE:=surrogate}"
: "${VALIDATED_FRONTIER_NSAMPLES:=$NSAMPLES}"
: "${VALIDATED_FRONTIER_SEQLEN:=$SEQLEN}"
# Held-out selection (review criticals C3/C5): the frontier-selecting KL
# must be measured on text disjoint from probe/cost/render calibration
# (house rule: "held-out split is disjoint from cost generation"). The
# loader is prefix-stable, so skipping the first NSAMPLES windows makes
# the validation windows token-disjoint by construction. Default ON —
# the prior in-sample wiring was a bug, not a behavior. Set
# VALIDATED_FRONTIER_SKIP_CALIB=0 only to reproduce historical runs.
: "${VALIDATED_FRONTIER_SKIP_CALIB:=$NSAMPLES}"
# Optional fully separate validation corpus (overrides skip-first
# disjointness with corpus-level disjointness when provided).
: "${VALIDATED_FRONTIER_DATASET:=$DATASET}"
# M26: final selection scores full-sequence KL (the gold-metric scope, §5),
# not the last_token triage screen. Selection is still re-validated on the
# served metric before ship, but aligning the selection scope with the gold
# metric removes a screen/gold mismatch. VALIDATED_FRONTIER_KL_SCOPE=last_token
# reproduces historical selections (the flagship 27B was picked under it).
: "${VALIDATED_FRONTIER_KL_SCOPE:=full_sequence}"
: "${VALIDATED_FRONTIER_PICK:=kneedle}"
: "${VALIDATED_FRONTIER_SAT_Z:=2.0}"
# Saturation (B*) needs a real per-bpp noise floor: validate_assignments_kl's
# kl_stderr is computed across calib-repeats, so a single repeat -> stderr=0 ->
# the saturation band collapses and B* degenerates to the asymptote. Auto-bump
# repeats to 4 when PICK=saturation (CLAUDE.md's --calib-repeats>=4 discipline);
# other picks keep 1 repeat (backwards-compatible). Override explicitly to pin.
if [[ "$VALIDATED_FRONTIER_PICK" == "saturation" ]]; then
  : "${VALIDATED_FRONTIER_CALIB_REPEATS:=4}"
else
  : "${VALIDATED_FRONTIER_CALIB_REPEATS:=1}"
fi
: "${VALIDATED_SOURCE_PREFETCH:=require}"
: "${VALIDATED_SOURCE_PREFETCH_MAX_GB:=0}"
: "${VALIDATED_SOURCE_PREFETCH_HEADROOM_GB:=16}"
: "${VALIDATED_SOURCE_PREFETCH_WORKERS:=2}"
: "${VALIDATED_FRONTIER_KL_CUDA_GRAPHS:=auto}"
: "${VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE:=0}"
# hooks materializes every assignment via forward hooks in one process
# (fast, but model + full render set must co-reside: OOMs on 35B-class MoE
# in the 128 GB unified pool). inplace loops one validate_assignments_kl
# invocation per Pareto point (weights installed in place, renders paged
# per point) and merges the per-point JSONs — the run_m4_validate_inplace
# pattern that fits 35B.
: "${VALIDATED_FRONTIER_MATERIALIZATION:=hooks}"
# COST_MODE=aura settings (defaults = the recipes that produced the regen-27b
# and 35B arm-E wins; see .claude/prismaquant-handover + memory notes).
: "${AURA_COST_NPROBES:=32}"
: "${AURA_COST_NSAMPLES:=8}"
: "${AURA_COST_SEQLEN:=128}"
: "${AURA_COST_CALIB_SEED:=42}"
: "${AURA_COST_LINEAR_CHUNKS:=8}"
: "${AURA_COST_PROBE_MICROBATCH:=8}"
: "${AURA_COST_MIN_FREE_GIB:=18}"
# auto: fp32 when the fp32-resident model fits with the min-free headroom
# (27B-class, the regen recipe), else bf16 (35B-class — fp32 is ~140 GiB
# against the 121 GiB pool and OOM-kills the box).
: "${AURA_COST_DTYPE:=auto}"
# Empirical packed-expert unit-KL stage (MoE hybrid): serving-unit RTN KL
# measured end-to-end, FP8 kept in the menu (real-KL rejects it, no bans).
: "${AURA_EXPERT_NSAMPLES:=16}"
: "${AURA_EXPERT_SEQLEN:=512}"

PROBE_PATH="${WORK_DIR}/artifacts/probe.pkl"
case "$COST_MODE" in
  local)
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    COST_PATH="${BASE_COST_PATH}"
    PRODUCTION_RENDER_COST_CACHE_PATH=""
    PRODUCTION_RENDER_COST_CACHE_DIR=""
    PRODUCTION_RENDER_COST_TAIL_QNAMES=""
    # One user knob drives BOTH ladder wirings (dense local cost + the
    # empirical expert stage): CB_LADDER_INTERP=1.
    if [[ "${CB_LADDER_INTERP:-0}" == "1" ]]; then
      export PRISMAQUANT_CB_LADDER_INTERP=1
    fi
    # CB M4-hybrid: stage [2d-CB] REPLACES every packed-expert row with the
    # empirical unit-KL (merge --replace-experts pops the local rows), so the
    # local stage's expert measurements — full-stack imatrix-weighted CB
    # encodes per rung, the dominant cost-stage wall on MoE (35B: ~1h45/shard)
    # — are discarded work. Skip them whenever the replacement is guaranteed.
    if [[ "${EXPORT_CONTAINER:-compressed-tensors}" == "nvfp4_cb" \
       && "${CB_EXPERT_EMPIRICAL:-1}" == "1" ]]; then
      export PRISMAQUANT_SKIP_PACKED_EXPERT_COST=1
      echo "[pipeline] [2/4] packed-expert local cost SKIPPED (CB hybrid replaces those rows)"
    fi
    ;;
  grouped-kl)
    echo "[pipeline] ERROR: COST_MODE=grouped-kl — the grouped-KL (fusion-matched) cost surrogate is archived under archive/grouped_kl_2026-05-28. It fixed a local allocator non-monotonicity but LOST the shipped vLLM A/B on Qwen3.6-27B (worse exact vLLM KL and direct WikiText PPL than the shipped 5.5 artifact); see archive/grouped_kl_2026-05-28/README.md. Use production-render-score (default), production-render-staged, or local." >&2
    exit 2
    ;;
  production-render-score|production-render)
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost_baseline.pkl"
    COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_render_score_cache.pkl"
    PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_render_score_weight_cache"
    PRODUCTION_RENDER_COST_TAIL_QNAMES=""
    ;;
  production-render-staged|production-render-tail)
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost_baseline.pkl"
    COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_render_score_staged_cache.pkl"
    PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_render_score_staged_weight_cache"
    PRODUCTION_RENDER_COST_TAIL_QNAMES="${WORK_DIR}/artifacts/production_render_score_tail_qnames.txt"
    ;;
  aura)
    # AURA downstream-KL-adjoint cost (aura_cost.py): predicted_dloss from
    # KL-Fisher probes x production-rendered dW. Served wins: -38% KL @4B,
    # -17.9% @27B vs the h_trace x output_mse baseline; regen-27b (#1
    # artifact) and the 35B arm-E hybrid both ran this recipe. On MoE
    # models the smooth cost is route-flip-blind for routed experts, so
    # packed experts get MEASURED empirical unit-KL costs instead
    # (prismaquant.expert_empirical_cost) merged into one hybrid payload.
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost_baseline.pkl"
    COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    AURA_COST_RAW="${WORK_DIR}/artifacts/cost_aura.pkl"
    if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
      # Principle #8: the cost's rendered dW, the frontier's measured KL,
      # and the exported bytes all come from ONE format-menu cache — build
      # the frontier cache early (identical settings to stage [4/4], which
      # then skip-if-exists) and point aura_cost at it. This is the
      # regen-27b prodcache_menu.pkl pattern, and it halves the ~60 GB
      # double-render a separate render-score cache would cost on 35B.
      PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_weight_cache_frontier_raw.pkl"
      PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_weight_cache_frontier"
    else
      PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_render_score_cache.pkl"
      PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_render_score_weight_cache"
    fi
    PRODUCTION_RENDER_COST_TAIL_QNAMES=""
    ;;
  *)
    echo "[pipeline] ERROR: COST_MODE must be local, production-render-score, production-render-staged, or aura" >&2
    exit 2
    ;;
esac

case "${HADAMARD_DUQUANT:-}" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *)
    echo "[pipeline] ERROR: Hadamard-DuQuant is archived under archive/hdq_2026-05-14 and is not available on the production path." >&2
    exit 2
    ;;
esac
case ",$PRODUCTION_CACHE_LEVERS," in
  *,hadamard_duquant,*)
    echo "[pipeline] ERROR: Hadamard-DuQuant is archived under archive/hdq_2026-05-14." >&2
    exit 2
    ;;
esac
case "${MULTI_SHOT_PASSES:-1}" in
  ""|1) ;;
  *)
    echo "[pipeline] ERROR: MULTI_SHOT_PASSES=${MULTI_SHOT_PASSES} — multi-shot recalibration is archived under archive/multi_shot_2026-05-19 (Qwen3-4B production-cal showed ΔKL=0 at 5/5 budgets; small-cal cross-eval showed mean -42.5% gap-closed with one budget regressing -154%). Not a production lever." >&2
    exit 2
    ;;
esac

mkdir -p "${WORK_DIR}"/{artifacts,act,work,logs,exported}

case "$SELECTION_MODE" in
  surrogate|validated-surrogate) ;;
  *)
    echo "[pipeline] ERROR: SELECTION_MODE must be surrogate or validated-surrogate" >&2
    exit 2
    ;;
esac
case "$MSE_PROMOTION" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *)
    if [[ "$SELECTION_MODE" != "validated-surrogate" ]]; then
      echo "[pipeline] ERROR: MSE_PROMOTION requires SELECTION_MODE=validated-surrogate; it is a post-frontier rewrite and would be inert under SELECTION_MODE=$SELECTION_MODE." >&2
      exit 2
    fi
    if [[ "$PRODUCTION_CACHE" == "0" || "$PRODUCTION_CACHE" == "false" || "$PRODUCTION_CACHE" == "False" ]]; then
      echo "[pipeline] ERROR: MSE_PROMOTION requires PRODUCTION_CACHE=1 so the selected assignment can be recached and exported from resident production weights." >&2
      exit 2
    fi
    ;;
esac

echo "[pipeline] config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  WORK_DIR=$WORK_DIR"
echo "  FORMATS=$FORMATS  TARGET_BITS=$TARGET_BITS"
echo "  NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN LAYERS_PER_SHARD=$LAYERS_PER_SHARD"
echo "  PREFETCH_LOOKAHEAD=$PREFETCH_LOOKAHEAD PREFETCH_WORKERS=$PREFETCH_WORKERS"
echo "  ACTIVATION_ROWS_LIMIT=$ACTIVATION_ROWS_LIMIT"
echo "  VISUAL_FORMAT=$VISUAL_FORMAT"
echo "  MTP_FORMAT=$MTP_FORMAT"
echo "  CALIBRATION_MODALITY=$CALIBRATION_MODALITY  MM_DATASET=$MM_DATASET"
echo "  PRODUCTION_CACHE=$PRODUCTION_CACHE PRODUCTION_RECACHE=$PRODUCTION_RECACHE"
echo "  PRODUCTION_CACHE_FORMATS=$PRODUCTION_CACHE_FORMATS"
echo "  PRODUCTION_CACHE_RENDER_SCOPE=$PRODUCTION_CACHE_RENDER_SCOPE"
echo "  PRODUCTION_CACHE_LEVERS=$PRODUCTION_CACHE_LEVERS"
echo "  PRODUCTION_CACHE_DISABLE_LEVERS=$PRODUCTION_CACHE_DISABLE_LEVERS"
echo "  COST_MODE=$COST_MODE"
if [[ "$COST_MODE" == "production-render-score" || "$COST_MODE" == "production-render" || "$COST_MODE" == "production-render-staged" || "$COST_MODE" == "production-render-tail" ]]; then
  echo "  PRODUCTION_RENDER_COST_NSAMPLES=$PRODUCTION_RENDER_COST_NSAMPLES PRODUCTION_RENDER_COST_SEQLEN=$PRODUCTION_RENDER_COST_SEQLEN PRODUCTION_RENDER_COST_SEED=$PRODUCTION_RENDER_COST_SEED SCORE_FIELD=$PRODUCTION_RENDER_COST_SCORE_FIELD"
fi
if [[ "$COST_MODE" == "aura" ]]; then
  echo "  AURA_COST_NPROBES=$AURA_COST_NPROBES AURA_COST_NSAMPLES=$AURA_COST_NSAMPLES AURA_COST_SEQLEN=$AURA_COST_SEQLEN AURA_COST_CALIB_SEED=$AURA_COST_CALIB_SEED AURA_COST_DTYPE=$AURA_COST_DTYPE"
  echo "  AURA_EXPERT_NSAMPLES=$AURA_EXPERT_NSAMPLES AURA_EXPERT_SEQLEN=$AURA_EXPERT_SEQLEN VALIDATED_FRONTIER_MATERIALIZATION=$VALIDATED_FRONTIER_MATERIALIZATION"
fi
echo "  EXPORT_GPTQ=$EXPORT_GPTQ EXPORT_SCALE_SWEEP=$EXPORT_SCALE_SWEEP"
echo "  PIPELINE_SPEC_PATH=$PIPELINE_SPEC_PATH"
echo "  PRISMAQUANT_NVFP4_SCALE_RULE=${PRISMAQUANT_NVFP4_SCALE_RULE:-static_6}"
echo "  PRODUCTION_CACHE_LRU_GB=$PRODUCTION_CACHE_LRU_GB PRODUCTION_CACHE_PREFETCH=$PRODUCTION_CACHE_PREFETCH"
echo "  VALIDATED_FRONTIER_SKIP_CALIB=$VALIDATED_FRONTIER_SKIP_CALIB VALIDATED_FRONTIER_DATASET=$VALIDATED_FRONTIER_DATASET"
echo "  SELECTION_MODE=$SELECTION_MODE VALIDATED_FRONTIER_NSAMPLES=$VALIDATED_FRONTIER_NSAMPLES VALIDATED_FRONTIER_SEQLEN=$VALIDATED_FRONTIER_SEQLEN VALIDATED_FRONTIER_PICK=$VALIDATED_FRONTIER_PICK"
echo

PIPELINE_SPEC_ARGS=(
  python3 -m prismaquant.pipeline
  --write-default-production "$PIPELINE_SPEC_PATH"
  --render-mechanisms "$PRODUCTION_CACHE_LEVERS"
  --disable-render-mechanisms "$PRODUCTION_CACHE_DISABLE_LEVERS"
  --model-path "$MODEL_PATH"
  --work-dir "$WORK_DIR"
  --formats "$FORMATS"
  --target-bits "$TARGET_BITS"
  --target-profile "$TARGET_PROFILE"
  --calibration-modality "$CALIBRATION_MODALITY"
  --selection-mode "$SELECTION_MODE"
  --production-cache "$PRODUCTION_CACHE"
  --production-recache "$PRODUCTION_RECACHE"
)
case "$PIPELINE_SPEC_VALIDATE" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *) PIPELINE_SPEC_ARGS+=(--validate) ;;
esac
"${PIPELINE_SPEC_ARGS[@]}"


# -----------------------------------------------------------------------
# Settings-hash guard (review critical C4): skip-if-exists stages must
# refuse to reuse artifacts built under different quality-affecting
# settings — silent reuse is the rendering-confound class that has
# invalidated A/Bs before. Each stage writes <artifact>.settings.json on
# build; on reuse a mismatch is a hard exit-2 naming the stale file.
# Legacy artifacts (no manifest) warn loudly but are not invalidated.
# -----------------------------------------------------------------------
require_stage_settings() {
  local artifact="$1" stage="$2"; shift 2
  python3 - "$artifact" "${artifact}.settings.json" "$stage" "$@" <<'GUARD'
import json, os, sys
artifact, manifest, stage, *pairs = sys.argv[1:]
cur = dict(p.split("=", 1) for p in pairs)
if os.path.exists(artifact):
    if not os.path.exists(manifest):
        print(f"[pipeline] WARNING: {stage}: reusing {artifact} which has no "
              "settings manifest (predates the settings-hash guard); cannot "
              "verify it matches the current settings", flush=True)
        sys.exit(0)
    prev = json.load(open(manifest))
    diffs = {k: (prev.get(k), cur.get(k))
             for k in sorted(set(prev) | set(cur))
             if prev.get(k) != cur.get(k)}
    if diffs:
        print(f"[pipeline] ERROR: {stage}: {artifact} was built under "
              "DIFFERENT settings; refusing silent reuse:", flush=True)
        for k, (a, b) in diffs.items():
            print(f"    {k}: artifact={a!r}  current={b!r}", flush=True)
        print(f"    -> delete {artifact} (and its .settings.json) to "
              "rebuild, or restore the original settings", flush=True)
        sys.exit(2)
    sys.exit(0)
os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
json.dump(cur, open(manifest, "w"), indent=1, sort_keys=True)
GUARD
  local rc=$?
  if [[ $rc -ne 0 ]]; then exit "$rc"; fi
}

RENDER_ENV_SETTINGS=(
  "PRISMAQUANT_NVFP4_SCALE_RULE=${PRISMAQUANT_NVFP4_SCALE_RULE:-}"
  # Manifest default must match the code default (gptq_damp_sweep_enabled:
  # OFF since 2026-06-12) so recorded provenance matches the render math.
  "PRISMAQUANT_GPTQ_DAMP_SWEEP=${PRISMAQUANT_GPTQ_DAMP_SWEEP:-0}"
  "PRISMAQUANT_GPTQ_DAMP=${PRISMAQUANT_GPTQ_DAMP:-}"
  "PRISMAQUANT_ACT_CLIP_QUANTILE=${PRISMAQUANT_ACT_CLIP_QUANTILE:-0.999}"
  "PRODUCTION_CACHE_LEVERS=$PRODUCTION_CACHE_LEVERS"
  "PRODUCTION_CACHE_DISABLE_LEVERS=${PRODUCTION_CACHE_DISABLE_LEVERS:-}"
)

# -----------------------------------------------------------------------
# 1. Sensitivity probe (per-Linear empirical Fisher diagonal trace,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
require_stage_settings "${PROBE_PATH}" probe \
  "MODEL_PATH=$MODEL_PATH" "DATASET=$DATASET" "NSAMPLES=$NSAMPLES" \
  "SEQLEN=$SEQLEN" "CALIBRATION_MODALITY=$CALIBRATION_MODALITY"
if [[ ! -f "${PROBE_PATH}" ]]; then
  echo "[pipeline] [1/4] running sensitivity probe ..."
  python3 -m prismaquant.incremental_probe \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
    --device "$DEVICE" --dtype bf16 \
    --output "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --work-dir "${WORK_DIR}/work" \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --prefetch-lookahead "$PREFETCH_LOOKAHEAD" \
    --prefetch-workers "$PREFETCH_WORKERS" \
    --prefetch-min-available-gb "$PREFETCH_MIN_AVAILABLE_GB" \
    --activation-rows-limit "$ACTIVATION_ROWS_LIMIT" \
    --calibration-modality "$CALIBRATION_MODALITY" \
    --mm-dataset "$MM_DATASET" \
    --mm-nsamples 8 --mm-max-text-len 128 \
    2>&1 | tee "${WORK_DIR}/logs/probe.log"
else
  # Reuse guard: make sure the pre-existing probe.pkl matches the
  # currently-requested calibration modality. Silently reusing a
  # text-only probe under multimodal (or vice versa) would produce
  # an assignment calibrated for the wrong activation distribution
  # — visible later as bad PPL that's hard to root-cause. Fail loud
  # and point the user at the file to delete.
  probe_modality=$(python3 -c "
import pickle, sys
try:
    with open(sys.argv[1], 'rb') as f:
        blob = pickle.load(f)
    meta = blob.get('meta', {}) if isinstance(blob, dict) else {}
    m = meta.get('calibration_modality') or meta.get('modality') or 'text-only'
    print(m)
except Exception as e:
    print(f'__error__:{e}', file=sys.stderr)
    sys.exit(2)
" "${PROBE_PATH}" 2>/dev/null || echo "__unknown__")
  if [[ "${probe_modality}" == "__unknown__" ]]; then
    echo "[pipeline] [1/4] probe.pkl exists but its calibration_modality"
    echo "             could not be read. Aborting to avoid mixing probes."
    echo "             Delete it explicitly to regenerate:"
    echo "               rm ${PROBE_PATH}"
    exit 2
  fi
  if [[ "${probe_modality}" != "${CALIBRATION_MODALITY}" ]]; then
    echo "[pipeline] [1/4] ABORT: probe.pkl was calibrated for"
    echo "             modality='${probe_modality}' but this run requests"
    echo "             CALIBRATION_MODALITY='${CALIBRATION_MODALITY}'."
    echo "             Reusing the probe would silently produce an"
    echo "             assignment calibrated on the wrong activations."
    echo ""
    echo "             Delete the stale probe to regenerate:"
    echo "               rm ${PROBE_PATH}"
    echo "             Or unset CALIBRATION_MODALITY to match the probe."
    exit 2
  fi
  echo "[pipeline] [1/4] probe.pkl exists (modality=${probe_modality}), skipping"
fi

# -----------------------------------------------------------------------
# 2. Cost measurement (per-(Linear, format) measured RTN error,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
require_stage_settings "${BASE_COST_PATH}" base-cost \
  "MODEL_PATH=$MODEL_PATH" "DATASET=$DATASET" "NSAMPLES=$NSAMPLES" \
  "SEQLEN=$SEQLEN" "FORMATS=$FORMATS"
if [[ ! -f "${BASE_COST_PATH}" ]]; then
  echo "[pipeline] [2/4] measuring per-(layer, format) cost ..."
  python3 -m prismaquant.incremental_measure_quant_cost \
    --model "$MODEL_PATH" \
    --probe "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --formats "$FORMATS" \
    --output "${BASE_COST_PATH}" \
    --work-dir "${WORK_DIR}/work" \
    --device "$DEVICE" --dtype bf16 \
    --mode batched --chunk-size 256 \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --skip-missing-activations \
    --no-include-lm-head \
    --swap-grow-limit-mb "${SWAP_GROW_LIMIT_MB:-2048}" \
    2>&1 | tee "${WORK_DIR}/logs/cost.log"
else
  echo "[pipeline] [2/4] baseline cost exists, skipping"
fi
# Fisher output-MSE allocator cost-status check removed; the archive guard
# at the top of this file errors out before reaching here if the var is set.

if [[ "$COST_MODE" == "production-render-score" || "$COST_MODE" == "production-render" ]]; then
  require_stage_settings "$PRODUCTION_RENDER_COST_CACHE_PATH" render-cost-cache \
    "MODEL_PATH=$MODEL_PATH" "DATASET=$DATASET" "FORMATS=$FORMATS" \
    "NS=$PRODUCTION_RENDER_COST_NSAMPLES" "SL=$PRODUCTION_RENDER_COST_SEQLEN" \
    "SEED=$PRODUCTION_RENDER_COST_SEED" "${RENDER_ENV_SETTINGS[@]}"
  if [[ ! -f "$PRODUCTION_RENDER_COST_CACHE_PATH" ]]; then
    echo "[pipeline] [2b/4] rendering production weights for allocator cost ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --formats "$FORMATS" \
      --render-scope format-menu \
      --n-calib-samples "$PRODUCTION_RENDER_COST_NSAMPLES" \
      --calib-seqlen "$PRODUCTION_RENDER_COST_SEQLEN" \
      --calib-seed "$PRODUCTION_RENDER_COST_SEED" \
      --dataset "$DATASET" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_score_cache.log"
  else
    echo "[pipeline] [2b/4] production-render cost cache exists, skipping"
  fi
  if [[ ! -f "$COST_PATH" ]]; then
    echo "[pipeline] [2c/4] synthesizing production-render allocator cost ..."
    PROD_RENDER_COST_ARGS=()
    case "$PRODUCTION_RENDER_COST_REQUIRE_SCORES" in
      1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        PROD_RENDER_COST_ARGS+=(--require-render-scores)
        ;;
    esac
    case "$PRODUCTION_RENDER_COST_REQUIRE_OUTPUT" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF|"") ;;
      *)
        PROD_RENDER_COST_ARGS+=(--require-output-metric)
        ;;
    esac
    python3 -m prismaquant.production_render_cost \
      --production-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --baseline-cost "$BASE_COST_PATH" \
      --output "$COST_PATH" \
      --formats "$FORMATS" \
      --score-field "$PRODUCTION_RENDER_COST_SCORE_FIELD" \
      "${PROD_RENDER_COST_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_cost.log"
  else
    echo "[pipeline] [2c/4] production-render allocator cost exists, skipping"
  fi
fi

if [[ "$COST_MODE" == "production-render-staged" || "$COST_MODE" == "production-render-tail" ]]; then
  PRODUCTION_RENDER_COST_NVFP4_CACHE="${WORK_DIR}/artifacts/production_render_score_staged_nvfp4_cache.pkl"
  PRODUCTION_RENDER_COST_TAIL_SUMMARY="${WORK_DIR}/artifacts/production_render_score_tail_summary.json"
  if [[ ! -f "$PRODUCTION_RENDER_COST_NVFP4_CACHE" ]]; then
    echo "[pipeline] [2b/4] rendering NVFP4 production weights for staged allocator cost ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_NVFP4_CACHE" \
      --formats NVFP4 \
      --render-scope format-menu \
      --n-calib-samples "$PRODUCTION_RENDER_COST_NSAMPLES" \
      --calib-seqlen "$PRODUCTION_RENDER_COST_SEQLEN" \
      --calib-seed "$PRODUCTION_RENDER_COST_SEED" \
      --dataset "$DATASET" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_score_nvfp4_cache.log"
  else
    echo "[pipeline] [2b/4] staged NVFP4 production-render cache exists, skipping"
  fi
  if [[ ! -f "$PRODUCTION_RENDER_COST_TAIL_QNAMES" ]]; then
    echo "[pipeline] [2c/4] selecting high-error NVFP4 tail for staged promotions ..."
    SELECT_TAIL_ARGS=()
    if [[ -n "$PRODUCTION_RENDER_COST_MIN_PROMOTE_SCORE" ]]; then
      SELECT_TAIL_ARGS+=(--select-tail-min-score "$PRODUCTION_RENDER_COST_MIN_PROMOTE_SCORE")
    fi
    if [[ -n "$PRODUCTION_RENDER_COST_MAX_PROMOTIONS" ]]; then
      SELECT_TAIL_ARGS+=(--select-tail-max-count "$PRODUCTION_RENDER_COST_MAX_PROMOTIONS")
    fi
    python3 -m prismaquant.production_render_cost \
      --production-cache "$PRODUCTION_RENDER_COST_NVFP4_CACHE" \
      --score-field "$PRODUCTION_RENDER_COST_SCORE_FIELD" \
      --select-tail-output "$PRODUCTION_RENDER_COST_TAIL_QNAMES" \
      --select-tail-summary "$PRODUCTION_RENDER_COST_TAIL_SUMMARY" \
      --select-tail-format NVFP4 \
      --select-tail-top-fraction "$PRODUCTION_RENDER_COST_PROMOTE_FRACTION" \
      "${SELECT_TAIL_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_tail.log"
  else
    echo "[pipeline] [2c/4] staged promotion tail exists, skipping"
  fi
  PRODUCTION_RENDER_COST_HIGH_FORMATS="$(python3 - "$FORMATS" <<'PY'
import sys
from prismaquant import format_registry as fr

seen = []
for raw in sys.argv[1].split(","):
    name = raw.strip()
    if not name:
        continue
    canon = fr.canonical_format_name(name)
    if canon not in {"BF16", "NVFP4"} and canon not in seen:
        seen.append(canon)
print(",".join(seen))
PY
)"
  if [[ -n "$PRODUCTION_RENDER_COST_HIGH_FORMATS" && ! -f "$PRODUCTION_RENDER_COST_CACHE_PATH" ]]; then
    echo "[pipeline] [2d/4] rendering staged promotion formats (${PRODUCTION_RENDER_COST_HIGH_FORMATS}) for high-error tail ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --formats "$PRODUCTION_RENDER_COST_HIGH_FORMATS" \
      --render-scope format-menu \
      --include-qnames-file "$PRODUCTION_RENDER_COST_TAIL_QNAMES" \
      --n-calib-samples "$PRODUCTION_RENDER_COST_NSAMPLES" \
      --calib-seqlen "$PRODUCTION_RENDER_COST_SEQLEN" \
      --calib-seed "$PRODUCTION_RENDER_COST_SEED" \
      --dataset "$DATASET" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_score_tail_cache.log"
  elif [[ -z "$PRODUCTION_RENDER_COST_HIGH_FORMATS" ]]; then
    PRODUCTION_RENDER_COST_CACHE_PATH="$PRODUCTION_RENDER_COST_NVFP4_CACHE"
    echo "[pipeline] [2d/4] no staged high formats requested"
  else
    echo "[pipeline] [2d/4] staged promotion-format cache exists, skipping"
  fi
  if [[ ! -f "$COST_PATH" ]]; then
    echo "[pipeline] [2e/4] synthesizing staged production-render allocator cost ..."
    PROD_RENDER_COST_ARGS=()
    case "$PRODUCTION_RENDER_COST_REQUIRE_OUTPUT" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF|"") ;;
      *)
        PROD_RENDER_COST_ARGS+=(--require-output-metric)
        ;;
    esac
    python3 -m prismaquant.production_render_cost \
      --production-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --baseline-cost "$BASE_COST_PATH" \
      --output "$COST_PATH" \
      --formats "$FORMATS" \
      --score-field "$PRODUCTION_RENDER_COST_SCORE_FIELD" \
      --missing-render-score-policy unavailable \
      --promotion-qnames-file "$PRODUCTION_RENDER_COST_TAIL_QNAMES" \
      --bf16-policy promotion-set \
      "${PROD_RENDER_COST_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_cost.log"
  else
    echo "[pipeline] [2e/4] staged production-render allocator cost exists, skipping"
  fi
fi

if [[ "$COST_MODE" == "aura" ]]; then
  # [2b] Production-faithful dW cache for the AURA adjoint. Under
  # SELECTION_MODE=validated-surrogate this IS the frontier cache (identical
  # path + settings to stage [4/4], which then skip-if-exists): ONE
  # format-menu cache supplies the cost's rendered dW, the frontier's
  # measured bytes, and the exported bytes (principle #8) — the regen-27b
  # prodcache_menu.pkl pattern.
  AURA_CACHE_FORMATS="$(python3 - "$FORMATS" <<'PY'
import sys
from prismaquant import format_registry as fr

seen = []
for raw in sys.argv[1].split(","):
    name = raw.strip()
    if not name:
        continue
    canon = fr.canonical_format_name(name)
    if canon != "BF16" and canon not in seen:
        seen.append(canon)
print(",".join(seen))
PY
)"
  if [[ -z "$AURA_CACHE_FORMATS" ]]; then
    echo "[pipeline] ERROR: COST_MODE=aura has no non-BF16 formats in FORMATS" >&2
    exit 2
  fi
  require_stage_settings "$PRODUCTION_RENDER_COST_CACHE_PATH" aura-dw-cache \
    "MODEL_PATH=$MODEL_PATH" "DATASET=$DATASET" "FORMATS=$AURA_CACHE_FORMATS" \
    "NS=$NSAMPLES" "SL=$SEQLEN" "SELECTION_MODE=$SELECTION_MODE" \
    "${RENDER_ENV_SETTINGS[@]}"
  if [[ ! -f "$PRODUCTION_RENDER_COST_CACHE_PATH" ]]; then
    echo "[pipeline] [2b/4] building format-menu production cache for AURA dW ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --formats "$AURA_CACHE_FORMATS" \
      --dataset "$DATASET" \
      --n-calib-samples "$NSAMPLES" \
      --calib-seqlen "$SEQLEN" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      --render-scope format-menu \
      $(if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then echo "--render-packed-experts"; fi) \
      2>&1 | tee "${WORK_DIR}/logs/aura_dw_cache.log"
  else
    echo "[pipeline] [2b/4] AURA dW production cache exists, skipping"
  fi

  # [2c] The AURA cost itself: KL-Fisher probes x production-rendered dW.
  # Packed-MoE experts are deliberately omitted here (the smooth adjoint is
  # route-flip-blind on them) and costed empirically in [2d].
  if [[ ! -f "$AURA_COST_RAW" ]]; then
    echo "[pipeline] [2c/4] measuring AURA downstream-KL-adjoint cost ..."
    python3 -m prismaquant.aura_cost \
      --model "$MODEL_PATH" \
      --output "$AURA_COST_RAW" \
      --formats "$FORMATS" \
      --production-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --require-production-cache \
      --n-probes "$AURA_COST_NPROBES" \
      --n-calib-samples "$AURA_COST_NSAMPLES" \
      --calib-seqlen "$AURA_COST_SEQLEN" \
      --calib-seed "$AURA_COST_CALIB_SEED" \
      --dtype "$AURA_COST_DTYPE" \
      --dataset "$DATASET" \
      --hook-harvest \
      --gradient-checkpointing \
      --n-linear-chunks "$AURA_COST_LINEAR_CHUNKS" \
      --probe-microbatch "$AURA_COST_PROBE_MICROBATCH" \
      --min-free-gib "$AURA_COST_MIN_FREE_GIB" \
      --accurate-chunk-bytes \
      --allow-packed-expert-omission \
      2>&1 | tee "${WORK_DIR}/logs/aura_cost.log"
  else
    echo "[pipeline] [2c/4] AURA cost exists, skipping"
  fi

  # [2d] Hybrid finalize: measured empirical unit-KL costs for any omitted
  # packed experts (FP8 kept in the menu — real-KL rejects it, no bans),
  # plus sidecar (MTP/visual) row backfill from the baseline cost. Backfilled
  # rows carry the baseline estimator and are recorded in provenance.
  if [[ ! -f "$COST_PATH" ]]; then
    OMITTED_EXPERTS="$(python3 - "$AURA_COST_RAW" <<'PY'
import pickle
import sys

payload = pickle.load(open(sys.argv[1], "rb"))
print(len(payload.get("provenance", {}).get("omitted_packed_experts", []) or []))
PY
)"
    if [[ "$OMITTED_EXPERTS" != "0" ]]; then
      echo "[pipeline] [2d/4] measuring empirical packed-expert unit-KL costs (${OMITTED_EXPERTS} omitted tensors; hybrid merge) ..."
      python3 -m prismaquant.expert_empirical_cost \
        --model "$MODEL_PATH" \
        --output "$COST_PATH" \
        --formats "$FORMATS" \
        --dataset "$DATASET" \
        --n-calib-samples "$AURA_EXPERT_NSAMPLES" \
        --calib-seqlen "$AURA_EXPERT_SEQLEN" \
        --merge-base "$AURA_COST_RAW" \
        --backfill-base "$BASE_COST_PATH" \
        2>&1 | tee "${WORK_DIR}/logs/expert_empirical_cost.log"
    else
      echo "[pipeline] [2d/4] no packed experts omitted; finalizing AURA cost (sidecar backfill) ..."
      python3 - "$AURA_COST_RAW" "$BASE_COST_PATH" "$COST_PATH" <<'PY'
import pickle
import sys

from prismaquant.expert_empirical_cost import backfill_missing_from_base

payload = pickle.load(open(sys.argv[1], "rb"))
payload.setdefault("stats", {})
payload.setdefault("costs", {})
base = pickle.load(open(sys.argv[2], "rb"))
added = backfill_missing_from_base(payload, base)
prov = dict(payload.get("provenance", {}) or {})
if added:
    prov["backfilled_from_base"] = added
    prov["backfill_base"] = sys.argv[2]
payload["provenance"] = prov
with open(sys.argv[3], "wb") as fh:
    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[pipeline] aura cost finalized -> {sys.argv[3]} "
      f"(backfilled {len(added)} sidecar rows: {added})")
PY
    fi
  else
    echo "[pipeline] [2d/4] AURA allocator cost exists, skipping"
  fi
fi

# [2d-CB] CB-lane hybrid: COST_MODE=local's weighted-recon cost is
# route-flip-blind on routed experts exactly like AURA's smooth cost
# (moe_cb_design.md §2) — REPLACE the expert-stack rows with measured
# empirical unit-KL, non-experts keep the local CB costs. No-op (beyond a
# model load) on dense models: zero packed-expert units -> the merged
# payload is the base unchanged. CB_EXPERT_EMPIRICAL=0 opts out.
if [[ "${EXPORT_CONTAINER:-compressed-tensors}" == "nvfp4_cb" \
   && "${CB_EXPERT_EMPIRICAL:-1}" == "1" ]]; then
  CB_LOCAL_RAW="${WORK_DIR}/artifacts/cost_local_raw.pkl"
  : "${CB_COL_WEIGHTS:=${WORK_DIR}/artifacts/cb_col_weights.pkl}"
  : "${CB_EXPERT_NSAMPLES:=16}"
  : "${CB_EXPERT_SEQLEN:=512}"
  CB_MERGED="$(python3 - "$COST_PATH" <<'PY'
import pickle, sys
try:
    payload = pickle.load(open(sys.argv[1], "rb"))
    print(1 if "expert_empirical_cost" in (
        payload.get("provenance", {}) or {}) else 0)
except Exception:
    print(0)
PY
)"
  if [[ "$CB_MERGED" != "1" ]]; then
    # The empirical pass renders CB candidates imatrix-WEIGHTED — harvest
    # the col-weights now (same skip-if-exists block as the [4/4] exporter;
    # measured cost and shipped bytes stay the one weighted render).
    if [[ ! -f "$CB_COL_WEIGHTS" ]]; then
      echo "[pipeline] [2d-CB] harvesting CB col-weights (imatrix) from ${WORK_DIR}/act ..."
      CB_ACT_DIR="${WORK_DIR}/act" CB_COL_WEIGHTS="$CB_COL_WEIGHTS" python3 - <<'PY'
import os, pickle
from prismaquant.export_gguf import build_imatrix_from_act_cache
act_dir = os.environ["CB_ACT_DIR"]
out = os.environ["CB_COL_WEIGHTS"]
cw = build_imatrix_from_act_cache(act_dir)
if not cw:
    raise SystemExit(
        f"[pipeline] ERROR: no activation cache under {act_dir!r}; the "
        f"empirical expert pass needs a col-weights vector per CB target.")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "wb") as fh:
    pickle.dump(cw, fh)
print(f"[pipeline] [2d-CB] wrote {out}: {len(cw)} Linears")
PY
    fi
    if [[ ! -f "$CB_LOCAL_RAW" ]]; then
      mv "$COST_PATH" "$CB_LOCAL_RAW"
    fi
    echo "[pipeline] [2d-CB] measuring empirical packed-expert unit-KL (CB hybrid; replace semantics) ..."
    CB_LADDER_ARGS=()
    if [[ "${CB_LADDER_INTERP:-0}" == "1" ]]; then
      CB_LADDER_ARGS+=(--cb-ladder-interp)
    fi
    # CB_EXPERT_SAMPLE=N: stratified expert subsample per unit (shared across
    # formats), unit KL extrapolated by expert count. 0 (default) = full
    # stacks. Cuts the empirical stage's encode volume ~E/N; export always
    # encodes every expert exactly.
    if [[ "${CB_EXPERT_SAMPLE:-0}" != "0" ]]; then
      CB_LADDER_ARGS+=(--expert-sample "$CB_EXPERT_SAMPLE")
    fi
    python3 -m prismaquant.expert_empirical_cost \
      --model "$MODEL_PATH" \
      --output "$COST_PATH" \
      --formats "$FORMATS" \
      --dataset "$DATASET" \
      --n-calib-samples "$CB_EXPERT_NSAMPLES" \
      --calib-seqlen "$CB_EXPERT_SEQLEN" \
      --merge-base "$CB_LOCAL_RAW" \
      --replace-experts \
      --col-weights "$CB_COL_WEIGHTS" \
      "${CB_LADDER_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/expert_empirical_cost_cb.log"
  else
    echo "[pipeline] [2d-CB] CB hybrid cost already merged, skipping"
  fi
fi

# -----------------------------------------------------------------------
# 3. Allocator (multi-choice knapsack over per-layer formats)
# -----------------------------------------------------------------------
echo "[pipeline] [3/4] running allocator at target=${TARGET_BITS} bpp ..."
# Choose visual-sensitivity mode from calibration modality:
#   text-only → uniform (Phase 1 --visual-format path, as before)
#   multimodal → fisher (Phase 2: DP places visual Linears from real
#                        multimodal Fisher; --visual-format acts as a
#                        fallback for un-probed visual Linears only)
if [[ "$CALIBRATION_MODALITY" == "multimodal" ]]; then
  VISUAL_SENSITIVITY=fisher
else
  VISUAL_SENSITIVITY=uniform
fi
ALLOCATOR_PARETO_ARGS=()
if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
  ALLOCATOR_PARETO_DIR="${WORK_DIR}/artifacts/pareto_assignments"
  ALLOCATOR_PARETO_ARGS=(--pareto-output-dir "$ALLOCATOR_PARETO_DIR")
else
  ALLOCATOR_PARETO_DIR=""
fi
ALLOCATOR_PROPAGATED_ARGS=()
if [[ -n "$ALLOC_PROPAGATED_SENSITIVITY_REPORT" ]]; then
  ALLOCATOR_PROPAGATED_ARGS+=(
    --propagated-sensitivity-report "$ALLOC_PROPAGATED_SENSITIVITY_REPORT"
    --propagated-sensitivity-scale "$ALLOC_PROPAGATED_SENSITIVITY_SCALE"
    --propagated-sensitivity-score-field "$ALLOC_PROPAGATED_SENSITIVITY_SCORE_FIELD"
    --propagated-sensitivity-format-extrapolation "$ALLOC_PROPAGATED_SENSITIVITY_FORMAT_EXTRAPOLATION"
  )
  if [[ -n "$ALLOC_PROPAGATED_SENSITIVITY_TARGET_FORMAT" ]]; then
    ALLOCATOR_PROPAGATED_ARGS+=(
      --propagated-sensitivity-target-format "$ALLOC_PROPAGATED_SENSITIVITY_TARGET_FORMAT"
    )
  fi
fi
python3 -m prismaquant.allocator \
  --probe "${PROBE_PATH}" \
  --costs "${COST_PATH}" \
  --target-bits "$TARGET_BITS" \
  --formats "$FORMATS" \
  --target-profile "$TARGET_PROFILE" \
  --pareto-targets "$PARETO_TARGETS" \
  --visual-format "$VISUAL_FORMAT" \
  --visual-sensitivity "$VISUAL_SENSITIVITY" \
  --mtp-format "$MTP_FORMAT" \
  --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
  --pareto-csv "${WORK_DIR}/artifacts/pareto.csv" \
  "${ALLOCATOR_PARETO_ARGS[@]}" \
  "${ALLOCATOR_PROPAGATED_ARGS[@]}" \
  2>&1 | tee "${WORK_DIR}/logs/allocator.log"

# -----------------------------------------------------------------------
# 4. Production cache + native compressed-tensors export
# -----------------------------------------------------------------------
PRODUCTION_CACHE_PATH=""
if [[ "$SELECTION_MODE" == "validated-surrogate" ]] && [[ "$PRODUCTION_CACHE" == "0" || "$PRODUCTION_CACHE" == "false" || "$PRODUCTION_CACHE" == "False" ]]; then
  echo "[pipeline] ERROR: SELECTION_MODE=validated-surrogate requires PRODUCTION_CACHE=1 so KL validates production-rendered weights." >&2
  exit 2
fi
if [[ "$PRODUCTION_CACHE" != "0" && "$PRODUCTION_CACHE" != "false" && "$PRODUCTION_CACHE" != "False" ]]; then
  PROD_CACHE_DIR="${WORK_DIR}/artifacts/production_weight_cache"
  PROD_CACHE_RAW="${WORK_DIR}/artifacts/production_weight_cache_raw.pkl"
  PROD_CACHE_RECACHED="${WORK_DIR}/artifacts/production_weight_cache_recached.pkl"
  CACHE_FORMATS="$PRODUCTION_CACHE_FORMATS"
  if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
    if [[ -z "$ALLOCATOR_PARETO_DIR" || ! -f "$ALLOCATOR_PARETO_DIR/manifest.json" ]]; then
      echo "[pipeline] ERROR: validated-surrogate selection requires allocator pareto assignments at $ALLOCATOR_PARETO_DIR" >&2
      exit 2
    fi
    if [[ "$CACHE_FORMATS" == "auto" ]]; then
      CACHE_FORMATS="$(python3 - "$FORMATS" <<'PY'
import sys
from prismaquant import format_registry as fr

seen = []
for raw in sys.argv[1].split(","):
    name = raw.strip()
    if not name:
        continue
    canon = fr.canonical_format_name(name)
    if canon != "BF16" and canon not in seen:
        seen.append(canon)
print(",".join(seen))
PY
)"
      echo "[pipeline] frontier production cache formats selected from FORMATS: ${CACHE_FORMATS:-none}"
    fi
    if [[ -z "$CACHE_FORMATS" ]]; then
      echo "[pipeline] ERROR: validated-surrogate selection has no non-BF16 cache formats" >&2
      exit 2
    fi
    PROD_CACHE_DIR="${PROD_CACHE_DIR}_frontier"
    PROD_CACHE_RAW="${WORK_DIR}/artifacts/production_weight_cache_frontier_raw.pkl"
    if [[ ! -f "$PROD_CACHE_RAW" ]]; then
      if [[ "${PRODUCTION_CACHE_UNION:-0}" == "1" ]]; then
        # Smart-union render: only render FP8_DYNAMIC fallbacks for Linears
        # whose NVFP4 output_mse is above the threshold percentile. Same
        # cache layout as format-menu, just fewer entries.
        echo "[pipeline] [4/4] building smart-union production cache for validated frontier ..."
        python3 -m tools.build_union_cache \
          --model "$MODEL_PATH" \
          --cost-pkl "$COST_PATH" \
          --input-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --output-cache-dir "$PROD_CACHE_DIR" \
          --output-pkl "$PROD_CACHE_RAW" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --levers "$PRODUCTION_CACHE_LEVERS" \
          --p-fp8 "${PRODUCTION_CACHE_UNION_P_FP8:-0.75}" \
          2>&1 | tee "${WORK_DIR}/logs/production_cache_frontier.log"
      else
        echo "[pipeline] [4/4] building format-menu production cache for validated frontier ..."
        python3 -m prismaquant.build_production_cache \
          --model "$MODEL_PATH" \
          --output "$PROD_CACHE_RAW" \
          --formats "$CACHE_FORMATS" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
          --enable "$PRODUCTION_CACHE_LEVERS" \
          --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
          --cache-dir "$PROD_CACHE_DIR" \
          --render-scope format-menu \
          --render-packed-experts \
          2>&1 | tee "${WORK_DIR}/logs/production_cache_frontier.log"
      fi
    else
      echo "[pipeline] [4/4] frontier production cache exists, skipping"
    fi

    VALIDATED_ASSIGNMENT_ARGS=()
    while IFS=$'\t' read -r label path; do
      [[ -n "$label" && -n "$path" ]] || continue
      VALIDATED_ASSIGNMENT_ARGS+=(--assignment "${label}=${path}")
    done < <(python3 - "$ALLOCATOR_PARETO_DIR/manifest.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
for row in payload.get("candidates", []):
    print(f"{row['label']}\t{row['path']}")
PY
)
    if [[ "${#VALIDATED_ASSIGNMENT_ARGS[@]}" -eq 0 ]]; then
      echo "[pipeline] ERROR: no Pareto assignments found for validated frontier" >&2
      exit 2
    fi

    VALIDATION_JSON="${WORK_DIR}/artifacts/validated_frontier_kl.json"
    VALIDATED_ASSIGNMENT_COUNT=$(( ${#VALIDATED_ASSIGNMENT_ARGS[@]} / 2 ))
    VAK_COMMON_ARGS=(
      --model "$MODEL_PATH"
      --probe "$PROBE_PATH"
      --costs "$COST_PATH"
      --base-assignment "${WORK_DIR}/artifacts/layer_config.json"
      --formats "$FORMATS"
      --dataset "$VALIDATED_FRONTIER_DATASET"
      --calib-skip-first "$(if [[ "$VALIDATED_FRONTIER_DATASET" == "$DATASET" ]]; then echo "$VALIDATED_FRONTIER_SKIP_CALIB"; else echo 0; fi)"
      --n-calib-samples "$VALIDATED_FRONTIER_NSAMPLES"
      --calib-seqlen "$VALIDATED_FRONTIER_SEQLEN"
      --calib-repeats "$VALIDATED_FRONTIER_CALIB_REPEATS"
      --work-dir "${WORK_DIR}/work/validate_kl"
      --dtype bf16
      --device "$DEVICE"
      --kl-scope "$VALIDATED_FRONTIER_KL_SCOPE"
      --kl-cuda-graphs "$VALIDATED_FRONTIER_KL_CUDA_GRAPHS"
      --source-prefetch "$VALIDATED_SOURCE_PREFETCH"
      --source-prefetch-max-gb "$VALIDATED_SOURCE_PREFETCH_MAX_GB"
      --source-prefetch-headroom-gb "$VALIDATED_SOURCE_PREFETCH_HEADROOM_GB"
      --source-prefetch-workers "$VALIDATED_SOURCE_PREFETCH_WORKERS"
      --production-weight-cache "$PROD_CACHE_RAW"
      --production-cache-dir-override "$PROD_CACHE_DIR"
      --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB"
      --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH"
      --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS"
    )
    if [[ "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "0" && "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "false" && "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "False" ]]; then
      VAK_COMMON_ARGS+=(--disable-frozen-weight-cache)
    fi
    if [[ "$VALIDATED_FRONTIER_MATERIALIZATION" == "inplace" ]]; then
      # inplace requires exactly one assignment per process (weights are
      # installed destructively); loop the Pareto points and merge the
      # per-point JSONs. This is the memory-fit path for 35B-class MoE —
      # hooks mode needs model + all renders co-resident and OOMs the
      # 128 GB unified pool.
      echo "[pipeline] [4/4] measuring real KL for ${VALIDATED_ASSIGNMENT_COUNT} Pareto assignments (inplace, one process per point) ..."
      VAK_PART_DIR="${WORK_DIR}/artifacts/validated_frontier_kl_parts"
      mkdir -p "$VAK_PART_DIR"
      VAK_PART_FILES=()
      for ((vi = 0; vi < ${#VALIDATED_ASSIGNMENT_ARGS[@]}; vi += 2)); do
        VAK_SPEC="${VALIDATED_ASSIGNMENT_ARGS[vi + 1]}"
        VAK_LABEL="${VAK_SPEC%%=*}"
        VAK_LABEL_SAFE="${VAK_LABEL//[^A-Za-z0-9._-]/_}"
        VAK_PART="${VAK_PART_DIR}/vak_${VAK_LABEL_SAFE}.json"
        VAK_PART_FILES+=("$VAK_PART")
        if [[ -f "$VAK_PART" ]]; then
          echo "[pipeline] [4/4] ${VAK_LABEL}: per-point KL exists, skipping"
          continue
        fi
        python3 -m prismaquant.validate_assignments_kl \
          "${VAK_COMMON_ARGS[@]}" \
          --assignment "$VAK_SPEC" \
          --assignment-materialization inplace \
          --output "$VAK_PART" \
          2>&1 | tee "${WORK_DIR}/logs/validated_frontier_kl_${VAK_LABEL_SAFE}.log"
      done
      python3 - "$VALIDATION_JSON" "${VAK_PART_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_path, *parts = sys.argv[1:]
merged = None
for part in parts:
    payload = json.loads(Path(part).read_text())
    if merged is None:
        merged = payload
    else:
        merged["results"].extend(payload.get("results", []))
if merged is None:
    raise SystemExit("[pipeline] ERROR: no per-point validation JSONs to merge")
merged["assignment_materialization"] = "inplace"
Path(out_path).write_text(json.dumps(merged, indent=2) + "\n")
print(f"[pipeline] merged {len(parts)} per-point KL JSONs -> {out_path} "
      f"({len(merged['results'])} results)")
PY
    else
      echo "[pipeline] [4/4] measuring real KL for ${VALIDATED_ASSIGNMENT_COUNT} Pareto assignments ..."
      python3 -m prismaquant.validate_assignments_kl \
        "${VAK_COMMON_ARGS[@]}" \
        "${VALIDATED_ASSIGNMENT_ARGS[@]}" \
        --assignment-materialization "$VALIDATED_FRONTIER_MATERIALIZATION" \
        --output "$VALIDATION_JSON" \
        2>&1 | tee "${WORK_DIR}/logs/validated_frontier_kl.log"
    fi

    echo "[pipeline] [4/4] selecting measured frontier point ..."
    python3 -m prismaquant.select_validated_frontier \
      --validation-json "$VALIDATION_JSON" \
      --mode "$VALIDATED_FRONTIER_PICK" \
      --sat-z "$VALIDATED_FRONTIER_SAT_Z" \
      --output-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
      --output-assignment "${WORK_DIR}/artifacts/layer_config_validated_assignment.json" \
      --output-summary "${WORK_DIR}/artifacts/validated_frontier_selection.json" \
      2>&1 | tee "${WORK_DIR}/logs/validated_frontier_select.log"

    if [[ "$MSE_PROMOTION" != "0" && "$MSE_PROMOTION" != "false" && "$MSE_PROMOTION" != "False" ]]; then
      MSE_PROMOTION_ARGS=()
      if [[ -n "$MSE_PROMOTION_TARGET_BPP" ]]; then
        MSE_PROMOTION_ARGS+=(--target-bpp "$MSE_PROMOTION_TARGET_BPP")
      elif [[ -n "$MSE_PROMOTION_MAX_BPP_DELTA" ]]; then
        MSE_PROMOTION_ARGS+=(--max-bpp-delta "$MSE_PROMOTION_MAX_BPP_DELTA")
      else
        echo "[pipeline] ERROR: MSE_PROMOTION requires MSE_PROMOTION_TARGET_BPP or MSE_PROMOTION_MAX_BPP_DELTA" >&2
        exit 2
      fi
      echo "[pipeline] applying MSE-driven promotion rewrite ..."
      python3 tools/build_mse_promotion_assignment.py \
        --base-assignment "${WORK_DIR}/artifacts/layer_config.json" \
        --model "$MODEL_PATH" \
        --costs "$COST_PATH" \
        --probe "$PROBE_PATH" \
        --output-layer-config "${WORK_DIR}/artifacts/layer_config_mse_promoted.json" \
        --output-report "${WORK_DIR}/artifacts/mse_promotion_report.json" \
        --categories "$MSE_PROMOTION_CATEGORIES" \
        --target-format "$MSE_PROMOTION_TARGET_FORMAT" \
        --group-by "$MSE_PROMOTION_GROUP_BY" \
        --metric "$MSE_PROMOTION_METRIC" \
        "${MSE_PROMOTION_ARGS[@]}" \
        2>&1 | tee "${WORK_DIR}/logs/mse_promotion.log"
      cp "${WORK_DIR}/artifacts/layer_config.json" \
         "${WORK_DIR}/artifacts/layer_config_before_mse_promotion.json"
      mv "${WORK_DIR}/artifacts/layer_config_mse_promoted.json" \
         "${WORK_DIR}/artifacts/layer_config.json"
    fi

    if [[ "$PRODUCTION_RECACHE" != "0" && "$PRODUCTION_RECACHE" != "false" && "$PRODUCTION_RECACHE" != "False" ]]; then
      SELECTED_DIGEST="$(python3 - "${WORK_DIR}/artifacts/layer_config.json" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()[:12])
PY
)"
      PROD_CACHE_RECACHED="${WORK_DIR}/artifacts/production_weight_cache_frontier_${SELECTED_DIGEST}_recached.pkl"
      if [[ ! -f "$PROD_CACHE_RECACHED" ]]; then
        echo "[pipeline] [4/4] re-fitting activation scales for selected measured-${VALIDATED_FRONTIER_PICK} assignment ..."
        python3 -m prismaquant.production_recache \
          --model "$MODEL_PATH" \
          --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --production-weight-cache "$PROD_CACHE_RAW" \
          --output "$PROD_CACHE_RECACHED" \
          --cache-dir-override "$PROD_CACHE_DIR" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --device "$DEVICE" \
          --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB" \
          --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH" \
          --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS" \
          --microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          2>&1 | tee "${WORK_DIR}/logs/production_recache.log"
      else
        echo "[pipeline] [4/4] selected-assignment recached production cache exists, skipping"
      fi
      PRODUCTION_CACHE_PATH="$PROD_CACHE_RECACHED"
    else
      PRODUCTION_CACHE_PATH="$PROD_CACHE_RAW"
    fi
  elif [[ "$CACHE_FORMATS" == "auto" ]]; then
    CACHE_FORMATS="$(python3 - "${WORK_DIR}/artifacts/layer_config.json" <<'PY'
import sys
from prismaquant.production_recache import _load_assignment

assignment = _load_assignment(sys.argv[1])
formats = sorted({fmt for fmt in assignment.values() if fmt.upper() != "BF16"})
print(",".join(formats))
PY
)"
    echo "[pipeline] production cache formats selected from assignment: ${CACHE_FORMATS:-none}"
  fi
  if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
    :
  elif [[ -z "$CACHE_FORMATS" ]]; then
    echo "[pipeline] production cache requested but no non-BF16 formats are in FORMATS; skipping cache"
  elif [[ "$PRODUCTION_RECACHE" != "0" && "$PRODUCTION_RECACHE" != "false" && "$PRODUCTION_RECACHE" != "False" ]]; then
    LC_DIGEST=$(sha256sum "${WORK_DIR}/artifacts/layer_config.json" | cut -c1-16)
    require_stage_settings "$PROD_CACHE_RECACHED" production-cache-recached \
      "MODEL_PATH=$MODEL_PATH" "DATASET=$DATASET" "NSAMPLES=$NSAMPLES" \
      "SEQLEN=$SEQLEN" "FORMATS=$FORMATS" "TARGET_BITS=$TARGET_BITS" \
      "ASSIGNMENT_DIGEST=$LC_DIGEST" "${RENDER_ENV_SETTINGS[@]}"
    if [[ ! -f "$PROD_CACHE_RECACHED" ]]; then
      if [[ ! -f "$PROD_CACHE_RAW" ]]; then
        echo "[pipeline] [4/4] building production cache + re-fitting activation scales ..."
        python3 -m prismaquant.build_production_cache \
          --model "$MODEL_PATH" \
          --output "$PROD_CACHE_RECACHED" \
          --formats "$CACHE_FORMATS" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
          --enable "$PRODUCTION_CACHE_LEVERS" \
          --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
          --cache-dir "$PROD_CACHE_DIR" \
          --render-scope "$PRODUCTION_CACHE_RENDER_SCOPE" \
          --render-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --recache-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --recache-microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          ${EXPERT_GATE_DATASET:+--expert-gate-dataset "$EXPERT_GATE_DATASET"} \
          2>&1 | tee "${WORK_DIR}/logs/production_cache.log"
      else
        echo "[pipeline] [4/4] re-fitting production activation scales ..."
        python3 -m prismaquant.production_recache \
          --model "$MODEL_PATH" \
          --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --production-weight-cache "$PROD_CACHE_RAW" \
          --output "$PROD_CACHE_RECACHED" \
          --cache-dir-override "$PROD_CACHE_DIR" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --device "$DEVICE" \
          --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB" \
          --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH" \
          --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS" \
          --microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          2>&1 | tee "${WORK_DIR}/logs/production_recache.log"
      fi
    else
      echo "[pipeline] [4/4] recached production cache exists, skipping"
    fi
    PRODUCTION_CACHE_PATH="$PROD_CACHE_RECACHED"
  else
    if [[ ! -f "$PROD_CACHE_RAW" ]]; then
      echo "[pipeline] [4/4] building production cache ..."
      python3 -m prismaquant.build_production_cache \
        --model "$MODEL_PATH" \
        --output "$PROD_CACHE_RAW" \
        --formats "$CACHE_FORMATS" \
        --dataset "$DATASET" \
        --n-calib-samples "$NSAMPLES" \
        --calib-seqlen "$SEQLEN" \
        --dtype bf16 \
        --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
        --enable "$PRODUCTION_CACHE_LEVERS" \
        --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
        --cache-dir "$PROD_CACHE_DIR" \
        --render-scope "$PRODUCTION_CACHE_RENDER_SCOPE" \
        --render-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
        ${EXPERT_GATE_DATASET:+--expert-gate-dataset "$EXPERT_GATE_DATASET"} \
        2>&1 | tee "${WORK_DIR}/logs/production_cache.log"
    else
      echo "[pipeline] [4/4] production cache exists, skipping"
    fi
    PRODUCTION_CACHE_PATH="$PROD_CACHE_RAW"
  fi
fi

if [[ "${EXPORT_CONTAINER:-compressed-tensors}" == "gguf" ]]; then
  # GGUF lane: one artifact serves llama.cpp natively and vLLM via the
  # GGUF path. Requires TARGET_PROFILE=gguf at allocation time (the
  # exporter hard-fails on non-GGUF formats). The skeleton (metadata +
  # tokenizer) comes from llama.cpp's own converter; we requantize bytes.
  : "${LLAMA_CPP_DIR:=/home/rob/dq-runs/llama.cpp}"
  : "${GGUF_SKELETON:=${WORK_DIR}/artifacts/skeleton.gguf}"
  : "${GGUF_TOKEN_EMBEDDING_FORMAT:=}"
  : "${GGUF_OUTPUT_FORMAT:=}"
  require_stage_settings "$GGUF_SKELETON" gguf-skeleton \
    "MODEL_PATH=$MODEL_PATH"
  if [[ ! -f "$GGUF_SKELETON" ]]; then
    echo "[pipeline] [4/4] building GGUF skeleton (convert_hf_to_gguf, bf16) ..."
    # Write-then-rename: convert_hf_to_gguf writes in place, so a crashed
    # conversion must not leave a truncated file the skip-gate trusts.
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_PATH" \
      --outtype bf16 --outfile "${GGUF_SKELETON}.tmp" \
      2>&1 | tee "${WORK_DIR}/logs/gguf_skeleton.log"
    mv "${GGUF_SKELETON}.tmp" "$GGUF_SKELETON"
  else
    echo "[pipeline] [4/4] GGUF skeleton exists, skipping"
  fi
  echo "[pipeline] [4/4] exporting to GGUF ..."
  GGUF_EXPORT_ARGS=(
    python3 -m prismaquant.export_gguf
    --skeleton "$GGUF_SKELETON"
    --layer-config "${WORK_DIR}/artifacts/layer_config.json"
    --out "${WORK_DIR}/exported.gguf"
    --device "$EXPORT_DEVICE"
  )
  # Keep export-side imatrix in lockstep with the cost measurement
  # (PRISMAQUANT_GGUF_IMATRIX, default on): measured cost and shipped
  # bytes must be the same rendering. The truthiness parse MUST match
  # measure_quant_cost._gguf_imatrix_enabled (set-but-empty = default on;
  # 0/false/no/off in any case = off), or the one flag whose job is
  # lockstep silently splits the two sides.
  _gguf_imatrix="${PRISMAQUANT_GGUF_IMATRIX:-1}"
  _gguf_imatrix="$(echo "$_gguf_imatrix" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  [[ -z "$_gguf_imatrix" ]] && _gguf_imatrix=1
  case "$_gguf_imatrix" in
    0|false|no|off) ;;
    *) GGUF_EXPORT_ARGS+=(--imatrix-from-act-cache "${WORK_DIR}/act") ;;
  esac
  [[ -n "$GGUF_TOKEN_EMBEDDING_FORMAT" ]] && \
    GGUF_EXPORT_ARGS+=(--token-embedding-format "$GGUF_TOKEN_EMBEDDING_FORMAT")
  [[ -n "$GGUF_OUTPUT_FORMAT" ]] && \
    GGUF_EXPORT_ARGS+=(--output-format "$GGUF_OUTPUT_FORMAT")
  "${GGUF_EXPORT_ARGS[@]}" 2>&1 | tee "${WORK_DIR}/logs/export.log"

  # Load + greedy-generate smoke on the actual serving runtime (the
  # validate_native_export analog for this lane). A PPL/p99-NLL ship gate
  # (validate_quantized_model analog) is still open work — see
  # docs/gguf_lane.md; this gate only proves the artifact loads and
  # produces tokens.
  LLAMA_COMPLETION="$LLAMA_CPP_DIR/build/bin/llama-completion"
  if [[ -x "$LLAMA_COMPLETION" ]]; then
    echo "[pipeline] [4/4] llama.cpp load+generate smoke ..."
    SMOKE_OUT=$("$LLAMA_COMPLETION" -m "${WORK_DIR}/exported.gguf" \
      -p "The quick brown fox" -n 16 --temp 0 -ngl 99 --no-display-prompt \
      < /dev/null 2>"${WORK_DIR}/logs/gguf_smoke.err") || {
      echo "[pipeline] ERROR: llama.cpp failed to load/generate from the exported artifact; see ${WORK_DIR}/logs/gguf_smoke.err" >&2
      exit 1
    }
    if [[ -z "${SMOKE_OUT//[[:space:]]/}" ]]; then
      echo "[pipeline] ERROR: llama.cpp generated no tokens from the exported artifact" >&2
      exit 1
    fi
    echo "[pipeline] [4/4] smoke output: ${SMOKE_OUT:0:120}"
  else
    echo "[pipeline] WARNING: $LLAMA_COMPLETION not built — skipping load+generate smoke (build with: cmake --build $LLAMA_CPP_DIR/build --target llama-completion)"
  fi

  echo
  echo "[pipeline] done."
  echo "  Artifact: ${WORK_DIR}/exported.gguf"
  echo "  Serve (llama.cpp): $LLAMA_CPP_DIR/build/bin/llama-server -m ${WORK_DIR}/exported.gguf -ngl 99"
  echo "  KL harness:        llama-perplexity --kl-divergence-base <base_logits> --kl-divergence"
  exit 0
fi

if [[ "${EXPORT_CONTAINER:-compressed-tensors}" == "nvfp4_cb" ]]; then
  # NVFP4-CB / FP8-CB codebook lane: one custom compressed-tensors-STYLE
  # artifact served by the out-of-tree vllm_prismaquant_plugin (LAYOUT.md is
  # the byte contract). Like the GGUF lane, the exporter requantizes the bf16
  # skeleton with an imatrix-weighted VQ search — it does NOT read a production
  # cache. Requires TARGET_PROFILE=nvfp4_cb + COST_MODE=local (gated above).
  : "${CB_OUT:=${WORK_DIR}/exported_nvfp4_cb}"
  : "${CB_COL_WEIGHTS:=${WORK_DIR}/artifacts/cb_col_weights.pkl}"
  # Codebook source: `lattice` (deterministic fixed FP4/FP8 lattice, no
  # sidecar) or `learned` (shared per-(role) codebooks trained at export
  # time, sidecar amortized ~0 bpw — the byte-competitive champion in
  # Phase 0; per-tensor learned is footprint-prohibitive, never used).
  : "${CB_CODEBOOK_SOURCE:=lattice}"
  : "${CB_CODEBOOK_ITERS:=4}"
  : "${CB_CODEBOOK_SEED:=0}"
  # Joint E4M3-legal scale sweep is default-ON (IQ-rendering parity; the
  # exp-1 CB-vs-IQ confound was a scale-rendering artifact). CB_SCALE_SWEEP=0
  # is the one-shot amax/grid-max A/B ablation only.
  : "${CB_SCALE_SWEEP:=1}"
  # fp4 scale coding: `v1` (bare E4M3 plane, default/served) or `two_tier`
  # (layout-v2 E8M0-super + 4-bit-sub; serve gates pending — do NOT ship).
  : "${CB_SCALE_CODING:=v1}"

  # Col-weights (per-input-column imatrix) — the CB exporter's weighted-VQ
  # importance, REQUIRED for every CB target (no silent RTN). Harvested from
  # the SAME activation cache the local cost stage built, exactly as the GGUF
  # lane's --imatrix-from-act-cache: measured cost and shipped bytes are the
  # one weighted render (the one-cache / lockstep contract, format-pipeline
  # §6). Skip-if-exists; override CB_COL_WEIGHTS to supply a pre-built flat
  # {qname: (in_features,) tensor} pickle (e.g. Fisher col-weights, exp-4).
  if [[ ! -f "$CB_COL_WEIGHTS" ]]; then
    echo "[pipeline] [4/4] harvesting CB col-weights (imatrix) from ${WORK_DIR}/act ..."
    CB_ACT_DIR="${WORK_DIR}/act" CB_COL_WEIGHTS="$CB_COL_WEIGHTS" python3 - <<'PY'
import os, pickle
from prismaquant.export_gguf import build_imatrix_from_act_cache
act_dir = os.environ["CB_ACT_DIR"]
out = os.environ["CB_COL_WEIGHTS"]
cw = build_imatrix_from_act_cache(act_dir)
if not cw:
    raise SystemExit(
        f"[pipeline] ERROR: no activation cache under {act_dir!r}; the CB "
        f"exporter needs a col-weights (imatrix) vector per CB target. Run "
        f"the probe+cost stages first (they populate {act_dir}).")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "wb") as fh:
    pickle.dump(cw, fh)
print(f"[pipeline] [4/4] wrote {out}: {len(cw)} Linears")
PY
  else
    echo "[pipeline] [4/4] CB col-weights exist, skipping"
  fi

  # Streaming exporter for 200-300B-class sources (Hy3 ~557GB, DSv4 ~295GB):
  # export_nvfp4_cb loads EVERY shard resident + accumulates EVERY output tensor
  # -> OOM on the 121GB box (dsv4_readiness.md gap 2). export_nvfp4_cb_streaming
  # holds ~one source tensor (one MoE layer's expert stack) + the codebooks.
  # EXPORT_STREAMING: auto (default; stream when the source exceeds
  # EXPORT_STREAMING_THRESHOLD_GB, default 80) / 1 / 0. Same CLI both ways.
  : "${EXPORT_STREAMING:=auto}"
  : "${EXPORT_STREAMING_THRESHOLD_GB:=80}"
  CB_EXPORT_MODULE="prismaquant.export_nvfp4_cb"
  _src_gb="$(du -sBG "$MODEL_PATH" 2>/dev/null | awk '{gsub(/G/,"",$1); print int($1)}')"
  case "$EXPORT_STREAMING" in
    1|true|True|TRUE|yes|Yes|YES) CB_EXPORT_MODULE="prismaquant.export_nvfp4_cb_streaming" ;;
    0|false|False|FALSE|no|No|NO) ;;
    auto|"")
      if [[ -n "$_src_gb" && "$_src_gb" -ge "$EXPORT_STREAMING_THRESHOLD_GB" ]]; then
        CB_EXPORT_MODULE="prismaquant.export_nvfp4_cb_streaming"
        echo "[pipeline] [4/4] source ~${_src_gb}GB >= ${EXPORT_STREAMING_THRESHOLD_GB}GB -> streaming exporter"
      fi
      ;;
    *)
      echo "[pipeline] ERROR: EXPORT_STREAMING must be auto/1/0" >&2
      exit 2
      ;;
  esac
  echo "[pipeline] [4/4] exporting to nvfp4_cb (codebook container; ${CB_EXPORT_MODULE}) ..."
  CB_EXPORT_ARGS=(
    python3 -m "$CB_EXPORT_MODULE"
    --model-dir "$MODEL_PATH"
    --layer-config "${WORK_DIR}/artifacts/layer_config.json"
    --out "$CB_OUT"
    --col-weights "$CB_COL_WEIGHTS"
    --codebook-source "$CB_CODEBOOK_SOURCE"
    --codebook-iters "$CB_CODEBOOK_ITERS"
    --codebook-seed "$CB_CODEBOOK_SEED"
    --scale-coding "$CB_SCALE_CODING"
    --device "$EXPORT_DEVICE"
  )
  case "$CB_SCALE_SWEEP" in
    0|false|False|FALSE|no|No|NO) CB_EXPORT_ARGS+=(--no-scale-sweep) ;;
    1|true|True|TRUE|yes|Yes|YES|auto|"") ;;
    *)
      echo "[pipeline] ERROR: CB_SCALE_SWEEP must be 0 or 1" >&2
      exit 2
      ;;
  esac
  "${CB_EXPORT_ARGS[@]}" 2>&1 | tee "${WORK_DIR}/logs/export.log"

  # TODO(Milestone C / production-cache): the fuller one-cache integration —
  # teach production-render-score to pass col_weights to the CB render so the
  # standard ProductionWeightCache identity holds (cost==KL==bytes through the
  # cache, like NVFP4/FP8) and the COST_MODE=local restriction can lift — is
  # DEFERRED per format-pipeline.md §6 ("Alternative (larger, better) path —
  # flag not adopt"). Phase 0 mirrors the GGUF skeleton-requantize model.
  # There is also NO in-lane serving smoke: CB artifacts serve ONLY via the
  # out-of-tree vllm_prismaquant_plugin (plugins/vllm_prismaquant/), so the
  # load+generate / served-KL gate runs in the plugin's serving env, not here
  # (docs/nvfp4-cb-plan/serve_prototype_0p6b.md). No rung is production-eligible
  # until it clears the served gold-metric KL/PPL gate AND the prefill perf
  # gate (INV-2, no Triton masquerade).
  echo
  echo "[pipeline] done."
  echo "  Artifact: ${CB_OUT}  (custom quant_method=prismaquant; LAYOUT.md contract)"
  echo "  Serve:    vLLM with the vllm_prismaquant_plugin installed"
  echo "            (plugins/vllm_prismaquant/); NOT stock compressed-tensors."
  echo "  Gate:     served KL-vs-BF16 + WikiText PPL in the plugin serving env."
  exit 0
fi

echo "[pipeline] [4/4] exporting to compressed-tensors ..."
EXPORT_ARGS=(
  python3 -m prismaquant.export_native_compressed
  --model "$MODEL_PATH"
  --layer-config "${WORK_DIR}/artifacts/layer_config.json"
  --output "${WORK_DIR}/exported"
  --device "$EXPORT_DEVICE"
  --activation-cache-dir "${WORK_DIR}/act"
)
case "$EXPORT_GPTQ" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-gptq) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--gptq) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_GPTQ must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
case "$EXPORT_SCALE_SWEEP" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-scale-sweep) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--scale-sweep) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_SCALE_SWEEP must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
if [[ -n "$PRODUCTION_CACHE_PATH" ]]; then
	  EXPORT_ARGS+=(
	    --production-weight-cache "$PRODUCTION_CACHE_PATH"
	    --production-cache-dir-override "$PROD_CACHE_DIR"
	    --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB"
	    --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS"
	  )
fi
"${EXPORT_ARGS[@]}" 2>&1 | tee "${WORK_DIR}/logs/export.log"

echo
echo "[pipeline] done."
echo "  Artifact: ${WORK_DIR}/exported"
echo "  Validate: python3 -m prismaquant.validate_native_export \\"
echo "              --model ${WORK_DIR}/exported"
echo "  Serve:    vllm serve ${WORK_DIR}/exported \\"
echo "              --quantization compressed-tensors --trust-remote-code"
