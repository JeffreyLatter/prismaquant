#!/usr/bin/env bash
# Proto-production PrismaSCOUT path:
#   1. Build or reuse a strict production weight cache.
#   2. Generate or reuse allocator Pareto seed assignments.
#   3. Remeasure the seed frontier with the production KL oracle.
#   4. Refine around the measured knee with centered output-Fisher Block-CLADO.
#   5. Emit a manifest and final assignment.
#
# This launcher is intentionally local-venv based. It is meant to be run from a
# PrismaQuant checkout on the quantization host, not inside a separate container.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/Qwen3-4B}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}" | tr '/:.' '---')}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/${MODEL_TAG}-prismascout-proto-production-${TS}}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
FLOOR_FORMAT="${FLOOR_FORMAT:-NVFP4}"
PIN_ARGS="${PIN_ARGS:---pin lm_head}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-kernels-community/flash-attn2}"

N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-64}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
KL_SCOPE="${KL_SCOPE:-last_token}"
MAX_LANES_PER_BATCH="${MAX_LANES_PER_BATCH:-4}"
CALIB_MICROBATCH="${CALIB_MICROBATCH:-auto}"

PROD_FORMATS="${PROD_FORMATS:-NVFP4}"
PROD_LEVERS="${PROD_LEVERS:-gptq,scale_sweep}"
PROD_N_CALIB_SAMPLES="${PROD_N_CALIB_SAMPLES:-${N_CALIB_SAMPLES}}"
PROD_CALIB_SEQLEN="${PROD_CALIB_SEQLEN:-${CALIB_SEQLEN}}"
PROD_MAX_ACT_ROWS="${PROD_MAX_ACT_ROWS:-512}"
PRODUCTION_CACHE_LRU_GB="${PRODUCTION_CACHE_LRU_GB:-16}"
PRODUCTION_CACHE_PREFETCH="${PRODUCTION_CACHE_PREFETCH:-batch}"
WEIGHT_SESSION_SNAPSHOT_DIR="${WEIGHT_SESSION_SNAPSHOT_DIR:-}"
OF_WEIGHT_SESSION_SNAPSHOT_DIR="${OF_WEIGHT_SESSION_SNAPSHOT_DIR:-${WEIGHT_SESSION_SNAPSHOT_DIR}}"

ALLOCATOR_PROBE="${ALLOCATOR_PROBE:-}"
ALLOCATOR_COSTS="${ALLOCATOR_COSTS:-}"
ALLOCATOR_TARGET_BITS="${ALLOCATOR_TARGET_BITS:-6.0}"
PARETO_TARGETS="${PARETO_TARGETS:-4.5,4.6,4.75,5.0,5.25,5.5,6.0,7.0,8.0}"
SEED_DIR="${SEED_DIR:-}"

REFINE_N_CALIB_SAMPLES="${REFINE_N_CALIB_SAMPLES:-${N_CALIB_SAMPLES}}"
REFINE_CALIB_SEQLEN="${REFINE_CALIB_SEQLEN:-${CALIB_SEQLEN}}"
REFINE_CALIB_MICROBATCH="${REFINE_CALIB_MICROBATCH:-1}"
REFINE_MAX_ITERATIONS="${REFINE_MAX_ITERATIONS:-1}"
REFINE_NEIGHBORS_VALIDATE="${REFINE_NEIGHBORS_VALIDATE:-4}"
REFINE_POLISH_MAX_PASSES="${REFINE_POLISH_MAX_PASSES:-8}"
REFINE_POLISH_BUDGET_CREEP="${REFINE_POLISH_BUDGET_CREEP:-0.05}"
REFINE_LOGIT_SCOPE="${REFINE_LOGIT_SCOPE:-last_token}"
OUTPUT_FISHER_REDUCTION_DEVICE="${OUTPUT_FISHER_REDUCTION_DEVICE:-auto}"
REFINE_SKIP_POLISH="${REFINE_SKIP_POLISH:-0}"
REFINE_PRODUCTION_CACHE_LRU_GB="${REFINE_PRODUCTION_CACHE_LRU_GB:-${PRODUCTION_CACHE_LRU_GB}}"
if [[ -n "${WEIGHT_SESSION_SNAPSHOT_DIR}" ]]; then
  REFINE_PRODUCTION_CACHE_PREFETCH="${REFINE_PRODUCTION_CACHE_PREFETCH:-initial_center}"
else
  REFINE_PRODUCTION_CACHE_PREFETCH="${REFINE_PRODUCTION_CACHE_PREFETCH:-none}"
fi

NO_ACTIVATION_QUANT="${NO_ACTIVATION_QUANT:-1}"
ENABLE_CUDA_GRAPHS="${ENABLE_CUDA_GRAPHS:-0}"
RUN_BUILD_CACHE="${RUN_BUILD_CACHE:-auto}"
RUN_ALLOCATOR="${RUN_ALLOCATOR:-auto}"
RUN_SEED_PROBE="${RUN_SEED_PROBE:-1}"
RUN_REFINE="${RUN_REFINE:-1}"

mkdir -p "${RUN_ROOT}"/{logs,allocator,probe_work,refine,cache}
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUN_ROOT}/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUN_ROOT}/cache/inductor}"
export PRISMAQUANT_DISABLE_RTN_COMPILE="${PRISMAQUANT_DISABLE_RTN_COMPILE:-1}"
export PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE="${PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE:-1}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

log() {
  printf '[proto] %s\n' "$*"
}

common_local_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  common_local_args+=(--local-files-only)
fi

activation_args=()
if [[ "${NO_ACTIVATION_QUANT}" == "1" ]]; then
  activation_args+=(--no-activation-quant)
fi

cuda_graph_args=()
if [[ "${ENABLE_CUDA_GRAPHS}" == "1" ]]; then
  cuda_graph_args+=(--enable-cuda-graphs)
fi

if [[ -n "${OF_WEIGHT_SESSION_SNAPSHOT_DIR}" ]]; then
  export PRISMAQUANT_OF_WEIGHT_SESSION_SNAPSHOT_DIR="${OF_WEIGHT_SESSION_SNAPSHOT_DIR}"
fi
if [[ -n "${WEIGHT_SESSION_SNAPSHOT_DIR}" ]]; then
  export PRISMAQUANT_WEIGHT_SESSION_SNAPSHOT_DIR="${WEIGHT_SESSION_SNAPSHOT_DIR}"
fi

weight_session_args=()
if [[ -n "${WEIGHT_SESSION_SNAPSHOT_DIR}" ]]; then
  weight_session_args+=(--weight-session-snapshot-dir "${WEIGHT_SESSION_SNAPSHOT_DIR}")
fi

refine_polish_args=()
if [[ "${REFINE_SKIP_POLISH}" == "1" ]]; then
  refine_polish_args+=(--skip-polish)
fi

PRODUCTION_CACHE_PKL="${PRODUCTION_CACHE_PKL:-${RUN_ROOT}/production_weight_cache.pkl}"
PRODUCTION_CACHE_DIR="${PRODUCTION_CACHE_DIR:-${RUN_ROOT}/production_weight_cache}"

if [[ "${RUN_BUILD_CACHE}" == "1" || ( "${RUN_BUILD_CACHE}" == "auto" && ! -f "${PRODUCTION_CACHE_PKL}" ) ]]; then
  log "building production cache: ${PRODUCTION_CACHE_PKL}"
  "${PYTHON_BIN}" -m prismaquant.build_production_cache \
    --model "${MODEL_PATH}" \
    --output "${PRODUCTION_CACHE_PKL}" \
    --cache-dir "${PRODUCTION_CACHE_DIR}" \
    --formats "${PROD_FORMATS}" \
    --enable "${PROD_LEVERS}" \
    --n-calib-samples "${PROD_N_CALIB_SAMPLES}" \
    --calib-seqlen "${PROD_CALIB_SEQLEN}" \
    --calib-split "${CALIB_SPLIT}" \
    --calib-seed "${CALIB_SEED}" \
    --max-act-rows "${PROD_MAX_ACT_ROWS}" \
    --dtype bf16 \
    2>&1 | tee "${RUN_ROOT}/logs/build_production_cache.log"
else
  log "reusing production cache: ${PRODUCTION_CACHE_PKL}"
fi

if [[ ! -f "${PRODUCTION_CACHE_PKL}" ]]; then
  log "missing production cache: ${PRODUCTION_CACHE_PKL}"
  exit 2
fi
if [[ ! -d "${PRODUCTION_CACHE_DIR}" ]]; then
  log "warning: production cache dir not found; using manifest path as-is: ${PRODUCTION_CACHE_DIR}"
fi

if [[ -z "${SEED_DIR}" ]]; then
  SEED_DIR="${RUN_ROOT}/allocator/pareto_assignments"
fi

if [[ "${RUN_ALLOCATOR}" == "1" || ( "${RUN_ALLOCATOR}" == "auto" && ! -d "${SEED_DIR}" ) ]]; then
  if [[ -z "${ALLOCATOR_PROBE}" || -z "${ALLOCATOR_COSTS}" ]]; then
    log "SEED_DIR does not exist and ALLOCATOR_PROBE/ALLOCATOR_COSTS were not supplied"
    log "set SEED_DIR=/path/to/allocator_pareto_assignments or provide allocator pickles"
    exit 2
  fi
  log "generating allocator Pareto seeds: ${SEED_DIR}"
  mkdir -p "${SEED_DIR}"
  "${PYTHON_BIN}" -m prismaquant.allocator \
    --probe "${ALLOCATOR_PROBE}" \
    --costs "${ALLOCATOR_COSTS}" \
    --model-override "${MODEL_PATH}" \
    --target-bits "${ALLOCATOR_TARGET_BITS}" \
    --formats "${FORMATS}" \
    --pareto-targets "${PARETO_TARGETS}" \
    --layer-config "${RUN_ROOT}/allocator/layer_config_target.json" \
    --pareto-csv "${RUN_ROOT}/allocator/pareto.csv" \
    --pareto-output-dir "${SEED_DIR}" \
    2>&1 | tee "${RUN_ROOT}/logs/allocator.log"
else
  log "reusing allocator seed dir: ${SEED_DIR}"
fi

mapfile -t seed_files < <(
  find "${SEED_DIR}" -maxdepth 1 -type f -name '*.json' ! -name 'manifest.json' | sort
)
if [[ "${#seed_files[@]}" -eq 0 ]]; then
  log "no seed assignment JSON files found in ${SEED_DIR}"
  exit 2
fi
log "seed assignments: ${#seed_files[@]}"

seed_args=()
for seed in "${seed_files[@]}"; do
  seed_args+=(--seed-assignment "${seed}")
done

SEED_PROBE_JSON="${RUN_ROOT}/seed_frontier_probe.json"
if [[ "${RUN_SEED_PROBE}" == "1" ]]; then
  log "measuring allocator seed frontier: ${SEED_PROBE_JSON}"
  "${PYTHON_BIN}" -m prismaquant.kl_sensitivity_probe \
    --model "${MODEL_PATH}" \
    --output "${SEED_PROBE_JSON}" \
    --work-root "${RUN_ROOT}/probe_work" \
    --floor-format "${FLOOR_FORMAT}" \
    --formats "${FORMATS}" \
    ${PIN_ARGS} \
    --calib-split "${CALIB_SPLIT}" \
    --n-calib-samples "${N_CALIB_SAMPLES}" \
    --calib-seqlen "${CALIB_SEQLEN}" \
    --calib-seed "${CALIB_SEED}" \
    --kl-scope "${KL_SCOPE}" \
    --max-lanes-per-batch "${MAX_LANES_PER_BATCH}" \
    --calib-microbatch "${CALIB_MICROBATCH}" \
    --candidate-recipe production \
    --candidate-search seed_only \
    --production-weight-cache "${PRODUCTION_CACHE_PKL}" \
    --production-cache-dir-override "${PRODUCTION_CACHE_DIR}" \
    --production-cache-lru-gb "${PRODUCTION_CACHE_LRU_GB}" \
    --production-cache-prefetch "${PRODUCTION_CACHE_PREFETCH}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --dtype bf16 \
    --device cuda \
    "${activation_args[@]}" \
    "${cuda_graph_args[@]}" \
    "${weight_session_args[@]}" \
    "${common_local_args[@]}" \
    "${seed_args[@]}" \
    2>&1 | tee "${RUN_ROOT}/logs/seed_frontier_probe.log"
else
  log "skipping seed frontier measurement; expecting ${SEED_PROBE_JSON}"
fi

if [[ ! -f "${SEED_PROBE_JSON}" ]]; then
  log "missing seed frontier probe output: ${SEED_PROBE_JSON}"
  exit 2
fi

REFINE_ROOT="${RUN_ROOT}/refine"
if [[ "${RUN_REFINE}" == "1" ]]; then
  log "running centered output-Fisher Block-CLADO refinement: ${REFINE_ROOT}"
  PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE=1 \
  PRISMAQUANT_DISABLE_RTN_COMPILE=1 \
  "${PYTHON_BIN}" -m prismaquant.iterate_block_clado \
    --model "${MODEL_PATH}" \
    --output-root "${REFINE_ROOT}" \
    --max-iterations "${REFINE_MAX_ITERATIONS}" \
    --formats "${FORMATS}" \
    --n-calib-samples "${REFINE_N_CALIB_SAMPLES}" \
    --calib-seqlen "${REFINE_CALIB_SEQLEN}" \
    --calib-microbatch "${REFINE_CALIB_MICROBATCH}" \
    --calib-split "${CALIB_SPLIT}" \
    --calib-seed "${CALIB_SEED}" \
    --dtype bf16 \
    --device cuda \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --measure-method output_fisher \
    --output-fisher-logit-scope "${REFINE_LOGIT_SCOPE}" \
    --output-fisher-reduction-device "${OUTPUT_FISHER_REDUCTION_DEVICE}" \
    --n-neighbors-validate "${REFINE_NEIGHBORS_VALIDATE}" \
    --polish-max-passes "${REFINE_POLISH_MAX_PASSES}" \
    --polish-noise-floor 1e-5 \
    --polish-budget-creep "${REFINE_POLISH_BUDGET_CREEP}" \
    --polish-steepest-first \
    "${refine_polish_args[@]}" \
    --production-weight-cache "${PRODUCTION_CACHE_PKL}" \
    --production-cache-dir-override "${PRODUCTION_CACHE_DIR}" \
    --production-cache-lru-gb "${REFINE_PRODUCTION_CACHE_LRU_GB}" \
    --production-cache-prefetch "${REFINE_PRODUCTION_CACHE_PREFETCH}" \
    --initial-center-assignment "${SEED_PROBE_JSON}" \
    "${activation_args[@]}" \
    "${weight_session_args[@]}" \
    "${common_local_args[@]}" \
    2>&1 | tee "${RUN_ROOT}/logs/refine.log"
else
  log "skipping refinement"
fi

log "writing manifest"
RUN_ROOT="${RUN_ROOT}" \
MODEL_PATH="${MODEL_PATH}" \
FORMATS="${FORMATS}" \
FLOOR_FORMAT="${FLOOR_FORMAT}" \
N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
CALIB_SEQLEN="${CALIB_SEQLEN}" \
KL_SCOPE="${KL_SCOPE}" \
REFINE_N_CALIB_SAMPLES="${REFINE_N_CALIB_SAMPLES}" \
REFINE_CALIB_SEQLEN="${REFINE_CALIB_SEQLEN}" \
REFINE_LOGIT_SCOPE="${REFINE_LOGIT_SCOPE}" \
PRODUCTION_CACHE_PKL="${PRODUCTION_CACHE_PKL}" \
PRODUCTION_CACHE_DIR="${PRODUCTION_CACHE_DIR}" \
SEED_DIR="${SEED_DIR}" \
SEED_PROBE_JSON="${SEED_PROBE_JSON}" \
REFINE_ROOT="${REFINE_ROOT}" \
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
from collections import Counter

from prismaquant import block_clado as bc


def load_json(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def sha256(path):
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return None


run_root = pathlib.Path(os.environ["RUN_ROOT"])
seed_probe_path = pathlib.Path(os.environ["SEED_PROBE_JSON"])
refine_root = pathlib.Path(os.environ["REFINE_ROOT"])
seed_probe = load_json(seed_probe_path) or {}
refine_summary = load_json(refine_root / "summary.json")
refine_best = load_json(refine_root / "best_assignment.json")
refine_payload = load_json(refine_root / "iter_0" / "block_clado.json")


def total_params_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    try:
        total = bc.total_param_count(payload)
    except Exception:
        return None
    return total if total > 0 else None


total_params = total_params_from_payload(refine_payload)


def point_kl(point):
    if not isinstance(point, dict):
        return None
    for key in ("measured_kl", "real_kl", "kl", "polished_kl"):
        value = point.get(key)
        if value is not None:
            return float(value)
    return None


def point_bpp(point):
    if not isinstance(point, dict):
        return None
    for key in ("bpp", "bits_per_param"):
        value = point.get(key)
        if value is not None:
            return float(value)
    bits = point.get("bits_total")
    if bits is not None and total_params:
        return float(bits) / float(total_params)
    return None


def summarize_frontier_point(point):
    if not isinstance(point, dict):
        return {}
    return {
        "label": point.get("label"),
        "source": point.get("source"),
        "kl": point_kl(point),
        "bpp": point_bpp(point),
        "bits_total": point.get("bits_total"),
        "measured_gain": point.get("measured_gain"),
        "assignment_hash": point.get("assignment_hash"),
        "promotion_count": point.get("promotion_count"),
    }


chosen = ((seed_probe.get("selection") or {}).get("chosen") or {})
seed_assignment = chosen.get("assignment") or seed_probe.get("chosen_assignment") or {}
final_source = "seed_probe"
final_assignment = dict(seed_assignment)

refine_meta = (refine_payload or {}).get("meta") or {}
refine_center_kl = refine_meta.get("center_kl")
refine_best_kl = point_kl(refine_best)
refine_improved_center = (
    isinstance(refine_best, dict)
    and isinstance(refine_best.get("assignment"), dict)
    and refine_center_kl is not None
    and refine_best_kl is not None
    and float(refine_best_kl) < float(refine_center_kl) - 1e-9
)
if refine_improved_center:
    final_source = "refine_best"
    final_assignment = dict(refine_best["assignment"])

final_path = run_root / "final_assignment.json"
final_payload = {
    "schema": "prismaquant.prismascout.final_assignment.v1",
    "source": final_source,
    "assignment": dict(sorted(final_assignment.items())),
    "format_counts": dict(sorted(Counter(final_assignment.values()).items())),
    "selection_reason": (
        "refinement_improved_center"
        if refine_improved_center
        else "seed_kept_refinement_did_not_improve_center"
    ),
}
final_path.write_text(json.dumps(final_payload, indent=2, sort_keys=True) + "\n")

floor = seed_probe.get("floor") or {}
seed_frontier_points = [
    summarize_frontier_point(point)
    for point in ((seed_probe.get("selection") or {}).get("frontier") or [])
]

manifest = {
    "schema": "prismaquant.prismascout.proto_production.v1",
    "git": {
        "head": git_value(["git", "rev-parse", "HEAD"]),
        "branch": git_value(["git", "branch", "--show-current"]),
        "status_porcelain": git_value(["git", "status", "--short"]),
    },
    "model": os.environ["MODEL_PATH"],
    "formats": os.environ["FORMATS"],
    "floor_format": os.environ["FLOOR_FORMAT"],
    "calibration": {
        "n_calib_samples": int(os.environ["N_CALIB_SAMPLES"]),
        "calib_seqlen": int(os.environ["CALIB_SEQLEN"]),
        "kl_scope": os.environ["KL_SCOPE"],
    },
    "production_cache": {
        "manifest": os.environ["PRODUCTION_CACHE_PKL"],
        "cache_dir": os.environ["PRODUCTION_CACHE_DIR"],
        "manifest_sha256": sha256(os.environ["PRODUCTION_CACHE_PKL"]),
    },
    "allocator_seeds": {
        "seed_dir": os.environ["SEED_DIR"],
        "seed_count": len(list(pathlib.Path(os.environ["SEED_DIR"]).glob("*.json"))) - (
            1 if (pathlib.Path(os.environ["SEED_DIR"]) / "manifest.json").exists() else 0
        ),
        "manifest_sha256": sha256(pathlib.Path(os.environ["SEED_DIR"]) / "manifest.json"),
    },
    "seed_frontier": {
        "path": str(seed_probe_path),
        "schema": seed_probe.get("schema"),
        "floor_kl": floor.get("kl"),
        "floor_bpp": point_bpp(floor),
        "floor_bits_total": floor.get("bits_total"),
        "chosen_label": chosen.get("label"),
        "chosen_kl": point_kl(chosen),
        "chosen_bpp": point_bpp(chosen),
        "chosen_bits_total": chosen.get("bits_total"),
        "chosen_measured_gain": chosen.get("measured_gain"),
        "chosen_assignment_hash": chosen.get("assignment_hash"),
        "frontier_points": seed_frontier_points,
        "production_cache_used": (seed_probe.get("diagnostics") or {}).get("production_cache_used"),
    },
    "refinement": {
        "root": str(refine_root),
        "n_calib_samples": int(os.environ["REFINE_N_CALIB_SAMPLES"]),
        "calib_seqlen": int(os.environ["REFINE_CALIB_SEQLEN"]),
        "logit_scope": os.environ["REFINE_LOGIT_SCOPE"],
        "center_kl": refine_center_kl,
        "best_kl": refine_best_kl,
        "accepted": bool(refine_improved_center),
        "rejection_reason": (
            None if refine_improved_center else "best_validated_not_better_than_center"
        ),
        "summary": refine_summary,
    },
    "final_assignment": {
        "path": str(final_path),
        "source": final_source,
        "format_counts": final_payload["format_counts"],
        "selection_reason": final_payload["selection_reason"],
    },
}
(run_root / "proto_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    "manifest": str(run_root / "proto_manifest.json"),
    "final_assignment": str(final_path),
    "seed_chosen_kl": manifest["seed_frontier"]["chosen_kl"],
    "seed_chosen_bpp": manifest["seed_frontier"]["chosen_bpp"],
    "final_source": final_source,
}, indent=2, sort_keys=True))
PY

log "done"
log "run root: ${RUN_ROOT}"
log "manifest: ${RUN_ROOT}/proto_manifest.json"
log "final assignment: ${RUN_ROOT}/final_assignment.json"
