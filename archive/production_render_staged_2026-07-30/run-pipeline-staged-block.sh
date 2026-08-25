# Verbatim staged block removed from prismaquant/run-pipeline.sh on 2026-07-30
# (re-vet R17). Stages [2b/4] .. [2e/4] of COST_MODE=production-render-staged.
# Restoring the lane means pasting this back immediately after the
# production-render-score [2c/4] block, restoring the shell defaults
#   PRODUCTION_RENDER_COST_PROMOTE_FRACTION / _MIN_PROMOTE_SCORE / _MAX_PROMOTIONS
#   PRODUCTION_RENDER_COST_TAIL_QNAMES (set in the COST_MODE case arm)
# and restoring the ten CLI args on prismaquant.production_render_cost.

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
