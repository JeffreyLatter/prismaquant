#!/usr/bin/env bash
# Side-by-side comparison: HALO-off baseline vs Hadamard-DuQuant.
#
# Runs two PrismaQuant pipelines:
#   (1) baseline: HADAMARD_DUQUANT=0 — the current production stack
#       (HALO off, NVFP4 + four_over_six + GPTQ + scale_sweep)
#   (2) experiment: HADAMARD_DUQUANT=1 — Hadamard-DuQuant on top of (1)
#
# Both run with identical calibration, target bpp, format menu, and levers
# so the only delta is the rotation axis. Outputs the two artifact paths
# and a side-by-side summary. The full first-ship gate (KL n=8/n=64,
# WikiText-2 PPL, C4 PPL, BF16-argmax) is run via the validator scripts
# the user invokes after this harness completes — see the printed
# "next steps" footer.
#
# Invocation: from a CUDA + vLLM-enabled environment
#   ./tools/compare_hadamard_duquant_to_baseline.sh
#
# Override defaults via env vars before invoking:
#   MODEL_PATH=/path/to/source-model
#   COMPARE_ROOT=/path/to/run-dirs        (default: dq-runs/hdq-compare-<ts>)
#   NSAMPLES=8 SEQLEN=512 TARGET_BITS=4.5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${MODEL_PATH:=/home/rob/.cache/huggingface/qwen35-0p8b-bf16-untied}"
: "${COMPARE_ROOT:=/home/rob/dq-runs/hdq-compare-$(date +%Y%m%dT%H%M%SZ)}"
: "${FORMATS:=NVFP4,MXFP8_E4M3,FP8_E4M3,BF16}"
: "${TARGET_BITS:=4.5}"
: "${NSAMPLES:=8}"
: "${SEQLEN:=512}"
: "${DATASET:=/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
: "${TARGET_PROFILE:=vllm_qwen3_5_packed_moe}"

BASELINE_WORK="$COMPARE_ROOT/baseline"
HDQ_WORK="$COMPARE_ROOT/hadamard_duquant"

mkdir -p "$BASELINE_WORK/logs" "$BASELINE_WORK/artifacts"
mkdir -p "$HDQ_WORK/logs"     "$HDQ_WORK/artifacts"

cat <<EOF
=================================================================
HDQ vs baseline comparison
-----------------------------------------------------------------
MODEL:                  $MODEL_PATH
COMPARE_ROOT:           $COMPARE_ROOT
  baseline:             $BASELINE_WORK
  hadamard-duquant:     $HDQ_WORK
FORMATS:                $FORMATS
TARGET_BITS:            $TARGET_BITS
NSAMPLES x SEQLEN:      $NSAMPLES x $SEQLEN
DATASET:                $DATASET
=================================================================
EOF

run_pipeline() {
  local work_dir="$1"; shift
  local label="$1"; shift
  local hdq_value="$1"; shift
  echo
  echo "▶ [$label] starting pipeline at WORK_DIR=$work_dir"
  echo "  HADAMARD_DUQUANT=$hdq_value HALO_MODE=off"
  echo
  env \
    MODEL_PATH="$MODEL_PATH" \
    WORK_DIR="$work_dir" \
    FORMATS="$FORMATS" \
    TARGET_BITS="$TARGET_BITS" \
    NSAMPLES="$NSAMPLES" \
    SEQLEN="$SEQLEN" \
    DATASET="$DATASET" \
    TARGET_PROFILE="$TARGET_PROFILE" \
    HADAMARD_DUQUANT="$hdq_value" \
    HALO_MODE=off \
    PRODUCTION_CACHE_LEVERS=gptq,scale_sweep \
    PRISMAQUANT_NVFP4_SCALE_RULE=four_over_six_mse \
    FISHER_OUTPUT_MSE_ALLOCATOR=1 \
    bash "$REPO_ROOT/prismaquant/run-pipeline.sh"
  echo "✓ [$label] complete"
}

run_pipeline "$BASELINE_WORK" "baseline"          0
run_pipeline "$HDQ_WORK"      "hadamard-duquant" 1

# -----------------------------------------------------------------------
# Summary diff
# -----------------------------------------------------------------------

echo
echo "================================================================="
echo "Pipeline summary — side by side"
echo "-----------------------------------------------------------------"

summarize() {
  local work_dir="$1"
  local label="$2"
  local layer_cfg="$work_dir/artifacts/layer_config.json"
  local manifest="$work_dir/exported/mixed_native_manifest.json"
  local hdq_picks="$work_dir/artifacts/hadamard_duquant_picks.json"
  local hdq_decisions="$work_dir/artifacts/hadamard_duquant_decisions.jsonl"
  echo
  echo "▣ $label"
  if [[ -f "$layer_cfg" ]]; then
    python3 - "$layer_cfg" <<'PY' || true
import json, sys, collections
cfg = json.load(open(sys.argv[1]))
dist = collections.Counter()
for v in cfg.values():
    if isinstance(v, dict):
        dist[v.get("bits", v.get("format", "?"))] += 1
    else:
        dist[str(v)] += 1
print(f"  layer_config entries: {len(cfg)}")
for k in sorted(dist):
    print(f"    {k}: {dist[k]}")
PY
  fi
  if [[ -f "$manifest" ]]; then
    python3 - "$manifest" <<'PY' || true
import json, sys
mf = json.load(open(sys.argv[1]))
hist = mf.get("format_histogram", {})
print(f"  format histogram: {hist}")
PY
  fi
  if [[ -f "$hdq_picks" ]]; then
    python3 - "$hdq_picks" <<'PY' || true
import json, sys, collections
picks = json.load(open(sys.argv[1])).get("picks", {})
dist = collections.Counter(v.split("+")[0] for v in picks.values())
print(f"  hdq picks: {len(picks)} clusters; {dict(dist)}")
PY
  fi
  if [[ -f "$hdq_decisions" ]]; then
    n_lines=$(wc -l < "$hdq_decisions")
    echo "  hdq decisions: $n_lines records → $hdq_decisions"
  fi
}

summarize "$BASELINE_WORK" "baseline (HADAMARD_DUQUANT=0)"
summarize "$HDQ_WORK"      "experiment (HADAMARD_DUQUANT=1)"

cat <<EOF

=================================================================
Next steps — first-ship gate (run from CUDA + vLLM 0.20):
-----------------------------------------------------------------
# 1. vLLM load smoke (both eager and CUDA-graph)
vllm serve $BASELINE_WORK/exported --quantization compressed-tensors --enforce-eager &
vllm serve $HDQ_WORK/exported     --quantization compressed-tensors --enforce-eager &

# 2. Calibration KL (n=8 and n=64, seq=512)
python3 -m prismaquant.validate_assignments_kl \\
  --model $MODEL_PATH \\
  --layer-config $HDQ_WORK/artifacts/layer_config.json \\
  --dataset $DATASET --n-samples 8 64 --seqlen 512

# 3. WikiText-2 + C4 PPL
python3 -m prismaquant.validate_native_export \\
  --model $HDQ_WORK/exported --ppl wikitext c4

# 4. BF16-argmax agreement
python3 -m prismaquant.validate_quantized_model \\
  --model $HDQ_WORK/exported --baseline $MODEL_PATH --metric argmax_agreement

# 5. lm-eval zero-shot suite
lm_eval --model vllm --model_args pretrained=$HDQ_WORK/exported,quantization=compressed-tensors \\
  --tasks hellaswag,arc_easy,winogrande,piqa --batch_size 8

# Compare each metric against the baseline run at $BASELINE_WORK. The
# first-ship criterion is non-regressing on ALL four (KL, PPL, argmax,
# vLLM load) at matched bpp.
=================================================================
EOF
