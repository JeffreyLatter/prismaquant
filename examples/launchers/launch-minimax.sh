#!/usr/bin/env bash
# MiniMax-M2.7 quantization run on v21 (probe-throughput-v21 branch).
#
# Targets: 90-95 GB on-disk artifact. Uses target_bits ≈ 3.20-3.30 in the
# allocator pareto sweep and lets the kneedle pick within that band.
#
# v21 features in use:
#   #1 PRISMAQUANT_DEFERRED_FISHER_SYNC=1 — device-resident accumulators,
#      ~94k → ~62 CUDA syncs per phase-3 sweep.
#   #5 PRISMAQUANT_DIRECT_CUDA_LOAD=1   — safe_open(device="cuda:0")
#      lands tensors directly on GPU.
#   #4 multi_chunk_probe --retain-cross-chunk-cache  — keeps LayerCache
#      contents across chunk boundaries.
#   #3 multi_chunk_probe --adaptive-sampling        — tracks per-domain
#      saliency, narrows linear-include for chunks 2+ to skip experts
#      whose rank has stabilized. Calibration tokens concentrate on the
#      contested ~5-15% of experts after the first couple of chunks.
#   #2 multi_chunk_probe --run-cost                 — cost step runs
#      in-process after probe so we skip a separate Python launch.
#
# Memory budget on Spark (128 GB UMA):
#   probe peak ~110 GB, cost step peak ~60 GB. Probe ctx is torn down
#   before cost begins so we never have both alive at once. Watch the
#   VmHWM lines in probe.log — sustained creep across chunks is the
#   OOM warning sign.
set -euo pipefail

if docker ps --format '{{.Names}}' | grep -E '^pq-(qwen|minimax|gemma|deepseek|llama|trial)-' | head -1; then
    echo "ABORT: another pq container is running."
    exit 1
fi

HOST_SNAP=$(ls -d /home/rob/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*/ 2>/dev/null | head -1)
[ -z "$HOST_SNAP" ] && { echo "MiniMax snapshot missing"; exit 1; }
CONTAINER_SNAP=${HOST_SNAP/\/home\/rob\/.cache\/huggingface/\/hfcache}
CONTAINER_SNAP=${CONTAINER_SNAP%/}

CAL_MIX=/home/rob/dq-runs/cal-mix-v1/cal_mix_shuf.jsonl
[ -f "$CAL_MIX" ] || { echo "cal mix missing: $CAL_MIX"; exit 1; }

HOST_WORK=/home/rob/dq-runs/minimax-m2p7-v21
mkdir -p "$HOST_WORK"/{chunks,artifacts,act,logs,work}

# Split cal_mix into 32 chunks of 12 samples each. multi_chunk_probe
# expects chunk_NN.jsonl naming. With adaptive sampling on, chunks 4+
# typically run on 5-15% of the experts so 32 chunks comes in close to
# v20's 16-chunk wall time but with much better expert-selection
# fidelity. Bare chunk_NN.jsonl → "_global" domain (single-domain run).
N_CHUNKS=32
SAMPLES_PER_CHUNK=12
SEQLEN=2048

# Idempotent split: only re-split if the chunks dir is empty or the
# total size doesn't match.
NEED_SPLIT=1
if [ -f "$HOST_WORK/chunks/chunk_$(printf '%02d' $((N_CHUNKS-1))).jsonl" ]; then
    NEED_SPLIT=0
fi
if [ "$NEED_SPLIT" -eq 1 ]; then
    echo "[split] $CAL_MIX -> $HOST_WORK/chunks/ ($N_CHUNKS x $SAMPLES_PER_CHUNK lines)"
    python3 - <<EOF
import pathlib
src = pathlib.Path("$CAL_MIX")
dst = pathlib.Path("$HOST_WORK/chunks")
dst.mkdir(parents=True, exist_ok=True)
n_chunks = $N_CHUNKS
spc = $SAMPLES_PER_CHUNK
with src.open() as f:
    for i in range(n_chunks):
        out = dst / f"chunk_{i:02d}.jsonl"
        with out.open("w") as o:
            for _ in range(spc):
                line = f.readline()
                if not line:
                    break
                o.write(line)
        print(f"  {out.name}: {spc} lines")
EOF
fi

docker rm -f pq-minimax-v21 2>/dev/null || true

AVAIL_GB=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
echo "[launch] host MemAvailable = ${AVAIL_GB} GB"

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-minimax-v21 \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v /models:/models \
  -v "$HOST_WORK":/work \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  -e CACHE_HEADROOM_GB=40 \
  -e PRISMAQUANT_DEFERRED_FISHER_SYNC=1 \
  -e PRISMAQUANT_DEFERRED_FISHER_COMPUTE=1 \
  -e PRISMAQUANT_ACT_CACHE_ASYNC=1 \
  -e PRISMAQUANT_ACT_CACHE_WORKERS=4 \
  -e PRISMAQUANT_DIRECT_CUDA_LOAD=1 \
  -w /prismaquant \
  --entrypoint bash vllm-fresh-b12x:latest \
  -c "
    set -euo pipefail
    pip install --quiet accelerate datasets 'transformers==4.57.5' 2>&1 | tail -1

    MODEL='$CONTAINER_SNAP'
    MERGED_PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl

    echo '[1/2] multi-chunk probe + in-process cost (v21) ...'
    python3 -m prismaquant.multi_chunk_probe \
      --chunks-dir /work/chunks \
      --model \"\$MODEL\" \
      --output \"\$MERGED_PROBE\" \
      --activation-cache-dir /work/act \
      --work-dir /work/work \
      --device cuda --dtype bf16 \
      --seqlen $SEQLEN \
      --layers-per-shard 4 \
      --unified-sweep \
      --no-include-mtp --no-include-visual --no-include-lm-head \
      --prefetch-lookahead 4 --prefetch-workers 2 \
      --prefetch-min-available-gb 30 \
      --activation-rows-limit 256 \
      --calibration-modality text-only \
      --retain-cross-chunk-cache \
      --adaptive-sampling \
      --adaptive-prune-ratio 0.375 \
      --run-cost \
      --cost-output \"\$COST\" \
      --cost-formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \
      --cost-work-dir /work/cost_work \
      --cost-mode batched --cost-chunk-size 256 \
      --cost-layers-per-shard 4 \
      --cost-swap-grow-limit-mb 4096 \
      2>&1 | tee /work/logs/probe_cost.log

    echo '[2/2] allocator (target 90-95 GB band) ...'
    python3 -m prismaquant.allocator \
      --model \"\$MODEL\" \
      --probe \"\$MERGED_PROBE\" \
      --costs \"\$COST\" \
      --formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \
      --target-bits 3.25 \
      --pareto-targets 3.10,3.16,3.20,3.25,3.30,3.40,3.50,3.60 \
      --enable-expert-prune \
      --prune-ratios 0.0,0.125,0.1875,0.25,0.3125,0.375 \
      --prune-alpha 0.15 \
      --prune-domain-policy union \
      --layer-config /work/artifacts/layer_config_prune.json \
      --pareto-csv /work/artifacts/pareto_prune.csv \
      2>&1 | tee /work/logs/allocator.log

    echo '[done] probe=' \$MERGED_PROBE ' cost=' \$COST ' layer_config=/work/artifacts/layer_config_prune.json'
    echo '       pareto=/work/artifacts/pareto_prune.csv'
    echo
    echo 'NEXT: review /work/artifacts/pareto_prune.csv, pick a target in 90-95 GB,'
    echo '      then run the export step (launch-minimax-export.sh).'
"

echo "[launch] container: pq-minimax-v21"
echo "[launch] tail:      docker logs -f pq-minimax-v21"
echo "[launch]   or:      tail -f $HOST_WORK/logs/probe_cost.log"
