#!/usr/bin/env bash
# DeepSeek-V4-Flash-Base quantization run.
#
# Source: deepseek-ai/DeepSeek-V4-Flash-Base — 274 GB FP8-native (FP8
# E4M3 + UE8M0 [128,128] block scales for routed experts and attention
# projections; BF16 for norms / embed / head / compressor / hyper-conn;
# F32 for hash-cluster heads + attn_sink + compressor.ape).
#
# Targets: best-effort compact Spark artifact using only formats that
# have native runtime support. Sub-4-bit custom formats are omitted
# because they require custom kernels.
#
# Vendoring + patches: detect_profile() automatically calls
# prismaquant.vendored.register_deepseek_v4() before any DSv4 weights
# load — registers the PR #45643 modeling class with AutoConfig +
# AutoModelForCausalLM and applies three small monkey-patches against
# transformers 5.5.4 (ALLOWED_LAYER_TYPES extension, sqrtsoftplus
# activation, rope_theta float|int annotation). No launcher action
# required beyond keeping the container's transformers at the default
# 5.5.4 (no `pip install transformers==X.Y.Z` line — see
# container_transformers_pin.md memory).
#
# Memory budget on Spark (128 GB UMA):
#   probe peak ~115 GB, cost step peak ~70 GB. The DSv4 source is
#   1.3× MiniMax M2.7's size, so peaks creep up. If VmHWM crosses
#   118 GB sustained, halve --layers-per-shard or --activation-rows-limit.
set -euo pipefail

if docker ps --format '{{.Names}}' | grep -E '^pq-(qwen|minimax|gemma|deepseek|llama|trial)-' | head -1; then
    echo "ABORT: another pq container is running."
    exit 1
fi

HOST_SNAP=$(ls -d /home/rob/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Base/snapshots/*/ 2>/dev/null | head -1)
[ -z "$HOST_SNAP" ] && { echo "DSv4-Flash-Base snapshot missing"; exit 1; }
CONTAINER_SNAP=${HOST_SNAP/\/home\/rob\/.cache\/huggingface/\/hfcache}
CONTAINER_SNAP=${CONTAINER_SNAP%/}

# Sanity: confirm safetensors are actually present (the metadata-only
# snapshot would still match the glob above).
HOST_SNAP_TRIM=${HOST_SNAP%/}
SHARD_COUNT=$(ls "$HOST_SNAP_TRIM"/*.safetensors 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -lt 46 ]; then
    echo "ABORT: only $SHARD_COUNT/46 safetensors shards in $HOST_SNAP_TRIM"
    echo "       wait for hf download to finish or re-run it"
    exit 1
fi

CAL_MIX=/home/rob/dq-runs/cal-mix-v1/cal_mix_shuf.jsonl
[ -f "$CAL_MIX" ] || { echo "cal mix missing: $CAL_MIX"; exit 1; }

HOST_WORK=/home/rob/dq-runs/dsv4-flash-base
mkdir -p "$HOST_WORK"/{chunks,artifacts,act,logs,work,cost_work,export-cache}

# Split cal_mix into chunks. DSv4-Flash-Base needs smaller per-chunk
# samples than MiniMax M2.7 because the model propagates hidden state
# in multi-stream form `[B, T, hc_mult, H]` (hc_mult=4) — phase-1
# activation tensors are ~3x bigger than the standard
# single-stream form. With 12 samples × 2048 × 4 × 4096 ≈ 35 GB just
# for the activation cache, the 121 GB Spark UMA OOMs at the
# device→host stack step. 6 samples halves to ~17 GB and leaves
# comfortable headroom alongside the 76 GB layer cache budget.
N_CHUNKS=32
SAMPLES_PER_CHUNK=6
SEQLEN=2048

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

docker rm -f pq-deepseek-v4 2>/dev/null || true

AVAIL_GB=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
echo "[launch] host MemAvailable = ${AVAIL_GB} GB"

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) \
  --name pq-deepseek-v4 \
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
    pip install --quiet accelerate datasets 2>&1 | tail -1

    MODEL='$CONTAINER_SNAP'
    MERGED_PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl

    echo '[1/2] multi-chunk probe + in-process cost ...'
    python3 -m prismaquant.multi_chunk_probe \\
      --chunks-dir /work/chunks \\
      --model \"\$MODEL\" \\
      --output \"\$MERGED_PROBE\" \\
      --activation-cache-dir /work/act \\
      --work-dir /work/work \\
      --device cuda --dtype bf16 \\
      --seqlen $SEQLEN \\
      --layers-per-shard 4 \\
      --unified-sweep \\
      --no-include-mtp --no-include-visual --no-include-lm-head \\
      --prefetch-lookahead 4 --prefetch-workers 2 \\
      --prefetch-min-available-gb 30 \\
      --activation-rows-limit 256 \\
      --calibration-modality text-only \\
      --retain-cross-chunk-cache \\
      --adaptive-sampling \\
      --adaptive-prune-ratio 0.375 \\
      --run-cost \\
      --cost-output \"\$COST\" \\
      --cost-formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \\
      --cost-work-dir /work/cost_work \\
      --cost-mode batched --cost-chunk-size 256 \\
      --cost-layers-per-shard 4 \\
      --cost-swap-grow-limit-mb 4096 \\
      2>&1 | tee /work/logs/probe_cost.log

    echo '[2/2] allocator (native-kernel format menu) ...'
    python3 -m prismaquant.allocator \\
      --model \"\$MODEL\" \\
      --probe \"\$MERGED_PROBE\" \\
      --costs \"\$COST\" \\
      --formats NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16 \\
      --target-bits 4.00 \\
      --pareto-targets 3.00,3.25,3.50,3.75,4.00,4.25,4.50,5.00 \\
      --enable-expert-prune \\
      --prune-ratios 0.0,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5 \\
      --prune-alpha 0.15 \\
      --prune-domain-policy union \\
      --layer-config /work/artifacts/layer_config.json \\
      --pareto-csv /work/artifacts/pareto.csv \\
      2>&1 | tee /work/logs/allocator.log

    echo '[done] probe=' \$MERGED_PROBE ' cost=' \$COST ' layer_config=/work/artifacts/layer_config.json'
    echo '       pareto=/work/artifacts/pareto.csv'
    echo
    echo 'NEXT: review /work/artifacts/pareto.csv, pick a target near 90 GB,'
    echo '      then run the export step (launch-dsv4-export.sh, mirroring'
    echo '      launch-minimax-allocate-export.sh w/ --export-cache-dir).'
"

echo "[launch] container: pq-deepseek-v4"
echo "[launch] tail:      docker logs -f pq-deepseek-v4"
echo "[launch]   or:      tail -f $HOST_WORK/logs/probe_cost.log"
