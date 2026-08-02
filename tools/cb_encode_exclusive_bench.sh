#!/usr/bin/env bash
# Exclusive-GPU old-vs-new validation for the CB encode optimization.
#
# Waits for the production cost container to exit, then measures BOTH arms
# with the GPU to itself: a real end-to-end single-layer driver run, plus a
# 3-warm-rep per-Linear bench, each with its own nvidia-smi dmon signature.
#
# Everything lands under $OUT. Never touches the production work-dir.
set -uo pipefail

PROD=pq-dsv4-cost-prod
SIDE=pq-perf-side
LOCK=/tmp/claude-1000/gpu-bench.lock
RUN=/home/rob/dq-runs/dsv4-flash-0731
OUT=$RUN/encoder-profile/exclusive
SCRATCH=$RUN/perf-bench-scratch
mkdir -p "$OUT" "$SCRATCH"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/bench.log"; }

log "waiting for $PROD to exit..."
while docker ps --format '{{.Names}}' | grep -qx "$PROD"; do sleep 60; done
log "$PROD is gone; settling 60s before exclusive measurement"
sleep 60

dmon_start() {  # $1 = tag
  nvidia-smi dmon -s pucm -d 1 -o T > "$OUT/dmon_$1.txt" 2>&1 &
  echo $!
}

# --- A: per-Linear bench, 3 warm reps, both arms -------------------------
for arm in base new; do
  pp=/pq; [ "$arm" = new ] && pp=/wt
  log "=== per-Linear bench [$arm] (PYTHONPATH=$pp) ==="
  d=$(dmon_start "perlinear_$arm")
  flock "$LOCK" docker exec -e PYTHONPATH=$pp -e PYTHONDONTWRITEBYTECODE=1 \
    "$SIDE" python3 /wt/tools/cb_encode_profile.py --reps 3 \
    --outdir "$OUT/$arm" >> "$OUT/perlinear_$arm.txt" 2>&1
  kill "$d" 2>/dev/null
  grep -E '^\[clean\]' "$OUT/perlinear_$arm.txt" | tee -a "$OUT/bench.log"
done

# --- B: real end-to-end single-layer driver run, both arms ---------------
for arm in base new; do
  pp=/pq; [ "$arm" = new ] && pp=/wt
  wd="$SCRATCH/work-$arm"; rm -rf "$wd"; mkdir -p "$wd"
  log "=== end-to-end layer 0 [$arm] ==="
  d=$(dmon_start "layer_$arm")
  t0=$(date +%s)
  flock "$LOCK" docker run --rm --gpus all --ipc=host --entrypoint bash \
    -v "$RUN":"$RUN" -v /home/rob/pq-perf-wt:/wt \
    -v /home/rob/prismaquant-ultraplan:/pq:ro \
    -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CB_CODEBOOK_SOURCE=lattice -e CB_SCALE_CODING=two_tier \
    -e CB_SCALE_SWEEP=1 -e PRISMAQUANT_CB_ENCODE_TIER=balanced \
    -e PYTHONPATH=$pp -e PYTHONDONTWRITEBYTECODE=1 \
    -e PRISMAQUANT_CB_EXT_DIR=$RUN/ext \
    -e PRISMAQUANT_CB_COL_WEIGHTS=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl \
    -e PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl.provenance.json \
    gridbook:test -c "
python3 -m prismaquant.incremental_measure_quant_cost \
  --model $RUN/source --cost-mode local \
  --probe $RUN/prod-cal-0p6/artifacts/probe.pkl \
  --activation-cache-dir $RUN/prod-cal-0p6/act \
  --formats 'NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36,BF16' \
  --output $wd/cost_full.pkl --work-dir $wd \
  --device cuda --dtype bf16 --mode batched --chunk-size 256 \
  --layers-per-shard 1 --start-layer 0 --end-layer 0 \
  --skip-missing-activations --no-include-lm-head
" >> "$OUT/layer_$arm.txt" 2>&1
  t1=$(date +%s)
  kill "$d" 2>/dev/null
  log "end-to-end layer 0 [$arm]: $((t1 - t0)) s"
  echo "$arm $((t1 - t0))" >> "$OUT/layer_seconds.txt"
done

# --- C: the shard rows the two arms produced must be identical -----------
log "=== shard row comparison ==="
flock "$LOCK" docker exec -e PYTHONPATH=/wt -e PYTHONDONTWRITEBYTECODE=1 \
  "$SIDE" python3 -c "
import pickle
a=pickle.load(open('$SCRATCH/work-base/shards/cost_shard_000.pkl','rb'))['costs']
b=pickle.load(open('$SCRATCH/work-new/shards/cost_shard_000.pkl','rb'))['costs']
assert set(a)==set(b), 'key sets differ'
bad=[]
for n in a:
    for f in a[n]:
        for k,v in a[n][f].items():
            w=b[n][f].get(k)
            if v!=w: bad.append((n,f,k,v,w))
print('rows', len(a), 'formats', len(next(iter(a.values()))))
print('EXACT ROW EQUALITY:', not bad, '(mismatches:', len(bad), ')')
for x in bad[:10]: print('  ', x)
" >> "$OUT/bench.log" 2>&1
tail -20 "$OUT/bench.log"

log "=== dmon summaries ==="
for f in "$OUT"/dmon_*.txt; do
  python3 - "$f" <<'PY' >> "$OUT/bench.log" 2>&1
import sys, statistics as st
rows=[]
for ln in open(sys.argv[1]):
    p=ln.split()
    if len(p)<8 or not p[1].isdigit(): continue
    try: rows.append((float(p[2]), float(p[5]), float(p[6])))
    except ValueError: pass
if rows:
    pw=[r[0] for r in rows]; sm=[r[1] for r in rows]; mm=[r[2] for r in rows]
    print(f"{sys.argv[1].split('/')[-1]}: n={len(rows)} "
          f"pw med={st.median(pw):.1f} max={max(pw):.1f} "
          f"sm med={st.median(sm):.0f} mem med={st.median(mm):.0f}")
PY
done
tail -8 "$OUT/bench.log"
log "DONE -> $OUT"
