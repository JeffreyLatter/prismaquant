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

# --- C: identity gate -----------------------------------------------------
# Layer 0's Linears all have 64 cached rows, so the row fix is a NO-OP there:
# one row bucket, the old code path's shape. The new arm must therefore
# reproduce the SHIPPED v1 shard exactly (modulo the added n_activation_rows).
# Layers 3+ WILL differ -- that is the defect being fixed -- which is why the
# gate is pinned on layer 0.
log "=== identity gate: new arm layer 0 vs shipped v1 shard ==="
flock "$LOCK" docker exec -e PYTHONPATH=/wt -e PYTHONDONTWRITEBYTECODE=1 \
  "$SIDE" python3 -c "
import pickle
SHIPPED='$RUN/prod-cal-0p6/work-prod/shards/cost_shard_000.pkl'
NEW='$SCRATCH/work-new/shards/cost_shard_000.pkl'
BASE='$SCRATCH/work-base/shards/cost_shard_000.pkl'
ship=pickle.load(open(SHIPPED,'rb'))['costs']
new=pickle.load(open(NEW,'rb'))['costs']
base=pickle.load(open(BASE,'rb'))['costs']
IGNORE={'n_activation_rows'}
def diff(a,b):
    bad=[]
    if set(a)!=set(b): return [('KEYSET',len(set(a)^set(b)))]
    for n in a:
        for f in a[n]:
            for k,v in a[n][f].items():
                if k in IGNORE: continue
                if b[n][f].get(k)!=v: bad.append((n,f,k,v,b[n][f].get(k)))
    return bad
d_ship=diff(ship,new); d_base=diff(base,new)
print('rows',len(ship))
print('new-vs-SHIPPED-v1 layer0 exact:',not d_ship,'(mismatches',len(d_ship),')')
for x in d_ship[:6]: print('   ',x)
print('new-vs-base layer0 exact:',not d_base,'(mismatches',len(d_base),')')
for x in d_base[:6]: print('   ',x)
rows=[e.get('n_activation_rows') for v in new.values() for e in v.values()]
rows=[r for r in rows if r is not None]
print('n_activation_rows present on',len(rows),'entries; distinct',sorted(set(rows))[:8])
open('$OUT/gate.txt','w').write('PASS' if not d_ship else 'FAIL')
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

# --- D: land the branch and relaunch the full run as v2 -------------------
if [ "$(cat "$OUT/gate.txt" 2>/dev/null)" != "PASS" ]; then
  log "IDENTITY GATE DID NOT PASS -- not merging, not relaunching."
  log "DONE (gate failed) -> $OUT"; exit 1
fi
log "identity gate PASS; fast-forwarding dsv4/flash-0731-92gb"
git -C /home/rob/prismaquant-ultraplan merge --ff-only perf/cb-encode-gpu \
  >> "$OUT/bench.log" 2>&1 || { log "MERGE FAILED"; exit 1; }
git -C /home/rob/prismaquant-ultraplan log --oneline -3 >> "$OUT/bench.log"

V2=$RUN/prod-cal-0p6-v2
mkdir -p "$V2/artifacts" "$V2/work-prod" "$V2/logs"
log "relaunching full 43-layer cost run as pq-dsv4-cost-prod2 -> $V2"
docker run -d --name pq-dsv4-cost-prod2 --gpus all --ipc=host --entrypoint bash \
  -v "$RUN":"$RUN" -v /home/rob/prismaquant-ultraplan:/pq \
  -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CB_CODEBOOK_SOURCE=lattice -e CB_SCALE_CODING=two_tier \
  -e CB_SCALE_SWEEP=1 -e PRISMAQUANT_CB_ENCODE_TIER=balanced \
  -e PYTHONPATH=/pq -e PRISMAQUANT_CB_EXT_DIR=$RUN/ext \
  -e PRISMAQUANT_CB_COL_WEIGHTS=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl \
  -e PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl.provenance.json \
  gridbook:test -c "
python3 -m prismaquant.incremental_measure_quant_cost \
  --model $RUN/source --cost-mode local \
  --probe $RUN/prod-cal-0p6/artifacts/probe.pkl \
  --activation-cache-dir $RUN/prod-cal-0p6/act \
  --formats 'NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36,BF16' \
  --output $V2/artifacts/cost_full.pkl --work-dir $V2/work-prod \
  --device cuda --dtype bf16 --mode batched --chunk-size 256 \
  --layers-per-shard 1 --start-layer 0 --end-layer 43 \
  --skip-missing-activations --no-include-lm-head \
  > $V2/logs/cost_prod2.log 2>&1
" >> "$OUT/bench.log" 2>&1
sleep 20
docker ps --filter name=pq-dsv4-cost-prod2 --format '{{.Names}} {{.Status}}' \
  | tee -a "$OUT/bench.log"
log "prod2 launched; log = $V2/logs/cost_prod2.log"
log "DONE -> $OUT"
