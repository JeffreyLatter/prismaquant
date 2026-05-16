# HDQ + NVFP4 W4A4 — Investigation Handover (2026-05-14)

## TL;DR

The HDQ joint-search solver was optimizing the wrong objective. Fixed.

- Added `_compute_cluster_loss_w4a4` (joint W+A STE quantization loss) in
  `prismaquant/hadamard_duquant.py:866`.
- Made W4A4 the default solver loss (`run-pipeline.sh:139`).
- Added five explicit knobs: solver-loss, solver-init, solver-weight-decay,
  solver-multi-init, solver-n-random-probes. All wired through CLI + env.
- Validated per-cluster on Qwen3.5-0.8B and Qwen3-4B; predicted aggregate
  reduction in Fisher-weighted output MSE: **−2.06% on Qwen3-4B**, peak
  cluster gain **−23.3% W4A4** (residual stream).
- NOT YET validated: end-to-end model quality (vLLM serve + PPL/KL).
  That is the next concrete experiment.

## The bug, in one paragraph

PrismaQuant ships NVFP4 as **W4A4** (per
`export_native_compressed.py:4501` — the `NVFP4_SCHEME` includes an
`input_activations` spec at G=16). The HDQ solver was scoring rotations
under a W-only STE loss `||x (W - Q_w(W M^T) M)^T||²` with un-quantized
BF16 activations. DuQuant's mechanism (rotation reshapes per-G-block
activation distributions, reducing the activation quantization error)
was invisible to the solver. Sylvester-init regressed at the W-only
score because its activation-side benefit wasn't measured; we then
falsely concluded "rotations don't help NVFP4."

User pushed back citing MXFP4 + DuQuant successes; I consulted Codex
which surfaced both the loss-mismatch hypothesis and two papers — DuQuant++
(arXiv:2604.17789) and MR-GPTQ (arXiv:2509.23202, ICLR 2026, "rotations
hurt NVFP4 RTN at small group size" — confirms the regime). A
forward-only geodesic sweep
(`tools/w4a4_geodesic_sweep.py`) confirmed the W4A4 loss has minima at
non-identity rotations on some clusters that the W-only loss can't see.

## Validation chain

1. **Geodesic sweep (forward-only, no solver)**: For each cluster,
   evaluate three losses (W-only / A-only / W4A4) along R(t) =
   orthogonalize((1−t) I + t R_init) for t ∈ [0,1]. On Qwen3.5-0.8B,
   peak W4A4 reduction was −1.13% (Sylvester at t=0.2 on v_o layer 7).
   On Qwen3-4B, **peak W4A4 reduction was −8.80%** (svd_v at t=0.7,
   residual layer 1) and full Sylvester at t=1.0 won on down_proj
   layer 2 with **−7.89% W4A4**. Same fix, much bigger payoff at scale.
2. **End-to-end 0.8B pipeline** (`/dq-runs/qwen35-0p8b-hdq-w4a4-20260514T054331Z`):
   Identity-init + W4A4 loss, 50/60 clusters benefit, predicted −1.45%
   Fisher-MSE. Exported artifact has 17 deduped rotation tensors baked
   into `transforms_config`.
3. **End-to-end 4B pipeline** (`/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z`):
   109/144 clusters benefit, predicted **−2.06% Fisher-MSE**. Peak
   residual cluster: **−23.30% W4A4**. 19 rotation tensors baked.
4. **Outlier suppression confirmed where the gain is real**: residual
   layer 1 mlp (Δw4a4 = −23.3%) shows **Δpeakiness = −11.91%** — exactly
   DuQuant's mechanism. Mid-tier clusters: modest peakiness reduction.
   No-gain clusters: Adam correctly stays at identity.

## Files changed

| file | change |
|---|---|
| `prismaquant/hadamard_duquant.py` | added `_compute_cluster_loss_w4a4`; `loss_kind`, `weight_decay` params on `solve_cluster_rotation`; new `init_strategy` values `sylvester_t<frac>` and `givens_balance` and `svd_v` |
| `prismaquant/joint_hadamard_format_search.py` | added `solver_loss`, `solver_weight_decay`, `solver_multi_init`, `solver_n_random_probes` to `search_cluster` and `run_joint_search`; probe selection is argmin over **production-renderer** score (not in-solver STE-loss) for consistency with sidecar reporting |
| `prismaquant/run_joint_hadamard_search.py` | CLI flags `--solver-loss`, `--solver-init`, `--solver-weight-decay`, `--solver-multi-init`, `--solver-n-random-probes` |
| `prismaquant/run-pipeline.sh` | env knobs `HADAMARD_DUQUANT_SOLVER_LOSS` (default `w4a4`), `HADAMARD_DUQUANT_RANDOM_PROBES` (default `0`) |
| `tools/w4a4_geodesic_sweep.py` | new — forward-only diagnostic that sweeps R(t) along I→R_init geodesic for three loss flavors |
| `tools/random_orthogonal_diagnostic.py` | new — basin-stuckness diagnostic; runs identity-init Adam + N Haar-uniform random-init Adams per cluster, reports how often random beats identity |
| `tools/_outlier_compare.py` | new — measures per-G-block peakiness before/after rotation, confirms whether rotation suppresses outliers |
| `docs/hdq_nvfp4_w4a4_loss_2026-05-14.md` | full investigation writeup |
| `memory/hdq_nvfp4_w4a4_loss_mismatch.md` | persistent project memory entry |
| `memory/feedback_no_heuristics.md` | persistent user-feedback entry |

## Path C exploration (and what landed)

User asked whether scaling-Sylvester guessing could be replaced with
data-inferred optima. Tested four alternatives:

1. **Multi-init basin search** (try {identity, sylvester_t0p3,
   sylvester_t0p5, svd_v} per cluster, pick W4A4 winner): aggregate
   Fisher-MSE on 0.8B = −1.30% vs identity-only −1.45%. **Worse** by
   0.15pp due to STE-loss vs renderer-score disagreement in the
   selection criterion (later fixed; see below).
2. **Weight decay on A_skew**: implemented as opt-in
   (`solver_weight_decay`). Identity-init Adam already produces small
   rotations (median ||M−I||=0.21 on 4B); data doesn't show overfit
   symptoms warranting lessening. Kept as a knob.
3. **Givens-balance constructive init** (closed-form per-pair angles
   from per-cluster GtG with explicit cost-reduction test, no heuristic
   threshold): on outlier-heavy synthetic, Adam-from-identity beats
   Givens-balance-Adam (+2.86% vs −0.00%). Pre-committing to a
   structured rotation constrains Adam's refinement direction. Kept as
   an option, not recommended.
4. **Channel permutation (cross-G)**: not implemented; would need a
   compressed-tensors Permutation transform or vLLM forward_pre_hook.
   The empirical ~2% Fisher-MSE ceiling at current scale doesn't
   justify the kernel/runtime work.

## Adaptive probe — implemented, partially validated

User requested an "adaptive probe": per-cluster, run identity-init
Adam + N random-init Adams, commit argmin over W4A4 scores. Landed:

- `solver_n_random_probes` parameter threaded end-to-end.
- Selection criterion is **production-renderer score**, not in-solver
  STE loss (the two can disagree; renderer matches what the sidecar
  reports). This is the principled fix to the multi-init mismatch.
- Argmin on shared calibration loss; no thresholds.

**Empirical results at 1k calibration (0.8B)**:

| config | Σ baseline | Σ with rot | picks | Δ% |
|---|---|---|---|---|
| identity (run A) | 1.367e-01 | 1.347e-01 | 50 | −1.451 |
| identity (rerun via v2 path) | 1.367e-01 | 1.346e-01 | 51 | −1.532 |
| probe v1 (STE selection) | 1.367e-01 | 1.343e-01 | 51 | −1.711 |
| probe v2 (renderer selection) | 1.367e-01 | 1.348e-01 | 54 | −1.420 |

**Cross-run noise dominates.** Two identity-only runs disagree on
49 of 60 clusters (per-cluster diffs up to ±2.8pp; aggregate diff
~0.08pp). At 1k calibration tokens, Adam-STE doesn't converge to a
stable per-cluster optimum across CUDA non-determinism, so probe
selection signal is sub-noise.

The probe **is correctly implemented** (argmin within-run on the
production-renderer score is monotone by construction). To make it
useful, scale up calibration: the random-orthogonal diagnostic that
found the +6.5pp signal used 16×2048 = 32k tokens, not 4×256 = 1k.

## Diagnostics that informed the design

### Random-orthogonal diagnostic on Qwen3-4B (`/dq-runs/qwen3-4b-hdq-randdiag-20260514T121528Z`)

For 6 clusters (3 "winners" with big identity-Adam gains + 3 "losers"
where identity-Adam barely helps), ran 30 Haar-uniform random-init
Adams plus identity. Result:

| cluster | identity gain | best random | n random > identity | landscape |
|---|---|---|---|---|
| residual layer 1 mlp | −23.98% | **−30.48%** | **5/30** | **multimodal — identity stuck** |
| residual layer 2 mlp | −12.57% | −1.64% | 0/30 | dominant identity |
| residual layer 5 mlp | −1.14% | +6.65% | 0/30 | dominant identity |
| residual layer 3 attn | −0.42% | +4.16% | 0/30 | dominant identity |
| residual layer 6 mlp | −0.66% | +55.0% | 0/30 | dominant identity |
| down_proj layer 1 | 0.00% | +128.0% | 0/30 | dominant identity |

**5 of 6 clusters are single-basin; 1 of 6 is multimodal.** The W4A4
landscape is heterogeneous: most clusters have a clear dominant basin
reachable from identity, but the deepest-gain cluster (residual layer 1
mlp, Δ−23.98%) has a better basin (Δ−30.48%) that needs random
sampling to find.

This justifies the adaptive probe **in principle** but the production
solver's small calibration washes out the signal. See above.

### Outlier-suppression validation (`tools/_outlier_compare.py`)

On Qwen3-4B identity-W4A4 artifact, sorted by W4A4 reduction:

| cluster | ||M−I||_F | Δw4a4 | Δpeakiness |
|---|---|---|---|
| layers.1.mlp.residual | 1.36 | **−23.3%** | **−11.91%** |
| layers.2.mlp.residual | 0.82 | −16.4% | **−6.59%** |
| layers.5.mlp.residual | 2.06 | −14.2% | −1.83% |
| layers.1.attn.residual | 1.30 | −13.4% | −0.82% |

Big-gain clusters DO show classic DuQuant outlier suppression
(7–12% peakiness reduction). Mid-tier clusters: smaller. No-gain
clusters: Adam correctly stays near identity (||M−I||<0.1, median).

## Open questions / what's next

1. **End-to-end model quality**: never run. Required test = `vllm serve`
   the 4B exported artifact + measure PPL/KL on wikitext vs an
   identical-config no-HDQ baseline. This is the make-or-break
   validation. Cost: ~1-2 hours, no new code needed.
2. **Larger calibration**: 32k+ tokens vs current 1k. The probe's
   sub-noise signal at 1k should clear noise at larger N. Production
   relevance depends on whether you ship with larger calibration.
3. **Qwen3.5-4B-untied staging**: the 4B test used Qwen3-4B (tied
   embeddings), not the mainline Qwen3.5-4B-untied. Untied variant
   not staged locally; needs setup before any release-grade smoke.
4. **Adaptive probe default**: kept off (`HADAMARD_DUQUANT_RANDOM_PROBES=0`).
   Should remain off until calibration is large enough for the probe
   signal to clear cross-run noise.

## Artifact map (where things live)

- 0.8B identity-W4A4 sidecar:
  `/dq-runs/qwen35-0p8b-hdq-w4a4-20260514T054331Z/artifacts/hadamard_duquant_sidecar.json`
- 0.8B exported artifact:
  `/dq-runs/qwen35-0p8b-hdq-w4a4-20260514T054331Z/exported/`
- **4B identity-W4A4 exported artifact** (the candidate for end-to-end test):
  `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/exported/`
- 4B sidecar: `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/artifacts/hadamard_duquant_sidecar.json`
- Geodesic sweep (4B): `/dq-runs/qwen3-4b-hdq-sweep-20260514T055243Z/sweep.jsonl`
- Random-orthogonal diagnostic (4B): `/dq-runs/qwen3-4b-hdq-randdiag-20260514T121528Z/diagnostic.jsonl`
- Adaptive probe v1 (STE selection, 0.8B): `/dq-runs/qwen35-0p8b-hdq-probe-20260514T133912Z/sidecar.json`
- Adaptive probe v2 (renderer selection, 0.8B): `/dq-runs/qwen35-0p8b-hdq-probe-v2-20260514T135132Z/sidecar.json`
- Identity-only re-run (cross-run noise baseline): `/dq-runs/qwen35-0p8b-hdq-baseline-v2-20260514T140334Z/sidecar.json`

## Tools (one-shot diagnostics, not part of production)

| tool | purpose |
|---|---|
| `tools/w4a4_geodesic_sweep.py` | forward-only sweep R(t) ∈ {I → R_init}, measure A-only/W-only/W4A4 |
| `tools/random_orthogonal_diagnostic.py` | per-cluster Haar-random sampling; tells you whether identity-Adam is stuck |
| `tools/_outlier_compare.py` | per-G-block peakiness/p99 max-abs before/after rotation |
| `/tmp/compare_loss_kinds.py` | compare two sidecars on rot-beats-norot count, per-kind median Δ |

## How to reproduce / re-run

End-to-end 4B with current defaults:

```bash
WORK_DIR=/dq-runs/qwen3-4b-hdq-XXX
docker run -d --gpus all --ipc=host --shm-size=8g \
  --user $(id -u):$(id -g) --name pq-4b-rerun \
  -v /home/rob/prismaquant:/work \
  -v /home/rob/.cache/huggingface:/hfcache \
  -v /home/rob/dq-runs:/dq-runs \
  -e HF_HOME=/hfcache -e HF_HUB_CACHE=/hfcache/hub \
  -e MODEL_PATH=/hfcache/Qwen3-4B \
  -e WORK_DIR=$WORK_DIR \
  -e DATASET=/dq-runs/calibration/diverse-v1.jsonl \
  -e NSAMPLES=4 -e SEQLEN=256 \
  -e TARGET_BITS=4.5 \
  -e TARGET_PROFILE=vllm_qwen3_5_packed_moe \
  -e PRODUCTION_CACHE_LEVERS=gptq,scale_sweep \
  -e PRISMAQUANT_NVFP4_SCALE_RULE=four_over_six_mse \
  -e HALO_MODE=off \
  -e HADAMARD_DUQUANT=1 \
  -e HADAMARD_DUQUANT_GROUP_SIZE=16 \
  # HADAMARD_DUQUANT_SOLVER_LOSS=w4a4   <- default
  # HADAMARD_DUQUANT_RANDOM_PROBES=0    <- default
  -e FISHER_OUTPUT_MSE_ALLOCATOR=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /work --entrypoint bash vllm-eugr-v020:latest \
  -c "bash /work/prismaquant/run-pipeline.sh 2>&1 | tee $WORK_DIR/pipeline.log"
```

Joint-search only (faster, ~10-30 min depending on model):

```bash
python3 -m prismaquant.run_joint_hadamard_search \
  --model /hfcache/Qwen3-4B \
  --sidecar-output $WORK_DIR/sidecar.json \
  --rotations-output $WORK_DIR/rotations.safetensors \
  --solver-loss w4a4 \
  --solver-init identity \
  --solver-n-random-probes 0 \
  --n-calib-samples 4 --calib-seqlen 256 \
  --dataset /dq-runs/calibration/diverse-v1.jsonl \
  --dtype bf16
```

## Key user feedback captured

- **"No heuristics when explicits exist"** (`memory/feedback_no_heuristics.md`):
  derive thresholds from the objective, not from arbitrary constants.
  Applied in adaptive-probe selection (argmin on actual renderer
  score, no significance threshold).
- **"Goal is to recover dynamic range and obviate clipping"**: the
  rotation work directly serves this. HDQ + W4A4 captures most of the
  outlier mass on clusters that need it without information loss.
  PrismaClip remains a separate (and currently disabled) clipping pass;
  HDQ doesn't formally replace it, but it addresses the same outlier
  problem from a lossless angle.

## Recommended next action

Run end-to-end PPL/KL test on the existing 4B exported artifact vs an
identical-config no-HDQ baseline. That answers the actual "does the
rotation deliver real model quality" question with the lowest
additional compute. If positive, accept and move on. If neutral, the
per-cluster Fisher-MSE win is too small to matter at the model level
and we should redirect effort to the higher-leverage problems
(channel permutation, larger calibration, other quant axes).
