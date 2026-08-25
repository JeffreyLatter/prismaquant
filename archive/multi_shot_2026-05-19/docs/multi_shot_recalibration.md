# Multi-shot Recalibration (Cheap Variant)

## Motivation

PrismaQuant's per-Linear cost is measured under BF16-upstream calibration
activations. Once the allocator commits a non-uniform assignment, every
downstream Linear sees activations that no longer match the ones its cost was
measured against — the local Taylor expansion the cost relies on is stale.
This is the same failure mode the archived `prismaclade_l3_non_additivity`
note describes: per-Linear costs measured under L2 context summed to the
wrong end-KL once L3 polish flipped many Linears simultaneously.

The LLM pruning literature (LLM-Surgeon, arXiv:2312.17244 §3.5) addresses
this by re-estimating per-Linear curvature `T` times around the partially
compressed operating point. PrismaQuant adopts the *cheap variant* of this
idea: refresh only the activation-conditioned cost numbers, keep the Fisher
diagonal trace and inner quantizer hyperparameters frozen.

## What gets refreshed shot-to-shot

| Quantity | Source | Refreshed? | Why |
|----------|--------|------------|-----|
| Per-Linear input activations | `multi_shot.recache_calibration_activations_for_cost` | **Yes** | This is the point of the loop. |
| Per-(Linear, format) `weight_mse` cost | `incremental_measure_quant_cost` | **Yes** | Cost recomputed against new activations. |
| Allocator assignment | `allocator` | **Yes** | Re-run on refreshed cost. |
| Fisher diagonal trace `H_trace` | `incremental_probe` | No | Costly to re-estimate; expected drift is small for 2-shot. |
| JSO clip grids (methodology) | `build_production_cache` | Unchanged | Same lever set, same calibration data, same BF16-forward — see "JSO semantics" below. |
| GPTQ damp value (methodology) | `build_production_cache` | Unchanged | Same reasoning. |
| Production weight cache (final) | `build_production_cache` after the loop | Yes | Built once against the final assignment. |

### JSO and damp semantics across shots

The intermediate per-shot production cache `build_production_cache` calls
**do re-execute** JSO and damp-sweep — they have to, since the per-shot
assignment differs and each Linear-format render is JSO-tuned. What stays
frozen is the **methodology**: the lever set (`PRODUCTION_CACHE_LEVERS`),
the calibration data + seqlen + sample count, and the BF16-forward used to
capture activations for JSO. The JSO clip grid that gets chosen for, say,
`fc2` under format NVFP4 at shot 1 is the same value that would be chosen
at shot 2 — the inputs to JSO have not changed, only the *set of Linears
being rendered* has. The intermediate-shot recomputations are wasteful in
that sense, but they are not "JSO unfreezing" — there is no rolling state
that could drift shot-to-shot.

This asymmetry is intentional and is the load-bearing cheapness of the
variant: only the cost step sees activations from the partially-quantized
upstream; JSO continues to be tuned against BF16 calibration just as in the
1-shot pipeline. A more expensive variant would also re-tune JSO under
quantized upstream; we expect the marginal win to be small relative to
implementation cost and are deferring it until the cheap variant shows
empirical signal.

## Usage

```bash
MULTI_SHOT_PASSES=2 \
MODEL_PATH=/models/Qwen3-4B \
WORK_DIR=./dq-runs/qwen3-4b-multi-shot-2 \
TARGET_BITS=4.5 \
SELECTION_MODE=surrogate \
./prismaquant/run-pipeline.sh
```

`MULTI_SHOT_PASSES=1` (default) reproduces the vanilla pipeline byte-for-byte.
Set `N > 1` to enable; the loop exits early if the assignment converges
(sha256 of `layer_config.json` matches the prior shot).

Intermediate artifacts land under `${WORK_DIR}/multi_shot/shot_<k>/`:

```
multi_shot/
├── shot_1/   # copy of the baseline allocator outputs
│   ├── layer_config.json
│   ├── pareto.csv
│   └── cost.pkl
├── shot_2/
│   ├── intermediate_prod_cache/      # rendered weights for shot_1 assignment
│   ├── intermediate_prod_cache.pkl
│   ├── act/                          # recached activations + metadata.json
│   ├── cost.pkl
│   ├── layer_config.json
│   ├── pareto.csv
│   └── logs/
└── shot_3/ ...
```

The final `${WORK_DIR}/artifacts/layer_config.json` always points to the
last shot's allocator output. The downstream production cache, recache, and
export stages run once at the end against that file.

## Restrictions (v1)

- `SELECTION_MODE=validated-surrogate` is not supported with
  `MULTI_SHOT_PASSES > 1`. Running both is a hard error: the validated-
  frontier path picks among Pareto candidates with measured KL, and we have
  not designed how multi-shot interacts with that selection. Use
  `SELECTION_MODE=surrogate`.
- The Fisher diagonal trace (`probe.pkl`) is not re-estimated. For
  `MULTI_SHOT_PASSES > 4` this likely drifts enough to matter; treat 2–3
  shots as the validated envelope until measured otherwise.

## Promotion gate

Currently **Research**. Promotes to **Candidate** only after:

1. Measured KL win on Qwen3-4B at ≥ 3 bpp budgets vs. the 1-shot baseline
   on the same calibration set.
2. The improvement clears the 1-shot vs. seed-noise floor (~40% of the
   between-seed variance observed during the L3-polish experiments — see
   `polish_overfit_2026_05_07.md`).
3. A 2-shot vs. 4-shot diminishing-returns check on at least one budget.

Per the design rules, any measured regression demotes back to Research.

## Empirical status (2026-05-18 → 2026-05-19) — NEGATIVE RESULT

Multi-shot recalibration was validated end-to-end on Qwen3-4B across both
small and production calibration sizes. **At production calibration (N=32
T=1024) the technique provides ΔKL = exactly 0 at every budget tested.**

### Production-calibration results (5/5 budgets)

| Budget | Baseline KL ± σ (4 reps) | Multi-shot KL ± σ | ΔKL |
|--------|---------------------------|--------------------|------|
| 4.70   | 0.13577 ± 0.036           | 0.13577 ± 0.036    | 0    |
| 4.85   | 0.13750 ± 0.023           | 0.13750 ± 0.023    | 0    |
| 5.00   | 0.09764 ± 0.005           | 0.09764 ± 0.005    | 0    |
| 5.50   | 0.10062 ± 0.024           | 0.10062 ± 0.024    | 0    |
| 6.00   | 0.09819 ± 0.014           | 0.09819 ± 0.014    | 0    |

All 5 budgets converge at shot 2 with the layer_config sha256 matching
shot 1's. The recalibrated cost numbers shift only ~0.6–1.2% on the
surrogate Δloss, which is below the DP's noise tolerance — the optimal
assignment is unchanged. Hence ΔKL is *exactly* 0, not "indistinguishable
from 0 within noise."

### What the small-cal Phase 0/1 measurement showed in retrospect

A preliminary run at N=8 T=512 × 4 repeats showed 3/3 directional wins
(mean ΔKL = -8.3%, best -16.6% at 6.0 bpp, z=-1.86). This *seemed* like
real signal at the time. The production-cal retest revealed it was noise
correction: at small calibration the per-Linear costs are noisy enough
that shot 1 picks a sub-optimal assignment; shot 2's recalibration pulls
it closer to the production-cal optimum. The KL "wins" were measured
against an artificially-bad baseline, not against a genuinely-optimal
reference.

### Why the hypothesis failed

The LLM-Surgeon §3.5-derived hypothesis was that per-Linear costs become
stale once many Linears change, and that iteration captures cross-Linear
interaction. The experiment shows the hypothesis does NOT hold for the
PrismaQuant allocator on Qwen3-4B with {NVFP4, MXFP8_E4M3, BF16} at
production calibration. The allocator's shot-1 pick is already a local
optimum that survives recalibration; the per-Linear cost surrogate
captures the dominant signal at production-cal accuracy.

This is consistent with `prismaclade_l3_non_additivity` in retrospect:
the L3-polish failure mode was about incremental refinement of an
already-optimal base assignment, not about whether the base allocator
itself benefits from recalibration. Multi-shot doesn't help where
L3-polish failed because they attack different problems.

### Where multi-shot might still help (untested, deferred)

1. **Larger models** (27B, 35B+) where cost noise may scale differently.
2. **Richer format menus** (adding NVINT3, MXFP6, …) — more decisions
   per Linear could surface interactions the 3-format menu does not.
3. **Different objectives** (perplexity vs. last-token KL) — KL may be
   insensitive to interactions perplexity catches.
4. **Different model families** with more correlated Linear sensitivities.
5. ~~**Calibration-efficiency claim**~~: a follow-up cross-evaluation
   tested whether small-cal + 2-shot recovers production-cal quality.
   It does not. Measured at production-grade eval (N=32 T=1024 × 4 reps):

   | Budget | Prod-cal base | Small-cal base | Small-cal 2-shot | Gap-to-prod closed |
   |--------|----------------|------------------|--------------------|---------------------|
   | 5.0    | 0.0976         | 0.1423           | 0.1314             | +24% |
   | 5.5    | 0.1006         | 0.1191           | 0.1474             | **-154%** (regression) |
   | 6.0    | 0.0982         | 0.1239           | 0.1235             | +2% |

   Mean gap-closed: -42.5%. The 95% assignment-similarity observation
   was misleading; the 5% of Linears that differ shift KL in
   unpredictable directions, and the small-cal 2-shot KL is on average
   *worse* than just running the small-cal pipeline once. **The
   calibration-efficiency hypothesis is also rejected.**

### Measurement-discipline lesson

The initial small-cal single-rep run showed a **+10% regression** at 5.5
that completely flipped to a **-5.2% win** with 4 calibration repeats.
Per-rep KL on Qwen3-4B at N=8 T=512 varies in a 0.04–0.25 range;
differential ΔKL across paired calibrations is much more stable. **Always
run `--calib-repeats ≥ 4`** for differential KL on this model scale.
A single-rep measurement at production calibration is also untrustworthy
for differential comparisons.

### Status vs. promotion gate

- (1) ≥3 budget wins: **0/5 at production cal** (3/3 at small cal,
  retracted as noise correction).
- (2) Clearing seed noise: trivially yes — there is no effect to clear.
- (3) 4-shot diminishing returns: moot if 2-shot already converges at
  shot 2.

Implementation stays **Research**. Do NOT promote to Candidate.

### Repro

Reusable orchestrator and analyzer:

```bash
# Orchestrator (parameterizable):
/home/rob/dq-runs/multi_shot_validate_orchestrator.sh
# Analyzer (reads kl_comparison.json or kl_repeats.json):
python3 /home/rob/dq-runs/analyze_multi_shot_validation.py <run_dir>
```

Three validation runs landed under `/home/rob/dq-runs/`:
- `multi-shot-validate-20260518T175439/`  (small cal, retracted signal)
- `multi-shot-validate-20260518T223633/`  (production cal, 3 budgets)
- `multi-shot-validate-20260519T012202-low-budget/`  (production cal, 2 budgets)

Each has a `comparison/analysis.{txt,json}` summarizing its measurement.

## References

- Frantar, Singh, Alistarh, *Optimal Brain Compression*, arXiv:2208.11580 — proves the OBS framework treats pruning and quantization as the same machinery, just substituting `(q(w)-w)` for `-w` in the saliency formula.
- van der Ouderaa et al., *The LLM Surgeon*, arXiv:2312.17244 — §3.5 introduces the `T`-shot recalibration that this implementation adapts.
- Yin et al., *Outlier Weighed Layerwise Sparsity (OWL)*, arXiv:2310.05175 — referenced as a complementary one-line layer-prior modulator; not implemented here, kept as a follow-up.
- Internal: `archive/cross_layer_2026-05-09/` for the prior cross-layer attempts (CLADO, propagated cost, L3 polish) that this work supersedes.
