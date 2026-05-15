# PrismaQuant

**Mixed-precision quantization for large language models, selected on real, end-to-end KL.**

PrismaQuant generates per-Linear format assignments cheaply with surrogate cost models (Fisher-weighted MSE under a multi-choice knapsack), then **selects the shipping artifact by measuring real KL on a held-out calibration split**. The output is a standard `compressed-tensors` checkpoint that vLLM serves natively — no patches, no custom kernels, no forked runtime.

```
vllm serve $WORK_DIR/exported --quantization compressed-tensors
```

---

## Headlines

**Qwen3.6-27B (PrismaSCOUT) — 11% smaller, 68% lower KL than the previous PrismaQuant ship.**

| Artifact | Size | bpp | Held-out KL ($8\!\times\!512$) |
|---|---:|---:|---:|
| PrismaQuant v1 (5.5 bpp) | 22.67 GB | 5.50 | 0.0475 |
| **PrismaSCOUT** (5.31 bpp) | **20.17 GB** | **5.31** | **0.0151** |
| Change | **−2.5 GB (−11%)** | −0.19 | **−0.0324 (−68%)** |

Same source weights, same per-tensor toolkit (HALO, GPTQ damp sweep, scale sweep, block-output match). Only the selection routine changed. Public artifact: [`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm) (DOI `10.57967/hf/8656`).

**Production-faithful polish — provisional, calibration re-measurement in flight.**

A second selection-only upgrade evaluates candidate format flips against the export-aligned per-Linear weight path (joint NVFP4 sibling-coherent input global scales, GPTQ reconstruction, scale sweep, and calibrated activation clip; block-output match and HALO remain on the export-only side). On Qwen3.6-27B it drops the polish-time KL from `0.0151` to `0.0054` at `5.39` bpp, on a matched 2×128 token calibration. **A re-measurement on the larger 8×512 split is in flight; treat the 0.0054 number as a polish-time signal until that completes.**

**Qwen3.6-35B-A3B at 4.75 bpp — wins 8 of 9 zero-shot metrics vs uniform NVFP4.**

| Task | BF16 | **PrismaQuant** | RedHatAI NVFP4 | Δ vs RedHat |
|---|---:|---:|---:|---:|
| arc_easy | 81.23 | **80.72** | 77.61 | **+3.11** (2.6σ) |
| arc_challenge | 54.86 | **54.35** | 51.79 | **+2.56** |
| piqa | 82.21 | **81.94** | 80.79 | **+1.14** |
| hellaswag (norm) | 83.47 | **82.91** | 82.21 | **+0.70** |
| winogrande | 75.69 | **73.48** | 70.80 | **+2.68** |

Mean Δ vs BF16: **−0.56 pp** for PrismaQuant, **−2.21 pp** for uniform NVFP4 (~4× closer to BF16). Ships 2 GB smaller. The over-aggression failure mode of uniform NVFP4 — collapsing the ~5% of genuinely sensitive Linears — shows up directly in numbers.

---

## Quick start

```bash
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export WORK_DIR=./dq-runs/qwen36
export FORMATS=NVFP4,MXFP8_E4M3,BF16
export TARGET_BITS=4.75

./prismaquant/run-pipeline.sh
```

Runs `probe → cost → allocator → native export` end-to-end; produces a `compressed-tensors` checkpoint at `$WORK_DIR/exported/`. Serve:

```bash
vllm serve $WORK_DIR/exported \
  --quantization compressed-tensors \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

For models too large to fit in RAM (200B+ MoE), PrismaQuant has a streaming layer-by-layer path that keeps peak memory bounded — no full-model load is ever required. Used in production for MiniMax M2.7 (228B) and DeepSeek-V4-Flash (671B).

For PrismaSCOUT-style validated-frontier selection on a probed model:

```bash
python -m prismaquant.iterate_perturbed_allocation --help
```

For production-faithful polish around a chosen assignment:

```bash
python -m prismaquant.polish_from_assignment \
  --model /path/to/source --payload payload.json \
  --assignment chosen_knee.json --output polished.json \
  --production-weight-cache prod.pkl \
  --delta-quantize \
  --n-calib-samples 8 --calib-seqlen 512
```

---

## How it works

Mixed-precision quantization decomposes naturally into two questions: **how should each Linear be rounded** (per-tensor toolkit — GPTQ, AutoRound, HALO, scale sweep) and **how many bits should each Linear get** (allocator). The first question is well-studied; the second is where PrismaQuant operates.

The classical answer is to assign a per-Linear sensitivity score and pack a multi-choice knapsack under a total-bit budget. This is the v1 PrismaQuant pipeline:

$$\Delta\mathrm{loss} \approx \tfrac{1}{2} \cdot H_\mathrm{trace} \cdot \mathrm{MSE}_W$$

`H_trace` is the empirical Fisher diagonal trace (one calibration pass), `MSE_W` is the measured per-format round-trip error on the actual weights, and the knapsack is solved in seconds. **Each bit goes where it buys the most likelihood.** For MoE, the knapsack lifts naturally to `(format, dropped_expert_ids)` — joint expert pruning + quantization is a first-class operation, not a post-processing pass.

The structural problem with that pipeline, observed across the mixed-precision allocation literature: per-Linear sensitivity scores are **biased estimators of joint quantization error**. When the allocator commits many flips at once, the additive surrogate's predicted loss systematically overshoots the measured KL by 30–50%. **CLADO** (Deng et al. 2023, [arXiv:2307.05657](https://arxiv.org/abs/2307.05657)) is the foundational treatment of this: it measures the residual pairwise quantization-error coupling between Linears on a small data subset and solves the resulting integer quadratic program directly. HAWQ-V3 uses second-order ILP; CoopQ takes a cooperative-game view. PrismaSCOUT takes a different tack:

> **Surrogates generate, real KL selects.**

PrismaSCOUT keeps the additive surrogate as a cheap candidate generator but routes every shipping decision through a real, end-to-end KL measurement on a held-out calibration split. The selection algorithm is a multi-level cost cascade:

- **L1 — probe.** Fisher-weighted MSE per `(Linear, format)`. Solve additive DP at the target budget. CPU-seconds.
- **L2 — perturbed-X fixed point.** Install activation hooks under the L1 assignment, cache calibration activations, re-measure per-`(Linear, format)` MSE under the perturbed activation distribution, re-solve the DP. Iterate to weighted-Hamming convergence. ~3 passes.
- **L3 — propagated end-KL.** Select a bounded neighborhood of uncertain Linears, measure paired BF16/candidate end-KL on each, solve a frozen DP over the L3 measurements at the budget.

A **validated-frontier kneedle** runs the cascade at multiple anchor budgets, validates each candidate on the held-out split, filters by `η`-dominance, and selects the elbow on measured `(bpp, KL)`. A **monotone coordinate-descent polish** then perturbs the chosen assignment locally and accepts only flips that strictly improve real KL — provably non-regressive on the polish-time evaluator.

The **production-faithful polish** is the most recent refinement: instead of evaluating polish flips on RTN-quantized proxy weights (which the v1 polish did), it evaluates on the export-aligned per-Linear weight path (joint NVFP4 sibling-coherent input global scales, GPTQ, scale sweep, calibrated activation clip; block-output match and HALO remain export-only). The polish move set is a set of **Block-CLADO decision units** — fused-sibling Linears (e.g., `q/k/v` in attention, `gate_up/down` in MoE) grouped into atomic flip targets, following the structural framing of CLADO (Deng et al. 2023). A delta-quantize `WeightSession` swaps one decision unit's weight in place per trial instead of cloning the model, which makes the polish tractable on a 27B model under a 121 GB UMA budget.

For full method derivations, the `_GradNormCapture` MoE Fisher estimator, the L3 paired-baseline construction, the calibration-disjointness discipline, and the rejected detours we considered (Lagrangian λ-bisection, sandwich proximal recalibration, block-DP over architectural cliques, sparse pairwise QUBO, top-K Hessian covering): see [`paper/main.pdf`](paper/main.pdf) and the source comments in `prismaquant/`.

---

## Pipeline

```
sensitivity_probe ──► probe.pkl     (Fisher H_trace per Linear + per-expert REAP saliency)
        │
measure_quant_cost ─► cost.pkl      (per-(Linear, format) MSE — L1)
        │
iterate_perturbed_allocation ─► validated_frontier.json   (L2 fixed point + L3 propagated KL,
        │                                                  validated kneedle, leave-one-out guard)
        │
polish_from_assignment ─► polished.json   (production-faithful polish from a low-bpp floor
        │                                  with delta-quantize WeightSession)
        │
export_native_compressed ─► exported/     (compressed-tensors checkpoint)
        │
validate_native_export   ─► vLLM forward + greedy decode + perplexity gate
```

For models that don't fit in RAM, the probe and cost stages run in **incremental streaming mode**: layers are loaded from disk one at a time, hooked, measured, unloaded. Peak memory is bounded by `~1 layer + a tunable cache`. Multi-chunk calibration (run probe N times across calibration shards, merge) lets you trade wall time for signal.

---

## Supported formats

| Family | Formats |
|--------|---------|
| NVIDIA microscaling | NVFP4, NVFP4A16 |
| MX (Open Compute) | MXFP4, MXFP6_E3M2, MXFP6_E2M3, MXFP8, MXFP8A16 |
| Integer | INT8_W8A16, INT4_W4A16_g128 |
| Native passthrough | BF16, FP8_SOURCE (preserves natively-FP8 source weights byte-exact) |

Hardware support:

|              | Blackwell (SM100+) | Ampere/Ada | vLLM serving today |
|--------------|:------------------:|:----------:|:------------------:|
| NVFP4        | ✓ (CUTLASS)        | Marlin emu | ✓                  |
| MXFP4        | ✓ (CUTLASS)        | Marlin emu | ✓                  |
| MXFP6        | ✓ (native)         | —          | ✗ (kernel pending) |
| MXFP8 / FP8  | ✓ (CUTLASS)        | ✓          | ✓                  |
| INT4 / INT8  | all NV             | all NV     | ✓ (Marlin)         |

Recommended bundle for shipping today: `--formats NVFP4,MXFP8_E4M3,BF16`. The allocator is constraint-aware: it never picks a format vLLM can't serve.

---

## Supported architectures

First-class profiles ship today:

- **Qwen3.5 / Qwen3.6** (dense + packed-3D MoE + MTP heads)
- **MiniMax M2 / M2.7** (nested per-expert MoE, native FP8 source)

Active integration:

- **DeepSeek-V3 / V3.1**
- **DeepSeek-V4-Flash** (waiting on `transformers` class)
- **GLM-4**

Adding a new architecture is a `model_profiles/` registration: declare the layer module path, the MoE structure (nested vs packed), the fused-sibling groups, and any pre-staging quirks. Most architectures land in 100–200 LoC.

---

## Status

Active development. The 27B PrismaSCOUT ship is the current public artifact. Active workstreams:

- **Production-faithful polish on 27B** — export of the 5.39 bpp polished artifact in flight at time of writing; downstream task evals (validator perplexity, GSM8K, IFEval, MMLU, tool-eval-bench) and 8×512 KL re-measurement queued.
- **MiniMax M2.7 at ~90 GB on Spark** — v22 throughput optimizations landed; probe + cost in flight.
- **DeepSeek-V4-Flash** — blocked on transformers `DeepseekV4ForCausalLM`; mirror flow ready.
- **Per-channel Fisher + per-channel weight MSE** — research; preserves the knapsack's optimal substructure at <10 MB extra storage per 35B model.

The full paper draft is at [`paper/main.pdf`](paper/main.pdf) — includes the methodology section on rejected detours (Lagrangian λ-bisection, sandwich proximal recalibration, block-DP, sparse pairwise QUBO, top-K Hessian covering, surrogate-only knee, probe-only knee predictor) and an honest accounting of downstream regressions on the shipped 27B artifact.

---

## Why PrismaQuant beats stronger algorithms with weaker scope

A common reaction: "AutoRound is a better rounding algorithm — why does PrismaQuant win?" Because that's the wrong comparison. AutoRound is a single-format rounder; PrismaQuant operates one level up.

PrismaQuant is a **format allocator** that composes on top of any rounding algorithm. The `FormatSpec` for each format carries its own `quantize_dequantize` function — drop in AutoRound's sign-gradient-descent rounding for the integer formats and you still get per-Linear mixed-precision selection on top. The bit budget goes farther at the same Pareto point, regardless of which rounding strategy fills each Linear.

The headline result against RedHatAI's `Qwen3.6-35B-A3B-NVFP4` (a uniform NVFP4 quantization with 342 hand-picked BF16 ignores) makes this concrete: **PrismaQuant ships 2 GB smaller, with 90 fewer Linears in BF16, and wins 8 of 9 zero-shot metrics**. The 90-Linear gap is exactly what end-to-end measurement buys over guessing.

---

## Citation

```bibtex
@software{prismaquant2026,
  title  = {PrismaQuant: Mixed-Precision LLM Quantization Selected on Real End-to-End KL},
  author = {Tand, Robert},
  year   = {2026},
  url    = {https://github.com/RobTand/prismaquant},
}
```

The full paper draft is at [`paper/main.pdf`](paper/main.pdf).

## Acknowledgements

PrismaQuant builds on a decade of mixed-precision quantization research. The closed-form cost model, Fisher-diagonal sensitivity estimator, multi-choice knapsack formulation, and per-expert REAP saliency are assembled from published ideas. Selected key influences:

- **Cross-layer dependency in allocation (foundational)** — CLADO (Deng et al. 2023, [arXiv:2307.05657](https://arxiv.org/abs/2307.05657)) for the integer-quadratic-programming formulation over pairwise quantization-error coupling and the decision-unit framing PrismaQuant's Block-CLADO pipeline builds on
- **Mixed-precision allocation** — HAWQ-V1/V2/V3 (Dong et al. 2019–2021), CoopQ (Zhao et al. 2025), AMQ (Lee et al. 2025)
- **Post-training quantization** — GPTQ (Frantar et al. 2022), AutoRound (Cheng et al. 2023), AWQ (Lin et al. 2023)
- **Outlier handling** — SqueezeLLM (Kim et al. 2023), SpQR (Dettmers et al. 2023), SmoothQuant (Xiao et al. 2022)
- **MoE pruning** — REAP (Lasby et al. 2025) for the dropout-loss saliency formulation
- **Pareto-knee detection** — Kneedle (Satopaa et al. 2011)
- **Foundation** — Cover & Thomas, *Elements of Information Theory* (2006), Chapter 13 on rate-distortion bit allocation
- **Geometry-aware rounding** — Chen et al. 2026 (GPTQ as Babai's nearest plane algorithm)

Full bibliography in [`paper/main.tex`](paper/main.tex).
