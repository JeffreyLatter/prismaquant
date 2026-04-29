# prismaquant

**Mixed-precision quantization for LLMs that actually measures what it's doing.**

Each Linear layer gets a different format based on its measured sensitivity. The high-curvature paths stay in BF16 or FP8; the rest go to NVFP4 or below. The output is a standard `compressed-tensors` checkpoint that vLLM serves natively — no patches, no custom kernels, no forked runtime.

```
vllm serve $WORK_DIR/exported --quantization compressed-tensors
```

That's it. The artifact is just a checkpoint vLLM already knows how to load.

---

## What you get

| Headline result on Qwen3.6-35B-A3B | Source BF16 | prismaquant 4.75 bpp |
|---|---:|---:|
| Disk size | 70 GB | **22 GB** (-69%) |
| Format mix | 100% BF16 | 124 NVFP4 + 26 MXFP8 + 252 BF16 |
| MTP head | 1.7 GB | **0.5 GB** |
| vLLM serves natively | ✓ | ✓ |
| MTP speculative decode | ✓ | ✓ (n=3 draft tokens) |

Zero-shot benchmarks vs RedHatAI's uniform NVFP4 quantization at the same size:

| Task | BF16 | **prismaquant** | RedHat NVFP4 | Δ vs RedHat |
|---|---:|---:|---:|---:|
| arc_easy | 81.23 | **80.72** | 77.61 | **+3.11** |
| arc_challenge | 54.86 | **54.35** | 51.79 | **+2.56** |
| piqa | 82.21 | **81.94** | 80.79 | **+1.14** |
| hellaswag (norm) | 83.47 | **82.91** | 82.21 | **+0.70** |
| winogrande | 75.69 | **73.48** | 70.80 | **+2.68** |

prismaquant wins **8 of 9** zero-shot metrics vs uniform NVFP4. **Mean Δ vs BF16: -0.56 pp** for prismaquant, **-2.21 pp** for uniform NVFP4 (~4× closer to BF16). And ships 2 GB smaller.

The single sharpest result: **arc_easy +3.11 pp, 2.6σ statistically significant**. That's the over-aggression failure mode (uniform NVFP4 collapsing the ~5% of genuinely sensitive Linears) showing up in numbers.

## Quick start

```bash
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export WORK_DIR=./dq-runs/qwen36
export FORMATS=NVFP4,MXFP8_E4M3,BF16
export TARGET_BITS=4.75

./prismaquant/run-pipeline.sh
```

That runs `probe → cost → allocator → native export` end-to-end and produces a `compressed-tensors` checkpoint at `$WORK_DIR/exported/`. Then serve:

```bash
vllm serve $WORK_DIR/exported \
  --quantization compressed-tensors \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

For models too large to fit in RAM (200 B+ MoE), prismaquant has a streaming layer-by-layer path that keeps peak memory bounded — no full-model load is ever required. Used in production for MiniMax M2.7 (228 B) and DeepSeek-V4-Flash (671 B).

## How it works

Most quantizers fail in one of two directions:

1. **Over-preservation.** Tools without a sensitivity model leave large chunks in BF16 "to be safe." Result: a 5–6 bpp artifact that could have been 4.2–4.5 bpp at the same quality.
2. **Over-aggression.** The same tools, tightened, cast genuinely-sensitive Linears to NVFP4 anyway because they can't tell a 3-MMLU-point Linear from a 0.1-point one. Quality collapses before the bit savings are realized.

Both waste DRAM — the resource that gates batch size, context length, and how many models you can rotate on a box. prismaquant replaces those guesses with a closed-form per-Linear cost:

$$\Delta\mathrm{loss} \approx \tfrac{1}{2} \cdot H_\mathrm{trace} \cdot \mathrm{MSE}_W$$

where `H_trace` is the empirical Fisher diagonal trace (one calibration pass) and `MSE_W` is the measured per-format round-trip error on the actual weights. The allocator solves a multi-choice knapsack over those estimates under a total-bit budget. **Each bit goes where it buys the most likelihood.**

For MoE, the choice is `(format, dropped_expert_ids)` priced in the same knapsack — joint expert pruning + quantization is a first-class operation, not a post-processing pass.

## Pipeline

```
sensitivity_probe ──► probe.pkl   (Fisher H_trace per Linear + per-expert REAP saliency)
        │
measure_quant_cost ─► cost.pkl    (per-(Linear, format) MSE)
        │
allocator ◄─────────┘
   │
   ▼ layer_config.json, pareto.csv
   │
export_native_compressed ─► exported/   (compressed-tensors checkpoint)
   │
   ▼
validate_native_export   ─► vLLM forward + greedy decode
```

For models that don't fit in RAM, the probe and cost stages run in **incremental streaming mode**: layers are loaded from disk one at a time, hooked, measured, and unloaded. Peak memory is bounded by `~1 layer + a tunable cache`. Multi-chunk calibration (run probe N times across calibration shards, merge) lets you trade wall time for signal.

The current MoE pipeline supports nested per-expert Linears (MiniMax-style) and packed-3D experts (Qwen3.5/3.6 / Mixtral). MTP (Multi-Token Prediction) heads are quantized end-to-end and exercised via vLLM's speculative decoding at serve time.

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

## Supported architectures

First-class profiles ship today:

- **Qwen3.5 / Qwen3.6** (dense + packed-3D MoE + MTP heads)
- **MiniMax M2 / M2.7** (nested per-expert MoE, native FP8 source)

Active integration:

- **DeepSeek-V3 / V3.1**
- **DeepSeek-V4-Flash** (waiting on `transformers` class)
- **GLM-4**

Adding a new architecture is a `model_profiles/` registration: declare the layer module path, the MoE structure (nested vs packed), the fused-sibling groups, and any pre-staging quirks. Most architectures land in 100–200 LoC.

## Status

Active development on the public-ship pipeline. Recent work (this branch series):

- **v21** — opt-in throughput optimizations: deferred Fisher sync, direct-CUDA safetensors load, adaptive per-expert sampling with per-domain saliency, cross-chunk LayerCache retention, in-process cost step.
- **v22** — GPU-utilization fixes: deferred per-Linear Fisher compute (decouples matmul from autograd dispatch), async batched activation cache writes, lazy weight-stats caching, sticky expert freezing, phase-1 sync elimination. **Net: ~2.5× faster probe wall on a 228 B MoE at the same calibration scale.**

The next ship targets are **MiniMax M2.7 at ~90 GB on DGX Spark** (running now) and **DeepSeek-V4-Flash** (next).

## Roadmap

Active:

- **MiniMax M2.7 at 90–95 GB on Spark** — currently in probe; v22 branch.
- **DeepSeek-V4-Flash** — blocked on transformers `DeepseekV4ForCausalLM`; mirror flow ready.
- **Additional native vLLM formats** — add formats only when the runtime has native kernels and compressed-tensors loader support.
- **Size-targeting allocator mode** — `--target-bytes 24G --kv-budget 8192` instead of `--target-bits 4.75`. The allocator already computes the full Pareto curve; this is a thin wrapper to pick the knee matching a hardware constraint.

Research:

- **Per-channel Fisher + per-channel weight MSE.** The current cost proxy uses scalar `H_trace`. Lifting to per-channel preserves the knapsack's optimal substructure at <10 MB extra storage per 35 B model. Hypothesis: meaningful additional accuracy on attention Linears + `down_proj` where outlier channels are the norm.
- **MTP-acceptance-aware allocation.** Jointly optimize body quality and draft-token acceptance rate at the allocator level, rather than treating MTP Linears as ordinary body Linears.
- **Multimodal calibration.** Real Fisher stats on visual encoder blocks via image+text calibration pairs.
- **Sparse-outlier co-quantization.** SpQR / SqueezeLLM-style top-k outlier extraction, dense remainder more aggressive — push the 4.75 bpp frontier toward 4.0 bpp at constant quality.
- **Domain-targeted calibration.** Code, reasoning, multilingual cal-mixes that should Pareto-dominate generic ultrachat for task-specific deployments. (Per-domain expert saliency tracking landed in v21.)

## Why prismaquant beats stronger algorithms with weaker scope

A common reaction: "Intel AutoRound is a better rounding algorithm — why does prismaquant win?" Because it's the wrong comparison. AutoRound is a single-format integer quantizer; prismaquant operates one level up.

prismaquant is a **format allocator** that composes on top of any rounding algorithm. The `FormatSpec` for each format carries its own `quantize_dequantize` function — drop in AutoRound's sign-gradient-descent rounding for the integer formats and you still get per-Linear mixed-precision selection on top. The bit budget goes farther at the same Pareto point, regardless of which rounding strategy fills each Linear.

The headline result against RedHatAI's `Qwen3.6-35B-A3B-NVFP4` (a uniform NVFP4 quantization with 342 hand-picked BF16 ignores) makes this concrete: **prismaquant ships 2 GB smaller, with 90 fewer Linears in BF16, and wins 8 of 9 zero-shot metrics**. The 90-Linear gap is exactly what measurement buys over guessing.

## Method notes (briefly)

<details>
<summary><b>Why Fisher and not Hutchinson?</b></summary>

The Fisher diagonal trace is exact for the model's empirical loss at the calibration data and requires no stochastic estimator. Hutchinson's diagonal estimator on the Hessian gives a noisy unbiased approximation; for our use case (per-Linear ranking under a knapsack) the Fisher diagonal already satisfies the second-order Taylor expansion of cross-entropy around the current weights, and we'd be adding variance for no gain.

</details>

<details>
<summary><b>Why measured RTN error over analytical formulas?</b></summary>

Closed-form quantization-error formulas (uniform-distribution assumption, etc.) drift on real weight distributions, especially around outliers. Measuring `MSE_W` directly per (Linear, format) eats one calibration pass of compute but eliminates a class of model-dependent error that would otherwise need empirical calibration coefficients per architecture. The cost is small relative to the probe; the win is that the allocation is grounded in what the rounding algorithm actually does to *these* weights.

</details>

<details>
<summary><b>What about inter-layer interactions?</b></summary>

The `0.5·H·MSE_W` proxy is additive across Linears — it ignores second-order interactions between two Linears' quantization errors. This is the single biggest assumption in the cost model, and we have a follow-up path (`measure_interactions.py` + `quadratic_refine_allocator.py`) that performs a sparse pairwise KL probe near the Pareto knee and runs a local quadratic refinement.

In practice, the additive frontier matches measured KL within ~5% on Qwen-class models we've calibrated against, so the refinement only matters near very tight bit budgets. Open question: how this scales to 200 B+ MoE without a dense pairwise probe.

</details>

<details>
<summary><b>This is not gradient descent</b></summary>

prismaquant doesn't fine-tune anything. The allocation is a one-shot knapsack solve over measured per-Linear numbers — no QAT, no rounding-error backprop, no auxiliary loss. The probe runs once per (model, calibration set); the allocator runs in seconds against the resulting pickles; the export quantizes per the recipe. Reproducible, bisectable, and cheap to re-run with a different format bundle or target.

</details>

<details>
<summary><b>Methodological caveats — the proxy's failure modes</b></summary>

The closed-form `Δloss ≈ 0.5·H·MSE_W` rests on these assumptions:

1. **Local quadratic.** The loss is well-approximated by a 2nd-order Taylor expansion within the quantization perturbation. Fails when the perturbation is large enough to cross a non-smooth region of the loss surface (rare for ≤ 4-bit quantization on calibrated models; can show up at 2–3 bit budgets without a calibration sweep).
2. **Diagonal Fisher.** We use the trace of the diagonal, not the off-diagonal coupling. Per-channel curvature differences within a Linear are aggregated. Per-channel Fisher (on the roadmap) lifts this without changing the knapsack structure.
3. **Independent Linears.** Inter-Linear quantization errors don't compound. See "interactions" caveat above.
4. **Calibration coverage.** The Fisher diagonal is empirical over the calibration set. A calibration set that doesn't cover the target serving distribution (e.g., quantizing on ultrachat then serving on coding) leaves Fisher mismatched. Domain-targeted calibration is the v22 fix.

These caveats apply equally to most published mixed-precision allocators (HAQ, HAWQ, etc.); prismaquant inherits the same theoretical scope.

</details>

For full derivations, the `_GradNormCapture` MoE Fisher estimator, the MTP quantization path, and the joint expert-prune solver: see source comments in `prismaquant/{sensitivity_probe,allocator,allocator_prune}.py` and the design notes in `docs/`.

## Citation

```bibtex
@software{prismaquant2026,
  title        = {prismaquant: Mixed-Precision Quantization via Fisher-Weighted Bit Allocation},
  author       = {Tand, Rob and contributors},
  year         = {2026},
  url          = {https://github.com/RobTand/prismaquant},
}
```

## Acknowledgements

prismaquant builds on a decade of mixed-precision quantization research. The closed-form cost model, Fisher-diagonal sensitivity estimator, multi-choice knapsack formulation, and per-expert REAP saliency are assembled from published ideas. Selected key influences:

- **Mixed-precision allocation** — HAQ (Wang et al. 2019), HAWQ-V1/V2/V3 (Dong et al. 2019–2021)
- **Post-training quantization** — GPTQ (Frantar et al. 2022), AutoRound (Cheng et al. 2023), AWQ (Lin et al. 2023)
- **Outlier handling** — SqueezeLLM (Kim et al. 2023), SpQR (Dettmers et al. 2023), SmoothQuant (Xiao et al. 2022)
- **MoE pruning** — REAP (Lasby et al. 2025) for the dropout-loss saliency formulation we use
- **Pareto-knee detection** — Kneedle (Satopaa et al. 2011)
- **Foundation** — *Elements of Information Theory* (Cover & Thomas, 2006), Chapter 13 on rate-distortion bit allocation

For the full bibliography and the derivations underlying the closed-form allocator math, the `_GradNormCapture` MoE Fisher estimator, and the MTP quantization path: see source comments in `prismaquant/{sensitivity_probe,allocator,allocator_prune,mtp_module}.py` and the design notes under `docs/`.
