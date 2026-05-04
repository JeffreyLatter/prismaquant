# PrismaSCOUT — Overview

PrismaSCOUT is the search-and-select algorithm at the heart of PrismaQuant. Where PrismaQuant is the full pipeline that turns a full-precision model into a quantized artifact, PrismaSCOUT is the routine that decides *which precision to spend on which weight matrix*.

The acronym: **S**urrogate-**C**ascaded **O**ptimization **U**nder **T**radeoff.

## From PrismaQuant v1 to PrismaSCOUT

PrismaSCOUT was always the goal — an allocator that selects on real end-to-end behavior, not on summed per-layer surrogates. We released the first PrismaQuant artifacts (built on the standard per-layer toolkit: HAWQ-style Hessian sensitivity, HALO, AutoRound, GPTQ, block-output matching, scale sweeps, rotations) to get the ball rolling and to see whether mixed-format quantization really had juice in it before pouring engineering effort into the harder pipeline. The community reaction made the answer clear: tens of thousands of downloads across the family in the first few weeks. That was the signal to commit to delivering on the original promise.

Each per-layer technique in the v1 stack is good at one job — take a single matrix and quantize it well. The v1 pipeline composed them per-layer and trusted that "each layer is individually well-quantized" would imply "the whole model is well-quantized." It usually did. But sometimes it didn't, and the failure mode was hard to predict from any single layer's metrics. A small perturbation in layer 7 — completely benign on its own — could compound through layers 8 through 40 and produce a noticeably worse model than the cost surrogate predicted. The reverse also happened: an allocation that *looked* expensive layer-by-layer turned out to be fine end-to-end, because the perturbations cancelled or got absorbed downstream. Neither effect was visible to a sum-of-per-layer-costs allocator. The per-layer view wasn't catastrophically wrong, but it was systematically biased: it overspent bits where local sensitivity was high but downstream impact was modest, and underspent bits where local sensitivity looked low but errors propagated badly through later attention layers.

PrismaSCOUT collapses the allocator design into a different shape: **surrogates generate, real measurement selects.** If you can measure the true loss for a candidate assignment, you no longer need a cost surrogate to make the final decision — you only need it to propose candidates cheaply. Cross-layer interactions and propagated effects stop being something you have to *model*. They become something you *observe*, automatically, every time you score a candidate. The previous PrismaQuant treated quantization quality as a sum of local effects and tried to allocate bits accordingly. PrismaSCOUT treats it as a single end-to-end signal and lets the search use that signal directly.

## About KL — the metric that decides what ships

KL divergence (short for Kullback-Leibler divergence) measures how different two probability distributions are. In quantization, the two distributions are: what the *original* full-precision model would predict at a given token, and what the *quantized* model predicts at that same token. Run the comparison across many tokens of held-out text and average. The result is a single number where 0 means the quantized model behaves identically to the original, and larger numbers mean it has drifted further.

KL is the natural metric for "did we preserve what the original model does?" because it captures the entire output distribution, not just the top guess. Two quantized models can both pick the same most-likely next token while producing very different distributions over the rest of the vocabulary — and that difference matters for sampling, instruction-following, and especially tool-use, where small shifts in token probability at decision points can flip a tool call or change an argument.

A rough sense of what KL values mean in practice for a competent base model:

| Setting | Typical KL |
|---|---|
| Same model, same weights, same input (sanity) | 0.00 |
| High-quality quant at 5–6 bpp | 0.01–0.05 |
| Medium quant at 4–4.5 bpp | 0.1–0.4 |
| Aggressive quant at 3 bpp | 0.5–2.0+ |

(Rough ranges from our own measurements; the quantization *method* matters as much as the bit-width.)

**Where PrismaSCOUT lands vs the previous PrismaQuant version on Qwen 27B:**

| Artifact | Size | bpp | Held-out KL |
|---|---|---|---|
| Previous PrismaQuant v1 | **22.67 GB** | 5.50 | **0.0475** |
| PrismaSCOUT | **20.17 GB** | 5.31 | **0.0151** |
| **Change** | **−2.5 GB (−11%)** | −0.19 | **−0.0324 (−68%)** |

Same source weights, same export-time quantization tricks. The new artifact is **11% smaller and produces a 68% lower KL divergence** from the original full-precision model. That's not a tradeoff — it's the same direction on both axes, and it's the direct empirical consequence of selecting on real end-to-end measurement instead of summed per-layer cost.

## How PrismaSCOUT works

PrismaSCOUT runs three escalating stages on the same problem:

- **L1 — cheap rank.** A coarse per-matrix loss surrogate ranks formats in milliseconds. The output is a rough budget-feasible assignment.
- **L2 — refined cost.** A more careful per-matrix measurement, still one matrix at a time, prunes obvious losers.
- **L3 — real evaluation.** The surviving candidates get rebuilt as actual quantized models, run against calibration text, and scored on how much their outputs drift from the original model. This is the slow step, but it sees the real artifact.

Each stage hands its top candidates to the next, so the expensive measurements only land on points that already look promising cheaply.

After L3, PrismaSCOUT does three things with the candidate set:

- **Filter to the Pareto frontier.** Drop any candidate where some other candidate is both smaller *and* better. What remains is the empirical efficient frontier.
- **Pick the knee on a held-out split.** Re-score frontier points on text the cost surrogates never saw. The kneedle algorithm runs on those out-of-sample pairs and selects the elbow.
- **Polish without regressing.** A coordinate-descent sweep tries small per-matrix perturbations around the knee. Any move that doesn't strictly improve real held-out KL is rolled back. The polished assignment is provably no worse than the chosen knee.

Under the hood: **L2** builds a sparse pairwise interaction model over per-layer format choices and solves it as a Lagrangian-relaxed QUBO — to our knowledge, the first time this game-theoretic decomposition has been applied at LLM scale for mixed-precision allocation. **L3** rebuilds candidate models in a small neighborhood around L2's converged point and re-solves using actual end-to-end KL. Sitting above the cascade, a **λ-sweep with one-pass Pareto archive DP** traces the achievable size-vs-quality frontier directly from L3 measurements: we deliberately abandon the traditional "fix a target bpp, pack to fit" formulation and let the kneedle pick the best point on the full frontier instead. The L3 measurement loop went through roughly half a dozen design rounds — fused NVFP4 Triton kernels, multi-lane CUDA graphs, a replay cache, aggressive memory management — before we landed on something with both respectable accuracy and tractable wall-clock at 27B+ scale.

## What it produces

A single layer-by-layer format assignment plus a full audit trail: every candidate measured, every dominance decision, the held-out KL of the chosen point, and a leave-one-out stability check confirming the knee isn't an artifact of one or two frontier points. The assignment is then exported by PrismaQuant and re-validated end-to-end on a real inference engine before it ships.

## Why it matters

PrismaSCOUT shipped a smaller, better Qwen 27B artifact from the same source weights and the same export-time tricks — only the selection routine changed. The lesson generalizes: per-layer quantization metrics are a useful proposal mechanism, but they are not the right shipping criterion. End-to-end held-out divergence is, and once you measure it, you should select on it.

## Relation to prior work

PrismaSCOUT was designed after surveying the modern literature on mixed-precision quantization. The relevant lineage includes HAWQ-V3 (2020) and the follow-on Hessian-sensitivity allocators; CLADO (2023, Cross-Layer ADmm Optimization) for explicit cross-layer error coupling; ParoQuant, IMPQ, and AMQ (2025) for various Pareto and budget formulations; the geometry-aware quantization line; and recent mixed-precision surveys.

The prior work converges on a recurring insight — per-layer surrogates underestimate cross-layer interactions — but treats it by building progressively more accurate *models* of those interactions: pairwise ADMM, integer programs over sensitivity matrices, structured Pareto solvers. PrismaSCOUT takes the opposite approach. Rather than modeling cross-layer effects more accurately, it directly *measures* end-to-end consequences for each candidate assignment and selects on the measurement. The L1/L2/L3 cascade exists precisely to keep that measurement cost manageable.

We adapted ideas from this lineage — much of which had not been run end-to-end at LLM scale before — and added a few new techniques of our own to make the measurement loop tractable on real models. The Qwen 27B comparison above replaces a previous PrismaQuant build that already incorporated HAWQ/HALO/AutoRound/GPTQ, i.e. a tuned per-layer state-of-the-art pipeline. PrismaSCOUT produces a smaller and meaningfully better artifact from the same weights and the same export tricks, and ships today as the production allocator inside PrismaQuant — with the candidate audit trail, leave-one-out stability check, held-out KL gate, and end-to-end serve validation that releasing public artifacts requires.
