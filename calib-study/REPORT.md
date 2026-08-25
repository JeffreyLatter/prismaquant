# Calibration Stability Study — DSv4-Flash-0731

## Verdict

**Concentration verdict: PASS, with a proxy qualification.** The motivating decomposition is reproduced exactly: the b-92 assignment has predicted loss `5376.606709145063`, of which the top 100 rows carry **93.6596%**, the top 500 **98.2164%**, and the top 1,775 **99.5871%**. Under the most faithful resample recoverable from the stored data—cached activation-energy subsets combined with binomial thinning of uncapped expert route counts—the top-100 identity at 50% calibration has Jaccard **0.9802 median** (0.9417–1.0000 5th–95th percentile), comfortably above the proposed 0.8 bar. The concentration itself stays at 93.66% median. On the evidence available, the hot set does not collapse when calibration is halved.

**Whole per-expert-calibration verdict: NOT decision-grade yet.** The stable hot set is not enough to validate unrestricted per-expert decisions: only **56.91%** of 11,008 expert units have at least eight routed tokens, exact CE-Fisher resampling is impossible from this cache, and no selected sample can be linked to a domain/source. The safe conclusion is: **the concentrated hot-set signal is robust to the observable activation/routing resample, but the current calibration is not decision-grade for arbitrary cold-expert allocation or composition claims.**

There is also an important correction to the motivating premise. Production `h_trace` was not estimated from the maximum 64 cached activation rows. It was accumulated by the probe over all routed tokens and normalized by the global 8,192-token calibration. The 64-row cap applies to the activation rows used for `output_mse`. The top 100 are not rare-route accidents: every one has at least 20 routed tokens, their median is 747.5, and 99/100 have 64 cached rows. The top 500 all have at least 11 routed tokens; 99.83% of the top 1,775 have at least eight.

## Exact baseline and data inventory

The study read only existing artifacts and ran with `CUDA_VISIBLE_DEVICES=''`; it did not invoke Docker or any CUDA API.

| Item | Observed |
|---|---:|
| Cost-table rows | 33,325 |
| Routed expert units | 11,008 = 43 layers × 256 experts |
| Activation-cache files | 27,820 |
| Missing/cold activation rows | 5,505 |
| Activation-cache bytes | 11,779,424,132 |
| Rows capped at 64 | 8,896 |
| Calibration | 16 samples × 512 tokens = 8,192 tokens |
| Router top-k | 6 |
| Fisher normalization | 8,192 global tokens |

The exact decomposition used the shipped b-92 format per qname and `0.5 × h_trace × output_mse`; source passthrough rows contribute zero. Its sum differs from `selection.json` by only floating-point summation error. Full paths, SHA-256s, the top-100 rows, and the cache histogram are in [data_inventory.json](data_inventory.json).

## 1. Estimator stability by row subsampling

The cache stores `{inputs, qname, row_indices}`. It does **not** store per-token Fisher contributions, output gradients, quantized per-row outputs, or per-row squared error. The closest honest quantities are therefore:

1. Activation proxy: per cached row, `mean(input² across features)`; per unit, the mean across a random fixed-size subset without replacement.
2. Route factor: `Binomial(n_tokens_seen, fraction) / (fraction × n_tokens_seen)`, using the probe's uncapped route count.
3. Assigned-loss proxy: the exact full-data assigned-row `0.5 × h_trace × output_mse`, multiplied by the activation-energy ratio and route factor.

Forty resamples were used at each fraction. This is a sensitivity analysis of the stored sufficient statistics, not an exact bootstrap of CE Fisher.

| Fraction | Activation-energy median RSD | Activation-energy p95 RSD | Combined route+energy median RSD | Combined p95 RSD | Loss-weighted combined RSD |
|---:|---:|---:|---:|---:|---:|
| 25% | 0.179% | 19.43% | 37.98% | 172.17% | 8.82% |
| 50% | 0.105% | 11.24% | 21.92% | 96.33% | 4.82% |
| 75% | 0.051% | 6.23% | 12.63% | 54.57% | 2.70% |

Decision implication: rare experts have large route-presence uncertainty, but rows that carry the predicted loss are much more stable. Conditional activation norm alone looks extremely stable and would be misleading if shown without routing. At 50%, its median RSD by stored rows falls from 0.294% at `n=2` to 0.200% at `n=8`, 0.165% at `n=16`, 0.106% at `n=32`, and 0.064% at the censored `n=64` group; p90 is much wider and is the more useful tail statistic.

A hard limitation is visible in the data: full-cache activation energy and production `h_trace` have Spearman **−0.3611** over activated positive-Fisher rows. Activation norm is not a substitute for the missing output-gradient Fisher factor. Consequently the RSD curve is useful for stress-testing decisions, not for attaching confidence intervals to production `h_trace`.

Data and plot: [estimator_fraction.csv](curves/estimator_fraction.csv), [estimator_by_n.csv](curves/estimator_by_n.csv), [estimator_fraction.svg](curves/estimator_fraction.svg), [estimator_stability.json](estimator_stability.json).

## 2. Hot-set identity and concentration stability

| Fraction | Top-100 Jaccard median (p05–p95) | Top-500 | Top-1,775 |
|---:|---:|---:|---:|
| 25% | 0.9417 (0.9048–0.9608) | 0.9305 (0.9082–0.9417) | 0.9325 (0.9241–0.9410) |
| 50% | **0.9802 (0.9417–1.0000)** | 0.9608 (0.9491–0.9685) | 0.9602 (0.9559–0.9657) |
| 75% | 0.9802 (0.9608–1.0000) | 0.9763 (0.9646–0.9841) | 0.9777 (0.9733–0.9822) |

The mass concentration is even more stable than identity. At 50% data, the median top-100/top-500/top-1,775 shares are 93.6599% / 98.2204% / 99.5891%, versus 93.6596% / 98.2164% / 99.5871% on full data.

Decision implication: the headline concentration clears the 0.8 top-100 criterion by a wide margin, including at the fifth percentile. A per-expert feature aimed specifically at the hot set has a real, data-robust target under this stress test. This does not authorize individual decisions for the cold tail.

Data and plot: [hotset_jaccard.csv](curves/hotset_jaccard.csv), [hotset_jaccard.svg](curves/hotset_jaccard.svg), [hotset_stability.json](hotset_stability.json).

## 3. Simplified allocation churn

The self-contained allocator performs per-row `argmin(dloss + λ × payload_bits)` and bisects λ to the b-92 body payload budget. It uses exact per-tensor payload formulas for K14, K15, K36, and BF16, excluding fixed shared codebook sidecars. Serving-atomic aggregation and exporter constraints are deliberately omitted, as requested.

Its full-data reference selects 31,300 K14 / 905 K15 / 998 K36 / 122 BF16 rows at predicted dloss 62.456. This is a stability reference, not a reproduction of the shipped solver: the shipped selection groups whole expert layers and has predicted dloss 5,376.607.

| Fraction | Format churn median (p05–p95) | Full-data dloss delta median (p05–p95) |
|---:|---:|---:|
| 25% | 0.632% (0.570–0.684%) | +0.696% (+0.481–+1.065%) |
| 50% | **0.366% (0.336–0.418%)** | +0.198% (−0.028–+0.391%) |
| 75% | 0.216% (0.180–0.243%) | −0.015% (−0.119–+0.269%) |

Decision implication: allocation is substantially more stable than the raw per-unit route RSD suggests. At half calibration, only about 122 of 33,325 rows change format at the median, and the full-data objective impact is about two-tenths of one percent. The small negative tail comes from discrete Lagrangian/budget underfill and should be read as “within solver granularity,” not as an improvement claim.

Data and plot: [allocation_churn.csv](curves/allocation_churn.csv), [allocation_churn.svg](curves/allocation_churn.svg), [allocation_churn.json](allocation_churn.json).

## 4. Coverage versus calibration size

Coverage uses `probe.stats[*].n_tokens_seen`, which is uncapped and identical across the three projections of every expert unit. This is better than inferring presence from the 64-row-capped activation cache.

| Gate | Current coverage |
|---|---:|
| ≥1 routed token | 83.33% |
| ≥8 routed tokens | **56.91%** |
| ≥64 routed tokens | 26.03% |

The population is extremely heterogeneous: mean routes/expert is 192, median is 12, and maximum is 8,192. A fitted Gamma-Poisson/negative-binomial model has dispersion `r=0.2093`. It predicts 23.98% zeros versus 16.67% observed and 60.01% at ≥8 versus 56.91% observed, so one NB is visibly imperfect.

Conditional on that no-structural-zero model, reaching 95% coverage at ≥8 requires a **20,787×** calibration multiplier, approximately **332,589 sequences × 512** (170.3M tokens). A layer-block bootstrap is enormously wide: 64,864 to 1,497,361 sequences at the 5th–95th percentiles. This is not a practical recipe recommendation; it is evidence that “just scale the same corpus” is not identified or economical. If more than 5% of experts are structural zeros for this composition, 95% is unattainable at any size. The current aggregate counts cannot distinguish structural zeros from extremely rare experts.

Decision implication: use an expert-coverage gate and targeted/composition-diverse calibration, not blind sample multiplication. The 64-row cache cap does not censor this coverage calculation because the probe retained uncapped route counts; it **does** censor activation/output-MSE convergence above 64 rows.

Data and plot: [coverage.csv](curves/coverage.csv), [coverage.svg](curves/coverage.svg), [coverage_model.json](coverage_model.json).

## 5. Composition

The actual recipe was:

- Local `/home/rob/dq-runs/calibration/diverse-v1.jsonl`.
- Loader seed 42 (the default); at seed 42 the loader does not reshuffle the corpus and takes random 512-token windows from the first qualifying records.
- 16 samples × 512 tokens = 8,192 exact calibration tokens.
- Corpus manifest: 256 text records targeting approximately 4,096 Qwen3.6 tokens each (about 1.05M target tokens total), shuffled at build seed 20260509.
- Intended aggregate mix: 40% prose, 20% code, 20% math, 20% multilingual. The manifest pins FineWeb-Edu; StarCoderData with a GitHub-code fallback; Proof-Pile-2 with Open-Web-Math fallback; and FLORES/XNLI fallback sources.

The calibration directory also contains `xdom-fit-v1.jsonl` (400 records) and `xdom-gate-v1.jsonl` (116 records), but this probe did not use them.

No valid domain split is recoverable. Every corpus record is only `{"text": ...}` after the builder shuffles and strips domain/source, while activation rows store only a qname and a Linear-local row index. Thus routing differences by domain, cross-domain `h_trace` rank correlations, and domain-unique cold experts cannot be computed without guessing.

Future probes must record stable `sample_id`, `source_id`, and `domain` on every corpus record; propagate sample/sequence/global-token identifiers into activation rows and expert-route records; and store per-token Fisher contributions plus per-format per-row output errors. The exact requested list is in [composition.json](composition.json).

## Calibration recipe recommendation

Measured recommendations:

1. **Do not reduce the current 64-row activation cap for allocation work.** Half of the stored rows (effectively about 32 for capped hot units) already gives top-100 Jaccard 0.9802, while the RSD-by-n tail improves through 64. There is no evidence here for lowering the cap, and no evidence above 64 because it is censored.
2. **Require at least eight uncapped routed tokens before allowing an independent per-expert decision.** This is the study's prespecified coverage threshold. Only 56.91% currently pass globally, although 100% of the top 500 and 99.83% of the top 1,775 pass. Experts below the gate should inherit a serving-atomic layer/group fallback rather than receive a noisy bespoke format.
3. **Treat the 16×512 recipe as sufficient for hot-set discovery under the tested proxy, not for full expert coverage.** The top-100 stability gate passes; the 95%-at-eight coverage gate fails.

Judgment, explicitly labeled:

4. **Prefer adaptive, route-targeted additional calibration over a fixed enormous sample count.** Stop only when ≥95% of allocatable expert units have ≥8 uncapped routes *or* are explicitly classified as structural/ineligible and assigned the fallback. The parametric same-corpus extrapolation is too wide to set a credible fixed `NSAMPLES`.
5. **Make composition stratified and testable.** Keep the intended prose/code/math/multilingual mixture, add explicit per-domain minimum samples, and report per-domain routing coverage and shared-expert rank correlation before promoting per-expert allocation.
6. **For the next probe, store exact resampling sufficient statistics.** At minimum: per-token Fisher contribution by Linear/expert, per-row output squared error per candidate format, sample/source/domain tags, and route records. Without them, the central CE-Fisher stability question remains proxy-only.

## Limitations

- **Proxy versus production estimator:** the cache lacks output-gradient factors, so CE-Fisher `h_trace` cannot be recomputed. The negative proxy/Fisher rank correlation shows this is material.
- **Output-MSE gap:** stored cost tables contain only a full-row mean. Input energy scales it but does not reconstruct directional `X·ΔW` error.
- **64-row censoring:** 8,896 cost rows and 99% of the top 100 hit the cap; convergence beyond 64 is unobserved.
- **Conditional subsampling:** fixed-size activation subsets are conditional on cached rows; binomial route thinning models presence independently and cannot reconstruct sample-correlated router behavior.
- **Single corpus/run:** all conclusions come from one DSv4 checkpoint, one corpus draw, and one assignment.
- **Composition missing:** aggregate intended mix is known, selected-record domains are not.
- **Coverage extrapolation:** the NB estimate assumes stationary composition and no structural zeros; its poor fit and huge interval are reasons not to operationalize the point estimate.
- **Simplified allocation:** it deliberately omits serving-atomic grouping and exact exporter sidecars, so its churn is convergence evidence only.

## Reproduction and checks

```bash
ionice -c 2 -n 7 nice -n 10 \
  env CUDA_VISIBLE_DEVICES='' \
  python3 calib-study/calibration_stability.py \
  --output calib-study --repeats 40 --seed 20260803

CUDA_VISIBLE_DEVICES='' pytest -q tests/test_calibration_stability_study.py
```

The test suite covers fixed-size subsampling on synthetic known distributions, Jaccard, and allocation churn. Machine-readable acceptance metrics are in [summary.json](summary.json).
