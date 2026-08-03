# DSv4 fused-rotation and LDLQ/CB pilot

Date: 2026-08-03

Branch: `feat/rotation-ldlq-pilot` (base `11c61b9`)

GPU: NVIDIA GB10, torch 2.11.0+cu130

Raw results: [`results.json`](results.json)

## Executive verdict

- **A — fused rotation: SKIP.** The best reproducible case is `layers.40.experts.81.up_proj` at K15, improving output SSE by 2.91% and 3.54% for seeds 0 and 1. Across all 18 seed/rung measurements the mean ratio is 1.0154 (a 1.54% regression), the range is 0.9646–1.1097, and only 2/18 clear a 3% win. Gini changes are small and mixed, so rotation does not reliably flatten the within-Linear tail or reliably reduce the tier-3 split prize. This is not enough to justify any serving fusion/folding work in a zero-online-transform system.
- **B — fixed-CB LDLQ feedback: NEEDS-MORE, strongly positive.** Every rung on every real Linear clears the 2–3% bar by a very large margin: calibration output SSE falls 90.7–97.5%. The incremental feedback assignment costs 1.14–2.09x plain assignment; with one Hessian build/factor amortized over the three rungs it costs 1.55–2.80x plain assignment. End-to-end prototype encode cost is only 1.10–1.43x the production encode because the already-optimized scale/codebook fit still dominates. However, `X` has only 64 rows (and the two expert Hessians are 4096-wide), so this is a rank-deficient in-sample objective with 1% damping. It is decisive evidence to advance a held-out test, not yet evidence to make feedback a production default.

The existing encoder is already optimized 2.18x. Paying the measured 1.10–1.43x total multiplier would retain approximately 1.52–1.99x of that speedup relative to the pre-optimization encoder, while requiring no serving transform and no format change.

## Contract and method

The three real harness tensors came from `/home/rob/dq-runs/dsv4-flash-0731/tier3-sample`: `layers.40.attn.wq_b` `[32768,1024]`, `layers.40.experts.81.up_proj` `[2048,4096]`, and `layers.20.experts.63.up_proj` `[2048,4096]`, each with 64 activation rows. The manifest's production context was reproduced exactly: `CB_CODEBOOK_SOURCE=lattice`, `CB_SCALE_CODING=two_tier`, `CB_SCALE_SWEEP=1`, `PRISMAQUANT_CB_ENCODE_TIER=balanced`, imatrix `mean(X.float()^2, dim=0)`, and the copied prebuilt extension cache at `/w/.ext`.

Experiment A calls PrismaQuant's `_cb_cost_quantize_dequantize` production entry. Experiment B calls the same underlying `nvfp4_cb_fields` path directly because it must retain the fitted fields, then proves the direct reconstruction and a plain fixed-field reassignment reproduce the saved production objective before measuring feedback. The recovered tier-3 `row_dispersion` library is reused unchanged for per-row SSE and Gini.

Commands:

```bash
cp -a /home/rob/dq-runs/dsv4-flash-0731/ext /w/.ext
export PRISMAQUANT_CB_EXT_DIR=/w/.ext PYTHONPATH=/w
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  scripts/rotation_ldlq_pilot.py --phase a --out rotpilot-out --ext-dir /w/.ext
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  scripts/rotation_ldlq_pilot.py --phase b --out rotpilot-out --ext-dir /w/.ext \
  --block-size 64 --damping-fraction 0.01
```

## A — fused randomized Hadamard rotation

For each seed, signs were drawn in float64 and a normalized fast Hadamard was applied on the right. Both the folded weight `W_rot = W @ H` and simulated fused activation `X_rot = X @ H` were stored in bf16 before the equivalence and quantization measurements.

### Serving-equivalence and baseline checks

| Linear | bf16 gap, seed 0 | bf16 gap, seed 1 | max baseline drift vs saved summary |
|---|---:|---:|---:|
| `layers.20.experts.63.up_proj` | 2.565740e-3 | 2.525157e-3 | 4.535e-10 |
| `layers.40.attn.wq_b` | 1.944192e-3 | 1.946319e-3 | 1.546e-10 |
| `layers.40.experts.81.up_proj` | 2.602070e-3 | 2.586240e-3 | 1.989e-9 |

All equivalence gaps are bf16-scale noise and below the fail-fast threshold of `5e-3`. Unrotated errors recomputed through the same per-row path match `tier3-sample/summary.json`; the largest relative drift is `1.99e-9`.

### Error ratio and tail dispersion

Ratios below 1 improve output SSE. `G0` is the unrotated Gini; `G1(s)` is the rotated Gini for that seed.

| Linear | K | ratio s0 | ratio s1 | G0 | G1(s0) | G1(s1) |
|---|---:|---:|---:|---:|---:|---:|
| `layers.20.experts.63.up_proj` | 12 | 1.0403 | 1.0467 | 0.6313 | 0.6133 | 0.6242 |
| `layers.20.experts.63.up_proj` | 15 | 0.9661 | 1.0059 | 0.6133 | 0.6274 | 0.6229 |
| `layers.20.experts.63.up_proj` | 18 | 1.0957 | 1.1097 | 0.6345 | 0.6082 | 0.6251 |
| `layers.40.attn.wq_b` | 12 | 1.0263 | 0.9867 | 0.4829 | 0.4807 | 0.4778 |
| `layers.40.attn.wq_b` | 15 | 1.0452 | 0.9993 | 0.4725 | 0.4776 | 0.4727 |
| `layers.40.attn.wq_b` | 18 | 1.0233 | 0.9970 | 0.4706 | 0.4708 | 0.4702 |
| `layers.40.experts.81.up_proj` | 12 | 1.0017 | 0.9764 | 0.3500 | 0.3538 | 0.3466 |
| `layers.40.experts.81.up_proj` | 15 | **0.9709** | **0.9646** | 0.3541 | 0.3486 | 0.3441 |
| `layers.40.experts.81.up_proj` | 18 | 1.0108 | 1.0112 | 0.3509 | 0.3598 | 0.3552 |

Seed sensitivity is material. The only case that repeats a near/above-3% win is the layer-40 expert at K15. The mid-layer expert K15 flips from a 3.39% win to a 0.59% loss. Gini deltas span -0.0263 to +0.0141 and average only -0.0023; lower Gini sometimes accompanies much worse total error (notably the mid-layer expert K18). Rotation therefore does not offer a stable "something for nothing," and it does not consistently erase the tier-3 tail opportunity.

## B — LDLQ/GPTQ-style fixed-CB reassignment

### Geometry and documented deviation

Scalar GPTQ is not directly representable by this CB format. An FP4 product-CB index represents an 8-column vector split across two independently assigned 4-column subtables, while one stored scale covers a 16-column group. Reassigning a scalar column would produce a state the serializer cannot encode.

The defensible variant used here is therefore:

1. Run the production encoder and freeze its codebook and two-tier scales.
2. Process 64-column blocks (aligned to both the 8-wide codeword and group-16 scale geometry).
3. Within a block, choose complete product-CB vectors by the production diagonal activation-weighted nearest-codeword metric. There is no illegal scalar splice and no scale refit.
4. Build `H = X^T X + 0.01 * mean(diag(X^T X)) * I`; form GPTQ's upper Cholesky factor `U` of `H^-1`.
5. After assigning block `A`, solve `E_A U_AA = W_A - Q_A`, then subtract `E_A U_A,*` from the unquantized columns. Feedback occurs between 64-column blocks, not sequentially inside a block. No activation-order permutation is used because it would break fixed CB tile geometry.

This changes encoder assignment only. The resulting weights use the original rung, codebook, scale plane, tile geometry, and serving kernel, so serving needs no online transform.

### Quality and overhead

`feedback/prod` is output-SSE ratio against the no-feedback production re-encode. `plain/prod` is the control fixed-field reassignment. `assign x` excludes Hessian construction; `amortized x` includes one Hessian build/factor divided across K12/K15/K18; `total encode x` compares production fit plus feedback and amortized factor against the production fit alone.

| Linear | K | feedback/prod | reduction | plain/prod | assign x | amortized x | total encode x |
|---|---:|---:|---:|---:|---:|---:|---:|
| `layers.20.experts.63.up_proj` | 12 | 0.0263 | 97.4% | 0.9998 | 2.09x | 2.80x | 1.43x |
| `layers.20.experts.63.up_proj` | 15 | 0.0247 | 97.5% | 1.0003 | 1.80x | 2.33x | 1.27x |
| `layers.20.experts.63.up_proj` | 18 | 0.0263 | 97.4% | 1.0006 | 1.43x | 1.72x | 1.17x |
| `layers.40.attn.wq_b` | 12 | 0.0927 | 90.7% | 1.0000 | 1.36x | 2.76x | 1.10x |
| `layers.40.attn.wq_b` | 15 | 0.0852 | 91.5% | 1.0001 | 1.31x | 2.18x | 1.15x |
| `layers.40.attn.wq_b` | 18 | 0.0871 | 91.3% | 0.9999 | 1.14x | 1.55x | 1.15x |
| `layers.40.experts.81.up_proj` | 12 | 0.0491 | 95.1% | 1.0005 | 2.08x | 2.80x | 1.43x |
| `layers.40.experts.81.up_proj` | 15 | 0.0339 | 96.6% | 0.9997 | 1.79x | 2.32x | 1.27x |
| `layers.40.experts.81.up_proj` | 18 | 0.0308 | 96.9% | 1.0012 | 1.44x | 1.73x | 1.17x |

The direct production-field re-encodes match saved production output error within 0.071%, and the plain reassignment matches those re-encodes within 0.125%. Thus the 10–40x error reduction is genuinely introduced by off-diagonal feedback under the measured calibration objective.

| Linear | form `X^T X` seconds | inverse-factor seconds | added diagonal damping |
|---|---:|---:|---:|
| `layers.20.experts.63.up_proj` | 0.0021 | 0.0378 | 0.136321 |
| `layers.40.attn.wq_b` | 0.1351 | 0.0841 | 0.00252328 |
| `layers.40.experts.81.up_proj` | 0.0021 | 0.0385 | 0.261095 |

The unusually large in-sample gain is plausible but also the reason not to overclaim: the expert Hessians are 4096x4096 from only 64 activation rows. Feedback has many null/near-null directions in which to compensate the calibration outputs. The next promotion gate must use disjoint held-out activation rows and must check whether the gain survives at the production calibration contract; without that, this result remains Research per PrismaQuant's design guidelines.

## Tests and recommendation

Targeted tests:

```text
12 passed, 14 warnings in 5.66s
```

The new tests include a float64 randomized-Hadamard identity and a tiny binary-codebook Hessian case exhaustively enumerated over all assignments. Plain nearest rounding selects `[-1,-1,-1]`; feedback selects the known global optimum `[-1,-1,+1]` and reduces the exact Hessian objective from 1.973608 to 0.788832.

Combined recommendation:

- **Rotation: skip.** Its mean result loses, its only repeated material win is one Linear/rung, its seed dependence is large, and it does not consistently flatten Gini.
- **LDLQ: needs-more, prioritize the held-out gate.** The measured effect is far beyond the 2–3% threshold at a tolerable total encoder multiplier and zero serving cost. If a disjoint-row test retains even a modest fraction of the 90–97% calibration gain, this should become an opt-in research encoder lever and proceed to the production calibration/serving validation ladder. Do not make it default from these 64 in-sample rows alone.
