# Tier-2 per-expert allocation counterfactual

Date: 2026-08-03. CPU-only offline analysis; no GPU or Docker command was
used. Reference artifacts were read from
`/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/` and all generated run
JSONs live beside this report under `tier2-out/` (ignored by Git because each
contains the full 33,325-row assignment).

## Result

Removing packed-stack uniformity while retaining each expert's fused w1+w3
(`w13`) coupling exposes a very large predicted tier-2 prize:

| Cell / arm | Collapsed baseline Δloss | Per-expert Δloss | Prize (baseline − counterfactual) | Prize | Exact bytes / budget | Headroom |
|---|---:|---:|---:|---:|---:|---:|
| b-92 free | 5,376.606709 | 23.424356 | 5,353.182353 | 99.5643% | 91,999,972,036 / 92,000,000,000 | 27,964 |
| b-92 floored | 5,376.606709 | 23.424356 | 5,353.182353 | 99.5643% | 91,999,972,036 / 92,000,000,000 | 27,964 |
| c-92 free | 1,325.831481 | 2.015491 | 1,323.815989 | 99.8480% | 102,862,821,888 / 102,862,838,300 | 16,412 |
| c-92 floored | 1,325.831481 | 2.015491 | 1,323.815989 | 99.8480% | 102,862,821,888 / 102,862,838,300 | 16,412 |

Every row passed the exact byte gate. The solver used 22,317 decision units:
11,008 fused expert-w13 pairs, 11,008 independent expert-w2 rows, and 301
ordinary body Linears. A separate post-solve check expanded all 33,325 rows
and reproduced the exact payload with the producer's CB assignment accountant,
including shared codebook sidecars once.

These prize numbers are **upper bounds**, not serving claims. They price no
mixed-format-per-stack launch overhead, kernel switching, metadata expansion,
or runtime implementation cost. They are predicted Δloss from the reconciled
cost table, not measured served KL/PPL. Lambda bisection is also a candidate
generator rather than a proof of global discrete-knapsack optimality.

## Reconciliation gate

The per-row pricer replays the allocator's cost precedence and the recorded
P5a family factors exactly once. Raw weight-only rows receive the factor;
already aggregated rows carrying `activation_pricing_applied=True` do not.

| Selection | Reconstructed Δloss | Recorded Δloss | Ratio |
|---|---:|---:|---:|
| artifacts-mxfp4-sm121 | 5,812.110642597935 | 5,812.110642597937 | 0.9999999999999997 |
| oldmenu-grid/b-92 | 5,376.606709145062 | 5,376.606709145063 | 0.9999999999999998 |
| oldmenu-grid/c-92 | 1,325.831480706076 | 1,325.831480706075 | 1.0000000000000007 |

All ratios are within the required 1e-3 of 1.0. The selection-recorded P5a
penalties replayed here are 138.139816300873 for `nvfp4_cb` and
193.6594503235084 for `fp8_cb`; the exact ratios above rule out the earlier
~142.6x double application.

## Format histograms (rows)

Free and floored assignments are identical within each cell.

| Format | b-92 | c-92 |
|---|---:|---:|
| NVFP4_CB_K14 | 30,399 | 25,117 |
| NVFP4_CB_K15 | 1,108 | 1,989 |
| FP8_CB_K36 | 71 | 87 |
| MXFP4_SOURCE | 1,704 | 6,021 |
| FP8_BLOCK_UE8M0_SOURCE | 43 | 111 |
| **Total** | **33,325** | **33,325** |

`FP8_BLOCK_UE8M0_SOURCE` is the additional body format recorded by both
baseline cells. Legality was replayed from each cell's
`format_applicability.json`, including its runtime/source-kind masks.

## Formats required per hypothetical packed stack

There are 86 stacks: w13 and w2 for each of 43 layers. Counts below are the
number of distinct formats that a mixed-stack runtime would need within that
stack.

| Cell | 1 format | 2 formats | 3 formats |
|---|---:|---:|---:|
| b-92, all stacks | 9 | 39 | 38 |
| b-92, w13 only | 6 | 13 | 24 |
| b-92, w2 only | 3 | 26 | 14 |
| c-92, all stacks | 0 | 6 | 80 |
| c-92, w13 only | 0 | 2 | 41 |
| c-92, w2 only | 0 | 4 | 39 |

The generated JSONs retain the per-layer stack-to-format lists. Explicit
w1+w3 validation found zero coupling violations in every arm.

## Protection-floor arms and decomposition

The exact probe table contains 5,505 zero-`h_trace` expert-linear rows spanning
1,835 experts. (The mounted table is authoritative for this report; its row
count differs from the approximate count in the task context.) The requested
menu already bottoms out at `NVFP4_CB_K14`, so `--floor-unmeasured` has no
sub-K14 candidate to remove. Consequently both floor arms are deliberately
no-ops:

| Cell | Floored − free Δloss | Zero-evidence-row component | Measured-row component | Assignment changes |
|---|---:|---:|---:|---:|
| b-92 | 0.0 | 0.0 | 0.0 | 0 |
| c-92 | 0.0 | 0.0 | 0.0 | 0 |

For completeness, decomposing the tier-2 prize itself gives zero savings on
the zero-evidence rows (their baseline and counterfactual prices are both
zero) and the full savings on measured rows: 5,353.182352884295 for b-92 and
1,323.815989499060 for c-92. A future menu containing K12/K13 would make the
floor effective; the CLI and unit test exercise that path.

## Commands

Each arm used this CPU-only CLI shape (with the matching cell, budget, output,
and optional floor flag):

```text
python3 scripts/tier2_per_expert_counterfactual.py \
  --baseline-dir <oldmenu-grid/{b,c}-92> \
  --probe <cell>/probe.pkl \
  --cost-table <artifacts>/cost_full.pkl \
  --budget <92000000000|102862838300> \
  --menu NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36,MXFP4_SOURCE,FP8_BLOCK_UE8M0_SOURCE \
  [--floor-unmeasured] \
  --output tier2-out/<cell>-<arm>.json
```

The four output files are `b-92-free.json`, `b-92-floored.json`,
`c-92-free.json`, and `c-92-floored.json`.
