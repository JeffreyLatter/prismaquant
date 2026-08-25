# Qwen3.5-0.8B product-K versus signed-S rung screen (2026-07-22)

This is the provenance record for excluding `NVFP4_CB_S13..S16` from new
production allocations while retaining their exporter and decoder support.
It is a local quantization-error screen, not served KL/PPL evidence.

> **Status 2026-08-17 — the family was DELETED, not merely de-menued.** This
> screen stands as written and is the quality half of the case. The deciding
> fact came later and is a serving one: `n_sub = 1` fails the predicate every
> native Gridbook FP4 route tests (`n_sub == 2 and type_size == 4*k + 9`), so
> no signed rung could ever reach a native kernel. Exporter and decoder
> support has been removed; `NVFP4_CB_S*` no longer parses. See
> `docs/ARCHITECTURE.md` §9.2.

## Reproduction identity

- Command: `scripts/run_0p8b_s_rung_headtohead.sh`
- Required PrismaQuant checkout, also recorded by the cost artifact:
  `9d2a88646d7c6ff33924d91ba0804b225c66b8cf`. Reproduction must run from that
  commit because the current production profile intentionally masks signed
  rungs from new allocations.
- Source: `Qwen/Qwen3.5-0.8B`; the local source weight shard used by the run has
  SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
  and `config.json` has SHA-256
  `72e780ebfb184003df8a1e1f4df0eb1ef933cc613faf5a2aa745e7ab7f43205e`.
- Calibration: 16 rows, sequence length 512, seed 42, dataset SHA-256
  `3d28ada7c8ff7dfed139ef61281a058925c1924730701dd99bf5a2958b973da4`.
- Formats: matched pairs `K13/S13`, `K14/S14`, `K15/S15`, `K16/S16`, with
  ladder interpolation disabled and the balanced encoder tier.
- Primary cost artifact: `artifacts/cost.pkl`, SHA-256
  `212b588acb30277759a121beab07ef6b999720da92db3dc1de892bd84df0d03f`.
- Assignment artifact: `artifacts/layer_config.json`, SHA-256
  `a87c1ff9fab6ac968bf24befb39d76300ebb1eaf4ec7e0929e72d0344a7e0e7f`.
- Pipeline contract: `artifacts/pipeline_spec.json`, SHA-256
  `ee5685c7c65606fdac1c10ea566f5d8e49b1eb2a82437a01c4bd55d51f6f06c8`.

The local run directory was `/home/rob/dq-runs/s-rung-headtohead`; paths are
included only to locate the retained artifacts on the validation host. The
hashes, not those machine-local paths, identify the evidence.

## Comparison and result

For each of 194 measured Linears and each `k` in 13 through 16, compare the
same-rate pair's `weight_mse` from `cost.pkl`. A product-K win is strictly
`weight_mse(Kk) < weight_mse(Sk)`; there were no ties.

| rung pair | comparisons | median signed penalty, `S/K - 1` |
|---|---:|---:|
| K13 / S13 | 194 | +2.2446% |
| K14 / S14 | 194 | +1.9624% |
| K15 / S15 | 194 | +0.4639% |
| K16 / S16 | 194 | +0.8432% |

Product K won 609 of 776 comparisons (**78.48%**, conventionally rounded to
79%); signed S won 167. At the 2.6-bpp allocator target, only six signed units
were selected versus 147 product-K units (`S13`: 2, `S16`: 4). This is enough
to remove the signed family from the production search menu, but not to remove
codec compatibility or claim a teacher-backed quality result.
