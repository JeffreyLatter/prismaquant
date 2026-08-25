# Smart-union production cache — archived 2026-07-30

**Kill order.** Architecture re-vet 2026-07-30, proposal **R18** (Lens 6 P4),
accepted by Robert as part of the wave-2 mandate: *"implement ALL proposals."*
See `docs/audits/architecture_re-vet_2026-07-30.md` §R18 and
`docs/ARCHITECTURE.md` §11.

## What it was

`PRODUCTION_CACHE_UNION=1` switched the `SELECTION_MODE=validated-surrogate`
frontier cache from a full format-menu render (every quantizable Linear ×
every quantized format) to a "smart union": NVFP4 for every eligible Linear,
plus an `FP8_DYNAMIC` fallback entry **only** for Linears whose NVFP4
`output_mse` sat above the `--p-fp8` percentile of the NVFP4 error
distribution. On Qwen3.5-0.8B that was 258 renders instead of 438
(`docs/results/union_cache_smoke_0p8b.md`, filed HISTORICAL).

## Durable lesson

**A render-budget heuristic that pre-decides which fallbacks are worth
measuring is a constraint on what the allocator may choose** — exactly the
class principle 1 vetoes. The union rule picks the FP8 candidate set from a
*percentile of a surrogate error distribution*, so any Linear below the
percentile is denied an FP8 rung before the real-KL frontier ever sees it. The
saving it bought (≈40% of the frontier render) was real but never load-bearing:
the frontier cache is built once per run, and the measurement it truncates is
the one the whole validated-surrogate path exists to make.

It also never earned its keep empirically — default `0` since it landed, zero
shipped artifacts, untouched since `7ac2d98` (2026-05-28), and its only
smoke doc was already filed HISTORICAL. `tools/patch_union_from_pareto.py` had
zero references anywhere in the tree.

## Contents

- `tools/build_union_cache.py` (282 L) — the union render driver: reads
  `cost.pkl` + the input `layer_config.json`, computes the NVFP4 `output_mse`
  percentile, and fills a `ProductionWeightCache` with the union entry set.
- `tools/patch_union_from_pareto.py` (97 L) — post-hoc patcher that added
  missing union entries implied by a Pareto assignment set. Zero references.

## What replaced it in the live tree

Nothing — the frontier path builds the full format menu
(`prismaquant.build_production_cache --render-scope format-menu
--render-packed-experts`, `run-pipeline.sh` stage 4/4 A), which is what every
shipped artifact used.

## Live-tree state after the wall

`run-pipeline.sh` no longer defines `PRODUCTION_CACHE_UNION` /
`PRODUCTION_CACHE_UNION_P_FP8`; setting either to a non-zero value now
**fails fast with `exit 2`** pointing here (the §3.5 archived-lever gate
convention).

## To resurrect

Move both files back to `tools/`, restore the `PRODUCTION_CACHE_UNION` defaults
and the `if [[ "${PRODUCTION_CACHE_UNION:-0}" == "1" ]]` dispatch inside the
frontier-cache branch of `run-pipeline.sh`, and delete the `exit 2` gate. No
library code was changed; `build_union_cache.py` imports only live modules.
