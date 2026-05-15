# SAO (Static Act-Order) Archive

Archived 2026-05-15 after 4-arm ablation on Qwen3-0.8B showed SAO does
not earn its keep on its own metric.

See `/home/rob/.claude/projects/-home-rob-prismaquant/memory/jso_sao_outcome.md`
for the numbers. Summary:

- Per-Linear activation-weighted MSE: SAO +0.66% (worse than vanilla GPTQ)
- Worst-hit kind: `self_attn.o_proj` (+22.6%), `linear_attn.out_proj` (+9.4%)
- Even oracle per-Linear cherry-pick: -0.65% (negligible)
- Per-kind deployable rule: -0.05% (essentially zero)

Mechanism: SAO permutes columns by `diag(H)` magnitude, then GPTQ runs
its column-by-column update through the permuted columns. But GPTQ's
update already uses the full Hessian for error propagation — the
column-order doesn't add information GPTQ doesn't already exploit.
On head-structured Linears the permutation breaks natural block
alignment.

## What's still in the live tree

The `static_act_order` lever name + per-Linear conditional branches
in `export_native_compressed.py` + the SAO scaffolding in
`production_weight_cache.py` / `kl_sensitivity_probe.py` remain as
dead code, callable only if `--enable …,static_act_order` is passed
to `build_production_cache.py`. `run-pipeline.sh` rejects
`static_act_order` in `PRODUCTION_CACHE_LEVERS` (commit pending).

## To resurrect

Re-add `static_act_order` to the run-pipeline lever allowlist and
re-enable the SAO arms in `tools/lift_gptq_ablation.py`. The Python
code paths still exist.
