# Orphan modules and tools — archived 2026-07-30

**Kill order.** Architecture re-vet 2026-07-30, proposal **R19** (Lens 6 P5),
accepted by Robert as part of the wave-2 mandate: *"implement ALL proposals."*
See `docs/audits/architecture_re-vet_2026-07-30.md` §R19 and
`docs/ARCHITECTURE.md` §11.

Every file here had **zero references** across `prismaquant/ tools/ scripts/
examples/ tests/` at wall time (test files moved with the module they test).
Nothing here lost a measurement — these are threads that closed and left their
instruments on the bench.

## Durable lesson

**A tool outlives the question it was built to answer, and there is no signal
when that happens.** Three of these belong to threads with recorded verdicts —
the damp sweep is OFF-final, the cross-layer program came back null, the AURA
additivity question was cleared — yet all three tools sat in `tools/` looking
exactly as live as `run-pipeline.sh`'s own stages. Reachability is the only
honest liveness signal a repo has, and it has to be *checked*, not assumed.
The counter-lesson is in `_fast_kernel_guard` below: an orphan is sometimes a
**missing caller**, not a dead idea, and the wall must say which.

## Contents, by closed thread

### Cross-layer program — null result

- `prismaquant/cross_layer_residual.py` (360 L) + `tests/test_cross_layer_residual.py`
  — `pair_interaction_stats`: measures the pairwise KL residual
  `KL(i,j) − KL(i) − KL(j)`. Its verdict is the reason it is here: the fp32
  residual is **+5–12% and diffuse, with 3 of 1180 pairs significant**
  (`xlayer_sensitivity_2026_06_09`), and the apparent non-additivity was a
  **bf16 artifact** — per-Linear KLs *do* add in fp32
  (`cross_layer_additivity_fp32`). That pair of findings retired both the
  pairwise-modelling literature lane (CLADO/QUBO) and, the same day, the L2/L3
  cascade itself (`archive/l3_propagated_2026-07-30/`). Read that wall first if
  you are tempted to re-open cross-layer cost modelling.

### Damp-sweep thread — OFF-final

- `tools/analyze_damp_sweep_log.py` — parsed per-Linear damp-sweep selections
  out of a render log. The sweep has been OFF with fixed damp 1.0 since
  2026-06-12: its evaluator was in-sample (held-out basins invert 31/31), the
  served A/B had fixed damp winning every gold-lane readout across two
  calibration draws at ~4.4× less render time, and the old "+137.5% if
  disabled" claim was a tier-5 hook screen that inverted on the gold lane.
- `tools/lift_gptq_ablation.py` — lever-lift ablation driver from the
  consolidation-era GPTQ studies (`archive/prismaclip_2026-05-14`,
  `archive/sao_2026-05-15`).
- `tools/collect_col_importance.py` — per-column importance collection, the
  input side of the retired column-permutation (SAO) and act-order studies.
- `tools/remeasure_cost_post_lever.py` — re-scored a cost table after toggling
  a render lever. Subsumed by `COST_MODE=production-render-score` / `aura`,
  which derive cost from the production render by construction.

### L3 / sensitivity thread

- `tools/sensitivity_coverage_summary.py` — coverage report over an L3
  sensitivity sweep. Its siblings
  (`sensitivity_propagated_group_report.py`,
  `build_propagated_sensitivity_cost_sweep.py`,
  `summarize_kl_sensitivity_response.py`) went to
  `archive/l3_propagated_2026-07-30/` the same day with the modules they
  import; this one had no import edge, only a dead thread.

### Export hygiene

- `prismaquant/collapse_config_groups.py` (83 L) — retro-fix CLI that collapsed
  per-expert `config_groups` enumerations in an already-shipped HF config into
  per-`(layer, projection)` regexes. The exporter emits collapsed groups
  natively (`_build_target_list`, idempotent, canonical vLLM scheme names), so
  the retro-fix has had nothing to fix since 2026-04-22. It also carried a
  documented hazard (CODEX finding): its collapse assumes within-layer expert
  format uniformity, which the exporter guarantees but the tool never checked.

### Serve-time kernel enforcement — an orphan that is a MISSING CALLER

- `prismaquant/_fast_kernel_guard.py` (92 L) + `tests/test_fast_kernel_guard.py`
  — `require_fast_kernels(model)`: reads the model profile's kernel
  requirements and hard-fails at startup if a required fast kernel (e.g.
  `causal-conv1d`, `flash-linear-attention`) is not importable, unless
  `PRISMAQUANT_ALLOW_PYTORCH_FALLBACK` is set. **This is the only mechanized
  piece of core principle 9's "routed to a *performant* kernel (not a slow
  fallback)" gate**, and it is walled only because its sole caller
  (`polish_from_assignment`) was archived in the 2026-05-15 consolidation —
  verified: no dynamic import anywhere in the tree resolves it (its own
  `importlib.import_module` is the kernel *probe*, not an import of the guard).

  **R19 said wall; the orphan report recommended REWIRE.** Resolved by walling
  it *and* booking the gap: `docs/ARCHITECTURE.md` §12 carries a LOW debt row
  recording that serve-time fast-kernel enforcement has had no caller since
  2026-05-15 and principle 9's kernel-performance gate is therefore **manual**.
  Whoever funds that row should start by moving this file back — the mechanism
  is written and tested; only the call site is missing.

## To resurrect

Move the file back to `prismaquant/` or `tools/` (and its test to `tests/`).
None of these had library dependents, so no call sites need re-wiring — the
point of the wall is that there were none to begin with.

## Deliberately NOT walled

`prismaquant/aura_additivity_gate.py` — R19 listed it among the orphans, Lens 1
(**R2**) wants it wired as AURA's standing per-artifact trust-region readout.
Robert's wave plan keeps it: **R2 wires it in a later wave.** It stays live.
