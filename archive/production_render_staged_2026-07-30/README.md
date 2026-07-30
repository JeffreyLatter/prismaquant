# `COST_MODE=production-render-staged` — archived 2026-07-30

**Kill order.** Architecture re-vet 2026-07-30, proposal **R17** (Lens 6 P3),
accepted by Robert as part of the wave-2 mandate: *"implement ALL proposals."*
See `docs/audits/architecture_re-vet_2026-07-30.md` §R17 and
`docs/ARCHITECTURE.md` §11. It is now the **eighth** `exit 2` archived-lever
gate in `run-pipeline.sh`.

## The killing measurement

`docs/results/production_render_staged_27b_results_2026-05-21.md` — Qwen3.6-27B
at 5.5 bpp, the staged lane against the plain production-render-score lane:

| metric | staged | baseline | verdict |
|---|---|---|---|
| last-token KL screen | **0.0232** | 0.0280 | staged "wins" |
| direct WikiText PPL | **10.83** | 8.33 | staged **regresses 30%** |

The doc's own conclusion is **"Do not ship."** Nothing has been run on the lane
since `8e071bf`.

## Durable lesson

**This is the canonical instance of the failure the graveyard exists to
record: a screen improved while the gold metric regressed.** It is the same
shape as grouped-KL (−3.52% local PPL, lost the vLLM A/B), as the staged-render
last-token-KL "win", and as `current_only` extrapolation (won the hook screen,
lost full-vocab KL). CLAUDE.md §5 principle 3 — *promote on the serving metric,
not the screen* — is written from these.

The *mechanism* of the failure is specific and worth keeping: the staged lane
renders **only NVFP4** in pass 1, ranks Linears by NVFP4 render score, and
renders the higher formats for the top `PROMOTE_FRACTION` alone. Every Linear
outside the tail therefore carries an **explicit `unavailable` cost** for every
promotion format, so the DP cannot even consider promoting it. That is a
render-budget heuristic deciding the allocator's candidate set — principle 1's
veto — and the 27B run shows what it buys: the tail ranking picked
linear-attention projections whose MXFP8 score was up to **10.7× worse** than
their NVFP4 score (130 of 149 tail members were worse or equal under MXFP8),
i.e. the ranking that chose the promotion set was not measuring what the
promotion would cost.

`grouped-kl` was `exit 2` for losing a vLLM A/B; this lane lost a direct PPL
comparison in its own result doc. Same class, same gate.

## Contents

- `prismaquant/production_render_staged.py` — `select_tail_from_render_scores`
  verbatim, plus the verbatim text of every branch removed from
  `prismaquant/production_render_cost.py` (the `bf16_policy="promotion-set"`
  arm, the `missing_render_score_policy="unavailable"` arm, the strictness
  check, the three `meta` keys, and `main()`'s tail dispatch). Importable
  against the live tree for replay, though nothing calls it.
- `run-pipeline-staged-block.sh` — the 108-line `[2b/4]`…`[2e/4]` staged
  block removed from `prismaquant/run-pipeline.sh`, verbatim.
- `tests/test_production_render_staged.py` — the two tests lifted out of
  `tests/test_production_render_cost.py`
  (`test_staged_render_cost_marks_unmeasured_promotions_unavailable`,
  `test_select_tail_from_nvfp4_render_scores`). A record; not runnable until
  the lane is restored.

## Live-tree state after the wall

- `run-pipeline.sh`: `COST_MODE=production-render-staged|production-render-tail`
  now `exit 2`s with the measurement above. The four staged shell defaults
  (`PRODUCTION_RENDER_COST_PROMOTE_FRACTION`, `_MIN_PROMOTE_SCORE`,
  `_MAX_PROMOTIONS`, `PRODUCTION_RENDER_COST_TAIL_QNAMES`) are gone, as is the
  staged arm of the config echo. The catch-all now reads *"COST_MODE must be
  local, production-render-score, or aura"*.
- `prismaquant/production_render_cost.py`: **the default path is byte-identical
  in behaviour.** Removed only the staged-only surface — `load_qnames_file`,
  `select_tail_from_render_scores`, the three staged keyword parameters of
  `synthesize_production_render_cost_payload` and their branches/meta keys, and
  the ten staged CLI arguments (`--select-tail-output`, `--select-tail-summary`,
  `--select-tail-format`, `--select-tail-top-fraction`,
  `--select-tail-min-score`, `--select-tail-min-count`,
  `--select-tail-max-count`, `--missing-render-score-policy`,
  `--promotion-qnames-file`, `--bf16-policy`). `--require-render-scores` still
  covers the strict "fail on missing render scores" behaviour that
  `--missing-render-score-policy=error` duplicated.

## To resurrect

1. Restore the three keyword parameters and their branches in
   `synthesize_production_render_cost_payload` (verbatim text is in
   `prismaquant/production_render_staged.py`), plus `load_qnames_file` and
   `select_tail_from_render_scores`.
2. Re-add the ten CLI arguments and `main()`'s tail dispatch.
3. Paste `run-pipeline-staged-block.sh` back after the `[2c/4]` block, restore
   the four shell defaults and the staged `COST_MODE` case arm, and delete the
   `exit 2`.
4. Move `tests/test_production_render_staged.py` back into
   `tests/test_production_render_cost.py`.
