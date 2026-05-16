# Polish Archive

Archived 2026-05-15 per user direction: "Polish, as well, as long as it's
not doing anything major." Confirmed it's not load-bearing:

- `run-pipeline.sh` does not invoke polish (only mentions it in a comment
  about what KL paths measure).
- `prismaquant/__init__.py` references `polish_from_assignment.py` in a
  docstring describing the workflow but does not import it.
- No other module in `prismaquant/` or `tests/` imports polish.

## Contents

- `prismaquant/polish.py` — measured single-flip coord-descent polish:
  for each pass evaluates every single-unit format flip and accepts the
  one with the lowest measured KL provided it strictly improves the
  metric. Fused-sibling aware. Pure measured KL gating, no surrogate.
- `prismaquant/polish_from_assignment.py` — CLI wrapper: loads a
  kneedle-style assignment, optional ProductionWeightCache, runs
  measured single-flip polish, writes polished assignment + trace JSON.

## Related (live in tree)

- `decision_units.py` — the fused-sibling unit / pair API both polish
  modules use. Stays in the live tree; load-bearing for
  `production_recache.py`, `production_weight_cache.py`, and
  `weight_session.py`.

## To resurrect

Move both files back to `prismaquant/`. No call sites or imports need
to be re-wired in the live tree (none were removed when archiving).
