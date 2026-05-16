# HALO Archive

Archived 2026-05-15 (commit-pending) after user judgement that HALO has not
demonstrated utility on any of our production models. The 2026-05-09 +
2026-05-13 smokes on Qwen3-0.8B and Qwen3-4B did not show a non-regressive
HALO win that survived measured KL.

This directory preserves:
- `prismaquant/halo.py` — full HALO module (per-Linear Hadamard rotations
  with gamma fold support, head-projection rotation, layer-projection
  rotation, validation helpers, metadata helpers)
- `tests/test_halo.py` — the HALO unit test suite

Live in main tree:
- A minimal `prismaquant/halo.py` stub remains so that conditional
  `if args.halo_mode == "random":` branches in build_production_cache,
  production_recache, validate_assignments_kl, and export_native_compressed
  resolve at import time. Any call into the stub raises a clear
  archive error pointing here. The default lever surface
  (PRODUCTION_CACHE_LEVERS, HALO_MODE in run-pipeline.sh) no longer
  references HALO; the pipeline cannot reach the conditional branch
  without explicit override.

To resurrect: move this directory's `halo.py` back to `prismaquant/halo.py`,
move `test_halo.py` back to `tests/`, re-add the HALO_MODE / HALO_SEED
env defaults in `run-pipeline.sh`, and re-register the `halo`
mechanism in `render_score.py`.
