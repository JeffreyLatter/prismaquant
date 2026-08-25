# REAP expert pruning archive (walled 2026-05-15)

This wall preserves the REAP-style MoE expert-pruning track:

- `prismaquant/observers/expert_saliency.py` — the saliency observer that scored
  routed experts during calibration.
- `prismaquant/expert_prune.py`, `prismaquant/allocator_prune.py` — the pruning
  mechanics and the allocator integration.
- `prismaquant/adaptive_sampling.py`, `prismaquant/multi_chunk_probe.py` — the probe
  support built to feed it.
- `docs/joint_optimizer.md` — the joint prune-plus-allocate design note.
- `tests/` — the five tests that covered the above.
- `examples/launchers/launch-{minimax,minimax-allocate-export,dsv4}.sh` — the
  launchers that exercised it.

It is archived — not merely default-off — because its cost model is **structurally**
wrong in a way no tuning fixes: dropping an expert does not remove its tokens, it
*redistributes* them to the router's next choices, and the saliency score under-counts
that misrouting cost entirely. A pruned model therefore looks cheap to the allocator
and is expensive in served KL.

**The durable lesson: a surrogate that never sees the cost it creates will keep
proposing it.** That is why pruning stays disabled for DSv4-Flash-Base despite the
size pressure that motivated it. The sanctioned ways to buy footprint on MoE models
are factorization, rotation, and sub-NVFP4 rungs — never expert removal.

Nothing in the live tree imports from this directory; the modules here are frozen at
their 2026-05-15 state and are not maintained against current profile/allocator APIs.
If you grepped for `expert_prune` or `expert_saliency` and landed here, that is the
wall working as intended. Do not revive into a production path without an explicit ask
**and** a cost model that prices redistribution.

See `docs/ARCHITECTURE.md` §11 ("History — what was tried and rejected") for this
wall's row in the graveyard table and its place among the other rejected methods.
