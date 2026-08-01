# PrismaClip / PrismaFisherClip archive (walled 2026-05-14)

This wall preserves the PrismaClip and PrismaFisherClip line of work: a family of
explicit weight-clipping solvers that searched for a per-Linear clipping threshold
*before* quantization, plus the Fisher-weighted variant that reused the probe's
`h_trace` to weight the clip objective.

It was retired in the 2026-05-15 consolidation because JSO (`joint_scale_opt`)
subsumed it outright — clipping is just another way of asking what the right block
scale is, and JSO already answers that question **inside** the GPTQ loop with
activation-weighted MSE over its per-group level grid, where a pre-pass clip solver
cannot see the error compensation it is about to perturb. Robert's verdict on the
standalone solver was blunt: *"it's useless."*

The smoke reports kept here (`docs/prismaclip_candidate_smoke_2026-05-11.md`,
`docs/prismafisherclip_smoke_2026-05-11.md`,
`docs/clip_fisher_mxfp8_ablation_2026-05-10.md`,
`docs/halo_4b_noclip_smoke_2026-05-13.md`,
`docs/qwen36_27b_fp8_frontier_2026-05-13.md`) are the measurement record behind that
call, and `docs/consolidation_2026-05-15.md` is the decision itself. This tree is
**documentation only** — no code was preserved and nothing in `prismaquant/` imports
from here.

**The durable lesson: when two mechanisms answer the same question, keep the one that
answers it inside the loop that consumes the answer.**

Do not revive a standalone clip solver into a production path without an explicit ask.
The useful successor question — deriving the per-Linear optimum from weights alone —
is tracked in `docs/design/unified_render_theory.md`.

See `docs/ARCHITECTURE.md` §11 ("History — what was tried and rejected") for this
wall's row in the graveyard table and its place among the other rejected methods.
