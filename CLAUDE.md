# Claude Handover

Before implementing new functionality, read `AGENTS.md` and
`docs/design_guidelines.md`. Those files are the normative design rules for
GPU-first execution, cache reuse, vLLM gating, and measurement discipline.

Before working on PrismaQuant, read in order:

- `.claude/prismaquant-handover-2026-05-02.md` — **CURRENT STATE.**
  PrismaSCOUT L3-redesign landed end-to-end. v5 proved 34% KL improvement
  over L2 at 4.5 bpp on Qwen 4B (KL 0.371 → 0.245). Kernel + graphs +
  replay combo blocked by GB10 OOM on 121 GB UMA. Recommended path: ship
  proven v5 config for Pareto sweep. Open issues + file map + commits
  list inside.
- `.claude/prismaquant-handover-2026-05-01.md` — earlier state. Allocator
  Δloss bug + the design that became PrismaSCOUT. User's conceptual core.
- `.claude/prismaquant-handover-2026-04-29.md` — older session context
  (HALO landing, wins stack, overnight 4B smoke).
