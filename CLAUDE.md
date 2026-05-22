# Claude Handover

Before implementing new functionality, read `AGENTS.md` and
`docs/design_guidelines.md`. Those files are the normative design rules for
GPU-first execution, cache reuse, vLLM gating, and measurement discipline.

Before working on PrismaQuant, read in order:

- `.claude/prismaquant-handover-2026-05-20.md` — **CURRENT STATE.**
  Grouped-KL cost surrogate validated on 27B (−3.52% PPL at 6.0 bpp,
  fixes non-monotonicity). JSO 4B A/B saw jso_off win 0.8-1.6% PPL but
  result has cost-surrogate confound; wall-off was committed then
  REVERTED. Next: launch queued 27B JSO isolation A/B
  (`/home/rob/dq-runs/qwen36_27b_jso_isolation.sh`). Open issues + file
  map + commits inside.
- `.claude/prismaquant-handover-2026-05-02.md` — earlier state.
  PrismaSCOUT L3-redesign landed end-to-end. v5 proved 34% KL improvement
  over L2 at 4.5 bpp on Qwen 4B.
- `.claude/prismaquant-handover-2026-05-01.md` — earlier state. Allocator
  Δloss bug + the design that became PrismaSCOUT.
- `.claude/prismaquant-handover-2026-04-29.md` — older session context.
