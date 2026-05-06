---
Followup to .claude/prismaquant-handover-2026-05-02.md (overnight autonomous handover).
Read that one FIRST — most context is there. This file captures only the deltas
and corrections discovered after the user resumed this morning at ~08:42 EDT.
---

# PrismaQuant / PrismaSCOUT Handover — 2026-05-02 PM

## Session bridge

Nothing executed between the autonomous overnight handover and this one. Last
container `pq-qwen4b-prismascout-v13` exited 137 at 02:07 EDT; user went to bed
01:55, woke ~08:42 EDT. No new commits, no live containers, no procs running.
System idle (~2 GB used / 119 GB free).

## Corrections to morning handover

**File paths for v5 anchor 4.50 outputs.** The morning doc said
`final_assignment_bpp_4.50.json` and `final_layer_config_bpp_4.50.json` live
inside `iterate_v5_out/anchor_bpp_4.50/`. They actually live one level up at
`iterate_v5_out/`. Anchor subdir contains only the per-iter caches and
`l3_propagated_costs.pkl`. Confirmed:

```
/home/rob/dq-runs/qwen3-4b-perturbed-x/iterate_v5_out/
├── final_assignment_bpp_4.50.json     ← 2026-05-02 00:29 (use this for export)
├── final_layer_config_bpp_4.50.json   ← 2026-05-02 00:29 (use this for export)
├── iteration_trace.jsonl
├── l3_polish_trace.jsonl
├── anchor_bpp_4.50/                   ← coord/L3 caches + l3_propagated_costs.pkl
└── anchor_bpp_5.00/                   ← INCOMPLETE — only iter_01 cache, no final
```

**v5 was multi-budget but only 4.50 finished.** The `anchor_bpp_5.00` subdir
exists with one L2 iter cache; it never reached coord descent. So v5 gave us
exactly one validated anchor (4.50). Pareto sweep still owed.

## State at handover

| Item | Value |
|---|---|
| Worktree | `/home/rob/prismaquant-propagated-cost` |
| Worktree HEAD | `7f7521a` ≡ `origin/main` (no local-only commits, only `?? .coverage`) |
| Main repo (`/home/rob/prismaquant`) | branch `feat/entmoot-moe-merge` with WIP — unrelated to PrismaSCOUT work |
| Containers | none running; v6-v13 all OOM-killed (exit 137) |
| Background tasks | none |
| Memory watchdog | not armed |

The main repo and the worktree are on **different branches** doing different
work. PrismaSCOUT lives entirely in the worktree at `feat/propagated-cost`.
The main repo's `feat/entmoot-moe-merge` is the user's separate WIP — leave it
alone unless explicitly asked.

## What still needs doing (unchanged from morning, with one tweak)

The morning handover laid out three options (A: ship proven config / B: test
Spec O / C: shrink memory and try all-on). All still apply. **The path that
ran successfully overnight (v5) used the proven config, so Option A is the
lowest-risk continuation.**

One concrete sequencing tweak: since `final_*_bpp_4.50.json` already exists,
the Pareto sweep could *exclude* 4.5 and just fill in 5.0/5.5/6.0/6.5/7.0,
saving ~75 min. Reuse the L3 costs pickle via `--resume-l3-costs-dir`.

## What to do first if you're picking up

1. Read the morning handover (`prismaquant-handover-2026-05-02.md`) for the full
   architecture/results context — this file deliberately doesn't repeat it.
2. `cd /home/rob/prismaquant-propagated-cost && git log --oneline -5` to confirm
   `7f7521a` is still HEAD and `origin/main`.
3. Decide A / B / C. Default to A unless the user wants validation of Spec O.
4. If A: launch v14 with **target-bits-list excluding 4.5** + the kernel/graph
   env vars OFF as in the morning handover's Option A snippet. Set a memory
   watchdog Monitor.
5. While Pareto runs, prep validation harness invocation for each anchor and
   the comparison-table generator.

## Wrapper-bug reminder (still active)

The codex agent wrappers used overnight reported "completed" before codex
itself finished. Pattern to remember: **always verify with `git log
origin/main..HEAD` and read the codex log directly** before trusting a
wrapper's done signal.

## Open questions to surface to user when they resume

- Does the v5 4.50 result warrant re-exporting the shipped Qwen 4B artifact
  *now* (before completing the Pareto sweep), or wait for the full curve?
- For 27B: still need the probe-based kneedle prediction approach. Probe data
  not yet generated. Worth queueing tonight if user is doing another autonomous
  block?
- Validation harness was wired (Spec B) but never run end-to-end against a
  PrismaSCOUT output. First full run will likely surface integration bugs;
  budget some debugging time.
