> **CLOSED PROCESS (archived 2026-07-30).** All 82 findings of the 2026-06-09
> review were dispositioned by 2026-06-22 (`CODEX_PROGRESS.md`, same
> directory). This file is a **record of a dispatch, not active instruction** —
> do not follow its ground rules or protocol as current policy; the live rules
> are `AGENTS.md` and `docs/design/design_guidelines.md`.
>
> One ledger line is now inverted on HEAD: MAJOR-M10's rationale
> (`CODEX_PROGRESS.md:10`) closes the hybrid expert-AURA cost as
> WONTFIX-FOR-NOW on the grounds that "nothing ships via AURA-MoE today (no
> COST_MODE=aura in run-pipeline)". `COST_MODE=aura` is now fully wired
> (`run-pipeline.sh:187` accepts it; sub-stages at `:314-336` and `:825-956`),
> with empirical expert costs for the MoE rungs — opt-in, not the default,
> which stays `production-render-score`.

# Codex review-batch dispatch — 2026-06-11

You are working through the non-critical findings (35 MAJOR, 42 MINOR,
5 INFO) of the 2026-06-09 exhaustive review of PrismaQuant. The findings
are in `CODEX_QUEUE_FINDINGS.md` (same directory). The 5 CRITICALs are
handled separately — do NOT touch: run-pipeline.sh stage-gating/holdout
wiring, validate_assignments_kl work-root default, or paper/figures/
fig_aura_rd_geometry.tex (those are in flight on another branch; if a
finding overlaps one of those files, fix only the non-critical aspect and
note the overlap in your commit message).

## Ground rules (mandatory, from CLAUDE.md / AGENTS.md — read both first)
- Work ONLY in this worktree (branch codex/review-batch). Never touch
  /home/rob/dq-runs/v1-damp-ab or any running job's files.
- NEVER write to /tmp. TMPDIR is preset; keep it.
- The platform measures and optimizes; it does not band-aid. No heuristic
  constants, no format bans, no post-hoc rewrites. If a finding's "fix"
  would violate a design principle, document why and skip it.
- Defaults stay backwards-compatible. New behavior behind env flags or
  opt-in CLI args unless the finding is a plain bug.
- Python: /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python, PYTHONPATH=.
- GPU: do not launch GPU-heavy work (full test suite, model loads) while
  another process owns the GPU — check `pgrep -f "build_production_cache|
  measure_vllm|export_native"` first; if busy, run CPU-safe tests only
  (pytest -k "not cuda" works for many; otherwise defer and note it).

## Per-finding protocol
For EACH finding (work top-down: MAJOR M1..M35, then MINOR, then INFO):
1. Re-verify the claim against current code (the review is 2 days old;
   some findings may already be fixed — e.g. anything touching the
   packed-expert gate, AutoRound-dict canonicalization, or expert render
   modes changed recently). If already fixed or stale: record
   `STALE` with the commit that fixed it.
2. If real: implement the minimal principled fix + a test that pins it
   (tests/ naming conventions in the repo). If the right fix is large or
   design-contentious: write a SKIPPED-NEEDS-DESIGN entry instead with a
   concrete proposal. Do not half-fix.
3. Commit per logical group (one finding or a tight cluster per commit),
   message format: `review-batch: <finding-ids> <summary>`.
4. Append one line per finding to CODEX_PROGRESS.md:
   `<id> FIXED <commit> | STALE <commit> | SKIPPED-NEEDS-DESIGN | WONTFIX <reason>`.

## Known-stale hints (verified fixed since the review)
- "zero tests for packed-expert GPTQ path" — partially addressed by
  tests/test_packed_expert_cross_domain_gate.py (commit 0bd5d9c); verify
  coverage rather than assuming.
- The fused-coherence/DefaultProfile cluster was reworked on 2026-06-08;
  re-verify against HEAD.
- The full-suite test failure test_meta_init_fla_priming (test-isolation
  bug) IS in your queue — fix the isolation, not the assert.

## Definition of done
- CODEX_PROGRESS.md has a line for all 82 findings.
- Full test suite green (612+ passing) once the GPU is free; CPU-safe
  subset green continuously.
- No commit touches serialization formats vLLM reads.
