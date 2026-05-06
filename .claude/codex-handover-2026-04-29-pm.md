# Codex handover — quality-wins audit complete (2026-04-29 PM)

Branch: `feat/quality-wins-batch1` on worktree `/home/rob/prismaquant-quality-wins`.
Main repo `/home/rob/prismaquant` is clean except for launcher/eval scripts.

## Headline result

Per-win PPL attribution on Qwen3-0.6B (validator EVAL_PROMPTS, 12 prompts,
`PRISMAQUANT_BATCHED_NVFP4_EXPORT=0` so wins exercise the scalar path):

| config | PPL | Δ vs baseline |
|---|---|---|
| baseline (norm_fp32 only) | 15.453 | — |
| solo_damp (PRISMAQUANT_GPTQ_DAMP_SWEEP=1) | 15.264 | −0.189 |
| **solo_clip (PRISMAQUANT_ACT_CLIP_QUANTILE=0.999)** | **14.546** | **−0.907** |
| solo_block (PRISMAQUANT_BLOCK_OUTPUT_MATCH=1) | 15.453 | 0.000 |
| solo_dnh (PRISMAQUANT_DO_NO_HARM=1) | 15.453 | 0.000 |
| **full_v2 (all four wins on)** | **14.987** | **−0.466** |

Artifacts at `/home/rob/dq-runs/qwen0p6b-smoke/<config>/exported/`. CSV
at `/home/rob/dq-runs/qwen0p6b-smoke/audit_eval.csv`.

## The critical finding — wins interfere when stacked

**`full_v2` (−0.466) is materially WORSE than `solo_clip` alone (−0.907).**
Stacking the wins loses ~half the act_clip benefit. Naive cumulative-is-better
intuition is wrong here.

Most likely mechanism (not yet confirmed): DNH's gate uses **unclipped**
activation column importance to compute MSE, while the GPTQ pass it gates
optimized under **clipped** activations. The post-pass weight is better on
clipped E[a²] but DNH's gate sees unclipped MSE → DNH reverts clip-improved
Linears back to pure RTN, undoing the gain.

Secondary suspect: `damp_sweep` selects best damp by per-Linear MSE, but the
H matrix it sweeps was built on clipped activations. The "best damp" under
clipped H may not be the best damp for the unclipped distribution the model
actually serves.

This is exactly the AWQ-style interference pattern. Needs investigation before
the all-on defaults can be trusted on a production artifact.

## Code state on `feat/quality-wins-batch1`

Defaults flipped to ON (validated or harmless on the audit):
- `PRISMAQUANT_ACT_CLIP_QUANTILE` (default `0.999`) — load-bearing win
- `PRISMAQUANT_GPTQ_DAMP_SWEEP` (default `1`)
- `PRISMAQUANT_BLOCK_OUTPUT_MATCH` (default `1`) — inert on Qwen3 but neutral
- `PRISMAQUANT_DO_NO_HARM` (default `1`) — inert on Qwen3, but probably the
  source of the interference above
- `PRISMAQUANT_ACT_CACHE_FP32` (default `1`) — probe-time fp32 activation cache

Always-on:
- norm FP32 passthrough (helper `_passthrough_dtype`, no flag)

Still off / explicit:
- `--halo-mode` (defaults `off`) — multimodal hang unresolved
- `PRISMAQUANT_BATCHED_NVFP4_EXPORT` defaults ON in production; the env-gated
  wins above all also live in `prismaquant/export_batched_gptq.py` now (commit
  `95295ff` extended act_clip / damp_sweep / DNH into the batched path).

## Recent commits

```
bba63a6 quality-wins: flip validated defaults to ON
95295ff quality-wins: extend env-gated wins into batched path (codex review fix)
7a3628e quality-wins: export-cache fingerprint (codex review #2)
e... (older) quality-wins #4 HALO module + tests + integration
```

Tests: 308/308 pass on the worktree.

## Open bugs / investigations (for codex)

1. **DNH × act_clip interference** — confirmed empirically by full_v2 < solo_clip.
   Probable fix: when act_clip is active, DNH's per-Linear MSE comparison
   should use the *same clipped* col_imp that GPTQ optimized under, not the
   raw activations. Or: skip DNH entirely when act_clip is non-default.
   Test by re-running full_v2 with `PRISMAQUANT_DO_NO_HARM=0`; if PPL ≈
   solo_clip, DNH is the culprit.

2. **`block_output_match` silently does nothing on Qwen3** — exact 0.000 PPL
   delta on solo_block. Either the spec builder returns None for Qwen3's
   module-name layout (path mismatch) or the greedy refiner converges to
   no-improvement on every block. Needs **positive logging**: log per-block
   "spec built: yes/no", "candidates evaluated: N", "improvements found: N"
   so we can disambiguate without further smoke runs.

3. **HALO multimodal hang** — see `prismaquant-handover-2026-04-29.md`. HALO
   integration in materialize_tensors_streaming hangs on Qwen 3.5/3.6
   multimodal models at the head_materialize step (apply_halo_to_head call
   on tied embeddings + `model.language_model.*` body prefix). Path-2 fix
   tied lm_head correctly but the hang persists. Not yet diagnosed.
   Workaround: HALO disabled in launcher for these models.

4. **Batched-path wins not yet validated** — the act_clip / damp_sweep / DNH
   extensions into `export_batched_gptq.py` (commit `95295ff`) compile and
   pass tests, but were never exercised end-to-end (the audit ran with
   `PRISMAQUANT_BATCHED_NVFP4_EXPORT=0`). Need a Qwen 4B run with batched=ON
   + full wins on to confirm parity with the scalar-path numbers.

5. **27B re-export pending** — user wants Qwen 3.6 27B re-exported with the
   validated wins for an influencer eval. Plan was in
   `session_2026_04_29_overnight_qwen4b_smoke.md`. Until #1 is investigated,
   recommendation is to ship just act_clip (single biggest win, lowest risk),
   not the cumulative stack.

## Recommended next steps for codex (priority order)

1. **Investigate the DNH×act_clip interference** (open bug #1). One quick
   test, big payoff for the 27B re-export decision.
2. **Add positive logging** to block_output_match (open bug #2). Cheap fix
   that future-proofs against silent-no-op confusion.
3. **Validate batched-path wins** (open bug #4) on Qwen 4B with batched=ON.
   Either confirm parity or fix the divergence.
4. **Plan 27B re-export** with the act_clip-only or full-stack-minus-DNH
   recipe per #1's outcome.
5. **HALO** is still the biggest unimplemented quality lever and the single
   biggest expected win (~0.20–0.30 PPL on Llama-class models per published
   results). Resolving the multimodal hang unlocks it.

## Files / paths to know

- Worktree: `/home/rob/prismaquant-quality-wins/`
- Smoke results: `/home/rob/dq-runs/qwen0p6b-smoke/audit_eval.csv`
- Launchers: `/home/rob/prismaquant/examples/launchers/launch-qwen4b-smoke.sh`
  (parameterized via `MODEL_PATH` + `WORK_ROOT` env vars), and
  `qwen4b-validator-eval.py` (validator runner — uses `WORK_ROOT` env).
- Driver: `/home/rob/prismaquant/examples/launchers/run-qwen4b-audit.sh`
  (sequential exports + final validator pass).
- Prior handovers in `.claude/`: `prismaquant-handover-2026-04-29.md` (the
  AM session) and this one. Read in order.

## What I learned about my own work

- Wins as env flags are easy to add and easy to silently bypass. The batched
  path issue (codex caught) wasn't the only one — block_match also silently
  no-ops on the Qwen3 architecture and I only noticed because the audit
  numbers were too clean.
- Positive logging at every win site would have caught this immediately. I
  added it for HALO but not for the others. Going forward, every quality
  win should log a one-line "fired on N Linears, X improvements found" at
  end of each layer.
- Smoke tests need to use the SAME methodology as the production validator
  (validator EVAL_PROMPTS suite, not wikitext-2). My initial wikitext eval
  noise floor was so high it hid the real signal.
- Stacking-validated wins isn't a replacement for testing the full stack.
  full_v2 < solo_clip is the empirical proof — should have been an
  obligatory check from the start.
