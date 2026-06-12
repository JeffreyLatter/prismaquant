MAJOR-M1 FIXED a92eb7f
MAJOR-M2 FIXED a92eb7f
MAJOR-M3 FIXED a92eb7f
MAJOR-M4 SKIPPED-NEEDS-DESIGN proposal: validated-surrogate on packed-MoE must either render expert entries for each Pareto assignment or fail/route to a selected-assignment production-cache build before KL/export; format-menu has no concrete expert assignment to render today.
MAJOR-M5 SKIPPED-NEEDS-DESIGN proposal: packed-expert allocator cost needs production render-score records from fill_packed_expert_cache_entries or an explicit hybrid cost contract; current RTN-cost/GPTQ-bytes split is real but changing cost semantics affects the paper-spine allocator.
MAJOR-M6 FIXED 3d2abcd
MAJOR-M7 STALE 0bd5d9c
MAJOR-M8 FIXED c856a6a
MAJOR-M9 FIXED c856a6a
MAJOR-M10 SKIPPED-NEEDS-DESIGN proposal: implement the codex-agreed empirical/hybrid packed-expert AURA cost with additivity gate; c856a6a adds fail-fast/explicit-escape so packed experts are no longer silently omitted.
MAJOR-M11 SKIPPED-NEEDS-DESIGN proposal: pipeline held-out validation wiring overlaps the queue's critical run-pipeline holdout branch; a92eb7f fixed validator/build seed forwarding, and the critical branch should pass a distinct held-out dataset/split/seed into validated-frontier selection.
MAJOR-M12 FIXED d9457f1
MAJOR-M13 FIXED d9457f1
MAJOR-M14 FIXED d9457f1
MAJOR-M15 FIXED d9457f1
MAJOR-M16 FIXED 82baaa2
MAJOR-M17 FIXED 1da086b
MAJOR-M18 FIXED ada08a8
MAJOR-M19 SKIPPED-NEEDS-DESIGN proposal: persist production-cache render artifacts at the packed-code/scale or render-metadata level so export can ship exactly the validated bytes; changing the cache payload/selection contract is larger than a queue-batch patch.
MAJOR-M20 FIXED 5bf80ac
MAJOR-M21 FIXED 3fd3dce
MAJOR-M22 FIXED 3fd3dce
MAJOR-M23 FIXED 718cfa3
MAJOR-M24 FIXED 0283d3f
MAJOR-M25 FIXED 0283d3f
MAJOR-M26 SKIPPED-NEEDS-DESIGN proposal: final validated-frontier KL scope/defaults overlap the queue's protected run-pipeline holdout/stage wiring; switch production final selection to a full-sequence held-out evaluator in that critical branch, or require an explicit smoke-only last-token override.
MAJOR-M27 FIXED 013b0c2
MAJOR-M28 FIXED 0b359eb
MAJOR-M29 FIXED 29f80d3
MAJOR-M30 FIXED 5ceccc2
MAJOR-M31 FIXED 5ceccc2
MAJOR-M32 STALE a92eb7f
MAJOR-M33 FIXED 46102e8
MAJOR-M34 FIXED a92eb7f
MAJOR-M35 FIXED 4b3f525
MINOR-M1 FIXED bc50a6c
MINOR-M2 SKIPPED-NEEDS-DESIGN proposal: replace the sparse-expert GPTQ-vs-RTN gate with a measured cross-domain/min-row policy and fit/eval weighting contract; changing the <20-row in-sample fallback would alter production expert-render semantics.
MINOR-M3 FIXED bc50a6c
MINOR-M4 FIXED bc50a6c
MINOR-M5 FIXED bc50a6c
MINOR-M6 FIXED bc50a6c
MINOR-M7 FIXED bc50a6c
MINOR-M8 FIXED bc50a6c
MINOR-M9 FIXED bc50a6c
MINOR-M10 FIXED bc50a6c
MINOR-M11 FIXED bc50a6c
MINOR-M12 FIXED bc50a6c
MINOR-M13 FIXED 03e3348
MINOR-M14 FIXED 03e3348
MINOR-M15 FIXED 03e3348
MINOR-M16 FIXED 03e3348
MINOR-M17 FIXED 7fc514c
MINOR-M18 FIXED 7fc514c
MINOR-M19 STALE a92eb7f
MINOR-M20 FIXED a07316c
MINOR-M21 FIXED 767d000
MINOR-M22 FIXED bc50a6c
MINOR-M23 FIXED bc50a6c
MINOR-M24 FIXED f679d2f
MINOR-M25 FIXED 2259287
MINOR-M26 FIXED 0ef119c
MINOR-M27 FIXED a026db4
MINOR-M28 FIXED 4549a18
MINOR-M29 FIXED 4549a18
MINOR-M30 FIXED f3a1241
MINOR-M31 FIXED f3a1241
MINOR-M32 FIXED eee0beb
MINOR-M33 SKIPPED-NEEDS-DESIGN proposal: model shared-KV as an explicit cross-layer gradient path in the Fisher pass-state contract; a local patch would either undercount silently or retain multi-layer KV graphs and break the isolated GPU-bound streaming design.
MINOR-M34 FIXED 0ef119c
