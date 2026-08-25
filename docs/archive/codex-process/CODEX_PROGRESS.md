MAJOR-M1 FIXED a92eb7f
MAJOR-M2 FIXED a92eb7f
MAJOR-M3 FIXED a92eb7f
MAJOR-M4 FIXED (close-skips 2026-06-22) — danger-half (silent-RTN confound + wasted KL hours) ALREADY-RESOLVED by fail-loud _materialize_assignment_inplace (validate_assignments_kl.py:577,443); the M4-named pre-check now actually fires for packed experts (include_packed_experts=True at _production_cache_assignment_diagnostics) with an actionable hint pointing at SELECTION_MODE=surrogate. Capability gap (validated-surrogate cannot ship a MoE artifact) declared unsupported — flagship is dense, experts ship RTN by policy (moe_expert_gptq_vs_rtn), so per-Pareto expert render (Option B) is wasted. Pinned: test_diagnostics_counts_packed_expert_misses_m4.
MAJOR-M5 WONTFIX-PRINCIPLED (close-skips 2026-06-22) — RTN-cost/GPTQ-bytes split is the served-VALIDATED production-faithful design, NOT a confound: served 27B KL A/B (memory aura_cost_rtn_vs_gptq_crossover, render held constant, bit-exact probe replay) = fp4 WASH (-0.0031, inside seed std), fp8 RTN-cost +36% BETTER. Forcing GPTQ-byte cost buys nothing at fp4, regresses fp8, and adds the dead-last/70x-slower per-expert GPTQ render into the cost step (Principle 3 + GPU-first). No functional change.
MAJOR-M6 FIXED 3d2abcd
MAJOR-M7 STALE 0bd5d9c
MAJOR-M8 FIXED c856a6a
MAJOR-M9 FIXED c856a6a
MAJOR-M10 FIXED+WONTFIX-FOR-NOW (close-skips 2026-06-22) — silent-omission half ALREADY-RESOLVED by c856a6a (_guard_packed_expert_coverage aura_cost.py:253-275,310 raises on any packed expert unless --allow-packed-expert-omission; pinned tests/test_aura_cost.py:296-316). Hybrid empirical-serving-unit expert AURA cost is WONTFIX-FOR-NOW: the codex gate (.claude/codex-aura-moe-gate-2026-06-09) proved AURA's gradient-probe estimator is structurally blind to routed-expert route-flip discontinuities (~10x format-gap error), so adding expert rows would violate Principle 1; AURA is the dense/attn spine, MoE ships via production-render-score; nothing ships via AURA-MoE today (no COST_MODE=aura in run-pipeline). Build only if a MoE-AURA artifact is roadmapped (gated on task #14).
MAJOR-M11 FIXED 722ec93 (verified close-skips 2026-06-22) — validated-frontier selection KL is held-out BY CONSTRUCTION: VALIDATED_FRONTIER_SKIP_CALIB defaults to NSAMPLES (run-pipeline.sh:190) so validation windows [NSAMPLES, NSAMPLES+VFN) are disjoint from the probe's [0,NSAMPLES); --calib-seed threaded on the --dataset path (validate_assignments_kl.py:493,511); optional VALIDATED_FRONTIER_DATASET for corpus separation. Pinned tests/test_validated_frontier_holdout.py. Ledger lagged; VALIDATED_FRONTIER_SKIP_CALIB=0 reproduces historical in-sample runs.
MAJOR-M12 FIXED d9457f1
MAJOR-M13 FIXED d9457f1
MAJOR-M14 FIXED d9457f1
MAJOR-M15 FIXED d9457f1
MAJOR-M16 FIXED 82baaa2
MAJOR-M17 FIXED 1da086b
MAJOR-M18 FIXED ada08a8
MAJOR-M19 FIXED+SERVED-VALIDATED (close-skips 2026-06-22, commit 6648c08) — the NVFP4 export re-derive now honors the render's RECORDED scale rule (cache.levers['nvfp4_scale_rule'], joint_mse when JSO on) instead of static_6, via PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE (default ON; =0 = legacy). NOT housekeeping — a real latent quality bug: 4B PAIRED A/B (same render, only the re-derive rule differs) = joint_mse 0.4699 KL / 26.017 PPL vs static_6 0.5033 / 26.892 → −6.6% served KL, −3.3% PPL, far outside the ~0.021 seed-noise band; the static_6 arm reproduces arm-D seed-42 exactly (legacy path byte-identical). Export was shipping every NVFP4 artifact ~6.6% worse-KL than its own validated render. Model-agnostic mechanism → re-exporting the flagship 27B would likely recover similar (measure before re-ship; see memory nvfp4_export_rederive_scale_fidelity). Pinned TestExportMatchRenderScaleRuleM19.
MAJOR-M20 FIXED 5bf80ac
MAJOR-M21 FIXED 3fd3dce
MAJOR-M22 FIXED 3fd3dce
MAJOR-M23 FIXED 718cfa3
MAJOR-M24 FIXED 0283d3f
MAJOR-M25 FIXED 0283d3f
MAJOR-M26 WONTFIX-SCREEN-THEN-GATE (close-skips 2026-06-22) — last_token frontier-selection KL is a triage SCREEN, and the chosen frontier point is re-validated on the served gold metric (validate_quantized_model + served KL A/B) before any ship, which is the methodological spine ("surrogates generate, real KL selects"); a mis-ranked screen point is caught by the served gate, not shipped. AVAILABLE UPGRADE (Robert's call, not required): Option A flips the run-pipeline default scope to full_sequence (gold-aligned per CLAUDE.md §5) — ~1 LoC + help-text fix, last_token reachable via VALIDATED_FRONTIER_KL_SCOPE for repro — but it changes the flagship-producing selection default and the full_sequence path needs an end-to-end validation run before defaulting.
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
MINOR-M2 WONTFIX-W-EVIDENCE + AVAILABLE-UPGRADE (close-skips 2026-06-22) — fit-weighting half is WONTFIX-with-evidence: the served 6-arm 35B finale chose RTN-static6 (memory moe_expert_gptq_vs_rtn) and arm F (MORE in-sample fit capacity: damp-sweep+act_order) served WORST on thin-Hessian routed experts, so row-weighting the GPTQ fit pushes the contraindicated direction (keep as a research lever, not a default). The cross-domain gate (0bd5d9c) is the served-validated GPTQ-vs-RTN policy and is reachable via --expert-gate-dataset. AVAILABLE UPGRADE (Robert's call): A1 wires a DISJOINT EXPERT_GATE_DATASET default into run-pipeline.sh for packed-MoE so a vanilla MoE run reproduces the proven recipe instead of the same-corpus n<20 in-sample fallback — needs a corpus choice + a served re-A/B (changes default MoE bytes).
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
MINOR-M33 FIXED — KV-cotangent path LANDED (measurement gap closed, guard INVERTED). The phase-3 reverse sweep now grafts grad-enabled leaves over a sharing layer's borrowed K/V, sums each consumer's leaf.grad per source (SharedStateCotangents, sensitivity_probe.py), and drives the STORING layer's backward with that sum alongside its own output cotangent (torch.autograd.backward([out,k,v],[g_out,gk,gv]) — one call, so each Linear's full-backward hook still fires once with the total gy). Consumers always sit above their producer and the sweep runs in reverse, so one pass suffices; no extra forwards, no disk state. ACCEPTANCE: fp64 synthetic-model equivalence vs ONE end-to-end autograd backward — cotangent rel err < 1e-12 and shipped h_trace BIT-IDENTICAL (rel err 0.0), while the pre-fix protocol under-counts the producer's k_proj by 85.1% and v_proj by 38.5% on the same fixture; the severed edge also truncated the chained grad_out of every layer BELOW the producer (all of layers.0 moved), which the fix repairs too. Guard now fires only when PRISMAQUANT_KV_COTANGENT=0 takes the path away (PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1 still overrides). No gemma4.py change needed — the profile's detached CPU capture is exactly right; the leaf is what carries the gradient. Pinned tests/test_kv_cotangent_path.py (17 tests) + rewritten tests/test_incremental_probe_kv_shared_guard.py. STILL UNMEASURED ON A REAL MODEL: no num_kv_shared_layers>0 checkpoint has been probed (no ship target yet), so the real-model magnitude is unknown — the toy percentages are a correctness demonstration, NOT a quality claim. NOTE: the ledger's separate 'MAJOR-M33 FIXED 46102e8' is a DIFFERENT finding (allocator incomplete-fused-group) — correctly fixed, not mislabeled.
MINOR-M34 FIXED 0ef119c
MINOR-M35 FIXED d7cbb09
MINOR-M36 FIXED 6a317f7
MINOR-M37 FIXED 081f73d
MINOR-M38 FIXED a8ef85a
MINOR-M39 FIXED 4ea6618
MINOR-M40 FIXED a9b703b
MINOR-M41 FIXED 7dc61c0
MINOR-M42 FIXED 7d24e70
INFO-I1 FIXED ed73ebf
INFO-I2 FIXED ed73ebf
INFO-I3 FIXED 95327e4
INFO-I4 FIXED 95327e4
INFO-I5 FIXED bcc9648
