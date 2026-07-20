# Hy3 295B ultra-low-bpp — NVFP4-CB lane (running log)

**Model:** tencent Hy3 295B-A21B (hy_v3, 80 layers, 192 experts top-8,
router `mlp.router.gate.weight` + `mlp.expert_bias`), BF16 source 557 GB
(non-preview release, verified). **NO QUALITY CLAIMS** (standing rule: a
295B cannot be KL-validated against its BF16 teacher on this box).
Validation = loads + coherent generation + bit-exact packing + speed vs
the shipped GGUF Hy3 2.8bpp (prefill 42 tok/s = the IQ tax the CB lane
exists to remove).

**Driver:** `scripts/run_hy3_prod_nvfp4cb.sh` — COST_MODE=local,
CB_EXPERT_EMPIRICAL=0 (+sample 16, ladder interp), two_tier (v2) coding,
TARGET_BITS=2.9, streaming export.

## Chain (all stages GPU, one box)
- Probe: 10×8-layer shards + tail (~1.7 h after LAYERS_PER_SHARD=8; auto
  had picked 1/shard ≈ 16 h).
- Cost: ~3 h, both rung-family floor-law ladders live.
- Allocation @2.9 (achieved 2.902): mixed-family body — experts
  fp4-CB (K16×36, K18×38, K20×30, K14×18) + fp8-CB K28×36; dense/attn
  fp8-CB (K32×327, K44×57, K36×33, K28×26) + fp4 tail; BF16 floor 57.
- Streaming export: ~105 GB expected, resumable (see ledger 5).

## First-contact bug ledger (2026-07-19, all committed)
1. **Streaming export box-OOM ×3** (global kernel OOM, exporter CPU RSS ~0,
   torch alloc 0 — invisible GPU-side): root cause
   `_pack_codes_to_bytes` materialized the (rows, nvec, k) int64 bitstream
   twice = **~155 GB** on a (192,3072,4096) expert stack. Fix: row-chunked
   bit-pack, bit-identity verified k=12..48 + roundtrip (`b4a8b3f`).
   Synthetic E=192 pack peak 173 → 27 GB. Diagnosed OFFLINE at 1/32 scale
   under `set_per_process_memory_fraction` after the live-relaunch pattern
   burned two extra runs — repro-before-relaunch is the rule.
2. Full broadcast col_weights copy (~10 GB/stack) → per-block gather,
   identity-verified (same commit).
3. Exporter now self-caps at 75% of the unified pool: future runaways
   raise a clean torch OOM naming the tensor, never a box-wide kill.
4. Unbounded open shard mmaps (233-shard source → ~1 TB VM) → LRU-4
   (`06e653a`). Not the OOM cause, but hygiene worth keeping.
5. **Resume support** in the streaming writer (`be300dd`): analytic
   offsets + deterministic producers → partial file resumes at the first
   incomplete entry (header must match bit-for-bit; sibling state groups
   backed up together). Byte-identity tested at 5 cut points. Salvaged the
   12 GB partial (resumed 763/2271).
6. **v2 encode pace**: the W×16 two-tier entry loop was launch-bound
   (33.6 s of 46.8 s at E=24 in 16,128×2 ms kernels) → batched via
   `_moment_err_groups_batched`, **2.9×, bit-identical across 32 configs**
   (torch.min first-occurrence == strict-<-first-legal). Expert stack
   ~15 → ~3 min; export ETA ~30 h → ~7-10 h (same commit).
7. Ops: export runs detached (`setsid`) — two of the four kills were
   session-side SIGKILLs, not the box. Monitor deadman keys on ARTIFACT
   mtime, not log mtime (20-entry log cadence stretches >30 min across
   expert stacks); pgrep patterns bracket-escaped (`pro[d]`) against
   self-match.

## Footprint (exact, read from the pre-written streaming header mid-export)
- **model.safetensors = 110.3 GB** (102.7 GiB): U8 cb_qweight 99.68 GB +
  BF16 sidecars 10.52 GB (57 BF16 layers) + F32 scales 0.10 GB. 2271
  tensors, no double-ship. Codebook sidecar (lattice, shared per role)
  negligible. This is **above the ~105 GB estimate** — 2.902 bpp over
  quantizable params understates the on-disk total because the BF16 floor
  is bpp-excluded but disk-resident.
- **Single-Spark fit: YES, context-capped.** 110.3 GB weights + ~2 GB
  framework on the ~121 GB usable pool. CB decode is transient (no load
  expansion), so resident weight footprint == disk. KV headroom (GQA 8
  kv-heads × 128 × 80 layers × fp16 = 0.328 MB/token):
  - 8k ctx: 113.0 GB resident (~8 GB spare) — comfortable
  - 16k ctx: 115.7 GB resident (~5 GB spare) — OK
  - 32k ctx: 121.0 GB — at the ceiling, no room for activations
  Native 262k context does NOT fit (true of any ~110 GB weight class on
  this box). **Serve with --max-model-len 8192–16384.** If practice shows
  the headroom too thin, 2.7 bpp is the PARETO_TARGETS fallback rung.

## Serve first-contact (2026-07-20)
- Artifact integrity: safetensors COMPLETE (110.3 GB, header end ==
  file size), config HYV3ForCausalLM + layout_version 2 + two_tier,
  cb_codebooks.pqcb loads via safetensors (24 shared lattice codebooks,
  sub-structure correct per rung). Codebooks are NOT torch.load/pickle —
  they are a safetensors file with a .pqcb extension (probe trap).
- hy_v3 serving arch: HYV3ForCausalLM + HYV3MTPModel are NATIVE in
  `vllm-node:latest` (vLLM 0.23.1-dev, tf 5.13) — the run-script's
  pre-launch gate (d) wrongly credited the GGUF work (that was
  llama.cpp, no vLLM adapter); the image covers it regardless.
- **Bug 9 (serving loader, hy_v3): stacked CB expert tensors KeyError at
  load.** `HYV3ForCausalLM.load_weights` loads experts at the TOP-LEVEL
  model via `expert_params_mapping` (per-expert names) and never calls the
  per-layer `FusedMoE.load_weights` the plugin wraps — so that wrap is dead
  code here (it works for Qwen3.5-MoE/35B, which delegates per-layer). Our
  stacked `experts.{gate_up_proj,down_proj}.cb_qweight` match no per-expert
  mapping → final `params_dict[name]` KeyError vs registered
  `experts.w13/w2_cb_qweight`. Fix: plugin-installed model-level
  load_weights wrap mapping stacked CB expert names → fused params (plain
  copy), delegating everything else. This is the top-level-loader analog of
  moe.py's per-layer wrap; generalizes to DSv4 (same convention). Serve
  memory params for the smoke: --enforce-eager --max-model-len 8192
  --gpu-memory-utilization 0.95 (110 GB weights leave ~5 GB for KV).

## Serve-bringup bug chain (2026-07-20, first serve of a top-level-loader MoE)
Each is a distinct hy_v3-specific first-contact bug; the artifact bytes are
correct (proven on disk), these are all serving-adapter gaps.
- **Bug 9a — top-level expert loader (FIXED, committed f202841 then
  refined).** hy_v3 loads experts at the top model via expert_params_mapping,
  bypassing the per-layer FusedMoE.load_weights the plugin wraps. Plugin-
  installed model-level load_weights wrap. The wrap install runs in the
  EngineCore process (verified via marker).
- **Bug 9b — routed_experts nesting (FIXED).** hy_v3's SharedFusedMoE nests
  the routed FusedMoE one level deeper: params are at
  `…mlp.experts.routed_experts.w13/w2_cb_qweight`, not `…mlp.experts.w13…`.
  A fixed-string suffix rewrite missed. Fix: `resolve_cb_expert_param`
  resolves the target by (`…mlp.experts.` prefix, leaf suffix) against the
  ACTUAL named_parameters — robust to this nesting and any future one.
  Confirmed working: both routed_experts params load.
- **Bug 9c — shared expert built unquantized (FIX IN FLIGHT).** hy_v3 passes
  `shared_experts=self.shared_mlp` into FusedMoE; vLLM builds the shared-MLP
  Linears as PLAIN BF16 `.weight` and NEVER calls the quant config's
  get_quant_method for them (proven: instrumented print never fired). But the
  export quantized shared_mlp to CB (190 tensors, 0.58 GB packed) → KeyError.
  Fix: decode the CB shared_mlp tensors to bf16 at load and populate the
  `.weight` params (the plugin already decodes CB at prefill; ~+1.7 GB
  resident). Unfused checkpoint gate/up → cat into vLLM's fused
  gate_up_proj.weight. NOTE for the clean re-export path: a hy_v3 serving
  profile should force shared_mlp→BF16 (like Gemma incomplete-fused groups),
  making this decode-at-load unnecessary — but that needs a re-export; the
  decode serves the existing artifact now.

## Pending
- After bug 9c: KV profiling at util 0.95 (tight — 110→112 GB resident,
  serve --max-model-len 4096-8192), first forward (attn + fp4-v2 grouped
  routed MoE + decoded shared expert), possible further first-contact bugs.
- Then: prefill vs GGUF 42 tok/s (the thesis number) + decode with the
  fp4-v2 grouped kernel (fa7cc90).
- Serve smoke: hy_v3 adapter in the serving image, v2-compose fp4-CB
  dense at scale, fp4-CB MoE via v2 (grouped decode kernel is fp8-only —
  the known extension for decode parity).
- Speed vs GGUF Hy3 2.8bpp: prefill is the thesis number; decode target
  parity after the grouped fp4-v2 extension.
