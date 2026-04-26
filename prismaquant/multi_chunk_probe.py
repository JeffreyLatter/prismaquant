"""In-process multi-chunk REAP probe driver.

Loads the source model + StreamingContext + LayerCache once, then iterates
N calibration chunks through the standard `incremental_probe.main()` loop
with per-chunk argument overrides. The shared StreamingContext is held in
`incremental_probe._PROBE_CTX_CACHE` (keyed by model+device+dtype) so
chunks 1..N hit warm offload + warm LayerCache instead of paying the
~5-10 min cold-start cost per chunk.

The Fisher accumulators across chunks are merged at the end via the
existing `merge_probe_pickles` (which sums additive fields like
h_trace_raw, h_w2_sum_raw, h_full per-weight, expert_saliency, and
n_tokens_seen). The activation cache from the LAST chunk is retained
for the cost stage.

Adaptive sampling (v21 #3) is opt-in via `--adaptive-sampling`. When
on, the driver tracks per-domain saliency via
`prismaquant.adaptive_sampling.AdaptiveExpertScheduler` and narrows
each chunk's `--linear-include` to skip experts whose rank has
stabilized in every observed domain. This is what lets later chunks
focus calibration tokens on contested experts instead of paying the
full per-chunk cost. Domain inference uses the chunk filename:
`chunk_<domain>_<idx>.jsonl` → domain `<domain>`; `chunk_<idx>.jsonl`
falls back to `_global` so single-domain runs work unchanged.

Per-domain saliency (v21): the merged probe.pkl gains an
`expert_saliency_per_domain[domain][router_q][expert_id]` field built
by token-weighted averaging across chunks tagged with that domain. The
allocator can consume this for union/intersection prune policies.

Usage:
    python -m prismaquant.multi_chunk_probe \\
        --chunks-dir /path/with/chunk_*.jsonl \\
        --model <hf_model_path> \\
        --output /path/to/merged_probe.pkl \\
        --activation-cache-dir /path/to/act \\
        --work-dir /path/to/work_root \\
        --h-detail-dir /path/to/h_detail \\
        --layers-per-shard 8 --prefetch-lookahead 4 \\
        [--adaptive-sampling] [--retain-cross-chunk-cache] \\
        [other incremental_probe args]
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

# Force the in-process ctx cache before importing the probe module so the
# very first ensure_ready call writes to the cache.
os.environ.setdefault("PRISMAQUANT_PROBE_CTX_CACHE", "1")

from prismaquant import incremental_probe as ip
from prismaquant import incremental_measure_quant_cost as cost_step
from prismaquant.adaptive_sampling import (
    AdaptiveExpertScheduler,
    aggregate_global_saliency,
    aggregate_per_domain_saliency,
    infer_chunk_domain,
)


def _make_chunk_argv(base_argv: list[str], chunk_jsonl: Path,
                     chunk_work_dir: Path, chunk_output: Path,
                     drop_keys: tuple[str, ...],
                     extra_args: list[str] | None = None) -> list[str]:
    """Rebuild sys.argv for one chunk by overriding --dataset, --work-dir,
    --output and stripping the multi-chunk-only keys.

    `extra_args` is appended at the end so adaptive-sampling overrides
    (e.g. a narrowed --linear-include) win over any earlier values.
    """
    out: list[str] = [base_argv[0]]
    skip_next = False
    skip_keys = tuple(drop_keys) + (
        ("--linear-include",) if extra_args else ()
    )
    for tok in base_argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in skip_keys:
            skip_next = True
            continue
        if any(tok.startswith(k + "=") for k in skip_keys):
            continue
        out.append(tok)
    # Append chunk-specific overrides (these win over any earlier values).
    out += [
        "--dataset", str(chunk_jsonl),
        "--work-dir", str(chunk_work_dir),
        "--output", str(chunk_output),
    ]
    if extra_args:
        out += extra_args
    return out


def _extract_arg_value(argv: list[str], key: str) -> str | None:
    """Pull a `--key VALUE` or `--key=VALUE` value out of an argv list."""
    for i, tok in enumerate(argv):
        if tok == key and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(key + "="):
            return tok[len(key) + 1:]
    return None


def _summarize_scheduler(sched: AdaptiveExpertScheduler) -> str:
    s = sched.summary()
    return (f"experts: total={s['total']} "
            f"frozen-keep={s['frozen_keep']} "
            f"frozen-drop={s['frozen_drop']} "
            f"contested={s['contested']} "
            f"chunks_by_domain={s['chunks_by_domain']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks-dir", required=True,
                    help="Directory containing chunk_NN.jsonl files")
    ap.add_argument("--output", required=True,
                    help="Final merged probe.pkl path")
    ap.add_argument("--work-dir", required=True,
                    help="Work root; per-chunk subdirs will be created here")
    ap.add_argument("--retain-cross-chunk-cache", action="store_true",
                    default=False,
                    help="Keep LayerCache contents across chunk boundaries. "
                         "Layer weights are model-invariant, so retaining "
                         "the cache makes chunk N+1's phase-1 forward warm "
                         "on the layers that survived end of chunk N. "
                         "Adds ~70 GB of resident bytes at chunk transitions; "
                         "safe on Spark (cache budget already accounts for it) "
                         "but disable on smaller boxes if memory is tight.")
    ap.add_argument("--adaptive-sampling", action="store_true",
                    default=False,
                    help="Enable per-domain expert saliency tracking and "
                         "narrow each chunk's --linear-include to skip "
                         "experts whose rank has stabilized across all "
                         "observed domains. Off by default for byte-for-"
                         "byte parity with v20 launches.")
    ap.add_argument("--adaptive-min-chunks", type=int, default=2,
                    help="Minimum number of observations per domain before "
                         "an expert can be marked frozen (default: 2).")
    ap.add_argument("--adaptive-stability", type=float, default=0.10,
                    help="Relative-range stability threshold across the "
                         "stability window. Smaller = stricter (default: 0.10).")
    ap.add_argument("--adaptive-keep-band", type=float, default=0.25,
                    help="Top fraction of router rank that counts as "
                         "frozen-keep (default: 0.25 → top 25%).")
    ap.add_argument("--adaptive-drop-band", type=float, default=0.10,
                    help="Bottom fraction of router rank that counts as "
                         "frozen-drop (default: 0.10 → bottom 10%).")
    ap.add_argument("--adaptive-disagreement-spread", type=float, default=0.5,
                    help="Per-domain rank-spread threshold above which an "
                         "expert stays contested even if individually "
                         "stable in each domain (default: 0.5).")
    ap.add_argument("--adaptive-prune-ratio", type=float, default=0.375,
                    help="Approximate prune cutoff used to define the "
                         "contested band around the prune decision "
                         "(default: 0.375, matching MiniMax launch).")
    # v21 #2: in-process cost step. Sequential rather than concurrent
    # — the multi-chunk probe pickle isn't ready until all chunks
    # complete, so the cost step still runs after probing is done.
    # The win is amortizing process / interpreter / streaming-context
    # build cost vs launching cost as a separate Python invocation.
    ap.add_argument("--run-cost", action="store_true", default=False,
                    help="After the merged probe.pkl is written, invoke "
                         "incremental_measure_quant_cost in-process. "
                         "Probe StreamingContext is torn down first so "
                         "the cost step's context build doesn't double "
                         "the UMA memory footprint.")
    ap.add_argument("--cost-output", default=None,
                    help="cost.pkl output path (required if --run-cost).")
    ap.add_argument("--cost-formats", default="NVFP4,MXFP8_E4M3,FP8_SOURCE,BF16",
                    help="Comma-separated format list for the cost step.")
    ap.add_argument("--cost-work-dir", default=None,
                    help="Cost-step work dir; defaults to "
                         "<work-dir>/cost_work when --run-cost.")
    ap.add_argument("--cost-mode", choices=["auto", "batched", "unbatched"],
                    default="batched")
    ap.add_argument("--cost-chunk-size", type=int, default=256)
    ap.add_argument("--cost-layers-per-shard", default=None,
                    help="Cost layers-per-shard. Defaults to the probe's "
                         "(picked up from passthrough args).")
    ap.add_argument("--cost-h-detail-dir", default=None,
                    help="Cost h-detail dir; off by default — h-detail "
                         "writes ~940 GB/chunk on MiniMax, defer unless "
                         "the disk has headroom.")
    args, passthrough = ap.parse_known_args()
    if args.retain_cross_chunk_cache:
        os.environ["PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK"] = "1"
        print("[multi-chunk] cross-chunk LayerCache retention: ENABLED",
              flush=True)

    chunks_dir = Path(args.chunks_dir)
    chunk_jsonls = sorted(chunks_dir.glob("chunk_*.jsonl"))
    if not chunk_jsonls:
        print(f"[multi-chunk] no chunk_*.jsonl files in {chunks_dir}",
              file=sys.stderr)
        return 1

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[multi-chunk] {len(chunk_jsonls)} chunks; sharing model+ctx "
          f"across all of them via PRISMAQUANT_PROBE_CTX_CACHE", flush=True)

    base_argv = ["prismaquant.incremental_probe"] + passthrough
    drop_keys = ("--chunks-dir", "--dataset", "--output", "--work-dir")
    base_linear_include = _extract_arg_value(base_argv, "--linear-include")

    sched: AdaptiveExpertScheduler | None = None
    sched_state_path = work_root / "adaptive_scheduler.json"
    if args.adaptive_sampling:
        if sched_state_path.exists():
            sched = AdaptiveExpertScheduler.load(sched_state_path)
            print(f"[multi-chunk] adaptive scheduler resumed from "
                  f"{sched_state_path} ({_summarize_scheduler(sched)})",
                  flush=True)
        else:
            sched = AdaptiveExpertScheduler(
                prune_ratio=args.adaptive_prune_ratio,
                stability_threshold=args.adaptive_stability,
                min_chunks_for_freeze=args.adaptive_min_chunks,
                keep_band=args.adaptive_keep_band,
                drop_band=args.adaptive_drop_band,
                disagreement_spread=args.adaptive_disagreement_spread,
            )
            print(f"[multi-chunk] adaptive sampling enabled: "
                  f"min_chunks={args.adaptive_min_chunks} "
                  f"stability={args.adaptive_stability} "
                  f"keep={args.adaptive_keep_band} "
                  f"drop={args.adaptive_drop_band} "
                  f"prune={args.adaptive_prune_ratio} "
                  f"spread={args.adaptive_disagreement_spread}",
                  flush=True)

    chunk_outputs: list[Path] = []
    chunk_domains: list[str] = []

    for i, chunk_jsonl in enumerate(chunk_jsonls):
        chunk_work_dir = work_root / f"chunk_{i:02d}"
        chunk_output = chunk_work_dir / "probe.pkl"
        domain = infer_chunk_domain(chunk_jsonl)
        chunk_domains.append(domain)

        if chunk_output.exists():
            print(f"[multi-chunk] chunk {i}: resume — {chunk_output} "
                  f"already exists, skipping (domain={domain})", flush=True)
            chunk_outputs.append(chunk_output)
            # Even on resume, fold this chunk's saliency into the
            # scheduler so subsequent chunks see consistent state.
            if sched is not None:
                with chunk_output.open("rb") as f:
                    pkl = pickle.load(f)
                sched.update_from_chunk_pickle(pkl, domain)
            continue
        chunk_work_dir.mkdir(parents=True, exist_ok=True)

        # Adaptive sampling: narrow --linear-include based on what
        # the scheduler considers contested. Falls back to base
        # include on the first chunk (no history yet).
        extra_args: list[str] = []
        if sched is not None and base_linear_include is not None:
            # Read expert_info from the previous chunk's pickle so the
            # narrowing has Linear→(router, eid) mapping. On chunk 0
            # there isn't one yet — fall through to base include.
            expert_info: dict[str, tuple[str, str]] = {}
            for prev in chunk_outputs:
                with prev.open("rb") as f:
                    prev_pkl = pickle.load(f)
                expert_info.update(prev_pkl.get("expert_info") or {})
            narrowed = sched.linear_include_for_next_chunk(
                base_include=base_linear_include,
                expert_info=expert_info,
            )
            if narrowed != base_linear_include:
                extra_args += ["--linear-include", narrowed]
                n_frozen = sum(
                    len(v) for v in sched.frozen_experts().values())
                print(f"[multi-chunk] chunk {i}: adaptive narrow — "
                      f"{n_frozen} frozen experts excluded; "
                      f"{_summarize_scheduler(sched)}", flush=True)

        chunk_argv = _make_chunk_argv(
            base_argv, chunk_jsonl, chunk_work_dir, chunk_output, drop_keys,
            extra_args=extra_args,
        )

        # Tag the chunk with its domain so the per-chunk pickle's meta
        # carries it. incremental_probe.main() reads this env var and
        # stashes it into meta["domain"].
        prior_domain = os.environ.get("PRISMAQUANT_PROBE_DOMAIN")
        os.environ["PRISMAQUANT_PROBE_DOMAIN"] = domain

        print(f"\n[multi-chunk] === chunk {i+1}/{len(chunk_jsonls)} ===",
              flush=True)
        print(f"[multi-chunk] dataset={chunk_jsonl.name} "
              f"domain={domain} "
              f"work_dir={chunk_work_dir.name}", flush=True)
        t0 = time.time()
        saved_argv = sys.argv
        sys.argv = chunk_argv
        try:
            ip.main()
        finally:
            sys.argv = saved_argv
            if prior_domain is None:
                os.environ.pop("PRISMAQUANT_PROBE_DOMAIN", None)
            else:
                os.environ["PRISMAQUANT_PROBE_DOMAIN"] = prior_domain
        elapsed = time.time() - t0
        print(f"[multi-chunk] chunk {i+1} done in {elapsed:.0f}s",
              flush=True)
        chunk_outputs.append(chunk_output)

        # Fold this chunk's saliency into the scheduler and persist.
        if sched is not None:
            with chunk_output.open("rb") as f:
                pkl = pickle.load(f)
            n = sched.update_from_chunk_pickle(pkl, domain)
            sched.save(sched_state_path)
            print(f"[multi-chunk] adaptive scheduler updated: +{n} "
                  f"(router,expert) entries for domain={domain}; "
                  f"{_summarize_scheduler(sched)}", flush=True)

    print(f"\n[multi-chunk] merging {len(chunk_outputs)} per-chunk "
          f"probe.pkls -> {out_path}", flush=True)
    ip.merge_probe_pickles(chunk_outputs, out_path)

    # Build per-domain + globally-aggregated saliency maps from raw
    # per-chunk pickles using the token-weighted average so the merged
    # value is correct regardless of differing chunk sizes.
    per_chunk_pairs: list[tuple[dict, str]] = []
    for p, d in zip(chunk_outputs, chunk_domains):
        with p.open("rb") as f:
            per_chunk_pairs.append((pickle.load(f), d))
    expert_saliency_per_domain = aggregate_per_domain_saliency(per_chunk_pairs)
    expert_saliency_global = aggregate_global_saliency(
        expert_saliency_per_domain, per_chunk_pairs)

    with out_path.open("rb") as f:
        merged = pickle.load(f)
    merged["expert_saliency"] = expert_saliency_global
    merged["expert_saliency_per_domain"] = expert_saliency_per_domain
    meta = dict(merged.get("meta", {}))
    meta["multi_chunk_count"] = len(chunk_outputs)
    meta["multi_chunk_total_nsamples"] = sum(
        pkl.get("meta", {}).get("nsamples", 0)
        for pkl, _ in per_chunk_pairs
    )
    meta["chunk_domains"] = chunk_domains
    meta["adaptive_sampling"] = bool(args.adaptive_sampling)
    if sched is not None:
        meta["adaptive_summary"] = sched.summary()
    merged["meta"] = meta
    with out_path.open("wb") as f:
        pickle.dump(merged, f)
    n_domains = len(expert_saliency_per_domain)
    n_routers = sum(len(v) for v in expert_saliency_per_domain.values())
    print(f"[multi-chunk] DONE — merged probe at {out_path} "
          f"(total_nsamples={meta['multi_chunk_total_nsamples']}, "
          f"per-domain saliency: {n_domains} domains × "
          f"~{n_routers // max(n_domains, 1)} routers)", flush=True)

    # v21 #2: in-process cost step. Sequential after the merged probe
    # is written. The cost step builds its own StreamingContext, so we
    # tear down the probe ctx first to keep peak UMA memory bounded.
    if args.run_cost:
        if not args.cost_output:
            print("[multi-chunk] --run-cost requires --cost-output",
                  file=sys.stderr)
            return 2
        # Tear down the probe's persistent ctx so cost can build its own
        # without doubling UMA pressure. After this point the probe is
        # done and no further chunk loop runs in this process.
        try:
            for key, (ctx, _tok) in list(ip._PROBE_CTX_CACHE.items()):
                try:
                    ctx.shutdown()
                except Exception as e:
                    print(f"[multi-chunk] ctx shutdown raised: {e!r}",
                          file=sys.stderr)
                ip._PROBE_CTX_CACHE.pop(key, None)
        except Exception as e:
            print(f"[multi-chunk] probe ctx teardown failed (continuing): "
                  f"{e!r}", file=sys.stderr)
        # Force a gc + cuda empty before cost starts so the cost ctx
        # build sees the freshest MemAvailable.
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        cost_work_dir = (
            Path(args.cost_work_dir) if args.cost_work_dir
            else (work_root / "cost_work")
        )
        cost_argv = [
            "prismaquant.incremental_measure_quant_cost",
            "--model", _extract_arg_value(base_argv, "--model") or "",
            "--probe", str(out_path),
            "--activation-cache-dir",
            _extract_arg_value(base_argv, "--activation-cache-dir") or "",
            "--output", str(args.cost_output),
            "--work-dir", str(cost_work_dir),
            "--device", _extract_arg_value(base_argv, "--device") or "cuda",
            "--dtype", _extract_arg_value(base_argv, "--dtype") or "bf16",
            "--formats", args.cost_formats,
            "--mode", args.cost_mode,
            "--chunk-size", str(args.cost_chunk_size),
            "--skip-missing-activations",
        ]
        cost_lps = (
            args.cost_layers_per_shard
            or _extract_arg_value(base_argv, "--layers-per-shard")
            or "auto"
        )
        cost_argv += ["--layers-per-shard", str(cost_lps)]
        if args.cost_h_detail_dir:
            cost_argv += ["--h-detail-dir", args.cost_h_detail_dir]
        # Sanity check the model & activation-cache-dir survived the
        # passthrough split — a missing value here would silently
        # produce a useless cost.pkl, so loud-fail instead.
        if not cost_argv[3] or not cost_argv[7]:
            print("[multi-chunk] --run-cost requires --model and "
                  "--activation-cache-dir to be in the probe passthrough "
                  "args.", file=sys.stderr)
            return 2

        print(f"\n[multi-chunk] === cost step (in-process) ===", flush=True)
        print(f"[multi-chunk] cost output: {args.cost_output}", flush=True)
        print(f"[multi-chunk] cost work-dir: {cost_work_dir}", flush=True)
        t_cost = time.time()
        saved_argv = sys.argv
        sys.argv = cost_argv
        try:
            cost_step.main()
        finally:
            sys.argv = saved_argv
        print(f"[multi-chunk] cost step done in "
              f"{time.time()-t_cost:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
