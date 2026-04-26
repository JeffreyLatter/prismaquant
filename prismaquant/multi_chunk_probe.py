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

Usage:
    python -m prismaquant.multi_chunk_probe \\
        --chunks-dir /path/with/chunk_*.jsonl \\
        --model <hf_model_path> \\
        --output /path/to/merged_probe.pkl \\
        --activation-cache-dir /path/to/act \\
        --work-dir /path/to/work_root \\
        --h-detail-dir /path/to/h_detail \\
        --layers-per-shard 8 --prefetch-lookahead 4 \\
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


def _make_chunk_argv(base_argv: list[str], chunk_jsonl: Path,
                     chunk_work_dir: Path, chunk_output: Path,
                     drop_keys: tuple[str, ...]) -> list[str]:
    """Rebuild sys.argv for one chunk by overriding --dataset, --work-dir,
    --output and stripping the multi-chunk-only keys."""
    out: list[str] = [base_argv[0]]
    i = 1
    skip_next = False
    for tok in base_argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in drop_keys:
            skip_next = True
            continue
        if any(tok.startswith(k + "=") for k in drop_keys):
            continue
        out.append(tok)
    # Append chunk-specific overrides (these win over any earlier values).
    out += [
        "--dataset", str(chunk_jsonl),
        "--work-dir", str(chunk_work_dir),
        "--output", str(chunk_output),
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks-dir", required=True,
                    help="Directory containing chunk_NN.jsonl files")
    ap.add_argument("--output", required=True,
                    help="Final merged probe.pkl path")
    ap.add_argument("--work-dir", required=True,
                    help="Work root; per-chunk subdirs will be created here")
    args, passthrough = ap.parse_known_args()

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

    chunk_outputs: list[Path] = []
    base_argv = ["prismaquant.incremental_probe"] + passthrough
    drop_keys = ("--chunks-dir", "--dataset", "--output", "--work-dir")

    for i, chunk_jsonl in enumerate(chunk_jsonls):
        chunk_work_dir = work_root / f"chunk_{i:02d}"
        chunk_output = chunk_work_dir / "probe.pkl"
        if chunk_output.exists():
            print(f"[multi-chunk] chunk {i}: resume — {chunk_output} "
                  f"already exists, skipping", flush=True)
            chunk_outputs.append(chunk_output)
            continue
        chunk_work_dir.mkdir(parents=True, exist_ok=True)
        # Per-chunk streaming-offload subdir: keep ctx warm across chunks
        # by sharing the offload at the work_root level instead of the
        # per-chunk subdir. The probe will use this when ctx is built
        # (chunk 0). Chunks 1..N reuse the cached ctx and never touch
        # offload setup again.
        # NOTE: we leave the ctx cache to handle reuse; per-chunk work_dir
        # only carries shards/, work/precompute, logs/.
        chunk_argv = _make_chunk_argv(
            base_argv, chunk_jsonl, chunk_work_dir, chunk_output, drop_keys)
        print(f"\n[multi-chunk] === chunk {i+1}/{len(chunk_jsonls)} ===",
              flush=True)
        print(f"[multi-chunk] dataset={chunk_jsonl.name} "
              f"work_dir={chunk_work_dir.name}", flush=True)
        t0 = time.time()
        # Save & restore sys.argv so each main() invocation parses fresh.
        saved_argv = sys.argv
        sys.argv = chunk_argv
        try:
            ip.main()
        finally:
            sys.argv = saved_argv
        elapsed = time.time() - t0
        print(f"[multi-chunk] chunk {i+1} done in {elapsed:.0f}s",
              flush=True)
        chunk_outputs.append(chunk_output)

    print(f"\n[multi-chunk] merging {len(chunk_outputs)} per-chunk "
          f"probe.pkls -> {out_path}", flush=True)
    ip.merge_probe_pickles(chunk_outputs, out_path)
    # Annotate the final pickle with multi-chunk metadata so downstream
    # consumers can see it was produced by N chunks.
    with out_path.open("rb") as f:
        merged = pickle.load(f)
    meta = dict(merged.get("meta", {}))
    meta["multi_chunk_count"] = len(chunk_outputs)
    meta["multi_chunk_total_nsamples"] = sum(
        pickle.loads(p.read_bytes()).get("meta", {}).get("nsamples", 0)
        for p in chunk_outputs
    )
    merged["meta"] = meta
    with out_path.open("wb") as f:
        pickle.dump(merged, f)
    print(f"[multi-chunk] DONE — merged probe at {out_path} "
          f"(total_nsamples={meta['multi_chunk_total_nsamples']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
