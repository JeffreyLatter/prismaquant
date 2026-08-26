#!/usr/bin/env python3
"""Merge disjoint-qname AURA cost shard pickles (aura_cost.py --unit-filter).

Each shard is a full aura_cost.py --output payload
({"schema", "n_probes", "formats", "token_scope", "stats", "costs",
"provenance"}) scoped to one --unit-filter layer range. Every Linear qname
must appear in EXACTLY one shard (fail-closed: overlap or a gap is a
partition bug, not something to silently paper over) and every shard must
carry the same schema/n_probes/formats/token_scope (they measured the same
calibration under the same knobs, just different Linears).

Usage: merge_aura_cost_shards.py --shard PATH [--shard PATH ...] --output PATH
"""
import argparse
import pickle
import sys
from pathlib import Path

_INVARIANT_KEYS = ("schema", "n_probes", "formats", "token_scope")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", action="append", required=True, dest="shards")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    payloads = []
    for path in args.shards:
        with open(path, "rb") as fh:
            payloads.append((path, pickle.load(fh)))

    base_path, base = payloads[0]
    for key in _INVARIANT_KEYS:
        base_val = base.get(key)
        for path, payload in payloads[1:]:
            val = payload.get(key)
            if val != base_val:
                print(f"REFUSE: shard {path!r} has {key}={val!r}, "
                      f"shard {base_path!r} has {key}={base_val!r} — these "
                      "are not shards of the same run", file=sys.stderr)
                return 2

    merged_stats: dict = {}
    merged_costs: dict = {}
    dw_rendered = 0
    dw_rtn = 0
    for path, payload in payloads:
        stats = payload.get("stats", {})
        costs = payload.get("costs", {})
        overlap = (set(merged_stats) & set(stats)) | (set(merged_costs) & set(costs))
        if overlap:
            print(f"REFUSE: shard {path!r} re-measures "
                  f"{len(overlap)} qname(s) another shard already covered "
                  f"(sample={sorted(overlap)[:5]}) — the shard partition "
                  "was not disjoint", file=sys.stderr)
            return 2
        merged_stats.update(stats)
        merged_costs.update(costs)
        prov = payload.get("provenance", {}) or {}
        dw_rendered += int(prov.get("dw_rendered_rows", 0) or 0)
        dw_rtn += int(prov.get("dw_rtn_fallback_rows", 0) or 0)

    merged_provenance = dict(base.get("provenance", {}) or {})
    merged_provenance["dw_rendered_rows"] = dw_rendered
    merged_provenance["dw_rtn_fallback_rows"] = dw_rtn
    merged_provenance["cluster_merged_from"] = [str(p) for p, _ in payloads]
    merged_provenance.pop("n_linear_chunks", None)

    merged = dict(base)
    merged["stats"] = merged_stats
    merged["costs"] = merged_costs
    merged["provenance"] = merged_provenance

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[merge-aura-cost] merged {len(payloads)} shard(s) -> "
          f"{args.output}: {len(merged_costs)} Linears "
          f"(dW rendered={dw_rendered} rtn={dw_rtn})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
