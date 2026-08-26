#!/usr/bin/env python3
"""Print N layer-contiguous --unit-filter regexes for clustering aura_cost.

Usage: aura_cost_shard_ranges.py MODEL_PATH N

Prints N lines, one --unit-filter regex per shard index (0..N-1), each
matching that shard's contiguous decoder-layer range under the model's own
profile-detected body-layer prefix. aura_cost.py's chunk count auto-sizes
from the scoped weight footprint (aura_cost._auto_n_chunks), so scoping a
box to a layer range genuinely shrinks its compute, not just its
bookkeeping. Layer-contiguous (not interleaved) so each shard's regex stays
a simple bounded alternation instead of one term per Linear.
"""
import re
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    model_path, n_str = sys.argv[1], sys.argv[2]
    n = int(n_str)
    if n < 1:
        print(f"N must be >= 1, got {n}", file=sys.stderr)
        return 2

    from prismaquant.incremental_probe import load_num_hidden_layers
    from prismaquant.model_profiles.registry import detect_profile

    num_layers = load_num_hidden_layers(model_path)
    profile = detect_profile(model_path)
    prefix = profile.body_layer_prefix()
    esc_prefix = re.escape(prefix)

    for i in range(n):
        lo = i * num_layers // n
        hi = (i + 1) * num_layers // n
        if lo >= hi:
            raise SystemExit(
                f"shard {i}/{n} got an empty layer range on a "
                f"{num_layers}-layer model; use fewer shards"
            )
        idxs = "|".join(str(j) for j in range(lo, hi))
        print(rf"{esc_prefix}\.(?:{idxs})\.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
