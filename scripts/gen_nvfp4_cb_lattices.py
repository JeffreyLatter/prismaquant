"""Generate the committed NVFP4-CB universal lattice cache.

Builds the grid-snapped Lloyd lattices used as the fixed (no-sidecar)
codebooks: fp4 full-mode d=8 tables (k=12..14), fp4 product-mode d=4
sub-tables (half-bits 6..12), fp8 product-mode d=2 sub-tables (quarter-bits
9..12), and fp4 POSITIVE d=8 magnitude tables for the signed-mode S13..S16
rungs (m = k-8 = 5..8).

Merge semantics: existing tables in data/nvfp4_cb_lattices.pt are PRESERVED
and only missing keys are built — Lloyd runs on CUDA when available, whose
float atomics can flip grid-snap ties, so rebuilding an existing table could
silently invalidate published measurements against it (exp-1). Delete the .pt
first to force a full rebuild.

    PYTHONPATH=. python scripts/gen_nvfp4_cb_lattices.py
"""
from __future__ import annotations

import torch

from prismaquant import nvfp4_cb_formats as cb


def main() -> None:
    out: dict[str, torch.Tensor] = {}
    if cb._DATA.exists():
        out.update(torch.load(cb._DATA, map_location="cpu",
                              weights_only=True))
    wanted: list[tuple[int, str, int, bool]] = []
    for k in (12, 13, 14):
        wanted.append((k, "fp4", cb.VEC_DIM, False))
    for k_half in range(6, 13):
        wanted.append((k_half, "fp4", cb.VEC_DIM // 2, False))
    for k_quarter in range(9, 13):
        wanted.append((k_quarter, "fp8", 2, False))
    for m in (5, 6, 7, 8):
        wanted.append((m, "fp4", cb.VEC_DIM, True))

    built = 0
    for k, grid, d, positive in wanted:
        key = cb._lattice_key(k, grid, d, positive)
        if key in out:
            print(f"kept  {key}: {tuple(out[key].shape)}")
            continue
        out[key] = cb._build_lattice(k, grid, d, positive=positive)
        built += 1
        print(f"built {key}: {tuple(out[key].shape)}")
    cb._DATA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cb._DATA)
    print(f"wrote {cb._DATA} ({len(out)} tables, {built} new)")


if __name__ == "__main__":
    main()
