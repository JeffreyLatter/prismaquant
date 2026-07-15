"""Generate the committed NVFP4-CB universal lattice cache.

Builds the grid-snapped Lloyd lattices used as the fixed (no-sidecar) codebooks:
fp4 full-mode d=8 tables for k in {12,13,14}, fp4 product-mode d=4 sub-tables
for the k=12..24 ladder (half-bits 6..12), and fp8 product-mode d=2 sub-tables
for the FP8_CB k=36..48 ladder (quarter-bits 9..12). Deterministic; the runtime
regenerates identical tables on a cache miss, so this only pre-warms.

    PYTHONPATH=. python scripts/gen_nvfp4_cb_lattices.py
"""
from __future__ import annotations

import torch

from prismaquant import nvfp4_cb_formats as cb


def main() -> None:
    out: dict[str, torch.Tensor] = {}
    grid = "fp4"
    for k in (12, 13, 14):
        key = cb._lattice_key(k, grid, cb.VEC_DIM)
        out[key] = cb._build_lattice(k, grid, cb.VEC_DIM)
        print(f"built {key}: {tuple(out[key].shape)}")
    for k_half in range(6, 13):
        key = cb._lattice_key(k_half, grid, cb.VEC_DIM // 2)
        out[key] = cb._build_lattice(k_half, grid, cb.VEC_DIM // 2)
        print(f"built {key}: {tuple(out[key].shape)}")
    for k_quarter in range(9, 13):
        key = cb._lattice_key(k_quarter, "fp8", 2)
        out[key] = cb._build_lattice(k_quarter, "fp8", 2)
        print(f"built {key}: {tuple(out[key].shape)}")
    cb._DATA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cb._DATA)
    print(f"wrote {cb._DATA} ({len(out)} tables)")


if __name__ == "__main__":
    main()
