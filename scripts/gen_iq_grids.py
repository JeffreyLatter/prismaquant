"""Generate the GGUF IQ codebook/grid data file consumed by
``prismaquant.gguf_formats``.

The IQ2_*/IQ3_* grids (magnitude codebooks of 8- or 4-element groups), the
7-bit sign codebook (``ksigns``), and the IQ4 non-linear value table are
lifted verbatim from gguf-py's *decoded* tables, which are the exact inverse
of its ``dequantize`` path. Extracting them here (rather than re-decoding the
packed ``grid_hex`` in the hot path) keeps the runtime torch-only and pins the
encoder's reconstruction to the same numbers the reader decodes.

Run: PYTHONPATH=. python scripts/gen_iq_grids.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gguf.quants import IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL

OUT = Path(__file__).resolve().parent.parent / "prismaquant" / "data" / "iq_grids.pt"


def main() -> None:
    data: dict[str, torch.Tensor] = {}
    for cls in (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S):
        cls.init_grid()
        grid = np.asarray(cls.grid, dtype=np.float32).reshape(cls.grid_shape)
        data[f"grid_{cls.qtype.name.lower()}"] = torch.from_numpy(grid.copy())
    data["ksigns"] = torch.from_numpy(
        np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy()
    )
    data["kvalues_iq4nl"] = torch.tensor(IQ4_NL.kvalues, dtype=torch.int8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT)
    print(f"wrote {OUT}")
    for k, v in data.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
