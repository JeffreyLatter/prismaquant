"""GPU residency guards for production hot paths."""
from __future__ import annotations

import torch


def require_cuda_hot_path(component: str, device: str | torch.device = "cuda") -> torch.device:
    """Return a CUDA device or raise before production work can run on CPU."""
    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"{component} requires CUDA. PrismaQuant production hot paths are "
            "GPU-or-bust; refusing to run on CPU."
        )
    return requested
