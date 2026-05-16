"""Activation row sampling helpers.

Calibration captures should not keep the first rows seen from a forward
stream. That biases every downstream solver toward early tokens. The helper
below assigns each observed row an independent random priority and keeps the
top-k priorities, which is equivalent to uniform sampling without replacement
over all rows seen so far while staying bounded to ``max_rows`` storage.
"""
from __future__ import annotations

import torch


def update_priority_reservoir(
    current_rows: torch.Tensor | None,
    current_priorities: torch.Tensor | None,
    new_rows: torch.Tensor,
    *,
    max_rows: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return an updated uniform row sample.

    ``current_priorities`` must contain the random priorities for
    ``current_rows`` and is kept on CPU. ``new_rows`` may live on CPU or CUDA;
    returned rows stay on the same device/dtype as the candidate row tensor.
    """
    limit = int(max_rows)
    if limit <= 0 or new_rows.numel() == 0:
        return current_rows, current_priorities
    if new_rows.dim() != 2:
        raise ValueError("priority reservoir expects 2D row tensors")

    incoming = new_rows.detach()
    new_priorities = torch.rand(
        int(incoming.shape[0]),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    if current_rows is None:
        candidates = incoming
        priorities = new_priorities
    else:
        if current_priorities is None:
            raise ValueError("current_priorities required with current_rows")
        if current_rows.dim() != 2:
            raise ValueError("current_rows must be 2D")
        if int(current_rows.shape[1]) != int(incoming.shape[1]):
            raise ValueError(
                "current_rows and new_rows must have the same feature width"
            )
        candidates = torch.cat([current_rows, incoming], dim=0)
        priorities = torch.cat([current_priorities.cpu(), new_priorities], dim=0)

    if int(candidates.shape[0]) <= limit:
        return candidates.clone(), priorities.clone()

    _, keep_cpu = torch.topk(priorities, k=limit, largest=True, sorted=False)
    keep_device = keep_cpu.to(device=candidates.device)
    return (
        candidates.index_select(0, keep_device).clone(),
        priorities.index_select(0, keep_cpu).clone(),
    )
