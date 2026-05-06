"""Single-point KL/Fisher utilities for LLM quantization surrogates.

For teacher logits ``z0`` and student logit perturbation ``dz``, forward KL
from the teacher distribution to the student distribution is locally

    0.5 * dz^T F dz

where ``F = diag(p) - p p^T`` is the categorical Fisher matrix at
``p = softmax(z0 / temperature)``, with an additional ``1 / temperature^2``
factor because the softmax is applied to scaled logits.  This module exposes
both an exact quadratic form for tests/diagnostics and a stochastic scalar
probe for adjoint sketches.
"""
from __future__ import annotations

import math
from typing import Literal

import torch


TokenScope = Literal["last", "all", "causal"]
ProbeDistribution = Literal["gaussian", "rademacher"]


def select_token_scope(logits: torch.Tensor, token_scope: str) -> torch.Tensor:
    """Select the token positions used by a KL/Fisher metric."""
    if logits.dim() < 2:
        raise ValueError("logits must have at least batch and vocab dimensions")
    scope = str(token_scope)
    if scope == "last":
        if logits.size(-2) < 1:
            raise ValueError("last-token KL/Fisher requires at least one token")
        return logits[..., -1:, :]
    if scope == "all":
        return logits
    if scope == "causal":
        if logits.size(-2) < 2:
            raise ValueError("causal KL/Fisher requires sequence length >= 2")
        return logits[..., :-1, :]
    raise ValueError(f"unknown KL/Fisher token scope: {token_scope!r}")


def token_count_for_logits(logits: torch.Tensor) -> int:
    """Return the number of categorical distributions in a logits tensor."""
    if logits.dim() < 2:
        raise ValueError("logits must have at least batch and vocab dimensions")
    return int(math.prod(int(v) for v in logits.shape[:-1]))


def fisher_quadratic_form(
    teacher_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    *,
    token_scope: str = "last",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return the local second-order forward-KL approximation.

    The returned scalar is normalized like ``kl_divergence`` in
    ``build_rtn_cache.py``: mean over selected token positions, sum over vocab.
    """
    if teacher_logits.shape != delta_logits.shape:
        raise ValueError("teacher_logits and delta_logits must have the same shape")
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be positive")

    z = select_token_scope(teacher_logits, token_scope).float() / temp
    dz = select_token_scope(delta_logits, token_scope).float() / temp
    probs = torch.softmax(z, dim=-1)
    mean = torch.sum(probs * dz, dim=-1, keepdim=True)
    variance = torch.sum(probs * (dz - mean) * (dz - mean), dim=-1)
    return 0.5 * variance.mean()


def fisher_probe_scalar(
    logits: torch.Tensor,
    *,
    seed: int,
    token_scope: str = "last",
    temperature: float = 1.0,
    distribution: str = "gaussian",
) -> torch.Tensor:
    """Return a scalar whose logit gradient is one KL/Fisher probe.

    If ``r`` is the gradient of this scalar with respect to logits, then
    ``E[r r^T]`` is the Fisher matrix of the selected teacher distributions,
    normalized by the number of selected token positions.  Backpropagating this
    scalar through a clean model therefore gives an unbiased low-rank probe of
    the single-point forward-KL quadratic surrogate.
    """
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be positive")

    selected = select_token_scope(logits, token_scope).float()
    scaled = selected / temp
    token_count = max(token_count_for_logits(selected), 1)
    with torch.no_grad():
        probs = torch.softmax(scaled.detach(), dim=-1)
        root = torch.sqrt(torch.clamp(probs, min=0.0))
        generator = torch.Generator(device=scaled.device)
        generator.manual_seed(int(seed))
        if distribution == "gaussian":
            noise = torch.randn(
                probs.shape,
                generator=generator,
                device=probs.device,
                dtype=torch.float32,
            )
        elif distribution == "rademacher":
            noise = torch.empty(
                probs.shape,
                device=probs.device,
                dtype=torch.float32,
            )
            noise.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        else:
            raise ValueError(f"unknown Fisher probe distribution: {distribution!r}")
        root_noise = root * noise
        probe = root_noise - probs * root_noise.sum(dim=-1, keepdim=True)
        probe = probe / math.sqrt(float(token_count))
    return torch.sum(scaled * probe)
