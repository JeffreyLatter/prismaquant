"""Router diagnostics for Entmoot MoE expert merging.

The merge planner uses these helpers to compare the ideal clustered router
target against stock K-row router approximations.  The shipped artifact must
use a normal K-row router; the old-router hard redirect is only a diagnostic
target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


EPS = 1e-12


@dataclass(frozen=True)
class RouterStrategyMetrics:
    """Held-out agreement between clustered old routing and a stock router."""

    strategy: str
    kl_mean: float
    top1_agreement: float
    topk_set_agreement: float
    num_tokens: int

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "kl_mean": self.kl_mean,
            "top1_agreement": self.top1_agreement,
            "topk_set_agreement": self.topk_set_agreement,
            "num_tokens": self.num_tokens,
        }


@dataclass(frozen=True)
class RouterStrategyChoice:
    """Selected stock-router strategy for one Entmoot layer plan."""

    selected_strategy: str
    metrics: tuple[RouterStrategyMetrics, ...]
    kl_cap: float
    top1_floor: float
    topk_floor: float

    def to_dict(self) -> dict:
        return {
            "selected_strategy": self.selected_strategy,
            "metrics": [m.to_dict() for m in self.metrics],
            "kl_cap": self.kl_cap,
            "top1_floor": self.top1_floor,
            "topk_floor": self.topk_floor,
        }


def topk_router_distribution(
    logits: torch.Tensor,
    *,
    top_k: int,
    softmax_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a sparse top-k router distribution with the same last dim.

    Qwen/Mixtral-style routers normalize softmax over selected top-k logits,
    not over all experts.  This helper mirrors that behavior and scatters the
    selected probabilities back into an `[tokens, experts]` tensor.
    """

    flat = _as_2d_logits(logits)
    k = min(int(top_k), int(flat.shape[-1]))
    if k <= 0:
        raise ValueError("top_k must be positive")
    top_v, top_i = flat.to(softmax_dtype).topk(k, dim=-1)
    probs = F.softmax(top_v, dim=-1).to(torch.float64)
    out = torch.zeros(flat.shape, dtype=torch.float64, device=flat.device)
    out.scatter_(1, top_i, probs)
    return out


def clustered_router_target(
    old_logits: torch.Tensor,
    orig_to_new_eid: Mapping[int | str, int],
    *,
    num_new_experts: int,
    top_k: int,
    softmax_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Aggregate an old E-way top-k router distribution into K clusters."""

    old_probs = topk_router_distribution(
        old_logits, top_k=top_k, softmax_dtype=softmax_dtype,
    )
    mapping = _mapping_vector(orig_to_new_eid, int(old_probs.shape[-1]))
    if int(mapping.max()) >= int(num_new_experts):
        raise ValueError("orig_to_new_eid maps outside num_new_experts")
    target = torch.zeros(
        (old_probs.shape[0], int(num_new_experts)),
        dtype=torch.float64,
        device=old_probs.device,
    )
    target.index_add_(1, mapping.to(old_probs.device), old_probs)
    return target


def stock_router_distribution(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
    router_bias: torch.Tensor | None = None,
    softmax_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Evaluate a stock linear router and return its top-k distribution."""

    hidden = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
    weight = router_weight.detach().to(device=hidden.device)
    logits = hidden.to(weight.dtype) @ weight.T
    if router_bias is not None:
        logits = logits + router_bias.detach().to(device=hidden.device, dtype=logits.dtype)
    return topk_router_distribution(
        logits, top_k=min(top_k, int(weight.shape[0])), softmax_dtype=softmax_dtype,
    )


def router_weight_for_strategy(
    old_router_weight: torch.Tensor,
    entry: Mapping,
    strategy: str,
) -> torch.Tensor:
    """Construct the K-row stock router weight for an Entmoot strategy."""

    old = old_router_weight.detach()
    if old.ndim != 2:
        raise ValueError("old_router_weight must be 2D")
    kept = [int(e) for e in entry["kept_expert_ids"]]
    if strategy in ("anchor", "anchor_rows"):
        idx = torch.as_tensor(kept, dtype=torch.long, device=old.device)
        return old.index_select(0, idx).contiguous()
    if strategy == "weighted_average":
        rows = []
        for new_eid in range(int(entry["num_experts_kept"])):
            weights = cluster_weights_by_orig(entry, new_eid, prefer_router=True)
            acc = None
            for orig_eid, coeff in sorted(weights.items()):
                part = old[int(orig_eid)].float() * float(coeff)
                acc = part if acc is None else acc + part
            rows.append(acc.to(dtype=old.dtype))
        return torch.stack(rows, dim=0).contiguous()
    raise ValueError(f"unknown router strategy {strategy!r}")


def router_kl(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Per-token KL(target || pred), both `[tokens, experts]` distributions."""

    if target.shape != pred.shape:
        raise ValueError("target and pred must have the same shape")
    t = target.to(torch.float64).clamp_min(EPS)
    p = pred.to(torch.float64).clamp_min(EPS)
    return (t * (t.log() - p.log())).sum(dim=-1)


def top1_agreement(target: torch.Tensor, pred: torch.Tensor) -> float:
    if target.shape != pred.shape:
        raise ValueError("target and pred must have the same shape")
    if target.numel() == 0:
        return 1.0
    return float((target.argmax(dim=-1) == pred.argmax(dim=-1)).to(torch.float64).mean())


def topk_set_agreement(target: torch.Tensor, pred: torch.Tensor, *, k: int) -> float:
    """Mean exact set agreement for top-k cluster ids."""

    if target.shape != pred.shape:
        raise ValueError("target and pred must have the same shape")
    kk = min(int(k), int(target.shape[-1]))
    if kk <= 0 or target.shape[0] == 0:
        return 1.0
    target_top = target.topk(kk, dim=-1).indices.sort(dim=-1).values
    pred_top = pred.topk(kk, dim=-1).indices.sort(dim=-1).values
    return float((target_top == pred_top).all(dim=-1).to(torch.float64).mean())


def evaluate_router_strategy(
    hidden_states: torch.Tensor,
    old_router_weight: torch.Tensor,
    entry: Mapping,
    *,
    strategy: str,
    top_k: int,
    old_router_bias: torch.Tensor | None = None,
    topk_agreement_k: int = 2,
    softmax_dtype: torch.dtype = torch.float32,
) -> RouterStrategyMetrics:
    """Measure one stock-router strategy against clustered old routing."""

    hidden = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
    old_logits = hidden.to(old_router_weight.dtype) @ old_router_weight.detach().T
    if old_router_bias is not None:
        old_logits = old_logits + old_router_bias.detach().to(
            device=hidden.device, dtype=old_logits.dtype,
        )
    target = clustered_router_target(
        old_logits,
        entry["orig_to_new_eid"],
        num_new_experts=int(entry["num_experts_kept"]),
        top_k=top_k,
        softmax_dtype=softmax_dtype,
    )
    stock_weight = router_weight_for_strategy(old_router_weight, entry, strategy)
    pred = stock_router_distribution(
        hidden,
        stock_weight,
        top_k=min(top_k, int(entry["num_experts_kept"])),
        softmax_dtype=softmax_dtype,
    )
    kl = router_kl(target, pred)
    return RouterStrategyMetrics(
        strategy=strategy,
        kl_mean=float(kl.mean()) if kl.numel() else 0.0,
        top1_agreement=top1_agreement(target, pred),
        topk_set_agreement=topk_set_agreement(target, pred, k=topk_agreement_k),
        num_tokens=int(hidden.shape[0]),
    )


def choose_router_strategy(
    hidden_states: torch.Tensor,
    old_router_weight: torch.Tensor,
    entry: Mapping,
    *,
    top_k: int,
    strategies: Sequence[str] = ("anchor", "weighted_average"),
    old_router_bias: torch.Tensor | None = None,
    kl_cap: float = 0.05,
    top1_floor: float = 0.95,
    topk_floor: float = 0.90,
    topk_agreement_k: int = 2,
    softmax_dtype: torch.dtype = torch.float32,
) -> RouterStrategyChoice:
    """Evaluate strategies and select the lowest-KL strategy passing gates."""

    metrics = tuple(
        evaluate_router_strategy(
            hidden_states,
            old_router_weight,
            entry,
            strategy=s,
            top_k=top_k,
            old_router_bias=old_router_bias,
            topk_agreement_k=topk_agreement_k,
            softmax_dtype=softmax_dtype,
        )
        for s in strategies
    )
    passing = [
        m for m in metrics
        if (
            m.kl_mean <= float(kl_cap)
            and m.top1_agreement >= float(top1_floor)
            and m.topk_set_agreement >= float(topk_floor)
        )
    ]
    pool = passing or list(metrics)
    selected = min(pool, key=lambda m: (m.kl_mean, 0 if m.strategy == "anchor" else 1))
    return RouterStrategyChoice(
        selected_strategy=selected.strategy,
        metrics=metrics,
        kl_cap=float(kl_cap),
        top1_floor=float(top1_floor),
        topk_floor=float(topk_floor),
    )


def cluster_weights_by_orig(
    entry: Mapping,
    new_eid: int,
    *,
    prefer_router: bool = False,
) -> dict[int, float]:
    """Return expert weights for a cluster from a manifest entry."""

    for cluster in entry.get("clusters", []):
        if int(cluster.get("new_expert_id", -1)) != int(new_eid):
            continue
        weights = None
        if prefer_router:
            weights = cluster.get("router_weights")
        if weights is None:
            weights = cluster.get("weights")
        if weights:
            return {int(k): float(v) for k, v in weights.items()}
        orig = [int(e) for e in cluster["original_expert_ids"]]
        return {e: 1.0 / max(len(orig), 1) for e in orig}
    raise KeyError(f"merge manifest missing cluster new_expert_id={new_eid}")


def _as_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim < 2:
        raise ValueError("logits must be a tensor with at least 2 dimensions")
    return logits.detach().reshape(-1, logits.shape[-1])


def _mapping_vector(mapping: Mapping[int | str, int], num_orig: int) -> torch.Tensor:
    out = []
    for eid in range(int(num_orig)):
        if eid in mapping:
            out.append(int(mapping[eid]))
        elif str(eid) in mapping:
            out.append(int(mapping[str(eid)]))
        else:
            raise ValueError(f"orig_to_new_eid missing expert {eid}")
    return torch.tensor(out, dtype=torch.long)

