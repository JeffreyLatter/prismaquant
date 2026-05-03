"""Entmoot: closed-form expert merge planning for MoE compression.

This module owns the sample-light part of MoE merging: given an expert
similarity Gram matrix and a marginal-value signal, estimate which experts
are redundant, rank them by expected merge loss, and build deterministic
clusters for reducing E experts to K experts.

The runtime/export path is intentionally not changed here.  The output is a
merge manifest that can later drive dense expert synthesis in the exporter
without requiring custom kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import math

import torch


EPS = 1e-12


@dataclass(frozen=True)
class ExpertSubsumability:
    """How much one expert is explained by a small basis of other experts."""

    expert_id: int
    marginal_value: float
    norm_sq: float
    neighbor_expert_ids: tuple[int, ...]
    coefficients: tuple[float, ...]
    residual_energy: float
    subsumed_fraction: float
    unique_fraction: float
    predicted_merge_loss: float
    memory_bytes: int | None = None
    savings_per_loss: float | None = None

    def to_dict(self) -> dict:
        out = {
            "expert_id": self.expert_id,
            "marginal_value": self.marginal_value,
            "norm_sq": self.norm_sq,
            "neighbor_expert_ids": list(self.neighbor_expert_ids),
            "coefficients": list(self.coefficients),
            "residual_energy": self.residual_energy,
            "subsumed_fraction": self.subsumed_fraction,
            "unique_fraction": self.unique_fraction,
            "predicted_merge_loss": self.predicted_merge_loss,
        }
        if self.memory_bytes is not None:
            out["memory_bytes"] = self.memory_bytes
        if self.savings_per_loss is not None:
            out["savings_per_loss"] = self.savings_per_loss
        return out


@dataclass(frozen=True)
class ExpertCluster:
    """A group of original experts represented by one new expert."""

    new_expert_id: int
    anchor_expert_id: int
    original_expert_ids: tuple[int, ...]
    weights: tuple[tuple[int, float], ...]
    centroid_residual_energy: float
    predicted_merge_loss: float
    marginal_value_sum: float
    router_weights: tuple[tuple[int, float], ...] | None = None
    tensor_weights: Mapping[str, tuple[tuple[int, float], ...]] | None = None
    merge_action: str | None = None
    decision: str | None = None
    residual_metrics: Mapping[str, float] | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        out = {
            "new_expert_id": self.new_expert_id,
            "anchor_expert_id": self.anchor_expert_id,
            "original_expert_ids": list(self.original_expert_ids),
            "weights": {str(k): v for k, v in self.weights},
            "centroid_residual_energy": self.centroid_residual_energy,
            "predicted_merge_loss": self.predicted_merge_loss,
            "marginal_value_sum": self.marginal_value_sum,
        }
        if self.router_weights is not None:
            out["router_weights"] = {str(k): v for k, v in self.router_weights}
        if self.tensor_weights is not None:
            out["tensor_weights"] = {
                str(name): {str(k): v for k, v in weights}
                for name, weights in sorted(self.tensor_weights.items())
            }
        if self.merge_action is not None:
            out["merge_action"] = self.merge_action
        if self.decision is not None:
            out["decision"] = self.decision
        if self.residual_metrics is not None:
            out["residual_metrics"] = dict(self.residual_metrics)
        if self.rejection_reason is not None:
            out["rejection_reason"] = self.rejection_reason
        return out


@dataclass(frozen=True)
class ExpertMergePlan:
    """Merge plan for one MoE router/layer."""

    router_qname: str | None
    num_experts_orig: int
    num_experts_kept: int
    clusters: tuple[ExpertCluster, ...]
    ranking: tuple[ExpertSubsumability, ...]
    total_predicted_merge_loss: float
    method: str = "entmoot_svd_gram_v1"
    router_strategy: str | None = None
    expert_decisions: tuple[Mapping[str, Any], ...] = ()
    diagnostics: Mapping[str, Any] | None = None

    def to_manifest_entry(self) -> dict:
        orig_to_new = {}
        for cluster in self.clusters:
            for eid in cluster.original_expert_ids:
                orig_to_new[str(eid)] = cluster.new_expert_id
        out = {
            "method": self.method,
            "num_experts_orig": self.num_experts_orig,
            "num_experts_kept": self.num_experts_kept,
            "kept_expert_ids": [c.anchor_expert_id for c in self.clusters],
            "orig_to_new_eid": orig_to_new,
            "clusters": [c.to_dict() for c in self.clusters],
            "expert_ranking": [r.to_dict() for r in self.ranking],
            "total_predicted_merge_loss": self.total_predicted_merge_loss,
        }
        if self.router_strategy is not None:
            out["router_strategy"] = self.router_strategy
        if self.expert_decisions:
            out["expert_decisions"] = [dict(d) for d in self.expert_decisions]
        if self.diagnostics is not None:
            out["diagnostics"] = dict(self.diagnostics)
        return out

    def to_manifest(self) -> dict:
        key = self.router_qname or "__unknown_router__"
        return {key: self.to_manifest_entry()}


def expert_weight_feature(
    projections: Mapping[str, torch.Tensor],
    *,
    projection_weights: Mapping[str, float] | None = None,
    normalize_projection: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Flatten one expert's projection tensors into a deterministic feature.

    ``normalize_projection=True`` makes gate/up/down contribute comparable
    directional information instead of letting the largest matrix dominate
    the similarity metric.
    """

    if not projections:
        raise ValueError("expert_weight_feature requires at least one projection")

    chunks: list[torch.Tensor] = []
    weights = projection_weights or {}
    for name in sorted(projections):
        weight = projections[name].detach().to(dtype=torch.float64).reshape(-1)
        if normalize_projection:
            weight = weight / weight.norm().clamp_min(eps)
        scale = float(weights.get(name, 1.0))
        chunks.append(weight * scale)
    return torch.cat(chunks, dim=0)


def gram_from_features(
    features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
    *,
    expert_ids: Sequence[int] | None = None,
    normalize: bool = False,
) -> tuple[list[int], torch.Tensor]:
    """Build an expert Gram matrix from dense/sketched feature vectors."""

    if isinstance(features, Mapping):
        ids = sorted(int(e) for e in features)
        rows = [features[e].detach().to(dtype=torch.float64).reshape(-1) for e in ids]
    else:
        if expert_ids is None:
            ids = list(range(len(features)))
        else:
            ids = [int(e) for e in expert_ids]
            if len(ids) != len(features):
                raise ValueError("expert_ids length must match feature count")
        rows = [f.detach().to(dtype=torch.float64).reshape(-1) for f in features]

    if not rows:
        raise ValueError("gram_from_features requires at least one feature")

    n = rows[0].numel()
    if any(r.numel() != n for r in rows):
        raise ValueError("all expert features must have the same length")

    X = torch.stack(rows, dim=0)
    if normalize:
        X = X / X.norm(dim=1, keepdim=True).clamp_min(EPS)
    return ids, X @ X.T


def cosine_from_gram(gram: torch.Tensor) -> torch.Tensor:
    """Convert a Gram matrix into cosine similarities."""

    G = _as_square_gram(gram)
    norms = G.diag().clamp_min(EPS).sqrt()
    return G / (norms[:, None] * norms[None, :]).clamp_min(EPS)


def rank_expert_subsumability(
    gram: torch.Tensor,
    expert_ids: Sequence[int],
    marginal_values: Mapping[int, float] | Sequence[float],
    *,
    basis_size: int = 8,
    ridge: float = 1e-6,
    memory_bytes: Mapping[int, int] | Sequence[int] | None = None,
) -> list[ExpertSubsumability]:
    """Rank experts by how cheaply they can be merged into nearby experts.

    For target expert j and basis S, solve a closed-form ridge projection

        min_c ||x_j - sum_i c_i x_i||^2 + λ||c||^2

    using only Gram entries.  The residual fraction is the non-subsumed part
    of expert j; multiplying by marginal value yields an expected merge loss.
    """

    ids = [int(e) for e in expert_ids]
    G = _validate_inputs(gram, ids)
    E = len(ids)
    if basis_size < 0:
        raise ValueError("basis_size must be non-negative")
    k = min(int(basis_size), max(E - 1, 0))
    marg = _value_vector(marginal_values, ids, "marginal_values")
    mem = _optional_int_vector(memory_bytes, ids, "memory_bytes")
    cos = cosine_from_gram(G)

    out: list[ExpertSubsumability] = []
    for j, eid in enumerate(ids):
        norm_j = float(G[j, j].clamp_min(0.0))
        if k == 0 or norm_j <= EPS:
            neighbors: list[int] = []
            coeff = torch.empty(0, dtype=torch.float64)
            residual = norm_j
        else:
            order = [
                i for i in sorted(
                    range(E), key=lambda i_: (-float(cos[j, i_]), ids[i_])
                )
                if i != j
            ][:k]
            A = G[order][:, order]
            b = G[order, j]
            diag_scale = float(A.diag().mean().clamp_min(EPS))
            A_reg = A + torch.eye(len(order), dtype=torch.float64) * (
                float(ridge) * diag_scale
            )
            try:
                coeff = torch.linalg.solve(A_reg, b)
            except RuntimeError:
                coeff = torch.linalg.lstsq(A_reg, b.unsqueeze(1)).solution.squeeze(1)
            residual_t = G[j, j] - 2.0 * torch.dot(coeff, b) + coeff @ A @ coeff
            residual = max(float(residual_t), 0.0)
            neighbors = [ids[i] for i in order]

        unique_fraction = 1.0 if norm_j <= EPS else _clamp01(residual / norm_j)
        subsumed_fraction = 1.0 - unique_fraction
        predicted_loss = max(float(marg[j]), 0.0) * unique_fraction
        mem_j = None if mem is None else int(mem[j])
        savings_per_loss = None
        if mem_j is not None:
            savings_per_loss = float(mem_j) / max(predicted_loss, EPS)
        out.append(ExpertSubsumability(
            expert_id=eid,
            marginal_value=float(marg[j]),
            norm_sq=norm_j,
            neighbor_expert_ids=tuple(neighbors),
            coefficients=tuple(float(x) for x in coeff.tolist()),
            residual_energy=residual,
            subsumed_fraction=subsumed_fraction,
            unique_fraction=unique_fraction,
            predicted_merge_loss=predicted_loss,
            memory_bytes=mem_j,
            savings_per_loss=savings_per_loss,
        ))

    return sorted(
        out,
        key=lambda r: (
            r.predicted_merge_loss,
            -r.subsumed_fraction,
            r.expert_id,
        ),
    )


def build_expert_merge_plan(
    gram: torch.Tensor,
    expert_ids: Sequence[int],
    marginal_values: Mapping[int, float] | Sequence[float],
    *,
    target_experts: int,
    router_qname: str | None = None,
    basis_size: int = 8,
    ridge: float = 1e-6,
    max_cluster_size: int | None = None,
) -> ExpertMergePlan:
    """Build a deterministic greedy merge plan from Gram/SVD statistics."""

    ids = [int(e) for e in expert_ids]
    G = _validate_inputs(gram, ids)
    E = len(ids)
    K = int(target_experts)
    if K <= 0:
        raise ValueError("target_experts must be positive")
    if K > E:
        raise ValueError("target_experts cannot exceed num experts")
    if max_cluster_size is not None and max_cluster_size <= 0:
        raise ValueError("max_cluster_size must be positive when provided")
    if max_cluster_size is not None and math.ceil(E / K) > max_cluster_size:
        raise ValueError("max_cluster_size is too small for target_experts")

    marg = _value_vector(marginal_values, ids, "marginal_values")
    clusters = [_ClusterState((i,), 0.0) for i in range(E)]

    while len(clusters) > K:
        best: tuple[float, tuple[int, ...], int, int, float] | None = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                merged = tuple(sorted(clusters[a].indices + clusters[b].indices))
                if max_cluster_size is not None and len(merged) > max_cluster_size:
                    continue
                residual = _weighted_centroid_residual(G, marg, merged)
                increase = residual - clusters[a].residual - clusters[b].residual
                anchor = _anchor_index(merged, marg, ids)
                candidate = (increase, tuple(ids[i] for i in merged), a, b, residual)
                if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
                    best = (increase, tuple(ids[i] for i in merged), a, b, residual)
        if best is None:
            raise ValueError("no feasible cluster merge found")
        _, _, a, b, residual = best
        merged_indices = tuple(sorted(clusters[a].indices + clusters[b].indices))
        clusters = [
            c for idx, c in enumerate(clusters)
            if idx not in (a, b)
        ]
        clusters.append(_ClusterState(merged_indices, residual))
        clusters.sort(key=lambda c: tuple(ids[i] for i in c.indices))

    clusters.sort(key=lambda c: ids[_anchor_index(c.indices, marg, ids)])
    final_clusters: list[ExpertCluster] = []
    for new_eid, state in enumerate(clusters):
        idxs = state.indices
        anchor_idx = _anchor_index(idxs, marg, ids)
        weights = _normalized_cluster_weights(marg, ids, idxs)
        loss = _weighted_centroid_residual(G, marg, idxs)
        final_clusters.append(ExpertCluster(
            new_expert_id=new_eid,
            anchor_expert_id=ids[anchor_idx],
            original_expert_ids=tuple(ids[i] for i in idxs),
            weights=weights,
            centroid_residual_energy=loss,
            predicted_merge_loss=loss,
            marginal_value_sum=float(marg[list(idxs)].sum()),
        ))

    ranking = rank_expert_subsumability(
        G, ids, marg.tolist(), basis_size=basis_size, ridge=ridge,
    )
    return ExpertMergePlan(
        router_qname=router_qname,
        num_experts_orig=E,
        num_experts_kept=K,
        clusters=tuple(final_clusters),
        ranking=tuple(ranking),
        total_predicted_merge_loss=sum(c.predicted_merge_loss for c in final_clusters),
    )


def select_rrqr_anchors_from_gram(
    gram: torch.Tensor,
    expert_ids: Sequence[int],
    *,
    target_experts: int,
    ridge: float = 1e-10,
) -> list[int]:
    """Select deterministic RRQR-style anchor experts from a Gram matrix.

    This is a Gram-space residual pivot selector: at each step it adds the
    column with the largest residual norm after projection onto the selected
    anchor span.  For Entmoot's small per-layer expert counts this gives the
    stability/reproducibility we need without depending on SciPy.
    """

    ids = [int(e) for e in expert_ids]
    G = _validate_inputs(gram, ids)
    E = len(ids)
    K = int(target_experts)
    if K <= 0:
        raise ValueError("target_experts must be positive")
    if K > E:
        raise ValueError("target_experts cannot exceed num experts")

    selected: list[int] = []
    selected_set: set[int] = set()
    residual = G.diag().clamp_min(0.0).clone()
    for _ in range(K):
        candidates = [i for i in range(E) if i not in selected_set]
        pivot = max(candidates, key=lambda i: (float(residual[i]), -ids[i]))
        selected.append(pivot)
        selected_set.add(pivot)
        S = torch.tensor(selected, dtype=torch.long)
        A = G[S][:, S]
        diag_scale = float(A.diag().mean().clamp_min(EPS))
        A_reg = A + torch.eye(len(selected), dtype=torch.float64) * (
            float(ridge) * diag_scale
        )
        B = G[S, :]
        try:
            coeff = torch.linalg.solve(A_reg, B)
        except RuntimeError:
            coeff = torch.linalg.lstsq(A_reg, B).solution
        proj_norm = (B * coeff).sum(dim=0)
        residual = (G.diag() - proj_norm).clamp_min(0.0)
        for i in selected:
            residual[i] = 0.0
    return [ids[i] for i in selected]


def build_router_id_merge_plan(
    features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    marginal_values: Mapping[int, float] | Sequence[float],
    *,
    target_experts: int,
    expert_ids: Sequence[int] | None = None,
    router_qname: str | None = None,
    routed_mass: Mapping[int, float] | Sequence[float] | None = None,
    sample_counts: Mapping[int, int] | Sequence[int] | None = None,
    activation_accept_threshold: float = 0.05,
    activation_tentative_threshold: float = 0.10,
    min_routed_mass: float = 0.0,
    min_samples: int = 0,
    ridge: float = 1e-10,
    router_strategy: str = "anchor",
    anchor_residuals: Mapping[tuple[int, int], float] | None = None,
    candidate_anchor_ids: Mapping[int, Sequence[int]] | None = None,
    synthesis_weights: Mapping[tuple[int, int], Mapping[int, float]] | None = None,
    tensor_synthesis_weights: Mapping[
        tuple[int, int], Mapping[str, Mapping[int, float]]
    ] | None = None,
) -> ExpertMergePlan:
    """Build the v1 router-aware hard-anchor Entmoot plan.

    Accepted non-anchor experts map to one selected anchor and receive
    one-hot expert synthesis weights, so the exported expert is exactly the
    surviving anchor.  Rejected experts become identity clusters and stay in
    the dense K-row checkpoint.
    """

    ids, G = _features_to_ids_and_gram(features, expert_ids=expert_ids)
    E = len(ids)
    K = int(target_experts)
    if K <= 0:
        raise ValueError("target_experts must be positive")
    if K > E:
        raise ValueError("target_experts cannot exceed num experts")
    if activation_accept_threshold < 0 or activation_tentative_threshold < 0:
        raise ValueError("activation thresholds must be non-negative")
    if activation_accept_threshold > activation_tentative_threshold:
        raise ValueError("activation_accept_threshold cannot exceed tentative threshold")

    marg = _value_vector(marginal_values, ids, "marginal_values")
    mass = (
        _value_vector(routed_mass, ids, "routed_mass")
        if routed_mass is not None else marg.clamp_min(0.0)
    )
    samples = _optional_int_vector(sample_counts, ids, "sample_counts")
    anchor_ids = select_rrqr_anchors_from_gram(
        G, ids, target_experts=K, ridge=ridge,
    )
    anchor_set = set(anchor_ids)
    id_to_idx = {eid: i for i, eid in enumerate(ids)}

    accepted_by_anchor: dict[int, list[int]] = {a: [a] for a in anchor_ids}
    cluster_weights_by_anchor: dict[int, dict[int, float]] = {
        a: {a: 1.0} for a in anchor_ids
    }
    cluster_tensor_weights_by_anchor: dict[
        int, dict[str, tuple[tuple[int, float], ...]]
    ] = {}
    cluster_loss_by_anchor: dict[int, float] = {a: 0.0 for a in anchor_ids}
    rejected: list[int] = []
    decisions_by_eid: dict[int, dict[str, Any]] = {}
    total_loss = 0.0

    for eid in ids:
        if eid not in anchor_set:
            continue
        j = id_to_idx[eid]
        decisions_by_eid[eid] = {
            "expert_id": eid,
            "merge_action": "anchor",
            "decision": "keep",
            "target_expert_id": eid,
            "activation_energy_relative": 0.0,
            "routed_mass": float(mass[j]),
            "samples": None if samples is None else int(samples[j]),
        }

    if synthesis_weights is not None:
        candidate_records: list[tuple[float, int, int, dict[int, float]]] = []
        best_seen: dict[int, tuple[float, int] | None] = {}
        support_by_eid: dict[int, bool] = {}
        for eid in ids:
            if eid in anchor_set:
                continue
            j = id_to_idx[eid]
            sample_count = None if samples is None else int(samples[j])
            support_ok = (
                float(mass[j]) >= float(min_routed_mass)
                and (sample_count is None or sample_count >= int(min_samples))
            )
            support_by_eid[eid] = support_ok
            candidates = _candidate_anchor_list(eid, anchor_ids, anchor_set, candidate_anchor_ids)
            best: tuple[float, int] | None = None
            for anchor in candidates:
                weights = _candidate_synthesis_weights(synthesis_weights, eid, anchor)
                if weights is None:
                    continue
                tensor_weights = _candidate_tensor_synthesis_weights(
                    tensor_synthesis_weights, eid, anchor,
                )
                if tensor_synthesis_weights is not None and tensor_weights is None:
                    continue
                rel = _anchor_relative_residual(
                    G, j, id_to_idx[anchor], eid, anchor, anchor_residuals,
                )
                if best is None or (rel, anchor) < best:
                    best = (rel, anchor)
                if support_ok and rel <= float(activation_accept_threshold):
                    candidate_records.append((rel, eid, anchor, weights))
            best_seen[eid] = best

        accepted_eids: set[int] = set()
        used_anchors: set[int] = set()
        for rel, eid, anchor, weights in sorted(candidate_records):
            if eid in accepted_eids or anchor in used_anchors:
                continue
            j = id_to_idx[eid]
            a_idx = id_to_idx[anchor]
            accepted_eids.add(eid)
            used_anchors.add(anchor)
            accepted_by_anchor[anchor].append(eid)
            cluster_weights_by_anchor[anchor] = weights
            tensor_weights = _candidate_tensor_synthesis_weights(
                tensor_synthesis_weights, eid, anchor,
            )
            if tensor_weights is not None:
                cluster_tensor_weights_by_anchor[anchor] = tensor_weights
            cluster_loss = max(float(marg[a_idx] + marg[j]), 0.0) * rel
            cluster_loss_by_anchor[anchor] = cluster_loss
            total_loss += cluster_loss
            decisions_by_eid[eid] = {
                "expert_id": eid,
                "merge_action": "merge",
                "decision": "accept",
                "target_expert_id": anchor,
                "activation_energy_relative": rel,
                "routed_mass": float(mass[j]),
                "samples": None if samples is None else int(samples[j]),
                "synthesis_weights": {str(k): v for k, v in weights.items()},
            }
            if tensor_weights is not None:
                decisions_by_eid[eid]["tensor_synthesis_weights"] = {
                    name: {str(k): v for k, v in w}
                    for name, w in sorted(tensor_weights.items())
                }

        for eid in ids:
            if eid in anchor_set or eid in accepted_eids:
                continue
            j = id_to_idx[eid]
            sample_count = None if samples is None else int(samples[j])
            best = best_seen.get(eid)
            rel = None if best is None else float(best[0])
            target = eid if best is None else int(best[1])
            if not support_by_eid.get(eid, False):
                reason = "support below threshold"
            elif best is None:
                reason = "no measured synthesis candidate"
            elif float(rel) <= float(activation_accept_threshold):
                reason = "synthesis candidate conflicted with a lower-residual pair"
            elif float(rel) <= float(activation_tentative_threshold):
                reason = "activation residual tentative; kept for v1 strict default"
            else:
                reason = "activation residual above threshold"
            rejected.append(eid)
            decisions_by_eid[eid] = {
                "expert_id": eid,
                "merge_action": "identity",
                "decision": "reject",
                "target_expert_id": target,
                "activation_energy_relative": rel,
                "routed_mass": float(mass[j]),
                "samples": sample_count,
                "rejection_reason": reason,
            }
    else:
        for eid in ids:
            j = id_to_idx[eid]
            if eid in anchor_set:
                continue

            candidates = _candidate_anchor_list(eid, anchor_ids, anchor_set, candidate_anchor_ids)
            best_anchor = min(
                candidates,
                key=lambda a: (
                    _anchor_relative_residual(
                        G, j, id_to_idx[a], eid, a, anchor_residuals,
                    ),
                    a,
                ),
            )
            rel = _anchor_relative_residual(
                G, j, id_to_idx[best_anchor], eid, best_anchor, anchor_residuals,
            )
            sample_count = None if samples is None else int(samples[j])
            support_ok = (
                float(mass[j]) >= float(min_routed_mass)
                and (sample_count is None or sample_count >= int(min_samples))
            )
            if not support_ok:
                reason = "support below threshold"
            elif rel <= float(activation_accept_threshold):
                reason = ""
            elif rel <= float(activation_tentative_threshold):
                reason = "activation residual tentative; kept for v1 strict default"
            else:
                reason = "activation residual above threshold"

            if support_ok and rel <= float(activation_accept_threshold):
                accepted_by_anchor[best_anchor].append(eid)
                cluster_weights_by_anchor[best_anchor][eid] = 0.0
                loss = max(float(marg[j]), 0.0) * rel
                cluster_loss_by_anchor[best_anchor] += loss
                total_loss += loss
                decisions_by_eid[eid] = {
                    "expert_id": eid,
                    "merge_action": "merge",
                    "decision": "accept",
                    "target_expert_id": best_anchor,
                    "activation_energy_relative": rel,
                    "routed_mass": float(mass[j]),
                    "samples": sample_count,
                }
            else:
                rejected.append(eid)
                decisions_by_eid[eid] = {
                    "expert_id": eid,
                    "merge_action": "identity",
                    "decision": "reject",
                    "target_expert_id": eid,
                    "activation_energy_relative": rel,
                    "routed_mass": float(mass[j]),
                    "samples": sample_count,
                    "rejection_reason": reason,
                }

    cluster_specs: list[tuple[int, list[int], str, str | None]] = []
    has_accepted_merges = any(len(orig_ids) > 1 for orig_ids in accepted_by_anchor.values())
    if has_accepted_merges:
        for anchor in anchor_ids:
            cluster_specs.append((anchor, sorted(accepted_by_anchor[anchor]), "anchor", None))
        for eid in sorted(rejected):
            cluster_specs.append((eid, [eid], "identity", decisions_by_eid[eid]["rejection_reason"]))
    else:
        for eid in ids:
            cluster_specs.append((
                eid,
                [eid],
                "identity",
                decisions_by_eid[eid].get("rejection_reason"),
            ))

    clusters: list[ExpertCluster] = []
    expert_decisions: list[dict[str, Any]] = []
    for new_eid, (anchor, orig_ids, action, rejection_reason) in enumerate(cluster_specs):
        orig_set = set(orig_ids)
        raw_weights = (
            {anchor: 1.0}
            if action == "identity"
            else cluster_weights_by_anchor.get(anchor, {anchor: 1.0})
        )
        weights = tuple((eid, float(raw_weights.get(eid, 0.0))) for eid in orig_ids)
        tensor_weights = (
            None if action == "identity"
            else cluster_tensor_weights_by_anchor.get(anchor)
        )
        router_weights = _router_mass_weights(mass, ids, orig_ids)
        residual_sum = float(cluster_loss_by_anchor.get(anchor, 0.0))
        decision = "keep" if action == "identity" else "accept"
        clusters.append(ExpertCluster(
            new_expert_id=new_eid,
            anchor_expert_id=anchor,
            original_expert_ids=tuple(orig_ids),
            weights=weights,
            router_weights=router_weights,
            tensor_weights=tensor_weights,
            centroid_residual_energy=residual_sum,
            predicted_merge_loss=residual_sum,
            marginal_value_sum=float(sum(marg[id_to_idx[eid]] for eid in orig_ids)),
            merge_action=action,
            decision=decision,
            residual_metrics={"activation_energy_relative_sum": residual_sum},
            rejection_reason=rejection_reason,
        ))
        for eid in orig_ids:
            d = dict(decisions_by_eid[eid])
            d["new_expert_id"] = new_eid
            d["cluster_anchor_expert_id"] = anchor
            if eid in orig_set:
                expert_decisions.append(d)

    ranking = rank_expert_subsumability(
        G, ids, marg.tolist(), basis_size=min(8, max(E - 1, 0)), ridge=ridge,
    )
    return ExpertMergePlan(
        router_qname=router_qname,
        num_experts_orig=E,
        num_experts_kept=len(clusters),
        clusters=tuple(clusters),
        ranking=tuple(ranking),
        total_predicted_merge_loss=total_loss,
        method="entmoot_router_id_v1",
        router_strategy=router_strategy,
        expert_decisions=tuple(sorted(expert_decisions, key=lambda d: int(d["expert_id"]))),
        diagnostics={
            "target_experts_requested": K,
            "activation_accept_threshold": float(activation_accept_threshold),
            "activation_tentative_threshold": float(activation_tentative_threshold),
            "min_routed_mass": float(min_routed_mass),
            "min_samples": int(min_samples),
            "anchor_selection": "rrqr_residual_pivot_v1",
            "activation_residual_source": (
                "same_input_expert_forward_v1"
                if anchor_residuals is not None else "feature_gram_proxy_v1"
            ),
        },
    )


def merge_plan_manifest(plans: Sequence[ExpertMergePlan]) -> dict[str, dict]:
    """Combine per-router plans into a JSON-serializable manifest."""

    manifest: dict[str, dict] = {}
    for idx, plan in enumerate(plans):
        key = plan.router_qname or f"__unknown_router_{idx}__"
        if key in manifest:
            raise ValueError(f"duplicate router key in merge plans: {key}")
        manifest[key] = plan.to_manifest_entry()
    return manifest


def save_merge_manifest(plans: Sequence[ExpertMergePlan], path: str | Path) -> None:
    """Write a merge manifest JSON sidecar."""

    p = Path(path)
    with open(p, "w") as f:
        json.dump(merge_plan_manifest(plans), f, indent=2, sort_keys=True)


@dataclass(frozen=True)
class _ClusterState:
    indices: tuple[int, ...]
    residual: float


def _as_square_gram(gram: torch.Tensor) -> torch.Tensor:
    G = torch.as_tensor(gram, dtype=torch.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("gram must be a square matrix")
    if not torch.isfinite(G).all():
        raise ValueError("gram must contain only finite values")
    return 0.5 * (G + G.T)


def _validate_inputs(gram: torch.Tensor, expert_ids: Sequence[int]) -> torch.Tensor:
    G = _as_square_gram(gram)
    if len(expert_ids) != G.shape[0]:
        raise ValueError("expert_ids length must match gram size")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("expert_ids must be unique")
    return G


def _value_vector(
    values: Mapping[int, float] | Sequence[float],
    expert_ids: Sequence[int],
    name: str,
) -> torch.Tensor:
    if isinstance(values, Mapping):
        missing = [e for e in expert_ids if e not in values and str(e) not in values]
        if missing:
            raise ValueError(f"{name} missing expert ids: {missing[:5]}")
        out = [
            float(values[e] if e in values else values[str(e)])
            for e in expert_ids
        ]
    else:
        if len(values) != len(expert_ids):
            raise ValueError(f"{name} length must match expert_ids")
        out = [float(v) for v in values]
    if any(not math.isfinite(v) for v in out):
        raise ValueError(f"{name} must contain only finite values")
    return torch.tensor(out, dtype=torch.float64)


def _optional_int_vector(
    values: Mapping[int, int] | Sequence[int] | None,
    expert_ids: Sequence[int],
    name: str,
) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        missing = [e for e in expert_ids if e not in values and str(e) not in values]
        if missing:
            raise ValueError(f"{name} missing expert ids: {missing[:5]}")
        out = [
            int(values[e] if e in values else values[str(e)])
            for e in expert_ids
        ]
    else:
        if len(values) != len(expert_ids):
            raise ValueError(f"{name} length must match expert_ids")
        out = [int(v) for v in values]
    if any(v < 0 for v in out):
        raise ValueError(f"{name} must be non-negative")
    return out


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _weighted_centroid_residual(
    gram: torch.Tensor,
    values: torch.Tensor,
    indices: Sequence[int],
) -> float:
    idx = torch.tensor(list(indices), dtype=torch.long)
    w = values[idx].clamp_min(0.0)
    if float(w.sum()) <= EPS:
        w = torch.ones_like(w)
    W = w.sum().clamp_min(EPS)
    Gc = gram[idx][:, idx]
    weighted_norm = torch.dot(w, Gc.diag())
    centroid_norm = (w[:, None] * Gc * w[None, :]).sum() / W
    return max(float(weighted_norm - centroid_norm), 0.0)


def _anchor_index(
    indices: Sequence[int],
    values: torch.Tensor,
    expert_ids: Sequence[int],
) -> int:
    return max(indices, key=lambda i: (float(values[i]), -expert_ids[i]))


def _normalized_cluster_weights(
    values: torch.Tensor,
    expert_ids: Sequence[int],
    indices: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    idxs = list(indices)
    w = values[idxs].clamp_min(0.0)
    if float(w.sum()) <= EPS:
        w = torch.ones_like(w)
    w = w / w.sum().clamp_min(EPS)
    return tuple((expert_ids[i], float(w[n])) for n, i in enumerate(idxs))


def _features_to_ids_and_gram(
    features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    *,
    expert_ids: Sequence[int] | None,
) -> tuple[list[int], torch.Tensor]:
    if isinstance(features, torch.Tensor):
        X = features.detach()
        if X.ndim != 2:
            raise ValueError("feature tensor must be 2D [experts, dims]")
        ids = list(range(int(X.shape[0]))) if expert_ids is None else [int(e) for e in expert_ids]
        if len(ids) != int(X.shape[0]):
            raise ValueError("expert_ids length must match feature rows")
        return ids, X.to(torch.float64) @ X.to(torch.float64).T
    return gram_from_features(features, expert_ids=expert_ids)


def _direct_relative_residual(gram: torch.Tensor, j: int, anchor: int) -> float:
    norm_j = float(gram[j, j].clamp_min(EPS))
    residual = gram[j, j] + gram[anchor, anchor] - 2.0 * gram[j, anchor]
    return _clamp01(max(float(residual), 0.0) / norm_j)


def _anchor_relative_residual(
    gram: torch.Tensor,
    j: int,
    anchor: int,
    eid: int,
    anchor_eid: int,
    anchor_residuals: Mapping[tuple[int, int], float] | None,
) -> float:
    if anchor_residuals is not None:
        value = anchor_residuals.get((int(eid), int(anchor_eid)))
        if value is not None:
            return max(float(value), 0.0)
    return _direct_relative_residual(gram, j, anchor)


def _candidate_anchor_list(
    eid: int,
    anchor_ids: Sequence[int],
    anchor_set: set[int],
    candidate_anchor_ids: Mapping[int, Sequence[int]] | None,
) -> list[int]:
    candidates = list(int(a) for a in anchor_ids)
    if candidate_anchor_ids is not None:
        raw_candidates = [
            int(a) for a in candidate_anchor_ids.get(int(eid), [])
            if int(a) in anchor_set
        ]
        if raw_candidates:
            candidates = raw_candidates
    if not candidates:
        raise ValueError(f"no candidate anchors for expert {eid}")
    return candidates


def _candidate_synthesis_weights(
    synthesis_weights: Mapping[tuple[int, int], Mapping[int, float]],
    eid: int,
    anchor: int,
) -> dict[int, float] | None:
    raw = synthesis_weights.get((int(eid), int(anchor)))
    if raw is None:
        return None
    expected = {int(anchor), int(eid)}
    out = {int(k): max(float(v), 0.0) for k, v in raw.items()}
    if set(out) != expected:
        raise ValueError(
            "synthesis weight keys must exactly match anchor and merged expert"
        )
    total = sum(out.values())
    if total <= EPS:
        raise ValueError("synthesis weights must have positive sum")
    return {k: v / total for k, v in sorted(out.items())}


def _candidate_tensor_synthesis_weights(
    tensor_synthesis_weights: Mapping[
        tuple[int, int], Mapping[str, Mapping[int, float]]
    ] | None,
    eid: int,
    anchor: int,
) -> dict[str, tuple[tuple[int, float], ...]] | None:
    if tensor_synthesis_weights is None:
        return None
    raw = tensor_synthesis_weights.get((int(eid), int(anchor)))
    if raw is None:
        return None
    out: dict[str, tuple[tuple[int, float], ...]] = {}
    for name, weights in raw.items():
        norm = _candidate_synthesis_weights(
            {(int(eid), int(anchor)): weights}, eid, anchor,
        )
        if norm is None:
            continue
        out[str(name)] = tuple((k, v) for k, v in sorted(norm.items()))
    return out


def _router_mass_weights(
    mass: torch.Tensor,
    expert_ids: Sequence[int],
    orig_ids: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    idx = [expert_ids.index(int(eid)) for eid in orig_ids]
    w = mass[idx].clamp_min(0.0)
    if float(w.sum()) <= EPS:
        w = torch.ones_like(w)
    w = w / w.sum().clamp_min(EPS)
    return tuple((int(eid), float(w[n])) for n, eid in enumerate(orig_ids))


def _load_saliency(path: str | Path, router: str | None) -> dict[int, float]:
    with open(path) as f:
        payload = json.load(f)
    if router is not None and router in payload:
        payload = payload[router]
    return {int(k): float(v) for k, v in payload.items()}


def _load_gram_npz(path: str | Path) -> tuple[list[int], torch.Tensor]:
    import numpy as np

    data = np.load(path)
    if "gram" not in data:
        raise ValueError("NPZ must contain a 'gram' array")
    gram = torch.from_numpy(data["gram"])
    if "expert_ids" in data:
        expert_ids = [int(x) for x in data["expert_ids"].tolist()]
    else:
        expert_ids = list(range(int(gram.shape[0])))
    return expert_ids, gram


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an Entmoot MoE expert merge manifest from a Gram matrix.",
    )
    parser.add_argument("--gram-npz", required=True, help="NPZ with gram and optional expert_ids")
    parser.add_argument("--saliency-json", required=True, help="JSON {eid: score} or {router: {eid: score}}")
    parser.add_argument("--router-qname", default=None)
    parser.add_argument("--target-experts", type=int, required=True)
    parser.add_argument("--basis-size", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--max-cluster-size", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    expert_ids, gram = _load_gram_npz(args.gram_npz)
    saliency = _load_saliency(args.saliency_json, args.router_qname)
    plan = build_expert_merge_plan(
        gram,
        expert_ids,
        saliency,
        target_experts=args.target_experts,
        router_qname=args.router_qname,
        basis_size=args.basis_size,
        ridge=args.ridge,
        max_cluster_size=args.max_cluster_size,
    )
    save_merge_manifest([plan], args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
