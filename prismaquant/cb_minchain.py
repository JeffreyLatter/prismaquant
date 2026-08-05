"""Monotone min-chain helpers for production product-CB encodes.

The encoder compares today's unconstrained free fit with the selected
predecessor embedded at the next rung.  Selection uses the reported weight-MSE
metric and the registered symmetric epsilon.  Each selected payload remains an
ordinary flat product-CB payload; there is no decoder or serving dependency on
the chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .cb_layout import VEC_DIM, codebook_subtable_shapes, family_for
from .nvfp4_cb_formats import (
    FP8_ELEMENT_MAX,
    _snap_to_grid,
    ldlq_reassign_cb_fields,
    nvfp4_cb_fields,
)


MINCHAIN_SCHEMA = "prismaquant.cb_minchain.v1"
MINCHAIN_CONTEXT_VERSION = "minchain-v1"
MINCHAIN_FLAG = "PRISMAQUANT_CB_MINCHAIN"
MINCHAIN_ANCHORS_ENV = "PRISMAQUANT_CB_MINCHAIN_ANCHORS"
MINCHAIN_HOLDBACKS_ENV = "PRISMAQUANT_CB_MINCHAIN_HOLDBACKS"
MINCHAIN_AUDIT_SEED_ENV = "PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED"
MINCHAIN_BACKSTOP_ENV = "PRISMAQUANT_CB_MINCHAIN_BACKSTOP"
MINCHAIN_AUDIT_MEDIAN_ENV = "PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN"
MINCHAIN_AUDIT_P95_ENV = "PRISMAQUANT_CB_MINCHAIN_AUDIT_P95"


def minchain_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Resolve the production flag, rejecting ambiguous spellings."""
    values = os.environ if environ is None else environ
    raw = str(values.get(MINCHAIN_FLAG, "0")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{MINCHAIN_FLAG} must be a boolean 0/1 setting, got {raw!r}")


def require_minchain_enabled() -> None:
    if not minchain_enabled():
        raise RuntimeError(f"min-chain execution requires {MINCHAIN_FLAG}=1")


# Pilot-era import compatibility.  The implementation and gate are production
# now; keeping the name avoids invalidating immutable pilot readers.
require_pilot_enabled = require_minchain_enabled


def _tensor_digest(hasher: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().contiguous().cpu()
    header = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    hasher.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    hasher.update(b"\0")


def solution_digest(fields: Mapping[str, Any]) -> str:
    """Digest the serialized solution planes and resolved flat codebook."""
    hasher = hashlib.sha256()
    hasher.update(MINCHAIN_SCHEMA.encode("ascii"))
    shape = fields.get("shape")
    if shape is not None:
        hasher.update(json.dumps({
            "logical_shape": [int(dim) for dim in shape]
        }, sort_keys=True).encode("utf-8"))
        hasher.update(b"\0")
    for key in ("indices", "scales", "signs", "scale_super", "scale_sub"):
        value = fields.get(key)
        if isinstance(value, torch.Tensor):
            _tensor_digest(hasher, key, value)
    codebook = fields.get("codebook")
    if isinstance(codebook, torch.Tensor):
        _tensor_digest(hasher, "codebook", codebook)
    elif codebook is not None:
        for index, table in enumerate(codebook):
            _tensor_digest(hasher, f"codebook.{index}", table)
    return hasher.hexdigest()


def chain_identity(
    *, winning_arm: str, solution: Mapping[str, Any],
    predecessor_digest: str | None,
) -> dict[str, Any]:
    if winning_arm not in {"free", "embed", "refine"}:
        raise ValueError(f"unknown min-chain arm {winning_arm!r}")
    if winning_arm in {"embed", "refine"} and not predecessor_digest:
        raise ValueError(f"{winning_arm} requires a predecessor digest")
    if winning_arm == "free" and predecessor_digest is not None:
        raise ValueError("free arm cannot claim a predecessor digest")
    return {
        "schema": MINCHAIN_SCHEMA,
        "chain_version": MINCHAIN_CONTEXT_VERSION,
        "winning_arm": winning_arm,
        "predecessor_digest": predecessor_digest,
        "solution_digest": solution_digest(solution),
    }


def recipe_solution_digest(recipe: Mapping[str, Any]) -> str:
    """Digest a deterministic encode recipe without rereading tensor planes.

    Large campaigns use this identity after source/imatrix/activation content
    has already been fail-closed at load.  It keeps bookkeeping off the hot
    path while still changing on every input, context, arm, predecessor, rung,
    or algorithm-version change.
    """
    payload = {
        "schema": MINCHAIN_SCHEMA,
        "chain_version": MINCHAIN_CONTEXT_VERSION,
        "deterministic_recipe": dict(recipe),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def chain_identity_from_digest(
    *, winning_arm: str, solution_digest_value: str,
    predecessor_digest: str | None,
) -> dict[str, Any]:
    if winning_arm not in {"free", "embed", "refine"}:
        raise ValueError(f"unknown min-chain arm {winning_arm!r}")
    if winning_arm in {"embed", "refine"} and not predecessor_digest:
        raise ValueError(f"{winning_arm} requires a predecessor digest")
    if winning_arm == "free" and predecessor_digest is not None:
        raise ValueError("free arm cannot claim a predecessor digest")
    digest = str(solution_digest_value).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("solution digest must be lowercase sha256 hex")
    return {
        "schema": MINCHAIN_SCHEMA,
        "chain_version": MINCHAIN_CONTEXT_VERSION,
        "winning_arm": winning_arm,
        "predecessor_digest": predecessor_digest,
        "solution_digest": digest,
        "digest_basis": "deterministic_content_gated_recipe",
    }


def validate_chain_identity(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Validate one persisted per-cell arm/digest record."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: min-chain identity is not an object")
    if value.get("schema") != MINCHAIN_SCHEMA:
        raise ValueError(f"{where}: unsupported min-chain schema {value.get('schema')!r}")
    if value.get("chain_version") != MINCHAIN_CONTEXT_VERSION:
        raise ValueError(
            f"{where}: min-chain version {value.get('chain_version')!r} "
            f"!= {MINCHAIN_CONTEXT_VERSION!r}"
        )
    arm = str(value.get("winning_arm", ""))
    predecessor = value.get("predecessor_digest")
    digest = str(value.get("solution_digest", "")).lower()
    if arm not in {"free", "embed", "refine"}:
        raise ValueError(f"{where}: unknown min-chain arm {arm!r}")
    if arm == "free" and predecessor is not None:
        raise ValueError(f"{where}: free arm cannot claim a predecessor digest")
    if arm in {"embed", "refine"} and not predecessor:
        raise ValueError(f"{where}: {arm} arm requires a predecessor digest")
    for label, raw_digest in (("solution", digest), ("predecessor", predecessor)):
        if raw_digest is None:
            continue
        text = str(raw_digest).lower()
        if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
            raise ValueError(f"{where}: {label} digest is not lowercase SHA-256")
    return dict(value)


def _expanded_codebook(
    predecessor: Mapping[str, Any], target_k: int,
    *, grid: str, mode: str,
) -> tuple[torch.Tensor, ...]:
    if mode != "product":
        raise ValueError("the min-chain pilot supports product CB only")
    old = tuple(predecessor["codebook"])
    n_sub = family_for(grid, mode).n_sub
    expected = codebook_subtable_shapes(target_k, mode, n_sub)
    if len(old) != len(expected):
        raise ValueError("predecessor subtable count differs from target")
    expanded = []
    for table, shape in zip(old, expected):
        if table.ndim != 2 or table.shape[1] != shape[1]:
            raise ValueError("predecessor subtable dimension differs")
        if table.shape[0] > shape[0]:
            raise ValueError("target rung is smaller than predecessor")
        if table.shape[0] == shape[0]:
            expanded.append(table.clone())
            continue
        padding = table[-1:].expand(shape[0] - table.shape[0], -1).clone()
        expanded.append(torch.cat((table, padding), dim=0))
    return tuple(expanded)


def embed_predecessor(
    predecessor: Mapping[str, Any], target_k: int,
    *, grid: str = "fp8", mode: str = "product",
) -> dict[str, Any]:
    """Embed a finished solution verbatim, padding only the flat tables."""
    require_pilot_enabled()
    out = dict(predecessor)
    out["codebook"] = _expanded_codebook(
        predecessor, target_k, grid=grid, mode=mode
    )
    return out


def _scaled_vectors(
    weight: torch.Tensor, scales: torch.Tensor, *, grid: str,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("refine_one_entry expects one 2-D slice")
    if grid != "fp8":
        raise ValueError("the registered pilot refines FP8-CB only")
    per_element = scales.to(torch.float32)
    if per_element.shape[-1] == 1:
        per_element = per_element.expand_as(weight)
    return (weight.float() / per_element.clamp_min(1e-12)).reshape(-1, VEC_DIM)


def _refine_new_centroid(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    old_table: torch.Tensor,
    *, iterations: int,
) -> torch.Tensor:
    # The candidate starts at the currently worst represented vector, then
    # competes against the frozen predecessor entries for at most three Lloyd
    # updates.  All old entries remain byte-verbatim.
    # Padding duplicates are semantically one competitor.  Deduplicating only
    # the distance evaluator keeps the emitted predecessor prefix untouched
    # while preventing repeated add-one steps from becoming exponential work.
    old_table = torch.unique(old_table, dim=0)
    chunk_rows = 1 << 15
    worst_values = []
    worst_vectors = []
    for start in range(0, vectors.shape[0], chunk_rows):
        stop = min(start + chunk_rows, vectors.shape[0])
        local_vectors = vectors[start:stop]
        local_weights = weights[start:stop]
        old_best = (
            (local_vectors[:, None, :] - old_table[None, :, :]).square()
            * local_weights[:, None, :]
        ).sum(-1).min(-1).values
        value, index = old_best.max(0)
        worst_values.append(value)
        worst_vectors.append(local_vectors[index])
    worst_chunk = torch.stack(worst_values).argmax()
    centroid = torch.stack(worst_vectors)[worst_chunk].clone()
    centroid = _snap_to_grid(centroid, "fp8").clamp(
        -FP8_ELEMENT_MAX, FP8_ELEMENT_MAX
    )
    for _ in range(iterations):
        numerator = torch.zeros_like(centroid)
        denominator = torch.zeros_like(centroid)
        selected_count = torch.zeros((), device=vectors.device, dtype=torch.int64)
        for start in range(0, vectors.shape[0], chunk_rows):
            stop = min(start + chunk_rows, vectors.shape[0])
            local_vectors = vectors[start:stop]
            local_weights = weights[start:stop]
            old_best = (
                (local_vectors[:, None, :] - old_table[None, :, :]).square()
                * local_weights[:, None, :]
            ).sum(-1).min(-1).values
            new_distance = (
                (local_vectors - centroid).square() * local_weights
            ).sum(-1)
            selected = new_distance < old_best
            chosen_weights = local_weights[selected]
            numerator += (local_vectors[selected] * chosen_weights).sum(0)
            denominator += chosen_weights.sum(0)
            selected_count += selected.sum()
        if int(selected_count.item()) == 0:
            break
        updated = _snap_to_grid(
            numerator / denominator.clamp_min(1e-30), "fp8"
        ).clamp(
            -FP8_ELEMENT_MAX, FP8_ELEMENT_MAX
        )
        if torch.equal(updated, centroid):
            break
        centroid = updated
    return centroid


def refine_one_entry(
    weight: torch.Tensor,
    predecessor: Mapping[str, Any],
    target_k: int,
    *,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor,
    grid: str = "fp8",
    mode: str = "product",
    iterations: int = 3,
) -> dict[str, Any]:
    """Cheap add-one-entry arm with frozen old tables and shared scales."""
    require_pilot_enabled()
    if not 0 <= int(iterations) <= 3:
        raise ValueError("min-chain refinement permits at most 3 iterations")
    embedded = embed_predecessor(
        predecessor, target_k, grid=grid, mode=mode
    )
    vectors = _scaled_vectors(weight, predecessor["scales"], grid=grid)
    weights = torch.broadcast_to(
        col_weights.to(device=weight.device, dtype=torch.float32), weight.shape
    )
    weights = weights.reshape(-1, VEC_DIM)
    sub_dim = VEC_DIM // len(embedded["codebook"])
    refined_tables = []
    for sub_index, (old_table, padded) in enumerate(zip(
        predecessor["codebook"], embedded["codebook"]
    )):
        old_count = old_table.shape[0]
        if old_count == padded.shape[0]:
            refined_tables.append(padded)
            continue
        sl = slice(sub_index * sub_dim, (sub_index + 1) * sub_dim)
        centroid = _refine_new_centroid(
            vectors[:, sl], weights[:, sl], old_table.float(),
            iterations=int(iterations),
        )
        tail = centroid.reshape(1, -1).expand(
            padded.shape[0] - old_count, -1
        ).clone()
        refined_tables.append(torch.cat((old_table, tail), dim=0))
    warm = {"scales": predecessor["scales"]}
    fields = nvfp4_cb_fields(
        weight,
        target_k,
        grid=grid,
        mode=mode,
        col_weights=col_weights,
        codebook=tuple(refined_tables),
        scale_sweep=True,
        warm_scale_state=warm,
        encode_tier="balanced",
    )
    fields = ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activation_rows,
        grid=grid,
        mode=mode,
    )
    if not torch.equal(fields["scales"], predecessor["scales"]):
        raise AssertionError("min-chain shared-scale refinement changed scales")
    return fields


def relative_epsilon(a: float, b: float, *, rtol: float = 1e-12) -> float:
    """Registered symmetric relative epsilon for min-chain comparisons."""
    return float(rtol) * max(abs(float(a)), abs(float(b)))


def epsilon_le(a: float, b: float, *, rtol: float = 1e-12) -> bool:
    """Return ``a <= b`` under the campaign's registered epsilon semantics."""
    return float(a) <= float(b) + relative_epsilon(a, b, rtol=rtol)


def select_arm(
    errors: Mapping[str, float], *, rtol: float = 1e-12,
) -> tuple[str, float]:
    """Select deterministically, treating epsilon-close values as ties.

    ``free`` has identity-stability priority, followed by ``embed`` and the
    legacy pilot-only ``refine`` arm.  The optimized A-FAST path supplies only
    ``free`` and ``embed``; keeping the optional third arm here preserves the
    immutable pilot-1 reader without putting refine back into the campaign.
    """
    if "free" not in errors:
        raise ValueError("min-chain selection requires the free arm")
    unknown = set(errors).difference({"free", "embed", "refine"})
    if unknown:
        raise ValueError(f"unknown min-chain arms: {sorted(unknown)}")
    order = tuple(arm for arm in ("free", "embed", "refine") if arm in errors)
    winner = "free"
    best = float(errors[winner])
    for arm in order[1:]:
        value = float(errors[arm])
        # A difference inside the registered epsilon is a tie.  Earlier arms
        # win ties, making exact and near-exact free identities stable.
        if value < best - relative_epsilon(value, best, rtol=rtol):
            winner, best = arm, value
    return winner, best


def _pava_decreasing(values: np.ndarray) -> np.ndarray:
    """Pilot-identical decreasing isotonic projection used before PCHIP."""
    means: list[float] = []
    counts: list[int] = []
    for value in (-np.asarray(values, dtype=float)).tolist():
        means.append(float(value))
        counts.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            count = counts[-2] + counts[-1]
            means[-2] = (
                means[-2] * counts[-2] + means[-1] * counts[-1]
            ) / count
            counts[-2] = count
            means.pop()
            counts.pop()
    return -np.asarray([
        mean for mean, count in zip(means, counts) for _ in range(count)
    ])


def pchip_monotone(
    x: list[float] | tuple[float, ...],
    y: list[float] | tuple[float, ...],
    xq: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Fritsch-Carlson PCHIP copied without mathematical change from pilot-2."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = _pava_decreasing(np.asarray(y, dtype=float))
    query = np.asarray(xq, dtype=float)
    if len(x_arr) < 2 or len(x_arr) != len(y_arr):
        raise ValueError("PCHIP requires matching x/y arrays with at least two points")
    if not bool(np.all(np.diff(x_arr) > 0)):
        raise ValueError("PCHIP anchors must be strictly increasing")
    h = np.diff(x_arr)
    d = np.diff(y_arr) / h
    n = len(x_arr)
    m = np.zeros(n, dtype=float)
    if n == 2:
        m[:] = d[0]
    else:
        for index in range(1, n - 1):
            if (
                d[index - 1] == 0
                or d[index] == 0
                or np.sign(d[index - 1]) != np.sign(d[index])
            ):
                m[index] = 0.0
            else:
                w1 = 2 * h[index] + h[index - 1]
                w2 = h[index] + 2 * h[index - 1]
                m[index] = (w1 + w2) / (
                    w1 / d[index - 1] + w2 / d[index]
                )

        def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
            value = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            if np.sign(value) != np.sign(d0):
                return 0.0
            if np.sign(d0) != np.sign(d1) and abs(value) > abs(3 * d0):
                return 3 * d0
            return float(value)

        m[0] = endpoint(h[0], h[1], d[0], d[1])
        m[-1] = endpoint(h[-1], h[-2], d[-1], d[-2])
    out = np.empty_like(query)
    for out_index, value in enumerate(query):
        index = min(max(int(np.searchsorted(x_arr, value) - 1), 0), n - 2)
        t = (value - x_arr[index]) / h[index]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        out[out_index] = (
            h00 * y_arr[index]
            + h10 * h[index] * m[index]
            + h01 * y_arr[index + 1]
            + h11 * h[index] * m[index + 1]
        )
    return out


def _parse_rungs(raw: str, *, name: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(raw).split(","):
        token = item.strip().upper()
        if not token:
            continue
        if "_K" in token:
            token = token.rsplit("_K", 1)[1]
        elif token.startswith("K"):
            token = token[1:]
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"{name} contains an invalid rung {item!r}") from exc
    return tuple(values)


@dataclass(frozen=True)
class MinChainInterpolationConfig:
    """Acceptance-amendment-v2 interpolation and audit policy."""

    anchors: tuple[int, ...]
    holdbacks: tuple[int, ...]
    audit_seed: int = 42
    backstop_tolerance: float = 0.25
    audit_median_tolerance: float = 0.05
    audit_p95_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if len(self.anchors) != 5 or tuple(sorted(set(self.anchors))) != self.anchors:
            raise ValueError("min-chain interpolation requires five distinct ascending anchors")
        if len(self.holdbacks) != 2 or not set(self.holdbacks).issubset(self.anchors):
            raise ValueError("min-chain interpolation requires two holdbacks drawn from anchors")
        if any(value < 0.0 for value in (
            self.backstop_tolerance,
            self.audit_median_tolerance,
            self.audit_p95_tolerance,
        )):
            raise ValueError("min-chain interpolation tolerances must be non-negative")

    @classmethod
    def from_rungs(
        cls,
        rungs: list[int] | tuple[int, ...],
        environ: Mapping[str, str] | None = None,
    ) -> "MinChainInterpolationConfig":
        values = os.environ if environ is None else environ
        ordered = tuple(sorted(set(int(rung) for rung in rungs)))
        if len(ordered) < 6:
            raise ValueError("min-chain interpolation needs five anchors plus an audit rung")
        raw_anchors = str(values.get(MINCHAIN_ANCHORS_ENV, "")).strip()
        if raw_anchors:
            anchors = _parse_rungs(raw_anchors, name=MINCHAIN_ANCHORS_ENV)
        else:
            last = len(ordered) - 1
            anchors = tuple(ordered[round(last * fraction / 4)] for fraction in range(5))
        raw_holdbacks = str(values.get(MINCHAIN_HOLDBACKS_ENV, "")).strip()
        holdbacks = (
            _parse_rungs(raw_holdbacks, name=MINCHAIN_HOLDBACKS_ENV)
            if raw_holdbacks else (anchors[1], anchors[3])
        )
        missing = sorted(set(anchors) - set(ordered))
        if missing:
            raise ValueError(f"configured min-chain anchors are absent from menu: {missing}")
        return cls(
            anchors=anchors,
            holdbacks=holdbacks,
            audit_seed=int(values.get(MINCHAIN_AUDIT_SEED_ENV, 42)),
            backstop_tolerance=float(values.get(MINCHAIN_BACKSTOP_ENV, 0.25)),
            audit_median_tolerance=float(values.get(MINCHAIN_AUDIT_MEDIAN_ENV, 0.05)),
            audit_p95_tolerance=float(values.get(MINCHAIN_AUDIT_P95_ENV, 0.15)),
        )

    def audit_rung(self, layer: int, rungs: list[int] | tuple[int, ...]) -> int:
        candidates = tuple(sorted(set(map(int, rungs)) - set(self.anchors)))
        if not candidates:
            raise ValueError("min-chain audit has no non-anchor rung to draw")
        return random.Random(self.audit_seed + int(layer)).choice(candidates)


def _relative_error(prediction: float, truth: float) -> float:
    return abs(float(prediction) - float(truth)) / max(abs(float(truth)), 1e-30)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def interpolation_acceptance_v2(
    anchor_errors: Mapping[int, list[float] | tuple[float, ...]],
    *,
    config: MinChainInterpolationConfig,
    target_rungs: list[int] | tuple[int, ...],
    audit_rung: int,
    audit_errors: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Apply accept-all + gross-CV backstop and the per-layer audit gate."""
    missing = [rung for rung in config.anchors if rung not in anchor_errors]
    if missing:
        raise ValueError(f"min-chain interpolation is missing anchors {missing}")
    n_slices = len(anchor_errors[config.anchors[0]])
    if any(len(anchor_errors[rung]) != n_slices for rung in config.anchors):
        raise ValueError("min-chain anchor vectors have inconsistent slice counts")
    if len(audit_errors) != n_slices:
        raise ValueError("min-chain audit vector has inconsistent slice count")
    predictions = {int(rung): [None] * n_slices for rung in target_rungs}
    backstop_failed: list[int] = []
    cv_errors: list[dict[str, float]] = []
    for cell in range(n_slices):
        local_cv: dict[str, float] = {}
        for holdback in config.holdbacks:
            fit = tuple(rung for rung in config.anchors if rung != holdback)
            prediction = pchip_monotone(
                fit,
                tuple(float(anchor_errors[rung][cell]) for rung in fit),
                (holdback,),
            )[0]
            local_cv[f"K{holdback}"] = _relative_error(
                float(prediction), float(anchor_errors[holdback][cell])
            )
        if any(value > config.backstop_tolerance for value in local_cv.values()):
            backstop_failed.append(cell)
        cv_errors.append(local_cv)
        values = tuple(float(anchor_errors[rung][cell]) for rung in config.anchors)
        fitted = pchip_monotone(config.anchors, values, tuple(target_rungs))
        for rung, prediction in zip(target_rungs, fitted):
            predictions[int(rung)][cell] = float(prediction)
    audit_prediction = pchip_monotone(
        config.anchors,
        tuple(float(anchor_errors[rung][cell]) for rung in config.anchors),
        (audit_rung,),
    ) if n_slices == 1 else None
    audit_relative: list[float] = []
    for cell in range(n_slices):
        prediction = (
            float(audit_prediction[0])
            if audit_prediction is not None
            else float(pchip_monotone(
                config.anchors,
                tuple(float(anchor_errors[rung][cell]) for rung in config.anchors),
                (audit_rung,),
            )[0])
        )
        audit_relative.append(_relative_error(prediction, float(audit_errors[cell])))
    audit_median = float(np.median(np.asarray(audit_relative, dtype=float)))
    audit_p95 = _percentile(audit_relative, 0.95)
    audit_pass = bool(
        audit_median <= config.audit_median_tolerance
        and audit_p95 <= config.audit_p95_tolerance
    )
    return {
        "predictions": predictions,
        "backstop_failed": backstop_failed,
        "backstop_passed": [cell for cell in range(n_slices) if cell not in set(backstop_failed)],
        "cv_relative_error": cv_errors,
        "backstop_tolerance": config.backstop_tolerance,
        "audit": {
            "rung": int(audit_rung),
            "median": audit_median,
            "p95": audit_p95,
            "relative_error": audit_relative,
            "pass": audit_pass,
        },
        "full_measure_layer": not audit_pass,
    }


def guarantee_accounting(
    *, aborted: bool, monotone_violations: int, zero_tax_violations: int,
) -> dict[str, Any]:
    """Keep an execution abort distinct from a mathematical violation count."""
    if aborted:
        return {
            "status": "ABORT",
            "monotone_violations": None,
            "zero_tax_violations": None,
        }
    return {
        "status": "PASS" if monotone_violations == zero_tax_violations == 0 else "FAIL",
        "monotone_violations": int(monotone_violations),
        "zero_tax_violations": int(zero_tax_violations),
    }


def select_reconstruction_slices(
    *,
    weight: torch.Tensor,
    free_reconstruction: torch.Tensor,
    qname: str,
    rung: int,
    format_name: str,
    content_guard: Mapping[str, Any],
    predecessor_reconstruction: torch.Tensor | None = None,
    predecessor_errors: list[float] | tuple[float, ...] | None = None,
    predecessor_identities: list[Mapping[str, Any]] | None = None,
    rtol: float = 1e-12,
) -> dict[str, Any]:
    """Select free/embed independently for every leading packed slice.

    This is the pilot-2 A-FAST production path: selection is unweighted
    weight-MSE, while the free encoder itself remains imatrix-weighted.  The
    returned reconstruction is the only arm replayed by activation QDQ.
    """
    if weight.shape != free_reconstruction.shape:
        raise ValueError("min-chain free reconstruction shape differs from source")
    if weight.ndim == 2:
        source = weight.unsqueeze(0)
        free = free_reconstruction.unsqueeze(0)
    elif weight.ndim == 3:
        source = weight
        free = free_reconstruction
    else:
        raise ValueError("min-chain packed selection expects a 2-D or 3-D weight")
    free_errors = (
        (source - free).float().square().mean(dim=(-2, -1)).detach().cpu().tolist()
    )
    n_slices = len(free_errors)
    if predecessor_reconstruction is not None:
        pred = (
            predecessor_reconstruction.unsqueeze(0)
            if predecessor_reconstruction.ndim == 2
            else predecessor_reconstruction
        )
        if pred.shape != source.shape:
            raise ValueError("min-chain predecessor reconstruction shape differs")
        if predecessor_errors is None or len(predecessor_errors) != n_slices:
            raise ValueError("min-chain predecessor error vector is incomplete")
        if predecessor_identities is None or len(predecessor_identities) != n_slices:
            raise ValueError("min-chain predecessor identity vector is incomplete")
    selected_errors: list[float] = []
    arms: list[str] = []
    identities: list[dict[str, Any]] = []
    embed_mask: list[bool] = []
    for cell, free_error in enumerate(free_errors):
        if predecessor_errors is None:
            arm, error = "free", float(free_error)
        else:
            arm, error = select_arm(
                {
                    "free": float(free_error),
                    "embed": float(predecessor_errors[cell]),
                },
                rtol=rtol,
            )
        predecessor_digest = None
        if arm == "embed":
            predecessor_digest = str(
                predecessor_identities[cell]["solution_digest"]
            )
        recipe = {
            "qname": str(qname),
            "format": str(format_name),
            "rung": int(rung),
            "slice": int(cell),
            "winning_arm": arm,
            "predecessor_digest": predecessor_digest,
            "content_guard": dict(content_guard),
        }
        identities.append(chain_identity_from_digest(
            winning_arm=arm,
            solution_digest_value=recipe_solution_digest(recipe),
            predecessor_digest=predecessor_digest,
        ))
        selected_errors.append(float(error))
        arms.append(arm)
        embed_mask.append(arm == "embed")
    if predecessor_reconstruction is None:
        selected = free_reconstruction
    else:
        mask = torch.as_tensor(
            embed_mask,
            device=free_reconstruction.device,
            dtype=torch.bool,
        ).view(-1, 1, 1)
        selected_3d = torch.where(mask, pred, free)
        selected = selected_3d[0] if weight.ndim == 2 else selected_3d
    if predecessor_errors is not None and any(
        not epsilon_le(value, predecessor, rtol=rtol)
        for value, predecessor in zip(selected_errors, predecessor_errors)
    ):
        raise AssertionError("min-chain monotonicity construction abort")
    if any(
        not epsilon_le(value, free_error, rtol=rtol)
        for value, free_error in zip(selected_errors, free_errors)
    ):
        raise AssertionError("min-chain zero-tax construction abort")
    return {
        "reconstruction": selected,
        "free_errors": [float(value) for value in free_errors],
        "selected_errors": selected_errors,
        "winning_arm": arms,
        "identities": identities,
    }


__all__ = [
    "MINCHAIN_ANCHORS_ENV",
    "MINCHAIN_AUDIT_MEDIAN_ENV",
    "MINCHAIN_AUDIT_P95_ENV",
    "MINCHAIN_AUDIT_SEED_ENV",
    "MINCHAIN_BACKSTOP_ENV",
    "MINCHAIN_CONTEXT_VERSION",
    "MINCHAIN_FLAG",
    "MINCHAIN_HOLDBACKS_ENV",
    "MINCHAIN_SCHEMA",
    "MinChainInterpolationConfig",
    "chain_identity",
    "chain_identity_from_digest",
    "epsilon_le",
    "embed_predecessor",
    "guarantee_accounting",
    "interpolation_acceptance_v2",
    "minchain_enabled",
    "pchip_monotone",
    "refine_one_entry",
    "recipe_solution_digest",
    "relative_epsilon",
    "require_minchain_enabled",
    "require_pilot_enabled",
    "select_arm",
    "select_reconstruction_slices",
    "solution_digest",
    "validate_chain_identity",
]
