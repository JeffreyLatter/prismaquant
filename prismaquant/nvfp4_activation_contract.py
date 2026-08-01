"""Single-owner NVFP4 W4A4 activation execution contract.

The compressed-tensors tensor ABI already names the calibrated scalar
``<target>.input_global_scale``.  PrismaQuant reuses that exact tensor for
FP4-CB instead of inventing a second spelling.  This module owns everything
that is not expressible by the compressed-tensors scheme itself:

* the versioned execution-contract and scale-policy identities;
* calibrated max-abs -> input-global-scale conversion;
* fused-sibling scale unification;
* the canonical mapping digest stamped into ``quant_config.json``; and
* the serve-faithful activation QDQ oracle used by producer tests/costs.

Old CB artifacts can lack both the contract record and scalar tensors; legacy
native artifacts may carry an unversioned/defaultable scalar.  Both remain
readable by their baseline paths, but neither is eligible for Gridbook fused
W4A4 dispatch.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import hashlib
import math
import os
from pathlib import Path
import re
import struct
from typing import Any

import torch


NVFP4_ACTIVATION_CONTRACT_KEY = "nvfp4_w4a4"
NVFP4_ACTIVATION_CONTRACT_SCHEMA = (
    "prismaquant.nvfp4_w4a4_activation.v1"
)
NVFP4_ACTIVATION_EXECUTION = "e2m1_group16_ue4m3_static"
NVFP4_INPUT_GLOBAL_SCALE_SUFFIX = "input_global_scale"

# Public numerical/compatibility constants.  Exporters and loaders must import
# these rather than grow a second activation-scale convention.  In particular,
# the uncalibrated value is a legacy compressed-tensors compatibility fallback;
# a versioned Gridbook activation contract never uses it implicitly.
UNCALIBRATED_INPUT_GLOBAL_SCALE = 1.0
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
FP4_GROUP_SIZE = 16

LEGACY_INPUT_GLOBAL_SCALE_POLICY = (
    "legacy_6_over_calibration_amax.v1"
)
FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY = (
    "full_e4m3_range_448x6_over_calibration_amax.v1"
)
MSE_GRID_INPUT_GLOBAL_SCALE_POLICY = "mse_grid_calibrated.v1"
NVFP4_INPUT_GLOBAL_SCALE_POLICIES = frozenset({
    LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
})

_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# Compatibility fallback for profiles that cannot expose serving fusion
# metadata.  This catalog lives here because activation calibration, legacy
# native export, and the Gridbook execution contract must never infer different
# sibling units.  New architectures should still declare their groups in the
# model profile/structure spec.
_FUSED_DENSE_PATTERNS = (
    (
        re.compile(
            r"^(?P<pre>.+)\.self_attn\.(?P<sib>q_proj|k_proj|v_proj)$"
        ),
        ("q_proj", "k_proj", "v_proj"),
    ),
    (
        re.compile(r"^(?P<pre>.+)\.mlp\.(?P<sib>gate_proj|up_proj)$"),
        ("gate_proj", "up_proj"),
    ),
    (
        re.compile(
            r"^(?P<pre>.+)\.mlp\.shared_expert\."
            r"(?P<sib>gate_proj|up_proj)$"
        ),
        ("gate_proj", "up_proj"),
    ),
    (
        re.compile(
            r"^(?P<pre>.+)\.linear_attn\."
            r"(?P<sib>in_proj_qkv|in_proj_z)$"
        ),
        ("in_proj_qkv", "in_proj_z"),
    ),
    (
        re.compile(
            r"^(?P<pre>.+)\.linear_attn\."
            r"(?P<sib>in_proj_a|in_proj_b)$"
        ),
        ("in_proj_a", "in_proj_b"),
    ),
)


def resolve_input_global_scale_policy(
    value: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve one explicit, stampable input-global-scale policy.

    ``PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE`` remains a compatibility
    input, but it is resolved once at export startup and the resulting policy
    identity is serialized.  No runtime consumer needs to consult the env.
    """

    aliases = {
        "legacy": LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        "legacy_6_over_calibration_amax": LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        LEGACY_INPUT_GLOBAL_SCALE_POLICY: LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        "full": FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        "full_e4m3": FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        "full_e4m3_range": FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        "full_e4m3_range_448x6_over_calibration_amax": (
            FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
        ),
        FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY: (
            FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
        ),
        "mse": MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        "mse_grid": MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        "mse_grid_calibrated": MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        MSE_GRID_INPUT_GLOBAL_SCALE_POLICY: MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
    }
    if value is None:
        env = os.environ if environ is None else environ
        value = (
            FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
            if str(env.get(
                "PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", "0"
            )).strip() == "1"
            else LEGACY_INPUT_GLOBAL_SCALE_POLICY
        )
    canonical = aliases.get(str(value).strip().lower())
    if canonical is None:
        raise ValueError(
            f"unknown NVFP4 input-global-scale policy {value!r}; expected "
            f"one of {sorted(NVFP4_INPUT_GLOBAL_SCALE_POLICIES)}"
        )
    return canonical


def input_global_scale_from_max_abs(
    max_abs: float,
    *,
    policy: str,
    nonpositive_fallback: float | None = None,
) -> float:
    """Return a finite positive F32-representable calibrated scalar.

    ``nonpositive_fallback`` exists only for legacy compressed-tensors export,
    whose historical all-zero behavior serialized ``1.0``.  It is opt-in so a
    versioned activation contract continues to reject missing/degenerate
    calibration.  Non-finite positive values and NaNs always fail closed.
    """

    canonical = resolve_input_global_scale_policy(policy)
    if canonical == MSE_GRID_INPUT_GLOBAL_SCALE_POLICY:
        raise ValueError(
            "mse_grid_calibrated.v1 requires activation samples; call "
            "select_mse_grid_input_global_scale"
        )
    value = float(max_abs)
    if value <= 0.0 and nonpositive_fallback is not None:
        return float(input_global_scale_tensor(nonpositive_fallback).item())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"NVFP4 activation calibration max_abs must be finite and > 0, "
            f"got {max_abs!r}"
        )
    numerator = FP4_E2M1_MAX
    if canonical == FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY:
        numerator *= FP8_E4M3_MAX
    # The artifact tensor is F32.  Digest and export the rounded value, not a
    # Python-f64 value that the loader can never observe.
    result = struct.unpack("<f", struct.pack("<f", numerator / value))[0]
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"NVFP4 input_global_scale is not finite/positive after F32 "
            f"rounding: max_abs={value}, policy={canonical}, value={result}"
        )
    return result


def input_global_scale_tensor(value: float) -> torch.Tensor:
    """Canonical compressed-tensors scalar representation: F32 shape ``[1]``."""

    rounded = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    if not math.isfinite(rounded) or rounded <= 0.0:
        raise ValueError(
            f"input_global_scale must be finite and > 0, got {value!r}"
        )
    return torch.tensor([rounded], dtype=torch.float32)


def resolve_input_global_scale_value(
    override: float | None = None,
    *,
    target: str | None = None,
    calibrated_scales: Mapping[str, float] | None = None,
    allow_uncalibrated_fallback: bool = False,
) -> float:
    """Resolve explicit override -> calibrated mapping -> legacy fallback.

    The fallback is deliberately disabled by default.  Calling with
    ``allow_uncalibrated_fallback=True`` preserves old native-export bytes but
    also means the result cannot attest the versioned fused-W4A4 contract.
    """

    value = override
    if value is None and target is not None and calibrated_scales:
        value = calibrated_scales.get(str(target))
    if value is None:
        if not allow_uncalibrated_fallback:
            raise ValueError(
                "NVFP4 activation contract has no calibrated "
                f"{NVFP4_INPUT_GLOBAL_SCALE_SUFFIX}"
                + (f" for {target!r}" if target is not None else "")
            )
        value = UNCALIBRATED_INPUT_GLOBAL_SCALE
    # Preserve legacy override/map semantics.  Artifact construction performs
    # the existing F32 cast; strict contract paths use input_global_scale_tensor
    # when they build and digest their physical mapping.
    return float(value)


def fused_dense_group(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the legacy fallback group prefix and sibling leaf names."""

    for pattern, members in _FUSED_DENSE_PATTERNS:
        match = pattern.match(str(name))
        if match:
            return match.group("pre"), members
    return None


def fused_sibling_group_key(
    name: str,
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
) -> str | None:
    """Resolve one canonical fused-sibling key.

    Versioned contract callers use the strict default: a profile that cannot
    attest its execution unit is an export error.  The legacy native exporter
    passes ``tolerate_profile_errors=True`` to preserve its historical
    profile-metadata fallback behavior without owning a second algorithm.
    """

    target = str(name)
    group_fn = getattr(profile, "fused_sibling_group", None)
    if callable(group_fn):
        try:
            group = group_fn(target)
        except Exception:
            if not tolerate_profile_errors:
                raise
            group = None
        if group:
            return str(group)

    mapping_fn = getattr(profile, "fused_sibling_leaf_mapping", None)
    if callable(mapping_fn) and "." in target:
        try:
            mapping = mapping_fn()
        except Exception:
            if not tolerate_profile_errors:
                raise
            mapping = None
        if mapping:
            prefix, leaf = target.rsplit(".", 1)
            for fused, members in mapping.items():
                if leaf in {str(member) for member in members}:
                    return f"{prefix}.{fused}"

    fallback = fused_dense_group(target)
    if fallback is None:
        return None
    prefix, members = fallback
    return f"{prefix}::__fused__:{','.join(members)}"


def group_fused_sibling_targets(
    targets: Iterable[str],
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Bucket targets by their runtime fused execution unit.

    Unfused targets remain singleton groups so calibration can use this one
    primitive without maintaining a parallel grouping loop.
    """

    groups: dict[str, list[str]] = {}
    for raw_target in targets:
        target = str(raw_target)
        group = fused_sibling_group_key(
            target,
            profile=profile,
            tolerate_profile_errors=tolerate_profile_errors,
        )
        groups.setdefault(str(group or target), []).append(target)
    return {key: tuple(members) for key, members in groups.items()}


def unify_fused_sibling_max_abs(
    max_abs_by_target: Mapping[str, float],
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
) -> dict[str, float]:
    """Use one conservative calibration maximum for every fused sibling.

    vLLM concatenates q/k/v and gate/up and applies one activation scale.  The
    largest max-abs (equivalently the smallest reciprocal scale) is shared.
    """

    result = {str(k): float(v) for k, v in max_abs_by_target.items()}
    groups = group_fused_sibling_targets(
        max_abs_by_target,
        profile=profile,
        tolerate_profile_errors=tolerate_profile_errors,
    )
    for members in groups.values():
        shared = max(float(result[name]) for name in members)
        for name in members:
            result[name] = shared
    return result


def unify_fused_sibling_input_global_scales(
    scales: Mapping[str, float],
    *,
    profile=None,
    tolerate_profile_errors: bool = False,
    diagnostic_prefix: str | None = None,
) -> dict[str, float]:
    """Conservatively join reciprocal scales across fused siblings.

    ``input_global_scale`` is proportional to ``1 / calibration_amax`` under
    every static policy, so the safe fused join is the minimum scale (the
    largest observed activation range).  Only siblings present in ``scales``
    participate; singleton/unfused targets pass through byte-for-byte.
    """

    groups = group_fused_sibling_targets(
        scales,
        profile=profile,
        tolerate_profile_errors=tolerate_profile_errors,
    )
    result = {str(name): float(value) for name, value in scales.items()}
    unified = 0
    max_drift = 0.0
    for members in groups.values():
        if len(members) < 2:
            continue
        values = [float(scales[member]) for member in members]
        shared = min(values)
        max_drift = max(
            max_drift,
            max(abs(shared - value) for value in values),
        )
        for member in members:
            result[member] = shared
        unified += 1
    if diagnostic_prefix and unified:
        print(
            f"{diagnostic_prefix} unified input_global_scale across "
            f"{unified} fused-sibling groups "
            f"(max pre-unify drift: {max_drift:.3e})",
            flush=True,
        )
    return result


def _activation_cache_candidates(targets: Iterable[str]) -> set[str]:
    candidates = {str(target) for target in targets}
    for target in tuple(candidates):
        for suffix in (".gate_up_proj", ".gate_proj", ".up_proj", ".down_proj"):
            if target.endswith(suffix):
                candidates.add(target[: -len(suffix)])
    return candidates


def load_activation_cache_samples(
    cache_dir: str | Path,
    targets: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Load only relevant input samples from the existing probe cache."""

    from prismaquant.measure_quant_cost import ActivationIndex

    root = Path(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"NVFP4 activation cache directory does not exist: {root}"
        )
    candidates = _activation_cache_candidates(targets)
    index = ActivationIndex(root, sorted(candidates))
    values: dict[str, torch.Tensor] = {}
    for name in sorted(candidates):
        if name not in index:
            continue
        tensor = index.load(name)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            raise ValueError(
                f"NVFP4 activation cache entry {name!r} has no input tensor"
            )
        tensor = tensor.detach().to("cpu").float().contiguous()
        value = float(tensor.abs().max().item())
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"NVFP4 activation cache entry {name!r} has invalid max_abs "
                f"{value!r}"
            )
        values[name] = tensor
    return values


def load_activation_cache_max_abs(
    cache_dir: str | Path,
    targets: Iterable[str],
) -> dict[str, float]:
    """Compatibility view of :func:`load_activation_cache_samples`."""

    return {
        name: float(tensor.abs().max().item())
        for name, tensor in load_activation_cache_samples(
            cache_dir,
            targets,
        ).items()
    }


def select_mse_grid_input_global_scale(
    activation_samples: Iterable[torch.Tensor],
    *,
    device: str | torch.device | None = None,
) -> float:
    """Pick a deterministic static G on the serve-QDQ MSE objective.

    The grid spans the legacy ``6/amax`` and full-E4M3 ``448*6/amax``
    endpoints at quarter-octave resolution.  Both endpoints are always
    included exactly.  A fused sibling group is optimized jointly by passing
    all of its cached samples here.  Equal scores select the smaller G (less
    aggressive clipping).
    """

    samples = [
        tensor.detach().reshape(-1, tensor.shape[-1]).float()
        for tensor in activation_samples
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
    ]
    if not samples:
        raise ValueError("MSE-grid NVFP4 calibration has no activation samples")
    if any(tensor.shape[-1] % FP4_GROUP_SIZE for tensor in samples):
        bad = [tuple(tensor.shape) for tensor in samples
               if tensor.shape[-1] % FP4_GROUP_SIZE]
        raise ValueError(
            f"MSE-grid NVFP4 calibration needs width divisible by 16: {bad}"
        )
    max_abs = max(float(tensor.abs().max().item()) for tensor in samples)
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        raise ValueError(
            f"MSE-grid NVFP4 calibration has invalid max_abs {max_abs!r}"
        )
    legacy = FP4_E2M1_MAX / max_abs
    full = FP8_E4M3_MAX * legacy
    factors = [2.0 ** (step / 4.0) for step in range(36)]
    factors.append(FP8_E4M3_MAX)
    candidates = sorted({
        struct.unpack("<f", struct.pack("<f", legacy * factor))[0]
        for factor in factors
        if legacy * factor <= full
    } | {struct.unpack("<f", struct.pack("<f", full))[0]})

    target_device = torch.device(device or "cpu")
    device_samples = [tensor.to(target_device) for tensor in samples]
    best_scale = candidates[0]
    best_error = math.inf
    for candidate in candidates:
        squared_error = 0.0
        count = 0
        for sample in device_samples:
            qdq = nvfp4_activation_qdq_served(sample, candidate).float()
            squared_error += float(
                (qdq - sample).square().sum(dtype=torch.float64).item()
            )
            count += int(sample.numel())
        error = squared_error / max(count, 1)
        if error < best_error:
            best_error = error
            best_scale = candidate
    return float(best_scale)


def calibrated_input_global_scales(
    targets: Iterable[str],
    *,
    activation_cache_dir: str | Path,
    policy: str,
    profile=None,
    supplemental_max_abs: Mapping[str, float] | None = None,
    supplemental_activations: Mapping[str, Any] | None = None,
    calibration_device: str | torch.device | None = None,
) -> dict[str, float]:
    """Resolve complete target coverage and return fused-coherent scalars.

    Packed gate/up targets consume the experts-module input and therefore may
    use that parent cache entry.  Packed down targets require their routed
    intermediate max-abs in ``supplemental_max_abs``; exporters synthesize it
    with the same checkpoint replay used by the imatrix harvester.
    """

    requested = tuple(sorted({str(target) for target in targets}))
    supplemental_samples: dict[str, torch.Tensor] = {}
    for name, raw_sample in (supplemental_activations or {}).items():
        sample = raw_sample
        if not isinstance(sample, torch.Tensor):
            validate = getattr(sample, "validate", None)
            if not callable(validate):
                raise TypeError(
                    f"supplemental activation {name!r} is neither a tensor "
                    "nor a validated routed sample"
                )
            validate()
            sample = getattr(sample, "values", None)
        if not isinstance(sample, torch.Tensor) or sample.numel() == 0:
            raise ValueError(
                f"supplemental activation {name!r} has no value-bearing rows"
            )
        supplemental_samples[str(name)] = (
            sample.detach().to("cpu").float().contiguous()
        )
    supplemental = {
        str(name): float(value)
        for name, value in (supplemental_max_abs or {}).items()
    }
    canonical_policy = resolve_input_global_scale_policy(policy)
    groups = group_fused_sibling_targets(requested, profile=profile)
    result: dict[str, float] = {}
    for members in groups.values():
        # Cache rows can be very large (tens of GB on 27B+ models).  Load and
        # fit one execution/fusion unit at a time; no policy needs samples from
        # unrelated modules.  This also makes the q/k/v and gate/up union an
        # explicit boundary rather than an accidental whole-model reduction.
        cached = load_activation_cache_samples(
            activation_cache_dir,
            members,
        )
        resolved_samples: dict[str, torch.Tensor] = {}
        resolved_max_abs: dict[str, float] = {}
        for target in members:
            sample = supplemental_samples.get(target, cached.get(target))
            value = supplemental.get(target)
            if sample is None and value is None and target.endswith((
                ".gate_up_proj", ".gate_proj", ".up_proj"
            )):
                parent = target.rsplit(".", 1)[0]
                sample = cached.get(parent)
            if sample is not None:
                value = float(sample.abs().max().item())
                resolved_samples[target] = sample
            if value is None:
                raise ValueError(
                    f"NVFP4 activation contract has no calibrated input for "
                    f"{target!r}; production export refuses an incomplete "
                    "scale mapping"
                )
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"NVFP4 activation contract has invalid max_abs for "
                    f"{target!r}: {value!r}"
                )
            resolved_max_abs[target] = float(value)
        if canonical_policy == MSE_GRID_INPUT_GLOBAL_SCALE_POLICY:
            missing_samples = [
                target for target in members if target not in resolved_samples
            ]
            if missing_samples:
                raise ValueError(
                    "MSE-grid NVFP4 activation calibration needs value-bearing "
                    f"samples for every fused target, missing {missing_samples}"
                )
            shared_scale = select_mse_grid_input_global_scale(
                (resolved_samples[target] for target in members),
                device=calibration_device,
            )
        else:
            shared_max_abs = max(resolved_max_abs[target] for target in members)
            shared_scale = input_global_scale_from_max_abs(
                shared_max_abs,
                policy=canonical_policy,
            )
        for target in members:
            result[target] = shared_scale
    return result


def target_values_sha256(
    scales: Mapping[str, float],
    *,
    policy: str,
) -> str:
    """Digest exact physical target names and their serialized F32 values."""

    canonical_policy = resolve_input_global_scale_policy(policy)
    digest = hashlib.sha256()
    for field in (NVFP4_ACTIVATION_CONTRACT_SCHEMA, canonical_policy):
        encoded = field.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    for target in sorted(scales):
        encoded = str(target).encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<f", float(scales[target])))
    return digest.hexdigest()


def build_execution_contract(
    scales: Mapping[str, float],
    *,
    policy: str,
    target_name: Callable[[str], str] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return the top-level record plus scales keyed by physical target name."""

    mapper = target_name or (lambda name: name)
    physical: dict[str, float] = {}
    for logical_name, raw_value in scales.items():
        name = str(mapper(str(logical_name)))
        if not name:
            raise ValueError(
                f"NVFP4 logical target {logical_name!r} maps to an empty "
                "physical prefix"
            )
        value = float(input_global_scale_tensor(raw_value).item())
        if name in physical:
            raise ValueError(
                f"multiple NVFP4 logical targets map to physical prefix "
                f"{name!r}; the activation namespace must be one-to-one"
            )
        physical[name] = value
    if not physical:
        raise ValueError("NVFP4 execution contract requires at least one target")
    canonical_policy = resolve_input_global_scale_policy(policy)
    record = {
        "schema": NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        "contract": NVFP4_ACTIVATION_EXECUTION,
        "group_size": FP4_GROUP_SIZE,
        "tensor_suffix": NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
        "value_dtype": "float32",
        "input_global_scale_policy": canonical_policy,
        "target_count": len(physical),
        # Config-group targets are not a sufficient namespace oracle: stock
        # compressed-tensors groups use regex targets, while nested models may
        # use a logical module name that differs from the checkpoint prefix.
        # Publish the exact physical prefixes hashed below and used for the
        # serialized ``<target>.input_global_scale`` tensors.
        "target_names": sorted(physical),
        "target_values_sha256": target_values_sha256(
            physical,
            policy=canonical_policy,
        ),
    }
    return record, physical


def nvfp4_activation_qdq_served(
    x: torch.Tensor,
    input_global_scale: float,
) -> torch.Tensor:
    """Serve-faithful static-G NVFP4 activation QDQ oracle.

    vLLM stores each 16-value block scale as UE4M3 and converts activation
    values to E2M1 with round-to-nearest, ties-to-even over the encoded positive
    index.  There is no minimum-scale clamp: with G=1, exactly
    ``6 * 2**-10`` ties the E4M3 scale to byte zero; a value just above it
    becomes byte one and the block is nonzero.

    Installed kernels use ``rcp.approx.ftz.f32`` in ``outputScale``.  Therefore
    arbitrary random Torch results are a numerical oracle, not a packed-byte
    equivalence claim; midpoint and underflow boundary cases are authoritative.
    """

    if x.shape[-1] % FP4_GROUP_SIZE != 0:
        raise ValueError(
            "nvfp4_activation_qdq_served needs last dim divisible by 16, "
            f"got {tuple(x.shape)}"
        )
    g = float(input_global_scale)
    if not math.isfinite(g) or g <= 0.0:
        raise ValueError(
            f"input_global_scale must be finite and > 0, got {g!r}"
        )
    original_shape = x.shape
    original_dtype = x.dtype
    grouped = x.reshape(-1, x.shape[-1] // FP4_GROUP_SIZE,
                        FP4_GROUP_SIZE).float()
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    stored_scale = (amax / FP4_E2M1_MAX * g).clamp(max=FP8_E4M3_MAX)
    stored_scale = stored_scale.to(torch.float8_e4m3fn).float()
    used_scale = stored_scale / g

    nonzero_scale = stored_scale != 0
    safe_scale = torch.where(
        nonzero_scale,
        used_scale,
        torch.ones_like(used_scale),
    )
    normalized = (grouped / safe_scale).clamp(
        -FP4_E2M1_MAX,
        FP4_E2M1_MAX,
    )
    positive = torch.tensor(
        _E2M1_POSITIVE,
        device=normalized.device,
        dtype=torch.float32,
    )
    magnitude = normalized.abs().contiguous()
    upper_index = torch.bucketize(magnitude, positive).clamp_max(
        positive.numel() - 1
    )
    lower_index = (upper_index - 1).clamp_min(0)
    lower = positive[lower_index]
    upper = positive[upper_index]
    lower_distance = (magnitude - lower).abs()
    upper_distance = (upper - magnitude).abs()
    tie = upper_distance == lower_distance
    # Positive E2M1 encodings are indices 0..7.  On an exact midpoint RNE
    # selects the candidate whose encoded index has an even least-significant
    # bit, rather than always selecting the lower magnitude.
    choose_upper = (upper_distance < lower_distance) | (
        tie & ((upper_index & 1) == 0)
    )
    rounded = torch.where(choose_upper, upper, lower).copysign(normalized)
    output = rounded * used_scale
    output = torch.where(nonzero_scale, output, torch.zeros_like(output))
    return output.reshape(original_shape).to(original_dtype)


__all__ = [
    "FP4_E2M1_MAX",
    "FP4_GROUP_SIZE",
    "FP8_E4M3_MAX",
    "FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY",
    "LEGACY_INPUT_GLOBAL_SCALE_POLICY",
    "MSE_GRID_INPUT_GLOBAL_SCALE_POLICY",
    "NVFP4_ACTIVATION_CONTRACT_KEY",
    "NVFP4_ACTIVATION_CONTRACT_SCHEMA",
    "NVFP4_ACTIVATION_EXECUTION",
    "NVFP4_INPUT_GLOBAL_SCALE_POLICIES",
    "NVFP4_INPUT_GLOBAL_SCALE_SUFFIX",
    "UNCALIBRATED_INPUT_GLOBAL_SCALE",
    "build_execution_contract",
    "calibrated_input_global_scales",
    "fused_dense_group",
    "fused_sibling_group_key",
    "group_fused_sibling_targets",
    "input_global_scale_from_max_abs",
    "input_global_scale_tensor",
    "load_activation_cache_max_abs",
    "load_activation_cache_samples",
    "nvfp4_activation_qdq_served",
    "resolve_input_global_scale_policy",
    "resolve_input_global_scale_value",
    "select_mse_grid_input_global_scale",
    "target_values_sha256",
    "unify_fused_sibling_max_abs",
    "unify_fused_sibling_input_global_scales",
]
