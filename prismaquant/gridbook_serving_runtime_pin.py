"""Strict reader for the current Gridbook serving-runtime release pin.

The historical ``gridbook_runtime_pin.v3`` remains immutable producer and
handoff evidence for the 0.8.5 render.  Serving is a distinct consumer
boundary: it requires the v4 ABI and an exact reviewed wheel digest.
This module deliberately accepts conspicuous pending sentinels structurally
so a release patch can be reviewed before Gridbook is cut, while every live
serve/ship gate rejects those sentinels.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA = (
    "prismaquant.gridbook_serving_runtime_pin.v1"
)
GRIDBOOK_SERVING_RUNTIME_REPOSITORY = (
    "https://github.com/RobTand/gridbook.git"
)
GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION = "0.8.8"
GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING = (
    "PENDING_GRIDBOOK_V0_8_8_RELEASE_COMMIT"
)
GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING = (
    "PENDING_GRIDBOOK_V0_8_8_WHEEL_SHA256"
)
# Resolved 2026-08-15 against the v0.8.8 release.  Neither value is guessed,
# and neither is transcribed from the other's source:
#   commit  -- annotated tag v0.8.8, which the release workflow refuses to
#              build unless the commit is reachable from origin/master.  It is
#              also the build recorded in the image tag
#              `gridbook:0.8.8-clean-064a4cb`.
#   wheel   -- read out of that image's installed distribution, from the PEP
#              610 `direct_url.json` `archive_info.hashes.sha256` of
#              gridbook-0.8.8-py3-none-any.whl.  This is the digest of the
#              wheel that is actually importable at serve time, which is the
#              only digest a serving pin may assert; a locally rebuilt wheel is
#              a DIFFERENT archive and must not be substituted here without
#              re-reading it from the served image.
#
# As with 0.8.7, this digest IS the PyPI wheel's: the image was built by
# installing the published `gridbook==0.8.8` archive from a local file rather
# than rebuilding it, so `pip download gridbook==0.8.8` satisfies the pin
# instead of tripping the wheel-cache trap documented in
# docs/audits/serving_wheel_cache_poisoning_2026-08-14.md.  Verified before
# use: all 60 members of the PyPI wheel are byte-identical to a local rebuild
# from the tag, so the published archive is this commit's content.
#
# WHY 0.8.8 AND NOT 0.8.7 for the Qwen3.8-27B CB lane: 0.8.7 shipped the
# quantized-embedding method and its dispatch branch, but vLLM dispatches from
# inside the layer constructor and `qwen3_5.py` builds its embedding with
# neither a quant_config nor a prefix -- so neither was ever reached and the
# artifact did not load at all.  0.8.8 supplies those arguments.
GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT = (
    "064a4cb093da10d7c35be03435bb6a525280a45f"
)
GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256 = (
    "a982e8842d0ce183eaad8978941a375dc984fa0697be7c4519dd741efd1153a3"
)
GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v4"
GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES = {
    "routed_moe_per_role_codebook_lut": 1,
    "source_fp8_block128_w8a16": 1,
    "dspark_construction_physical_bridge": 1,
}
_REQUIRED_MEMBERS = {
    "schema",
    "repository",
    "commit",
    "version",
    "version_is_release",
    "wheel_sha256",
    "runtime_contract_schema",
    "required_abi_features",
}
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GridbookServingRuntimePinError(ValueError):
    """The current serving pin is missing, pending, or malformed."""


@dataclass(frozen=True)
class GridbookServingRuntimePin:
    schema: str
    repository: str
    commit: str
    version: str
    version_is_release: bool
    wheel_sha256: str
    runtime_contract_schema: str
    required_abi_features: Mapping[str, int]

    @property
    def commit_is_resolved(self) -> bool:
        return _FULL_COMMIT_RE.fullmatch(self.commit) is not None

    @property
    def wheel_is_resolved(self) -> bool:
        return _SHA256_RE.fullmatch(self.wheel_sha256) is not None


def parse_gridbook_serving_runtime_pin(
    payload: Mapping[str, Any],
    *,
    where: str = "gridbook_serving_runtime_pin.json",
) -> GridbookServingRuntimePin:
    if not isinstance(payload, Mapping) or set(payload) != _REQUIRED_MEMBERS:
        observed = sorted(payload) if isinstance(payload, Mapping) else []
        raise GridbookServingRuntimePinError(
            f"{where}: expected exactly {sorted(_REQUIRED_MEMBERS)}, "
            f"got {observed}"
        )
    if payload["schema"] != GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA:
        raise GridbookServingRuntimePinError(
            f"{where}: unsupported schema {payload['schema']!r}"
        )
    if payload["repository"] != GRIDBOOK_SERVING_RUNTIME_REPOSITORY:
        raise GridbookServingRuntimePinError(
            f"{where}: repository differs from the reviewed Gridbook origin"
        )
    commit = payload["commit"]
    if not isinstance(commit, str) or (
        _FULL_COMMIT_RE.fullmatch(commit) is None
        and commit != GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: commit must be full lowercase SHA or the exact pending sentinel"
        )
    wheel = payload["wheel_sha256"]
    if not isinstance(wheel, str) or (
        _SHA256_RE.fullmatch(wheel) is None
        and wheel != GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: wheel_sha256 must be SHA-256 or the exact pending sentinel"
        )
    if payload["version"] != GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION:
        raise GridbookServingRuntimePinError(
            f"{where}: version must be {GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION!r}"
        )
    released = payload["version_is_release"]
    if not isinstance(released, bool):
        raise GridbookServingRuntimePinError(
            f"{where}: version_is_release must be a JSON boolean"
        )
    if (commit == GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING or
            wheel == GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING) and released:
        raise GridbookServingRuntimePinError(
            f"{where}: pending commit/wheel cannot be marked released"
        )
    if payload["runtime_contract_schema"] != (
        GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: serving runtime contract must be v4"
        )
    features = payload["required_abi_features"]
    if not isinstance(features, Mapping) or set(features) != set(
        GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: ABI feature closure differs"
        )
    normalized: dict[str, int] = {}
    for name, expected in GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES.items():
        value = features[name]
        if type(value) is not int or value != expected:
            raise GridbookServingRuntimePinError(
                f"{where}: required_abi_features.{name} must equal {expected}"
            )
        normalized[name] = value
    return GridbookServingRuntimePin(
        schema=str(payload["schema"]),
        repository=str(payload["repository"]),
        commit=commit,
        version=str(payload["version"]),
        version_is_release=released,
        wheel_sha256=wheel,
        runtime_contract_schema=str(payload["runtime_contract_schema"]),
        required_abi_features=MappingProxyType(normalized),
    )


def _reject_duplicate_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GridbookServingRuntimePinError(
                f"gridbook_serving_runtime_pin.json: duplicate member {key!r}"
            )
        result[key] = value
    return result


@lru_cache(maxsize=1)
def load_gridbook_serving_runtime_pin() -> GridbookServingRuntimePin:
    location = (
        Path(__file__).resolve().parent
        / "gridbook_runtime"
        / "gridbook_serving_runtime_pin.json"
    )
    try:
        payload = json.loads(
            location.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GridbookServingRuntimePinError(
            f"{location}: cannot read serving pin: {exc}"
        ) from exc
    return parse_gridbook_serving_runtime_pin(payload, where=str(location))


def require_exact_gridbook_serving_runtime_release(
    pin: GridbookServingRuntimePin,
) -> None:
    if not pin.commit_is_resolved or not pin.wheel_is_resolved:
        raise GridbookServingRuntimePinError(
            f"Gridbook {GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION} serving "
            "commit/wheel digest is still pending"
        )
    if (
        pin.commit != GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT
        or pin.wheel_sha256 != GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256
        or pin.version != GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION
        or pin.version_is_release is not True
    ):
        raise GridbookServingRuntimePinError(
            "Gridbook serving runtime differs from the exact reviewed "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION} release"
        )


__all__ = [
    "GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES",
    "GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING",
    "GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA",
    "GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA",
    "GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT",
    "GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION",
    "GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256",
    "GRIDBOOK_SERVING_RUNTIME_REPOSITORY",
    "GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING",
    "GridbookServingRuntimePin",
    "GridbookServingRuntimePinError",
    "load_gridbook_serving_runtime_pin",
    "parse_gridbook_serving_runtime_pin",
    "require_exact_gridbook_serving_runtime_release",
]
