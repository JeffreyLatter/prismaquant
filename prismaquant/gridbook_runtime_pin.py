"""Strict, torch-free reader for PrismaQuant's immutable Gridbook pin.

This module reads producer/consumer compatibility data only.  It never imports
the external Gridbook package: production compatibility continues to cross the
repository boundary through ``gridbook_runtime_pin.json``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json
import re
from typing import Any


GRIDBOOK_RUNTIME_PIN_SCHEMA = "prismaquant.gridbook_runtime_pin.v2"
GRIDBOOK_RUNTIME_REPOSITORY = "https://github.com/RobTand/gridbook.git"
GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION = "0.8.4"
_REQUIRED_MEMBERS = {
    "schema",
    "repository",
    "commit",
    "version",
    "version_is_release",
}
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_PIN_VERSION_RE = re.compile(
    r"[0-9]+(?:[.][0-9]+)*(?:[A-Za-z0-9.+-]*)?"
)
_FINAL_NUMERIC_VERSION_RE = re.compile(r"[0-9]+(?:[.][0-9]+)+")


class GridbookRuntimePinError(ValueError):
    """The tracked Gridbook pin is missing or structurally invalid."""


@dataclass(frozen=True)
class GridbookRuntimePin:
    schema: str
    repository: str
    commit: str
    version: str
    version_is_release: bool


def parse_gridbook_runtime_pin(
    payload: Mapping[str, Any],
    *,
    where: str = "gridbook_runtime_pin.json",
) -> GridbookRuntimePin:
    """Validate the complete v2 pin payload without permissive defaults."""

    if not isinstance(payload, Mapping):
        raise GridbookRuntimePinError(f"{where}: pin must be a JSON object")
    members = set(payload)
    if members != _REQUIRED_MEMBERS:
        raise GridbookRuntimePinError(
            f"{where}: expected exactly {sorted(_REQUIRED_MEMBERS)}, "
            f"got {sorted(members)}"
        )

    schema = payload["schema"]
    if schema != GRIDBOOK_RUNTIME_PIN_SCHEMA:
        raise GridbookRuntimePinError(
            f"{where}: unsupported schema {schema!r}"
        )
    repository = payload["repository"]
    if repository != GRIDBOOK_RUNTIME_REPOSITORY:
        raise GridbookRuntimePinError(
            f"{where}: repository must be {GRIDBOOK_RUNTIME_REPOSITORY!r}"
        )
    commit = payload["commit"]
    if not isinstance(commit, str) or _FULL_COMMIT_RE.fullmatch(commit) is None:
        raise GridbookRuntimePinError(
            f"{where}: commit must be a lowercase full 40-hex SHA"
        )
    version = payload["version"]
    if not isinstance(version, str) or _PIN_VERSION_RE.fullmatch(version) is None:
        raise GridbookRuntimePinError(
            f"{where}: invalid package version {version!r}"
        )
    version_is_release = payload["version_is_release"]
    if not isinstance(version_is_release, bool):
        raise GridbookRuntimePinError(
            f"{where}: version_is_release must be a JSON boolean"
        )
    return GridbookRuntimePin(
        schema=schema,
        repository=repository,
        commit=commit,
        version=version,
        version_is_release=version_is_release,
    )


def _reject_duplicate_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GridbookRuntimePinError(
                f"gridbook_runtime_pin.json: duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


@lru_cache(maxsize=1)
def load_gridbook_runtime_pin() -> GridbookRuntimePin:
    """Read and validate the one packaged Gridbook runtime pin."""

    location = resources.files("prismaquant").joinpath(
        "gridbook_runtime", "gridbook_runtime_pin.json"
    )
    try:
        raw = location.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GridbookRuntimePinError(
            f"{location}: cannot read Gridbook runtime pin: {exc}"
        ) from exc
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_members)
    except GridbookRuntimePinError:
        raise
    except json.JSONDecodeError as exc:
        raise GridbookRuntimePinError(
            f"{location}: malformed JSON: {exc}"
        ) from exc
    return parse_gridbook_runtime_pin(payload, where=str(location))


def _final_numeric_version(version: str) -> tuple[int, ...] | None:
    text = str(version)
    if _FINAL_NUMERIC_VERSION_RE.fullmatch(text) is None:
        return None
    return tuple(int(part) for part in text.split("."))


def supports_routed_moe_per_role_codebook_lut(
    pin: GridbookRuntimePin,
) -> bool:
    """Whether this pin's version carries the routed per-role LUT ABI.

    ``version_is_release`` is intentionally not part of this capability gate.
    An immutable pre-tag release-preparation commit can carry the ABI while
    correctly reporting that no release tag exists yet.  Release status gates
    served-rung credit elsewhere; the exact commit plus compatibility CI binds
    this version claim to Gridbook's packaged runtime contract.
    """

    actual = _final_numeric_version(pin.version)
    minimum = _final_numeric_version(
        GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION
    )
    assert minimum is not None
    return actual is not None and actual >= minimum


__all__ = [
    "GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION",
    "GRIDBOOK_RUNTIME_REPOSITORY",
    "GRIDBOOK_RUNTIME_PIN_SCHEMA",
    "GridbookRuntimePin",
    "GridbookRuntimePinError",
    "load_gridbook_runtime_pin",
    "parse_gridbook_runtime_pin",
    "supports_routed_moe_per_role_codebook_lut",
]
