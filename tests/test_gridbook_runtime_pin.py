"""Strict producer-side interpretation of the immutable Gridbook pin."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import gridbook_runtime_pin as pinmod


REPO = Path(__file__).resolve().parents[1]
PIN_PATH = (
    REPO / "prismaquant" / "gridbook_runtime" / "gridbook_runtime_pin.json"
)


def _payload(version="0.8.2", *, version_is_release=True):
    return {
        "schema": pinmod.GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": version,
        "version_is_release": version_is_release,
    }


def test_strict_reader_matches_the_tracked_pin():
    pinmod.load_gridbook_runtime_pin.cache_clear()
    parsed = pinmod.load_gridbook_runtime_pin()
    tracked = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert parsed == pinmod.GridbookRuntimePin(**tracked)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("schema", "unknown", "unsupported schema"),
        ("repository", "file:///gridbook.git", "github.com/RobTand/gridbook"),
        ("commit", "ABC", "lowercase full 40-hex"),
        ("version", "not-a-version", "invalid package version"),
        ("version_is_release", 1, "JSON boolean"),
    ),
)
def test_strict_parser_rejects_malformed_pin_members(field, value, match):
    payload = _payload()
    payload[field] = value
    with pytest.raises(pinmod.GridbookRuntimePinError, match=match):
        pinmod.parse_gridbook_runtime_pin(payload)


def test_strict_parser_rejects_missing_and_unknown_members():
    missing = _payload()
    missing.pop("commit")
    with pytest.raises(pinmod.GridbookRuntimePinError, match="expected exactly"):
        pinmod.parse_gridbook_runtime_pin(missing)
    extra = {**_payload(), "abi_guess": True}
    with pytest.raises(pinmod.GridbookRuntimePinError, match="expected exactly"):
        pinmod.parse_gridbook_runtime_pin(extra)


@pytest.mark.parametrize(
    "version,version_is_release,expected",
    (
        ("0.8.2", True, False),
        ("0.8.3", False, True),
        ("0.8.3", True, True),
        ("0.8.4", True, True),
        ("0.9.0", True, True),
        ("0.8.3rc1", False, False),
        ("0.8.3+local", False, False),
    ),
)
def test_routed_moe_per_role_lut_gate_is_final_numeric_versioned(
    version, version_is_release, expected
):
    pin = pinmod.parse_gridbook_runtime_pin(
        _payload(version, version_is_release=version_is_release)
    )
    assert pinmod.supports_routed_moe_per_role_codebook_lut(pin) is expected
