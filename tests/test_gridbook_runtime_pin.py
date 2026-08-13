"""Strict producer-side interpretation of the immutable Gridbook pin."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import gridbook_runtime_pin as pinmod
from prismaquant.shipcard import _released_gridbook_runtime_pin


REPO = Path(__file__).resolve().parents[1]
PIN_PATH = (
    REPO / "prismaquant" / "gridbook_runtime" / "gridbook_runtime_pin.json"
)


def _payload(
    version="0.8.5",
    *,
    version_is_release=True,
    features=None,
):
    return {
        "schema": pinmod.GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": version,
        "version_is_release": version_is_release,
        "runtime_contract_schema": pinmod.GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
        "required_abi_features": dict(
            pinmod.GRIDBOOK_REQUIRED_ABI_FEATURES
            if features is None else features
        ),
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


def test_capabilities_are_feature_gated_not_version_inferred():
    pin = pinmod.parse_gridbook_runtime_pin(_payload("99.0.0"))
    assert pinmod.supports_routed_moe_per_role_codebook_lut(pin)
    assert pinmod.supports_source_fp8_block128_w8a16(pin)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("runtime_contract_schema", "gridbook.runtime-contract.v2", "schema"),
        ("required_abi_features", {}, "contain exactly"),
        (
            "required_abi_features",
            {
                "routed_moe_per_role_codebook_lut": 1,
                "source_fp8_block128_w8a16": True,
            },
            "integer 1",
        ),
    ),
)
def test_strict_parser_rejects_runtime_contract_drift(field, value, match):
    payload = _payload()
    payload[field] = value
    with pytest.raises(pinmod.GridbookRuntimePinError, match=match):
        pinmod.parse_gridbook_runtime_pin(payload)


def test_unresolved_release_commit_is_conspicuous_and_fail_closed():
    pinmod.load_gridbook_runtime_pin.cache_clear()
    pin = pinmod.load_gridbook_runtime_pin()
    assert pin.commit == pinmod.GRIDBOOK_RUNTIME_COMMIT_PENDING
    assert pin.version_is_release is False
    with pytest.raises(pinmod.GridbookRuntimePinError, match="unresolved"):
        pinmod.require_resolved_gridbook_runtime_pin(pin)


def test_shipcard_gates_refuse_the_staged_release_pin():
    pinmod.load_gridbook_runtime_pin.cache_clear()
    with pytest.raises(pinmod.GridbookRuntimePinError, match="unresolved"):
        _released_gridbook_runtime_pin()
