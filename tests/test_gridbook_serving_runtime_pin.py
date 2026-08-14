from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from prismaquant.gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING,
    GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING,
    GridbookServingRuntimePinError,
    load_gridbook_serving_runtime_pin,
    parse_gridbook_serving_runtime_pin,
    require_exact_gridbook_serving_runtime_release,
)


def _resolved_payload() -> dict:
    return {
        "schema": "prismaquant.gridbook_serving_runtime_pin.v1",
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": "0.8.6",
        "version_is_release": True,
        "wheel_sha256": "b" * 64,
        "runtime_contract_schema": "gridbook.runtime-contract.v4",
        "required_abi_features": {
            "routed_moe_per_role_codebook_lut": 1,
            "source_fp8_block128_w8a16": 1,
            "dspark_construction_physical_bridge": 1,
        },
    }


def test_packaged_serving_pin_is_resolved_and_loads_in_shell():
    """The packaged pin resolves to v0.8.6, and the shell helper accepts it.

    Until 2026-08-14 this asserted the opposite -- that the packaged pin was
    the PENDING sentinel and that the shell helper exited 2 on it. That was
    the correct assertion while 0.8.6 was untagged. Now that the release
    exists, the same two code paths are asserted from the other side: the pin
    must resolve, and the helper that every serve script sources must export
    the resolved identity rather than refuse. The refusal path is not dropped
    -- it is exercised on a synthetic pending pin in the test below, which is
    where it belongs, since a fixture cannot go stale the way the packaged
    file just did.
    """
    pin = load_gridbook_serving_runtime_pin()
    assert pin.commit != GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
    assert pin.wheel_sha256 != GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    assert pin.commit_is_resolved and pin.wheel_is_resolved
    assert pin.version == "0.8.6"
    assert pin.version_is_release is True
    require_exact_gridbook_serving_runtime_release(pin)

    helper = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_serving_runtime.sh"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin && printf "%s\\t%s\\t%s\\n"'
            ' "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"'
            ' "$GRIDBOOK_RUNTIME_WHEEL_SHA256"',
            "bash",
            str(helper),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    # The shell and Python readers must agree; they are two independent parsers
    # of the same file and a serve script only ever sees the shell one.
    assert result.stdout.strip().split("\t") == [
        pin.commit, pin.version, pin.wheel_sha256,
    ]


def test_shell_helper_still_refuses_a_pending_serving_pin(tmp_path):
    """A pending pin must still fail the shell loader closed, exit 2."""
    helper = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_serving_runtime.sh"
    )
    asset_dir = tmp_path / "gridbook_runtime"
    asset_dir.mkdir()
    shutil.copy(helper, asset_dir / "gridbook_serving_runtime.sh")
    payload = json.loads(
        (helper.parent / "gridbook_serving_runtime_pin.json")
        .read_text(encoding="utf-8"))
    payload["commit"] = GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
    payload["wheel_sha256"] = GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    payload["version_is_release"] = False
    (asset_dir / "gridbook_serving_runtime_pin.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin',
            "bash",
            str(asset_dir / "gridbook_serving_runtime.sh"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "invalid or pending serving pin" in result.stderr


def test_resolved_serving_pin_requires_closed_v4_feature_set():
    pin = parse_gridbook_serving_runtime_pin(_resolved_payload())
    assert pin.commit_is_resolved
    assert pin.wheel_is_resolved
    bad = deepcopy(_resolved_payload())
    bad["required_abi_features"].pop("dspark_construction_physical_bridge")
    with pytest.raises(GridbookServingRuntimePinError, match="closure differs"):
        parse_gridbook_serving_runtime_pin(bad)


def test_serving_helper_accepts_only_the_exact_resolved_wheel(tmp_path):
    wheel = tmp_path / "gridbook-0.8.6-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "gridbook-0.8.6.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gridbook\nVersion: 0.8.6\n",
        )
        archive.writestr("gridbook/__init__.py", "")
    payload = _resolved_payload()
    payload["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    source = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_serving_runtime.sh"
    )
    helper = runtime_dir / source.name
    shutil.copyfile(source, helper)
    (runtime_dir / "gridbook_serving_runtime_pin.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin; '
            'gridbook_serving_runtime_verify_wheel "$2"',
            "bash",
            str(helper),
            str(wheel),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(wheel.resolve())

    payload["wheel_sha256"] = "b" * 64
    (runtime_dir / "gridbook_serving_runtime_pin.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin; '
            'gridbook_serving_runtime_verify_wheel "$2"',
            "bash",
            str(helper),
            str(wheel),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode != 0
    assert "differs from pin" in rejected.stderr


@pytest.mark.parametrize("member", ("commit", "wheel_sha256"))
def test_pending_serving_identity_cannot_claim_release(member):
    payload = _resolved_payload()
    payload[member] = (
        GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
        if member == "commit"
        else GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    )
    with pytest.raises(GridbookServingRuntimePinError, match="cannot be marked"):
        parse_gridbook_serving_runtime_pin(payload)
