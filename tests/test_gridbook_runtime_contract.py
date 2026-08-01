"""Pinned producer/runtime compatibility without a mirrored Gridbook tree."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re

import pytest


REPO = Path(__file__).resolve().parents[1]
PIN = REPO / "scripts" / "lib" / "gridbook_runtime_pin.json"
REQUIRE_CONTRACT = os.environ.get(
    "PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT") == "1"

pytestmark = pytest.mark.skipif(
    not REQUIRE_CONTRACT,
    reason="run by the pinned Gridbook compatibility CI job",
)


def _pin() -> dict:
    value = json.loads(PIN.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", value["commit"])
    return value


def _direct_url() -> dict:
    dist = importlib.metadata.distribution("gridbook")
    matches = [file for file in (dist.files or ())
               if file.name == "direct_url.json"
               and ".dist-info" in str(file.parent)]
    assert len(matches) == 1, (
        "the compatibility job must install Gridbook from the pinned VCS URL")
    path = Path(dist.locate_file(matches[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_contract() -> dict:
    from gridbook.runtime_contract import load_runtime_contract

    return load_runtime_contract()


def test_installed_gridbook_is_the_exact_external_vcs_pin():
    pin = _pin()
    spec = importlib.util.find_spec("gridbook")
    assert spec is not None and spec.origin
    origin = Path(spec.origin).resolve()
    assert not origin.is_relative_to(REPO), (
        f"Gridbook imported from inside PrismaQuant: {origin}")
    assert importlib.metadata.version("gridbook") == pin["version"]

    direct = _direct_url()
    vcs = direct.get("vcs_info") or {}
    assert vcs.get("vcs") == "git"
    assert vcs.get("commit_id") == pin["commit"]
    assert vcs.get("requested_revision") == pin["commit"]


def test_declared_cb_lanes_equal_gridbook_supported_producer_profiles():
    from prismaquant.model_profiles import registry

    declared = {
        cls().name for cls in registry._REGISTERED
        if "nvfp4_cb" in cls().supported_export_lanes()
    }
    supported = set(
        _runtime_contract()["producer_profiles"]["supported_ids"])
    assert declared == supported


def test_cb_rungs_layouts_and_quant_method_fit_the_runtime_contract():
    from prismaquant.cb_layout import (
        CODEWORDS_PER_SUPERBLOCK,
        INDEX_BIT_ORDER,
        INDEX_BYTES_PER_K,
        SCALE_CODING_TWO_TIER,
        SCALE_CODING_V1,
        SUBINDEX_SPLIT,
        SUPERBLOCK,
        VEC_DIM,
        bit_split,
        type_size,
    )
    from prismaquant.format_registry import list_formats

    contract = _runtime_contract()
    assert contract["quant_method"]["canonical"] == "gridbook"
    assert "prismaquant" in contract["quant_method"]["legacy"]
    by_family = {entry["family"]: entry for entry in contract["formats"]}

    names = {spec.name for spec in list_formats()}
    producer_k = {
        int(name.rsplit("K", 1)[1]) for name in names
        if name.startswith("NVFP4_CB_K")
    }
    producer_s = {
        int(name.rsplit("S", 1)[1]) for name in names
        if name.startswith("NVFP4_CB_S")
    }
    producer_fp8 = {
        int(name.rsplit("K", 1)[1]) for name in names
        if name.startswith("FP8_CB_K")
    }
    assert producer_k == set(by_family["NVFP4_CB_K"]["rungs"])
    assert producer_s <= set(by_family["NVFP4_CB_S"]["rungs"])
    assert producer_fp8 == set(by_family["FP8_CB_K"]["rungs"])

    packing = contract["packing"]
    assert packing == {
        "vector_dim": VEC_DIM,
        "superblock_weights": SUPERBLOCK,
        "codewords_per_superblock": CODEWORDS_PER_SUPERBLOCK,
        "index_bytes_per_k": INDEX_BYTES_PER_K,
        "index_bit_order": INDEX_BIT_ORDER,
        "subindex_split": SUBINDEX_SPLIT,
    }
    # Pin the named split rule to the producer implementation, including odd
    # rungs where ceil-first versus floor-first changes every sidecar shape.
    assert bit_split(13, 2) == (7, 6)
    assert bit_split(29, 4) == (8, 7, 7, 7)

    expected_family_fields = {
        "NVFP4_CB_K": ("NVFP4_CB_K{k}", "fp4", "product", 2),
        "NVFP4_CB_S": ("NVFP4_CB_S{k}", "fp4", "signed", 1),
        "FP8_CB_K": ("FP8_CB_K{k}", "fp8", "product", 4),
    }
    for family, (pattern, grid, mode, n_sub) in expected_family_fields.items():
        entry = by_family[family]
        assert (entry["name_pattern"], entry["grid"], entry["mode"],
                entry["n_sub"]) == (pattern, grid, mode, n_sub)

    layout = contract["layout"]
    assert layout["supported"] == [1, 2]
    assert layout["default_when_absent"] == 1
    assert layout["field"] == "layout_version"
    assert layout["scale_coding_field"] == "scale_coding.kind"
    assert layout["scale_coding_default_when_absent"] == SCALE_CODING_V1
    rules = {
        (rule["grid"], rule["layout_version"], rule["scale_coding"]):
            rule["scale_plane_bytes"]
        for rule in layout["type_size_rules"]
    }
    assert rules == {
        ("fp4", 1, SCALE_CODING_V1): 16,
        ("fp4", 2, SCALE_CODING_TWO_TIER): 9,
        ("fp8", 1, SCALE_CODING_V1): 0,
    }
    for family, entry in by_family.items():
        for version in entry["layout_versions"]:
            coding = (SCALE_CODING_TWO_TIER
                      if entry["grid"] == "fp4" and version == 2
                      else SCALE_CODING_V1)
            scale_bytes = rules[(entry["grid"], version, coding)]
            for k in entry["rungs"]:
                assert type_size(k, entry["grid"], coding) == (
                    packing["index_bytes_per_k"] * k + scale_bytes
                ), (family, version, k)

    assert by_family["NVFP4_CB_K"]["layout_versions"] == [1, 2]
    assert by_family["NVFP4_CB_K"]["moe_layout_versions"] == [2]
    assert by_family["NVFP4_CB_S"]["layout_versions"] == [1, 2]
    assert by_family["NVFP4_CB_S"]["moe_layout_versions"] == [2]
    assert by_family["FP8_CB_K"]["layout_versions"] == [1]
    assert by_family["FP8_CB_K"]["moe_layout_versions"] == [1]
