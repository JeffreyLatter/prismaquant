"""Producer CB layout facts have one torch-free source of truth."""
from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from prismaquant import cb_layout


REPO = Path(__file__).resolve().parents[1]


def test_cb_layout_module_is_torch_free(tmp_path):
    probe = r'''
import builtins
import importlib.util
import pathlib
import sys

original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] == "torch":
        raise AssertionError("cb_layout imported torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("cb_layout_standalone", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.SUPERBLOCK == 256
assert module.CB_FORMAT_NAMES
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe,
         str(REPO / "prismaquant" / "cb_layout.py")],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_registry_parser_and_packer_share_the_layout_source():
    from prismaquant import format_registry
    from prismaquant import layer_config
    from prismaquant import nvfp4_cb_formats

    registry_names = {
        spec.name for spec in format_registry.list_formats()
        if spec.family in {"nvfp4_cb", "fp8_cb"}
    }
    assert registry_names == cb_layout.CB_FORMAT_NAMES
    assert layer_config._NVFP4_CB_FORMAT_NAMES == cb_layout.CB_FORMAT_NAMES
    assert nvfp4_cb_formats.VEC_DIM == cb_layout.VEC_DIM
    assert nvfp4_cb_formats.SUPERBLOCK == cb_layout.SUPERBLOCK
    assert nvfp4_cb_formats.FP4_GROUP == cb_layout.FP4_GROUP
    for family in cb_layout.FAMILIES:
        for k in family.rungs:
            for version in family.layout_versions:
                coding = (cb_layout.SCALE_CODING_TWO_TIER
                          if family.grid == "fp4" and version == 2
                          else cb_layout.SCALE_CODING_V1)
                assert nvfp4_cb_formats.nvfp4_cb_type_size(
                    k, family.grid, coding
                ) == cb_layout.type_size(k, family.grid, coding)


def test_exact_accountant_uses_layout_subtable_shapes():
    from prismaquant.nvfp4_cb_footprint import codebook_subtable_shapes

    for family in cb_layout.FAMILIES:
        for k in family.rungs:
            name = family.name(k)
            assert codebook_subtable_shapes(name) == (
                cb_layout.codebook_subtable_shapes(
                    k, family.mode, family.n_sub
                )
            )


def test_family_and_subtable_rules_are_canonical():
    fp4_product = cb_layout.family_for("fp4", "product")
    fp4_signed = cb_layout.family_for("FP4", "SIGNED")
    fp8_product = cb_layout.family_for("fp8", "product")

    assert fp4_product.n_sub == 2
    assert fp4_signed.n_sub == 1
    assert fp8_product.n_sub == 4
    assert cb_layout.subtable_bit_widths(13, "product", 2) == (7, 6)
    assert cb_layout.subtable_bit_widths(13, "signed", 1) == (5,)
    assert cb_layout.subtable_bit_widths(29, "product", 4) == (8, 7, 7, 7)

    with pytest.raises(ValueError, match="unknown CB grid/mode"):
        cb_layout.family_for("fp4", "full")
    with pytest.raises(ValueError, match="signed CB requires"):
        cb_layout.subtable_bit_widths(13, "signed", 2)


def test_product_menu_and_lattice_generator_derive_from_layout():
    from scripts.gen_nvfp4_cb_lattices import required_lattice_specs

    suffix = ("NVFP4", "FP8_DYNAMIC", "BF16")
    expected_menu = ",".join((*cb_layout.PRODUCT_CB_FORMATS, *suffix))
    assert cb_layout.product_format_menu(*suffix) == expected_menu

    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "print_cb_format_menu.py"),
         *suffix],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected_menu

    required = set(required_lattice_specs())
    for family in cb_layout.FAMILIES:
        for k in family.rungs:
            widths = cb_layout.subtable_bit_widths(
                k, family.mode, family.n_sub
            )
            shapes = cb_layout.codebook_subtable_shapes(
                k, family.mode, family.n_sub
            )
            for width, (_, dimension) in zip(widths, shapes):
                assert (
                    width,
                    family.grid,
                    dimension,
                    family.mode == "signed",
                ) in required

    for script in (
        "run_27b_cb_20gb.sh",
        "run_laguna_s21_prod.sh",
        "run_hy3_prod_joint.sh",
    ):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert "print_cb_format_menu.py" in text
        assert "range(12, 25)" not in text
        assert "range(28, 49)" not in text


def test_production_serving_profile_cb_allowlist_matches_layout():
    from prismaquant.serving_profiles import load_serving_profile

    raw = json.loads((
        REPO / "prismaquant" / "serving_profile_specs" / "nvfp4_cb.json"
    ).read_text(encoding="utf-8"))
    raw_production = next(
        rule for rule in raw["format_rules"]
        if rule["id"] == "nvfp4_cb_container_formats"
    )
    assert raw_production["allow_formats_from"] == [
        "prismaquant.cb_layout:PRODUCT_CB_FORMAT_NAMES"
    ]

    profile = load_serving_profile("nvfp4_cb")
    production = next(
        rule for rule in profile.format_rules
        if rule.id == "nvfp4_cb_container_formats"
    )
    declared_cb = {
        name for name in production.allow_formats
        if cb_layout.parse_format_name(name) is not None
    }
    assert declared_cb == cb_layout.PRODUCT_CB_FORMAT_NAMES

    shape = next(rule for rule in profile.shape_rules
                 if rule.id == "cb_superblock_shape")
    assert set(shape.formats) == cb_layout.CB_FORMAT_NAMES
