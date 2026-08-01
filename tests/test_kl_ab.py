"""The serve fingerprint and the comparator that refuses to cross it (R15).

§7.4's rule was prose: "A/B arms must have identical extension residency; deltas
under ~±20% across differing serving stacks are not evidence". These tests pin
the two halves of the mechanization — a fingerprint that ignores *which
artifact* was served (so a legitimate A/B still compares) but not *what stack
served it*, and a comparator that refuses a cross-stack delta and downgrades it
to an honest range when overridden.
"""
from __future__ import annotations

import json
import pathlib

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from tools.kl_ab import CROSS_STACK_BAND, compare
from tools.kl_ab import main as kl_ab_cli
from tools.serve_fingerprint import (
    EXTENSION_PATTERN,
    collect_manifest,
    elide_argv_paths,
    fingerprint,
    gridbook_runtime_pin,
    manifest_differences,
    resident_extensions,
    self_manifest,
)

BASE_MANIFEST = {
    "image": "vllm-node:latest",
    "gpu_name": "NVIDIA GB10",
    "driver_version": "595.42",
    "enforce_eager": True,
    "quantization": "compressed-tensors",
    "package_versions": {"vllm": "0.21.0", "torch": "2.11.0"},
    "resident_extensions": ["_gridbook_C.so"],
    "launch_flags": ["vllm", "serve", "<path>", "--enforce-eager"],
    # excluded from the fingerprint:
    "created": "2026-07-30T10:00:00",
    "launch_argv": ["vllm", "serve", "/dqruns/a/exported", "--enforce-eager"],
    "model": "/dqruns/a/exported",
    "processes": [{"pid": 1, "cmdline": "vllm serve"}],
}


def _result(value, *, manifest=None, metric="kl_confident_mean", model="/a"):
    payload = {"model": model, metric: value}
    if manifest is not None:
        payload["serve_manifest"] = manifest
        payload["serve_fingerprint"] = fingerprint(manifest)
    return payload


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_argv_paths_are_elided_so_an_ab_shares_a_fingerprint():
    argv = ["vllm", "serve", "/dqruns/arm-a/exported", "--max-model-len", "8192"]
    assert elide_argv_paths(argv) == [
        "vllm", "serve", "<path>", "--max-model-len", "8192"]

    arm_a = dict(BASE_MANIFEST)
    arm_b = dict(BASE_MANIFEST,
                 model="/dqruns/b/exported",
                 launch_argv=["vllm", "serve", "/dqruns/b/exported"],
                 created="2026-07-31T11:00:00",
                 processes=[{"pid": 7, "cmdline": "vllm serve"}])
    assert fingerprint(arm_a) == fingerprint(arm_b), (
        "two artifacts served the same way must share a fingerprint, or every "
        "A/B would refuse itself")


@pytest.mark.parametrize("key, value", [
    ("resident_extensions", []),                       # the ±17% mechanism
    ("enforce_eager", False),
    ("image", "vllm-node:2026-07-01"),
    ("quantization", None),
    ("package_versions", {"vllm": "0.22.0", "torch": "2.11.0"}),
    ("launch_flags", ["vllm", "serve", "<path>"]),
    ("gpu_name", "NVIDIA H100"),
    ("gridbook_runtime_pin", {
        "commit": "f" * 40, "version": "0.4.1",
    }),
])
def test_stack_changes_move_the_fingerprint(key, value):
    changed = dict(BASE_MANIFEST, **{key: value})
    assert fingerprint(changed) != fingerprint(BASE_MANIFEST)
    assert key in manifest_differences(BASE_MANIFEST, changed)


def test_extension_pattern_matches_the_tracked_sos():
    for path in (
        "/gb_snap/gridbook/_gridbook_C.cpython-312-aarch64-linux-gnu.so",
        "/usr/lib/python3/site-packages/flashinfer/_kernels.so",
        "/usr/lib/python3/site-packages/causal_conv1d/_C.so",
        "/usr/lib/python3/site-packages/fla/ops/_triton.so",
        "/repo/prismaquant/kernels/nvfp4_fused.so",
    ):
        assert EXTENSION_PATTERN.search(path), path
    assert not EXTENSION_PATTERN.search("/usr/lib/libcudart.so.13")


def test_external_gridbook_pin_is_recorded_in_the_stack(monkeypatch):
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_VERSION", "0.4.1")
    assert gridbook_runtime_pin() == {
        "commit": "a" * 40,
        "version": "0.4.1",
    }
    manifest = collect_manifest(
        pids=[__import__("os").getpid()], launch_argv=["vllm", "serve", "/m"])
    assert manifest["gridbook_runtime_pin"] == gridbook_runtime_pin()


def test_self_manifest_reads_this_process(tmp_path):
    manifest = self_manifest(image="test-image")
    assert manifest["source"] == "in_process"
    assert manifest["image"] == "test-image"
    assert len(manifest["serve_fingerprint"]) == 64
    assert isinstance(manifest["resident_extensions"], list)
    # recomputing over the written manifest is stable (the fingerprint field
    # itself is excluded from its own input)
    round_tripped = json.loads(json.dumps(manifest))
    assert fingerprint(round_tripped) == manifest["serve_fingerprint"]


def test_unreadable_maps_never_masquerade_as_nothing_resident():
    """Reading a root-owned container process's maps from the host is denied,
    and that denial looks exactly like an empty extension list."""
    assert resident_extensions([999999999]) == []

    blind = collect_manifest(pids=[999999999])
    seeing = collect_manifest(pids=[__import__("os").getpid()])
    assert blind["resident_extensions"] == []
    assert blind["residency_readable"] is False
    assert seeing["residency_readable"] is True
    assert fingerprint(blind) != fingerprint(
        dict(blind, residency_readable=True)), (
        "an unverified scan must not fingerprint as a verified empty one")


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------
def test_same_fingerprint_reports_a_delta():
    a = _result(0.0200, manifest=BASE_MANIFEST)
    b = _result(0.0100, manifest=BASE_MANIFEST)
    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 0
    assert "(matched)" in text
    assert "-50.00%" in text


def test_cross_fingerprint_refuses_without_the_flag():
    a = _result(0.01134, manifest=BASE_MANIFEST)
    b = _result(0.01328,
                manifest=dict(BASE_MANIFEST, resident_extensions=[]))
    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 3
    assert "REFUSED" in text
    assert "resident_extensions" in text
    assert "delta (" not in text, "a refusal must not quote a delta at all"
    assert "-14" not in text and "+17" not in text


def test_cross_fingerprint_downgrades_to_a_range_inside_the_band():
    """The dated 27B case: 0.01134 vs 0.01328 on residency alone."""
    a = _result(0.01134, manifest=BASE_MANIFEST)
    b = _result(0.01328,
                manifest=dict(BASE_MANIFEST, resident_extensions=[]))
    code, lines = compare(a, b, metric="kl_confident_mean",
                          allow_cross_fingerprint=True)
    text = "\n".join(lines)
    assert code == 0
    assert "CROSS-STACK RANGE (not a delta)" in text
    assert f"±{CROSS_STACK_BAND * 100:.0f}%" in text
    assert "NOT EVIDENCE" in text


def test_cross_fingerprint_outside_the_band_is_still_a_range():
    a = _result(0.0400, manifest=BASE_MANIFEST)
    b = _result(0.0100, manifest=dict(BASE_MANIFEST, enforce_eager=False))
    code, lines = compare(a, b, metric="kl_confident_mean",
                          allow_cross_fingerprint=True)
    text = "\n".join(lines)
    assert code == 0
    assert "outside the ±20% band" in text
    assert "NOT EVIDENCE" not in text
    assert "enforce_eager" in text


def test_legacy_json_without_a_fingerprint_compares_with_a_warning():
    a = _result(0.0200)
    b = _result(0.0100)
    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 0
    assert "WARNING: no serve_fingerprint" in text
    assert "-50.00%" in text


def test_spec_decode_taint_is_called_out():
    a = _result(0.02, manifest=BASE_MANIFEST)
    b = dict(_result(0.01, manifest=BASE_MANIFEST), spec_decode_detected=True)
    text = "\n".join(compare(a, b, metric="kl_confident_mean")[1])
    assert "DRAFT model's" in text


def test_differing_git_commit_is_noted_but_not_refused():
    a = dict(_result(0.02, manifest=BASE_MANIFEST), git_commit="a" * 40)
    b = dict(_result(0.01, manifest=BASE_MANIFEST), git_commit="b" * 40)
    code, lines = compare(a, b, metric="kl_confident_mean")
    assert code == 0
    assert any("different git_commit" in line for line in lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_metric_autoselect_and_exit_codes(tmp_path, capsys):
    same = tmp_path / "a.json"
    other = tmp_path / "b.json"
    same.write_text(json.dumps(_result(4.10, manifest=BASE_MANIFEST,
                                       metric="ppl")))
    other.write_text(json.dumps(
        _result(4.05, manifest=dict(BASE_MANIFEST, image="other:latest"),
                metric="ppl")))

    assert kl_ab_cli([str(same), str(other)]) == 3
    assert "REFUSED" in capsys.readouterr().out

    assert kl_ab_cli([str(same), str(other), "--allow-cross-fingerprint"]) == 0
    out = capsys.readouterr().out
    assert "metric: ppl" in out
    assert "CROSS-STACK RANGE" in out
