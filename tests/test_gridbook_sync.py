"""Drift gate for the two gridbook trees.

``plugins/gridbook/`` is the development tree; ``RobTand/gridbook`` is the
release project that publishes it to PyPI. They drifted badly once (37 files,
an entire HIP kernel lane and the fill guard missing on the release side) with
nothing to notice. ``scripts/sync_gridbook.py --check`` is the mechanical
answer; this test is what makes it run.

The gate compares the release repo against **committed** prismaquant content,
so in-flight kernel authoring in the working tree never trips it — only a
commit that lands in ``plugins/gridbook/`` without a matching sync does.

It also honours ``sync_gridbook.HELD_PATHS`` (the AMD/HIP lane, held until it
has a serving metric), because the gate and the sync must agree on what the
release project is *supposed* to contain: a held path living only in
prismaquant is the intended steady state, not drift.

The release repo is a local clone, not a fixed part of this checkout, so its
location comes from ``GRIDBOOK_REPO`` (default ``/home/rob/gridbook``) and the
test SKIPS when it is absent. A skip here means "cannot check", never "clean".
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNC_SCRIPT = os.path.join(REPO_ROOT, "scripts", "sync_gridbook.py")
GRIDBOOK_REPO = os.environ.get("GRIDBOOK_REPO", "/home/rob/gridbook")


def test_gridbook_release_repo_is_in_sync():
    if not os.path.isdir(GRIDBOOK_REPO):
        pytest.skip(
            f"gridbook release repo not present at {GRIDBOOK_REPO} "
            "(set GRIDBOOK_REPO to point at a clone to enable this gate)")

    proc = subprocess.run(
        [sys.executable, SYNC_SCRIPT, "--check", "--dest", GRIDBOOK_REPO],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = proc.stdout.decode()

    if proc.returncode == 2:
        pytest.skip(f"sync script refused the destination:\n{report}")

    assert proc.returncode == 0, (
        "The gridbook release repo has drifted from plugins/gridbook/.\n"
        "Run `python3 scripts/sync_gridbook.py` and commit in the release "
        f"repo.\n\n{report}")


def _sync_module():
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    try:
        import sync_gridbook
    finally:
        sys.path.pop(0)
    return sync_gridbook


def test_held_paths_are_the_amd_lane_and_are_actually_filtered():
    """Tier-2 release policy: AMD/HIP kernels are held until a serving metric
    exists. Two things are asserted, and the second is the one that matters —
    that the hold is *applied*, not merely declared. The source listing must
    drop paths that ls-tree still reports, or HELD_PATHS is decoration."""
    sync = _sync_module()

    assert {p for p, _ in sync.HELD_PATHS} == {
        "gridbook/csrc_hip", "gridbook/hip_ext.py", "gridbook/linear_hip.py"}
    for path, reason in sync.HELD_PATHS:
        assert "serving metric" in reason, (
            f"{path}'s hold must name the condition that releases it")

    raw = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--",
         "plugins/gridbook/gridbook"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, check=True).stdout.decode().split()
    tracked = {p.split("plugins/gridbook/", 1)[1] for p in raw}
    assert any(sync.held(p) for p in tracked), (
        "no HELD path is tracked at HEAD — this test would pass vacuously; "
        "either the AMD lane was deleted or HELD_PATHS has gone stale")

    synced = sync.source_entries("HEAD")
    leaked = sorted(p for p in synced if sync.held(p))
    assert not leaked, f"HELD paths reached the sync plan: {leaked}"


def test_hip_delegation_in_linear_is_synced_and_degrades_to_none():
    """The hold is on PATHS, not on CONTENT. linear.py keeps its one guarded
    HIP delegation and is synced verbatim; with linear_hip.py held the import
    raises and `_HIP = None`, which is the state every CUDA box was already in.
    Forking linear.py instead would put a content exception in this gate."""
    sync = _sync_module()
    assert sync.held("gridbook/linear.py") is None, (
        "linear.py must stay synced — see the RELEASE POLICY block in "
        "scripts/sync_gridbook.py for why the guard is kept rather than forked")

    source = open(os.path.join(REPO_ROOT, "plugins", "gridbook", "gridbook",
                               "linear.py"), encoding="utf-8").read()
    guard = source.split("from . import linear_hip", 1)
    assert len(guard) == 2, "the guarded HIP delegation vanished from linear.py"
    assert "except Exception" in guard[1].split("\n\n", 1)[0], (
        "the linear_hip import is no longer wrapped in a bare-degradation "
        "except; with the lane held, that except IS the release behaviour")


def test_sync_script_never_targets_distribution_scaffolding():
    """The release repo owns LICENSE / pyproject / docs / CI; the sync must be
    confined to the two package subtrees. Checked against the declared mapping
    rather than by running it, so it holds on a machine with no clone."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    try:
        import sync_gridbook
    finally:
        sys.path.pop(0)

    destinations = {dst for _, dst in sync_gridbook.SUBTREES}
    assert destinations == {"gridbook", "tests"}, (
        "sync_gridbook.SUBTREES grew a destination outside the package tree; "
        f"got {sorted(destinations)}")
    for name in sync_gridbook.DEST_SCAFFOLDING:
        assert not any(name.startswith(f"{dst}/") for dst in destinations)
