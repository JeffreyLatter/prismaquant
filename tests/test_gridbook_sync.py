"""Drift gate for the two gridbook trees.

``plugins/gridbook/`` is the development tree; ``RobTand/gridbook`` is the
release project that publishes it to PyPI. They drifted badly once (37 files,
an entire HIP kernel lane and the fill guard missing on the release side) with
nothing to notice. ``scripts/sync_gridbook.py --check`` is the mechanical
answer; this test is what makes it run.

The gate compares the release repo against **committed** prismaquant content,
so in-flight kernel authoring in the working tree never trips it — only a
commit that lands in ``plugins/gridbook/`` without a matching sync does.

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
