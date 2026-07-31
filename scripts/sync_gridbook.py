#!/usr/bin/env python3
"""One-way sync of the gridbook PACKAGE from prismaquant to the release repo.

TWO TREES, ONE PACKAGE
======================
``plugins/gridbook/`` in this repo is the **development tree** — where the
kernel campaign lands, alongside the encoder/allocator work that produces the
artifacts those kernels serve.  ``RobTand/gridbook`` (locally ``/home/rob/
gridbook``, overridable with ``GRIDBOOK_REPO``) is the **release project** — the
public GitHub repo and the PyPI ``gridbook`` distribution.  The release repo
carries distribution scaffolding the in-tree copy has no business owning
(``LICENSE``, ``pyproject.toml``, ``MANIFEST.in``, ``Dockerfile``, ``docs/``,
``scripts/``, ``.github/``, ``CITATION.cff``, ``CONTRIBUTING.md``,
``ROADMAP.md``, ``README.md``), and this script never touches any of it.

The only thing that flows, and it flows one way:

    plugins/gridbook/gridbook/  ->  <release>/gridbook/
    plugins/gridbook/tests/     ->  <release>/tests/

WHY GIT, NOT THE WORKING TREE
=============================
The source is read from a **committed revision** (``HEAD`` by default), never
from the working tree.  Robert's rule for this path is "all kernels should be
part of gridbook project *when they're ready*", and a commit is the only
mechanical definition of ready this repo already has.  In-flight kernel
authoring — untracked ``.cu``/``.hpp`` files, unstaged edits to ``codec.py`` —
is therefore excluded automatically rather than by a hand-maintained denylist
that would go stale the moment the next kernel lands.  The summary prints what
it skipped for that reason so the exclusion is visible, not silent.

MIRROR SEMANTICS
================
Within the two synced subtrees this is a mirror, deletions included: a file the
dev tree deleted (``tests/bench_r2_terms.py``, dropped in ``fd1e524``) is stale
in the release repo, and leaving it behind is exactly the rot this script
exists to stop.  Deletions are listed individually in the summary before
anything is written.  Nothing outside the two destination subtrees is read,
written, or considered — asserted per path, not just intended.

USAGE
=====
    python3 scripts/sync_gridbook.py --check     # drift report, exit 1 on drift
    python3 scripts/sync_gridbook.py             # show plan, then apply
    python3 scripts/sync_gridbook.py --rev v0.2  # sync some other revision

``--check`` is wired to ``tests/test_gridbook_sync.py`` so the trees cannot
silently diverge again.  Exit codes: 0 in sync / applied, 1 drift (``--check``),
2 refused (bad destination, unexpected layout).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# repo-relative source subtree -> destination-repo-relative subtree.
SUBTREES: tuple[tuple[str, str], ...] = (
    ("plugins/gridbook/gridbook", "gridbook"),
    ("plugins/gridbook/tests", "tests"),
)

DEFAULT_DEST = "/home/rob/gridbook"

# Build/test detritus. Never tracked in git, so these only matter when scanning
# the destination for files to delete — an untracked __pycache__ over there is
# not drift.
IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache",
                          ".ruff_cache", "build", "dist"})
IGNORED_SUFFIXES = (".pyc", ".pyo", ".egg-info")

# Distribution scaffolding the release repo owns. Listed so the guard below can
# prove we are pointed at the release repo and not, say, at a stale checkout of
# the plugin, and so the "never touched" claim is checkable rather than implied.
DEST_SCAFFOLDING = ("LICENSE", "pyproject.toml", "MANIFEST.in", "README.md")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Refused(Exception):
    """Destination or layout is not what this script is allowed to write to."""


def _git(args: list[str], cwd: str, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=cwd, check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out.stdout if binary else out.stdout.decode()


def _ignored(rel_path: str) -> bool:
    parts = rel_path.split("/")
    if any(p in IGNORED_DIRS for p in parts):
        return True
    return any(p.endswith(IGNORED_SUFFIXES) for p in parts)


def source_entries(rev: str) -> dict[str, tuple[str, str]]:
    """``{dest_rel_path: (git_mode, blob_sha)}`` for the committed revision."""
    entries: dict[str, tuple[str, str]] = {}
    for src_sub, dst_sub in SUBTREES:
        listing = _git(["ls-tree", "-r", "-z", rev, "--", src_sub], REPO_ROOT)
        for record in listing.split("\0"):
            if not record:
                continue
            meta, path = record.split("\t", 1)
            mode, obj_type, sha = meta.split()
            if obj_type != "blob":          # submodules/symlinks are not ours
                continue
            rel = os.path.relpath(path, src_sub)
            if _ignored(rel):
                continue
            entries[f"{dst_sub}/{rel}"] = (mode, sha)
    return entries


def dest_entries(dest_root: str) -> dict[str, str]:
    """``{dest_rel_path: absolute_path}`` for files currently in the mirrored
    subtrees. Only these paths are ever eligible for deletion."""
    found: dict[str, str] = {}
    for _, dst_sub in SUBTREES:
        base = os.path.join(dest_root, dst_sub)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for name in filenames:
                abs_path = os.path.join(dirpath, name)
                rel = os.path.relpath(abs_path, dest_root)
                if _ignored(rel):
                    continue
                found[rel] = abs_path
    return found


def guard_destination(dest_root: str) -> None:
    """Refuse anything that is not recognisably the gridbook release repo, or
    whose tests are laid out differently from ours. Flattening someone else's
    structure is worse than not syncing."""
    if not os.path.isdir(dest_root):
        raise Refused(f"destination does not exist: {dest_root}")
    if not os.path.isdir(os.path.join(dest_root, ".git")):
        raise Refused(f"destination is not a git repo: {dest_root}")
    for name in DEST_SCAFFOLDING:
        if not os.path.exists(os.path.join(dest_root, name)):
            raise Refused(
                f"{dest_root} is missing {name} — this does not look like the "
                "gridbook release repo; refusing to write into it")
    pyproject = open(os.path.join(dest_root, "pyproject.toml")).read()
    if 'name = "gridbook"' not in pyproject:
        raise Refused(f"{dest_root}/pyproject.toml does not declare the "
                      "gridbook distribution; refusing")
    for _, dst_sub in SUBTREES:
        sub = os.path.join(dest_root, dst_sub)
        if not os.path.isdir(sub):
            raise Refused(
                f"{sub} does not exist. This script mirrors an existing layout; "
                "it will not invent one. Check the release repo by hand.")
    tests_dir = os.path.join(dest_root, "tests")
    flat = [n for n in os.listdir(tests_dir) if n.startswith("test_")
            and n.endswith(".py")]
    if not flat:
        raise Refused(
            f"{tests_dir} holds no top-level test_*.py — its layout differs "
            "from plugins/gridbook/tests/ (flat). Refusing to flatten it.")


def _within(path: str, root: str) -> bool:
    return os.path.commonpath([os.path.abspath(path),
                               os.path.abspath(root)]) == os.path.abspath(root)


def build_plan(rev: str, dest_root: str):
    """(adds, updates, deletes) as lists of dest-relative paths, plus the blob
    payloads needed to write them."""
    src = source_entries(rev)
    dst = dest_entries(dest_root)

    adds, updates, deletes = [], [], []
    payload: dict[str, tuple[bytes, bool]] = {}

    for rel, (mode, sha) in sorted(src.items()):
        blob = _git(["cat-file", "blob", sha], REPO_ROOT, binary=True)
        executable = mode == "100755"
        abs_dest = os.path.join(dest_root, rel)
        if rel not in dst:
            adds.append(rel)
            payload[rel] = (blob, executable)
        else:
            current = open(abs_dest, "rb").read()
            is_exec = os.access(abs_dest, os.X_OK)
            if current != blob or is_exec != executable:
                updates.append(rel)
                payload[rel] = (blob, executable)

    for rel in sorted(dst):
        if rel not in src:
            deletes.append(rel)

    return adds, updates, deletes, payload


def print_plan(adds, updates, deletes, dest_root: str, rev: str,
               skipped: list[str]) -> None:
    print(f"gridbook sync   source: {REPO_ROOT} @ {rev} (committed content only)")
    print(f"                dest:   {dest_root}")
    for src_sub, dst_sub in SUBTREES:
        print(f"                {src_sub}/ -> {dst_sub}/")
    if skipped:
        print(f"\nEXCLUDED — in flight, not committed at {rev} "
              f"({len(skipped)} path(s)):")
        for path in skipped:
            print(f"    ~ {path}")
    print(f"\n{len(adds)} added, {len(updates)} updated, {len(deletes)} deleted")
    for rel in adds:
        print(f"    + {rel}")
    for rel in updates:
        print(f"    M {rel}")
    for rel in deletes:
        print(f"    - {rel}")
    if not (adds or updates or deletes):
        print("    (trees are in sync)")


def in_flight_paths(rev: str) -> list[str]:
    """Working-tree paths under the synced subtrees that are NOT part of ``rev``
    — the kernels that are not ready yet. Reported, never synced."""
    subs = [s for s, _ in SUBTREES]
    out = _git(["status", "--porcelain", "-z", "--untracked-files=all", "--",
                *subs], REPO_ROOT)
    paths = []
    for record in out.split("\0"):
        if len(record) > 3:
            paths.append(record[3:])
    return sorted(paths)


def apply_plan(adds, updates, deletes, payload, dest_root: str) -> None:
    roots = [os.path.join(dest_root, dst) for _, dst in SUBTREES]
    for rel in adds + updates:
        abs_dest = os.path.join(dest_root, rel)
        if not any(_within(abs_dest, r) for r in roots):
            raise Refused(f"refusing to write outside the synced subtrees: {rel}")
        blob, executable = payload[rel]
        os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
        with open(abs_dest, "wb") as fh:
            fh.write(blob)
        mode = os.stat(abs_dest).st_mode
        os.chmod(abs_dest, (mode | 0o111) if executable else (mode & ~0o111))
    for rel in deletes:
        abs_dest = os.path.join(dest_root, rel)
        if not any(_within(abs_dest, r) for r in roots):
            raise Refused(f"refusing to delete outside the synced subtrees: {rel}")
        os.remove(abs_dest)
        parent = os.path.dirname(abs_dest)
        while parent not in roots and _within(parent, dest_root) and \
                not os.listdir(parent):
            os.rmdir(parent)
            parent = os.path.dirname(parent)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; write nothing")
    ap.add_argument("--rev", default="HEAD",
                    help="prismaquant revision to sync from (default HEAD; the "
                         "working tree is never used)")
    ap.add_argument("--dest", default=os.environ.get("GRIDBOOK_REPO", DEFAULT_DEST),
                    help="gridbook release repo (env GRIDBOOK_REPO, default "
                         f"{DEFAULT_DEST})")
    args = ap.parse_args(argv)

    dest_root = os.path.abspath(args.dest)
    try:
        guard_destination(dest_root)
        adds, updates, deletes, payload = build_plan(args.rev, dest_root)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print_plan(adds, updates, deletes, dest_root, args.rev,
               in_flight_paths(args.rev))
    drift = bool(adds or updates or deletes)

    if args.check:
        if drift:
            print("\nDRIFT: the release repo is out of sync. Run "
                  "`python3 scripts/sync_gridbook.py` to bring it forward.",
                  file=sys.stderr)
            return 1
        return 0

    if not drift:
        return 0
    try:
        apply_plan(adds, updates, deletes, payload, dest_root)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"\napplied to {dest_root}. Nothing was committed, tagged, or pushed "
          "— releasing is a human action.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
