#!/usr/bin/env python3
"""Publish an exported artifact to the Hub — the ONE blocking gate (R16).

**The ruling (Robert, 2026-07-30).** Pipeline ship gates stay *advisory*: the
build venv cannot import `vllm`, so `run-pipeline.sh` cannot run a serve-lane
gate and never should (ARCHITECTURE §7.1). The blocking point is **publication**.
Nothing forces a quality number while an artifact sits on disk; the moment it
becomes public, the shipcard must be closed.

    python3 tools/publish_artifact.py <artifact_dir> --repo-id rdtand/<name>

Before it uploads anything — and before it even prints the command it *would*
run — this refuses unless `prismaquant.shipcard.verify` passes on the artifact's
`shipcard.json`, listing every unfilled or failing slot. `huggingface_hub` is
imported lazily: the build venv may not have it, in which case the tool prints
the exact upload command and exits 0 without pretending to have published.

Escape hatch, deliberately loud: `--force-unverified` publishes an artifact whose
card is not closed, but only after the operator **re-types the artifact
directory's basename** (interactively, or via `--confirm-name`), and it stamps
`forced_unverified: true` into the shipcard together with the problems that were
overridden — so the artifact itself carries the record that it was published
ungated.

Exit codes: 0 published (or command printed) · 1 refused · 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.shipcard import (  # noqa: E402
    SHIPCARD_FILENAME,
    compute_model_sha,
    load_shipcard,
    unfilled_slots,
    verify,
    write_shipcard,
)

REPO_TYPES = ("model", "dataset", "space")


def _shipcard_path(artifact_dir: Path, explicit: str | None) -> Path:
    return Path(explicit) if explicit else artifact_dir / SHIPCARD_FILENAME


def upload_command(args: argparse.Namespace) -> str:
    """The exact CLI equivalent of the `upload_folder` call this would make.

    Printed verbatim when `huggingface_hub` is not importable (the build venv
    often lacks it) and under `--dry-run`, so "we could not upload" never
    degrades into "we did something else".
    """
    cmd = [
        "hf", "upload", args.repo_id, str(Path(args.artifact_dir).resolve()),
        args.path_in_repo, "--repo-type", args.repo_type,
    ]
    if args.private:
        cmd.append("--private")
    if args.commit_message:
        cmd += ["--commit-message", args.commit_message]
    for pattern in args.allow_patterns or []:
        cmd += ["--include", pattern]
    for pattern in args.ignore_patterns or []:
        cmd += ["--exclude", pattern]
    return " ".join(shlex.quote(part) for part in cmd)


def check_shipcard(
    artifact_dir: Path,
    shipcard_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    """`(card, problems)` — an unreadable/missing card is itself a refusal."""
    if not shipcard_path.is_file():
        return None, [
            f"no shipcard at {shipcard_path} — an artifact with no ship record "
            "has never been gated; re-export at a commit that writes one, or "
            "open it with prismaquant.shipcard.build_shipcard"
        ]
    try:
        card = load_shipcard(shipcard_path)
    except Exception as exc:
        return None, [f"{shipcard_path} is not a readable shipcard: {exc!r}"]
    return card, verify(card, model_dir=artifact_dir)


def _confirm_forced(artifact_dir: Path, confirm_name: str | None) -> bool:
    """Re-typing the basename is the confirmation. Deliberate, not a y/n."""
    expected = artifact_dir.resolve().name
    typed = confirm_name
    if typed is None:
        if not sys.stdin.isatty():
            print(
                "[publish] REFUSED: --force-unverified needs the artifact "
                f"directory basename re-typed ({expected!r}); no tty, so pass "
                "--confirm-name",
                file=sys.stderr)
            return False
        typed = input(
            f"[publish] Type the artifact directory name to publish it "
            f"UNVERIFIED ({expected}): ")
    if str(typed).strip() != expected:
        print(
            f"[publish] REFUSED: typed {str(typed).strip()!r} != {expected!r}; "
            "the confirmation must match the artifact directory basename",
            file=sys.stderr)
        return False
    return True


def stamp_forced(
    shipcard_path: Path,
    card: dict[str, Any] | None,
    artifact_dir: Path,
    problems: list[str],
    repo_id: str,
) -> None:
    """Record on the artifact that it was published without a closed card."""
    if card is None:
        print(
            "[publish] WARN no shipcard to stamp — the forced publish is "
            "recorded in this log only",
            file=sys.stderr)
        return
    card["forced_unverified"] = True
    history = list(card.get("forced_unverified_history") or [])
    history.append({
        "repo_id": repo_id,
        "model_sha": compute_model_sha(artifact_dir),
        "problems": list(problems),
        "unfilled_slots": unfilled_slots(card),
    })
    card["forced_unverified_history"] = history
    write_shipcard(shipcard_path, card)
    print(f"[publish] stamped forced_unverified=true into {shipcard_path}")


def _upload(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import upload_folder  # noqa: PLC0415
    except Exception as exc:
        print(f"[publish] huggingface_hub is not importable here ({exc!r}); "
              "the artifact is VERIFIED and ready. Run this from an "
              "environment that has it:")
        print(f"  {upload_command(args)}")
        return 0
    if args.dry_run:
        print("[publish] --dry-run; the upload this would run is:")
        print(f"  {upload_command(args)}")
        return 0
    print(f"[publish] uploading {args.artifact_dir} -> {args.repo_id} "
          f"({args.repo_type})")
    url = upload_folder(
        repo_id=args.repo_id,
        folder_path=str(Path(args.artifact_dir).resolve()),
        path_in_repo=args.path_in_repo,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        allow_patterns=args.allow_patterns or None,
        ignore_patterns=args.ignore_patterns or None,
    )
    print(f"[publish] done: {url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact_dir", help="exported/ directory to publish")
    ap.add_argument("--repo-id", required=True, help="e.g. rdtand/<name>")
    ap.add_argument("--repo-type", default="model", choices=REPO_TYPES)
    ap.add_argument("--path-in-repo", default=".")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message", default=None)
    ap.add_argument("--allow-patterns", nargs="*", default=None)
    ap.add_argument("--ignore-patterns", nargs="*", default=None)
    ap.add_argument("--shipcard", default=None,
                    help=f"default: <artifact_dir>/{SHIPCARD_FILENAME}")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify, then print the upload command instead of "
                         "running it")
    ap.add_argument("--force-unverified", action="store_true",
                    help="publish despite an unclosed shipcard; requires the "
                         "artifact directory basename re-typed and stamps "
                         "forced_unverified into the card")
    ap.add_argument("--confirm-name", default=None,
                    help="the re-typed basename, for non-interactive use of "
                         "--force-unverified")
    args = ap.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_dir():
        print(f"[publish] ERROR: {artifact_dir} is not a directory",
              file=sys.stderr)
        return 2

    shipcard_path = _shipcard_path(artifact_dir, args.shipcard)
    card, problems = check_shipcard(artifact_dir, shipcard_path)

    if problems:
        print(f"[publish] REFUSED — {len(problems)} problem(s) with "
              f"{shipcard_path}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if not args.force_unverified:
            print("[publish] nothing was uploaded and no upload command was "
                  "printed. Close the slots (python -m prismaquant.shipcard_cli show|fill) or "
                  "re-run with --force-unverified.", file=sys.stderr)
            return 1
        if not _confirm_forced(artifact_dir, args.confirm_name):
            return 2
        print("[publish] WARNING: publishing an UNVERIFIED artifact by "
              "explicit override.", file=sys.stderr)
        stamp_forced(shipcard_path, card, artifact_dir, problems, args.repo_id)
    else:
        print(f"[publish] shipcard OK — every slot closed and matching for "
              f"{json.dumps(str(artifact_dir.resolve()))}")

    return _upload(args)


if __name__ == "__main__":
    raise SystemExit(main())
