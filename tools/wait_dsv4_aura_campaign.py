#!/usr/bin/env python3
"""Bind the live DSv4 AURA invocation and publish its terminal receipt.

The public entry point accepts only the fixed DSv4 release campaign.  It first
materializes the clean current Git commit as a content-addressed runtime
snapshot, then re-executes this tool from that snapshot before importing
PrismaQuant.  ``wait`` never starts, stops, or restarts the campaign service.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence


RUN_ROOT = Path("/home/rob/dq-runs/dsv4-flash-0731")
WORK_DIR = RUN_ROOT / "aura-cb-reprice-streamed-cached"
HISTORICAL_ROOT = RUN_ROOT / "aura-cb-reprice" / "checkpoints" / "aura"
SNAPSHOT_CACHE = RUN_ROOT / "runtime-source-cache"
SERVICE_UNIT = "pq-aura-dsv4-streamed-cached.service"
_INTERNAL = "PQ_DSV4_COMPLETION_SNAPSHOT_REEXEC"
_SNAPSHOT = "PQ_DSV4_COMPLETION_SNAPSHOT"
_COMMIT = "PQ_DSV4_COMPLETION_COMMIT"
_TREE = "PQ_DSV4_COMPLETION_TREE"
_CLOSURE = "PQ_DSV4_COMPLETION_CLOSURE_SHA256"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class WaiterError(RuntimeError):
    """The fixed release campaign cannot be attested safely."""


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise WaiterError(detail.strip()) from exc
    return result.stdout.strip()


def _json_command(command: Sequence[str]) -> dict[str, Any]:
    raw = _run(command)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WaiterError("runtime snapshot tool returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise WaiterError("runtime snapshot tool did not return an object")
    return value


def _git(repo: Path, *args: str) -> str:
    return _run(("git", "-C", str(repo), *args))


def _outer_reexec(argv: Sequence[str]) -> None:
    repo = Path(__file__).resolve(strict=True).parent.parent
    top = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != repo:
        raise WaiterError(f"waiter is outside its Git root: {repo} != {top}")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise WaiterError(
            "release checkout must be clean before arming the terminal waiter"
        )
    commit = _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if _COMMIT_RE.fullmatch(commit) is None or _COMMIT_RE.fullmatch(tree) is None:
        raise WaiterError("release Git commit/tree identity is invalid")
    cache = Path(os.environ.get("PQ_RUNTIME_SNAPSHOT_CACHE", str(SNAPSHOT_CACHE)))
    helper = repo / "tools" / "prismaquant_runtime_snapshot.py"
    identity = _json_command((
        sys.executable,
        str(helper),
        "materialize",
        "--source-root",
        str(repo),
        "--cache-root",
        str(cache),
        "--commit",
        commit,
    ))
    snapshot = Path(str(identity.get("snapshot", ""))).resolve(strict=True)
    closure = str(identity.get("closure_sha256", ""))
    if (
        identity.get("commit") != commit
        or identity.get("tree") != tree
        or _SHA256_RE.fullmatch(closure) is None
    ):
        raise WaiterError("materialized runtime identity differs from clean HEAD")
    if (
        _git(repo, "rev-parse", "--verify", "HEAD^{commit}") != commit
        or _git(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise WaiterError("release checkout changed while snapshotting the waiter")
    inner = snapshot / "tools" / Path(__file__).name
    if not inner.is_file() or inner.is_symlink():
        raise WaiterError("reviewed runtime snapshot lacks the completion waiter")
    environment = dict(os.environ)
    environment.update({
        _INTERNAL: "1",
        _SNAPSHOT: str(snapshot),
        _COMMIT: commit,
        _TREE: tree,
        _CLOSURE: closure,
        "PYTHONPATH": str(snapshot),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PRISMAQUANT_IDENTITY_GIT_COMMIT": commit,
        "PRISMAQUANT_IDENTITY_GIT_DIRTY": "0",
    })
    os.execve(
        sys.executable,
        [sys.executable, str(inner), *argv],
        environment,
    )


def _verify_inner_snapshot() -> dict[str, Any]:
    snapshot_raw = os.environ.get(_SNAPSHOT, "")
    commit = os.environ.get(_COMMIT, "")
    tree = os.environ.get(_TREE, "")
    closure = os.environ.get(_CLOSURE, "")
    if (
        not snapshot_raw
        or _COMMIT_RE.fullmatch(commit or "") is None
        or _COMMIT_RE.fullmatch(tree or "") is None
        or _SHA256_RE.fullmatch(closure or "") is None
    ):
        raise WaiterError("internal runtime-snapshot identity is incomplete")
    snapshot = Path(snapshot_raw).resolve(strict=True)
    expected_script = snapshot / "tools" / Path(__file__).name
    if Path(__file__).resolve(strict=True) != expected_script:
        raise WaiterError("completion waiter is not executing from its snapshot")
    helper = snapshot / "tools" / "prismaquant_runtime_snapshot.py"
    identity = _json_command((
        sys.executable,
        str(helper),
        "verify",
        "--snapshot",
        str(snapshot),
        "--expected-commit",
        commit,
        "--expected-tree",
        tree,
        "--expected-closure-sha256",
        closure,
    ))
    return identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    wait = subparsers.add_parser(
        "wait", help="bind the already-running service and publish one receipt"
    )
    wait.add_argument("--timeout-seconds", type=int, default=12 * 60 * 60)
    subparsers.add_parser(
        "verify", help="re-audit an existing receipt and all campaign inputs"
    )
    return parser


def _inner_main(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_identity = _verify_inner_snapshot()
    expected_init = (
        Path(str(snapshot_identity["snapshot"])) / "prismaquant" / "__init__.py"
    ).resolve(strict=True)
    import importlib.util

    spec = importlib.util.find_spec("prismaquant")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected_init:
        raise WaiterError("Python resolves PrismaQuant outside the reviewed snapshot")
    from prismaquant.dsv4_campaign_completion import (
        CampaignCompletionError,
        PRODUCER_SCHEMA,
        audit_campaign_closure,
        build_completion_receipt,
        completion_receipt_path,
        load_completion_receipt,
        publish_completion_receipt,
        verify_receipt_against_current_campaign,
        wait_for_bound_systemd_service,
    )

    receipt_path = completion_receipt_path(WORK_DIR)
    producer = {
        "schema": PRODUCER_SCHEMA,
        "snapshot": str(snapshot_identity["snapshot"]),
        "commit": str(snapshot_identity["commit"]),
        "tree": str(snapshot_identity["tree"]),
        "closure_sha256": str(snapshot_identity["closure_sha256"]),
    }
    try:
        if args.command == "verify":
            receipt = verify_receipt_against_current_campaign(
                receipt_path,
                work_dir=WORK_DIR,
                historical_root=HISTORICAL_ROOT,
                expected_producer_commit=producer["commit"],
            )
            _verify_inner_snapshot()
            return {
                "status": "verified",
                "receipt": str(receipt_path.resolve(strict=True)),
                "receipt_sha256": receipt["receipt_sha256"],
            }
        if receipt_path.exists() or receipt_path.is_symlink():
            raise CampaignCompletionError(
                f"completion receipt already exists; use verify: {receipt_path}"
            )
        service = wait_for_bound_systemd_service(
            SERVICE_UNIT, timeout_seconds=args.timeout_seconds
        )
        _verify_inner_snapshot()
        campaign = audit_campaign_closure(WORK_DIR, HISTORICAL_ROOT)
        _verify_inner_snapshot()
        receipt = build_completion_receipt(
            service=service,
            producer=producer,
            campaign=campaign,
        )
        publish_completion_receipt(receipt_path, receipt)
        loaded = load_completion_receipt(receipt_path)
        if loaded != receipt:
            raise CampaignCompletionError("durable completion receipt changed")
        _verify_inner_snapshot()
        return {
            "status": "published",
            "receipt": str(receipt_path.resolve(strict=True)),
            "receipt_sha256": receipt["receipt_sha256"],
            "unit_count": campaign["unit_count"],
            "historical_overlap_count": campaign["historical_overlap"][
                "exact_payload_match_count"
            ],
            "service_invocation_id": service["invocation_id"],
        }
    except CampaignCompletionError:
        raise


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if os.environ.get(_INTERNAL) != "1":
            _outer_reexec(raw_argv)
            raise AssertionError("execve returned")
        args = _parser().parse_args(raw_argv)
        result = _inner_main(args)
    except (OSError, WaiterError, RuntimeError) as exc:
        print(f"dsv4-campaign-completion: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
