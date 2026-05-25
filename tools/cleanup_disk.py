"""Prune transient PrismaQuant disk debris.

This is intentionally conservative: it removes compile/runtime caches,
Python bytecode caches, scratch work directories, and core dumps, but it
does not remove exported models, artifacts, metrics, logs, calibration data,
venvs, or Hugging Face model caches outside run-local cache directories.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import os
import shutil
import sys
import time
from pathlib import Path


GIB = 1024**3
KEEP_MARKER = ".prismaquant-keep"

DIR_BASENAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
    "recache_work",
}

RUN_CACHE_SUFFIXES = {
    ("cache",),
    ("cache", "triton"),
    ("cache", "torchinductor"),
    ("cache", "hf"),
    ("metrics", "triton_cache"),
    ("metrics", "torchinductor_cache"),
    ("metrics", "hf_cache"),
    ("triton_cache",),
    ("torchinductor_cache",),
    ("hf_cache",),
    ("mse_promotion", "propagated_group_work"),
    ("mse_promotion", "propagated_serving_unit_work"),
}

FILE_GLOBS = {
    "core.[0-9]*",
    "*.core",
    "hs_err_pid*.log",
}

PROTECTED_PARTS = {
    "artifacts",
    "calibration",
    "exported",
    "exported_4p75",
    "exported_5p53",
    "exported_current_only",
    "exported_scale_5p0",
    "exported_scale_10p0",
    "exported_validated_kneedle_4p70",
    "logs",
    "venvs",
}


@dataclasses.dataclass(frozen=True)
class Candidate:
    path: Path
    kind: str
    size_bytes: int
    age_hours: float


def _now() -> float:
    return time.time()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _dir_size(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path, topdown=True, onerror=lambda _err: None):
        dirs[:] = [d for d in dirs if not Path(root, d).is_symlink()]
        for name in files:
            item = Path(root, name)
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def _path_size(path: Path) -> int:
    try:
        if path.is_dir() and not path.is_symlink():
            return _dir_size(path)
        return path.stat().st_size
    except OSError:
        return 0


def _age_hours(path: Path, now: float) -> float:
    try:
        return max(0.0, (now - path.stat().st_mtime) / 3600.0)
    except OSError:
        return 0.0


def _has_keep_marker(path: Path) -> bool:
    cur = path if path.is_dir() else path.parent
    for parent in [cur, *cur.parents]:
        if (parent / KEEP_MARKER).exists():
            return True
    return False


def _matches_file_glob(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in FILE_GLOBS)


def _has_protected_part(path: Path, stop_at: Path | None = None) -> bool:
    parts = path.parts
    if stop_at is not None and _is_relative_to(path, stop_at):
        rel_parts = path.relative_to(stop_at).parts
        parts = rel_parts
    return any(part in PROTECTED_PARTS for part in parts)


def _matches_run_cache_suffix(path: Path, run_root: Path) -> bool:
    try:
        rel = path.relative_to(run_root).parts
    except ValueError:
        return False
    for suffix in RUN_CACHE_SUFFIXES:
        if len(rel) >= len(suffix) and rel[-len(suffix) :] == suffix:
            return True
    return False


def _candidate_kind(path: Path, run_root: Path, repo_root: Path) -> str | None:
    if path.name == KEEP_MARKER:
        return None
    if path.is_dir():
        if path.name in DIR_BASENAMES:
            return "dev-cache"
        if _matches_run_cache_suffix(path, run_root):
            return "run-cache"
        if _is_relative_to(path, repo_root / "tmp"):
            return "repo-tmp"
        return None
    if path.is_file() and _matches_file_glob(path):
        return "crash-dump"
    return None


def _should_skip(path: Path, kind: str, run_root: Path, repo_root: Path) -> bool:
    if _has_keep_marker(path):
        return True
    if kind == "run-cache":
        return False
    if kind == "dev-cache":
        return False
    if kind == "repo-tmp":
        return not _is_relative_to(path, repo_root / "tmp")
    if kind == "crash-dump":
        return False
    if _is_relative_to(path, run_root) and _has_protected_part(path, run_root):
        return True
    return False


def iter_candidates(
    *,
    run_root: Path,
    repo_root: Path,
    min_age_hours: float,
    now: float | None = None,
) -> list[Candidate]:
    now = _now() if now is None else now
    roots = [repo_root, run_root]
    candidates: dict[Path, Candidate] = {}

    for root in roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, topdown=True, onerror=lambda _err: None):
            current_path = Path(current)
            if current_path != root and _has_keep_marker(current_path):
                dirs[:] = []
                continue

            child_names = list(dirs) + list(files)
            for name in child_names:
                path = current_path / name
                kind = _candidate_kind(path, run_root, repo_root)
                if kind is None:
                    continue
                if _should_skip(path, kind, run_root, repo_root):
                    continue
                age = _age_hours(path, now)
                if age < min_age_hours:
                    continue
                candidates[path.resolve()] = Candidate(
                    path=path,
                    kind=kind,
                    size_bytes=_path_size(path),
                    age_hours=age,
                )

            dirs[:] = [
                name
                for name in dirs
                if not _candidate_kind(current_path / name, run_root, repo_root)
            ]

    return sorted(
        candidates.values(),
        key=lambda c: (c.kind, -c.size_bytes, str(c.path)),
    )


def remove_candidate(candidate: Candidate) -> None:
    path = candidate.path
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def format_bytes(value: int) -> str:
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    mib = 1024**2
    if value >= mib:
        return f"{value / mib:.2f} MiB"
    kib = 1024
    if value >= kib:
        return f"{value / kib:.2f} KiB"
    return f"{value} B"


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/rob/dq-runs"))
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--min-age-hours", type=float, default=72.0)
    parser.add_argument("--apply", action="store_true", help="delete candidates")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="append a cleanup summary to this log file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only print/delete the first N candidates; 0 means no limit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()

    candidates = iter_candidates(
        run_root=run_root,
        repo_root=repo_root,
        min_age_hours=args.min_age_hours,
    )
    if args.limit:
        candidates = candidates[: args.limit]

    total = sum(c.size_bytes for c in candidates)
    action = "delete" if args.apply else "would delete"
    lines = [
        f"PrismaQuant cleanup: {action} {len(candidates)} paths, {format_bytes(total)}",
        f"run_root={run_root}",
        f"repo_root={repo_root}",
        f"min_age_hours={args.min_age_hours}",
    ]
    for c in candidates:
        lines.append(
            f"{c.kind:10s} {format_bytes(c.size_bytes):>10s} "
            f"{c.age_hours:8.1f}h {c.path}"
        )

    if args.apply:
        for candidate in candidates:
            remove_candidate(candidate)

    output = "\n".join(lines)
    print(output)
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            handle.write("\n")
            handle.write(output)
            handle.write("\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
