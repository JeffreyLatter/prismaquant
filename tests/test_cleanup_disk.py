from __future__ import annotations

import os
import time
from pathlib import Path

from tools.cleanup_disk import iter_candidates


def _touch_old(path: Path, *, hours: float) -> None:
    timestamp = time.time() - hours * 3600.0
    os.utime(path, (timestamp, timestamp))


def test_cleanup_finds_old_transient_cache_without_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "dq-runs"
    run_cache = run_root / "run-a" / "cache"
    cache = run_cache / "triton"
    artifact = run_root / "run-a" / "artifacts" / "layer_config.json"
    pycache = repo / "prismaquant" / "__pycache__"

    cache.mkdir(parents=True)
    (cache / "kernel.so").write_bytes(b"x" * 17)
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    pycache.mkdir(parents=True)
    (pycache / "module.pyc").write_bytes(b"x")

    _touch_old(cache / "kernel.so", hours=100)
    _touch_old(cache, hours=100)
    _touch_old(run_cache, hours=100)
    _touch_old(pycache / "module.pyc", hours=100)
    _touch_old(pycache, hours=100)

    candidates = iter_candidates(
        run_root=run_root,
        repo_root=repo,
        min_age_hours=72,
        now=time.time(),
    )
    paths = {candidate.path for candidate in candidates}

    assert run_cache in paths
    assert pycache in paths
    assert artifact not in paths
    assert artifact.parent not in paths


def test_cleanup_respects_keep_marker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "dq-runs"
    cache = run_root / "run-a" / "cache" / "triton"
    cache.mkdir(parents=True)
    (run_root / "run-a" / ".prismaquant-keep").write_text("", encoding="utf-8")
    (cache / "kernel.so").write_bytes(b"x")
    _touch_old(cache, hours=100)

    candidates = iter_candidates(
        run_root=run_root,
        repo_root=repo,
        min_age_hours=72,
        now=time.time(),
    )

    assert candidates == []


def test_cleanup_skips_recent_cache(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "dq-runs"
    cache = run_root / "run-a" / "cache" / "triton"
    cache.mkdir(parents=True)
    (cache / "kernel.so").write_bytes(b"x")

    candidates = iter_candidates(
        run_root=run_root,
        repo_root=repo,
        min_age_hours=72,
        now=time.time(),
    )

    assert candidates == []


def test_cleanup_does_not_treat_package_core_modules_as_dumps(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "dq-runs"
    site_package = run_root / "venvs" / "env" / "lib" / "python3.12" / "site-packages"
    site_package.mkdir(parents=True)
    (site_package / "core.py").write_text("# module\n", encoding="utf-8")
    (site_package / "core.abi3.so").write_bytes(b"so")
    (site_package / "core").write_text("schema\n", encoding="utf-8")
    real_dump = run_root / "old-run" / "core.12345"
    real_dump.parent.mkdir(parents=True)
    real_dump.write_bytes(b"dump")

    for path in (
        site_package / "core.py",
        site_package / "core.abi3.so",
        site_package / "core",
        real_dump,
    ):
        _touch_old(path, hours=100)

    candidates = iter_candidates(
        run_root=run_root,
        repo_root=repo,
        min_age_hours=72,
        now=time.time(),
    )
    paths = {candidate.path for candidate in candidates}

    assert real_dump in paths
    assert site_package / "core.py" not in paths
    assert site_package / "core.abi3.so" not in paths
    assert site_package / "core" not in paths
