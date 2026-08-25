#!/usr/bin/env python3
"""Internal stdlib-only MemAvailable sampler for the DSpark arm collector.

This is not an evidence-report authoring interface.  The source-closed parent
collector creates a private state directory and pre-opens a one-byte arming
pipe; this helper only samples into that directory until the parent collector
creates the stop marker.  Keeping the observer in a fresh Python process
avoids retaining the package's PyTorch/Transformers import footprint during a
long model startup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from typing import NoReturn, Sequence


PHASES = ("startup", "ready", "warmup", "measured", "post")
ERROR_SCHEMA = "prismaquant.dspark_memory_sampler_error.v1"


class SamplerError(RuntimeError):
    """The private memory observer cannot continue safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _exclusive_write(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mem_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    value = int(fields[1]) * 1024
                    if value > 0:
                        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise SamplerError("cannot read /proc/meminfo MemAvailable") from exc
    raise SamplerError("/proc/meminfo has no valid MemAvailable")


def _phase(state_dir: Path) -> str:
    path = state_dir / "phase"
    if path.is_symlink() or not path.is_file():
        raise SamplerError("memory sampler phase file is absent or unsafe")
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise SamplerError("memory sampler phase is unreadable") from exc
    if value not in PHASES:
        raise SamplerError(f"memory sampler phase is invalid: {value!r}")
    return value


def _append_sample(path: Path, *, sequence: int, phase: str) -> None:
    flags = os.O_WRONLY | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise SamplerError("memory sample ledger is not a regular file")
        row = {
            "sequence": sequence,
            "observed_at": _utc_now(),
            "phase": phase,
            "mem_available_bytes": _mem_available(),
        }
        data = (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _stop_requested(state_dir: Path) -> bool:
    marker = state_dir / "stop"
    if not marker.exists():
        return False
    if marker.is_symlink() or not marker.is_file():
        raise SamplerError("memory sampler stop marker is unsafe")
    return True


def _record_error(state_dir: Path, exc: BaseException) -> None:
    payload = {
        "schema": ERROR_SCHEMA,
        "observed_at": _utc_now(),
        "error_type": type(exc).__name__,
        "detail": str(exc),
    }
    try:
        _exclusive_write(
            state_dir / "error.json",
            (
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8"),
        )
    except BaseException:
        pass


def _sample(state_dir: Path, interval_ms: int) -> None:
    sequence = 0
    next_deadline = time.monotonic()
    last_wall: datetime | None = None
    while True:
        wall = datetime.now(timezone.utc)
        if last_wall is not None and wall <= last_wall:
            raise SamplerError("memory sampler wall clock regressed")
        _append_sample(
            state_dir / "samples.jsonl",
            sequence=sequence,
            phase=_phase(state_dir),
        )
        if sequence == 0:
            _exclusive_write(state_dir / "ready", b"ready\n")
        sequence += 1
        last_wall = wall
        if _stop_requested(state_dir):
            break
        next_deadline += interval_ms / 1000.0
        delay = next_deadline - time.monotonic()
        if not math.isfinite(delay) or delay < -1.0:
            raise SamplerError("memory sampler scheduling fell behind")
        time.sleep(max(0.0, delay))
    _exclusive_write(state_dir / "stopped", b"stopped\n")


def _fail(message: str) -> NoReturn:
    raise SamplerError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--start-fd", required=True, type=int)
    parser.add_argument("--interval-ms", required=True, type=int)
    args = parser.parse_args(argv)
    raw_state = Path(args.state_dir)
    if (
        not raw_state.is_absolute()
        or raw_state.is_symlink()
        or args.start_fd < 3
        or args.interval_ms <= 0
    ):
        return 2
    try:
        state_dir = raw_state.resolve(strict=True)
    except OSError:
        return 2
    if not state_dir.is_dir():
        return 2
    try:
        try:
            armed = os.read(args.start_fd, 1)
        finally:
            os.close(args.start_fd)
        if armed != b"1":
            _fail("memory sampler was not armed by its collector")
        _sample(state_dir, args.interval_ms)
    except BaseException as exc:
        _record_error(state_dir, exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
