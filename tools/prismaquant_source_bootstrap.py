#!/usr/bin/env python3
"""Import PrismaQuant from the verified source snapshot without PYTHONPATH.

Release containers invoke this file from the read-only snapshot itself.  Its
own canonical parent is therefore the only source root it will add to
``sys.path``.  A transported root is an assertion, not an alternate search
location: it must resolve to that same parent, Python safe-path mode must be
active, and ``PYTHONPATH`` must be absent from the process environment.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import re
import runpy
import sys
import types
from typing import Sequence


SOURCE_ROOT_ENV = "PQ_RUNTIME_PRISMAQUANT_ROOT"
_MODULE = re.compile(r"prismaquant(?:[.][A-Za-z_][A-Za-z0-9_]*)+")
_RELEASE_TOOLS = {
    "serve-fingerprint": "tools/serve_fingerprint.py",
}


class SourceBootstrapError(ValueError):
    """The requested source root is not the bootstrap's verified snapshot."""


def activate_prismaquant_source(
    source_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Prepend this tool's exact snapshot root and verify import resolution.

    When ``source_root`` or :data:`SOURCE_ROOT_ENV` is supplied, strict release
    mode is enabled.  Ordinary source-tree imports remain supported for unit
    tests and developer tools, but release mode additionally proves that no
    environment-level Python search override can reach the server process.
    """
    tool_path = Path(__file__)
    if tool_path.is_symlink() or not tool_path.is_file():
        raise SourceBootstrapError("source bootstrap must be one real file")
    code_root = tool_path.resolve(strict=True).parents[1]

    transported = source_root
    if transported is None:
        transported = os.environ.get(SOURCE_ROOT_ENV)
    strict = transported is not None
    if strict:
        if "PYTHONPATH" in os.environ:
            raise SourceBootstrapError(
                "PYTHONPATH must be absent in a release source bootstrap"
            )
        if os.environ.get("PYTHONSAFEPATH") != "1" or not sys.flags.safe_path:
            raise SourceBootstrapError(
                "PYTHONSAFEPATH=1 and active Python safe-path mode are required "
                "in a release source bootstrap"
            )
        if (
            os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or not sys.dont_write_bytecode
        ):
            raise SourceBootstrapError(
                "PYTHONDONTWRITEBYTECODE=1 and active no-bytecode mode are "
                "required in a release source bootstrap"
            )
        if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
            raise SourceBootstrapError(
                "PYTHONNOUSERSITE=1 and disabled user-site mode are required "
                "in a release source bootstrap"
            )
        raw_root = Path(os.fspath(transported))
        if not raw_root.is_absolute() or raw_root.is_symlink():
            raise SourceBootstrapError(
                "transported PrismaQuant source root must be absolute and non-symlink"
            )
        try:
            selected_root = raw_root.resolve(strict=True)
        except OSError as exc:
            raise SourceBootstrapError(
                "transported PrismaQuant source root is unreadable"
            ) from exc
        if selected_root != code_root:
            raise SourceBootstrapError(
                "transported PrismaQuant source root differs from the bootstrap "
                f"snapshot: {selected_root} != {code_root}"
            )
    else:
        selected_root = code_root

    package_root = selected_root / "prismaquant"
    expected_origin_path = package_root / "__init__.py"
    if (
        package_root.is_symlink()
        or not package_root.is_dir()
        or expected_origin_path.is_symlink()
        or not expected_origin_path.is_file()
    ):
        raise SourceBootstrapError(
            "selected source root has no real PrismaQuant package"
        )
    expected_origin = expected_origin_path.resolve(strict=True)

    selected_text = str(selected_root)
    sys.path[:] = [entry for entry in sys.path if entry != selected_text]
    sys.path.insert(0, selected_text)
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("prismaquant")
    if spec is None or not isinstance(spec.origin, str):
        raise SourceBootstrapError("Python cannot resolve PrismaQuant")
    try:
        observed_origin = Path(spec.origin).resolve(strict=True)
    except OSError as exc:
        raise SourceBootstrapError(
            "resolved PrismaQuant package origin is unreadable"
        ) from exc
    if observed_origin != expected_origin:
        raise SourceBootstrapError(
            "Python resolves PrismaQuant outside the selected snapshot: "
            f"{observed_origin} != {expected_origin}"
        )
    return selected_root


def _install_exact_package_namespace(root: Path) -> None:
    """Expose exact submodules without executing PrismaQuant's heavy init.

    ``serve_fingerprint`` is intentionally stdlib-only and must not import
    torch merely to call the stdlib-only shipcard hash routine.  The snapshot
    bootstrap has already proven ``root/prismaquant/__init__.py`` is the exact
    selected package origin, so a package namespace rooted only there safely
    permits the whitelisted tool's lazy ``prismaquant.shipcard`` import.
    """
    if "prismaquant" in sys.modules:
        raise SourceBootstrapError(
            "PrismaQuant was imported before the release-tool bootstrap"
        )
    package_root = (root / "prismaquant").resolve(strict=True)
    module = types.ModuleType("prismaquant")
    module.__file__ = str(package_root / "__init__.py")
    module.__package__ = "prismaquant"
    module.__path__ = [str(package_root)]
    spec = importlib.machinery.ModuleSpec(
        "prismaquant", loader=None, is_package=True
    )
    spec.origin = module.__file__
    spec.submodule_search_locations = [str(package_root)]
    module.__spec__ = spec
    sys.modules["prismaquant"] = module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--source-root")
    run_module = commands.add_parser("run-module")
    run_module.add_argument("--source-root")
    run_module.add_argument("module")
    run_module.add_argument("arguments", nargs=argparse.REMAINDER)
    run_tool = commands.add_parser(
        "run-tool",
        help="execute one explicitly approved tracked release tool",
    )
    run_tool.add_argument("--source-root")
    run_tool.add_argument("tool", choices=tuple(sorted(_RELEASE_TOOLS)))
    run_tool.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = activate_prismaquant_source(args.source_root)
    if args.command == "check":
        print(root)
        return 0
    if args.command == "run-module" and _MODULE.fullmatch(args.module) is None:
        raise SourceBootstrapError(
            "run-module accepts only a qualified prismaquant module"
        )
    if args.command == "run-module":
        _install_exact_package_namespace(root)
        sys.argv = [args.module, *args.arguments]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        return 0
    relative = _RELEASE_TOOLS[args.tool]
    tool = root / relative
    if tool.is_symlink() or not tool.is_file():
        raise SourceBootstrapError(
            f"approved release tool is absent or unsafe: {relative}"
        )
    resolved_tool = tool.resolve(strict=True)
    try:
        resolved_tool.relative_to(root)
    except ValueError as exc:
        raise SourceBootstrapError(
            f"approved release tool escapes the selected snapshot: {relative}"
        ) from exc
    _install_exact_package_namespace(root)
    sys.argv = [str(resolved_tool), *args.arguments]
    runpy.run_path(str(resolved_tool), run_name="__main__")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourceBootstrapError as exc:
        print(f"prismaquant-source-bootstrap: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
