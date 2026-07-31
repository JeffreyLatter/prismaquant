"""JIT build/load of the ROCm/HIP CB kernel extension (``gridbook/csrc_hip``).

The HIP twin of :mod:`gridbook.cuda_ext`, deliberately mirroring its idiom:
lazy first-use build, fail-soft to the existing Triton path with one loud
warning, sources located through ``importlib.resources`` so an in-repo
checkout, ``pip install -e`` and a wheel install resolve identically, and a
build cache that is **never** ``/tmp`` (``PRISMAQUANT_CB_EXT_DIR`` if set, else
``~/.cache/prismaquant-cb-hip-ext``).

Three things about the HIP build that are not obvious and cost real time to
discover, recorded here because they are load-bearing:

1. **``.hip`` sources are compiled by hipcc but are NOT protected from
   hipify.** ``torch.utils.cpp_extension`` treats ``.hip`` as a device-source
   extension on ROCm (``_is_cuda_file``), yet ``load()`` still passes every
   source through ``hipify_python.hipify(..., hipify_extra_files_only=True)``,
   which processes ``extra_files`` regardless of extension.  The sources in
   ``csrc_hip`` are therefore written in HIP spellings already and use
   ``hipLaunchKernelGGL(HIP_KERNEL_NAME(...))`` rather than ``<<<>>>``: hipify's
   kernel-launch rewriter is textual and mis-parses template arguments
   containing commas, which every kernel here has.  Nothing in these files is
   rewritten, so what is compiled is what is in the repo.

2. **The offload arch must be pinned.** ``_get_rocm_arch_flags`` honours
   ``PYTORCH_ROCM_ARCH`` and any user ``--offload-arch``; if neither is present
   it asks torch, which on a multi-arch build can emit code for architectures
   this kernel has not been validated on.  We pass ``--offload-arch`` explicitly
   from the detected device (or ``PRISMAQUANT_CB_HIP_ARCH``).

3. **Fedora needs an explicit runtime link.** On Fedora 44 / ROCm 7.1.1 a plain
   hipcc link fails with ``undefined symbol: __hipUnregisterFatBinary``; adding
   ``-L/usr/lib64 -lamdhip64`` fixes it.  Harmless elsewhere, so it is passed
   unconditionally when that library exists.

No fast-math: the activation-QDQ kernel's division and conversion rounding must
match torch bit-for-bit.
"""
from __future__ import annotations

import os
import sys

_ext = None
_tried = False

_ROCM_HINT = (
    "install a ROCm toolchain matching your torch build (hipcc + "
    "rocm-hip-devel) and make sure `hipcc` is on PATH or ROCM_PATH points at "
    "it; set PRISMAQUANT_CB_EXT_DIR to a writable, persistent directory to "
    "keep the one-time JIT build across restarts"
)

_PKG = getattr(__spec__, "parent", None) or __package__ or "gridbook"

# Device sources, in link order.  The torch binding lives in its own TU so the
# kernels stay torch-free and can also be linked into the standalone self-test.
# The torch binding is a `.cpp` on purpose (host compiler): see the header
# comment in cb_hip_torch.cpp.  The header is listed so staging copies it too.
_SOURCES = ("cb_hip_torch.cpp", "cb_gemv_hip.hip", "cb_gemm_hip.hip")
_HEADERS = ("cb_decode_hip.h",)


class IncompleteInstallError(FileNotFoundError):
    """A packaged HIP source is missing from the installed package.

    Distinct from "no ROCm": a packaging defect, not a property of the machine,
    and reported differently so the two are never confused.
    """


def csrc_hip_dir() -> str:
    """Absolute path to the packaged HIP sources (``gridbook/csrc_hip``)."""
    from importlib.resources import files

    try:
        return os.fspath(files(_PKG) / "csrc_hip")
    except Exception as exc:  # noqa: BLE001
        raise IncompleteInstallError(
            f"cannot locate the gridbook package resources (anchor {_PKG!r}): "
            f"{type(exc).__name__}: {exc}. This is a packaging defect, not a "
            f"missing ROCm toolchain — reinstall gridbook.") from exc


def _require_sources() -> str:
    d = csrc_hip_dir()
    missing = [n for n in _SOURCES if not os.path.isfile(os.path.join(d, n))]
    if missing:
        raise IncompleteInstallError(
            f"gridbook is installed without its HIP sources: {missing} not "
            f"found under {d}. This is a packaging defect, not a missing ROCm "
            f"toolchain — reinstall gridbook.")
    return d


def is_rocm() -> bool:
    """True when the loaded torch is a ROCm build with a live HIP device."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()


def target_arch() -> str | None:
    """The gfx target to compile for: env override, else the live device."""
    env = os.environ.get("PRISMAQUANT_CB_HIP_ARCH")
    if env:
        return env
    try:
        import torch

        name = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return None
    # gcnArchName carries feature suffixes ("gfx1151:xnack-"); --offload-arch
    # accepts them, but the bare name is what the kernels were validated on.
    return name.split(":")[0] if name else None


def _include_next_shim(build_dir: str) -> list[str]:
    """Repair libstdc++'s ``#include_next`` when torch puts ``/usr/include`` on
    the ``-isystem`` list (Fedora's system torch package does).

    Symptom: every HIP source fails with ``fatal error: 'math.h' file not
    found`` raised from ``/usr/include/c++/<v>/cmath:55: #include_next
    <math.h>`` — while the identical source compiles fine under a bare
    ``hipcc``, which is what makes it look like a source bug and is not.

    Cause, in order: clang's HIP wrapper force-includes
    ``cuda_wrappers/cmath``, which ``#include_next``s the real ``<cmath>``;
    ``#include_next`` resumes at the directory AFTER the one holding the
    current file, so the real ``<cmath>`` is found in the GCC C++ directory
    near the end of the search order; its own ``#include_next <math.h>`` then
    has almost nothing left to search — and ``/usr/include``, which is what it
    wants, was hoisted to the front by torch's ``-isystem`` and de-duplicated
    out of its natural late position.  Adding ``-I`` for the C++ directories
    does NOT help: ``-I`` is searched before the clang resource directory, so
    ``#include_next`` skips those entries entirely.

    Fix: give the search order one more directory AFTER everything, holding
    symlinks to the top-level C headers, via ``-idirafter``.  Clang de-dups by
    directory identity, and this is a genuinely distinct directory, so it
    survives; ``math.h`` resolves through the symlink and its own nested
    ``#include``s (``bits/*``) resolve normally from ``/usr/include``.

    Inert when ``/usr/include/c++`` is absent or the sentinel already exists.
    """
    import glob

    if not glob.glob("/usr/include/c++/*/cmath"):
        return []
    shim = os.path.join(build_dir, "_include_next_shim")
    sentinel = os.path.join(shim, ".complete")
    if not os.path.exists(sentinel):
        os.makedirs(shim, exist_ok=True)
        for src in glob.glob("/usr/include/*.h"):
            dst = os.path.join(shim, os.path.basename(src))
            if not os.path.lexists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass
        with open(sentinel, "w") as fh:
            fh.write("symlinks to /usr/include/*.h; see hip_ext.py\n")
    return [f"-idirafter{shim}"]


def _stage_sources(src_dir: str, build_dir: str) -> tuple[str, list[str]]:
    """Copy the packaged sources into the build directory and compile from
    there.

    Two reasons, both real: torch's ROCm ``load()`` runs hipify over its source
    list and writes any rewritten file next to the ORIGINAL, which for a wheel
    install means writing into site-packages (often read-only, and always
    wrong); and a staged tree makes the build hermetic, so an edit in a
    checkout cannot half-apply to a cached build.  Copy is by
    (size, mtime) so a rebuild is only triggered by a genuine change.
    """
    import shutil

    staged_dir = os.path.join(build_dir, "src")
    os.makedirs(staged_dir, exist_ok=True)
    out = []
    for name in _SOURCES + _HEADERS:
        src = os.path.join(src_dir, name)
        dst = os.path.join(staged_dir, name)
        if (not os.path.exists(dst)
                or os.path.getsize(dst) != os.path.getsize(src)
                or os.path.getmtime(dst) < os.path.getmtime(src)):
            shutil.copy2(src, dst)
        if name in _SOURCES:
            out.append(dst)
    return staged_dir, out


def get_ext():
    """The compiled HIP extension module, or None if unavailable.

    None is not an error: every caller falls back to the Triton decode path,
    exactly as :func:`gridbook.cuda_ext.get_ext` does when nvcc is missing.
    """
    global _ext, _tried
    if _tried:
        return _ext
    _tried = True
    if os.environ.get("PRISMAQUANT_CB_HIP", "1") == "0":
        return None
    if not is_rocm():
        return None
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load

        src_dir = _require_sources()
        build_dir = os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-hip-ext")
        os.makedirs(build_dir, exist_ok=True)
        staged_dir, staged = _stage_sources(src_dir, build_dir)

        # The shim goes on BOTH flag lists: torch puts `-isystem /usr/include`
        # on the host compile too, so the host TU hits the identical
        # `#include_next <stdlib.h>` failure that the device TUs hit.
        shim = _include_next_shim(build_dir)
        cflags = ["-O3"] + shim
        flags = ["-O3"] + shim
        arch = target_arch()
        if arch:
            # PYTORCH_ROCM_ARCH, not extra_cuda_cflags.  `_get_rocm_arch_flags`
            # scans the *cxx* flag list for a user-supplied `--offload-arch`,
            # and extra_cuda_cflags is appended after that scan, so passing it
            # there does NOT suppress torch's default arch list — the build then
            # compiles every arch the torch wheel supports (24 of them on the
            # Fedora package) and fails on the first unrelated one.  The env var
            # is checked first and short-circuits the whole list.
            os.environ.setdefault("PYTORCH_ROCM_ARCH", arch)
            flags.append(f"--offload-arch={arch}")
        ldflags = []
        for lib in ("/usr/lib64", "/opt/rocm/lib"):
            if os.path.exists(os.path.join(lib, "libamdhip64.so")):
                ldflags += [f"-L{lib}", "-lamdhip64"]
                break

        _ext = load(name="prismaquant_cb_hip_ext",
                    sources=staged,
                    extra_include_paths=[staged_dir],
                    extra_cflags=cflags,
                    extra_cuda_cflags=flags,
                    extra_ldflags=ldflags,
                    build_directory=build_dir, verbose=False)
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"Falling back to the Triton decode path.",
              file=sys.stderr, flush=True)
        _ext = None
    except Exception as exc:  # noqa: BLE001 — any build/env failure -> fallback
        print(f"[prismaquant-cb] WARNING: gridbook's HIP decode extension "
              f"could not be built ({type(exc).__name__}: {exc}); falling back "
              f"to the Triton decode path (slow prototype). To get the HIP "
              f"path: {_ROCM_HINT}.",
              file=sys.stderr, flush=True)
        _ext = None
    return _ext
