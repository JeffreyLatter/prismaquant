"""Gridbook is an immutable external runtime, never vendored into PrismaQuant."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "lib" / "gridbook_runtime.sh"
PIN = REPO / "scripts" / "lib" / "gridbook_runtime_pin.json"
LIVE_SCRIPTS = (
    "canary_ladder.sh",
    "serve_hy3_smoke.sh",
    "serve_hy3_teb.sh",
    "serve_laguna_smoke.sh",
    "serve_qwen27b_smoke.sh",
    "smoke_nvfp4_cb_delegation.sh",
)


def _bash(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "bash", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_gridbook_pin_is_one_full_immutable_commit():
    pins = list((REPO / "scripts").rglob("*gridbook*pin*.json"))
    assert pins == [PIN]
    payload = json.loads(PIN.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "repository", "commit", "version"}
    assert payload["schema"] == "prismaquant.gridbook_runtime_pin.v1"
    assert payload["repository"] == "https://github.com/RobTand/gridbook.git"
    assert re.fullmatch(r"[0-9a-f]{40}", payload["commit"])
    assert re.fullmatch(r"[0-9]+(?:[.][0-9]+)+(?:[A-Za-z0-9.+-]*)?",
                        payload["version"])


def test_no_gridbook_runtime_or_tests_are_vendored():
    assert not (REPO / "plugins" / "gridbook").exists()
    assert not (REPO / "scripts" / "sync_gridbook.py").exists()
    assert not (REPO / "tests" / "test_gridbook_sync.py").exists()


def test_producer_does_not_import_external_gridbook_runtime():
    violations: list[str] = []
    for path in (REPO / "prismaquant").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "gridbook" or name.startswith("gridbook.")
                   for name in names):
                violations.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not violations, (
        "PrismaQuant is the producer and must not import Gridbook runtime code: "
        f"{violations}")


def test_every_live_script_uses_the_one_external_runtime_helper():
    for name in LIVE_SCRIPTS:
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "gridbook_runtime.sh" in text, name
        assert "gridbook_runtime_prepare" in text, name
        assert "GRIDBOOK_RUNTIME_DOCKER_ARGS" in text, name
        assert (
            "install-container" in text
            or "gridbook_runtime_install_container" in text
        ), name
        assert "set -euo pipefail" in text, name
        assert "plugins/gridbook" not in text, name
        assert "--quantization prismaquant" not in text, name
    delegation = (REPO / "scripts" /
                  "smoke_nvfp4_cb_delegation.sh").read_text(encoding="utf-8")
    assert "--quantization gridbook" in delegation


def test_helper_and_live_scripts_are_valid_bash():
    paths = [HELPER, *(REPO / "scripts" / name for name in LIVE_SCRIPTS)]
    for path in paths:
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 0, f"{path}: {proc.stderr}"


def _make_gridbook_checkout(root: Path) -> str:
    (root / "gridbook").mkdir(parents=True)
    (root / "gridbook" / "__init__.py").write_text(
        '__version__ = "9.9.9"\n', encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gridbook"\nversion = "9.9.9"\n',
        encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Gridbook Test"],
                   cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"],
                   cwd=root, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def test_checkout_override_requires_exact_clean_commit(tmp_path):
    checkout = tmp_path / "gridbook"
    checkout.mkdir()
    commit = _make_gridbook_checkout(checkout)
    command = (
        f'. "{HELPER}"; '
        'gridbook_runtime_verify_checkout "$1" "$2" 9.9.9')
    clean = _bash(command, str(checkout), commit)
    assert clean.returncode == 0, clean.stderr
    assert clean.stdout.strip() == str(checkout)

    # Exercise the Docker-root compatibility contract: every Git read in the
    # verifier must mark only the exact resolved checkout as safe. A wrapper
    # fails the verification if any rev-parse/status call omits that option.
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper = wrapper_dir / "git"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'expected=${GRIDBOOK_TEST_SAFE_DIRECTORY:?}\n'
        'if [[ "$1" != "-c" || "$2" != "safe.directory=$expected" '
        '|| "$3" != "-C" || "$4" != "$expected" ]]; then\n'
        '  printf "unsafe git invocation: %q " "$@" >&2\n'
        '  printf "\\n" >&2\n'
        "  exit 97\n"
        "fi\n"
        f'exec "{real_git}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    safe_command = (
        f'. "{HELPER}"; '
        'PATH="$1:$PATH" GRIDBOOK_TEST_SAFE_DIRECTORY="$2" '
        'gridbook_runtime_verify_checkout "$2" "$3" 9.9.9')
    safe = _bash(safe_command, str(wrapper_dir), str(checkout), commit)
    assert safe.returncode == 0, safe.stderr
    assert safe.stdout.strip() == str(checkout)

    (checkout / "untracked").write_text("dirty", encoding="utf-8")
    dirty = _bash(command, str(checkout), commit)
    assert dirty.returncode == 2
    assert "is dirty" in dirty.stderr

    wrong = _bash(command, str(checkout), "0" * 40)
    assert wrong.returncode == 2
    assert "does not equal pinned" in wrong.stderr


def test_container_install_reloads_and_enforces_the_tracked_pin():
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    wrong_commit = "f" * 40 if pin["commit"] != "f" * 40 else "e" * 40
    commit = _bash(
        f'PQ_GRIDBOOK_RUNTIME_SOURCE=/not-used '
        f'PQ_GRIDBOOK_RUNTIME_COMMIT={wrong_commit} '
        f'PQ_GRIDBOOK_RUNTIME_VERSION={pin["version"]} '
        f'bash "{HELPER}" install-container')
    assert commit.returncode == 2
    assert "does not equal tracked pin" in commit.stderr

    version = _bash(
        f'PQ_GRIDBOOK_RUNTIME_SOURCE=/not-used '
        f'PQ_GRIDBOOK_RUNTIME_COMMIT={pin["commit"]} '
        f'PQ_GRIDBOOK_RUNTIME_VERSION=999.0.0 '
        f'bash "{HELPER}" install-container')
    assert version.returncode == 2
    assert "does not equal tracked pin" in version.stderr


def test_runtime_helper_has_no_wheel_or_runtime_kind_branch():
    text = HELPER.read_text(encoding="utf-8")
    assert "GRIDBOOK_RUNTIME_WHEEL" not in text
    assert "PQ_GRIDBOOK_RUNTIME_KIND" not in text
    assert "gridbook_runtime_verify_wheel" not in text


def test_container_install_path_is_owned_only_by_runtime_helper():
    marker = "/tmp/gridbook-runtime-"
    assert marker in HELPER.read_text(encoding="utf-8")
    for path in (REPO / "scripts").rglob("*.sh"):
        if path == HELPER:
            continue
        assert marker not in path.read_text(encoding="utf-8"), path
    canary = (REPO / "scripts" / "canary_ladder.sh").read_text(
        encoding="utf-8"
    )
    assert canary.count("gridbook_runtime_container_install_target") == 2
