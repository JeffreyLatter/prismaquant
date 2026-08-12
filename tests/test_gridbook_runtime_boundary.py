"""Gridbook is an immutable external runtime, never vendored into PrismaQuant."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO / "prismaquant" / "gridbook_runtime"
HELPER = ASSET_DIR / "gridbook_runtime.sh"
PIN = ASSET_DIR / "gridbook_runtime_pin.json"
LIVE_SCRIPTS = (
    "canary_ladder.sh",
    "serve_dsv4_cb_validate.sh",
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
    pins = [
        path
        for root in (REPO / "prismaquant", REPO / "scripts")
        for path in root.rglob("*gridbook*pin*.json")
    ]
    assert pins == [PIN]
    payload = json.loads(PIN.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "repository", "commit", "version",
                            "version_is_release"}
    assert payload["schema"] == "prismaquant.gridbook_runtime_pin.v2"
    assert payload["repository"] == "https://github.com/RobTand/gridbook.git"
    assert re.fullmatch(r"[0-9a-f]{40}", payload["commit"])
    assert re.fullmatch(r"[0-9]+(?:[.][0-9]+)+(?:[A-Za-z0-9.+-]*)?",
                        payload["version"])
    assert isinstance(payload["version_is_release"], bool)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_rung_tables_may_only_credit_a_runtime_that_was_actually_released():
    """The version-drift ratchet.

    ``rungs_by_runtime_version`` is keyed by runtime VERSION, but the pin's
    identity is its COMMIT -- and a feature merge does not bump
    ``gridbook.__version__``. So a pin can advance to a post-release master
    commit while keeping the same version string, and a rung table keyed on
    that string silently starts describing a runtime nobody released. Three
    different commits self-reported 0.7.0 during the source-passthrough work,
    which is exactly how this gap was found.

    Two properties close it without inventing a release-history table:

      * no key may name a runtime NEWER than the one actually pinned -- a
        table cannot promise rungs from a version this producer has never
        resolved; and
      * the pinned version may appear as a key only when the pinned commit IS
        that version's release (``version_is_release``). An unreleased pin
        backs nothing, which is the fail-closed direction the spec's own
        "when the pin advances, ADD the version key" rule already asks for.
    """
    payload = json.loads(PIN.read_text(encoding="utf-8"))
    pin_version = payload["version"]
    spec = json.loads(
        (REPO / "prismaquant" / "serving_profile_specs" / "nvfp4_cb.json")
        .read_text(encoding="utf-8"))

    keyed: set[str] = set()
    for lane in spec["serving_lanes"]:
        fused = lane.get("fused_mid_m")
        if fused is not None:
            keyed |= set(fused["rungs_by_runtime_version"])

    for version in sorted(keyed):
        assert _version_tuple(version) <= _version_tuple(pin_version), (
            f"serving profile credits runtime {version}, which is newer than "
            f"the pinned runtime {pin_version}: a rung table cannot promise a "
            f"version the producer has never resolved")

    if not payload["version_is_release"]:
        assert pin_version not in keyed, (
            f"the pin names version {pin_version} but its commit "
            f"{payload['commit']} is not that version's release "
            f"(version_is_release is false), so no rung table may credit it -- "
            f"either pin the release tag commit or drop the {pin_version} key")


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
    discovered = {
        path.name
        for path in (REPO / "scripts").glob("*.sh")
        if any(
            marker in path.read_text(encoding="utf-8")
            for marker in (
                "gridbook_runtime_prepare",
                "install-container",
                "gridbook_runtime_install_container",
            )
        )
    }
    assert discovered == set(LIVE_SCRIPTS)
    for name in LIVE_SCRIPTS:
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "gridbook_runtime.sh" in text, name
        assert "prismaquant/gridbook_runtime/gridbook_runtime.sh" in text, name
        assert "gridbook_runtime_prepare" in text, name
        assert "GRIDBOOK_RUNTIME_DOCKER_ARGS" in text, name
        assert "PQ_GRIDBOOK_RUNTIME_HELPER" in text, name
        assert (
            "install-container" in text
            or "gridbook_runtime_install_container" in text
        ), name
        assert "set -euo pipefail" in text, name
        assert "plugins/gridbook" not in text, name
        assert "/repo/scripts/lib/gridbook_runtime.sh" not in text, name
        assert "--quantization prismaquant" not in text, name
        # Import safety is intentionally owned once by the shared Docker
        # argument vector.  A launcher-local override could appear after that
        # vector and silently win Docker's last-value environment resolution.
        if name == "serve_dsv4_cb_validate.sh":
            # This launcher has a host-side, artifact-build-commit snapshot
            # bootstrap before Gridbook preparation. Its strict source
            # bootstrap requires PYTHONSAFEPATH in the host environment; the
            # container value remains owned by GRIDBOOK_RUNTIME_DOCKER_ARGS.
            assert "-e PYTHONSAFEPATH" not in text, name
            assert "--env PYTHONSAFEPATH" not in text, name
        else:
            assert "PYTHONSAFEPATH" not in text, name
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


def test_standalone_checkout_rejects_linked_git_metadata(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    commit = _make_gridbook_checkout(repository)
    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "worktree", "add", "--quiet", "--detach", str(linked), commit],
        cwd=repository,
        check=True,
    )
    command = (
        f'. "{HELPER}"; '
        'gridbook_runtime_verify_standalone_checkout "$1" "$2" 9.9.9'
    )
    refused = _bash(command, str(linked), commit)
    assert refused.returncode == 2
    assert "not self-contained" in refused.stderr

    accepted = _bash(command, str(repository), commit)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == str(repository)


def test_prepare_mounts_runtime_source_and_contract_read_only(tmp_path):
    checkout = tmp_path / "gridbook"
    checkout.mkdir()
    commit = _make_gridbook_checkout(checkout)

    assets = tmp_path / "contract"
    assets.mkdir()
    copied_helper = assets / HELPER.name
    shutil.copy2(HELPER, copied_helper)
    (assets / PIN.name).write_text(json.dumps({
        "schema": "prismaquant.gridbook_runtime_pin.v2",
        "repository": "https://github.com/example/gridbook.git",
        "commit": commit,
        "version": "9.9.9",
        "version_is_release": False,
    }), encoding="utf-8")

    prepared = _bash(
        'GRIDBOOK_RUNTIME_CHECKOUT="$1"; '
        'GRIDBOOK_RUNTIME_CACHE_DIR="$3"; '
        '. "$2"; '
        'gridbook_runtime_prepare; '
        'printf "%s\\n" "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}"',
        str(checkout), str(copied_helper), str(tmp_path / "cache"),
    )
    assert prepared.returncode == 0, prepared.stderr
    args = prepared.stdout.splitlines()
    materialized = tmp_path / "cache" / commit
    assert f"{materialized}:/opt/prismaquant-gridbook-source:ro" in args
    assert (materialized / ".git").is_dir()
    assert (
        f"{assets}:/opt/prismaquant-gridbook-runtime-contract:ro" in args
    )
    assert (
        "PQ_GRIDBOOK_RUNTIME_HELPER="
        "/opt/prismaquant-gridbook-runtime-contract/gridbook_runtime.sh"
    ) in args
    assert "--workdir" in args
    assert "/" in args
    assert "--env" in args
    assert "PYTHONSAFEPATH=1" in args
    assert all("/repo/" not in arg for arg in args)


def test_prepare_materializes_linked_worktree_as_standalone_checkout(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    commit = _make_gridbook_checkout(repository)
    checkout = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--quiet", "--detach", str(checkout), commit],
        cwd=repository,
        check=True,
    )
    assert (checkout / ".git").is_file()

    assets = tmp_path / "contract"
    assets.mkdir()
    copied_helper = assets / HELPER.name
    shutil.copy2(HELPER, copied_helper)
    (assets / PIN.name).write_text(json.dumps({
        "schema": "prismaquant.gridbook_runtime_pin.v2",
        "repository": "https://github.com/example/gridbook.git",
        "commit": commit,
        "version": "9.9.9",
        "version_is_release": False,
    }), encoding="utf-8")
    cache = tmp_path / "cache"
    prepared = _bash(
        'GRIDBOOK_RUNTIME_CHECKOUT="$1"; '
        'GRIDBOOK_RUNTIME_CACHE_DIR="$3"; '
        '. "$2"; '
        'gridbook_runtime_prepare; '
        'printf "%s\\n" "$GRIDBOOK_RUNTIME_SOURCE"',
        str(checkout), str(copied_helper), str(cache),
    )
    assert prepared.returncode == 0, prepared.stderr
    materialized = cache / commit
    assert prepared.stdout.strip() == str(materialized)
    assert (materialized / ".git").is_dir()
    assert not (materialized / ".git" / "objects" / "info" / "alternates").exists()
    verified = subprocess.run(
        ["git", "-C", str(materialized), "fsck", "--no-dangling"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert verified.returncode == 0, verified.stderr


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


def test_container_install_preserves_exact_vcs_provenance():
    text = HELPER.read_text(encoding="utf-8")
    assert (
        'git+file://${install_source}@${GRIDBOOK_RUNTIME_COMMIT}' in text
    )
    assert "--force-reinstall" in text
    assert "cp -a --no-preserve=ownership" in text
    assert "--no-build-isolation" in text
    assert '"commit_id": expected_commit' in text
    assert '"requested_revision": expected_commit' in text
    assert '"vcs": "git"' in text
    assert "importlib.import_module(\"gridbook\")" in text
    assert "module_file != installed_init" in text
    assert "entry.relative_to(package_root)" in text
    assert "imported_version != expected" in text


def test_runtime_helper_owns_safe_import_path_and_neutral_workdir():
    text = HELPER.read_text(encoding="utf-8")
    assert '--workdir /' in text
    assert '--env "PYTHONSAFEPATH=1"' in text
    assert "export PYTHONSAFEPATH=1" in text


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
