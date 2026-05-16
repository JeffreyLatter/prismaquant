#!/usr/bin/env python3
"""validate_native_export.py — load a compressed-tensors checkpoint via
vLLM and do a single forward + greedy decode. Binary check: either
vLLM accepts the format and produces tokens, or it doesn't.

Usage (from inside a vllm-node container):
    python -m prismaquant.validate_native_export \\
        --model dq-runs-new/qwen36-fresh/exported \\
        --prompt "The capital of France is" \\
        --max-new-tokens 16

The script can optionally upgrade the container's flashinfer to a
serving-profile-pinned version before loading; this is needed for some vLLM builds
that ship with a flashinfer that can't dispatch the NVFP4 MoE backend
on Blackwell. Pass `--no-flashinfer-upgrade` to skip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_FLASHINFER_PACKAGES = ("flashinfer-python", "flashinfer-cubin")


def maybe_upgrade_flashinfer(
    version: str,
    *,
    package_names: tuple[str, ...] = _DEFAULT_FLASHINFER_PACKAGES,
    env: dict[str, str] | None = None,
) -> None:
    """Upgrade flashinfer-python and flashinfer-cubin to `version` and
    set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass the AOT-cache pin
    that lags behind PyPI. No-op if already at the target version.
    """
    for key, value in (env or {"FLASHINFER_DISABLE_VERSION_CHECK": "1"}).items():
        os.environ.setdefault(key, value)
    try:
        import flashinfer
        if getattr(flashinfer, "__version__", "0.0") == version:
            return
    except ImportError:
        pass
    package_specs = [f"{name}=={version}" for name in package_names]
    print(f"[validate] upgrading {', '.join(package_names)} to {version}",
          flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "-q",
        *package_specs,
    ])


def _flashinfer_runtime_package(target_profile: str):
    from .serving_profiles import load_serving_profile

    profile = load_serving_profile(target_profile)
    spec = profile.runtime_package("flashinfer")
    if spec is None:
        return (
            "0.6.8.post1",
            _DEFAULT_FLASHINFER_PACKAGES,
            {"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
        )
    return (
        spec.version or "0.6.8.post1",
        spec.pip_packages or _DEFAULT_FLASHINFER_PACKAGES,
        spec.env_dict() or {"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
    )


def _resolve_validation_target_profile(
    model_dir: str | Path,
    requested: str | None,
) -> str:
    """Resolve the serving profile for an exported checkpoint smoke."""
    from .model_profiles.registry import detect_profile
    from .serving_profiles import resolve_target_profile

    try:
        profile = detect_profile(str(model_dir))
    except Exception:
        profile = None
    return resolve_target_profile(
        profile,
        requested,
        default="vllm_packed_moe",
    )


def summarize_quantization_config(cfg_path: Path) -> None:
    cfg = json.load(open(cfg_path))
    qc = cfg.get("quantization_config", {})
    print(f"[validate] quant_method: {qc.get('quant_method', '<missing>')}")
    print(f"[validate] format:       {qc.get('format', '<missing>')}")
    for gn, g in qc.get("config_groups", {}).items():
        w = g.get("weights", {})
        print(f"[validate]   {gn}: bits={w.get('num_bits')} "
              f"strategy={w.get('strategy')} group={w.get('group_size')} "
              f"format={g.get('format')} n_targets={len(g.get('targets', []))}")
    print(f"[validate]   ignore: {len(qc.get('ignore', []))} entries")


def _speculative_config_uses_embedded_mtp(spec: dict) -> bool:
    method = str(spec.get("method") or "").lower()
    return method == "mtp" or method.endswith("_mtp")


def main():
    from .serving_profiles import serving_profile_names

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Compressed-tensors checkpoint directory.")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--target-profile", default=None,
                    choices=serving_profile_names(),
                    help="Serving profile whose runtime package pins should "
                         "be used for validation preflight. Defaults to the "
                         "exported model profile's configured serving "
                         "profile, then vllm_packed_moe.")
    ap.add_argument("--flashinfer-version", default=None,
                    help="Override the serving profile's flashinfer package "
                         "version before loading vLLM.")
    ap.add_argument("--no-flashinfer-upgrade", action="store_true",
                    help="Skip the flashinfer pre-flight upgrade.")
    ap.add_argument("--speculative-config", default=None,
                    help="JSON string for vLLM SpeculativeConfig. Use this "
                         "to exercise MTP heads, e.g. "
                         "'{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 3, "
                         "\"model\": \"<same model dir>\"}'.")
    ap.add_argument("--no-enforce-eager", action="store_true",
                    help="Allow vLLM compile/CUDA-graph execution instead of "
                         "forcing eager mode. Use after the eager smoke passes.")
    args = ap.parse_args()

    model_dir = Path(args.model)
    target_profile = _resolve_validation_target_profile(
        model_dir,
        args.target_profile,
    )
    print(f"[validate] target profile: {target_profile}", flush=True)

    if not args.no_flashinfer_upgrade:
        version, package_names, env = _flashinfer_runtime_package(target_profile)
        if args.flashinfer_version:
            version = args.flashinfer_version
        maybe_upgrade_flashinfer(version, package_names=package_names, env=env)

    summarize_quantization_config(model_dir / "config.json")

    print(f"[validate] starting vLLM ...", flush=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    spec = None
    if args.speculative_config:
        spec = json.loads(args.speculative_config)
        # If caller omitted "model", default to the same checkpoint for
        # draft-model style configs. MTP-family methods are the exception:
        # their extra heads travel with the target checkpoint, and vLLM expects
        # model to be absent/null so it can take the embedded-MTP path.
        if "model" not in spec and not _speculative_config_uses_embedded_mtp(spec):
            spec["model"] = str(model_dir)
        print(f"[validate] speculative config: {spec}", flush=True)
    llm = LLM(
        model=str(model_dir),
        quantization="compressed-tensors",
        trust_remote_code=True,
        enforce_eager=not args.no_enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        speculative_config=spec,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    out = llm.generate([args.prompt], sp)
    print(f"[validate] generated:", flush=True)
    for o in out:
        print(f"  prompt: {o.prompt!r}", flush=True)
        print(f"  output: {o.outputs[0].text!r}", flush=True)


if __name__ == "__main__":
    main()
