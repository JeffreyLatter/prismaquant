from __future__ import annotations

import json

import pytest

import prismaquant._fast_kernel_guard as fast_kernel_guard


def _write_config(path, payload):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(payload))
    return path


def test_fast_kernel_guard_reads_profile_requirements(tmp_path, monkeypatch):
    model_path = _write_config(
        tmp_path / "qwen35",
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
    )

    real_import = fast_kernel_guard.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name in {"causal_conv1d", "fla"}:
            raise ImportError
        return real_import(name, *args, **kwargs)

    assert fast_kernel_guard._requirements_for_model(str(model_path))

    monkeypatch.delenv("PRISMAQUANT_ALLOW_PYTORCH_FALLBACK", raising=False)
    monkeypatch.setattr(fast_kernel_guard.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError) as exc:
        fast_kernel_guard.require_fast_kernels(str(model_path))

    message = str(exc.value)
    assert "causal-conv1d" in message
    assert "flash-linear-attention" in message


def test_fast_kernel_guard_skips_profiles_without_requirements(tmp_path, monkeypatch):
    model_path = _write_config(
        tmp_path / "qwen3",
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
    )

    def fake_import(_name, *args, **kwargs):
        raise ImportError

    monkeypatch.delenv("PRISMAQUANT_ALLOW_PYTORCH_FALLBACK", raising=False)
    monkeypatch.setattr(fast_kernel_guard.importlib, "import_module", fake_import)

    fast_kernel_guard.require_fast_kernels(str(model_path))


def test_fast_kernel_guard_keeps_remote_id_fallback():
    requirements = fast_kernel_guard._requirements_for_model("Qwen3.6-35B-A3B")

    assert ("causal_conv1d", "causal-conv1d (Dao-AILab/causal-conv1d)") in requirements
