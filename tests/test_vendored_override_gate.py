"""The profile registry must refuse a model_type whose vendored override died.

Issue #19's real defect was silence: `register_qwen3()` returned cleanly, set
its "registered" flag, and the run then executed UPSTREAM modelling code with
no exception anywhere. `prismaquant.vendored` now verifies its own overrides
and records the dead ones — but `detect_profile` calls
`register_vendored_modeling()` inside a `try/except: pass` (correctly: a
vendoring failure must not break profile *detection*), so the recorded failure
has to be consulted explicitly or the swallow re-hides it.

These tests pin that consultation, not the vendored machinery itself.
"""
from __future__ import annotations

import pytest

import prismaquant.vendored as vendored
from prismaquant.model_profiles.registry import _refuse_dead_vendored_override


@pytest.fixture(autouse=True)
def _clean_override_errors():
    """OVERRIDE_ERRORS is process-global; never leak a synthetic entry."""
    saved = dict(vendored.OVERRIDE_ERRORS)
    vendored.OVERRIDE_ERRORS.clear()
    vendored.OVERRIDE_ERRORS.update(saved)
    yield
    vendored.OVERRIDE_ERRORS.clear()
    vendored.OVERRIDE_ERRORS.update(saved)


def test_gate_is_inert_when_the_override_is_healthy():
    # On the reference box the qwen3 override resolves, so nothing is recorded
    # and the gate must not fire — this is the common path and it must stay
    # free of false positives.
    vendored.OVERRIDE_ERRORS.clear()
    assert _refuse_dead_vendored_override("qwen3") is None
    assert _refuse_dead_vendored_override("llama") is None


def test_gate_raises_for_a_recorded_dead_override():
    vendored.OVERRIDE_ERRORS["qwen3"] = (
        "synthetic: AutoModelForCausalLM.register no-op'd (config __module__ "
        "filter) and the config shim could not be installed either"
    )
    with pytest.raises(RuntimeError) as exc:
        _refuse_dead_vendored_override("qwen3")
    msg = str(exc.value)
    # The message must say WHICH model_type, and that the consequence is
    # running upstream code — the whole point is that this is otherwise silent.
    assert "qwen3" in msg
    assert "UPSTREAM" in msg
    # and it must carry the vendored layer's own detail rather than replacing it
    assert "config __module__" in msg


def test_gate_is_scoped_to_the_failing_model_type():
    """One dead override must not block unrelated architectures."""
    vendored.OVERRIDE_ERRORS["qwen3"] = "synthetic failure"
    assert _refuse_dead_vendored_override("gemma4") is None
    assert _refuse_dead_vendored_override("deepseek_v4") is None
    with pytest.raises(RuntimeError):
        _refuse_dead_vendored_override("qwen3")


def test_gate_survives_a_missing_vendored_package(monkeypatch):
    """A tree without the vendored package must still detect profiles.

    The gate is a safety net, not a dependency: if `prismaquant.vendored`
    cannot be imported at all there is no override to be silently wrong about.
    """
    import builtins

    real_import = builtins.__import__

    def _no_vendored(name, *args, **kwargs):
        if name == "prismaquant.vendored" or name.startswith(
                "prismaquant.vendored."):
            raise ImportError("synthetic: vendored package absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_vendored)
    assert _refuse_dead_vendored_override("qwen3") is None
