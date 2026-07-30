"""MINOR-M33, now INVERTED: KV-sharing models (num_kv_shared_layers>0) are
probed normally, because the reverse sweep routes each consumer's Fisher
cotangent back to the layer that produced the borrowed K/V (the KV-cotangent
path — equivalence against an end-to-end backward is pinned in
tests/test_kv_cotangent_path.py). The guard now fires only when that path is
UNAVAILABLE, i.e. when PRISMAQUANT_KV_COTANGENT=0 reinstates the severed
cotangent and its k_proj/v_proj under-count — never ship a silently-biased
allocation."""
import json

import prismaquant.incremental_probe as ip


def test_config_num_kv_shared_layers_top_level(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 4, "num_kv_shared_layers": 3}))
    monkeypatch.setattr(ip, "stage_text_only", lambda p: str(tmp_path))
    assert ip.config_num_kv_shared_layers(str(tmp_path)) == 3


def test_config_num_kv_shared_layers_text_config(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_kv_shared_layers": 2}}))
    monkeypatch.setattr(ip, "stage_text_only", lambda p: str(tmp_path))
    assert ip.config_num_kv_shared_layers(str(tmp_path)) == 2


def test_config_num_kv_shared_layers_absent_is_zero(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 4}))
    monkeypatch.setattr(ip, "stage_text_only", lambda p: str(tmp_path))
    assert ip.config_num_kv_shared_layers(str(tmp_path)) == 0


def test_guard_silent_when_the_cotangent_path_is_available(monkeypatch):
    """The point of landing the KV-cotangent path: KV sharing is no longer a
    reason to refuse to probe."""
    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 2)
    monkeypatch.delenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", raising=False)
    monkeypatch.delenv("PRISMAQUANT_KV_COTANGENT", raising=False)
    assert ip.kv_shared_fisher_block_reason("any/model") is None


def test_guard_fires_when_the_cotangent_path_is_disabled(monkeypatch):
    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 2)
    monkeypatch.delenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", raising=False)
    monkeypatch.setenv("PRISMAQUANT_KV_COTANGENT", "0")
    msg = ip.kv_shared_fisher_block_reason("any/model")
    assert msg is not None
    assert "num_kv_shared_layers=2" in msg
    assert "MINOR-M33" in msg


def test_guard_silent_when_no_kv_sharing(monkeypatch):
    for cotangent in ("0", "1"):
        monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 0)
        monkeypatch.delenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", raising=False)
        monkeypatch.setenv("PRISMAQUANT_KV_COTANGENT", cotangent)
        assert ip.kv_shared_fisher_block_reason("any/model") is None


def test_guard_override_allows_probe(monkeypatch):
    """The accept-the-under-count escape hatch still works, for anyone
    deliberately reproducing a pre-fix probe."""
    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 5)
    monkeypatch.setenv("PRISMAQUANT_KV_COTANGENT", "0")
    monkeypatch.setenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", "1")
    assert ip.kv_shared_fisher_block_reason("any/model") is None
