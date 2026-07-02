"""MINOR-M33: the streaming Fisher probe must fail loud on KV-sharing models
(num_kv_shared_layers>0), where it under-counts the storing layer's
k_proj/v_proj h_trace, rather than ship a silently-biased allocation."""
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


def test_guard_fires_when_kv_shared_and_no_override(monkeypatch):
    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 2)
    monkeypatch.delenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", raising=False)
    msg = ip.kv_shared_fisher_block_reason("any/model")
    assert msg is not None
    assert "num_kv_shared_layers=2" in msg
    assert "MINOR-M33" in msg


def test_guard_silent_when_no_kv_sharing(monkeypatch):
    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 0)
    monkeypatch.delenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", raising=False)
    assert ip.kv_shared_fisher_block_reason("any/model") is None


def test_guard_override_allows_probe(monkeypatch):
    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 5)
    monkeypatch.setenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", "1")
    assert ip.kv_shared_fisher_block_reason("any/model") is None
