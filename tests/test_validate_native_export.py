from prismaquant.validate_native_export import _speculative_config_uses_embedded_mtp


def test_speculative_config_embedded_mtp_detection():
    assert _speculative_config_uses_embedded_mtp({"method": "mtp"})
    assert _speculative_config_uses_embedded_mtp({"method": "qwen3_5_mtp"})
    assert _speculative_config_uses_embedded_mtp({"method": "qwen3_next_mtp"})

    assert not _speculative_config_uses_embedded_mtp({"method": "ngram"})
    assert not _speculative_config_uses_embedded_mtp({"method": "draft_model"})
    assert not _speculative_config_uses_embedded_mtp({})
