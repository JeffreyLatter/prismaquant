"""Mixed-container dispatch: CB groups (with a "scheme" key) stay ours; stock
compressed-tensors groups (no "scheme" key) are delegated to a real
CompressedTensorsConfig (serving-kernel.md §2). vLLM-only, so skip-guarded —
run in the vllm-node container:

  docker run --rm --gpus all -v /home/rob/prismaquant:/repo --entrypoint bash \
    vllm-node:latest -c 'pip install -e /repo/plugins/vllm_prismaquant --no-deps -q; \
    PYTHONPATH=/repo/plugins/vllm_prismaquant python3 -m pytest \
    /repo/plugins/vllm_prismaquant/tests/test_delegation.py -q'
"""
import pytest

pytest.importorskip("vllm")
from vllm_prismaquant.config import PrismaQuantConfig  # noqa: E402


def _mixed_config():
    nvfp4_w = {"num_bits": 4, "type": "float", "strategy": "tensor_group",
               "group_size": 16, "symmetric": True, "dynamic": False,
               "scale_dtype": "torch.float8_e4m3fn"}
    nvfp4_a = {"num_bits": 4, "type": "float", "strategy": "tensor_group",
               "group_size": 16, "symmetric": True, "dynamic": "local",
               "scale_dtype": "torch.float8_e4m3fn"}
    return {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "group_cb": {
                "format": "FP8_CB_K44",
                "targets": ["model.layers.0.mlp.down_proj"],
                "scheme": {"grid": "fp8", "mode": "product", "k": 44,
                           "n_sub": 4, "type_size": 176, "group_size": 0,
                           "vec_dim": 8, "codebook_group": "down_proj",
                           "codebook_source": "learned",
                           "codebook_ref": [
                               "cb_codebook.down_proj.FP8_CB_K44.sub0"]}},
            "group_nvfp4": {
                "format": "nvfp4-pack-quantized",
                "weights": nvfp4_w, "input_activations": nvfp4_a,
                "targets": ["re:.*mlp.gate_proj$", "re:.*mlp.up_proj$",
                            "re:.*self_attn.*_proj$"]},
        },
        "ignore": ["lm_head", "model.embed_tokens"],
    }


def test_cb_stock_split_and_ct_built():
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    assert c.ct_config is not None, "stock CT config not built for mixed artifact"
    assert len(c.target_scheme) == 1                     # 1 CB target
    assert "model.layers.0.mlp.down_proj" in c._cb_targets


def test_cb_prefix_stays_ours():
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    # CB target resolves to a CB scheme (fused-aware).
    assert c._scheme_for_prefix("model.layers.0.mlp.down_proj") is not None
    # stock prefix does NOT CB-resolve (falls through to CT delegation).
    assert c._scheme_for_prefix("model.layers.0.mlp.gate_proj") is None


def test_ct_owns_stock_and_ignores_cb():
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        find_matched_target,
    )
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    # CB modules are in CT's ignore so CT never tries to own them.
    assert "model.layers.0.mlp.down_proj" in c.ct_config.ignore
    # CT recognises a stock prefix as one of its targets.
    tgt = find_matched_target(
        "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj",
        list(c.ct_config.target_scheme_map.keys()),
        c.ct_config.packed_modules_mapping)
    assert tgt is not None


def test_uniform_cb_has_no_ct():
    """A pure-CB (uniform) artifact must not build a CT config (no stock groups)."""
    cfg = _mixed_config()
    del cfg["config_groups"]["group_nvfp4"]
    c = PrismaQuantConfig.from_config(cfg)
    c._ensure_resolved()
    assert c.ct_config is None
