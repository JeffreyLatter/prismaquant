"""Export must ignore the absent sibling + fused name of an incomplete fused
group whose present members are BF16.

vLLM fuses q/k/v -> qkv_proj and requires one scheme across the group. Gemma4
attention_k_eq_v layers synthesize v=k and ship NO v_proj. If q/k are BF16 but
v_proj/qkv_proj aren't in the ignore list, vLLM rejects the mixed fused group
("Found a different quantization schemes for ['q_proj','k_proj','v_proj']").
build_quantization_config must add the absent v_proj AND the fused qkv_proj to
ignore so the fused module loads uniformly unquantized.
"""
from prismaquant.export_native_compressed import build_quantization_config
from prismaquant.model_profiles.gemma4 import Gemma4Profile


def test_kv_eq_v_layer_ignores_absent_vproj_and_fused():
    p = Gemma4Profile()
    # sliding layer 0: complete q/k/v, quantized. k_eq_v layer 5: q/k BF16, v absent.
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "NVFP4",
        "model.layers.0.self_attn.v_proj": "NVFP4",
    }
    bf16_passthrough = {
        "model.layers.5.self_attn.q_proj",
        "model.layers.5.self_attn.k_proj",
        # v_proj absent (synthesized v=k)
    }
    cfg = build_quantization_config(assignment, bf16_passthrough, profile=p)
    ign = set(cfg["ignore"])

    # the absent v_proj AND the fused qkv_proj for the k_eq_v layer must be ignored
    assert any(x.endswith("layers.5.self_attn.v_proj") for x in ign), ign
    assert any(x.endswith("layers.5.self_attn.qkv_proj") for x in ign), ign

    # the complete, quantized sliding layer must NOT have its qkv_proj ignored
    assert not any(x.endswith("layers.0.self_attn.qkv_proj") for x in ign), ign
