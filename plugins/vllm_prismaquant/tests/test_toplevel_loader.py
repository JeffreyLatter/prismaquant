"""Top-level stacked-CB expert-loader wrapper (moe_toplevel_loader.py).

Regression cover for the HYV3 (Hy3) serving bug: archs that load MoE experts at
the top-level model via a per-expert ``expert_params_mapping`` (and ``continue``
past ``mlp.experts`` in their stacked-params loop) NEVER call the per-layer
``FusedMoE.load_weights``, so our instance-level CB hook is dead code for them
and the stock loader ``KeyError``s on our stacked ``…experts.gate_up_proj.
cb_qweight``. The wrapper here intercepts exactly those, copies them into the
registered fused params, and delegates everything else unchanged.

torch-only (no vLLM): the wrapper module imports just torch, so this runs in any
venv with torch. The full 194-tensor artifact mapping proof is the offline
validate_hy3_cb_loader.py oracle (create_weights + real quant_config schemes).
"""
import pytest
import torch

from vllm_prismaquant.moe_toplevel_loader import (
    install_toplevel_cb_expert_loader,
    map_cb_expert_name,
)


def test_map_cb_expert_name_positive():
    P = "model.layers.7.mlp.experts."
    assert map_cb_expert_name(P + "gate_up_proj.cb_qweight") == P + "w13_cb_qweight"
    assert map_cb_expert_name(P + "down_proj.cb_qweight") == P + "w2_cb_qweight"
    assert map_cb_expert_name(P + "gate_up_proj.weight_scale") == P + "w13_weight_scale"
    assert map_cb_expert_name(P + "down_proj.weight_scale") == P + "w2_weight_scale"
    # prefix-agnostic (pure suffix rewrite): works with or without model. prefix
    assert map_cb_expert_name("layers.7.mlp.experts.down_proj.cb_qweight") == \
        "layers.7.mlp.experts.w2_cb_qweight"


def test_map_cb_expert_name_excludes_non_experts():
    # The .experts. anchor (not just the leaf name) excludes shared_mlp / dense
    # MLP / router / attention — those must reach the ORIGINAL loader.
    for n in [
        "model.layers.72.mlp.shared_mlp.gate_proj.cb_qweight",   # shared expert
        "model.layers.72.mlp.shared_mlp.up_proj.cb_qweight",
        "model.layers.5.mlp.shared_mlp.gate_up_proj.cb_qweight",  # synth fused shared
        "model.layers.0.mlp.gate_proj.cb_qweight",               # dense L0
        "model.layers.0.mlp.down_proj.cb_qweight",               # dense L0 down
        "model.layers.1.mlp.router.gate.weight",                 # router
        "model.layers.1.self_attn.qkv_proj.cb_qweight",          # dense attn
        "model.layers.1.mlp.gate.expert_bias",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]:
        assert map_cb_expert_name(n) is None, n


def test_wrapper_routes_experts_and_delegates_rest():
    E, HID, INTER, BYTES = 4, 16, 8, 3

    class _FakeCausalLM:                      # fresh class -> fresh sentinel
        def __init__(self):
            self._params = {
                "model.layers.1.mlp.experts.w13_cb_qweight":
                    torch.zeros(E, 2 * INTER, BYTES, dtype=torch.uint8),
                "model.layers.1.mlp.experts.w2_cb_qweight":
                    torch.zeros(E, HID, BYTES, dtype=torch.uint8),
                "model.layers.1.mlp.experts.w13_weight_scale":
                    torch.zeros(E, 2 * INTER, dtype=torch.float32),
                "model.layers.1.mlp.experts.w2_weight_scale":
                    torch.zeros(E, HID, dtype=torch.float32),
                "model.layers.0.mlp.down_proj.cb_qweight":       # dense, delegated
                    torch.zeros(HID, BYTES, dtype=torch.uint8),
                "model.embed_tokens.weight": torch.zeros(10, HID),
            }
            self.delegated = []

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):     # ORIGINAL (stub stock loader)
            loaded = set()
            for name, w in weights:
                self.delegated.append(name)
                if name in self._params:
                    self._params[name].copy_(w)
                    loaded.add(name)
            return loaded

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    m = _FakeCausalLM()
    ckpt = [
        ("model.layers.1.mlp.experts.gate_up_proj.cb_qweight",
         torch.full((E, 2 * INTER, BYTES), 7, dtype=torch.uint8)),
        ("model.layers.1.mlp.experts.down_proj.cb_qweight",
         torch.full((E, HID, BYTES), 9, dtype=torch.uint8)),
        ("model.layers.1.mlp.experts.gate_up_proj.weight_scale",
         torch.full((E, 2 * INTER), 2.0)),
        ("model.layers.1.mlp.experts.down_proj.weight_scale",
         torch.full((E, HID), 3.0)),
        ("model.layers.0.mlp.down_proj.cb_qweight",
         torch.full((HID, BYTES), 5, dtype=torch.uint8)),
        ("model.embed_tokens.weight", torch.full((10, HID), 1.0)),
    ]
    loaded = m.load_weights(iter(ckpt))      # generator input (streaming semantics)

    # expert stacks copied into the registered fused params
    assert torch.all(m._params["model.layers.1.mlp.experts.w13_cb_qweight"] == 7)
    assert torch.all(m._params["model.layers.1.mlp.experts.w2_cb_qweight"] == 9)
    assert torch.allclose(m._params["model.layers.1.mlp.experts.w13_weight_scale"],
                          torch.tensor(2.0))
    assert torch.allclose(m._params["model.layers.1.mlp.experts.w2_weight_scale"],
                          torch.tensor(3.0))
    # dense / embedding delegated to the original and copied there
    assert torch.all(m._params["model.layers.0.mlp.down_proj.cb_qweight"] == 5)
    assert torch.allclose(m._params["model.embed_tokens.weight"], torch.tensor(1.0))
    # NO expert tensor leaked to the original loader (no double-load)
    assert not any(".experts." in n for n in m.delegated), m.delegated
    # loaded set = 4 mapped expert params (model. prefix) + 2 delegated
    assert loaded == {
        "model.layers.1.mlp.experts.w13_cb_qweight",
        "model.layers.1.mlp.experts.w2_cb_qweight",
        "model.layers.1.mlp.experts.w13_weight_scale",
        "model.layers.1.mlp.experts.w2_weight_scale",
        "model.layers.0.mlp.down_proj.cb_qweight",
        "model.embed_tokens.weight",
    }


def test_wrapper_defers_unmappable_expert_name():
    # A suffix match whose target param is absent (PP/EP-missing, or an MTP/spec
    # layer the original filters) must DEFER to the original, never hard-fail.
    class _FakeCausalLM:
        def __init__(self):
            self._params = {}            # no expert params registered here
            self.delegated = []

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):
            self.delegated = [n for n, _ in weights]
            return set()

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    m = _FakeCausalLM()
    m.load_weights(iter([
        ("model.layers.80.mlp.experts.gate_up_proj.cb_qweight",
         torch.zeros(2, 2, 1, dtype=torch.uint8)),
    ]))
    assert m.delegated == ["model.layers.80.mlp.experts.gate_up_proj.cb_qweight"]


def test_wrapper_shape_mismatch_raises():
    class _FakeCausalLM:
        def __init__(self):
            self._params = {
                "model.layers.1.mlp.experts.w13_cb_qweight":
                    torch.zeros(4, 6, 3, dtype=torch.uint8),
            }

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):
            for _ in weights:
                pass
            return set()

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    m = _FakeCausalLM()
    with pytest.raises(ValueError, match="contract violated"):
        m.load_weights(iter([
            ("model.layers.1.mlp.experts.gate_up_proj.cb_qweight",
             torch.zeros(4, 6, 99, dtype=torch.uint8)),   # wrong byte width
        ]))


def test_install_is_idempotent():
    class _FakeCausalLM:
        def load_weights(self, weights):
            return set()

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    wrapped = _FakeCausalLM.load_weights
    install_toplevel_cb_expert_loader(_FakeCausalLM)     # second call: no-op
    assert _FakeCausalLM.load_weights is wrapped
    assert _FakeCausalLM._pq_cb_wrapped is True
