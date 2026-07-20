"""Top-level stacked-CB expert loader shim.

Some MoE architectures load their experts at the **top-level** ``*ForCausalLM``
(equivalently, the top ``*Model``) ``load_weights`` via an
``expert_params_mapping`` keyed on *per-expert* checkpoint names
(``…experts.{eid}.gate_proj.``), and explicitly ``continue`` past
``mlp.experts`` in their ``stacked_params_mapping`` loop. Such models **never**
call the per-layer ``FusedMoE.load_weights`` — so the instance-level CB load
hook that ``PrismaQuantCBMoEMethod.create_weights`` installs on the FusedMoE
module (``moe.py`` ``_cb_load_weights``) is *dead code* for them.

HunYuan V3 (``HYV3ForCausalLM``) is exactly this shape. Our exporter writes
**stacked** CB expert tensors — one tensor per role holding all experts:

    model.layers.N.mlp.experts.gate_up_proj.cb_qweight   uint8 (E, 2·inter, bytes)
    model.layers.N.mlp.experts.down_proj.cb_qweight      uint8 (E, hidden,  bytes)
    model.layers.N.mlp.experts.gate_up_proj.weight_scale f32   (E, 2·inter)   # fp8 rungs only
    model.layers.N.mlp.experts.down_proj.weight_scale    f32   (E, hidden)    # fp8 rungs only

These match neither the arch's ``stacked_params_mapping`` (experts skipped) nor
its ``expert_params_mapping`` (which matches per-expert ``experts.{eid}.…``
names, not our fused+stacked ``experts.gate_up_proj.…``), so the arch's loader
falls to its final ``params_dict[name]`` and ``KeyError``s.

``PrismaQuantCBMoEMethod.create_weights`` registers these stacked tensors on the
FusedMoE module (path ``…experts``) verbatim as params ``w13_cb_qweight`` /
``w2_cb_qweight`` (+ fp8 ``w13_weight_scale`` / ``w2_weight_scale``) — SAME
shapes as the checkpoint tensors. So loading is a plain ``copy_`` with no
per-expert split and no transpose.

This module installs a thin wrapper on the top-level ``load_weights`` that
intercepts exactly those stacked-CB expert tensors, copies each into its
registered fused param, and delegates *every other* tensor (dense CB,
router.gate, shared_mlp, expert_bias, norms, embeddings, lm_head, attention)
unchanged to the original loader. One line per arch registers it
(``install_toplevel_cb_expert_loader(SomeForCausalLM)``); DSv4 and any other
top-level-expert-mapping MoE arch reuse it as-is.

Prefix note: we wrap the OUTERMOST class (e.g. ``HYV3ForCausalLM``), whose
incoming weight names and ``named_parameters()`` BOTH carry the ``model.``
prefix (the raw safetensors stream). The KeyError in the bug report
(``layers.1.mlp.experts.down_proj.cb_qweight``, no ``model.``) originates one
level down in ``HYV3Model.load_weights``, because ``AutoWeightsLoader`` strips
the ``model.`` prefix before delegating to the child. By intercepting at the top
level we never let those tensors reach that child, and prefix handling stays
self-consistent: both the incoming name and the mapped param name carry
``model.``. The mapping is a pure suffix rewrite, so it is prefix-agnostic
regardless; the ``params_dict`` membership check keeps it robust if a future
vLLM changes prefix handling (an unmapped name simply defers to the original).
"""
from __future__ import annotations

import torch

# Checkpoint expert-tensor suffix  ->  registered FusedMoE param suffix.
# The leading ``.experts.`` anchor is load-bearing: it excludes ``shared_mlp``
# and dense MLP projections that share the ``gate_up_proj`` / ``down_proj`` leaf
# names (e.g. ``…mlp.shared_mlp.gate_proj.cb_qweight``,
# ``…layers.0.mlp.down_proj.cb_qweight``), which must go to the ORIGINAL loader.
_CB_EXPERT_SUFFIX_MAP: dict[str, str] = {
    ".experts.gate_up_proj.cb_qweight": ".experts.w13_cb_qweight",
    ".experts.down_proj.cb_qweight": ".experts.w2_cb_qweight",
    ".experts.gate_up_proj.weight_scale": ".experts.w13_weight_scale",
    ".experts.down_proj.weight_scale": ".experts.w2_weight_scale",
}


def map_cb_expert_name(name: str) -> str | None:
    """Map a stacked-CB expert checkpoint tensor name to the FusedMoE param
    name that ``PrismaQuantCBMoEMethod.create_weights`` registers, preserving
    any module-nesting prefix (``model.`` etc.). Returns ``None`` when *name* is
    not a stacked-CB expert tensor (so the caller delegates it unchanged)."""
    for suffix, replacement in _CB_EXPERT_SUFFIX_MAP.items():
        if name.endswith(suffix):
            return name[: -len(suffix)] + replacement
    return None


def install_toplevel_cb_expert_loader(model_cls: type) -> None:
    """Idempotently wrap ``model_cls.load_weights`` so stacked-CB expert tensors
    load directly into the registered FusedMoE params, and everything else
    delegates to the original loader.

    Safe to call repeatedly (guarded by a ``_pq_cb_wrapped`` class sentinel) and
    safe if the model has no CB experts at serve time (the wrapper only fires on
    matching names; all others pass straight through)."""
    if getattr(model_cls, "_pq_cb_wrapped", False):
        return
    orig_load_weights = model_cls.load_weights

    def load_weights(self, weights):  # noqa: ANN001, ANN202
        # named_parameters() here carries the same module-nesting prefix as the
        # incoming checkpoint names (both ``model.…`` at the top level), so the
        # suffix-rewritten target is a direct key.
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        def _passthrough():
            # A generator (not a materialized list): only one checkpoint tensor
            # is live at a time, preserving the streaming/mmap semantics the
            # original loader relies on for a 100 GB+ model. CB expert tensors
            # are copied inline as a side effect and recorded in ``loaded``;
            # every other tensor is yielded on to the original loader.
            for name, w in weights:
                mapped = map_cb_expert_name(name)
                if mapped is not None:
                    param = params_dict.get(mapped)
                    if param is not None:
                        if tuple(param.shape) != tuple(w.shape):
                            raise ValueError(
                                f"prismaquant CB expert '{name}' -> '{mapped}': "
                                f"checkpoint shape {tuple(w.shape)} != param "
                                f"shape {tuple(param.shape)} — stacked "
                                "(E, out, bytes) contract violated")
                        param.data.copy_(w.to(param.dtype))
                        loaded.add(mapped)
                        continue
                    # Suffix matched but the target param is absent on this rank:
                    # PP/EP-missing expert, or an MTP/spec layer the original's
                    # own filter drops. Defer to the original loader, which knows
                    # how to skip it — never a hard failure here.
                yield name, w

        loaded |= orig_load_weights(self, _passthrough())
        return loaded

    model_cls.load_weights = load_weights
    model_cls._pq_cb_wrapped = True
