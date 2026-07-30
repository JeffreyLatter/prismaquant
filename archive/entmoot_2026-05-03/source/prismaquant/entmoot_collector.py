"""Activation/router sketch collection for Entmoot.

The collector records bounded per-expert samples from MoE calibration
forwards.  Qwen3.5/3.6 packed experts are supported first because they are
the initial Entmoot validation target and already use a compact packed-3D
expert module (`gate_up_proj` + `down_proj`).
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.observers.expert_saliency import saliency_from_packed_moe


@dataclass(frozen=True)
class ExpertSketchStats:
    expert_id: int
    samples: int
    seen: int
    routed_mass: float

    def to_dict(self) -> dict:
        return {
            "expert_id": self.expert_id,
            "samples": self.samples,
            "seen": self.seen,
            "routed_mass": self.routed_mass,
        }


class ReservoirTensorBuffer:
    """Reservoir-sample aligned tensors for one expert."""

    def __init__(
        self,
        max_samples: int,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = int(max_samples)
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.hidden: list[torch.Tensor] = []
        self.output: list[torch.Tensor] = []
        self.route_weight: list[torch.Tensor] = []
        self.seen = 0
        self.routed_mass = 0.0

    @property
    def samples(self) -> int:
        return len(self.output)

    def add(
        self,
        hidden: torch.Tensor,
        output: torch.Tensor,
        route_weight: torch.Tensor,
    ) -> None:
        hidden = hidden.detach().to("cpu", dtype=torch.float32)
        output = output.detach().to("cpu", dtype=torch.float32)
        route_weight = route_weight.detach().to("cpu", dtype=torch.float32).reshape(-1)
        if hidden.ndim != 2 or output.ndim != 2:
            raise ValueError("hidden and output samples must be 2D")
        if hidden.shape[0] != output.shape[0] or hidden.shape[0] != route_weight.shape[0]:
            raise ValueError("hidden/output/route_weight row counts must match")
        for i in range(int(output.shape[0])):
            self.seen += 1
            self.routed_mass += float(route_weight[i])
            row = self.seen - 1
            if len(self.output) < self.max_samples:
                self.hidden.append(hidden[i].clone())
                self.output.append(output[i].clone())
                self.route_weight.append(route_weight[i].reshape(()).clone())
                continue
            j = int(torch.randint(
                low=0,
                high=self.seen,
                size=(1,),
                generator=self.generator,
            ).item())
            if j < self.max_samples:
                self.hidden[j] = hidden[i].clone()
                self.output[j] = output[i].clone()
                self.route_weight[j] = route_weight[i].reshape(()).clone()

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.output:
            return (
                torch.empty(0, 0, dtype=torch.float32),
                torch.empty(0, 0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )
        return (
            torch.stack(self.hidden, dim=0),
            torch.stack(self.output, dim=0),
            torch.stack(self.route_weight, dim=0).reshape(-1),
        )

    def state_dict(self) -> dict:
        hidden, output, route_weight = self.stacked()
        return {
            "max_samples": self.max_samples,
            "seen": self.seen,
            "routed_mass": self.routed_mass,
            "hidden": hidden,
            "output": output,
            "route_weight": route_weight,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReservoirTensorBuffer":
        buf = cls(int(state["max_samples"]))
        buf.seen = int(state.get("seen", 0))
        buf.routed_mass = float(state.get("routed_mass", 0.0))
        hidden = state.get("hidden", torch.empty(0, 0))
        output = state.get("output", torch.empty(0, 0))
        route_weight = state.get("route_weight", torch.empty(0))
        if isinstance(hidden, torch.Tensor) and hidden.numel() > 0:
            buf.hidden = [row.detach().cpu().clone() for row in hidden]
        if isinstance(output, torch.Tensor) and output.numel() > 0:
            buf.output = [row.detach().cpu().clone() for row in output]
        if isinstance(route_weight, torch.Tensor) and route_weight.numel() > 0:
            buf.route_weight = [
                row.detach().cpu().reshape(()).clone()
                for row in route_weight.reshape(-1)
            ]
        return buf


class LayerSketchBuffer:
    """Bounded Entmoot sketches for one MoE layer/router."""

    def __init__(
        self,
        router_qname: str,
        *,
        num_experts: int,
        max_samples_per_expert: int = 256,
        seed: int = 0,
    ) -> None:
        self.router_qname = str(router_qname)
        self.num_experts = int(num_experts)
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.max_samples_per_expert = int(max_samples_per_expert)
        self.experts = {
            eid: ReservoirTensorBuffer(
                self.max_samples_per_expert,
                seed=int(seed) + eid,
            )
            for eid in range(self.num_experts)
        }
        self.total_tokens = 0

    def add_expert_batch(
        self,
        expert_id: int,
        hidden: torch.Tensor,
        output: torch.Tensor,
        route_weight: torch.Tensor,
    ) -> None:
        eid = int(expert_id)
        if eid < 0 or eid >= self.num_experts:
            raise ValueError(f"expert_id {eid} outside [0, {self.num_experts})")
        self.experts[eid].add(hidden, output, route_weight)

    def add_router_batch(self, hidden: torch.Tensor) -> None:
        if hidden.ndim < 2:
            return
        self.total_tokens += int(hidden.reshape(-1, hidden.shape[-1]).shape[0])

    def stats(self) -> list[ExpertSketchStats]:
        return [
            ExpertSketchStats(
                expert_id=eid,
                samples=buf.samples,
                seen=buf.seen,
                routed_mass=buf.routed_mass,
            )
            for eid, buf in sorted(self.experts.items())
        ]

    def routed_mass(self) -> dict[int, float]:
        return {eid: buf.routed_mass for eid, buf in sorted(self.experts.items())}

    def output_feature_matrix(
        self,
        *,
        normalize: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        """Return `[E, H]` router-weighted mean output features."""

        rows: list[torch.Tensor] = []
        hidden_dim = None
        for eid in range(self.num_experts):
            _hidden, output, weight = self.experts[eid].stacked()
            if output.numel() == 0:
                if hidden_dim is None:
                    rows.append(torch.empty(0, dtype=torch.float64))
                else:
                    rows.append(torch.zeros(hidden_dim, dtype=torch.float64))
                continue
            hidden_dim = int(output.shape[-1])
            w = weight.to(torch.float64).clamp_min(0.0)
            if float(w.sum()) <= 0.0:
                w = torch.ones_like(w)
            feat = (output.to(torch.float64) * w[:, None]).sum(dim=0) / w.sum()
            rows.append(feat)

        if hidden_dim is None:
            raise ValueError("no output samples collected")
        rows = [
            r if r.numel() else torch.zeros(hidden_dim, dtype=torch.float64)
            for r in rows
        ]
        X = torch.stack(rows, dim=0)
        if normalize:
            X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return list(range(self.num_experts)), X

    def to_summary(self) -> dict:
        return {
            "router_qname": self.router_qname,
            "num_experts": self.num_experts,
            "max_samples_per_expert": self.max_samples_per_expert,
            "total_tokens": self.total_tokens,
            "experts": [s.to_dict() for s in self.stats()],
        }

    def state_dict(self) -> dict:
        return {
            "router_qname": self.router_qname,
            "num_experts": self.num_experts,
            "max_samples_per_expert": self.max_samples_per_expert,
            "total_tokens": self.total_tokens,
            "experts": {
                str(eid): buf.state_dict()
                for eid, buf in sorted(self.experts.items())
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LayerSketchBuffer":
        layer = cls(
            str(state["router_qname"]),
            num_experts=int(state["num_experts"]),
            max_samples_per_expert=int(state["max_samples_per_expert"]),
        )
        layer.total_tokens = int(state.get("total_tokens", 0))
        experts = state.get("experts", {})
        for eid_s, buf_state in experts.items():
            eid = int(eid_s)
            if eid in layer.experts:
                layer.experts[eid] = ReservoirTensorBuffer.from_state_dict(buf_state)
        return layer


class EntmootActivationCollector:
    """Install Qwen packed-MoE hooks and collect Entmoot layer sketches."""

    def __init__(
        self,
        model: nn.Module,
        *,
        packed_moe_blocks: list[dict] | None = None,
        max_samples_per_expert: int = 256,
        seed: int = 0,
    ) -> None:
        self.max_samples_per_expert = int(max_samples_per_expert)
        self.seed = int(seed)
        self.layers: dict[str, LayerSketchBuffer] = {}
        self._patched: list[nn.Module] = []
        entries = packed_moe_blocks if packed_moe_blocks is not None else saliency_from_packed_moe(model)
        for idx, entry in enumerate(entries):
            self._install_qwen_packed_patch(model, entry, seed=self.seed + idx * 1009)

    def remove_hooks(self) -> None:
        for mod in list(self._patched):
            original = getattr(mod, "_pq_entmoot_original_forward", None)
            if original is not None:
                mod.forward = original
            for attr in (
                "_pq_entmoot_patched",
                "_pq_entmoot_collector",
                "_pq_entmoot_router",
                "_pq_entmoot_original_forward",
            ):
                if hasattr(mod, attr):
                    delattr(mod, attr)
        self._patched.clear()

    def summaries(self) -> dict[str, dict]:
        return {router: buf.to_summary() for router, buf in sorted(self.layers.items())}

    def state_dict(self) -> dict:
        return {
            "format": "entmoot_activation_collector_v1",
            "max_samples_per_expert": self.max_samples_per_expert,
            "layers": {
                router: buf.state_dict()
                for router, buf in sorted(self.layers.items())
            },
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    def _install_qwen_packed_patch(
        self,
        model: nn.Module,
        entry: Mapping[str, Any],
        *,
        seed: int,
    ) -> bool:
        router_qname = entry.get("router_qname")
        experts_qname = entry.get("experts_qname")
        num_experts = entry.get("num_experts")
        if not (router_qname and experts_qname and num_experts):
            return False
        try:
            experts_mod = model.get_submodule(str(experts_qname))
        except AttributeError:
            return False
        gate_up = getattr(experts_mod, "gate_up_proj", None)
        down = getattr(experts_mod, "down_proj", None)
        act_fn = getattr(experts_mod, "act_fn", None)
        if not (isinstance(gate_up, nn.Parameter) and gate_up.dim() == 3):
            return False
        if not (isinstance(down, nn.Parameter) and down.dim() == 3):
            return False
        if act_fn is None or hasattr(experts_mod, "_pq_entmoot_patched"):
            return False

        self.layers[str(router_qname)] = LayerSketchBuffer(
            str(router_qname),
            num_experts=int(num_experts),
            max_samples_per_expert=self.max_samples_per_expert,
            seed=seed,
        )
        experts_mod._pq_entmoot_patched = True
        experts_mod._pq_entmoot_collector = self
        experts_mod._pq_entmoot_router = str(router_qname)
        experts_mod._pq_entmoot_original_forward = experts_mod.forward
        experts_mod.forward = types.MethodType(
            _qwen3_5_moe_experts_entmoot_forward,
            experts_mod,
        )
        self._patched.append(experts_mod)
        return True


def _qwen3_5_moe_experts_entmoot_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Qwen3.5/3.6 packed-experts forward with sketch collection."""

    collector = getattr(self, "_pq_entmoot_collector", None)
    router_qname = getattr(self, "_pq_entmoot_router", None)
    layer_buf = None
    if collector is not None and router_qname is not None:
        layer_buf = collector.layers.get(router_qname)
        if layer_buf is not None:
            layer_buf.add_router_batch(hidden_states)

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit_ints = (
            torch.greater(expert_mask.sum(dim=(-1, -2)), 0)
            .nonzero().flatten().tolist()
        )

    for e_int in expert_hit_ints:
        if e_int == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e_int])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[e_int]).chunk(2, dim=-1)
        inter = self.act_fn(gate) * up
        expert_out = F.linear(inter, self.down_proj[e_int])

        if layer_buf is not None and expert_out.numel() > 0:
            route_weight = top_k_weights[token_idx, top_k_pos]
            layer_buf.add_expert_batch(e_int, current_state, expert_out, route_weight)

        routed = expert_out * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, routed.to(final_hidden_states.dtype),
        )

    return final_hidden_states


def load_collector_state(path: str | Path) -> dict[str, LayerSketchBuffer]:
    """Load a saved Entmoot collector artifact."""

    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("collector artifact is not a mapping")
    if payload.get("format") != "entmoot_activation_collector_v1":
        raise ValueError("unsupported collector artifact format")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError("collector artifact missing layers")
    return {
        str(router): LayerSketchBuffer.from_state_dict(state)
        for router, state in layers.items()
    }

