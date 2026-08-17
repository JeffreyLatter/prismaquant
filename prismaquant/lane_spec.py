"""`LaneSpec` — one lane-uniform ship gate across native / CB / GGUF.

Re-vet **R16** (`docs/audits/architecture_re-vet_2026-07-30.md`), which closes
the measurement half of debt **D26**.

**The asymmetry was wiring, not capability.** `validate_quantized_model.py` is
already runtime-agnostic — it drives an OpenAI-compatible endpoint over HTTP
(`--base-url`, `--model-name`) and knows nothing about compressed-tensors;
`grep 'gguf\\|nvfp4_cb'` across every `validate_*.py` returns zero hits. What
was missing is a *declaration* of what each lane's serve command, endpoint,
gate set and KL evaluator are, so "the bar" is defined once instead of being
native's bar by default and nothing elsewhere.

Same idiom as `serving_profile_specs/`: JSON declarations plus a dataclass with
`from_dict`, so a new lane is a data file rather than a branch.

**Gates are ADVISORY.** They are *recorded* — each maps to a `shipcard.json`
slot, and the shipcard is what refuses (R13) — but nothing in this module fails
a run. The verdict's open half was whether gates should become **blocking**,
which changes every lane's run; that is deferred to Robert and is recorded as
such in the re-vet outcome. `LaneSpec.advisory_gates` is `True` for every lane
and a test pins it, so a future flip is a deliberate edit rather than a drift.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "prismaquant.lane_spec.v1"

_SPEC_DIR = Path(__file__).resolve().parent / "lane_specs"


@dataclass(frozen=True)
class LaneEndpoint:
    """How a served artifact of this lane is talked to."""

    kind: str                       # openai | llama_server | none
    base_url: str | None = None
    health_path: str | None = None
    metrics_path: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneEndpoint":
        return cls(
            kind=str(payload["kind"]),
            base_url=_opt_str(payload.get("base_url")),
            health_path=_opt_str(payload.get("health_path")),
            metrics_path=_opt_str(payload.get("metrics_path")),
        )


@dataclass(frozen=True)
class LaneGate:
    """One numeric gate, and the shipcard slot its record closes."""

    id: str
    runner: str
    shipcard_slot: str | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneGate":
        return cls(
            id=str(payload["id"]),
            runner=str(payload["runner"]),
            shipcard_slot=_opt_str(payload.get("shipcard_slot")),
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class LaneKLEvaluator:
    """The lane's held-out KL evaluator, behind the `validate_assignments_kl`
    interface: `(mean, per_sequence, stats)` with the gold lane's key names."""

    kind: str                       # validate_assignments_kl | llama_perplexity
    entrypoint: str
    note: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneKLEvaluator":
        return cls(
            kind=str(payload["kind"]),
            entrypoint=str(payload["entrypoint"]),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class LaneActivationContract:
    """Which formats' activation quantization this lane's runtime EXECUTES.

    The format registry declares NVFP4 a W4A4 format. That is a fact about the
    FORMAT, not about the LANE. Gridbook's CB runtime decodes to BF16 and runs
    a BF16 GEMM -- what its own docstring calls "the exact native BF16 bridge"
    -- unless a fused activation mode is explicitly selected by a
    process-global env selector. Every gate and gold serve on the nvfp4_cb lane
    leaves those selectors unset, so an NVFP4_CB unit's activations are never
    quantized there and its A-side cost is exactly zero.

    Pricing an A-side the runtime does not execute is a CURRENCY error, not a
    conservative overestimate. It makes a format look more expensive than it
    is, and the DP then spends real weight bytes escaping a cost of zero: on
    DSv4-Flash at 87.403 GB it promoted 2,307 units to FP8_CB and funded that
    by dropping the bulk of the model from codebook rung K16 to K12 (four fewer
    index bits on ~19k units). Discovered 2026-08-17; the same mispricing is on
    the Qwen3.8-27B CB-A allocation, which used `cost_aura_anchored_aqua.pkl`
    while serving with the same selectors unset.

    ``executes`` is the authority the A-side pricing must intersect with. It is
    a set of **glob patterns** over format names, because the answer is per
    FAMILY and the rungs within a family are open-ended: the CB lane bridges
    every ``NVFP4_CB_K*`` but genuinely serves every ``FP8_CB_K*`` as W8A8
    (`gridbook/linear.py` feeds quantized ``xq`` with per-token dynamic scales
    into ``native_cutlass_scaled_mm``; `moe.py` declares
    ``_FP8_GROUPED_CONTRACT = "fp8_per_token_dynamic"``). Listing rungs
    explicitly would silently under-declare the day a new one is added, which
    is the same silent-default failure this class exists to remove.

    An empty set is a meaningful answer -- not a missing declaration.
    """

    executes: frozenset[str]
    rationale: str
    selectors_must_be_unset: tuple[str, ...] = ()

    def matches(self, format_name: str) -> bool:
        """Does this lane execute ``format_name``'s activation quantization?"""
        return any(fnmatch.fnmatchcase(format_name, pattern)
                   for pattern in self.executes)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LaneActivationContract":
        if "executes" not in payload:
            raise ValueError(
                "served_activation_quantization must state `executes` "
                "explicitly; an absent list is not an empty list, and "
                "guessing it is the bug this field exists to prevent")
        return cls(
            executes=frozenset(str(f) for f in payload["executes"]),
            rationale=str(payload.get("rationale", "")),
            selectors_must_be_unset=tuple(
                str(s) for s in payload.get("selectors_must_be_unset", ())),
        )


@dataclass(frozen=True)
class LaneSpec:
    id: str
    export_container: str
    runtime: str
    description: str
    endpoint: LaneEndpoint
    kl_evaluator: LaneKLEvaluator
    serve_scripts: tuple[str, ...] = ()
    serve_command: tuple[str, ...] = ()
    gates: tuple[LaneGate, ...] = ()
    serving_profiles: tuple[str, ...] = ()
    advisory_gates: bool = True
    notes: tuple[str, ...] = field(default=())
    served_activation_quantization: LaneActivationContract | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneSpec":
        schema = str(payload.get("schema", SCHEMA))
        if schema != SCHEMA:
            raise ValueError(f"unknown lane spec schema {schema!r}")
        serve = payload.get("serve", {}) or {}
        return cls(
            id=str(payload["id"]),
            export_container=str(payload["export_container"]),
            runtime=str(payload["runtime"]),
            description=str(payload.get("description", "")),
            endpoint=LaneEndpoint.from_dict(payload["endpoint"]),
            kl_evaluator=LaneKLEvaluator.from_dict(payload["kl_evaluator"]),
            serve_scripts=tuple(str(s) for s in serve.get("scripts", ())),
            serve_command=tuple(str(s) for s in serve.get("command", ())),
            gates=tuple(
                LaneGate.from_dict(g) for g in payload.get("gates", ())),
            serving_profiles=tuple(
                str(p) for p in payload.get("serving_profiles", ())),
            advisory_gates=bool(payload.get("advisory_gates", True)),
            notes=tuple(str(n) for n in payload.get("notes", ())),
            served_activation_quantization=(
                LaneActivationContract.from_dict(
                    payload["served_activation_quantization"])
                if payload.get("served_activation_quantization") is not None
                else None),
        )

    def gate(self, gate_id: str) -> LaneGate | None:
        for g in self.gates:
            if g.id == gate_id:
                return g
        return None

    def shipcard_slots(self) -> tuple[str, ...]:
        return tuple(g.shipcard_slot for g in self.gates if g.shipcard_slot)

    def render_serve_command(self, **values: str) -> tuple[str, ...]:
        """Substitute `${NAME}` placeholders in the declared serve command.

        A missing placeholder is a `KeyError` naming it — a serve command with
        an unresolved `${MODEL}` is exactly the class of mistake that produces
        a container serving the wrong artifact.
        """
        from string import Template

        out: list[str] = []
        for token in self.serve_command:
            out.append(Template(token).substitute(values))
        return tuple(out)


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def lane_spec_names() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in _SPEC_DIR.glob("*.json")))


@lru_cache(maxsize=None)
def load_lane_spec(lane_id: str) -> LaneSpec:
    path = _SPEC_DIR / f"{lane_id}.json"
    if not path.is_file():
        raise KeyError(
            f"unknown lane {lane_id!r}; known lanes: {lane_spec_names()}")
    return LaneSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def lane_spec_for_container(export_container: str) -> LaneSpec:
    """The lane declaration for an `EXPORT_CONTAINER` value."""
    want = str(export_container).strip()
    for name in lane_spec_names():
        spec = load_lane_spec(name)
        if spec.export_container == want:
            return spec
    raise KeyError(
        f"no lane spec declares export_container={want!r}; "
        f"known: {[load_lane_spec(n).export_container for n in lane_spec_names()]}"
    )


def all_lane_specs() -> tuple[LaneSpec, ...]:
    return tuple(load_lane_spec(name) for name in lane_spec_names())


def lane_gate_report(spec: LaneSpec, shipcard: Mapping[str, Any] | None = None
                     ) -> list[dict[str, Any]]:
    """Advisory status of every declared gate against a shipcard payload.

    Returns one row per gate: `{gate, runner, shipcard_slot, filled,
    advisory}`. Nothing here refuses — `python -m prismaquant.shipcard_cli verify` is the
    refusal, and this is the lane-uniform view of what it will refuse on.
    """
    slots: Mapping[str, Any] = {}
    if shipcard:
        slots = shipcard.get("slots", {}) or {}
    rows: list[dict[str, Any]] = []
    for gate in spec.gates:
        filled = (
            gate.shipcard_slot is not None
            and slots.get(gate.shipcard_slot) is not None
        )
        rows.append({
            "gate": gate.id,
            "runner": gate.runner,
            "shipcard_slot": gate.shipcard_slot,
            "filled": bool(filled),
            "advisory": bool(spec.advisory_gates),
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Show a lane's declared ship gate")
    p.add_argument("--lane", default=None, help="lane id (default: all)")
    p.add_argument("--export-container", default=None)
    p.add_argument("--shipcard", default=None,
                   help="shipcard.json to report gate fill against")
    args = p.parse_args(argv)

    if args.export_container:
        specs = (lane_spec_for_container(args.export_container),)
    elif args.lane:
        specs = (load_lane_spec(args.lane),)
    else:
        specs = all_lane_specs()

    card = None
    if args.shipcard:
        card = json.loads(Path(args.shipcard).read_text(encoding="utf-8"))

    for spec in specs:
        print(f"[lane] {spec.id}: container={spec.export_container} "
              f"runtime={spec.runtime} endpoint={spec.endpoint.kind} "
              f"kl={spec.kl_evaluator.kind} "
              f"gates={'ADVISORY' if spec.advisory_gates else 'BLOCKING'}")
        for row in lane_gate_report(spec, card):
            state = "filled" if row["filled"] else "UNFILLED"
            print(f"    {row['gate']:<28} {row['runner']:<44} "
                  f"{row['shipcard_slot'] or '-':<20} {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
