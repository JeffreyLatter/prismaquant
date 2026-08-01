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
