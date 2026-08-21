"""Discover every weight-bearing computation in a model by traversal.

This module is the discovery walker of ``docs/design/model_coverage_ledgers.md``
("walk what runs, not what's declared") and the implementation of the R5 design
contract. It exists to kill one failure class: a parameter that feeds a matmul
through a module class the pipeline's own enumeration skips, and therefore never
becomes an allocator decision. The motivating instance is DeepSeek-V4's
``attn.wo_a`` — consumed by a grouped einsum/bmm, 17.9% of decode read traffic,
shipped by omission.

The walk has one root pair and one output:

* **Root A — the module tree.** Every named parameter and buffer becomes a
  :class:`WalkNode`, enumerated with ``remove_duplicate=False`` so tied weights
  (an embedding that is also the logits projection) keep every name.
* **Root B — one traced forward.** A :class:`WeightUseInterceptor`
  (``TorchFunctionMode``) records every matmul-family call the model executes,
  together with the parameters that feed it. Each resolved
  (parameter, op) pair becomes a :class:`WalkEdge`. The trace normally runs
  under ``FakeTensorMode`` on a meta-loaded model, so intake costs no GPU and
  no weight I/O; the interceptor is agnostic to which mode hosts it
  (``execution="real"`` runs the same interceptor over a real CPU forward).

Every node must then be **claimed** by exactly one disposition —
``decide`` (enters the allocator's domain), ``pin(reason)`` (held at source
precision on purpose), or ``exclude(reason)`` (outside the artifact's scope).
Claims come from :class:`ClaimRule` lists; the model profile supplies them
(``ModelProfile.walk_claim_rules()``). A matmul-fed node with no claim
**fails the walk**, with the node named and the op cited. That is the whole
point: ``wo_a`` claimed as ``pin(probe cannot price grouped operands yet)`` on
day one is a known debt with a name; ``wo_a`` absent is what shipped.

Operand resolution
------------------
An operand maps to its parameter by **storage identity**: the walker indexes
the ``UntypedStorage`` of every named parameter and buffer, so a view or a
per-expert slice maps to its parent parameter.

Two measured facts shape the implementation (torch 2.11, 2026-08-21):

* On meta tensors ``untyped_storage().data_ptr()`` is degenerate — every
  storage reports ``0`` — so the identity key is the ``StorageImpl`` address
  (``untyped_storage()._cdata``; ``data_ptr()`` is the fallback when the
  private field is absent). This implements the contract's storage-identity
  choice; the contract's literal ``data_ptr()`` spelling cannot distinguish
  meta storages.
* Under ``FakeTensorMode``, the *output* of a view op on a non-fake parameter
  is a fresh ``FakeTensor`` whose storage does not alias the parameter's. The
  view call itself is still visible at ``__torch_function__`` level, so the
  interceptor propagates parameter identity through a fixed allowlist of
  alias and cast ops (``view``, ``__getitem__``, ``transpose``, ``to``,
  ``contiguous``, …) and records the hop chain on the edge (``via``).

An operand that resolves to nothing — not a named tensor, not a model input,
not a tensor computed during the traced forward — is **unresolved**. An
unresolved floating-point operand in a multiplicand position is a walk
failure: it means a weight this pipeline cannot name (a parameter that was
``.to()``'d or reconstructed at init time loses storage identity and lands
here, reported rather than misattributed). Additive operands (``F.linear``'s
bias, the first argument of ``addmm``/``baddbmm``) are recorded on edges with
``role="additive"`` but are exempt from both requirements: a bias is not a
GEMM multiplicand, so it is not a weight the allocator prices.

``F.scaled_dot_product_attention`` is deliberately not captured (no weights
among its operands). ``F.embedding`` produces no edge, but its weight is
recorded as consumed (:class:`EmbeddingUse`) and still requires a claim from
root A.

Honest limits
-------------
* The trace discovers what the traced forward executes. A module the trace
  never runs is still discovered by root A and dispositioned there;
  ``trace_coverage`` records which modules executed so the gap is visible,
  not silent. Container modules (``ModuleList``, ``ModuleDict``, parameter
  containers) never execute a forward by construction and are listed
  separately.
* Identity propagates through alias and cast ops only. A weight that reaches
  a matmul through arithmetic (a dequant pattern such as
  ``weight.to(x.dtype) * scale``) is an intermediate to this walker; the
  multiply's output resolves to "computed during forward" and produces no
  edge. Extending provenance through elementwise ops is deliberately out of
  scope — the walk's job is discovery, not dataflow analysis — and the
  parameter itself still requires a claim if any op consumes it directly.
* Data-dependent control flow executes shape-wise under fake tensors;
  all-expert participation on a fake-traced MoE is the desired discovery, but
  an op that requires concrete values (``.item()``, ``nonzero``) stops the
  fake trace. That is what ``execution="real"`` exists for.

Intake for a new architecture
-----------------------------
1. Meta-load the model (``with torch.device("meta"): AutoModel...``).
2. ``walk_model(model, claim_rules=profile.walk_claim_rules())``.
3. Read the failure list. Every named parameter is either a real allocator
   unit the profile must let through (``decide``), a known debt to pin with a
   reason, or an exclusion to declare. Do not silence a failure without
   writing the reason down — the reasons land on the shipcard.

This module imports only torch and the standard library, so it can wrap any
torch model; the prismaquant-specific claim policy lives on the model profile.
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

__all__ = [
    "Claim",
    "ClaimRule",
    "EmbeddingUse",
    "TraceCoverage",
    "UnresolvedOperand",
    "WalkEdge",
    "WalkError",
    "WalkFailure",
    "WalkNode",
    "WalkResult",
    "WeightUseInterceptor",
    "walk_model",
]

DISPOSITIONS = ("decide", "pin", "exclude")

# ---------------------------------------------------------------------------
# Op tables
# ---------------------------------------------------------------------------

# Matmul-family capture set, keyed by callable identity (the object
# `__torch_function__` receives), with the positions of additive (non-
# multiplicand) tensor arguments. `F.scaled_dot_product_attention` is
# excluded on purpose: none of its operands are weights.
_ADDITIVE_NONE: frozenset[int] = frozenset()
_MATMUL_FUNCS: dict[Any, tuple[str, frozenset[int]]] = {
    F.linear: ("linear", frozenset({2})),           # (input, weight, bias)
    torch.matmul: ("matmul", _ADDITIVE_NONE),
    torch.Tensor.matmul: ("matmul", _ADDITIVE_NONE),
    torch.Tensor.__matmul__: ("matmul", _ADDITIVE_NONE),
    torch.Tensor.__rmatmul__: ("matmul", _ADDITIVE_NONE),
    torch.mm: ("mm", _ADDITIVE_NONE),
    torch.bmm: ("bmm", _ADDITIVE_NONE),
    torch.mv: ("mv", _ADDITIVE_NONE),
    torch.addmm: ("addmm", frozenset({0})),
    torch.addbmm: ("addbmm", frozenset({0})),
    torch.addmv: ("addmv", frozenset({0})),
    torch.baddbmm: ("baddbmm", frozenset({0})),
    torch.einsum: ("einsum", _ADDITIVE_NONE),
    torch.tensordot: ("tensordot", _ADDITIVE_NONE),
}
# Keyword spellings of additive positions.
_ADDITIVE_KWARGS: dict[str, frozenset[str]] = {
    "linear": frozenset({"bias"}),
    "addmm": frozenset({"input"}),
    "addbmm": frozenset({"input"}),
    "addmv": frozenset({"input"}),
    "baddbmm": frozenset({"input"}),
}

_EMBEDDING_FUNCS = {F.embedding, F.embedding_bag}

# Ops through which parameter identity propagates: the output *is* the
# parameter's payload under an alias, a reshape, or a value-preserving cast.
# Matched by `__name__` because most are bound methods with many callable
# spellings. Arithmetic is deliberately absent (see "Honest limits").
_ALIAS_OP_NAMES = frozenset({
    "__getitem__", "alias", "as_strided", "bfloat16", "chunk", "clone",
    "contiguous", "detach", "double", "expand", "expand_as", "flatten",
    "float", "half", "movedim", "moveaxis", "narrow", "permute", "ravel",
    "reshape", "select", "split", "squeeze", "swapaxes", "swapdims", "t",
    "to", "transpose", "type", "type_as", "unbind", "unflatten", "unsqueeze",
    "view", "view_as",
})

_CONTAINER_CLASSES = (
    nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict,
)


def _storage_key(tensor: torch.Tensor) -> int | None:
    """Identity key of a tensor's underlying storage.

    Uses the ``StorageImpl`` address (``_cdata``): views and slices of one
    parameter share it, distinct parameters differ, and — unlike
    ``data_ptr()``, which is 0 for every meta storage — it stays unique on
    the meta device. Falls back to ``data_ptr()`` if the private field ever
    disappears; returns None for tensors without accessible storage.
    """
    try:
        storage = tensor.untyped_storage()
    except Exception:
        return None
    cdata = getattr(storage, "_cdata", None)
    if cdata is not None:
        return int(cdata)
    return int(storage.data_ptr())


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


# ---------------------------------------------------------------------------
# Output dataclasses (all JSON-serializable through `to_json_dict`)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WalkNode:
    """One named parameter or buffer from the module tree (root A)."""

    name: str
    kind: str                       # "parameter" | "buffer"
    persistent: bool                # False only for non-persistent buffers
    shape: tuple[int, ...]
    dtype: str
    stored_bytes: int
    owner_module: str               # qualified name of the owning module
    module_class: str               # class name of the owning module
    module_class_mro: tuple[str, ...]  # class names, most-derived first
    aliases: tuple[str, ...]        # every OTHER name sharing this storage

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["shape"] = list(self.shape)
        d["module_class_mro"] = list(self.module_class_mro)
        d["aliases"] = list(self.aliases)
        return d


@dataclasses.dataclass(frozen=True)
class WalkEdge:
    """One (parameter, matmul-family op) consumption discovered by the trace
    (root B)."""

    param: str                      # primary node name
    param_aliases: tuple[str, ...]  # tied names sharing the storage
    op: str                         # "linear" | "matmul" | "einsum" | ...
    equation: str | None            # einsum equation, when the op has one
    role: str                       # "multiplicand" | "additive"
    operand_index: int              # position among the op's tensor operands
    operand_shape: tuple[int, ...]  # shape as consumed (view/slice shape)
    operand_dtype: str
    operand_shapes: tuple[tuple[int, ...], ...]  # every tensor operand's shape
    stored_bytes: int               # bytes of the full parameter node
    module: str                     # module executing the op ("" = root)
    via: tuple[str, ...]            # alias/cast hops from parameter to operand
    calls: int = 1                  # identical consumptions in the trace

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["param_aliases"] = list(self.param_aliases)
        d["operand_shape"] = list(self.operand_shape)
        d["operand_shapes"] = [list(s) for s in self.operand_shapes]
        d["via"] = list(self.via)
        return d


@dataclasses.dataclass(frozen=True)
class EmbeddingUse:
    """An `F.embedding` consumption: no edge, but the weight still requires a
    claim from root A."""

    param: str
    param_aliases: tuple[str, ...]
    module: str

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["param_aliases"] = list(self.param_aliases)
        return d


@dataclasses.dataclass(frozen=True)
class UnresolvedOperand:
    """A matmul operand that resolves to nothing the walk can name."""

    op: str
    equation: str | None
    module: str
    operand_index: int
    operand_shape: tuple[int, ...]
    operand_dtype: str
    role: str
    is_floating: bool

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["operand_shape"] = list(self.operand_shape)
        return d


@dataclasses.dataclass(frozen=True)
class Claim:
    """The disposition of one node, with the reason it carries."""

    disposition: str                # "decide" | "pin" | "exclude"
    reason: str
    rule_index: int                 # index into the applied rule list

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class WalkFailure:
    """One reason the walk fails: an unclaimed matmul-fed node, or an
    unresolved floating multiplicand."""

    kind: str                       # "unclaimed" | "unresolved"
    node: str | None                # the parameter/buffer name, when known
    op: str
    equation: str | None
    module: str
    detail: str

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TraceCoverage:
    """Which modules the traced forward executed. `containers` never execute
    a forward by construction (ModuleList and friends) and are listed apart
    so `not_executed` means what it says."""

    executed: tuple[str, ...]
    not_executed: tuple[str, ...]
    containers: tuple[str, ...]

    def to_json_dict(self) -> dict:
        return {
            "executed": list(self.executed),
            "not_executed": list(self.not_executed),
            "containers": list(self.containers),
        }


@dataclasses.dataclass(frozen=True)
class WalkResult:
    """The single enumeration every consumer derives from."""

    nodes: tuple[WalkNode, ...]
    edges: tuple[WalkEdge, ...]
    claims: dict[str, Claim]
    unclaimed: tuple[str, ...]      # nodes with no claim (fatal only if fed)
    embedding_uses: tuple[EmbeddingUse, ...]
    unresolved_operands: tuple[UnresolvedOperand, ...]
    failures: tuple[WalkFailure, ...]
    trace_coverage: TraceCoverage
    execution: str                  # "fake" | "real"

    @property
    def ok(self) -> bool:
        return not self.failures

    def node(self, name: str) -> WalkNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def edges_for(self, name: str) -> tuple[WalkEdge, ...]:
        return tuple(e for e in self.edges
                     if e.param == name or name in e.param_aliases)

    def raise_if_failed(self) -> None:
        if self.failures:
            raise WalkError(self)

    def to_json_dict(self) -> dict:
        return {
            "execution": self.execution,
            "ok": self.ok,
            "nodes": [n.to_json_dict() for n in self.nodes],
            "edges": [e.to_json_dict() for e in self.edges],
            "claims": {k: v.to_json_dict()
                       for k, v in sorted(self.claims.items())},
            "unclaimed": list(self.unclaimed),
            "embedding_uses": [u.to_json_dict() for u in self.embedding_uses],
            "unresolved_operands": [
                u.to_json_dict() for u in self.unresolved_operands],
            "failures": [f.to_json_dict() for f in self.failures],
            "trace_coverage": self.trace_coverage.to_json_dict(),
        }


class WalkError(RuntimeError):
    """The walk failed. The message names every failing node and cites the op
    that fed it; `result` carries the full enumeration for diagnosis."""

    def __init__(self, result: WalkResult):
        self.result = result
        lines = ["model walk failed:"]
        for f in result.failures:
            where = f"{f.op}" + (f" '{f.equation}'" if f.equation else "")
            lines.append(
                f"  [{f.kind}] {f.node or '<unnamed tensor>'} fed to {where} "
                f"in module '{f.module or '<root>'}': {f.detail}"
            )
        lines.append(
            "Every matmul-fed parameter needs a claim: decide, pin(reason), "
            "or exclude(reason) — see ModelProfile.walk_claim_rules()."
        )
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Claim rules
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClaimRule:
    """One declarative claim: the first rule that matches a node claims it.

    Every populated matcher must hold for the rule to match. `module_class`
    matches anywhere in the owning module's MRO, so `"Linear"` claims
    subclasses too. `predicate` exists for policies a pattern cannot express
    (a profile's `is_pinned_name`); it receives the :class:`WalkNode`.
    """

    disposition: str
    reason: str
    name_regex: str | None = None
    leaf: str | None = None          # last dotted component of the node name
    module_class: str | None = None  # class name anywhere in the owner's MRO
    kind: str | None = None          # "parameter" | "buffer"
    persistent: bool | None = None   # match only (non-)persistent nodes
    max_ndim: int | None = None
    min_ndim: int | None = None
    floating: bool | None = None     # match only (non-)floating dtypes
    predicate: Callable[[WalkNode], bool] | None = None

    def __post_init__(self):
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"disposition must be one of {DISPOSITIONS}, "
                f"got {self.disposition!r}")
        if not self.reason:
            raise ValueError("a ClaimRule must carry a reason")

    def matches(self, node: WalkNode) -> bool:
        if self.name_regex is not None and not re.search(
                self.name_regex, node.name):
            return False
        if self.leaf is not None and node.name.rsplit(".", 1)[-1] != self.leaf:
            return False
        if (self.module_class is not None
                and self.module_class not in node.module_class_mro):
            return False
        if self.kind is not None and node.kind != self.kind:
            return False
        if self.persistent is not None and node.persistent is not self.persistent:
            return False
        ndim = len(node.shape)
        if self.max_ndim is not None and ndim > self.max_ndim:
            return False
        if self.min_ndim is not None and ndim < self.min_ndim:
            return False
        if self.floating is not None:
            if _dtype_is_floating(node.dtype) is not self.floating:
                return False
        if self.predicate is not None and not self.predicate(node):
            return False
        return True


def _dtype_is_floating(dtype_str: str) -> bool:
    dtype = getattr(torch, dtype_str.removeprefix("torch."), None)
    return isinstance(dtype, torch.dtype) and (
        dtype.is_floating_point or dtype.is_complex)


def apply_claim_rules(
    nodes: Sequence[WalkNode], rules: Sequence[ClaimRule],
) -> dict[str, Claim]:
    claims: dict[str, Claim] = {}
    for node in nodes:
        for index, rule in enumerate(rules):
            if rule.matches(node):
                claims[node.name] = Claim(
                    disposition=rule.disposition,
                    reason=rule.reason,
                    rule_index=index,
                )
                break
    return claims


# ---------------------------------------------------------------------------
# The interceptor (root B)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _OpRecord:
    op: str
    equation: str | None
    module: str
    operands: list  # per tensor operand: (key, shape, dtype, role)


class WeightUseInterceptor(TorchFunctionMode):
    """Records every matmul-family call and resolves its operands to named
    tensors by storage identity.

    The interceptor is host-mode agnostic: it works identically over a
    ``FakeTensorMode`` trace of a meta-loaded model and over a real CPU
    forward, because it reads operands at ``__torch_function__`` level —
    before any dispatch-level fake conversion replaces them.

    ``origin`` maps a storage key to (names, via-chain); it is seeded with
    the model's named tensors and extended through alias/cast ops during the
    trace. Every tensor whose key enters ``origin`` or ``computed`` is
    appended to ``_keepalive`` for the duration of the trace, so a freed
    storage's address can never be reused by a later tensor and
    misattributed — the one failure mode storage-identity resolution must
    never have.
    """

    def __init__(self, named_storages: Mapping[int, tuple[str, ...]]):
        super().__init__()
        self.origin: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {
            key: (names, ()) for key, names in named_storages.items()
        }
        self.computed: set[int] = set()
        self.records: list[_OpRecord] = []
        self.embedding_records: list[tuple[tuple[str, ...] | None, str]] = []
        self.module_stack: list[str] = []
        self._keepalive: list[torch.Tensor] = []

    # -- module attribution (driven by the walker's forward hooks) ---------
    @property
    def current_module(self) -> str:
        return self.module_stack[-1] if self.module_stack else ""

    # -- torch function protocol ------------------------------------------
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        matmul = _MATMUL_FUNCS.get(func)
        if matmul is not None:
            self._record_matmul(matmul[0], matmul[1], args, kwargs)
        elif func in _EMBEDDING_FUNCS:
            self._record_embedding(args, kwargs)

        name = getattr(func, "__name__", "")
        if name in _ALIAS_OP_NAMES:
            self._propagate_alias(name, args, out)

        for tensor in _iter_tensors(out):
            key = _storage_key(tensor)
            if key is not None and key not in self.origin:
                if key not in self.computed:
                    self.computed.add(key)
                    self._keepalive.append(tensor)
        return out

    def mark_inputs(self, tensors: Iterable[torch.Tensor]) -> None:
        """Register the forward's inputs so they resolve as activations."""
        for tensor in tensors:
            key = _storage_key(tensor)
            if key is not None and key not in self.origin:
                self.computed.add(key)
                self._keepalive.append(tensor)

    # -- internals ---------------------------------------------------------
    def _record_matmul(self, op: str, additive_positions: frozenset[int],
                       args: tuple, kwargs: dict) -> None:
        equation = None
        tensors: list[tuple[torch.Tensor, str]] = []
        position = 0
        for arg in args:
            if isinstance(arg, str) and op == "einsum" and equation is None:
                equation = arg
                continue
            for tensor in _iter_tensors(arg):
                role = ("additive" if position in additive_positions
                        else "multiplicand")
                tensors.append((tensor, role))
                position += 1
        additive_kwargs = _ADDITIVE_KWARGS.get(op, frozenset())
        for key_name, value in kwargs.items():
            for tensor in _iter_tensors(value):
                role = ("additive" if key_name in additive_kwargs
                        else "multiplicand")
                tensors.append((tensor, role))
        record = _OpRecord(
            op=op, equation=equation, module=self.current_module, operands=[])
        for tensor, role in tensors:
            record.operands.append((
                _storage_key(tensor),
                tuple(tensor.shape),
                str(tensor.dtype),
                role,
                tensor.dtype.is_floating_point or tensor.dtype.is_complex,
            ))
        self.records.append(record)

    def _record_embedding(self, args: tuple, kwargs: dict) -> None:
        # F.embedding(input, weight, ...); F.embedding_bag(input, weight, ...)
        weight = kwargs.get("weight")
        if weight is None:
            tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
            weight = tensor_args[1] if len(tensor_args) > 1 else None
        names = None
        if isinstance(weight, torch.Tensor):
            key = _storage_key(weight)
            entry = self.origin.get(key) if key is not None else None
            if entry is not None:
                names = entry[0]
        self.embedding_records.append((names, self.current_module))

    def _propagate_alias(self, op_name: str, args: tuple, out: Any) -> None:
        source = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if source is None:
            return
        key = _storage_key(source)
        entry = self.origin.get(key) if key is not None else None
        if entry is None:
            return
        names, via = entry
        hop = via if via and via[-1] == op_name else via + (op_name,)
        for tensor in _iter_tensors(out):
            out_key = _storage_key(tensor)
            if out_key is not None and out_key not in self.origin:
                self.origin[out_key] = (names, hop)
                self._keepalive.append(tensor)

    def release(self) -> None:
        """Drop the keepalive references once the trace is consumed."""
        self._keepalive.clear()


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def _named_tensor_index(model: nn.Module):
    """Root A: every named parameter and buffer, tied names preserved."""
    modules = dict(model.named_modules())
    non_persistent: set[str] = set()
    for qname, module in modules.items():
        for leaf in getattr(module, "_non_persistent_buffers_set", ()):
            non_persistent.add(f"{qname}.{leaf}" if qname else leaf)

    entries: list[tuple[str, str, torch.Tensor]] = []
    for name, param in model.named_parameters(remove_duplicate=False):
        entries.append((name, "parameter", param))
    for name, buf in model.named_buffers(remove_duplicate=False):
        entries.append((name, "buffer", buf))

    by_key: dict[int, list[str]] = {}
    for name, _, tensor in entries:
        key = _storage_key(tensor)
        if key is not None:
            by_key.setdefault(key, []).append(name)

    nodes: list[WalkNode] = []
    for name, kind, tensor in entries:
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(owner)
        mro: tuple[str, ...] = ()
        if module is not None:
            mro = tuple(
                cls.__name__ for cls in type(module).__mro__
                if cls not in (object,))
        key = _storage_key(tensor)
        aliases = tuple(n for n in by_key.get(key, []) if n != name)
        nodes.append(WalkNode(
            name=name,
            kind=kind,
            persistent=(kind == "parameter" or name not in non_persistent),
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            stored_bytes=tensor.numel() * tensor.element_size(),
            owner_module=owner,
            module_class=type(module).__name__ if module is not None else "",
            module_class_mro=mro,
            aliases=aliases,
        ))
    named_storages = {key: tuple(names) for key, names in by_key.items()}
    return nodes, named_storages


def _model_device(model: nn.Module) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _default_example_inputs(model: nn.Module, seq_len: int,
                            device: torch.device) -> dict:
    """A language-model default: token ids, cache off. Callers with other
    input contracts pass `example_inputs` explicitly."""
    ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    return {"input_ids": ids}


def _call_forward(model: nn.Module, example_inputs) -> None:
    if isinstance(example_inputs, Mapping):
        kwargs = dict(example_inputs)
        if "use_cache" not in kwargs and _forward_accepts(model, "use_cache"):
            kwargs["use_cache"] = False
        model(**kwargs)
    elif isinstance(example_inputs, tuple):
        model(*example_inputs)
    else:
        model(example_inputs)


def _forward_accepts(model: nn.Module, name: str) -> bool:
    """Signature check, not try-and-retry: a retry after a TypeError raised
    mid-forward would trace part of the model twice."""
    import inspect

    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def walk_model(
    model: nn.Module,
    example_inputs: Mapping[str, Any] | tuple | torch.Tensor | None = None,
    *,
    claim_rules: Sequence[ClaimRule] = (),
    execution: str = "fake",
    strict: bool = True,
    seq_len: int = 8,
) -> WalkResult:
    """Walk a model: enumerate its named tensors, trace one forward, resolve
    every matmul-family operand, and claim every node.

    Args:
        model: any ``nn.Module``. For ``execution="fake"``, meta-loaded is
            the intended (and cheapest) state.
        example_inputs: kwargs mapping, positional tuple, or a single tensor
            for the traced forward. Defaults to ``input_ids`` of shape
            ``(1, seq_len)`` on the model's device, with ``use_cache=False``
            when the forward accepts it.
        claim_rules: ordered :class:`ClaimRule` list; first match claims a
            node. Model profiles supply these via ``walk_claim_rules()``.
        execution: ``"fake"`` runs the forward under ``FakeTensorMode``
            (no weight I/O, works on meta); ``"real"`` runs a plain forward
            (root-B fallback for models fake tensors cannot execute).
        strict: raise :class:`WalkError` on any failure. ``strict=False``
            returns the result with ``failures`` populated instead.
        seq_len: length of the synthesized default input.

    Returns:
        A :class:`WalkResult` — the single enumeration downstream consumers
        derive from.

    Raises:
        WalkError: when ``strict`` and a matmul-fed node is unclaimed or a
            floating multiplicand cannot be resolved to any named tensor.
    """
    if execution not in ("fake", "real"):
        raise ValueError(f"execution must be 'fake' or 'real', got {execution!r}")

    nodes, named_storages = _named_tensor_index(model)
    device = _model_device(model)
    if execution == "real" and device.type == "meta":
        raise ValueError(
            "execution='real' needs materialized weights; this model is on "
            "the meta device. Use execution='fake', or load real weights.")
    if example_inputs is None:
        example_inputs = _default_example_inputs(model, seq_len, device)

    interceptor = WeightUseInterceptor(named_storages)
    input_tensors = list(_iter_tensors(
        list(example_inputs.values())
        if isinstance(example_inputs, Mapping) else example_inputs))
    interceptor.mark_inputs(input_tensors)

    executed: set[str] = set()
    hook_handles = []
    module_names = dict(model.named_modules())

    def _pre_hook(qname):
        def hook(module, args, kwargs=None):
            executed.add(qname)
            interceptor.module_stack.append(qname)
        return hook

    def _post_hook(qname):
        def hook(module, args, output):
            if interceptor.module_stack and \
                    interceptor.module_stack[-1] == qname:
                interceptor.module_stack.pop()
        return hook

    for qname, module in module_names.items():
        hook_handles.append(
            module.register_forward_pre_hook(_pre_hook(qname)))
        hook_handles.append(
            module.register_forward_hook(_post_hook(qname)))

    try:
        with torch.no_grad():
            if execution == "fake":
                from torch._subclasses.fake_tensor import FakeTensorMode

                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
                with interceptor, fake_mode:
                    _call_forward(model, example_inputs)
            else:
                with interceptor:
                    _call_forward(model, example_inputs)
    finally:
        for handle in hook_handles:
            handle.remove()

    result = _assemble(
        nodes, interceptor, executed, module_names, claim_rules, execution)
    interceptor.release()
    if strict:
        result.raise_if_failed()
    return result


def _assemble(
    nodes: list[WalkNode],
    interceptor: WeightUseInterceptor,
    executed: set[str],
    module_names: dict[str, nn.Module],
    claim_rules: Sequence[ClaimRule],
    execution: str,
) -> WalkResult:
    node_by_name = {n.name: n for n in nodes}

    # Resolve operands -> edges / unresolved.
    edge_counts: dict[tuple, int] = {}
    unresolved: list[UnresolvedOperand] = []
    for record in interceptor.records:
        all_shapes = tuple(shape for _, shape, _, _, _ in record.operands)
        for index, (key, shape, dtype, role, floating) in enumerate(
                record.operands):
            entry = interceptor.origin.get(key) if key is not None else None
            if entry is not None:
                names, via = entry
                primary = names[0]
                node = node_by_name.get(primary)
                edge_key = (
                    primary, tuple(names[1:]), record.op, record.equation,
                    role, index, shape, dtype, all_shapes,
                    node.stored_bytes if node else 0, record.module, via,
                )
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
                continue
            if key is not None and key in interceptor.computed:
                continue  # an activation / traced intermediate
            unresolved.append(UnresolvedOperand(
                op=record.op, equation=record.equation, module=record.module,
                operand_index=index, operand_shape=shape,
                operand_dtype=dtype, role=role, is_floating=floating,
            ))

    edges = tuple(sorted(
        (WalkEdge(
            param=k[0], param_aliases=k[1], op=k[2], equation=k[3],
            role=k[4], operand_index=k[5], operand_shape=k[6],
            operand_dtype=k[7], operand_shapes=k[8], stored_bytes=k[9],
            module=k[10], via=k[11], calls=count)
         for k, count in edge_counts.items()),
        key=lambda e: (e.param, e.module, e.op, e.operand_index, e.role,
                       e.operand_shape, e.via, e.equation or ""),
    ))

    embedding_uses = tuple(sorted(
        {EmbeddingUse(
            param=names[0], param_aliases=tuple(names[1:]), module=module)
         for names, module in interceptor.embedding_records
         if names is not None},
        key=lambda u: (u.param, u.module),
    ))

    # Claims.
    claims = apply_claim_rules(nodes, claim_rules)
    unclaimed = tuple(sorted(
        n.name for n in nodes if n.name not in claims))

    # Failures.
    failures: list[WalkFailure] = []
    seen_failure_nodes: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.role != "multiplicand":
            continue
        for name in (edge.param, *edge.param_aliases):
            if name in claims or (name, edge.op) in seen_failure_nodes:
                continue
            seen_failure_nodes.add((name, edge.op))
            failures.append(WalkFailure(
                kind="unclaimed", node=name, op=edge.op,
                equation=edge.equation, module=edge.module,
                detail=(
                    f"parameter {name!r} (shape "
                    f"{list(node_by_name[name].shape)}, "
                    f"{node_by_name[name].stored_bytes} bytes) feeds this op "
                    "and no claim rule matched it"),
            ))
    for use in embedding_uses:
        for name in (use.param, *use.param_aliases):
            if name in claims or (name, "embedding") in seen_failure_nodes:
                continue
            seen_failure_nodes.add((name, "embedding"))
            failures.append(WalkFailure(
                kind="unclaimed", node=name, op="embedding", equation=None,
                module=use.module,
                detail=(f"embedding weight {name!r} is consumed by "
                        "F.embedding and no claim rule matched it"),
            ))
    for operand in unresolved:
        if not (operand.is_floating and operand.role == "multiplicand"):
            continue
        failures.append(WalkFailure(
            kind="unresolved", node=None, op=operand.op,
            equation=operand.equation, module=operand.module,
            detail=(
                f"floating operand #{operand.operand_index} (shape "
                f"{list(operand.operand_shape)}, {operand.operand_dtype}) "
                "matches no named parameter or buffer and was not computed "
                "by the traced forward — a weight this walk cannot name "
                "(was it .to()'d or reconstructed outside the forward?)"),
        ))

    # Trace coverage.
    containers = tuple(sorted(
        qname for qname, module in module_names.items()
        if isinstance(module, _CONTAINER_CLASSES)))
    container_set = set(containers)
    executed_names = tuple(sorted(
        q for q in executed if q not in container_set))
    not_executed = tuple(sorted(
        qname for qname in module_names
        if qname not in executed and qname not in container_set))

    return WalkResult(
        nodes=tuple(sorted(nodes, key=lambda n: (n.name, n.kind))),
        edges=edges,
        claims=claims,
        unclaimed=unclaimed,
        embedding_uses=embedding_uses,
        unresolved_operands=tuple(unresolved),
        failures=tuple(failures),
        trace_coverage=TraceCoverage(
            executed=executed_names,
            not_executed=not_executed,
            containers=containers,
        ),
        execution=execution,
    )
