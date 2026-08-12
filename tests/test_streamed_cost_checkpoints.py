from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import prismaquant.aura_cost as aura
import prismaquant.expert_empirical_cost as expert_cost
import prismaquant.production_weight_cache as pwc
from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
from prismaquant.model_profiles.default import DefaultProfile


class _FakeStreamingContext:
    """Logical one-layer residency context for exact CPU contract tests."""

    def __init__(self, model):
        self.model = model
        self.base_model = model.model
        self.layers = model.model.layers
        self.layers_prefix = "model.layers."
        self.num_layers = len(self.layers)
        self.device = torch.device("cpu")
        self.dtype = next(model.parameters()).dtype
        self.active: set[int] = set()
        self.max_active = 0
        self.install_calls = 0

    def install(self, layer):
        self.install_calls += 1
        self.active.add(int(layer))
        self.layers[int(layer)]._fixture_stream_resident = True
        self.max_active = max(self.max_active, len(self.active))
        return "fixture"

    def unload(self, layer):
        self.active.discard(int(layer))
        self.layers[int(layer)]._fixture_stream_resident = False
        return 0

    def schedule_prefetch(self, _layer):
        return None

    def shutdown(self):
        self.active.clear()


class _DenseLayer(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, **_kwargs):
        if getattr(self, "_fixture_requires_stream_residency", False):
            assert getattr(self, "_fixture_stream_resident", False)
        return torch.tanh(self.proj(hidden_states))


class _DenseTinyLM(nn.Module):
    def __init__(self, state=None, vocab=23, width=16):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(layer_types=())
        self.model.embed_tokens = nn.Embedding(vocab, width)
        self.model.layers = nn.ModuleList([_DenseLayer(width), _DenseLayer(width)])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(width, vocab, bias=False)
        if state is not None:
            self.load_state_dict(state)

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden)))


class _RenderedCache:
    def __init__(self, model, fmt):
        self.metadata = {
            "calib_hash": "fixture-calibration",
            "cb_cache_pair_identity": {
                "schema": "prismaquant.production_weight_cache.cb_pair_set.v1",
                "identity_sha256": "e" * 64,
                "artifact_sha256": "d" * 64,
                "entries": 2,
                "published_entries": 2,
                "calibration_hashes": ["fixture-calibration"],
                "git_commits": ["9" * 40],
                "producer_source_sha256": ["8" * 64],
            },
        }
        self.weights = {
            (name, fmt): mod.weight.detach().clone() + 0.03125
            for name, mod in model.named_modules()
            if name in {
                "model.layers.0.proj",
                "model.layers.1.proj",
            }
        }

    def get(self, name, fmt):
        return self.weights.get((name, fmt))

    def compact_for_pickle(self):
        return 0


def _cb_provenance():
    return {
        "cb_cost_provenance_schema": "test.cb.provenance.v1",
        "cb_render_identity": {
            "schema": "test.cb.render_identity.v1",
            "col_weights_sha256": "f" * 64,
            "scale_coding": "two_tier",
            "scale_sweep_scope": "all",
            "ldlq_scope": "none",
            "layout_version": 2,
        },
    }


def _model_identity(label: str):
    shard_digest = hashlib.sha256(label.encode()).hexdigest()
    value = {
        "config": {"fixture": True},
        "weight_map": {"fixture.weight": "fixture.weight"},
        "shards": [{
            "path": f"/fixture/{label}.safetensors",
            "size": 1,
            "sha256": shard_digest,
        }],
    }
    return {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": label,
        "resolved_commit": None,
        "content_sha256": canonical_json_sha256(
            value, where="fixture streamed model identity"
        ),
        **value,
    }


def _dense_runner(state):
    model = _DenseTinyLM(state).eval()
    for layer in model.model.layers:
        layer._fixture_requires_stream_residency = True
    context = _FakeStreamingContext(model)
    return model, context, StreamedCausalLM(context, DefaultProfile())


def test_streamed_aura_cost_rows_are_exactly_resident_rows():
    torch.manual_seed(104)
    seed_model = _DenseTinyLM().eval()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    calib = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    resident = _DenseTinyLM(state).eval()
    resident_result = aura.compute_aura_cost(
        resident,
        calib,
        ["NVFP4"],
        n_probes=3,
        n_linear_chunks=1,
        min_free_gib=0.0,
        production_cache=_RenderedCache(resident, "NVFP4"),
        require_production_cache=True,
        dw_dtype="float32",
        profile=DefaultProfile(),
    )

    streamed_model, context, runner = _dense_runner(state)
    streamed_result = aura.compute_aura_cost_streamed(
        runner,
        calib,
        ["NVFP4"],
        n_probes=3,
        min_free_gib=0.0,
        production_cache=_RenderedCache(streamed_model, "NVFP4"),
        require_production_cache=True,
        dw_dtype="float32",
        profile=DefaultProfile(),
    )

    assert streamed_result["stats"] == resident_result["stats"]
    assert streamed_result["costs"] == resident_result["costs"]
    assert context.max_active == 1


def _run_streamed_aura(
    state, calib, checkpoint_dir, *, resume, monkeypatch
):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(
        pwc,
        "production_cache_cb_render_provenance",
        lambda *_args, **_kwargs: _cb_provenance(),
    )
    model, context, runner = _dense_runner(state)
    result = aura.compute_aura_cost_streamed(
        runner,
        calib,
        ["FP8_CB_K28"],
        n_probes=2,
        production_cache=_RenderedCache(model, "FP8_CB_K28"),
        require_production_cache=True,
        dw_dtype="float32",
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        model_identity=_model_identity("dense-v1"),
        profile=DefaultProfile(),
    )
    return result, context


def test_streamed_aura_interrupt_resume_and_identity_refusal(
    tmp_path, monkeypatch
):
    torch.manual_seed(105)
    seed_model = _DenseTinyLM().eval()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    calib = torch.tensor([[1, 2, 3, 4]])
    expected, _ = _run_streamed_aura(
        state, calib, tmp_path / "whole", resume=False, monkeypatch=monkeypatch
    )

    original_writer = aura._write_aura_unit_checkpoint
    writes = {"count": 0}

    def interrupt_after_first(*args, **kwargs):
        original_writer(*args, **kwargs)
        writes["count"] += 1
        if writes["count"] == 1:
            raise RuntimeError("fixture interruption")

    monkeypatch.setattr(
        aura, "_write_aura_unit_checkpoint", interrupt_after_first
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        _run_streamed_aura(
            state,
            calib,
            tmp_path / "resumed",
            resume=False,
            monkeypatch=monkeypatch,
        )
    assert writes["count"] == 1

    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", original_writer)
    actual, _ = _run_streamed_aura(
        state, calib, tmp_path / "resumed", resume=True, monkeypatch=monkeypatch
    )
    assert actual["stats"] == expected["stats"]
    assert actual["costs"] == expected["costs"]

    complete_model, complete_context, complete_runner = _dense_runner(state)
    complete = aura.compute_aura_cost_streamed(
        complete_runner,
        calib,
        ["FP8_CB_K28"],
        n_probes=2,
        production_cache=_RenderedCache(complete_model, "FP8_CB_K28"),
        require_production_cache=True,
        dw_dtype="float32",
        checkpoint_dir=tmp_path / "resumed",
        resume=True,
        model_identity=_model_identity("dense-v1"),
        profile=DefaultProfile(),
    )
    assert complete["costs"] == expected["costs"]
    assert complete_context.install_calls == 0

    mismatch_model, mismatch_context, mismatch_runner = _dense_runner(state)
    with pytest.raises(RuntimeError, match="calibration.*refusing"):
        aura.compute_aura_cost_streamed(
            mismatch_runner,
            torch.tensor([[1, 2, 3, 9]]),
            ["FP8_CB_K28"],
            n_probes=2,
            production_cache=_RenderedCache(
                mismatch_model, "FP8_CB_K28"
            ),
            require_production_cache=True,
            dw_dtype="float32",
            checkpoint_dir=tmp_path / "resumed",
            resume=True,
            model_identity=_model_identity("dense-v1"),
            profile=DefaultProfile(),
        )
    # The mismatch is validated before capture_boundaries installs layer 0.
    assert mismatch_context.install_calls == 0


class _ExactAnchorRenderer:
    def __init__(self, formats_by_qname):
        self.formats_by_qname = {
            name: tuple(formats)
            for name, formats in formats_by_qname.items()
        }
        self.identity = {
            "schema": "test.production_anchor_renderer.v1",
            "formats_by_qname": {
                name: list(formats)
                for name, formats in self.formats_by_qname.items()
            },
            "cb_render_identity": None,
            "arm_identity": {"arm": "fixture-production"},
        }
        self.render_count = 0
        self.max_live_rendered = 0

    def render_layer(self, *, layer, modules, formats_by_qname):
        del layer
        offsets = {"NVFP4": 0.03125, "FP8_E4M3": 0.015625}
        offsets["FP8_CB_K28"] = 0.0078125
        rendered = {
            (name, fmt): modules[name].weight.detach().clone() + offsets[fmt]
            for name, formats in formats_by_qname.items()
            for fmt in formats
        }
        self.render_count += len(rendered)
        self.max_live_rendered = max(self.max_live_rendered, len(rendered))
        return rendered


def test_streamed_aura_production_anchors_are_sparse_and_never_use_rtn(
    monkeypatch,
):
    torch.manual_seed(108)
    seed_model = _DenseTinyLM().eval()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    model, _context, runner = _dense_runner(state)
    q0 = "model.layers.0.proj"
    q1 = "model.layers.1.proj"
    render_plan = {
        q0: ("NVFP4", "FP8_E4M3"),
        q1: ("NVFP4",),
    }
    unit_plan = {
        q0: (*render_plan[q0], "BF16"),
        q1: (*render_plan[q1], "BF16"),
    }
    renderer = _ExactAnchorRenderer(render_plan)

    def refuse_delta_w(*_args, **_kwargs):
        raise AssertionError("production-anchor AURA touched RTN/cache dW")

    monkeypatch.setattr(aura, "_delta_w", refuse_delta_w)
    payload = aura.compute_aura_cost_streamed(
        runner,
        torch.tensor([[1, 2, 3, 4]]),
        ["NVFP4", "FP8_DYNAMIC", "BF16"],
        n_probes=2,
        dw_dtype="float32",
        formats_by_qname=unit_plan,
        anchor_renderer=renderer,
        profile=DefaultProfile(),
    )

    # Three declared anchors, not two units x two nonterminal rungs.
    assert renderer.render_count == 3
    assert payload["provenance"]["production_anchor_expected_renders"] == 3
    assert payload["provenance"][
        "production_anchor_no_full_menu_materialization"
    ] is True
    assert set(payload["costs"][q0]) == {
        "NVFP4", "FP8_E4M3", "BF16"
    }
    assert set(payload["costs"][q1]) == {"NVFP4", "BF16"}
    for name, formats in render_plan.items():
        for fmt in formats:
            row = payload["costs"][name][fmt]
            assert row["dw_source"] == "production_render"
            assert row["production_anchor_measured"] is True
    assert model is runner.model


def _run_checkpointed_anchor_diagnostic(
    state, calib, checkpoint_dir, *, resume, monkeypatch,
):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    _model, context, runner = _dense_runner(state)
    plan = {
        "model.layers.0.proj": ("FP8_CB_K28",),
        "model.layers.1.proj": ("FP8_CB_K28",),
    }
    renderer = _ExactAnchorRenderer(plan)
    renderer.identity["cb_render_identity"] = {
        "schema": "test.sparse.anchor.cb.v1",
        "formats_by_qname": {
            name: list(formats) for name, formats in plan.items()
        },
    }
    payload = aura.compute_aura_cost_streamed(
        runner,
        calib,
        ["FP8_CB_K28"],
        n_probes=2,
        dw_dtype="float32",
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        model_identity=_model_identity("anchor-diagnostic-v1"),
        formats_by_qname=plan,
        anchor_renderer=renderer,
        diagnostic_weight_mse_pairs=list(
            (name, "FP8_CB_K28") for name in plan
        ),
        profile=DefaultProfile(),
    )
    return payload, context, renderer


def test_production_anchor_weight_mse_diagnostic_resumes_exactly(
    tmp_path, monkeypatch,
):
    torch.manual_seed(111)
    seed_model = _DenseTinyLM().eval()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    calib = torch.tensor([[1, 2, 3, 4]])
    expected, _, _ = _run_checkpointed_anchor_diagnostic(
        state,
        calib,
        tmp_path / "whole-anchor",
        resume=False,
        monkeypatch=monkeypatch,
    )

    original_writer = aura._write_aura_unit_checkpoint
    writes = {"count": 0}

    def interrupt_after_first(*args, **kwargs):
        original_writer(*args, **kwargs)
        writes["count"] += 1
        if writes["count"] == 1:
            raise RuntimeError("anchor diagnostic interruption")

    monkeypatch.setattr(
        aura, "_write_aura_unit_checkpoint", interrupt_after_first
    )
    with pytest.raises(RuntimeError, match="anchor diagnostic interruption"):
        _run_checkpointed_anchor_diagnostic(
            state,
            calib,
            tmp_path / "resume-anchor",
            resume=False,
            monkeypatch=monkeypatch,
        )
    monkeypatch.setattr(
        aura, "_write_aura_unit_checkpoint", original_writer
    )
    actual, _, renderer = _run_checkpointed_anchor_diagnostic(
        state,
        calib,
        tmp_path / "resume-anchor",
        resume=True,
        monkeypatch=monkeypatch,
    )
    assert actual["costs"] == expected["costs"]
    assert renderer.render_count == 1
    for rows in actual["costs"].values():
        row = rows["FP8_CB_K28"]
        assert row["weight_mse_diagnostic"] == pytest.approx(
            0.0078125 ** 2
        )
        assert row["weight_mse_is_cost_input"] is False
    assert actual["provenance"]["production_anchor_cost_currency"] == (
        "aura_only"
    )
    assert actual["provenance"]["weight_mse_diagnostic_rows"] == 2

    complete, complete_context, complete_renderer = (
        _run_checkpointed_anchor_diagnostic(
            state,
            calib,
            tmp_path / "resume-anchor",
            resume=True,
            monkeypatch=monkeypatch,
        )
    )
    assert complete["costs"] == expected["costs"]
    assert complete_context.install_calls == 0
    assert complete_renderer.render_count == 0
    assert complete["provenance"][
        "production_anchor_restored_renders"
    ] == 2


class _Expert(nn.Module):
    def __init__(self, width=16, intermediate=32):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, hidden):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden))
            * self.up_proj(hidden)
        )


class _ExpertLayer(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = nn.ModuleList([_Expert(width), _Expert(width)])

    def forward(self, hidden_states, input_ids=None, **_kwargs):
        if getattr(self, "_fixture_requires_stream_residency", False):
            assert getattr(self, "_fixture_stream_resident", False)
        assert input_ids is not None
        flat = hidden_states.reshape(-1, hidden_states.size(-1))
        routes = input_ids.reshape(-1).remainder(len(self.mlp.experts))
        routed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.mlp.experts):
            selected = routes == expert_id
            if bool(selected.any()):
                routed[selected] = expert(flat[selected])
        return hidden_states + routed.reshape_as(hidden_states)


class _ExpertTinyLM(nn.Module):
    def __init__(self, state=None, vocab=23, width=16):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(hc_mult=1, layer_types=())
        self.model.embed_tokens = nn.Embedding(vocab, width)
        self.model.layers = nn.ModuleList([
            _ExpertLayer(width),
            _ExpertLayer(width),
        ])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(width, vocab, bias=False)
        if state is not None:
            self.load_state_dict(state)

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, input_ids=input_ids)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden)))


def _expert_runner(state):
    model = _ExpertTinyLM(state).eval()
    for layer in model.model.layers:
        layer._fixture_requires_stream_residency = True
    context = _FakeStreamingContext(model)
    profile = DeepseekV4Profile()
    return model, context, StreamedCausalLM(context, profile), profile


def test_never_routed_expert_keeps_real_zero_production_anchor(monkeypatch):
    torch.manual_seed(109)
    seed_model = _ExpertTinyLM().eval()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    _model, _context, runner, profile = _expert_runner(state)
    # Every input id is even, so expert 1 receives no routed token in either
    # layer. Its real rendered dW is still measured; smooth AURA honestly
    # returns zero rather than borrowing a sibling level or adding epsilon.
    qname = "model.layers.0.mlp.experts.1.gate_proj"
    renderer = _ExactAnchorRenderer({qname: ("NVFP4",)})
    monkeypatch.setattr(
        aura,
        "_delta_w",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cold production anchor touched RTN")
        ),
    )
    payload = aura.compute_aura_cost_streamed(
        runner,
        torch.tensor([[2, 4, 6, 8]]),
        ["NVFP4", "MXFP4_SOURCE"],
        n_probes=2,
        dw_dtype="float32",
        formats_by_qname={qname: ("NVFP4", "MXFP4_SOURCE")},
        anchor_renderer=renderer,
        include_routed_experts=True,
        profile=profile,
    )
    row = payload["costs"][qname]["NVFP4"]
    assert row["predicted_dloss"] == 0.0
    assert row["x2_per_probe"] == [0.0, 0.0]
    assert row["dw_source"] == "production_render"
    assert row["production_anchor_measured"] is True
    assert row["production_anchor_zero"] is True
    assert payload["costs"][qname]["MXFP4_SOURCE"][
        "predicted_dloss"
    ] == 0.0


def test_production_anchor_cold_render_requires_exact_declared_profile_scope(
    monkeypatch,
):
    import prismaquant.streaming_production_cache as streaming
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    torch.manual_seed(110)
    model = _ExpertTinyLM().eval()
    profile = DeepseekV4Profile()
    qname = "model.layers.0.mlp.experts.1.gate_proj"

    class _NoActivations:
        def __contains__(self, _qname):
            return False

        def load_with_row_indices(self, _qname):
            raise AssertionError("declared cold expert tried to load acts")

    kwargs = dict(
        act_index=_NoActivations(),
        formats_by_qname={qname: ("NVFP4_CB_K12",)},
        levers={"gptq": True, "weighted_vq": True},
        profile=profile,
        device="cpu",
        col_weights={qname: torch.ones(16)},
        cb_serialization_context=CBSerializationContext.production(
            scale_sweep=True,
            ldlq=True,
            codebook_source="lattice",
        ),
        calibration_hash="a" * 64,
        arm_identity={"arm": "fixture-production-ldlq"},
        model_identity=_model_identity("cold-render-source"),
        max_act_rows=8,
        producer_git_commit="1" * 40,
        producer_source_sha256="2" * 64,
    )
    with pytest.raises(RuntimeError, match="undeclared_missing"):
        streaming.StreamedProductionAnchorRenderer(model, **kwargs)

    calls = []

    def render(weight, fmt, *, activations, ldlq_missing_activation_ok, **_kw):
        calls.append((fmt, dict(activations), ldlq_missing_activation_ok))
        return weight.detach().clone() + 0.03125

    monkeypatch.setattr(streaming, "render_production_weight", render)
    renderer = streaming.StreamedProductionAnchorRenderer(
        model,
        cold_expert_provenance={
            "rule": "unrouted_expert_neutral_prior:layer_routed_mean",
            "names": [qname],
        },
        **kwargs,
    )
    rendered = renderer.render_layer(
        layer=0,
        modules={qname: model.get_submodule(qname)},
        formats_by_qname={qname: ("NVFP4_CB_K12",)},
    )
    assert set(rendered) == {(qname, "NVFP4_CB_K12")}
    assert calls == [("NVFP4_CB_K12", {}, True)]
    assert renderer.render_count == 1
    assert renderer.cache.weights == {}
    completed = renderer.bind_completed_source_weight_identities({
        qname: renderer.source_weight_identity_for(qname)
    })
    assert completed["source_weights"]["complete"] is True
    assert completed["cb_render_identity"]["source_weights_complete"] is True
    assert completed["cb_render_identity"]["render_scope"] == (
        "sparse_production_anchors"
    )


def test_production_anchor_resolves_raw_named_expert_activation_cache(
    monkeypatch,
):
    """DSv4-Flash keys its activation cache under RAW per-expert names.

    ``canonical_linear_name`` remaps those onto the packed Qwen-style
    spelling, which is cached for nobody here -- so resolving through it
    alone reports every routed expert as an undeclared activation miss, and
    (had the coverage gate not been fail-closed) would have rendered 66k
    experts against the never-routed neutral prior instead of their real
    activations.
    """
    import prismaquant.streaming_production_cache as streaming
    from prismaquant.measure_quant_cost import canonical_linear_name
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    torch.manual_seed(111)
    model = _ExpertTinyLM().eval()
    profile = DeepseekV4Profile()
    qname = "model.layers.0.mlp.experts.1.gate_proj"
    # The remap is real for this profile: without the raw-name fallback the
    # renderer would look up a key the cache has never held.
    assert canonical_linear_name(qname, profile) != qname

    class _RawNamedActivations:
        def __init__(self, names):
            self._names = set(names)
            self.loaded: list[str] = []

        def __contains__(self, name):
            return name in self._names

        def load_with_row_indices(self, name):
            if name not in self._names:
                raise KeyError(name)
            self.loaded.append(name)
            cols = int(model.get_submodule(qname).weight.shape[1])
            return torch.ones(4, cols), None

    act_index = _RawNamedActivations([qname])
    calls = []

    def render(weight, fmt, *, activations, **_kw):
        calls.append((fmt, sorted(activations)))
        return weight.detach().clone() + 0.03125

    monkeypatch.setattr(streaming, "render_production_weight", render)
    # No cold declaration: the expert IS cached, just under the raw name.
    renderer = streaming.StreamedProductionAnchorRenderer(
        model,
        act_index=act_index,
        formats_by_qname={qname: ("NVFP4_CB_K12",)},
        levers={"gptq": True, "weighted_vq": True},
        profile=profile,
        device="cpu",
        col_weights={qname: torch.ones(16)},
        cb_serialization_context=CBSerializationContext.production(
            scale_sweep=True,
            ldlq=True,
            codebook_source="lattice",
        ),
        calibration_hash="b" * 64,
        arm_identity={"arm": "fixture-raw-named-acts"},
        model_identity=_model_identity("raw-named-act-source"),
        max_act_rows=8,
        producer_git_commit="3" * 40,
        producer_source_sha256="4" * 64,
    )
    rendered = renderer.render_layer(
        layer=0,
        modules={qname: model.get_submodule(qname)},
        formats_by_qname={qname: ("NVFP4_CB_K12",)},
    )
    assert set(rendered) == {(qname, "NVFP4_CB_K12")}
    # Loaded under the raw name, and the render saw the REAL activations
    # rather than the empty cold-prior mapping.
    assert act_index.loaded == [qname]
    assert calls == [("NVFP4_CB_K12", [qname])]


def test_streamed_expert_cost_rows_are_exactly_resident_rows():
    torch.manual_seed(106)
    seed_model = _ExpertTinyLM().eval()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    calib = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    profile = DeepseekV4Profile()
    resident = _ExpertTinyLM(state).eval()
    expected = expert_cost.measure_expert_unit_costs(
        resident,
        profile,
        calib,
        ["NVFP4", "BF16"],
        progress=False,
    )

    _model, context, runner, profile = _expert_runner(state)
    actual = expert_cost.measure_expert_unit_costs_streamed(
        runner,
        profile,
        calib,
        ["NVFP4", "BF16"],
        progress=False,
    )
    assert actual == expected
    # The pinned target plus the currently traversed other layer is the bound.
    assert context.max_active <= 2


def _run_streamed_expert(
    state, checkpoint_dir, *, resume, model_identity
):
    model, context, runner, profile = _expert_runner(state)
    result = expert_cost.measure_expert_unit_costs_streamed(
        runner,
        profile,
        torch.tensor([[1, 2, 3, 4]]),
        ["NVFP4", "BF16"],
        progress=False,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        model_identity=model_identity,
    )
    return result, context


def test_streamed_expert_interrupt_resume_and_identity_refusal(
    tmp_path, monkeypatch
):
    torch.manual_seed(107)
    seed_model = _ExpertTinyLM().eval()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    identity = _model_identity("tiny-model-v1")
    expected, _ = _run_streamed_expert(
        state, tmp_path / "whole", resume=False, model_identity=identity
    )

    original_writer = expert_cost._write_expert_unit_checkpoint
    writes = {"count": 0}

    def interrupt_after_first(*args, **kwargs):
        original_writer(*args, **kwargs)
        writes["count"] += 1
        if writes["count"] == 1:
            raise RuntimeError("fixture interruption")

    monkeypatch.setattr(
        expert_cost, "_write_expert_unit_checkpoint", interrupt_after_first
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        _run_streamed_expert(
            state,
            tmp_path / "resumed",
            resume=False,
            model_identity=identity,
        )
    assert writes["count"] == 1

    monkeypatch.setattr(
        expert_cost, "_write_expert_unit_checkpoint", original_writer
    )
    actual, _ = _run_streamed_expert(
        state,
        tmp_path / "resumed",
        resume=True,
        model_identity=identity,
    )
    assert actual == expected

    _model, mismatch_context, mismatch_runner, profile = _expert_runner(state)
    with pytest.raises(RuntimeError, match="model.content_sha256.*refusing"):
        expert_cost.measure_expert_unit_costs_streamed(
            mismatch_runner,
            profile,
            torch.tensor([[1, 2, 3, 4]]),
            ["NVFP4", "BF16"],
            progress=False,
            checkpoint_dir=tmp_path / "resumed",
            resume=True,
            model_identity=_model_identity("tiny-model-v2"),
        )
    assert mismatch_context.install_calls == 0
