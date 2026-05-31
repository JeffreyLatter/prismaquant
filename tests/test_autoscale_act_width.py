"""Regression: autoscale must size retained-activation memory on the widest
per-Linear activation (``max(hidden, intermediate)``), not ``hidden_size``.

A transformer MLP's ``down_proj`` reads an ``intermediate_size``-wide input and
the cost step's batched render materializes ``intermediate_size``-wide fp32
outputs. On Gemma4-31B (intermediate 21504 vs hidden 5376, 4×) sizing on
``hidden_size`` undershot host RAM ~4×, so the cost step's autoscale picked
``layers_per_shard=14`` and the memory watchdog aborted mid-render. Sizing on
``_act_width`` shrinks the pick to a value whose working set fits.
"""
import json

from prismaquant import autoscale as A


def test_act_width_takes_max_of_hidden_and_ffn():
    cfg = {"hidden_size": 5376, "intermediate_size": 21504,
           "num_hidden_layers": 60}
    assert A._act_width(cfg) == 21504


def test_act_width_reads_text_config_and_moe_keys():
    cfg = {"text_config": {"hidden_size": 4096, "moe_intermediate_size": 1536,
                           "intermediate_size": 12288}}
    assert A._act_width(cfg) == 12288


def test_act_width_collapses_to_hidden_when_no_ffn_declared():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32}
    assert A._act_width(cfg) == 4096


def test_estimate_scales_active_bytes_with_act_width():
    """Wide-MLP estimate must exceed the hidden-only estimate in proportion to
    the width ratio (activation term only; the gradient term is fixed)."""
    base = dict(model_path="/nonexistent", num_layers=60, hidden_size=5376,
                nsamples=8, seqlen=1024)
    _, active_hidden = A.estimate_per_layer_bytes(**base)               # act_width None -> hidden
    _, active_wide = A.estimate_per_layer_bytes(**base, act_width=21504)
    grad = 1 * 1024 ** 3  # fallback per-layer weight when no safetensors on disk
    act_hidden = active_hidden - grad
    act_wide = active_wide - grad
    assert abs(act_wide / act_hidden - 21504 / 5376) < 0.01


def _write_fake_model(tmp_path, cfg: dict, disk_gb: float):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    # one fake safetensors blob so _model_weight_bytes_on_disk is realistic
    blob = tmp_path / "model.safetensors"
    blob.write_bytes(b"\0" * 4096)
    import os
    os.truncate(blob, int(disk_gb * 1024 ** 3))
    return str(tmp_path)


def test_pick_layers_per_shard_shrinks_for_large_mlp(tmp_path):
    """Same disk size + host budget: a 4×-intermediate model must pick strictly
    fewer layers/shard than an intermediate==hidden model — this is the OOM
    that the watchdog caught on Gemma4-31B."""
    common = dict(num_hidden_layers=60, hidden_size=5376)
    wide = _write_fake_model(tmp_path / "wide",
                             {**common, "intermediate_size": 21504}, disk_gb=62.0)
    narrow = _write_fake_model(tmp_path / "narrow",
                               {**common, "intermediate_size": 5376}, disk_gb=62.0)
    avail = int(118 * 1024 ** 3)
    lps_wide, _ = A.pick_layers_per_shard(
        wide, nsamples=8, seqlen=1024, available_ram_bytes=avail)
    lps_narrow, _ = A.pick_layers_per_shard(
        narrow, nsamples=8, seqlen=1024, available_ram_bytes=avail)
    assert lps_wide < lps_narrow, (lps_wide, lps_narrow)
    assert lps_wide >= 1
