# Validation Harness

PrismaQuant validates candidate layer configs before shipping quantized model
artifacts. The harness measures held-out WikiText perplexity, MMLU multiple
choice accuracy, and End-KL against a fixed WikiText calibration batch.
Materialized vLLM artifacts also run the downstream serving suite: PPL/mean
NLL, log-likelihood task checks, ToolEvalBench, and eager/graph vLLM smokes.

## Validate and Register

```bash
python -m prismaquant.validation_harness \
  --model /hfcache/Qwen3-4B \
  --layer-config /work/iterate_out/final_layer_config_bpp_4.50.json \
  --register \
  --notes "iterated L3 v1 candidate"
```

Useful options:

```bash
--cache-dir /home/rob/dq-runs/prismaquant-validation-cache
--registry /home/rob/dq-runs/prismaquant-artifact-registry.json
--artifact-path /exports/qwen3-4b-nvfp4
--target-bpp 4.50 --achieved-bpp 4.49
--device cuda --dtype bf16
```

## Compare Records

```bash
python -m prismaquant.validation_harness compare \
  --candidate-id <new_record_id> \
  --baseline-id <shipped_record_id>
```

## List Records for a Model

```bash
python -m prismaquant.validation_harness list --model /hfcache/Qwen3-4B
```

## Metrics

- `ppl_wikitext`: perplexity over WikiText-2 raw test text. Lower is better.
- `mean_nll` / log-likelihood checks: report the underlying token NLL or
  task log-likelihood where available. Lower NLL and higher task likelihood
  are better.
- `ppl_mmlu_acc`: accuracy on a 200-question diverse MMLU sample. Higher is
  better.
- `end_kl`: KL on a fixed WikiText calibration batch using the same perturbed
  allocation machinery as the allocator. Lower is better.
- ToolEvalBench: run sequential hardmode ToolEvalBench against the served vLLM
  artifact with deterministic sampling (`--temperature 0 --seed 1234
  --no-think --hardmode --parallel 1`). Higher score/points are better; keep
  the full report and vLLM server log with the validation record.
- `model_sha`: SHA-256 of the model path contents when local, or the HF model
  reference string when remote.
- `layer_config_sha`: SHA-256 of canonical JSON for the layer config.

## Gate

Candidate ships only if `pass=True` vs the highest-rated existing record for
the same source model. `pass=True` requires strictly lower `end_kl`, no more
than 0.5% regression on WikiText perplexity or MMLU accuracy, no material
log-likelihood regression on the downstream task suite, and no ToolEvalBench
quality regression against the matched baseline unless explicitly accepted as
a bitrate/latency tradeoff.
