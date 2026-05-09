# Diverse Calibration v1

`diverse-v1.jsonl` is the default calibration recipe for the next PrismaQuant
phase. It is intentionally a local JSONL artifact so probes, cost tables, KL
validation, and exports can all consume the same text distribution.

## Build

```bash
python3 tools/build_diverse_calibration.py \
  --tokenizer /home/rob/.cache/huggingface/qwen36-27b-bf16 \
  --output /home/rob/dq-runs/calibration/diverse-v1.jsonl
```

The output has one manifest row followed by 256 `{"text": "..."}` rows, each
targeting about 4096 Qwen3.6 tokens. The existing probe loader skips the
manifest row because it has no `text` or `messages` field.

## Mix

- 40% prose: `HuggingFaceFW/fineweb-edu`, `sample-10BT`.
- 20% code: primary `bigcode/starcoderdata` (`python`, `c`, `cpp`); fallback
  `HSH-Intelligence/github-code-corpus-sample` filtered to Python/C/C++ when
  StarCoderData is gated.
- 20% math: primary `EleutherAI/proof-pile-2` algebraic-stack proof/code
  files; fallback `open-web-math/open-web-math`.
- 20% multilingual: requested `Helsinki-NLP/flores200` is not currently
  present on the Hub, and public FLORES200 mirrors are script-backed or gated
  in the installed loader. The builder records pinned FLORES200 metadata and
  uses pinned `facebook/xnli` `all_languages` as the reproducible fallback.

All source revisions are pinned in `MANIFEST` at the top of
[tools/build_diverse_calibration.py](/home/rob/prismaquant/tools/build_diverse_calibration.py).

## Pipeline

`run-pipeline.sh` defaults `DATASET` to:

```bash
/home/rob/dq-runs/calibration/diverse-v1.jsonl
```

For quick offline smoke runs before building the file, override:

```bash
DATASET=ultrachat_200k ./prismaquant/run-pipeline.sh
```
