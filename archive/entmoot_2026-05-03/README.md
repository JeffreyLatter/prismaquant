# Entmoot Archive

This directory preserves the Entmoot MoE merge experiments without wiring them
into the active PrismaQuant runtime.

Contents:

- `source/prismaquant/`: archived Entmoot and expert-merge prototype modules.
- `source/tests/`: tests that were written for the prototype modules.
- `examples/launchers/`: launcher and validator scripts used during the
  experiments.
- `patches/active-integration.diff`: patch containing the active-code wiring
  that was intentionally not kept in `prismaquant/`.
- `scratch_summaries/`: lightweight JSON/JSONL experiment summaries. Large
  tensor artifacts from `scratch/entmoot` were intentionally omitted.

The archive is for reference and future resurrection only. Import paths here
are not package modules, and no production code imports from this tree.
