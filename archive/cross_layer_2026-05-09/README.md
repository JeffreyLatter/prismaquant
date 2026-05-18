# Cross-Layer Research Archive

This archive preserves the shelved CLADO and SMRF ports plus their validation
notes. They are research context only and are not imported or registered by
the production package.

Contents:

- `prismaquant/research_components/block_clado.py`
- `prismaquant/research_components/block_clado_runtime.py`
- `prismaquant/research_components/smrf.py`
- `prismaquant/research_components/smrf_runtime.py`
- `tests/test_block_clado_runtime.py`
- `tests/test_smrf_runtime.py`
- `docs/qwen3_4b_clado_assignment_optimization.md`
- `docs/qwen3_4b_smrf_optimization.md`
- `docs/qwen3_27b_smrf_validation.md`

Status:

- CLADO was made runtime-legal over fused siblings and layer/subblock scopes,
  but measured KL and vLLM perplexity regressed versus standard PQ at matched
  bitrate.
- SMRF generated plausible low-budget candidates, but the more rigorous 27B
  validation rejected the refined winner: matched standard PQ had lower KL,
  and vLLM perplexity did not improve.

To revive either path, copy it out of archive as a new opt-in research
component and re-run the production validation gate with resident
`ProductionWeightCache`, measured KL repeats, held-out PPL/mean NLL,
log-likelihood checks, ToolEvalBench, and vLLM eager/graph smokes.
