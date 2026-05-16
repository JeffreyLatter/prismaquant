# ReSpinQuant Archive

Archived on 2026-05-13.

This directory preserves the ReSpinQuant and residual-adapter investigation as
research context. It is no longer part of the live PrismaQuant production
surface.

Rationale:

- Full layer-wise ReSpinQuant changes the residual-stream basis between layers.
  Without runtime residual-basis adapter code, that transform cannot be exported
  as a vanilla vLLM compressed-tensors artifact.
- The optional residual-adapter path requires custom serving code. That violates
  the current production constraint unless the runtime/plugin support decision
  is explicitly reopened.
- The `.8B` trained smoke did not improve the render score after PrismaQuant's
  existing local mechanisms. It produced a worse ReSpin static score than the
  non-rotation baseline, and rank-32 transition approximation error remained too
  high for a credible production candidate.
- Keeping this code live made the pipeline and docs imply a path forward that is
  not viable under the no-custom-runtime-code constraint.

Contents:

- `prismaquant/respinquant.py` compatibility-scout logic.
- `prismaquant/respinquant_core.py` SpinQuant/ReSpin-family training math.
- `prismaquant/residual_adapter.py` residual-adapter manifest helpers.
- `tools/train_respinquant_rotations.py` GPU-only simplified rotation trainer.
- `tools/create_respin_equivalent_variant.py` research materializer.
- `tools/create_residual_adapter_variant.py` residual-adapter artifact creator.
- `tools/respinquant_scout.py` compatibility scout.
- `tools/respin_render_attribution.py` local attribution replay.
- `tests/test_respinquant*.py` and `tests/test_residual_adapter.py`.
- `docs/` result notes and plugin documentation from the investigation.

Live replacement:

- None. ReSpinQuant is shelved. Continue with vanilla-vLLM-safe numerical
  methods in the production cache and allocator path: per-Linear format
  selection, AWQ, GPTQ/fisher-GPTQ, FourOverSix, scale sweep, activation
  recache, and validated frontier selection.

Reopening criteria:

- A runtime-support decision explicitly allows the residual adapter in serving,
  or the method is reformulated so the exported graph remains vanilla vLLM.
- The implementation is measured against the current production stack with the
  same calibration contract and clears the normal KL/bpp/runtime gates.
