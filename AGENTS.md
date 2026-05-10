# PrismaQuant Agent Rules

These rules are mandatory for coding agents working in this repository.
Before implementing new functionality, read this file and
`docs/design_guidelines.md`.

## Core Principles

1. **GPU-bound by default.** Production probes, cache fills, recache,
   polish, export, and validation must be designed so the GPU is the
   bottleneck. A hot path that is CPU-bound, disk-bound, or NVMe-bound is a
   bug unless the user explicitly requested an offline data-prep step.
2. **Use the existing cache and prefetch system.** Do not create a parallel
   cache, preload, or residency mechanism for rendered weights or
   activations. Extend `ProductionWeightCache`, `PerturbedActivationCache`,
   the streaming model prefetch path, or the existing pipeline wiring.
   Production paths should fail fast when required resident data cannot be
   prefetched instead of silently streaming from NVMe.
3. **Right quantization for the right layer.** Preserve PrismaQuant's core
   contract: per-Linear empirical selection with measured quality/cost
   tradeoffs. Avoid model-wide defaults unless they are only a fallback or
   have been validated against the per-Linear path.
4. **Only ship performant vLLM-supported formats.** A format can appear in
   research menus before it is production-ready, but it must not become a
   production default until vLLM loads it, generates correctly, and uses a
   performant kernel on representative shapes.
5. **Measure on the same calibration contract.** New levers need apples-to-
   apples KL, bpp, and runtime measurements. Compare against the relevant
   shipped or current baseline using the same calibration set, sequence
   length, layer assignment semantics, and production cache behavior.
6. **Reuse local abstractions.** Prefer the existing format registry,
   allocator, production cache, recache, validation harness, and pipeline
   flags. If an abstraction is missing, add it at the shared layer rather
   than building a one-off call site.
7. **Keep cross-layer machinery archived unless explicitly requested.**
   The archived CLADO, propagated-cost, output-Fisher, PrismaSCOUT iteration,
   QUBO, and polish-of-many code is research context, not a production
   shipping lever.

## Implementation Checklist

Before editing:

- Identify the existing mechanism this change should extend.
- Decide how the change stays GPU-bound and resident-prefetched.
- Define the vLLM compatibility gate if formats, export metadata, kernels,
  or compressed-tensors layout are touched.
- Define the KL/bpp/runtime comparison and calibration set.

Before finishing:

- Run targeted tests and compile checks for touched modules.
- Add or update tests for new policies, format gates, cache residency, or
  validation behavior.
- Record measured results, commands, and log paths in docs when a claim is
  based on a run.
- Leave experimental methods opt-in until the validation gate in
  `docs/design_guidelines.md` is satisfied.

