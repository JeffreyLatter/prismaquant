# HDQ Archive (2026-05-14)

Hadamard-DuQuant/HDQ was archived on 2026-05-14 after the native Hadamard and
learned dense rotation experiments failed the production bar.

Why it was retired:

- vLLM-compatible native Hadamard rotations had mixed results: ungated PPL
  improved versus saved vanilla NVFP4, but KL regressed versus vanilla NVFP4.
- Dense learned online rotations only loaded through a local vLLM/qutlass
  selector patch and fell onto an unusably slow path in smoke testing.
- Held-out gates rejected most native rotations, which means the search signal
  did not generalize reliably enough to justify production complexity.
- The solver was sensitive to calibration size, sampling, STE/prod-render
  mismatch, and per-cluster greedy decisions.

Contents:

- `prismaquant/`: archived HDQ implementation, allocator bridge, cache state,
  export transforms, joint search, and CLI entry point.
- `tests/`: HDQ-specific tests moved out of the active pytest tree.
- `tools/`: HDQ smoke, comparison, and diagnostic scripts.
- `docs/`: investigation handover and measured-loss notes.

The live pipeline now rejects `hadamard_duquant` production levers and does not
load HDQ sidecars, rotations, transform metadata, or cache routing. Reviving
this archive should be treated as a new research effort with fresh
apples-to-apples KL, PPL, bpp, runtime, and vLLM-kernel evidence.
