# Fold-Scale And OrthoG Archive

Archived on 2026-05-13.

This directory contains the removed AWQ-v2, SmoothQuant joint-format search,
and BlockOrtho-G experiments. It also includes pre-removal snapshots of the
production-cache and export files that wired those experiments into the live
path. They were removed from production cache, export, pipeline, allocator,
and render-score paths so future work can proceed on a Hadamard/DuQuant-style
path without accidentally composing these older input axis transforms.

Production code now rejects `awq`, `smoothquant`, and `block_rotation` levers.
The files here are research context only.
