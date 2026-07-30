"""vLLM out-of-tree plugin for the NVFP4-CB / FP8-CB product-codebook formats.

Registers a ``QuantizationConfig`` (``"gridbook"``, with ``"prismaquant"`` kept
as a legacy alias) plus the linear and fused-MoE methods that serve
codebook-quantized weights. The CUDA kernels ship as sources under
``gridbook/csrc`` and are JIT-compiled on first use; without nvcc the plugin
falls back to a correct-but-slow Triton path.

``register`` is lazy so ``import gridbook.codec`` / ``import gridbook.kernels``
(the format and correctness tests) work without vLLM installed.
"""

# Development head. PyPI serves 0.1.1 (built from the standalone /home/rob/gridbook
# release repo); this in-tree copy is AHEAD of it in kernel work and is released
# only through that repo, so it carries a dev suffix rather than a release number.
__version__ = "0.2.0.dev0"


def register() -> None:
    from .plugin import register as _register
    _register()
