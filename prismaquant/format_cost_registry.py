"""Reference :class:`FormatCostPlugin` implementations backed by the real registry.

These are not a parallel model of what a format does -- they call
``FormatSpec.quantize_dequantize`` and ``FormatSpec.activation_quantize_dequantize``
directly, so the error a plugin reports is produced by the same code that
renders. That is the same rendering-identity requirement that makes the
surrogate, the KL validation and the exported bytes comparable; a plugin that
re-implemented a format's rounding would reintroduce exactly the confound the
"one cache mechanism" rule exists to prevent.

Because the registry already declares ``act_bits`` and
``act_quant_changes_input`` per format, AQUA-AURA needs no new format metadata:
``NVFP4`` (W4A4) and ``NVFP4A16`` (W4A16) are already distinct entries with
identical weight bits, which is the cleanest possible demonstration that the
activation term is doing the work.

The activation error is *measured* through the format's own activation
quantizer, applied to a synthetic activation whose per-channel scale is taken
from the card, rather than assumed to be a uniform grid. Caveat, stated plainly:
real activation quantizers are often per-token or per-group, so driving them
with a per-channel synthetic sample approximates the grouping. This is a
screening surrogate, not a served measurement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .format_registry import FormatSpec, get_format
from .format_cost_protocol import FormatDescriptor
from .sensitivity_card import SensitivityUnit

# Formats that copy an already-matching source tensor verbatim. The allocator
# may pick these only when the source is already that precision; synthesizing
# BF16 from a dequantized FP8 would waste 8 bpp.
PASSTHROUGH_SOURCE_DTYPE = {
    "BF16": "bfloat16",
    "FP8_SOURCE": "float8_e4m3fn",
    "FP8_BLOCK_UE8M0_SOURCE": "float8_e4m3fn",
    "MXFP4_SOURCE": "float4_e2m1fn",
}

#: The production menu. Deliberately small: these are the formats vLLM serves
#: natively today. An author targeting another platform passes their own list.
PRODUCTION_MENU = ("NVFP4", "FP8_E4M3", "BF16", "FP8_SOURCE")

#: A menu that isolates the activation axis: same weight format, different
#: activation handling. This is the AQUA-AURA A/B in menu form.
ACTIVATION_AXIS_MENU = ("NVFP4", "NVFP4A16", "FP8_E4M3", "MXFP8A16")


def descriptor_for(spec: FormatSpec, *, shape: tuple[int, int],
                   speed_index: float | None = None) -> FormatDescriptor:
    """Adapt a registry ``FormatSpec`` into the costing seam's descriptor.

    ``effective_bits_for_shape`` is used rather than ``weight_bits`` because the
    byte budget spends *stored* bits including scale/codebook overhead, and
    because codebook formats report ``weight_bits == 0``.
    """
    # act_quant_changes_input is THE predicate (format_registry defines it);
    # it is carried across as explicit data rather than re-derived downstream.
    quantizes = bool(spec.act_quant_changes_input)
    return FormatDescriptor(
        name=spec.name,
        weight_bits=float(spec.effective_bits_for_shape(shape)),
        act_bits=spec.act_bits if quantizes else None,
        quantizes_activations=quantizes,
        group_size=spec.group_size or None,
        passthrough=spec.name in PASSTHROUGH_SOURCE_DTYPE,
        requires_source_dtype=PASSTHROUGH_SOURCE_DTYPE.get(spec.name),
        speed_index=speed_index,
    )


@dataclasses.dataclass
class RegistryFormatPlugin:
    """Price a registry format using its own quantizers.

    ``device`` matters: this is a GPU-first codebase and the RTN kernels are
    written for it. Costing a full model on CPU is a bug, not a fallback.
    """

    descriptor: FormatDescriptor
    spec: FormatSpec
    device: str = "cuda"
    act_samples: int = 256
    seed: int = 0

    @classmethod
    def build(cls, name: str, *, shape: tuple[int, int], device: str = "cuda",
              speed_index: float | None = None) -> "RegistryFormatPlugin":
        spec = get_format(name)
        return cls(descriptor=descriptor_for(spec, shape=shape,
                                             speed_index=speed_index),
                   spec=spec, device=device)

    # ------------------------------------------------------------ weight side

    def weight_error(self, unit: SensitivityUnit,
                     weight: np.ndarray) -> np.ndarray:
        """Elementwise squared weight error under the format's own RTN render.

        Passthrough formats are lossless by construction, so their error is
        exactly zero -- not "small", zero. Returning a measured epsilon there
        would let float noise decide a passthrough-vs-quantized comparison.
        """
        if self.descriptor.passthrough:
            return np.zeros((unit.out_features, unit.in_features),
                            dtype=np.float32)
        w = torch.as_tensor(np.asarray(weight), dtype=torch.bfloat16,
                            device=self.device)
        with torch.no_grad():
            q = self.spec.quantize_dequantize(w)
            err = (w.float() - q.float()) ** 2
        return err.cpu().numpy()

    # -------------------------------------------------------- activation side

    def activation_error_variance(self, unit: SensitivityUnit,
                                  ) -> np.ndarray | None:
        """Per-input-channel variance of this format's activation-quant error.

        Measured by pushing a synthetic activation through the format's own
        ``activation_quantize_dequantize``. The synthetic sample is scaled per
        channel to the card's measured second moment and clipped to its measured
        absmax, so the quantizer sees the dynamic range the channel actually
        spans -- which is what drives activation error.

        Returns None when the format does not quantize activations, or when the
        card carries no activation statistics. None is not zero: an unmeasured
        activation cost must never read as a free one.
        """
        if not self.descriptor.quantizes_activations:
            return None
        if unit.act_sq_sum is None:
            return None

        sigma = np.sqrt(np.asarray(unit.act_sq_sum, dtype=np.float64)
                        / max(1, unit.n_tokens))
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        base = torch.randn(self.act_samples, unit.in_features,
                           generator=gen, dtype=torch.float32)
        x = base * torch.as_tensor(sigma, dtype=torch.float32)
        if unit.act_absmax is not None:
            cap = torch.as_tensor(np.asarray(unit.act_absmax),
                                  dtype=torch.float32)
            x = torch.clamp(x, -cap, cap)

        x = x.to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            xq = self.spec.activation_quantize_dequantize(x)
            err = (x.float() - xq.float()) ** 2
            per_channel = err.mean(dim=0)
        out = per_channel.cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(out)):
            return None
        return out


def build_menu(names, *, shape: tuple[int, int], device: str = "cuda"):
    """Build plugins for a named menu. This is the whole 'arbitrary menu' story."""
    return [RegistryFormatPlugin.build(n, shape=shape, device=device)
            for n in names]
