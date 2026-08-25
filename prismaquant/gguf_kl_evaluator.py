"""GGUF lane KL evaluator: `llama-perplexity --kl-divergence-base`, behind the
`validate_assignments_kl` interface.

Re-vet **R16** (`LaneSpec`), which closes the measurement half of **D26**.

**Why an adapter and not the resident evaluator.** `validate_assignments_kl`
measures KL by materialising rendered weights into a live torch model. A GGUF
artifact cannot be materialised that way without leaving the format under test —
its k-quant / IQ blocks are what is being measured, and llama.cpp's dequant is
part of the artifact's behaviour. The lane's own harness is already the right
instrument (`docs/lanes/gguf.md`); what was missing is that its numbers came
back as console text instead of as rows a selector can rank.

**The contract this satisfies.** `measure_assignment_kl` returns
`(mean, per_sequence, stats)` and `stats` uses the **gold lane's key names**
(`kl_mean`, `kl_p99`, `kl_max`, `nll_mean`) so a GGUF row and a native row are
directly comparable for the first time — the same reason R9 chose those names.

**What it honestly cannot give.** `llama-perplexity` reports *aggregate*
quantiles over tokens, not per-sequence values. `per_sequence` is therefore
empty and `stats["kl_tail_domain"] = "aggregate"` — the tail keys are the
harness's own token quantiles, NOT recomputed from samples we hold. A tail-veto
(R9) reading `kl_p99` on this lane is reading a token-domain statistic; that is
recorded rather than papered over.

Parsing, not running, is the testable part, so it is a pure function over the
harness's stdout and is pinned against canned output from both spellings
llama.cpp has shipped. Running the binary is the integration path.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Both spellings llama.cpp has shipped for the KL block: the current
#: "Mean    KLD:" and the older "Mean    KL divergence         :".
_KLD = r"(?:KLD|KL divergence)"
_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

_PATTERNS: dict[str, str] = {
    "kl_mean": rf"^\s*Mean\s+{_KLD}\s*:\s*{_NUM}",
    "kl_max": rf"^\s*Maximum\s+{_KLD}\s*:\s*{_NUM}",
    "kl_p999": rf"^\s*99\.9%\s+{_KLD}\s*:\s*{_NUM}",
    "kl_p99": rf"^\s*99\.0%\s+{_KLD}\s*:\s*{_NUM}",
    "kl_median": rf"^\s*Median\s+{_KLD}\s*:\s*{_NUM}",
    "kl_min": rf"^\s*Minimum\s+{_KLD}\s*:\s*{_NUM}",
    "ppl_q": rf"^\s*Mean PPL\(Q\)\s*:\s*{_NUM}",
    "ppl_base": rf"^\s*Mean PPL\(base\)\s*:\s*{_NUM}",
    "top1_agreement_pct": rf"^\s*Same top p\s*:\s*{_NUM}",
}

#: stderr columns: llama.cpp prints `value ± stderr` for a few statistics.
_STDERR_PATTERNS: dict[str, str] = {
    "kl_stderr": rf"^\s*Mean\s+{_KLD}\s*:\s*{_NUM}\s*(?:±|\+/-)\s*{_NUM}",
    "top1_agreement_stderr_pct": (
        rf"^\s*Same top p\s*:\s*{_NUM}\s*(?:±|\+/-)\s*{_NUM}"),
}


class LlamaPerplexityParseError(ValueError):
    """The harness produced output with no KL block — a run that failed, was
    invoked without `--kl-divergence`, or a binary whose format changed."""


def parse_llama_perplexity_kl(text: str) -> dict[str, float]:
    """Parse `llama-perplexity --kl-divergence` output into a flat dict.

    Tolerant of both known spellings and of the `±` / `+/-` variants. Absent
    fields are simply absent — no zero-filling, because a missing statistic and
    a statistic that measured zero are different facts.
    """
    out: dict[str, float] = {}
    for key, pattern in _PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    for key, pattern in _STDERR_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            out[key] = float(m.group(2))
    if "kl_mean" not in out:
        raise LlamaPerplexityParseError(
            "no KL block in llama-perplexity output — was it run with "
            "--kl-divergence --kl-divergence-base <base_logits.bin>?"
        )
    return out


def build_llama_perplexity_command(
    *,
    model: str | Path,
    base_logits: str | Path,
    corpus: str | Path | None = None,
    chunks: int = 64,
    n_gpu_layers: int = 99,
    binary: str | Path = "llama-perplexity",
    extra_args: Sequence[str] = (),
) -> list[str]:
    """The lane's KL command, as declared in `docs/lanes/gguf.md`."""
    cmd = [
        str(binary),
        "-m", str(model),
        "--kl-divergence-base", str(base_logits),
        "--kl-divergence",
        "--chunks", str(int(chunks)),
        "-ngl", str(int(n_gpu_layers)),
    ]
    if corpus is not None:
        cmd += ["-f", str(corpus)]
    cmd += [str(a) for a in extra_args]
    return cmd


def kl_stats_from_parsed(parsed: Mapping[str, float]) -> dict[str, Any]:
    """Map the harness's fields onto the gold lane's key names.

    Deliberately the SAME names `_kl_repeat_summary` / the gold lane emit, so a
    GGUF row drops straight into a frontier table. `kl_tail_domain` records
    that the sample unit here is a TOKEN and the quantiles are the harness's,
    not ours — the honest counterpart to R9's `"sequence"`.
    """
    stats: dict[str, Any] = {
        "kl_mean": float(parsed["kl_mean"]),
        # Deprecated alias kept for one cycle, exactly as the resident path.
        "last_token_kl": float(parsed["kl_mean"]),
        "kl_tail_domain": "aggregate",
        "kl_evaluator": "llama_perplexity",
    }
    for src, dst in (
        ("kl_stderr", "kl_stderr"),
        ("kl_p99", "kl_p99"),
        ("kl_p999", "kl_p999"),
        ("kl_max", "kl_max"),
        ("kl_median", "kl_median"),
        ("kl_min", "kl_min"),
        ("ppl_q", "ppl"),
        ("ppl_base", "ppl_base"),
        ("top1_agreement_pct", "top1_agreement_pct"),
        ("top1_agreement_stderr_pct", "top1_agreement_stderr_pct"),
    ):
        if src in parsed:
            stats[dst] = float(parsed[src])
    if "ppl" in stats and stats["ppl"] > 0:
        # A PPL-family rung-2 statistic in the same row, for free: NLL is
        # ln(PPL) by definition, so this is a rename, not a new measurement.
        stats["nll_mean"] = float(math.log(stats["ppl"]))
    return stats


def measure_assignment_kl(
    *,
    model: str | Path,
    base_logits: str | Path,
    corpus: str | Path | None = None,
    chunks: int = 64,
    n_gpu_layers: int = 99,
    binary: str | Path = "llama-perplexity",
    extra_args: Sequence[str] = (),
    output_text: str | None = None,
    timeout: float | None = None,
) -> tuple[float, list[float], dict[str, Any]]:
    """`validate_assignments_kl`-shaped result for a GGUF artifact.

    Returns `(mean, per_sequence, stats)`. `per_sequence` is **empty** by
    construction (see the module docstring). Pass `output_text` to parse
    already-captured harness output — that is the path unit tests take, and the
    path a re-analysis of an archived log takes; otherwise the binary runs.
    """
    if output_text is None:
        cmd = build_llama_perplexity_command(
            model=model, base_logits=base_logits, corpus=corpus,
            chunks=chunks, n_gpu_layers=n_gpu_layers, binary=binary,
            extra_args=extra_args,
        )
        proc = subprocess.run(
            cmd, text=True, capture_output=True, timeout=timeout, check=False)
        # llama-perplexity writes its statistics block to stderr on some
        # builds and stdout on others; concatenate rather than guess.
        output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 and "kl_mean" not in output_text:
            raise RuntimeError(
                f"llama-perplexity exited {proc.returncode}: "
                f"{(proc.stderr or '')[-2000:]}"
            )
    parsed = parse_llama_perplexity_kl(output_text)
    stats = kl_stats_from_parsed(parsed)
    return float(stats["kl_mean"]), [], stats


def frontier_row(
    label: str,
    bpp: float,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """One `select_validated_frontier` row from a parsed GGUF measurement.

    `measured_rows` whitelists its columns, so this emits exactly the names it
    reads (`label`, `bpp`, `kl`) plus the tail columns the R9 veto looks for.
    """
    row: dict[str, Any] = {
        "label": str(label),
        "bpp": float(bpp),
        "kl": float(stats["kl_mean"]),
        "kl_mean": float(stats["kl_mean"]),
    }
    for key in ("kl_stderr", "kl_p99", "kl_max", "nll_mean",
                "kl_tail_domain", "kl_evaluator", "ppl"):
        if key in stats:
            row[key] = stats[key]
    return row
