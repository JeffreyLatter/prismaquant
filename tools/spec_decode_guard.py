#!/usr/bin/env python3
"""Spec-decode refusal for the gold lane.

`validate_quantized_model.py` has scraped `/metrics` for `vllm:spec_decode` and
refused to return a verdict since the draft-logprobs postmortem — but that guard
lived *only* there. The gold lane (`tools/measure_vllm_full_kl.py`,
`tools/measure_vllm_wikitext_ppl.py`), which is metric authority rung 1, had no
such guard at all (R13).

The gold tools construct their own in-process `LLM`, so the detection is the
engine's own config rather than a Prometheus scrape; both forms live here so the
refusal text and semantics stay identical.

Returned/recorded `spec_decode_detected`:

* `False` — inspected and spec-decode is off (the only state the shipcard accepts
  on a `gold.*` record)
* `True`  — refuse: vLLM routes echo/prompt logprobs through the DRAFT model, so
  the NLL/KL would be the 1-layer MTP head's, not the artifact's
* `None`  — could not inspect. Not treated as "off": `tools/shipcard.py verify`
  refuses an unknown, because an unverified negative is what the original trap
  looked like.
"""
from __future__ import annotations

import urllib.request
from typing import Any

REFUSAL = (
    "spec-decode is active on this serve. vLLM routes echo/prompt logprobs "
    "through the DRAFT model when speculative decoding is configured, so the "
    "numbers this tool would produce are the 1-layer MTP head's, NOT the "
    "artifact's. Re-run against a serve WITHOUT --speculative-config (use a "
    "second serve for MTP acceptance). See docs/ARCHITECTURE.md 7.5."
)


def spec_decode_from_metrics(base_url: str, timeout: float = 15.0) -> bool | None:
    """True iff `/metrics` exposes the `vllm:spec_decode_*` family.

    vLLM registers those counters at startup whenever spec-decode is
    configured, before any draft runs — their presence is a config-time signal.
    Returns None when `/metrics` is unreachable (unknown, not "off").
    """
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout) as r:
            text = r.read().decode("utf-8")
    except Exception:
        return None
    return "vllm:spec_decode" in text


def spec_decode_from_engine(llm: Any) -> bool | None:
    """True iff a live in-process `LLM` was built with a speculative config."""
    engine = getattr(llm, "llm_engine", None) or llm
    for owner_name in ("vllm_config", "engine_config", "config"):
        owner = getattr(engine, owner_name, None)
        if owner is None:
            continue
        if hasattr(owner, "speculative_config"):
            return getattr(owner, "speculative_config") is not None
    if hasattr(engine, "speculative_config"):
        return getattr(engine, "speculative_config") is not None
    return None


def refuse_if_spec_decode(
    *,
    llm: Any | None = None,
    base_url: str | None = None,
    allow: bool = False,
    context: str = "gold lane",
) -> bool | None:
    """Detect, print, and raise unless `allow`. Returns the detection result."""
    detected: bool | None = None
    if base_url:
        detected = spec_decode_from_metrics(base_url)
    if detected is None and llm is not None:
        detected = spec_decode_from_engine(llm)

    if detected is True:
        message = f"[{context}] REFUSED: {REFUSAL}"
        if not allow:
            raise SystemExit(message)
        print(f"{message}\n[{context}] --allow-spec-decode given: continuing "
              "with a number that is NOT the artifact's", flush=True)
    elif detected is None:
        print(f"[{context}] WARN could not determine whether spec-decode is "
              "active; the shipcard will refuse this record until it can",
              flush=True)
    return detected
