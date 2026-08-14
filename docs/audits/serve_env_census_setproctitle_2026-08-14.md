# Serve environment census refused every correct server (`setproctitle`)

**Date:** 2026-08-14
**Status:** FIXED for future serves; **the DSv4-Flash 0731 artifact remains ungateable at its own build commit** (see §5).
**Severity:** Blocking — no serve slot on any lane could be closed against a live vLLM server.

## 1. Symptom

The DSv4 ship gate's `native_export.eager` arm refused, after the server had come
up cleanly (model loaded 95.12 GiB in 82.6 s, serve-manifest fingerprint
`9932b5127f88946c`, 14 resident extensions):

```
eager: CBEndpointValidationError: serve manifest environment is not the exact
live-process Gridbook contract; expected exact values for [16 names]
```

Artifact `artifact-aura-cb-112p69`, `model_sha`
`7968018ad458f96ac6ebebb73e81016a865af7c9fb7887f531cb22f3d234e2cd`, build commit
`3c23cf07`, image `eugr/spark-vllm@sha256:7bf752a9…`.

## 2. Evidence

`server_process_environment` in the eager serve manifest:

| pid   | role           | allowlisted vars seen | sha            |
|-------|----------------|-----------------------|----------------|
| 1     | container init | 16 / 16               | `d763e752558e` |
| 14038 | API server     | 16 / 16               | `d763e752558e` |
| 13287 | **EngineCore** | **1 / 16**            | `bb3f5acd81cc` |

`len(distinct) != 1` → `consensus = None` → `consistent: False` → refusal
(`tools/serve_fingerprint.py:366-432`).

## 3. Root cause — measured, not inferred

The census reads `/proc/<pid>/environ`. vLLM's EngineCore renames itself:
`vllm/v1/engine/core.py:1285` → `vllm.utils.system_utils.set_process_title`
(`:184-198`) → `setproctitle.setproctitle("VLLM::EngineCore_DP0")`. On Linux
`setproctitle` overwrites the contiguous argv+envp memory block, which
**destroys `/proc/<pid>/environ`** while leaving the process's real `os.environ`
untouched.

Measured inside the pinned image, same process, across that one call:

```
BEFORE  /proc/self/environ: all 6 probed vars    os.environ: all 6
AFTER   /proc/self/environ: []                   os.environ: all 6
=> /proc lost: all 6.   => os.environ lost: none.
```

pid 1 and the API server never rename themselves, so their `/proc/environ`
survives intact and identical. The single surviving
`PRISMAQUANT_CB_GROUPED_TRIM` on EngineCore is leftover bytes, not a
configuration fact.

**The artifact was never implicated.** The environment did reach EngineCore;
`os.environ` — what the CB kernels read — was correct throughout. This also
settles a question raised while diagnosing: the accepted PB0-vs-PB A/B compared
genuinely different configurations (prefill 604/729/707, 1.61–1.91× over PB0;
decode +8.62%, 18.442 vs 16.979). None of that evidence is affected.

## 4. Scope — it was structural, not lane-specific

- `validate_cb_endpoint.py:460` imports the general `SERVER_ENV_ALLOWLIST`. The
  newer `DSPARK_SERVER_ENV_ALLOWLIST` (`dspark_serving_profile.py:308`) changes
  *which names* are compared, not *how they are read*, and
  `dspark_serving_profile.py:1560` calls the same validator.
- `validate_cb_performance.py:1524` independently requires `consistent is True`
  and `unreadable_pids == []`.
- No launcher sets `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so EngineCore is always a
  separate, self-renamed process.

So the check could not be satisfied by any artifact, on any lane, at any commit.
It was a first-exercise failure: earlier runs died at the snapshot ledger and the
venv before ever reaching it, so it had never once been green — the same shape as
the probe that crashed on every invocation behind a green suite.

## 5. Why this does not unblock the 0731 bytes

The gate is bound to the artifact's own build commit: it refuses unless the serve
checkout is clean at `ARTIFACT_BUILD_COMMIT`
(`scripts/serve_dsv4_cb_validate.sh:79-81`), re-execs itself from a
content-addressed snapshot of that commit (`:141`), and that snapshot contains
`scripts/` and `tools/` as well as `prismaquant/`. `GRIDBOOK_RUNTIME_DOCKER_ARGS`
is *assigned* inside `gridbook_runtime_prepare`, not appended, so it is not a
caller-extensible hook either.

That binding is working exactly as designed, and the consequence is exact: **no
change, however correct, can make the 0731 artifact's gate pass.** Closing its
slots requires rebuilding at a commit that carries this fix, which produces a new
`model_sha` and detaches the accepted perf evidence from the shipped bytes. That
is a deliberate decision and is Rob's, not the gate's.

## 6. Fix

Both runtime Docker vectors now set `SPT_NOENV=1`
(`prismaquant/gridbook_runtime/gridbook_runtime.sh`,
`gridbook_serving_runtime.sh`), which confines the process title to the argv area.
The title truncates to argv length (`VLLM::En`), which is cosmetic; kernels,
memory and numerics are untouched.

**This is not a relaxation.** The census still compares every allowlisted name's
value exactly — `SPT_NOENV` only makes those values legible, so a genuinely
mismatched EngineCore environment still fails. It is deliberately excluded from
every compared allowlist: a variable whose purpose is to make the measurement
possible must not become part of the thing measured.

Guarded by `tests/test_gridbook_runtime_boundary.py`
(`test_both_runtime_vectors_keep_proc_environ_readable`, plus the prepared-vector
assertion), mutation-checked: deleting the `SPT_NOENV` line fails the guard.

## 7. Lesson

The house question — *measurement gap or optimizer gap?* — applied to a gate.
A refusal is evidence about the measurement as much as about the artifact, and
the first move on a red gate is to check that the check can be satisfied at all.
This one never could. Related: **verify a gate has been green at least once
before treating its refusal as a finding about the thing under test.**
