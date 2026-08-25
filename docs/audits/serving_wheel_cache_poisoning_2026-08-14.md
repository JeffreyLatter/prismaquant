# A rejected wheel was cached under the pinned digest and bricked the serving lane

**Date:** 2026-08-14 · **Status:** CLOSED (fixed + regression-tested, v0.12.3)
**Files:** `prismaquant/gridbook_runtime/gridbook_serving_runtime.sh`,
`tests/test_gridbook_serving_runtime_pin.py`

## 1. Symptom

Every attempt to prepare the DSpark serving runtime failed with:

```
wheel SHA-256 e21185600a9dbbdb… differs from pin d085506a74869ac6…
```

and kept failing afterwards, including when a *correct* wheel was supplied.
Nothing in the message named the cache, so the natural response — supply the
right wheel — could not work.

## 2. Root cause — reproduced, not inferred

The serving-wheel cache is keyed by the **pinned digest**:
`$CACHE/<wheel_sha256>/<wheel>`. The materializer's first branch trusts that
directory name: if the directory exists it takes the single wheel inside and
verifies it, and **never consults a supplied wheel or a download**. So caching
an unverified wheel is not a wasted byte — it is permanent.

Pre-fix, the publish step was a bare call:

```bash
gridbook_serving_runtime_verify_wheel "${published[0]}" >/dev/null
if mv -T -- "$tmp" "$destination" 2>/dev/null; then
```

That verify's failure did not abort. The materializer's only caller reaches it
as `wheel="$(_gridbook_serving_materialize_wheel)" || return`, and **Bash
disables `errexit` for a command substitution whose enclosing command is part of
a `||` list — re-arming `set -e` inside that subshell does not restore it**. The
`set -euo pipefail` at the top of the function is therefore inert on the only
path that runs it. The rejection printed, the `mv` proceeded anyway, and the
lane was bricked. (The doubled error line in the logs is the tell: verify #1 was
ignored, then verify #2 ran after the `mv` and its status propagated.)

Reproduced deterministically: quarantining the cache entry and re-running
prepare recreated it, containing the rejected wheel.

## 3. Why the wrong wheel was there at all

The pin names the wheel **read out of the served image**
`gridbook:0.8.6-clean-dde15e0` — the runtime the accepted prefill/decode numbers
were taken on (`prismaquant/gridbook_serving_runtime_pin.py`). A plain
`pip download gridbook==0.8.6` fetches the **PyPI** wheel, which is a different
archive and does not satisfy the pin.

The two are **content-identical**: all 58 archive members byte-for-byte equal,
differing only in zip container metadata, and the PyPI wheel was built by the
release run from exactly the pinned commit `dde15e04`. Wheels are simply not
byte-reproducible.

**This defect was armed by publishing.** Before gridbook 0.8.6 existed on PyPI,
`pip download` failed outright and cached nothing. After the 2026-08-14 publish,
the download succeeded, and one prepare without a supplied wheel was enough to
brick the lane on that machine. See `docs/ARCHITECTURE.md` §"DSpark serving has
a second, current-consumer pin" for the unresolved policy question of *which*
wheel the digest should name.

## 4. Fix

Three sites now test the verification status explicitly rather than relying on
`errexit`, which is provably inert here:

- the **publish** step refuses to `mv` an unverified wheel into the cache;
- the **supplied-wheel** step fails with a clear message instead of leaving
  `$wheel` empty and surfacing a confusing `cp` error;
- the **fast path** names the offending directory:
  `cached wheel does not match the pin; remove <dir> and retry`.

Regression tests: `test_a_rejected_wheel_is_never_published_into_the_digest_cache`
drives the **download** path through a `pip download` shim, because that is the
only path where the defect fires — with a *supplied* mismatched wheel the
pre-fix code exits earlier on the empty-staging-directory check and the test
would be vacuous (it was, on the first draft, and passed against the unfixed
code). `test_a_poisoned_cache_entry_names_the_directory_to_remove` covers the
message. Both are mutation-proven against the true pre-fix file.

## 5. Operational note

The cache on this box was cleared and correctly repopulated from the pinned
image wheel, so **the lane is unbricked here**. Any run of the DSpark serving
lane must either supply
`GRIDBOOK_SERVING_RUNTIME_WHEEL=/home/rob/dq-runs/dsv4-flash-0731/mtp-throughput-research/gridbook-086-clean-dde15e0-image/gridbook-0.8.6-py3-none-any.whl`
or start from a correctly populated cache. The poisoned entry is preserved as
evidence at `/home/rob/dq-runs/spt-noenv-smoke/poisoned-cache-evidence/`.

## 6. Lessons

1. **`set -e` is inert inside `$( )` when the enclosing command is in a `||`
   list.** Any safety check on that path must test its status explicitly. This
   is a whole class of latent bug, not one line.
2. **A content-addressed cache keyed by an *expected* digest must never be
   populated before verification** — a bad entry is self-perpetuating, because
   the lookup that would replace it is the one being short-circuited.
3. **Publishing can arm a dormant bug.** The failure mode did not exist until
   the artifact became downloadable; "we changed nothing in that code path" was
   true and irrelevant.
4. **Wheels are not byte-reproducible.** A digest pin must name the artifact
   that will actually serve, and an error message about a digest must say
   *which file* is being rejected and where it came from.
