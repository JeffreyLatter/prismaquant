# Fork Patches

Six numbered patches representing all local fork edits on top of `upstream/main`
(https://github.com/RobTand/prismaquant).  Each patch is a single logical change;
the series applies cleanly to any `upstream/main` checkout.

## Applying to a fresh upstream clone

```bash
git clone https://github.com/RobTand/prismaquant my-fork
cd my-fork
git am patches/0001-*.patch
git am patches/0002-*.patch
git am patches/0003-*.patch
git am patches/0004-*.patch
git am patches/0005-*.patch
git am patches/0006-*.patch
```

Or apply all at once:

```bash
git am patches/*.patch
```

If a patch has a conflict (upstream changed the same lines), resolve it, then
`git add <file> && git am --continue`.

## Re-generating patches after new fork work

After adding new commits to the `fork-edits` branch:

```bash
git fetch upstream
git format-patch upstream/main..fork-edits -o patches/
```

## Keeping fork-edits current with upstream

When upstream releases new commits:

```bash
git fetch upstream
git checkout fork-edits
git rebase upstream/main
# resolve any conflicts, then:
git format-patch upstream/main..fork-edits -o patches/
```

## Patches

| File | Section | What it does |
|---|---|---|
| 0001 | Section 1 | MoE expert coverage false-positive fix |
| 0002 | Section 2 | GB10 CUDA mask default off |
| 0003 | Section 3 | Phase-1 OOM fix on dense 27B+ models |
| 0004 | Section 4 | test-pipeline.sh local harness |
| 0005 | Section 5 | Qwen3-Coder-Next (qwen3_next) model profile |
| 0006 | Section 6 | Streaming production cache for oversized models |

See `FORK_EDITS.md` in the repo root for the design rationale behind each section.

## fla source changes (Section 2, external)

The GB10 allocator and TileLang guard fixes live in `/tmp/fla-src` (editable
install), not in this repo.  They are documented in FORK_EDITS.md Section 2 but
cannot be captured as git patches here.  Re-apply them manually after a fresh
`pip install -e /path/to/fla-src`.
