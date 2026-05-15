"""Archive stub for the HALO module.

The full HALO implementation lives in ``archive/halo_2026-05-15/prismaquant/halo.py``
after the 2026-05-15 consolidation — HALO never demonstrated a reproducible
non-regressive measured-KL win on production models and was removed from the
recommended stack.

Conditional ``if args.halo_mode == "random":`` branches in
build_production_cache, production_recache, validate_assignments_kl, and
export_native_compressed still import names from this module. They resolve
here, but any actual call raises ``HaloArchived`` so an explicit
``--halo-mode random`` invocation fails immediately.

To resurrect: copy ``archive/halo_2026-05-15/prismaquant/halo.py`` back
over this file.
"""
from __future__ import annotations


class HaloArchived(RuntimeError):
    """Raised when archived HALO code is invoked from production paths."""


_ARCHIVE_MESSAGE = (
    "HALO is archived under archive/halo_2026-05-15/ and is not available "
    "on the production path. To use it for research, restore "
    "archive/halo_2026-05-15/prismaquant/halo.py over prismaquant/halo.py."
)


def _archived(*_args, **_kwargs):
    raise HaloArchived(_ARCHIVE_MESSAGE)


apply_random_halo_to_model = _archived
apply_halo_to_head = _archived
apply_halo_to_layer = _archived
halo_hidden_from_config = _archived
halo_config_bool = _archived
validate_halo_export_support = _archived
halo_metadata = _archived
random_hadamard = _archived
