# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Prompt-facing OSI core and DATUS extension authoring specifications."""

from __future__ import annotations

import re
from importlib import resources

from datus.tools.semantic_tools.exceptions import SemanticCoreException

_SPEC_RESOURCE = "osi-core-0.2.0.dev0.spec.yaml"
_SPEC_TITLE_MARKER = "# Apache Ossie - Core Metadata Spec"
_DIALECTS_BLOCK_RE = re.compile(
    r"(# Supported expression language dialects\ndialects:\n)"
    r'(?:  - "[^"]+"[^\n]*\n)+',
)

_NATIVE_NOTES = """\

---
# Dosi native authoring notes
# - Keep exactly one semantic_model object per file; Dosi does not merge model
#   fragments before validation or execution.
# - Every expression dialect must be `{dialect}` for this datasource.
# - Dosi execution metadata belongs in vendor_name: DATUS custom_extensions.
#   Use the active DATUS extension authoring specification for supported keys.
# - The native Dosi parser/compiler is authoritative after every mutation.
"""


def authoring_spec_text(dialect: str) -> str:
    """Render the vendored OSI core spec for the active SQL dialect."""

    raw = resources.files("datus.tools.semantic_tools.dosi.schema").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8")
    title_at = raw.find(_SPEC_TITLE_MARKER)
    if title_at >= 0:
        raw = raw[title_at:]
    replacement = (
        "# Supported expression language dialects\n"
        "dialects:\n"
        f'  - "{dialect}"              # the only dialect executed in this deployment\n'
    )
    rendered, substitutions = _DIALECTS_BLOCK_RE.subn(replacement, raw, count=1)
    if substitutions != 1:
        raise SemanticCoreException(
            "the vendored OSI core spec no longer exposes a recognizable "
            f"dialects block; re-check {_SPEC_RESOURCE!r} after updating it"
        )
    return rendered + _NATIVE_NOTES.format(dialect=dialect)


def datus_extension_authoring_spec_text(dialect: str = "<osi_dialect>") -> str:
    """Return the native engine's DATUS authoring contract."""

    from .engine import load_binding

    rendered = load_binding().render_datus_authoring_spec()
    if not isinstance(rendered, str) or not rendered.strip():
        raise SemanticCoreException("the installed dosi-engine returned an empty DATUS authoring contract")
    return rendered


def datus_extension_authoring_spec_digest() -> str:
    """Return a cache key for the active engine's authoring contract."""

    from .engine import datus_authoring_contract_digest

    return datus_authoring_contract_digest()
