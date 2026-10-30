# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Prompt-facing OSI core and DATUS extension authoring specifications."""

from __future__ import annotations

import re
from importlib import resources
from typing import Any
from urllib.parse import quote

import yaml

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


def compact_authoring_spec_text(dialect: str) -> str:
    """Keep every vendored core shape and enum without its explanatory prose."""
    documents = [document for document in yaml.safe_load_all(authoring_spec_text(dialect)) if document is not None]
    return yaml.safe_dump_all(documents, sort_keys=False, allow_unicode=True, width=110)


def _capability_reference_name(key: str) -> str:
    return f"capability-{quote(key, safe='')}.yaml"


def authoring_overview_text() -> str:
    """Render all active capabilities without eagerly injecting their definitions."""
    from .engine import datus_authoring_contract

    contract = datus_authoring_contract()
    overview = {
        key: value
        for key, value in contract.items()
        if key not in {"capabilities", "definitions", "examples", "authoring_guidance"}
    }
    overview["capabilities"] = [
        {
            **{key: row[key] for key in ("key", "extension", "carriers")},
            "reference": _capability_reference_name(row["key"]),
        }
        for row in contract["capabilities"]
    ]
    overview["authoring_guidance"] = contract["authoring_guidance"]
    return yaml.safe_dump(overview, sort_keys=False, allow_unicode=True, width=110)


def _capability_spec(contract: dict[str, Any], capability: str) -> dict[str, Any]:
    """Select complete native rules and resolve their schema references."""
    supported = {row["key"] for row in contract["capabilities"]}
    selected = {capability}
    details = {
        key: value
        for key, value in contract.items()
        if key not in {"capabilities", "carriers", "definitions", "examples", "authoring_guidance"}
    }
    details["contract_digest"] = datus_extension_authoring_spec_digest()
    details["capabilities"] = [row for row in contract["capabilities"] if row["key"] in selected]
    details["carriers"] = {
        carrier: {key: rule for key, rule in rules.items() if key in selected}
        for carrier, rules in contract["carriers"].items()
        if selected.intersection(rules)
    }
    definitions = {}

    def resolve_references(value: Any) -> None:
        if isinstance(value, dict):
            ref = value.get("schema_ref")
            if ref:
                prefix, _, name = str(ref).partition(".")
                if prefix != "definitions" or name not in contract["definitions"]:
                    raise SemanticCoreException(f"Unresolved authoring schema reference: {ref}")
                if name not in definitions:
                    definitions[name] = contract["definitions"][name]
                    resolve_references(definitions[name])
            for child in value.values():
                resolve_references(child)
        elif isinstance(value, list):
            for child in value:
                resolve_references(child)

    resolve_references(details["carriers"])
    details["definitions"] = definitions
    extensions = {row["extension"] for row in details["capabilities"]}
    details["authoring_guidance"] = [
        row for row in contract["authoring_guidance"] if row["capability"].split()[0] in extensions
    ]
    details["examples"] = {
        name: example for name, example in contract["examples"].items() if selected.intersection(example)
    }
    related = set().union(*(supported.intersection(example) for example in details["examples"].values())) - selected
    details["related_references"] = {key: _capability_reference_name(key) for key in sorted(related)}
    return details


def authoring_reference_documents() -> dict[str, str]:
    """Generate reference files from the installed engine and vendored core spec."""
    from .engine import datus_authoring_contract

    contract = datus_authoring_contract()
    documents = {
        "osi-core.yaml": authoring_spec_text("<osi_dialect>"),
        "datus.yaml": datus_extension_authoring_spec_text(),
    }
    for row in contract["capabilities"]:
        documents[_capability_reference_name(row["key"])] = yaml.safe_dump(
            _capability_spec(contract, row["key"]), sort_keys=False, allow_unicode=True, width=110
        )
    return documents
