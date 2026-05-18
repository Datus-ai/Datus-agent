# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""On-disk manifest shared by the report and dashboard subagents.

Written once at artifact creation time (``start_new_report`` /
``start_new_dashboard``) to ``<root>/<id>/manifest.json``. Consumers:

* Datus-SaaS list pages — pull ``name`` and ``description`` to render
  human-friendly cards instead of raw ``rpt_<...>`` / ``dash_<...>`` ids.
* Datus-CLI HTML compile — falls back to ``name`` for the page title.
* IDE explorer — surface ``name`` next to the artifact directory.

The two LLM-supplied fields (``name``, ``description``) are required —
we treat a missing/blank value as a programming error rather than
quietly defaulting, so the list pages never end up with a card that
just says ``Untitled report``.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ArtifactKind = Literal["report", "dashboard"]

# LLM-supplied slug used directly as the artifact's directory name (no
# random suffix, no prefix). Constrained to filesystem-friendly chars so
# we never need to URL-escape it; max 80 keeps the full ``reports/<slug>/render/app.jsx``
# path under typical OS limits.
ARTIFACT_SLUG_PATTERN = r"^[a-z0-9_]{1,80}$"
ARTIFACT_SLUG_RE = re.compile(ARTIFACT_SLUG_PATTERN)


class ArtifactManifest(BaseModel):
    """Persisted at ``<root>/<slug>/manifest.json``.

    Field choices:

    * ``slug`` is the LLM-supplied stable identifier; it doubles as the
      on-disk directory name (``reports/<slug>/`` /
      ``dashboards/<slug>/``). The LLM is responsible for choosing a
      slug that doesn't collide with any existing artifact directory
      (system prompt mandates a ``glob`` of the kind root before
      calling ``start_new_*``).
    * ``name`` and ``description`` are **required, non-empty**. The
      system prompt forces the LLM to produce both at ``start_new_*``
      time so the artifact is never orphaned without a display name.
    * ``kind`` mirrors the parent directory (``"reports"`` →
      ``"report"``, ``"dashboards"`` → ``"dashboard"``); callers that
      read the file by path already know which kind it is, but keeping
      the field self-describing means a single backend route can serve
      both shapes by inspecting one file.
    * ``created_at`` is the UTC timestamp at which the manifest was
      first written.
    * ``updated_at`` is a persisted ISO-8601 UTC timestamp refreshed on
      every ``save_query`` / ``save_query_template`` / ``bind_existing_*``
      call so list pages can order by recency. ``None`` until the first
      mutation after creation; older manifests written before this field
      was introduced deserialize cleanly with ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(
        ...,
        pattern=ARTIFACT_SLUG_PATTERN,
        description="Filesystem-friendly slug; doubles as the artifact's directory name.",
    )
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable display name (any language).")
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="One-paragraph description of what this artifact does.",
    )
    kind: ArtifactKind = Field(..., description="report | dashboard")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp at second precision.")
    updated_at: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 UTC timestamp refreshed on every save_query / save_query_template / "
            "bind_existing_* call so list pages can order by recency. ``None`` until the "
            "first mutation after creation. Older manifests written before this field was "
            "introduced deserialize cleanly with ``None``."
        ),
    )
    datasources: List[str] = Field(
        default_factory=list,
        description=(
            "Distinct datasource labels referenced by this artifact's queries, populated "
            "incrementally as ``save_query`` / ``save_query_template`` is called. Used by "
            "the subagent that gets spawned for follow-up questions to know which "
            "connectors to bind. Stable order = first-seen order so diffs stay readable."
        ),
    )
    key_tables: List[str] = Field(
        default_factory=list,
        description=(
            "Distinct table references across this artifact's queries, preserving the "
            "qualification each SQL used (``finbench.main.Account`` stays as is; bare "
            "``Account`` stays bare). Code-generated at finalize time by parsing every "
            "``queries/*.sql`` and ``queries/*.sql.j2`` with sqlglot, stripping CTE "
            "aliases, and collapsing same-table-different-qualification entries to the "
            "more informative form — the LLM never writes this field. The follow-up "
            "ask agent reads it both to skip ``list_tables`` / ``describe_table`` "
            "round-trips AND to copy the exact qualified reference when planning new "
            "SQL on related tables (so the query runs on strict-schema dialects without "
            "guessing the prefix). Empty when finalize hasn't run yet or when SQL "
            "parsing failed across the board (rare). Sorted alphabetically for diff "
            "stability."
        ),
    )
