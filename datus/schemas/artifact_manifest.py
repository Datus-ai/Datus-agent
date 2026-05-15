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
from typing import Literal

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
      first written. We deliberately do NOT track an ``updated_at``
      here — the file is write-once for now (no update-manifest tool).
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
