# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""On-disk schema snapshot for an artifact's ``manifest.key_tables``.

Written once at finalize time (after ``manifest.key_tables`` is
code-aggregated from the saved SQL bodies) to
``<root>/<slug>/analysis/key_tables_schema.json``. The companion
``ask_*`` consultant inlines the snapshot into its system prompt so
follow-up SQL planning skips ``describe_table`` round-trips for the
tables the artifact already touched.

Contract — snapshot only:

* Captures the columns that existed when the artifact was finalized.
  Live DDL drift is **expected** to happen; the prompt instructs the
  LLM to re-fetch via ``describe_table('<table>')`` whenever the user
  asks about the current/latest state, or about a column not in this
  list, or about tables outside ``manifest.key_tables``.
* Best-effort. ``describe_table`` may fail for individual tables
  (permission revoked, table renamed/dropped post-creation). Per-table
  ``error`` field carries the failure reason instead of stranding the
  whole sidecar.
* Mirrors :func:`datus.tools.func_tool.database.DBFuncTool.describe_table`
  return shape so the bake step is a thin pass-through and the prompt
  rendering knows exactly which fields can be present.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class KeyTableColumn(BaseModel):
    """One column of a snapshotted key_table.

    Fields mirror ``DBFuncTool.describe_table()`` output. ``is_dimension``
    is only populated when a semantic model existed for the table at
    finalize time — its absence is meaningful (no semantic info), so
    we distinguish ``None`` (unknown / no model) from ``False``
    (model exists but column is a measure / metric, not a dimension).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = ""
    comment: str = ""
    is_dimension: Optional[bool] = None


class KeyTableSchema(BaseModel):
    """Snapshot of one entry from ``manifest.key_tables``.

    ``name`` mirrors the manifest entry verbatim (fully qualified per
    SQL convention, e.g. ``jeff_shop.raw_orders``). ``error`` is
    populated iff ``describe_table`` failed for this table; when set,
    ``columns`` is empty and the prompt renderer surfaces a "schema
    unavailable; call describe_table()" hint instead of a column list.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    columns: List[KeyTableColumn] = Field(default_factory=list)
    error: Optional[str] = None


class KeyTablesSchemaFile(BaseModel):
    """Top-level wrapper for ``analysis/key_tables_schema.json``.

    Single-field wrapper rather than a bare list so the file shape is
    forward-compatible — e.g. adding ``generated_at`` or
    ``connector_version`` later is a non-breaking change for any
    consumer that already validates against this model.
    """

    model_config = ConfigDict(extra="forbid")

    tables: List[KeyTableSchema] = Field(default_factory=list)
