# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""GenVisualReport Agentic Node Models.

Schemas for the ``gen_visual_report`` subagent, which produces a React-JSX
report artifact (``main.jsx`` + ``queries/*.sql`` + ``queries/*.json``).
See ``docs/gen-report-artifact.md`` for the wire contract this file enforces.
"""

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datus.schemas.base import BaseInput, BaseResult

# LLM-supplied slug doubles as the on-disk directory name; constrained
# to a filesystem-friendly subset so we never need to URL-escape it.
REPORT_SLUG_RE = re.compile(r"^[a-z0-9_]{1,80}$")
QUERY_SLUG_RE = re.compile(r"^[a-z0-9_]{1,64}$")
# Permissive sqlId form accepted by ``useQuerySql`` inside main.jsx — the
# runtime normalizes any of ``queries/foo``, ``queries/foo.json``, ``foo`` to
# the bare slug ``foo``. Save-time validation uses :func:`extract_query_slug`.
DATA_REF_RE = re.compile(r"^(?:queries/)?([a-z0-9_]{1,64})(?:\.(?:json|sql))?$")


ColumnSemanticType = Literal["string", "integer", "number", "date", "boolean"]


def extract_query_slug(data_ref: str) -> Optional[str]:
    """Return the bare slug for a ``useQuerySql`` argument, or None if invalid.

    Mirrors the iframe runtime's ``normalizeSqlId`` so static validation in
    :class:`ReportArtifactTools.save_main_jsx` agrees with what the renderer
    will look up at runtime.
    """
    if not isinstance(data_ref, str):
        return None
    match = DATA_REF_RE.fullmatch(data_ref.strip())
    if not match:
        return None
    return match.group(1)


class QueryColumnMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    type: ColumnSemanticType


class QueryResultFile(BaseModel):
    """Schema for ``queries/<slug>.json`` files."""

    model_config = ConfigDict(extra="forbid")

    executed_at: str = Field(..., description="ISO 8601 UTC")
    datasource: str
    row_count: int = Field(..., ge=0)
    columns: List[QueryColumnMeta] = Field(..., min_length=1)
    rows: List[Dict[str, Any]] = Field(...)

    @model_validator(mode="after")
    def _row_count_matches(self) -> "QueryResultFile":
        if len(self.rows) != self.row_count:
            raise ValueError(f"row_count={self.row_count} does not match len(rows)={len(self.rows)}")
        column_names = {c.name for c in self.columns}
        for idx, row in enumerate(self.rows):
            row_keys = set(row.keys())
            extra = row_keys - column_names
            if extra:
                raise ValueError(f"row {idx} contains keys not declared in columns: {sorted(extra)}")
            missing = column_names - row_keys
            if missing:
                raise ValueError(f"row {idx} is missing keys declared in columns: {sorted(missing)}")
        return self


class GenVisualReportNodeInput(BaseInput):
    """Input model for GenVisualReportAgenticNode."""

    user_message: str = Field(..., description="User's analysis question (required)")
    catalog: Optional[str] = Field(None, description="Database catalog")
    database: Optional[str] = Field(None, description="Database name")
    db_schema: Optional[str] = Field(None, description="Database schema")
    prompt_version: Optional[str] = Field(None, description="Prompt template version override")


class GenVisualReportNodeResult(BaseResult):
    """Result model for GenVisualReportAgenticNode."""

    response: str = Field(default="", description="Natural language summary shown after the artifact is produced")
    report_slug: Optional[str] = Field(None, description="LLM-chosen slug; doubles as the report's directory name.")
    app_jsx_path: Optional[str] = Field(None, description="Relative path to render/app.jsx under project_root")
    render_file_count: int = Field(default=0, description="Number of files persisted under reports/<slug>/render/")
    html_path: Optional[str] = Field(None, description="Path to compiled index.html (CLI mode only)")
    query_count: int = Field(default=0, description="Number of queries persisted under queries/")
    tokens_used: int = Field(default=0, description="Total tokens used during this run")
