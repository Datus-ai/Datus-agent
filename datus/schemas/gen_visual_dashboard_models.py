# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""GenVisualDashboard Agentic Node Models.

Schemas for the ``gen_visual_dashboard`` subagent, which produces a
parameterized React-JSX dashboard artifact (``render/*.jsx`` +
``queries/<slug>.sql.j2`` + ``queries/<slug>.params.json``).

Unlike ``gen_visual_report`` (which persists pre-executed JSON results),
a dashboard's queries are Jinja2 templates with declared parameters. At
view time the backend renders the template with the user-supplied filter
values and executes the resulting SQL live against the bound datasource.

See ``gen_visual_dashboard_system_1.0.j2`` for the wire contract this
file enforces.
"""

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.gen_visual_report_models import (  # re-use the cross-artifact primitives
    DATA_REF_RE,
    QUERY_SLUG_RE,
    QueryColumnMeta,
    extract_query_slug,
)

# LLM-supplied slug doubles as the on-disk directory name; constrained
# to a filesystem-friendly subset so we never need to URL-escape it.
DASHBOARD_SLUG_RE = re.compile(r"^[a-z0-9_]{1,80}$")

# Header line that declares the params a template body references. The
# first non-blank line of a saved template must begin with this prefix.
DATUS_PARAMS_HEADER_RE = re.compile(r"^\s*--\s*@datus-params(?:\s+(.*?))?\s*$", re.MULTILINE)

# Single param declaration syntax: ``name:type[:optional]`` where type is
# one of the scalar types or its array variant (``string[]``, ``date[]``…).
# ``[?]`` shorthand is accepted as a tail-marker for optional.
_PARAM_TYPE_BASE = r"string|integer|number|date|boolean"
_PARAM_DECL_RE = re.compile(
    rf"^([a-z_][a-z0-9_]*)\s*:\s*({_PARAM_TYPE_BASE})(\s*\[\s*\])?\s*(?::\s*(optional)|(\?))?\s*$",
    re.IGNORECASE,
)

# Anything we explicitly accept inside a ``params`` dict on the JS side.
ParamScalarType = Literal["string", "integer", "number", "date", "boolean"]
ParamFullType = Literal[
    "string",
    "integer",
    "number",
    "date",
    "boolean",
    "string[]",
    "integer[]",
    "number[]",
    "date[]",
    "boolean[]",
]

__all__ = [
    "DASHBOARD_SLUG_RE",
    "DATA_REF_RE",
    "QUERY_SLUG_RE",
    "DATUS_PARAMS_HEADER_RE",
    "ParamScalarType",
    "ParamFullType",
    "QueryColumnMeta",
    "TemplateParamDecl",
    "QueryTemplateMetaFile",
    "GenVisualDashboardNodeInput",
    "GenVisualDashboardNodeResult",
    "extract_query_slug",
    "parse_datus_params_header",
]


class TemplateParamDecl(BaseModel):
    """One declared parameter of a saved query template."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z_][a-z0-9_]*$")
    type: ParamFullType
    required: bool = Field(default=True)

    @property
    def is_array(self) -> bool:
        return self.type.endswith("[]")

    @property
    def base_type(self) -> ParamScalarType:
        return self.type[:-2] if self.is_array else self.type  # type: ignore[return-value]


class QueryTemplateMetaFile(BaseModel):
    """Schema for ``queries/<slug>.params.json`` files.

    Companion to the ``.sql.j2`` template body — declares the params the
    template references plus the column metadata inferred from the trial
    render the LLM did with ``sample_params``.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    description: str = Field(default="")
    datasource: str
    params: List[TemplateParamDecl] = Field(..., description="Parameters declared by the SQL template")
    columns: List[QueryColumnMeta] = Field(..., min_length=1, description="Inferred from the sample render")
    sample_params: Dict[str, Any] = Field(default_factory=dict, description="Values used for the trial render")
    sample_row_count: int = Field(..., ge=0)
    saved_at: str = Field(..., description="ISO 8601 UTC")

    @model_validator(mode="after")
    def _params_unique(self) -> "QueryTemplateMetaFile":
        seen: set[str] = set()
        for p in self.params:
            if p.name in seen:
                raise ValueError(f"duplicate param declaration: {p.name!r}")
            seen.add(p.name)
        return self


class GenVisualDashboardNodeInput(BaseInput):
    """Input model for GenVisualDashboardAgenticNode."""

    user_message: str = Field(..., description="User's dashboard question (required)")
    catalog: Optional[str] = Field(None, description="Database catalog")
    database: Optional[str] = Field(None, description="Database name")
    db_schema: Optional[str] = Field(None, description="Database schema")
    prompt_version: Optional[str] = Field(None, description="Prompt template version override")


class GenVisualDashboardNodeResult(BaseResult):
    """Result model for GenVisualDashboardAgenticNode."""

    response: str = Field(default="", description="Natural language summary shown after the artifact is produced")
    dashboard_slug: Optional[str] = Field(
        None,
        description="LLM-chosen slug; doubles as the dashboard's directory name.",
    )
    app_jsx_path: Optional[str] = Field(None, description="Relative path to render/app.jsx under project_root")
    render_file_count: int = Field(default=0, description="Number of files persisted under dashboards/<slug>/render/")
    template_count: int = Field(default=0, description="Number of templates persisted under queries/")
    tokens_used: int = Field(default=0, description="Total tokens used during this run")
    artifact_kind: Literal["dashboard"] = Field(
        default="dashboard",
        description="Artifact kind, fixed for this node. Carried into the SSE artifact payload so the frontend can pick the right viewer without reading the action_type.",
    )
    artifact_mode: Optional[Literal["new", "edit"]] = Field(
        default=None,
        description="Whether the LLM chose start_new_dashboard (new) or bind_existing_dashboard (edit). None when the run failed before binding.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Display name copied from manifest.json so the frontend can render the artifact card without a second round-trip.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Short description copied from manifest.json.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC timestamp copied from manifest.json; reflects creation time even for 'edit' mode.",
    )
    finalize_warnings: List[str] = Field(
        default_factory=list,
        description="Best-effort warnings from the analysis-finalize stage (see ReportNodeResult for details).",
    )
    finalize_error: Optional[str] = Field(
        default=None,
        description="Set when finalize LLM call / validation failed; the analysis/ trio is then absent or stale.",
    )


def parse_datus_params_header(sql_template: str) -> List[TemplateParamDecl]:
    """Parse the ``-- @datus-params ...`` header out of a template.

    Returns the list of declared params. Raises ``ValueError`` if the
    header is missing, malformed, or declares the same param twice. The
    header must appear on the first non-blank line that starts with
    ``--`` (so leading comments are allowed only after this header).
    """
    first_real_line = ""
    for raw in sql_template.splitlines():
        if not raw.strip():
            continue
        first_real_line = raw
        break

    if not first_real_line.lstrip().startswith("--"):
        raise ValueError(
            "Template is missing the required ``-- @datus-params ...`` header on its first non-blank line."
        )

    match = DATUS_PARAMS_HEADER_RE.match(first_real_line)
    if not match:
        raise ValueError(
            "First template line is a SQL comment but is not a ``-- @datus-params ...`` declaration. "
            "Move any other leading comments below the declaration."
        )

    # group(1) is None when the optional ``\s+(.*?)`` is absent — i.e. the
    # bare-keyword empty-header form ``-- @datus-params``.
    body = (match.group(1) or "").strip()
    parts = [p.strip() for p in body.split(",") if p.strip()]
    # ``-- @datus-params`` with no body (or only a stray comma) is the
    # canonical way to declare "this template takes no parameters" — the
    # header is still mandatory so the runtime contract stays explicit, but
    # an empty body means there is nothing to bind. The save_query_template
    # validator's "every declared param must appear as :name in the body"
    # check is then vacuously satisfied for genuinely static queries (e.g.
    # a catalog-level supply-cost rollup that doesn't depend on date or
    # store filters).
    if not parts:
        return []

    seen: set[str] = set()
    decls: List[TemplateParamDecl] = []
    for part in parts:
        decl_match = _PARAM_DECL_RE.match(part)
        if not decl_match:
            raise ValueError(
                f"Param declaration {part!r} is malformed. Expected ``<name>:<type>[:optional]`` "
                "where <type> is one of string / integer / number / date / boolean (optionally [] for array)."
            )
        name = decl_match.group(1).lower()
        base = decl_match.group(2).lower()
        is_array = decl_match.group(3) is not None
        optional_kw = decl_match.group(4)
        optional_short = decl_match.group(5)
        required = optional_kw is None and optional_short is None
        if name in seen:
            raise ValueError(f"Duplicate param {name!r} in ``-- @datus-params`` header.")
        seen.add(name)
        full_type = f"{base}[]" if is_array else base
        decls.append(TemplateParamDecl(name=name, type=full_type, required=required))  # type: ignore[arg-type]
    return decls
