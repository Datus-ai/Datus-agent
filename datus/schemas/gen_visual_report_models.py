# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenVisualReport Agentic Node Models.

Schemas for the gen_visual_report subagent, which produces a structured
report artifact (manifest.json + queries/*.sql + queries/*.json) instead
of a single Markdown string. See docs/gen-report-artifact.md for the
contract this file enforces.
"""

import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from datus.schemas.base import BaseInput, BaseResult

REPORT_ID_RE = re.compile(r"^rpt_[a-z0-9_-]{1,80}$")
QUERY_SLUG_RE = re.compile(r"^[a-z0-9_]{1,64}$")
DATA_REF_RE = re.compile(r"^queries/[a-z0-9_]{1,64}$")
SECTION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


ColumnSemanticType = Literal["string", "integer", "number", "date", "boolean"]
ColorScale = Literal["delta", "heatmap", "traffic"]
ColumnAlign = Literal["left", "center", "right"]

PREDEFINED_FORMATS = {
    "currency_usd",
    "currency_cny",
    "percent",
    "integer",
    "date_short",
    "date_long",
}

D3_FORMAT_RE = re.compile(r"^[\s\d.,%~$+\-fegspboxd]+$")


class TableColumnSpec(BaseModel):
    """One column definition for a table section."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        ..., min_length=1, max_length=128, description="Data key, must exist in referenced query columns"
    )
    label: str = Field(..., min_length=1, max_length=128, description="Header display text")
    type: ColumnSemanticType = Field(..., description="Semantic type for alignment / sorting / formatting defaults")
    format: Optional[str] = Field(
        None,
        description="Predefined enum (currency_usd / percent / ...) or a raw d3-format string",
    )
    align: Optional[ColumnAlign] = Field(None, description="Override default alignment for this type")
    color_scale: Optional[ColorScale] = Field(None, description="Optional visual encoding for the column")

    @field_validator("format")
    @classmethod
    def _check_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        if value in PREDEFINED_FORMATS:
            return value
        if D3_FORMAT_RE.fullmatch(value):
            return value
        raise ValueError(
            f"format must be one of {sorted(PREDEFINED_FORMATS)} or a valid d3-format string, got {value!r}"
        )


class MarkdownSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=SECTION_ID_RE.pattern)
    type: Literal["markdown"] = "markdown"
    content: str = Field(..., min_length=1, description="Markdown source (GFM)")


class SectionHeader(BaseModel):
    """Structured chapter / section heading.

    Replaces ``# Title`` markdown blobs with a typed primitive so renderers
    can apply consistent typography, optional numbered badges, and an
    accompanying one-line lede. Pick this over ``markdown`` whenever the
    block is acting as a hierarchical heading; reserve ``markdown`` for
    prose, lists, and inline formatting.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=SECTION_ID_RE.pattern)
    type: Literal["section_header"] = "section_header"
    title: str = Field(..., min_length=1, max_length=200, description="Heading text")
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional one-line lede shown beneath the title in a softer style",
    )
    level: int = Field(
        ...,
        ge=1,
        le=7,
        description=(
            "Visual hierarchy, 1 (top-level chapter) to 7 (deepest sub-section). "
            "Renderers map this to font size + weight + spacing."
        ),
    )
    number: Optional[int] = Field(
        None,
        ge=1,
        le=999,
        description=(
            "Optional 1-based ordinal displayed as a badge (e.g. '03') before the title. "
            "Omit when the section is unnumbered."
        ),
    )


class ChartSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=SECTION_ID_RE.pattern)
    type: Literal["chart"] = "chart"
    title: Optional[str] = Field(None, max_length=200)
    data_ref: str = Field(..., description="Reference to a query, formatted as 'queries/<slug>'")
    spec: Dict[str, Any] = Field(
        ...,
        description="Vega-Lite spec; must NOT include a 'data' field — renderer injects rows",
    )
    narrative: Optional[str] = Field(None, description="Plain text or Markdown explanation under the chart")

    @field_validator("data_ref")
    @classmethod
    def _check_data_ref(cls, value: str) -> str:
        if not DATA_REF_RE.fullmatch(value):
            raise ValueError(f"data_ref must match {DATA_REF_RE.pattern}, got {value!r}")
        return value

    @field_validator("spec")
    @classmethod
    def _check_spec(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if "data" in value:
            raise ValueError("chart.spec must not contain a 'data' field; renderer injects rows from data_ref")
        if "mark" not in value and "layer" not in value and "facet" not in value and "concat" not in value:
            raise ValueError("chart.spec must define one of: mark / layer / facet / concat")
        return value


class TableSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=SECTION_ID_RE.pattern)
    type: Literal["table"] = "table"
    title: Optional[str] = Field(None, max_length=200)
    data_ref: str = Field(..., description="Reference to a query, formatted as 'queries/<slug>'")
    columns: List[TableColumnSpec] = Field(..., min_length=1, max_length=64)

    @field_validator("data_ref")
    @classmethod
    def _check_data_ref(cls, value: str) -> str:
        if not DATA_REF_RE.fullmatch(value):
            raise ValueError(f"data_ref must match {DATA_REF_RE.pattern}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _check_unique_fields(self) -> "TableSection":
        seen: set[str] = set()
        for col in self.columns:
            if col.field in seen:
                raise ValueError(f"duplicate column field {col.field!r} in table {self.id}")
            seen.add(col.field)
        return self


class DividerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=SECTION_ID_RE.pattern)
    type: Literal["divider"] = "divider"


class LayoutSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=SECTION_ID_RE.pattern)
    type: Literal["layout"] = "layout"
    columns: List[int] = Field(..., min_length=1, max_length=6, description="Column-width ratios, positive integers")
    children: List["Section"] = Field(..., description="Length must equal columns.length")

    @field_validator("columns")
    @classmethod
    def _check_columns(cls, value: List[int]) -> List[int]:
        if any(c <= 0 for c in value):
            raise ValueError("layout.columns entries must be positive integers")
        return value

    @model_validator(mode="after")
    def _check_children_length(self) -> "LayoutSection":
        if len(self.children) != len(self.columns):
            raise ValueError(
                f"layout {self.id}: children length ({len(self.children)}) "
                f"must equal columns length ({len(self.columns)})"
            )
        return self


Section = Annotated[
    Union[MarkdownSection, SectionHeader, ChartSection, TableSection, LayoutSection, DividerSection],
    Field(discriminator="type"),
]

LayoutSection.model_rebuild()


def _iter_sections(sections: List["Section"]):
    for sec in sections:
        yield sec
        if isinstance(sec, LayoutSection):
            yield from _iter_sections(sec.children)


class ReportManifest(BaseModel):
    """Top-level manifest persisted as `manifest.json` inside the report dir."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"] = "1.0"
    id: str = Field(..., description="Report id, must match the folder name")
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    created_at: str = Field(..., description="ISO 8601 UTC timestamp")
    datasource: Optional[str] = Field(None, description="Primary datasource logical name")
    sections: List[Section] = Field(..., min_length=1)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not REPORT_ID_RE.fullmatch(value):
            raise ValueError(f"report id must match {REPORT_ID_RE.pattern}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _check_unique_section_ids(self) -> "ReportManifest":
        seen: set[str] = set()
        for sec in _iter_sections(self.sections):
            if sec.id in seen:
                raise ValueError(f"duplicate section id {sec.id!r}")
            seen.add(sec.id)
        return self

    def collect_data_refs(self) -> List[str]:
        """Return all `data_ref` values referenced by chart/table sections."""
        refs: List[str] = []
        for sec in _iter_sections(self.sections):
            if isinstance(sec, (ChartSection, TableSection)):
                refs.append(sec.data_ref)
        return refs


class QueryColumnMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    type: ColumnSemanticType


class QueryResultFile(BaseModel):
    """Schema for `queries/<slug>.json` files."""

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
            extra = set(row.keys()) - column_names
            if extra:
                raise ValueError(f"row {idx} contains keys not declared in columns: {sorted(extra)}")
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
    report_id: Optional[str] = Field(None, description="Generated report id (matches the folder name)")
    manifest_path: Optional[str] = Field(None, description="Relative path to manifest.json under project_root")
    html_path: Optional[str] = Field(None, description="Path to compiled index.html (CLI mode only)")
    query_count: int = Field(default=0, description="Number of queries persisted under queries/")
    tokens_used: int = Field(default=0, description="Total tokens used during this run")
