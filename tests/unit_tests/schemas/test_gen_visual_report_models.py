# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for gen_visual_report pydantic schemas.

These tests pin the wire contract documented in
``docs/gen-report-artifact.md`` (saas repo). Every public field rule
covered here is something downstream renderers (CLI HTML and SaaS
``DatusReportRender``) rely on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datus.schemas.gen_visual_report_models import (
    ChartSection,
    GenVisualReportNodeInput,
    GenVisualReportNodeResult,
    LayoutSection,
    QueryResultFile,
    ReportManifest,
    SectionHeader,
    TableColumnSpec,
    TableSection,
)

# ----------------------------------------------------------------------------- #
# TableColumnSpec                                                               #
# ----------------------------------------------------------------------------- #


class TestTableColumnSpec:
    def test_minimal_columns_round_trip(self):
        col = TableColumnSpec(field="sales", label="Sales", type="number")
        assert col.field == "sales"
        assert col.label == "Sales"
        assert col.type == "number"
        assert col.format is None
        assert col.align is None
        assert col.color_scale is None

    def test_predefined_format_accepted(self):
        col = TableColumnSpec(field="sales", label="Sales", type="number", format="currency_usd")
        assert col.format == "currency_usd"

    def test_d3_format_string_accepted(self):
        col = TableColumnSpec(field="growth", label="YoY", type="number", format=".1%")
        assert col.format == ".1%"

    def test_empty_format_normalized_to_none(self):
        col = TableColumnSpec(field="x", label="X", type="number", format="")
        assert col.format is None

    def test_unknown_predefined_format_rejected(self):
        with pytest.raises(ValidationError):
            TableColumnSpec(field="x", label="X", type="number", format="currency_eur")

    def test_garbage_format_string_rejected(self):
        with pytest.raises(ValidationError):
            TableColumnSpec(field="x", label="X", type="number", format="hello world!")

    def test_align_color_scale_values(self):
        col = TableColumnSpec(
            field="growth",
            label="YoY",
            type="number",
            align="right",
            color_scale="delta",
        )
        assert col.align == "right"
        assert col.color_scale == "delta"

    def test_invalid_align_rejected(self):
        with pytest.raises(ValidationError):
            TableColumnSpec(field="x", label="X", type="number", align="diagonal")

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            TableColumnSpec(field="x", label="X", type="object")


# ----------------------------------------------------------------------------- #
# Section types                                                                 #
# ----------------------------------------------------------------------------- #


class TestSections:
    def test_chart_data_ref_must_match_pattern(self):
        with pytest.raises(ValidationError):
            ChartSection(
                id="blk_chart",
                data_ref="not_a_query_ref",
                spec={"mark": "bar", "encoding": {}},
            )

    def test_chart_spec_must_not_have_data(self):
        with pytest.raises(ValidationError):
            ChartSection(
                id="blk_chart",
                data_ref="queries/q1",
                spec={"data": {"values": [{"x": 1}]}, "mark": "bar", "encoding": {}},
            )

    def test_chart_spec_must_have_top_level_mark_or_composition(self):
        with pytest.raises(ValidationError):
            ChartSection(id="blk_chart", data_ref="queries/q1", spec={"encoding": {}})

    def test_chart_minimal_valid(self):
        sec = ChartSection(
            id="blk_chart",
            data_ref="queries/q_sales",
            spec={
                "mark": "bar",
                "encoding": {
                    "x": {"field": "month", "type": "ordinal"},
                    "y": {"field": "sales", "type": "quantitative"},
                },
            },
        )
        assert sec.data_ref == "queries/q_sales"
        assert sec.spec["mark"] == "bar"

    def test_table_requires_at_least_one_column(self):
        with pytest.raises(ValidationError):
            TableSection(id="blk_tbl", data_ref="queries/q1", columns=[])

    def test_table_columns_must_be_unique(self):
        with pytest.raises(ValidationError):
            TableSection(
                id="blk_tbl",
                data_ref="queries/q1",
                columns=[
                    TableColumnSpec(field="x", label="X", type="number"),
                    TableColumnSpec(field="x", label="X again", type="number"),
                ],
            )

    def test_layout_children_count_must_equal_columns(self):
        with pytest.raises(ValidationError):
            LayoutSection(
                id="blk_layout",
                columns=[2, 1],
                children=[{"id": "blk_layout_c0", "type": "markdown", "content": "only one"}],
            )

    def test_layout_positive_columns_only(self):
        with pytest.raises(ValidationError):
            LayoutSection(
                id="blk_layout",
                columns=[0, 1],
                children=[
                    {"id": "a", "type": "markdown", "content": "a"},
                    {"id": "b", "type": "markdown", "content": "b"},
                ],
            )

    def test_section_header_minimal_valid(self):
        sec = SectionHeader(id="blk_h_overview", title="总体统计概览", level=1)
        assert sec.type == "section_header"
        assert sec.level == 1
        assert sec.number is None
        assert sec.description is None

    def test_section_header_with_number_and_description(self):
        sec = SectionHeader(
            id="blk_h_overview",
            number=3,
            title="Numerical Distribution Features",
            description="Observe distribution shape, dispersion, and skewness of key numeric columns",
            level=2,
        )
        assert sec.number == 3
        assert sec.level == 2
        assert sec.description.startswith("Observe")

    def test_section_header_level_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SectionHeader(id="blk_h", title="x", level=0)
        with pytest.raises(ValidationError):
            SectionHeader(id="blk_h", title="x", level=8)

    def test_section_header_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            SectionHeader(id="blk_h", title="x", level=1, number=0)

    def test_section_header_title_required(self):
        with pytest.raises(ValidationError):
            SectionHeader(id="blk_h", title="", level=1)

    def test_section_header_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            SectionHeader.model_validate(
                {"id": "blk_h", "type": "section_header", "title": "x", "level": 1, "unknown": "value"}
            )

    def test_section_header_inside_manifest_discriminated(self):
        """The manifest's discriminated union must accept type=section_header."""
        manifest = ReportManifest.model_validate(
            {
                "id": "rpt_demo_002",
                "title": "demo",
                "created_at": "2026-05-13T10:00:00Z",
                "sections": [
                    {
                        "id": "blk_h_overview",
                        "type": "section_header",
                        "number": 1,
                        "title": "Overview",
                        "description": "High-level KPIs",
                        "level": 1,
                    },
                    {"id": "blk_body", "type": "markdown", "content": "body"},
                ],
            }
        )
        # The first section round-trips as a SectionHeader instance.
        header = manifest.sections[0]
        assert isinstance(header, SectionHeader)
        assert header.number == 1

    def test_layout_nested_valid(self):
        sec = LayoutSection(
            id="blk_layout",
            columns=[1, 1],
            children=[
                {"id": "child_a", "type": "markdown", "content": "left"},
                {"id": "child_b", "type": "divider"},
            ],
        )
        assert len(sec.children) == 2


# ----------------------------------------------------------------------------- #
# ReportManifest                                                                #
# ----------------------------------------------------------------------------- #


def _valid_manifest(**overrides):
    payload = {
        "id": "rpt_demo_001",
        "title": "demo",
        "created_at": "2026-05-13T10:00:00Z",
        "sections": [
            {"id": "blk_001", "type": "markdown", "content": "# hello"},
            {
                "id": "blk_002",
                "type": "chart",
                "data_ref": "queries/sales_by_store",
                "spec": {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "month", "type": "ordinal"},
                        "y": {"field": "sales", "type": "quantitative"},
                    },
                },
            },
            {
                "id": "blk_003",
                "type": "table",
                "data_ref": "queries/sales_by_store",
                "columns": [
                    {"field": "month", "label": "M", "type": "integer"},
                    {"field": "sales", "label": "Sales", "type": "number", "format": "currency_usd"},
                ],
            },
            {"id": "blk_004", "type": "divider"},
        ],
    }
    payload.update(overrides)
    return payload


class TestReportManifest:
    def test_minimal_manifest_round_trip(self):
        manifest = ReportManifest.model_validate(_valid_manifest())
        assert manifest.id == "rpt_demo_001"
        assert manifest.version == "1.0"
        assert len(manifest.sections) == 4
        refs = manifest.collect_data_refs()
        assert refs == ["queries/sales_by_store", "queries/sales_by_store"]

    def test_id_must_match_pattern(self):
        with pytest.raises(ValidationError):
            ReportManifest.model_validate(_valid_manifest(id="bad id with spaces"))

    def test_id_uppercase_rejected(self):
        with pytest.raises(ValidationError):
            ReportManifest.model_validate(_valid_manifest(id="rpt_HAS_UPPER"))

    def test_section_ids_must_be_unique(self):
        bad = _valid_manifest()
        bad["sections"].append({"id": "blk_001", "type": "divider"})
        with pytest.raises(ValidationError):
            ReportManifest.model_validate(bad)

    def test_layout_children_uniqueness_enforced_globally(self):
        bad = _valid_manifest()
        bad["sections"] = [
            {"id": "blk_dup", "type": "markdown", "content": "x"},
            {
                "id": "blk_layout",
                "type": "layout",
                "columns": [1, 1],
                "children": [
                    {"id": "blk_dup", "type": "markdown", "content": "duplicate inside"},
                    {"id": "blk_other", "type": "divider"},
                ],
            },
        ]
        with pytest.raises(ValidationError):
            ReportManifest.model_validate(bad)

    def test_extra_fields_rejected(self):
        bad = _valid_manifest()
        bad["unknown"] = "value"
        with pytest.raises(ValidationError):
            ReportManifest.model_validate(bad)

    def test_sections_required_non_empty(self):
        bad = _valid_manifest()
        bad["sections"] = []
        with pytest.raises(ValidationError):
            ReportManifest.model_validate(bad)


# ----------------------------------------------------------------------------- #
# QueryResultFile                                                               #
# ----------------------------------------------------------------------------- #


class TestQueryResultFile:
    def test_round_trip(self):
        qr = QueryResultFile.model_validate(
            {
                "executed_at": "2026-05-13T10:00:00Z",
                "datasource": "pg_main",
                "row_count": 2,
                "columns": [
                    {"name": "store_name", "type": "string"},
                    {"name": "sales", "type": "number"},
                ],
                "rows": [
                    {"store_name": "A", "sales": 100.0},
                    {"store_name": "B", "sales": 250.5},
                ],
            }
        )
        assert qr.row_count == 2
        assert len(qr.rows) == 2
        assert qr.columns[0].name == "store_name"

    def test_row_count_must_match_rows_length(self):
        with pytest.raises(ValidationError):
            QueryResultFile.model_validate(
                {
                    "executed_at": "2026-05-13T10:00:00Z",
                    "datasource": "pg_main",
                    "row_count": 5,
                    "columns": [{"name": "a", "type": "integer"}],
                    "rows": [{"a": 1}],
                }
            )

    def test_row_keys_must_be_in_columns(self):
        with pytest.raises(ValidationError):
            QueryResultFile.model_validate(
                {
                    "executed_at": "2026-05-13T10:00:00Z",
                    "datasource": "pg_main",
                    "row_count": 1,
                    "columns": [{"name": "a", "type": "integer"}],
                    "rows": [{"a": 1, "rogue_field": "nope"}],
                }
            )


# ----------------------------------------------------------------------------- #
# Input / Result models                                                         #
# ----------------------------------------------------------------------------- #


class TestNodeIO:
    def test_input_requires_user_message(self):
        with pytest.raises(ValidationError):
            GenVisualReportNodeInput()  # type: ignore[call-arg]

    def test_input_round_trip(self):
        inp = GenVisualReportNodeInput(user_message="hi", catalog="cat", database="db", db_schema="s")
        assert inp.user_message == "hi"
        assert inp.catalog == "cat"
        assert inp.db_schema == "s"

    def test_result_defaults(self):
        result = GenVisualReportNodeResult(success=True)
        assert result.success is True
        assert result.response == ""
        assert result.report_id is None
        assert result.html_path is None
        assert result.query_count == 0
        assert result.tokens_used == 0
