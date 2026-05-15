# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for gen_visual_report pydantic schemas.

The artifact contract is now ``main.jsx + queries/*``. This file pins:

* the query-result schema (``QueryResultFile``) — both the runner and the
  iframe ``useQuerySql`` hook rely on this shape;
* the sqlId normalizer (``extract_query_slug``) — keeps the agent-side
  static check aligned with the runtime's ``normalizeSqlId``;
* the node IO models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datus.schemas.gen_visual_report_models import (
    GenVisualReportNodeInput,
    GenVisualReportNodeResult,
    QueryResultFile,
    extract_query_slug,
)

# ----------------------------------------------------------------------------- #
# extract_query_slug                                                            #
# ----------------------------------------------------------------------------- #


class TestExtractQuerySlug:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("queries/sales_by_store", "sales_by_store"),
            ("queries/sales_by_store.json", "sales_by_store"),
            ("queries/sales_by_store.sql", "sales_by_store"),
            ("sales_by_store", "sales_by_store"),
            ("sales_by_store.json", "sales_by_store"),
            ("  queries/sales_by_store  ", "sales_by_store"),
        ],
    )
    def test_accepts_each_form(self, raw, expected):
        assert extract_query_slug(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "queries/Sales_With_Uppercase",
            "queries/has spaces",
            "queries/sales/twolevel",
            "../queries/escape",
            "../etc/passwd",
            "",
            "a" * 70,  # exceeds {1,64}
        ],
    )
    def test_rejects_malformed(self, raw):
        assert extract_query_slug(raw) is None

    def test_non_string_returns_none(self):
        assert extract_query_slug(None) is None  # type: ignore[arg-type]
        assert extract_query_slug(123) is None  # type: ignore[arg-type]


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

    def test_unknown_column_type_rejected(self):
        with pytest.raises(ValidationError):
            QueryResultFile.model_validate(
                {
                    "executed_at": "2026-05-13T10:00:00Z",
                    "datasource": "pg_main",
                    "row_count": 0,
                    "columns": [{"name": "a", "type": "object"}],
                    "rows": [],
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
        assert result.report_slug is None
        assert result.app_jsx_path is None
        assert result.render_file_count == 0
        assert result.html_path is None
        assert result.query_count == 0
        assert result.tokens_used == 0

    def test_result_with_app_jsx_path(self):
        result = GenVisualReportNodeResult(
            success=True,
            report_slug="demo",
            app_jsx_path="reports/demo/render/app.jsx",
            render_file_count=3,
            query_count=5,
        )
        assert result.app_jsx_path == "reports/demo/render/app.jsx"
        assert result.render_file_count == 3
        assert result.query_count == 5
