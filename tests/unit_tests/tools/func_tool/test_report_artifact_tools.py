# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for report artifact tools.

Covers:
* ``ReportArtifactTools.start_new_report`` / ``bind_existing_report`` — the
  LLM-driven intent declaration that picks "create new" vs "edit existing"
  before any write tool runs.
* ``_require_active`` guard — save_query / save_main_jsx fail-fast when no
  report is bound.
* ``save_query`` — column inference, SQL persistence, schema validation,
  datasource resolution failures, slug overwrite.
* ``save_main_jsx`` — basic syntax check, sqlId reference cross-check
  against persisted queries, atomic rewrite, byte cap.
* ``ReportFilesystemFuncTool`` — deny rules for ``main.jsx`` and
  ``queries/*`` paths; allow rules for everything else.

No mocks; we use a real SQLite database wired through ``DBFuncTool`` so
``save_query`` exercises the same code path it will in production.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.tools.func_tool import DBFuncTool, ReportArtifactTools, ReportFilesystemFuncTool
from datus.tools.func_tool.report_artifact_tools import _allocate_report_id, _infer_column_type, _slugify_title

# ----------------------------------------------------------------------------- #
# Fixtures                                                                      #
# ----------------------------------------------------------------------------- #


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Path:
    """Build a tiny SQLite db with a `sales` table for save_query tests."""
    db_path = tmp_path / "demo.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE sales (store_name TEXT, month INTEGER, sales REAL, growth REAL, asof TEXT)")
        conn.executemany(
            "INSERT INTO sales VALUES (?,?,?,?,?)",
            [
                ("Manhattan #1", 1, 1000.98, 0.18, "2026-01-01"),
                ("Brooklyn #3", 1, 3000.24, -0.05, "2026-01-01"),
                ("Manhattan #1", 2, 1200.50, 0.10, "2026-02-01"),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def db_func_tool(sqlite_db: Path) -> DBFuncTool:
    from datus.tools.db_tools.config import SQLiteConfig
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector

    connector = SQLiteConnector(SQLiteConfig(db_path=str(sqlite_db)))
    return DBFuncTool(connector_or_manager=connector)


@pytest.fixture
def unbound_tools(db_func_tool: DBFuncTool, project_root: Path) -> ReportArtifactTools:
    """Instance with no active report — for testing the binding lifecycle."""
    agent_config = SimpleNamespace(project_root=str(project_root))
    return ReportArtifactTools(agent_config=agent_config, db_func_tool=db_func_tool)


@pytest.fixture
def report_tools(unbound_tools: ReportArtifactTools) -> ReportArtifactTools:
    """Instance with an active fresh report bound via start_new_report."""
    result = unbound_tools.start_new_report(title="demo test")
    assert result.success == 1, result.error
    return unbound_tools


# ----------------------------------------------------------------------------- #
# helpers                                                                       #
# ----------------------------------------------------------------------------- #


class TestSlugifyTitle:
    def test_ascii_lowercased_and_underscored(self):
        assert _slugify_title("Sales By Store") == "sales_by_store"

    def test_strips_non_ascii(self):
        # Pure Chinese title yields empty slug; caller falls back to "report".
        assert _slugify_title("销售分析") == ""

    def test_collapses_punctuation(self):
        assert _slugify_title("Q1 — North/East Sales!!") == "q1_north_east_sales"

    def test_caps_length(self):
        very_long = "a" * 200
        assert _slugify_title(very_long, max_len=32) == "a" * 32


class TestAllocateReportId:
    def test_format_matches_pattern(self, project_root: Path):
        new_id = _allocate_report_id("sales report", project_root)
        assert new_id.startswith("rpt_sales_report_")
        parts = new_id.split("_")
        assert parts[0] == "rpt"
        assert parts[-1] != "" and len(parts[-1]) == 6

    def test_falls_back_to_report_when_slug_empty(self, project_root: Path):
        new_id = _allocate_report_id("销售", project_root)
        assert new_id.startswith("rpt_report_")

    def test_avoids_collision(self, project_root: Path):
        first = _allocate_report_id("collision", project_root)
        (project_root / "reports" / first).mkdir(parents=True)
        # The next call must pick a different id, not return first again.
        second = _allocate_report_id("collision", project_root)
        assert second != first


# ----------------------------------------------------------------------------- #
# start_new_report / bind_existing_report                                       #
# ----------------------------------------------------------------------------- #


class TestStartNewReport:
    def test_allocates_id_and_creates_dirs(self, unbound_tools: ReportArtifactTools, project_root: Path):
        result = unbound_tools.start_new_report(title="east sales")
        assert result.success == 1
        payload = result.result
        new_id = payload["report_id"]
        assert new_id.startswith("rpt_east_sales_")
        assert payload["mode"] == "new"
        assert payload["report_dir"] == f"reports/{new_id}"

        assert unbound_tools.report_id == new_id
        assert unbound_tools.mode == "new"
        assert (project_root / "reports" / new_id / "queries").is_dir()

    def test_empty_title_falls_back_to_report(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.start_new_report(title="")
        assert result.success == 1
        assert result.result["report_id"].startswith("rpt_report_")


class TestBindExistingReport:
    def test_binds_when_directory_and_main_jsx_exist(self, unbound_tools: ReportArtifactTools, project_root: Path):
        existing = project_root / "reports" / "rpt_existing_demo_260513_aaaaaa"
        (existing / "queries").mkdir(parents=True)
        (existing / "main.jsx").write_text("export default function R() { return null; }\n")

        result = unbound_tools.bind_existing_report("rpt_existing_demo_260513_aaaaaa")
        assert result.success == 1, result.error
        assert result.result["mode"] == "edit"
        assert unbound_tools.report_id == "rpt_existing_demo_260513_aaaaaa"
        assert unbound_tools.mode == "edit"

    def test_rejects_missing_directory(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.bind_existing_report("rpt_nope_260513_bbbbbb")
        assert result.success == 0
        assert "not found" in (result.error or "").lower()
        assert unbound_tools.report_id is None

    def test_rejects_missing_main_jsx(self, unbound_tools: ReportArtifactTools, project_root: Path):
        # Directory exists but no main.jsx — caller never finished a previous run.
        incomplete = project_root / "reports" / "rpt_partial_260513_cccccc"
        (incomplete / "queries").mkdir(parents=True)

        result = unbound_tools.bind_existing_report("rpt_partial_260513_cccccc")
        assert result.success == 0
        assert "main.jsx" in (result.error or "")
        assert unbound_tools.report_id is None

    def test_rejects_invalid_id_format(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.bind_existing_report("not-a-valid-id!")
        assert result.success == 0
        assert "match" in (result.error or "").lower()


class TestRequireActive:
    """save_query / save_main_jsx must fail-fast when no report is bound."""

    def test_save_query_rejects_when_unbound(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.save_query(name="q", sql="SELECT 1 AS a")
        assert result.success == 0
        error = (result.error or "").lower()
        assert "no active report" in error
        assert "start_new_report" in error
        assert "bind_existing_report" in error

    def test_save_main_jsx_rejects_when_unbound(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.save_main_jsx("export default function R() { return null; }")
        assert result.success == 0
        assert "no active report" in (result.error or "").lower()


# ----------------------------------------------------------------------------- #
# _infer_column_type                                                            #
# ----------------------------------------------------------------------------- #


class TestInferColumnType:
    def test_all_none_is_string(self):
        assert _infer_column_type([None, None]) == "string"

    def test_all_booleans(self):
        assert _infer_column_type([True, False, True]) == "boolean"

    def test_all_integers(self):
        assert _infer_column_type([1, 2, 3]) == "integer"

    def test_mixed_int_float_is_number(self):
        assert _infer_column_type([1, 2.5, 3]) == "number"

    def test_iso_date_strings(self):
        assert _infer_column_type(["2026-01-01", "2026-02-01"]) == "date"

    def test_iso_datetime_strings(self):
        assert _infer_column_type(["2026-01-01T10:00:00Z", "2026-02-01T11:00:00Z"]) == "date"

    def test_falls_back_to_string(self):
        assert _infer_column_type(["alpha", "beta"]) == "string"


# ----------------------------------------------------------------------------- #
# save_query                                                                    #
# ----------------------------------------------------------------------------- #


class TestSaveQuery:
    def test_persists_sql_and_json(self, report_tools: ReportArtifactTools, project_root: Path):
        result = report_tools.save_query(
            name="sales_by_store",
            sql="SELECT store_name, month, sales, growth FROM sales ORDER BY store_name, month",
            description="Monthly sales by store",
        )
        assert result.success == 1
        payload = result.result
        assert payload["name"] == "sales_by_store"
        assert payload["data_ref"] == "queries/sales_by_store"
        assert payload["row_count"] == 3

        report_id = report_tools.report_id or ""
        assert report_id.startswith("rpt_demo_test_"), f"unexpected report id: {report_id!r}"
        sql_file = project_root / "reports" / report_id / "queries" / "sales_by_store.sql"
        json_file = project_root / "reports" / report_id / "queries" / "sales_by_store.json"
        assert sql_file.exists()
        assert json_file.exists()

        sql_text = sql_file.read_text()
        assert sql_text.startswith("-- Monthly sales by store")
        assert "SELECT store_name" in sql_text

        json_payload = json.loads(json_file.read_text())
        assert json_payload["row_count"] == 3
        assert json_payload["rows"][0]["store_name"] == "Brooklyn #3"
        column_types = {c["name"]: c["type"] for c in json_payload["columns"]}
        assert column_types["store_name"] == "string"
        assert column_types["month"] == "integer"
        assert column_types["sales"] == "number"

    def test_invalid_slug_rejected(self, report_tools: ReportArtifactTools):
        result = report_tools.save_query(name="Bad Name!", sql="SELECT 1 AS a")
        assert result.success == 0
        assert "match" in (result.error or "")

    def test_empty_sql_rejected(self, report_tools: ReportArtifactTools):
        result = report_tools.save_query(name="empty", sql="   ")
        assert result.success == 0

    def test_write_operations_rejected(self, report_tools: ReportArtifactTools):
        result = report_tools.save_query(name="delete_attempt", sql="DELETE FROM sales WHERE 1=1")
        assert result.success == 0
        assert "read-only" in (result.error or "").lower()

    def test_reuse_name_overwrites(self, report_tools: ReportArtifactTools, project_root: Path):
        first = report_tools.save_query(name="reusable", sql="SELECT store_name FROM sales LIMIT 1")
        assert first.success == 1
        second = report_tools.save_query(name="reusable", sql="SELECT month FROM sales LIMIT 2")
        assert second.success == 1
        report_id = report_tools.report_id
        sql_file = project_root / "reports" / report_id / "queries" / "reusable.sql"
        assert "SELECT month FROM sales" in sql_file.read_text()


# ----------------------------------------------------------------------------- #
# save_main_jsx                                                                 #
# ----------------------------------------------------------------------------- #


_VALID_JSX = """\
import React from 'react';
import { useDatusArtifact } from '@datus/web-artifact';

export default function Demo() {
  const { useQuerySql } = useDatusArtifact();
  const { data } = useQuerySql('queries/sales_by_store');
  return React.createElement('div', null, JSON.stringify(data?.rows ?? []));
}
"""


class TestSaveMainJsx:
    def test_writes_validated_source(self, report_tools: ReportArtifactTools, project_root: Path):
        # Provide the referenced query first so the cross-check passes.
        run = report_tools.save_query(name="sales_by_store", sql="SELECT store_name, month, sales FROM sales")
        assert run.success == 1
        report_id = report_tools.report_id

        result = report_tools.save_main_jsx(_VALID_JSX)
        assert result.success == 1, result.error
        payload = result.result
        assert payload["main_jsx_path"] == f"reports/{report_id}/main.jsx"
        assert "queries/sales_by_store" in payload["query_refs"]

        main_jsx_file = project_root / "reports" / report_id / "main.jsx"
        assert main_jsx_file.exists()
        assert "useDatusArtifact" in main_jsx_file.read_text()

    def test_dangling_query_ref_rejected(self, report_tools: ReportArtifactTools, project_root: Path):
        result = report_tools.save_main_jsx(_VALID_JSX)
        assert result.success == 0
        assert "save_query" in (result.error or "")
        report_id = report_tools.report_id
        # The file MUST NOT have been written when validation fails.
        assert not (project_root / "reports" / report_id / "main.jsx").exists()

    def test_empty_source_rejected(self, report_tools: ReportArtifactTools):
        result = report_tools.save_main_jsx("   ")
        assert result.success == 0
        assert "empty" in (result.error or "").lower()

    def test_missing_default_export_rejected(self, report_tools: ReportArtifactTools):
        no_export = "import React from 'react';\nfunction X() { return null; }\n"
        result = report_tools.save_main_jsx(no_export)
        assert result.success == 0
        assert "export default" in (result.error or "")

    def test_template_literal_sqlid_skipped_in_static_check(
        self, report_tools: ReportArtifactTools, project_root: Path
    ):
        """Template-string sqlIds resolve at runtime; static check should not block them."""
        run = report_tools.save_query(name="sales_by_store", sql="SELECT store_name, month, sales FROM sales")
        assert run.success == 1
        jsx_with_template = """\
import React, { useState } from 'react';
import { useDatusArtifact } from '@datus/web-artifact';
const MONTHS = ['jan', 'feb'];
export default function R() {
  const { useQuerySql } = useDatusArtifact();
  const [m] = useState('jan');
  // literal that must resolve:
  const a = useQuerySql('queries/sales_by_store');
  // template that must NOT trigger the static check:
  const b = useQuerySql(`queries/sales_by_month_${m}`);
  return null;
}
"""
        result = report_tools.save_main_jsx(jsx_with_template)
        assert result.success == 1, result.error
        # Only the literal slug is reported back.
        assert result.result["query_refs"] == ["queries/sales_by_store"]

    def test_oversize_source_rejected(self, report_tools: ReportArtifactTools):
        big = "export default function R(){return null}\n" + ("// padding line\n" * 40000)
        result = report_tools.save_main_jsx(big)
        assert result.success == 0
        assert "exceeds" in (result.error or "").lower()


# ----------------------------------------------------------------------------- #
# ReportFilesystemFuncTool deny rules                                           #
# ----------------------------------------------------------------------------- #


class TestReportFilesystemFuncTool:
    def test_write_main_jsx_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x").mkdir(parents=True)
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("reports/rpt_x/main.jsx", "export default () => null;")
        assert result.success == 0
        assert "save_main_jsx" in (result.error or "")

    def test_write_queries_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "queries").mkdir(parents=True)
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("reports/rpt_x/queries/q.sql", "SELECT 1")
        assert result.success == 0
        assert "save_query" in (result.error or "")

    def test_write_outside_reports_allowed(self, project_root: Path):
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("notes.md", "# scratch")
        assert result.success == 1
        assert (project_root / "notes.md").exists()

    def test_edit_main_jsx_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x").mkdir(parents=True)
        main_jsx = project_root / "reports" / "rpt_x" / "main.jsx"
        main_jsx.write_text("export default () => null;\n")
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.edit_file("reports/rpt_x/main.jsx", "null", "false")
        assert result.success == 0
        assert "save_main_jsx" in (result.error or "")

    def test_read_main_jsx_still_allowed(self, project_root: Path):
        (project_root / "reports" / "rpt_x").mkdir(parents=True)
        main_jsx = project_root / "reports" / "rpt_x" / "main.jsx"
        main_jsx.write_text("export default function R(){return null;}\n")
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.read_file("reports/rpt_x/main.jsx")
        assert result.success == 1
        assert "export default" in (result.result or "")
