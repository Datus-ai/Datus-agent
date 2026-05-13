# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for report artifact tools.

Covers:
* ``ReportArtifactTools.save_query`` — column inference, SQL persistence,
  schema validation, datasource resolution failures.
* ``ReportArtifactTools.save_manifest`` — schema validation, data_ref
  cross-check against persisted queries, atomic rewrite.
* ``ReportFilesystemFuncTool`` — deny rules for ``manifest.json`` and
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
from datus.tools.func_tool.report_artifact_tools import _infer_column_type

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
def report_tools(sqlite_db: Path, project_root: Path) -> ReportArtifactTools:
    from datus.tools.db_tools.config import SQLiteConfig
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector

    connector = SQLiteConnector(SQLiteConfig(db_path=str(sqlite_db)))
    db_func_tool = DBFuncTool(connector_or_manager=connector)

    agent_config = SimpleNamespace(project_root=str(project_root))
    return ReportArtifactTools(
        agent_config=agent_config,
        report_id="rpt_demo_test_001",
        db_func_tool=db_func_tool,
    )


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

        sql_file = project_root / "reports" / "rpt_demo_test_001" / "queries" / "sales_by_store.sql"
        json_file = project_root / "reports" / "rpt_demo_test_001" / "queries" / "sales_by_store.json"
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
        sql_file = project_root / "reports" / "rpt_demo_test_001" / "queries" / "reusable.sql"
        assert "SELECT month FROM sales" in sql_file.read_text()


# ----------------------------------------------------------------------------- #
# save_manifest                                                                 #
# ----------------------------------------------------------------------------- #


def _basic_manifest(report_id: str = "rpt_demo_test_001") -> dict:
    return {
        "id": report_id,
        "title": "demo",
        "created_at": "2026-05-13T10:00:00Z",
        "sections": [
            {"id": "blk_001", "type": "markdown", "content": "# hi"},
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
        ],
    }


class TestSaveManifest:
    def test_writes_validated_manifest(self, report_tools: ReportArtifactTools, project_root: Path):
        run = report_tools.save_query(
            name="sales_by_store",
            sql="SELECT store_name, month, sales FROM sales",
        )
        assert run.success == 1

        result = report_tools.save_manifest(_basic_manifest())
        assert result.success == 1, result.error
        manifest_path = project_root / "reports" / "rpt_demo_test_001" / "manifest.json"
        assert manifest_path.exists()
        on_disk = json.loads(manifest_path.read_text())
        assert on_disk["id"] == "rpt_demo_test_001"
        assert len(on_disk["sections"]) == 2

    def test_missing_data_ref_blocks_write(self, report_tools: ReportArtifactTools, project_root: Path):
        manifest_path = project_root / "reports" / "rpt_demo_test_001" / "manifest.json"
        assert not manifest_path.exists()

        result = report_tools.save_manifest(_basic_manifest())
        assert result.success == 0
        assert "save_query" in (result.error or "")
        assert not manifest_path.exists()

    def test_wrong_report_id_blocks_write(self, report_tools: ReportArtifactTools):
        bad = _basic_manifest(report_id="rpt_other")
        result = report_tools.save_manifest(bad)
        assert result.success == 0
        assert "report id" in (result.error or "").lower()

    def test_invalid_manifest_blocks_write(self, report_tools: ReportArtifactTools, project_root: Path):
        report_tools.save_query(name="q", sql="SELECT 1 AS a")
        bad = _basic_manifest()
        # Inject a layout with mismatched children/columns.
        bad["sections"] = [
            {
                "id": "blk_layout",
                "type": "layout",
                "columns": [1, 1],
                "children": [
                    {"id": "blk_layout_c0", "type": "markdown", "content": "only one"},
                ],
            }
        ]
        result = report_tools.save_manifest(bad)
        assert result.success == 0
        assert "validation" in (result.error or "").lower()
        manifest_path = project_root / "reports" / "rpt_demo_test_001" / "manifest.json"
        assert not manifest_path.exists()


# ----------------------------------------------------------------------------- #
# ReportFilesystemFuncTool deny rules                                           #
# ----------------------------------------------------------------------------- #


class TestReportFilesystemFuncTool:
    def test_write_manifest_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x").mkdir(parents=True)
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("reports/rpt_x/manifest.json", '{"id":"rpt_x"}')
        assert result.success == 0
        assert "save_manifest" in (result.error or "")

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

    def test_edit_manifest_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x").mkdir(parents=True)
        manifest = project_root / "reports" / "rpt_x" / "manifest.json"
        manifest.write_text('{"id":"rpt_x"}')
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.edit_file("reports/rpt_x/manifest.json", "rpt_x", "rpt_y")
        assert result.success == 0
        assert "save_manifest" in (result.error or "")

    def test_read_manifest_still_allowed(self, project_root: Path):
        (project_root / "reports" / "rpt_x").mkdir(parents=True)
        manifest = project_root / "reports" / "rpt_x" / "manifest.json"
        manifest.write_text('{"id":"rpt_x"}')
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.read_file("reports/rpt_x/manifest.json")
        assert result.success == 1
        assert "rpt_x" in (result.result or "")
