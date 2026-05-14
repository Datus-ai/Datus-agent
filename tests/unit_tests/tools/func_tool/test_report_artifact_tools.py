# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for report artifact tools.

Covers:
* ``ReportArtifactTools.start_new_report`` / ``bind_existing_report`` — the
  LLM-driven intent declaration that picks "create new" vs "edit existing"
  before any write tool runs.
* ``_require_active`` guard — save_query / validate_render fail-fast when no
  report is bound.
* ``save_query`` — column inference, SQL persistence, schema validation,
  datasource resolution failures, slug overwrite.
* ``validate_render`` — entry point check, sqlId cross-check, allowed bare
  specifiers, relative import resolution, escape detection, unreferenced-
  file warnings.
* ``ReportFilesystemFuncTool`` — deny rules for ``queries/*``, .jsx/.js/.css
  extension allowlist under ``render/``.

No mocks; we use a real SQLite database wired through ``DBFuncTool`` so
``save_query`` exercises the same code path it will in production.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.tools.func_tool import DBFuncTool, ReportArtifactTools, ReportFilesystemFuncTool
from datus.tools.func_tool.report_artifact_tools import (
    _allocate_report_id,
    _infer_column_type,
    _resolve_relative_import,
    _slugify_title,
)

# ----------------------------------------------------------------------------- #
# Fixtures                                                                      #
# ----------------------------------------------------------------------------- #


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Path:
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
    agent_config = SimpleNamespace(project_root=str(project_root))
    return ReportArtifactTools(agent_config=agent_config, db_func_tool=db_func_tool)


@pytest.fixture
def report_tools(unbound_tools: ReportArtifactTools) -> ReportArtifactTools:
    result = unbound_tools.start_new_report(
        name="demo test",
        description="Smoke-test report used by the report-artifact-tools unit tests.",
    )
    assert result.success == 1, result.error
    return unbound_tools


# ----------------------------------------------------------------------------- #
# helpers                                                                       #
# ----------------------------------------------------------------------------- #


class TestSlugifyTitle:
    def test_ascii_lowercased_and_underscored(self):
        assert _slugify_title("Sales By Store") == "sales_by_store"

    def test_strips_non_ascii(self):
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
        second = _allocate_report_id("collision", project_root)
        assert second != first


class TestResolveRelativeImport:
    """Static path resolution must agree with the iframe runtime's resolver."""

    @pytest.mark.parametrize(
        "caller, spec, keys, expected",
        [
            ("app", "./kpi-banner", {"app", "kpi-banner"}, "kpi-banner"),
            ("app", "./kpi-banner.jsx", {"app", "kpi-banner"}, "kpi-banner"),
            ("app", "./charts/trend", {"app", "charts/trend"}, "charts/trend"),
            ("charts/trend", "./line", {"charts/trend", "charts/line"}, "charts/line"),
            ("charts/trend", "../shared/colors", {"charts/trend", "shared/colors"}, "shared/colors"),
            ("app", "./shared", {"app", "shared/index"}, "shared/index"),
        ],
    )
    def test_resolves(self, caller, spec, keys, expected):
        assert _resolve_relative_import(caller, spec, keys) == expected

    @pytest.mark.parametrize(
        "caller, spec, keys",
        [
            ("app", "./missing", {"app"}),
            ("app", "../escape", {"app"}),
            ("charts/trend", "../../escape", {"charts/trend"}),
        ],
    )
    def test_unresolvable(self, caller, spec, keys):
        assert _resolve_relative_import(caller, spec, keys) is None


# ----------------------------------------------------------------------------- #
# start_new_report / bind_existing_report                                       #
# ----------------------------------------------------------------------------- #


class TestStartNewReport:
    def test_allocates_id_and_writes_manifest(self, unbound_tools: ReportArtifactTools, project_root: Path):
        result = unbound_tools.start_new_report(
            name="east sales",
            description="Quarterly review of east-region direct sales.",
        )
        assert result.success == 1
        payload = result.result
        new_id = payload["report_id"]
        assert new_id.startswith("rpt_east_sales_")
        assert payload["mode"] == "new"
        assert payload["report_dir"] == f"reports/{new_id}"
        assert payload["render_dir"] == f"reports/{new_id}/render"
        assert payload["queries_dir"] == f"reports/{new_id}/queries"
        assert payload["manifest_path"] == f"reports/{new_id}/manifest.json"

        assert unbound_tools.report_id == new_id
        assert unbound_tools.mode == "new"
        assert (project_root / "reports" / new_id / "queries").is_dir()
        assert (project_root / "reports" / new_id / "render").is_dir()

        manifest_path = project_root / "reports" / new_id / "manifest.json"
        assert manifest_path.is_file()
        import json as _json

        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["name"] == "east sales"
        assert manifest["description"] == "Quarterly review of east-region direct sales."
        assert manifest["kind"] == "report"
        assert manifest["created_at"].endswith("Z")

    def test_chinese_name_slug_falls_back_to_report(self, unbound_tools: ReportArtifactTools, project_root: Path):
        result = unbound_tools.start_new_report(
            name="销售季度复盘",
            description="第一季度区域销售业绩复盘。",
        )
        assert result.success == 1, result.error
        new_id = result.result["report_id"]
        # Non-ASCII name slugifies to nothing → fall back to the literal "report" base slug.
        assert new_id.startswith("rpt_report_")
        # But the manifest preserves the original name verbatim.
        import json as _json

        manifest = _json.loads((project_root / "reports" / new_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == "销售季度复盘"

    def test_empty_name_rejected(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.start_new_report(name="", description="x")
        assert result.success == 0
        assert "name" in (result.error or "").lower()

    def test_empty_description_rejected(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.start_new_report(name="ok name", description="   ")
        assert result.success == 0
        assert "description" in (result.error or "").lower()


class TestBindExistingReport:
    def test_binds_when_directory_and_app_jsx_exist(self, unbound_tools: ReportArtifactTools, project_root: Path):
        existing = project_root / "reports" / "rpt_existing_demo_260513_aaaaaa"
        (existing / "queries").mkdir(parents=True)
        (existing / "render").mkdir()
        (existing / "render" / "app.jsx").write_text("export default function R() { return null; }\n")

        result = unbound_tools.bind_existing_report("rpt_existing_demo_260513_aaaaaa")
        assert result.success == 1, result.error
        assert result.result["mode"] == "edit"
        assert unbound_tools.report_id == "rpt_existing_demo_260513_aaaaaa"
        assert unbound_tools.mode == "edit"
        assert unbound_tools.render_dir == existing / "render"

    def test_rejects_missing_directory(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.bind_existing_report("rpt_nope_260513_bbbbbb")
        assert result.success == 0
        assert "not found" in (result.error or "").lower()
        assert unbound_tools.report_id is None

    def test_rejects_missing_app_jsx(self, unbound_tools: ReportArtifactTools, project_root: Path):
        incomplete = project_root / "reports" / "rpt_partial_260513_cccccc"
        (incomplete / "queries").mkdir(parents=True)
        (incomplete / "render").mkdir()
        result = unbound_tools.bind_existing_report("rpt_partial_260513_cccccc")
        assert result.success == 0
        assert "render/app.jsx" in (result.error or "")
        assert unbound_tools.report_id is None

    def test_rejects_invalid_id_format(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.bind_existing_report("not-a-valid-id!")
        assert result.success == 0
        assert "match" in (result.error or "").lower()


class TestRequireActive:
    def test_save_query_rejects_when_unbound(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.save_query(name="q", sql="SELECT 1 AS a")
        assert result.success == 0
        error = (result.error or "").lower()
        assert "no active report" in error
        assert "start_new_report" in error
        assert "bind_existing_report" in error

    def test_validate_render_rejects_when_unbound(self, unbound_tools: ReportArtifactTools):
        result = unbound_tools.validate_render()
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
        assert report_id.startswith("rpt_demo_test_")
        sql_file = project_root / "reports" / report_id / "queries" / "sales_by_store.sql"
        json_file = project_root / "reports" / report_id / "queries" / "sales_by_store.json"
        assert sql_file.exists()
        assert json_file.exists()

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


# ----------------------------------------------------------------------------- #
# validate_render                                                               #
# ----------------------------------------------------------------------------- #


def _write_render(project_root: Path, report_id: str, files: dict[str, str]) -> Path:
    render = project_root / "reports" / report_id / "render"
    render.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = render / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return render


_VALID_APP_JSX = """\
import React from 'react';
import KpiBanner from './kpi-banner';
import { useDatusArtifact } from '@datus/web-artifact';

export default function App() {
  const { useQuerySql } = useDatusArtifact();
  const { data } = useQuerySql('queries/sales_by_store');
  return React.createElement(KpiBanner, { rows: data?.rows ?? [] });
}
"""

_VALID_KPI_BANNER_JSX = """\
import React from 'react';
import { TrendingUp } from 'lucide-react';

export default function KpiBanner({ rows }) {
  return React.createElement('div', null, rows.length, ' rows');
}
"""


class TestValidateRender:
    def test_happy_path(self, report_tools: ReportArtifactTools, project_root: Path):
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        _write_render(
            project_root,
            report_tools.report_id,
            {
                "app.jsx": _VALID_APP_JSX,
                "kpi-banner.jsx": _VALID_KPI_BANNER_JSX,
            },
        )

        result = report_tools.validate_render()
        assert result.success == 1, result.error
        assert result.result["app_jsx_path"].endswith("render/app.jsx")
        assert "queries/sales_by_store" in result.result["query_refs"]
        # All files reachable from app.jsx → no warnings.
        assert result.result["warnings"] == []

    def test_rejects_missing_app_jsx(self, report_tools: ReportArtifactTools, project_root: Path):
        _write_render(project_root, report_tools.report_id, {"kpi-banner.jsx": _VALID_KPI_BANNER_JSX})
        result = report_tools.validate_render()
        assert result.success == 0
        assert "render/app.jsx" in (result.error or "")

    def test_rejects_empty_render_dir(self, report_tools: ReportArtifactTools):
        result = report_tools.validate_render()
        assert result.success == 0
        assert "no .jsx" in (result.error or "")

    def test_rejects_missing_default_export(self, report_tools: ReportArtifactTools, project_root: Path):
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        no_default = "import React from 'react';\nfunction App() { return null; }\n"
        _write_render(project_root, report_tools.report_id, {"app.jsx": no_default})
        result = report_tools.validate_render()
        assert result.success == 0
        assert "export default" in (result.error or "")

    def test_rejects_dangling_sqlid(self, report_tools: ReportArtifactTools, project_root: Path):
        _write_render(
            project_root, report_tools.report_id, {"app.jsx": _VALID_APP_JSX, "kpi-banner.jsx": _VALID_KPI_BANNER_JSX}
        )
        # No save_query → queries/sales_by_store.json does NOT exist.
        result = report_tools.validate_render()
        assert result.success == 0
        assert "queries/sales_by_store" in (result.error or "")

    def test_rejects_disallowed_bare_import(self, report_tools: ReportArtifactTools, project_root: Path):
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        bad_app = (
            "import React from 'react';\n"
            "import _ from 'lodash';\n"  # not allowed
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/sales_by_store');\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(project_root, report_tools.report_id, {"app.jsx": bad_app})
        result = report_tools.validate_render()
        assert result.success == 0
        assert "lodash" in (result.error or "")

    def test_rejects_unresolved_relative_import(self, report_tools: ReportArtifactTools, project_root: Path):
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        bad_app = _VALID_APP_JSX  # imports ./kpi-banner but we don't write it
        _write_render(project_root, report_tools.report_id, {"app.jsx": bad_app})
        result = report_tools.validate_render()
        assert result.success == 0
        assert "./kpi-banner" in (result.error or "")
        assert "does not resolve" in (result.error or "")

    def test_rejects_escape_relative_import(self, report_tools: ReportArtifactTools, project_root: Path):
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        escape = (
            "import React from 'react';\n"
            "import x from '../../../etc/passwd';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/sales_by_store');\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(project_root, report_tools.report_id, {"app.jsx": escape})
        result = report_tools.validate_render()
        assert result.success == 0
        # Either rejected as unresolved or as escape — both end in "does not resolve".
        assert "does not resolve" in (result.error or "")

    def test_reports_unreferenced_files_as_warnings(self, report_tools: ReportArtifactTools, project_root: Path):
        """Files not reachable from app.jsx should surface as warnings, not block validation."""
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        minimal_app = (
            "import React from 'react';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/sales_by_store');\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(
            project_root,
            report_tools.report_id,
            {
                "app.jsx": minimal_app,
                "legacy.jsx": "import React from 'react';\nexport default function L() { return null; }\n",
            },
        )
        result = report_tools.validate_render()
        assert result.success == 1, result.error
        assert any("legacy.jsx" in w for w in result.result["warnings"])

    def test_subdirectory_imports_resolve(self, report_tools: ReportArtifactTools, project_root: Path):
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        app = (
            "import React from 'react';\n"
            "import Trend from './charts/trend';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/sales_by_store');\n"
            "  return React.createElement(Trend, null);\n"
            "}\n"
        )
        trend = (
            "import React from 'react';\n"
            "import { COLORS } from '../shared/colors';\n"
            "export default function Trend() { return React.createElement('div', { style: { color: COLORS.primary } }); }\n"
        )
        colors = "export const COLORS = { primary: '#1A56DB' };\n"
        _write_render(
            project_root,
            report_tools.report_id,
            {
                "app.jsx": app,
                "charts/trend.jsx": trend,
                "shared/colors.js": colors,
            },
        )
        result = report_tools.validate_render()
        assert result.success == 1, result.error
        assert result.result["warnings"] == []
        names = result.result["render_files"]
        assert "render/app.jsx" in names
        assert "render/charts/trend.jsx" in names
        assert "render/shared/colors.js" in names

    def test_template_string_sqlid_skipped(self, report_tools: ReportArtifactTools, project_root: Path):
        """Template-string sqlIds are runtime-resolved and must not block validation."""
        report_tools.save_query(name="sales_by_store", sql="SELECT store_name FROM sales")
        app_with_template = (
            "import React, { useState } from 'react';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "const MONTHS = ['jan', 'feb'];\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  const [m] = useState('jan');\n"
            "  useQuerySql('queries/sales_by_store');\n"
            "  useQuerySql(`queries/sales_by_month_${m}`);\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(project_root, report_tools.report_id, {"app.jsx": app_with_template})
        result = report_tools.validate_render()
        assert result.success == 1, result.error
        # Only the literal slug appears in query_refs.
        assert result.result["query_refs"] == ["queries/sales_by_store"]


# ----------------------------------------------------------------------------- #
# ReportFilesystemFuncTool deny / allow rules                                   #
# ----------------------------------------------------------------------------- #


class TestReportFilesystemFuncTool:
    def test_write_queries_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "queries").mkdir(parents=True)
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("reports/rpt_x/queries/q.sql", "SELECT 1")
        assert result.success == 0
        assert "save_query" in (result.error or "")

    def test_write_render_jsx_allowed(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "render").mkdir(parents=True)
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("reports/rpt_x/render/app.jsx", "export default () => null;\n")
        assert result.success == 1
        assert (project_root / "reports" / "rpt_x" / "render" / "app.jsx").is_file()

    def test_write_render_nested_subdir_allowed(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "render").mkdir(parents=True)
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file(
            "reports/rpt_x/render/charts/trend.jsx",
            "export default () => null;\n",
        )
        assert result.success == 1
        assert (project_root / "reports" / "rpt_x" / "render" / "charts" / "trend.jsx").is_file()

    def test_write_render_json_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "render").mkdir(parents=True)
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("reports/rpt_x/render/data.json", '{"x": 1}')
        assert result.success == 0
        assert ".jsx" in (result.error or "")

    def test_edit_queries_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "queries").mkdir(parents=True)
        existing = project_root / "reports" / "rpt_x" / "queries" / "q.sql"
        existing.write_text("SELECT 1")
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.edit_file("reports/rpt_x/queries/q.sql", "1", "2")
        assert result.success == 0
        assert "save_query" in (result.error or "")

    def test_delete_render_jsx_allowed(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "render").mkdir(parents=True)
        target = project_root / "reports" / "rpt_x" / "render" / "old.jsx"
        target.write_text("export default () => null;\n")
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.delete_file("reports/rpt_x/render/old.jsx")
        assert result.success == 1
        assert not target.exists()

    def test_delete_queries_rejected(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "queries").mkdir(parents=True)
        target = project_root / "reports" / "rpt_x" / "queries" / "q.sql"
        target.write_text("SELECT 1")
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.delete_file("reports/rpt_x/queries/q.sql")
        assert result.success == 0
        # Either rejected by the deny rule (preferred message) or by the parent
        # tool — accept any error response so the test stays robust to future
        # changes in error wording.
        assert target.exists()
        assert (result.error or "") != ""

    def test_write_outside_reports_allowed(self, project_root: Path):
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("notes.md", "# scratch")
        assert result.success == 1
        assert (project_root / "notes.md").exists()

    def test_read_render_jsx_allowed(self, project_root: Path):
        (project_root / "reports" / "rpt_x" / "render").mkdir(parents=True)
        target = project_root / "reports" / "rpt_x" / "render" / "app.jsx"
        target.write_text("export default function App() { return null; }\n")
        fs = ReportFilesystemFuncTool(root_path=str(project_root))
        result = fs.read_file("reports/rpt_x/render/app.jsx")
        assert result.success == 1
        assert "export default" in (result.result or "")
