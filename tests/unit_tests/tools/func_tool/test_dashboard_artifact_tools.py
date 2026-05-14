# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for dashboard artifact tools.

Covers:
* Helpers: ``sql_quote_scalar``, ``resolve_bind_placeholders``,
  ``render_dashboard_template``.
* ``DashboardArtifactTools.start_new_dashboard`` /
  ``bind_existing_dashboard`` — LLM-driven intent declaration. The LLM
  supplies the ``slug`` directly; the tool refuses to overwrite an
  existing directory.
* ``_require_active`` guard — save_query_template / validate_render fail
  fast when no dashboard is bound.
* ``save_query_template`` — header parsing, type coercion through the
  trial render, column inference, persistence on disk, error paths.
* ``validate_render`` — entry point check, useQuerySql 2-arg requirement,
  params-key contract against the saved template declaration, import
  allowlist, escape detection, unreferenced-file warnings.
* ``DashboardFilesystemFuncTool`` — deny rules for ``queries/*``,
  ``.jsx/.js/.css`` allowlist under ``render/``.

No mocks; we use a real SQLite database wired through ``DBFuncTool`` so
``save_query_template`` exercises the production code path.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.schemas.gen_visual_dashboard_models import TemplateParamDecl
from datus.tools.func_tool import DashboardArtifactTools, DashboardFilesystemFuncTool, DBFuncTool
from datus.tools.func_tool.dashboard_artifact_tools import (
    render_dashboard_template,
    resolve_bind_placeholders,
    sql_quote_scalar,
)

# ----------------------------------------------------------------------------- #
# Fixtures                                                                      #
# ----------------------------------------------------------------------------- #


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "demo.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE sales (region TEXT, month TEXT, revenue REAL)")
        conn.executemany(
            "INSERT INTO sales VALUES (?,?,?)",
            [
                ("NA", "2026-01", 1500.0),
                ("EU", "2026-01", 1200.0),
                ("NA", "2026-02", 1800.0),
                ("APAC", "2026-02", 950.0),
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
def unbound_tools(db_func_tool: DBFuncTool, project_root: Path) -> DashboardArtifactTools:
    agent_config = SimpleNamespace(project_root=str(project_root))
    return DashboardArtifactTools(agent_config=agent_config, db_func_tool=db_func_tool)


@pytest.fixture
def dashboard_tools(unbound_tools: DashboardArtifactTools) -> DashboardArtifactTools:
    result = unbound_tools.start_new_dashboard(
        slug="demo_test",
        name="demo test",
        description="Smoke-test dashboard used by the dashboard-artifact-tools unit tests.",
    )
    assert result.success == 1, result.error
    return unbound_tools


# ----------------------------------------------------------------------------- #
# Helper functions                                                              #
# ----------------------------------------------------------------------------- #


class TestSqlQuoteScalar:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, "NULL"),
            (True, "TRUE"),
            (False, "FALSE"),
            (42, "42"),
            (3.14, "3.14"),
            ("hello", "'hello'"),
            ("o'brien", "'o''brien'"),
            ("2026-01-01", "'2026-01-01'"),
        ],
    )
    def test_quoting(self, value, expected):
        assert sql_quote_scalar(value) == expected


class TestResolveBindPlaceholders:
    def test_scalar_substitution(self):
        decls = [TemplateParamDecl(name="d", type="date"), TemplateParamDecl(name="n", type="integer")]
        out = resolve_bind_placeholders("WHERE d = :d AND n > :n", decls, {"d": "2026-01-01", "n": 5})
        assert out == "WHERE d = '2026-01-01' AND n > 5"

    def test_array_expanded_to_in_list(self):
        decls = [TemplateParamDecl(name="xs", type="string[]")]
        out = resolve_bind_placeholders("WHERE r IN :xs", decls, {"xs": ["NA", "EU"]})
        assert out == "WHERE r IN ('NA', 'EU')"

    def test_empty_array_yields_unsatisfiable_predicate(self):
        decls = [TemplateParamDecl(name="xs", type="string[]")]
        out = resolve_bind_placeholders("WHERE r IN :xs", decls, {"xs": []})
        assert "(NULL)" in out

    def test_missing_required_raises(self):
        decls = [TemplateParamDecl(name="x", type="string", required=True)]
        with pytest.raises(ValueError, match="missing from sample_params"):
            resolve_bind_placeholders("SELECT :x", decls, {})

    def test_missing_optional_left_in_place(self):
        decls = [TemplateParamDecl(name="x", type="string", required=False)]
        out = resolve_bind_placeholders("SELECT :x", decls, {})
        assert out == "SELECT :x"

    def test_unknown_placeholder_left_in_place(self):
        # Caller may use ``:foo`` for something other than a declared param
        # (e.g. a sentinel). Validator catches at execute time; we don't crash.
        decls = [TemplateParamDecl(name="x", type="string")]
        out = resolve_bind_placeholders("SELECT :foo", decls, {"x": "v"})
        assert out == "SELECT :foo"

    def test_postgres_double_colon_cast_preserved(self):
        decls = [TemplateParamDecl(name="x", type="integer")]
        out = resolve_bind_placeholders("SELECT :x::integer", decls, {"x": 7})
        # ``::integer`` is a PostgreSQL cast — not a placeholder.
        assert "7::integer" in out

    def test_array_with_non_list_raises(self):
        decls = [TemplateParamDecl(name="xs", type="string[]")]
        with pytest.raises(ValueError, match="declared as an array"):
            resolve_bind_placeholders("SELECT :xs", decls, {"xs": "not-a-list"})


class TestRenderDashboardTemplate:
    def test_full_render_pipeline(self):
        tpl = (
            "-- @datus-params start_date:date, regions:string[]:optional\n"
            "SELECT region, SUM(revenue) FROM sales\n"
            "WHERE month >= :start_date\n"
            "{% if regions %}AND region IN :regions{% endif %}\n"
            "GROUP BY region"
        )
        decls = [
            TemplateParamDecl(name="start_date", type="date", required=True),
            TemplateParamDecl(name="regions", type="string[]", required=False),
        ]
        out = render_dashboard_template(tpl, decls, {"start_date": "2026-01-01", "regions": ["NA"]})
        assert "WHERE month >= '2026-01-01'" in out
        assert "AND region IN ('NA')" in out

    def test_optional_param_branch_skipped_when_omitted(self):
        tpl = (
            "-- @datus-params start_date:date, regions:string[]:optional\n"
            "SELECT 1 WHERE :start_date IS NOT NULL\n"
            "{% if regions %}AND region IN :regions{% endif %}"
        )
        decls = [
            TemplateParamDecl(name="start_date", type="date", required=True),
            TemplateParamDecl(name="regions", type="string[]", required=False),
        ]
        out = render_dashboard_template(tpl, decls, {"start_date": "2026-01-01"})
        assert "AND region IN" not in out

    def test_jinja_parse_error_raises(self):
        decls = [TemplateParamDecl(name="x", type="string")]
        with pytest.raises(ValueError, match="Jinja2 parse error"):
            render_dashboard_template("{% if x %}", decls, {"x": "v"})


# ----------------------------------------------------------------------------- #
# start_new_dashboard / bind_existing_dashboard                                 #
# ----------------------------------------------------------------------------- #


class TestStartNewDashboard:
    def test_uses_supplied_slug_and_writes_manifest(self, unbound_tools: DashboardArtifactTools, project_root: Path):
        result = unbound_tools.start_new_dashboard(
            slug="north_sales",
            name="north sales",
            description="Live north-region sales dashboard with date and channel filters.",
        )
        assert result.success == 1
        payload = result.result
        assert payload["dashboard_slug"] == "north_sales"
        assert payload["mode"] == "new"
        assert payload["dashboard_dir"] == "dashboards/north_sales"
        assert payload["render_dir"] == "dashboards/north_sales/render"
        assert payload["queries_dir"] == "dashboards/north_sales/queries"
        assert payload["manifest_path"] == "dashboards/north_sales/manifest.json"

        assert unbound_tools.dashboard_slug == "north_sales"
        assert (project_root / "dashboards" / "north_sales" / "queries").is_dir()
        assert (project_root / "dashboards" / "north_sales" / "render").is_dir()

        manifest_path = project_root / "dashboards" / "north_sales" / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["slug"] == "north_sales"
        assert manifest["name"] == "north sales"
        assert manifest["description"] == "Live north-region sales dashboard with date and channel filters."
        assert manifest["kind"] == "dashboard"
        assert manifest["created_at"].endswith("Z")

    def test_chinese_name_is_preserved_in_manifest(self, unbound_tools: DashboardArtifactTools, project_root: Path):
        result = unbound_tools.start_new_dashboard(
            slug="sales_live",
            name="销售看板",
            description="实时销售看板，按地区过滤。",
        )
        assert result.success == 1, result.error
        manifest = json.loads(
            (project_root / "dashboards" / "sales_live" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["slug"] == "sales_live"
        assert manifest["name"] == "销售看板"

    def test_empty_name_rejected(self, unbound_tools: DashboardArtifactTools):
        result = unbound_tools.start_new_dashboard(slug="ok", name=" ", description="x")
        assert result.success == 0
        assert "name" in (result.error or "").lower()

    def test_empty_description_rejected(self, unbound_tools: DashboardArtifactTools):
        result = unbound_tools.start_new_dashboard(slug="ok", name="ok", description="")
        assert result.success == 0
        assert "description" in (result.error or "").lower()

    @pytest.mark.parametrize(
        "bad_slug",
        [
            "",
            "Has-Hyphen",
            "has space",
            "中文",
            "a" * 81,
        ],
    )
    def test_invalid_slug_rejected(self, unbound_tools: DashboardArtifactTools, bad_slug: str):
        result = unbound_tools.start_new_dashboard(slug=bad_slug, name="ok", description="ok")
        assert result.success == 0
        assert "slug" in (result.error or "").lower()

    def test_existing_directory_rejected(self, unbound_tools: DashboardArtifactTools, project_root: Path):
        (project_root / "dashboards" / "preexisting").mkdir(parents=True)
        result = unbound_tools.start_new_dashboard(slug="preexisting", name="x", description="y")
        assert result.success == 0
        assert "already exists" in (result.error or "").lower()


class TestBindExistingDashboard:
    def test_binds_when_directory_and_app_jsx_exist(self, unbound_tools: DashboardArtifactTools, project_root: Path):
        existing = project_root / "dashboards" / "existing_demo"
        (existing / "queries").mkdir(parents=True)
        (existing / "render").mkdir()
        (existing / "render" / "app.jsx").write_text("export default function D() { return null; }\n")

        result = unbound_tools.bind_existing_dashboard("existing_demo")
        assert result.success == 1, result.error
        assert result.result["mode"] == "edit"
        assert result.result["dashboard_slug"] == "existing_demo"
        assert unbound_tools.dashboard_slug == "existing_demo"

    def test_rejects_missing_directory(self, unbound_tools: DashboardArtifactTools):
        result = unbound_tools.bind_existing_dashboard("nope")
        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    def test_rejects_missing_app_jsx(self, unbound_tools: DashboardArtifactTools, project_root: Path):
        incomplete = project_root / "dashboards" / "partial"
        (incomplete / "queries").mkdir(parents=True)
        (incomplete / "render").mkdir()
        result = unbound_tools.bind_existing_dashboard("partial")
        assert result.success == 0
        assert "render/app.jsx" in (result.error or "")

    def test_rejects_invalid_slug_format(self, unbound_tools: DashboardArtifactTools):
        result = unbound_tools.bind_existing_dashboard("Not-A-Valid-Slug!")
        assert result.success == 0
        assert "match" in (result.error or "").lower()


class TestRequireActive:
    def test_save_query_template_rejects_when_unbound(self, unbound_tools: DashboardArtifactTools):
        result = unbound_tools.save_query_template(
            name="q", sql_template="-- @datus-params x:string\nSELECT :x", sample_params={"x": "v"}
        )
        assert result.success == 0
        error = (result.error or "").lower()
        assert "no active dashboard" in error
        assert "start_new_dashboard" in error
        assert "bind_existing_dashboard" in error

    def test_validate_render_rejects_when_unbound(self, unbound_tools: DashboardArtifactTools):
        result = unbound_tools.validate_render()
        assert result.success == 0
        assert "no active dashboard" in (result.error or "").lower()


# ----------------------------------------------------------------------------- #
# save_query_template                                                           #
# ----------------------------------------------------------------------------- #


_REVENUE_TEMPLATE = (
    "-- @datus-params month_floor:string, regions:string[]:optional\n"
    "SELECT region, SUM(revenue) AS revenue FROM sales\n"
    "WHERE month >= :month_floor\n"
    "{% if regions %}AND region IN :regions{% endif %}\n"
    "GROUP BY region\n"
    "ORDER BY revenue DESC"
)


class TestSaveQueryTemplate:
    def test_persists_sql_and_params_json(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        result = dashboard_tools.save_query_template(
            name="revenue_by_region",
            sql_template=_REVENUE_TEMPLATE,
            sample_params={"month_floor": "2026-01", "regions": ["NA", "EU"]},
            description="Revenue per region with optional filter",
        )
        assert result.success == 1, result.error
        payload = result.result
        assert payload["name"] == "revenue_by_region"
        assert payload["data_ref"] == "queries/revenue_by_region"
        names = [c["name"] for c in payload["columns"]]
        assert names == ["region", "revenue"]
        # 2 rows are inside NA / EU + month >= 2026-01.
        assert payload["sample_row_count"] == 2

        dash_slug = dashboard_tools.dashboard_slug or ""
        sql_path = project_root / "dashboards" / dash_slug / "queries" / "revenue_by_region.sql.j2"
        params_path = project_root / "dashboards" / dash_slug / "queries" / "revenue_by_region.params.json"
        assert sql_path.exists()
        assert params_path.exists()
        meta = json.loads(params_path.read_text())
        decl_names = [p["name"] for p in meta["params"]]
        assert decl_names == ["month_floor", "regions"]

    def test_invalid_slug_rejected(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.save_query_template(
            name="Bad Name!", sql_template="-- @datus-params x:string\nSELECT :x", sample_params={"x": "v"}
        )
        assert result.success == 0
        assert "match" in (result.error or "")

    def test_empty_template_rejected(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.save_query_template(name="empty", sql_template="   ", sample_params={})
        assert result.success == 0
        assert "must not be empty" in (result.error or "")

    def test_unknown_sample_param_rejected(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.save_query_template(
            name="bad_params",
            sql_template=_REVENUE_TEMPLATE,
            sample_params={"month_floor": "2026-01", "rogue": 1},
        )
        assert result.success == 0
        assert "not declared" in (result.error or "")

    def test_missing_required_sample_param_rejected(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.save_query_template(
            name="bad_params2",
            sql_template=_REVENUE_TEMPLATE,
            sample_params={},
        )
        assert result.success == 0
        assert "missing required" in (result.error or "")

    def test_missing_header_rejected(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.save_query_template(name="no_header", sql_template="SELECT 1 AS a", sample_params={})
        assert result.success == 0
        assert "@datus-params" in (result.error or "")

    def test_write_operations_rejected(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.save_query_template(
            name="bad_write",
            sql_template=("-- @datus-params region:string\nDELETE FROM sales WHERE region = :region"),
            sample_params={"region": "NA"},
        )
        assert result.success == 0
        assert "read-only" in (result.error or "").lower()

    def test_sample_params_must_be_dict(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.save_query_template(
            name="bad_args",
            sql_template="-- @datus-params x:string\nSELECT :x",
            sample_params="not a dict",  # type: ignore[arg-type]
        )
        assert result.success == 0
        assert "JSON object" in (result.error or "")


# ----------------------------------------------------------------------------- #
# validate_render                                                               #
# ----------------------------------------------------------------------------- #


def _seed_template(dashboard_tools: DashboardArtifactTools, slug: str = "revenue_by_region") -> None:
    """Persist a sample template the validator can match against."""
    result = dashboard_tools.save_query_template(
        name=slug,
        sql_template=_REVENUE_TEMPLATE,
        sample_params={"month_floor": "2026-01"},
    )
    assert result.success == 1, result.error


def _write_render(project_root: Path, dashboard_slug: str, files: dict[str, str]) -> Path:
    render = project_root / "dashboards" / dashboard_slug / "render"
    render.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = render / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return render


_VALID_APP_JSX = """\
import React from 'react';
import { useDatusArtifact } from '@datus/web-artifact';

export default function App() {
  const { useQuerySql } = useDatusArtifact();
  const { data } = useQuerySql('queries/revenue_by_region', { month_floor: '2026-01' });
  return React.createElement('pre', null, JSON.stringify(data?.rows ?? []));
}
"""


class TestValidateRender:
    def test_happy_path(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        _seed_template(dashboard_tools)
        _write_render(project_root, dashboard_tools.dashboard_slug, {"app.jsx": _VALID_APP_JSX})

        result = dashboard_tools.validate_render()
        assert result.success == 1, result.error
        assert result.result["app_jsx_path"].endswith("render/app.jsx")
        assert "queries/revenue_by_region" in result.result["query_refs"]

    def test_rejects_useQuerySql_without_params_arg(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        _seed_template(dashboard_tools)
        app_no_params = (
            "import React from 'react';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/revenue_by_region');\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(project_root, dashboard_tools.dashboard_slug, {"app.jsx": app_no_params})
        result = dashboard_tools.validate_render()
        assert result.success == 0
        assert "second `params` argument" in (result.error or "")

    def test_rejects_missing_required_param_key(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        _seed_template(dashboard_tools)
        app_missing_key = (
            "import React from 'react';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/revenue_by_region', {});\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(project_root, dashboard_tools.dashboard_slug, {"app.jsx": app_missing_key})
        result = dashboard_tools.validate_render()
        assert result.success == 0
        assert "missing required" in (result.error or "")
        assert "month_floor" in (result.error or "")

    def test_rejects_unknown_param_key(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        _seed_template(dashboard_tools)
        app_extra_key = (
            "import React from 'react';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/revenue_by_region', { month_floor: '2026-01', extra: 1 });\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(project_root, dashboard_tools.dashboard_slug, {"app.jsx": app_extra_key})
        result = dashboard_tools.validate_render()
        assert result.success == 0
        assert "unknown" in (result.error or "")
        assert "extra" in (result.error or "")

    def test_rejects_dangling_slug(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        # No save_query_template — but app.jsx references one.
        _write_render(project_root, dashboard_tools.dashboard_slug, {"app.jsx": _VALID_APP_JSX})
        result = dashboard_tools.validate_render()
        assert result.success == 0
        assert "queries/revenue_by_region" in (result.error or "")

    def test_rejects_missing_default_export(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        _seed_template(dashboard_tools)
        no_default = "import React from 'react';\nfunction App() { return null; }\n"
        _write_render(project_root, dashboard_tools.dashboard_slug, {"app.jsx": no_default})
        result = dashboard_tools.validate_render()
        assert result.success == 0
        assert "export default" in (result.error or "")

    def test_rejects_disallowed_bare_import(self, dashboard_tools: DashboardArtifactTools, project_root: Path):
        _seed_template(dashboard_tools)
        bad_app = (
            "import React from 'react';\n"
            "import _ from 'lodash';\n"
            "import { useDatusArtifact } from '@datus/web-artifact';\n"
            "export default function App() {\n"
            "  const { useQuerySql } = useDatusArtifact();\n"
            "  useQuerySql('queries/revenue_by_region', { month_floor: '2026-01' });\n"
            "  return null;\n"
            "}\n"
        )
        _write_render(project_root, dashboard_tools.dashboard_slug, {"app.jsx": bad_app})
        result = dashboard_tools.validate_render()
        assert result.success == 0
        assert "lodash" in (result.error or "")

    def test_rejects_empty_render_dir(self, dashboard_tools: DashboardArtifactTools):
        result = dashboard_tools.validate_render()
        assert result.success == 0
        assert "no .jsx" in (result.error or "")


# ----------------------------------------------------------------------------- #
# DashboardFilesystemFuncTool deny / allow rules                                #
# ----------------------------------------------------------------------------- #


class TestDashboardFilesystemFuncTool:
    def test_write_queries_rejected(self, project_root: Path):
        (project_root / "dashboards" / "x" / "queries").mkdir(parents=True)
        fs = DashboardFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("dashboards/x/queries/q.sql.j2", "-- @datus-params x:string\nSELECT :x")
        assert result.success == 0
        assert "save_query_template" in (result.error or "")

    def test_write_render_jsx_allowed(self, project_root: Path):
        (project_root / "dashboards" / "x" / "render").mkdir(parents=True)
        fs = DashboardFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("dashboards/x/render/app.jsx", "export default () => null;\n")
        assert result.success == 1
        assert (project_root / "dashboards" / "x" / "render" / "app.jsx").is_file()

    def test_write_render_json_rejected(self, project_root: Path):
        (project_root / "dashboards" / "x" / "render").mkdir(parents=True)
        fs = DashboardFilesystemFuncTool(root_path=str(project_root))
        result = fs.write_file("dashboards/x/render/data.json", '{"x": 1}')
        assert result.success == 0
        assert ".jsx" in (result.error or "")

    def test_edit_queries_rejected(self, project_root: Path):
        (project_root / "dashboards" / "x" / "queries").mkdir(parents=True)
        existing = project_root / "dashboards" / "x" / "queries" / "q.sql.j2"
        existing.write_text("-- @datus-params x:string\nSELECT :x")
        fs = DashboardFilesystemFuncTool(root_path=str(project_root))
        result = fs.edit_file("dashboards/x/queries/q.sql.j2", ":x", ":x AS v")
        assert result.success == 0
        assert "save_query_template" in (result.error or "")

    def test_delete_queries_rejected(self, project_root: Path):
        (project_root / "dashboards" / "x" / "queries").mkdir(parents=True)
        target = project_root / "dashboards" / "x" / "queries" / "q.params.json"
        target.write_text("{}")
        fs = DashboardFilesystemFuncTool(root_path=str(project_root))
        result = fs.delete_file("dashboards/x/queries/q.params.json")
        assert result.success == 0
        assert target.exists()
