# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.api.services.dashboard_service`` — CI level, zero external deps.

Covers the on-disk artifact bundle walk, the on-disk template-pair loader,
and the agent-only branches of ``DashboardService.run_query``:

* ``published_version is None`` (IDE live-edit preview) feeds the render
  from ``dashboards/<slug>/queries/<slug>.{sql.j2,params.json}``.
* ``published_version`` set with no ``published_template_loader`` is
  rejected with ``INVALID_PUBLISHED_VERSION`` — the agent-only deployment
  has no Postgres snapshot table, so the loader injection seam is the
  only way to enable that branch.

The Datus-backend-side wrapper covers the published-snapshot path
through its own ``tests/unit/test_dashboard_service_run_query.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datus.api.services.dashboard_service import (
    DashboardService,
    _load_local_template_pair,
)

_SAMPLE_SQL_J2 = "SELECT * FROM sales WHERE region = :region;\n"
_SAMPLE_META = {
    "slug": "by_region",
    "description": "Sales by region",
    "datasource": "warehouse",
    "params": [{"name": "region", "type": "string", "required": True}],
    "columns": [{"name": "region", "type": "string"}, {"name": "amount", "type": "number"}],
    "sample_params": {"region": "APAC"},
    "sample_row_count": 1,
    "saved_at": "2026-05-20T00:00:00Z",
}
_SAMPLE_MANIFEST = {
    "slug": "demo",
    "name": "Demo Dashboard",
    "description": "Just a demo",
    "kind": "dashboard",
    "created_at": "2026-05-20T00:00:00Z",
}
_SAMPLE_APP_JSX = "import React from 'react';\nexport default function App() { return null; }\n"


def _write_dashboard(
    project_files_root: Path,
    *,
    dashboard_slug: str = "demo",
    query_slug: str = "by_region",
    with_template: bool = True,
) -> Path:
    """Lay out a minimal on-disk dashboard fixture under
    ``<project_files_root>/dashboards/<slug>/``.

    Returns the dashboard directory.
    """
    dashboard_dir = project_files_root / "dashboards" / dashboard_slug
    (dashboard_dir / "render").mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "render" / "app.jsx").write_text(_SAMPLE_APP_JSX, encoding="utf-8")
    (dashboard_dir / "manifest.json").write_text(json.dumps(_SAMPLE_MANIFEST), encoding="utf-8")
    if with_template:
        queries_dir = dashboard_dir / "queries"
        queries_dir.mkdir(parents=True, exist_ok=True)
        (queries_dir / f"{query_slug}.sql.j2").write_text(_SAMPLE_SQL_J2, encoding="utf-8")
        (queries_dir / f"{query_slug}.params.json").write_text(json.dumps(_SAMPLE_META), encoding="utf-8")
    return dashboard_dir


def _patch_executor(monkeypatch, *, captured: dict) -> None:
    """Replace the DB-execution suffix of ``run_query`` so tests focus on
    the template-source switch / render output, not the live connector path.

    The agent service late-imports ``datus.tools.func_tool`` at call time so
    monkeypatching ``DBFuncTool`` on the module attribute is safe.
    """

    class _FakeExecResult:
        success = True
        sql_return = [{"region": "APAC", "amount": 100}]

    class _FakeConnector:
        def execute_query(self, sql, result_format="list"):
            captured["sql"] = sql
            captured["result_format"] = result_format
            return _FakeExecResult()

    class _FakeDBFuncTool:
        def __init__(self, *, agent_config, sub_agent_name):
            captured["sub_agent_name"] = sub_agent_name

        def _get_connector(self, datasource):
            captured["datasource"] = datasource
            return _FakeConnector()

    import datus.tools.func_tool as func_tool_mod

    monkeypatch.setattr(func_tool_mod, "DBFuncTool", _FakeDBFuncTool)


# ---------------------------------------------------------------------------
# _load_local_template_pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_local_template_pair_reads_files_from_disk(tmp_path: Path):
    _write_dashboard(tmp_path)

    result = await _load_local_template_pair(tmp_path, "demo", "by_region")

    assert result.success is True
    sql_template, meta_text = result.data
    assert sql_template == _SAMPLE_SQL_J2
    assert json.loads(meta_text) == _SAMPLE_META


@pytest.mark.asyncio
async def test_load_local_template_pair_missing_returns_template_not_found(tmp_path: Path):
    _write_dashboard(tmp_path, with_template=False)

    result = await _load_local_template_pair(tmp_path, "demo", "missing")

    assert result.success is False
    assert result.errorCode == "TEMPLATE_NOT_FOUND"


@pytest.mark.asyncio
async def test_load_local_template_pair_rejects_invalid_dashboard_slug(tmp_path: Path):
    # Slug with traversal / invalid chars — fails the slug regex guard.
    result = await _load_local_template_pair(tmp_path, "../escape", "by_region")

    assert result.success is False
    assert result.errorCode == "INVALID_DASHBOARD_SLUG"


# ---------------------------------------------------------------------------
# DashboardService.get_detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_detail_returns_bundle_and_templates(tmp_path: Path):
    _write_dashboard(tmp_path)

    result = await DashboardService(agent_config=None).get_detail(
        project_files_root=tmp_path,
        dashboard_slug="demo",
    )

    assert result.success is True
    detail = result.data
    assert detail.slug == "demo"
    assert detail.name == "Demo Dashboard"
    assert detail.description == "Just a demo"

    # Flat files list includes the render entry and the queries pair —
    # manifest.json itself is intentionally absent (structured form lives
    # on ``manifest``).
    file_paths = {f.path for f in detail.files}
    assert "render/app.jsx" in file_paths
    assert "queries/by_region.sql.j2" in file_paths
    assert "queries/by_region.params.json" in file_paths
    assert "manifest.json" not in file_paths

    # The parsed templates sidecar carries the saved params/columns/datasource
    # so the outer-panel UI can drive filter affordances without re-parsing
    # the .params.json bytes from ``files``.
    assert len(detail.templates) == 1
    assert detail.templates[0].slug == "by_region"
    assert detail.templates[0].datasource == "warehouse"

    # Publication-side fields (subagent / dashboard_id / published_version /
    # published_at) are not part of the agent-side ``DashboardDetail``
    # schema — they live on Datus-backend's ``PublishedDashboardDetail``
    # subclass. The presence of any such attribute here would mean the
    # subclass leaked into agent code.
    assert not hasattr(detail, "subagent")
    assert not hasattr(detail, "dashboard_id")
    assert not hasattr(detail, "published_version")
    assert not hasattr(detail, "published_at")


@pytest.mark.asyncio
async def test_get_detail_rejects_invalid_slug(tmp_path: Path):
    result = await DashboardService(agent_config=None).get_detail(
        project_files_root=tmp_path,
        dashboard_slug="../escape",
    )

    assert result.success is False
    assert result.errorCode == "INVALID_DASHBOARD_SLUG"


@pytest.mark.asyncio
async def test_get_detail_missing_dashboard_returns_not_found(tmp_path: Path):
    result = await DashboardService(agent_config=None).get_detail(
        project_files_root=tmp_path,
        dashboard_slug="never_existed",
    )

    assert result.success is False
    assert result.errorCode == "DASHBOARD_NOT_FOUND"


@pytest.mark.asyncio
async def test_get_detail_missing_manifest_returns_not_found(tmp_path: Path):
    """``render/app.jsx`` exists but ``manifest.json`` is missing — the
    bundle is unrenderable and must surface a deterministic error."""
    dashboard_dir = tmp_path / "dashboards" / "demo"
    (dashboard_dir / "render").mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "render" / "app.jsx").write_text(_SAMPLE_APP_JSX, encoding="utf-8")

    result = await DashboardService(agent_config=None).get_detail(
        project_files_root=tmp_path,
        dashboard_slug="demo",
    )

    assert result.success is False
    assert result.errorCode == "DASHBOARD_NOT_FOUND"


# ---------------------------------------------------------------------------
# DashboardService.run_query — live-edit (no published_version)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_query_without_published_version_uses_local_template(monkeypatch, tmp_path: Path):
    """``published_version`` omitted → on-disk template feeds the render
    and the rendered SQL substitutes the supplied param.
    """
    _write_dashboard(tmp_path)
    captured: dict = {}
    _patch_executor(monkeypatch, captured=captured)

    result = await DashboardService(agent_config=MagicMock()).run_query(
        project_files_root=tmp_path,
        dashboard_slug="demo",
        query_slug="by_region",
        params={"region": "APAC"},
        published_version=None,
    )

    assert result.success is True
    assert result.data.row_count == 1
    assert result.data.datasource == "warehouse"
    # The rendered SQL substitutes the param — confirms we read the
    # on-disk ``.sql.j2`` and ran it through ``render_dashboard_template``.
    assert "APAC" in captured["sql"]
    # The agent service hands the canonical sub-agent name to ``DBFuncTool``
    # so the connector picks the same datasource binding the LLM saved.
    assert captured["sub_agent_name"] == "gen_visual_dashboard"
    assert captured["datasource"] == "warehouse"


@pytest.mark.asyncio
async def test_run_query_rejects_invalid_query_slug(tmp_path: Path):
    """Defence-in-depth: the slug regex guard fires before any I/O so a
    crafted slug can't reach the filesystem walker."""
    result = await DashboardService(agent_config=MagicMock()).run_query(
        project_files_root=tmp_path,
        dashboard_slug="demo",
        query_slug="../etc/passwd",
        params={},
    )

    assert result.success is False
    assert result.errorCode == "INVALID_QUERY_SLUG"


@pytest.mark.asyncio
async def test_run_query_rejects_non_dict_params(tmp_path: Path):
    """``params`` must be a JSON object so the param coercion step has
    something to walk."""
    result = await DashboardService(agent_config=MagicMock()).run_query(
        project_files_root=tmp_path,
        dashboard_slug="demo",
        query_slug="by_region",
        params=["not", "a", "dict"],  # type: ignore[arg-type]
    )

    assert result.success is False
    assert result.errorCode == "INVALID_PARAMS"


# ---------------------------------------------------------------------------
# DashboardService.run_query — published_version branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_query_published_version_without_loader_is_rejected(tmp_path: Path):
    """Agent-only deployments have no Postgres snapshot table; without an
    injected ``published_template_loader`` the published-version branch must
    refuse cleanly so callers don't silently fall through to the on-disk
    path with wrong semantics."""
    result = await DashboardService(agent_config=MagicMock()).run_query(
        project_files_root=tmp_path,
        dashboard_slug="demo",
        query_slug="by_region",
        params={"region": "APAC"},
        published_version=1,
        published_template_loader=None,
    )

    assert result.success is False
    assert result.errorCode == "INVALID_PUBLISHED_VERSION"


@pytest.mark.asyncio
async def test_run_query_published_version_below_one_is_rejected(tmp_path: Path):
    """Even with a loader wired, a non-positive ``published_version`` must
    fail validation before calling out."""

    async def _never_called(_version):  # pragma: no cover - guards a defence
        raise AssertionError("loader should not be called when version is invalid")

    result = await DashboardService(agent_config=MagicMock()).run_query(
        project_files_root=tmp_path,
        dashboard_slug="demo",
        query_slug="by_region",
        params={"region": "APAC"},
        published_version=0,
        published_template_loader=_never_called,
    )

    assert result.success is False
    assert result.errorCode == "INVALID_PUBLISHED_VERSION"


@pytest.mark.asyncio
async def test_run_query_published_version_uses_injected_loader(monkeypatch, tmp_path: Path):
    """When a loader is supplied, the on-disk tree is ignored — exercises
    the same seam the SaaS backend uses to feed
    ``visual_dashboard_versions`` snapshots into render+execute."""
    # Seed an on-disk dashboard with a sentinel SQL that would leak into
    # the rendered output if the on-disk path was hit by mistake.
    dashboard_dir = tmp_path / "dashboards" / "demo"
    (dashboard_dir / "queries").mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "queries" / "by_region.sql.j2").write_text(
        "SELECT 'LOCAL_LEAKED' AS sentinel;\n", encoding="utf-8"
    )
    (dashboard_dir / "queries" / "by_region.params.json").write_text(json.dumps(_SAMPLE_META), encoding="utf-8")

    loader_calls: list = []

    async def _loader(version: int):
        from datus.api.models.base_models import Result

        loader_calls.append(version)
        return Result(success=True, data=(_SAMPLE_SQL_J2, json.dumps(_SAMPLE_META)))

    captured: dict = {}
    _patch_executor(monkeypatch, captured=captured)

    result = await DashboardService(agent_config=MagicMock()).run_query(
        project_files_root=tmp_path,
        dashboard_slug="demo",
        query_slug="by_region",
        params={"region": "APAC"},
        published_version=2,
        published_template_loader=_loader,
    )

    assert result.success is True
    assert loader_calls == [2]
    # The on-disk sentinel must NOT appear — confirms the loader's output
    # won the source-selection.
    assert "LOCAL_LEAKED" not in captured["sql"]
    assert "APAC" in captured["sql"]
