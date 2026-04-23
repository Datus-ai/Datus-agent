# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Integration tests for ``/<service>.<method>`` CLI routing.

Scope = integration (end-to-end through ``AgentConfig`` → services config →
``ServiceClientRegistry`` → ``BIFuncTool._build_adapter`` → stub adapter →
Rich-rendered output).

Cost = CI (no external services, no network). The ``datus_bi_core`` module
is patched with an in-memory stub so ``BIFuncTool`` can construct a real
adapter without touching Superset/Grafana.

These cover wiring that unit tests intentionally mock out: agent.yml
services block → dashboard_config → registry discovery → lazy factory →
capability gating → schema-driven argument parsing → ``FuncToolResult``
rendering.
"""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from datus.cli.service_commands import ServiceCommands
from datus.configuration.agent_config import AgentConfig

# ---------------------------------------------------------------------------
# Stub datus_bi_core module — registered adapter class is instantiated by
# BIFuncTool._build_adapter via adapter_registry.get(platform).
# ---------------------------------------------------------------------------


class _MockDashboardWriteMixin:
    def create_dashboard(self, spec):
        return _Model(id=10, name=getattr(spec, "title", ""))

    def update_dashboard(self, dashboard_id, spec):
        return _Model(id=dashboard_id, name=getattr(spec, "title", ""))

    def delete_dashboard(self, dashboard_id):
        return True


class _MockChartWriteMixin:
    def create_chart(self, spec, dashboard_id=None):
        return _Model(id=1, name=getattr(spec, "title", ""))

    def update_chart(self, chart_id, spec):
        return _Model(id=chart_id, name=getattr(spec, "title", ""))

    def delete_chart(self, chart_id):
        return True

    def add_chart_to_dashboard(self, dashboard_id, chart_id):
        return True


class _MockDatasetWriteMixin:
    def create_dataset(self, spec):
        return _Model(id=1, name=getattr(spec, "name", ""))

    def delete_dataset(self, dataset_id):
        return True

    def list_bi_databases(self):
        return [{"id": 1, "name": "prod-pg"}]


class _Model:
    """Lightweight dict-backed object that mimics a Pydantic model."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return {k: v for k, v in self.__dict__.items()}


class _Page:
    """Mimics ``datus_bi_core.models.PaginatedResult``: ``.items`` + ``.total``."""

    def __init__(self, items, total=None):
        self.items = items
        self.total = len(items) if total is None else total


class _FullStubSupersetAdapter(_MockDashboardWriteMixin, _MockChartWriteMixin, _MockDatasetWriteMixin):
    """Stub that implements every capability — full feature surface."""

    def __init__(self, api_base_url, auth_params, dialect):
        self.api_base_url = api_base_url
        self.auth_params = auth_params
        self.dialect = dialect

    def list_dashboards(self, search="", limit=50, offset=0):
        items = [
            _Model(id=1, name="Finance Overview"),
            _Model(id=2, name="Sales Overview"),
        ]
        if search:
            items = [i for i in items if search.lower() in i.name.lower()]
        return _Page(items[offset : offset + limit], total=len(items))

    def get_dashboard_info(self, dashboard_id):
        if dashboard_id == "missing":
            return None
        return _Model(id=dashboard_id, name="Finance Overview", description="", chart_ids=[1, 2])

    def list_charts(self, dashboard_id, limit=50, offset=0):
        items = [_Model(id=1, name="Revenue Trend", chart_type="line")]
        return _Page(items[offset : offset + limit], total=len(items))

    def get_chart(self, chart_id, dashboard_id=None):
        if chart_id == "missing":
            return None
        return _Model(id=chart_id, name="Revenue Trend", chart_type="line")

    def list_datasets(self, dashboard_id="", limit=50, offset=0):
        items = [_Model(id=1, name="orders", dialect="postgresql")]
        return _Page(items[offset : offset + limit], total=len(items))

    def get_chart_data(self, chart_id, dashboard_id=None, limit=None):
        rows = [{"month": "2024-01", "revenue": 1000}, {"month": "2024-02", "revenue": 1500}]
        if limit is not None:
            rows = rows[:limit]
        return _Model(
            chart_id=chart_id,
            columns=["month", "revenue"],
            rows=rows,
            row_count=len(rows),
            sql="SELECT * FROM orders",
            extra={},
        )


class _ReadOnlyStubAdapter:
    """Stub with no write mixins and no advertised ``get_chart_data``."""

    def __init__(self, api_base_url, auth_params, dialect):
        self.api_base_url = api_base_url
        self.auth_params = auth_params
        self.dialect = dialect

    def list_dashboards(self, search="", limit=50, offset=0):
        items = [_Model(id=1, name="Read Only")]
        return _Page(items[offset : offset + limit], total=len(items))

    def get_dashboard_info(self, dashboard_id):
        return _Model(id=dashboard_id, name="Read Only", description="", chart_ids=[])

    def list_charts(self, dashboard_id, limit=50, offset=0):
        return _Page([], total=0)

    def get_chart(self, chart_id, dashboard_id=None):
        return None

    def list_datasets(self, dashboard_id="", limit=50, offset=0):
        return _Page([], total=0)


def _build_bi_core_mock(full_platform="superset", readonly_platform="grafana_ro") -> MagicMock:
    """Assemble the datus_bi_core module stub shared across tests."""
    mock = MagicMock()
    mock.DashboardWriteMixin = _MockDashboardWriteMixin
    mock.ChartWriteMixin = _MockChartWriteMixin
    mock.DatasetWriteMixin = _MockDatasetWriteMixin

    class _AuthParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    mock.AuthParam = _AuthParam

    class _Registry:
        def __init__(self):
            self._adapters = {
                full_platform: _FullStubSupersetAdapter,
                readonly_platform: _ReadOnlyStubAdapter,
            }

        def discover_adapters(self):
            pass

        def get(self, name):
            return self._adapters.get(name)

    mock.adapter_registry = _Registry()

    # ``BIAdapterBase`` is consulted by ``_supports_chart_data``; providing a
    # base class without ``get_chart_data`` forces ``isinstance`` / method
    # comparison to fall through the "override" path for adapters that DO
    # define the method.
    class _BIAdapterBase:
        pass

    mock.BIAdapterBase = _BIAdapterBase

    # ``datus_bi_core.models`` namespace for create_* methods.
    class _Spec:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    mock.models.ChartSpec = _Spec
    mock.models.DatasetSpec = _Spec
    mock.models.DashboardSpec = _Spec
    return mock


# ---------------------------------------------------------------------------
# CLI harness
# ---------------------------------------------------------------------------


class _FakeCLI:
    """Minimal DatusCLI stand-in: real agent_config + captured Rich console."""

    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.console = Console(file=io.StringIO(), no_color=True, width=200)
        self._bg_loop = None


def _configure_bi_services(agent_config, entries):
    """Register BI services on a real ``AgentConfig`` and re-run
    ``init_dashboard`` so ``dashboard_config`` reflects the mutation."""
    agent_config.services.bi_tools = dict(entries)
    agent_config.init_dashboard(agent_config.services.bi_tools)


@pytest.fixture
def bi_core_stub():
    """Patch ``datus_bi_core`` in ``sys.modules`` for the duration of a test."""
    mock = _build_bi_core_mock()
    with pytest.MonkeyPatch().context() as m:
        m.setitem(sys.modules, "datus_bi_core", mock)
        m.setitem(sys.modules, "datus_bi_core.models", mock.models)
        yield mock


def _build_agent_config(tmp_path, bi_tools):
    """Construct a minimal real ``AgentConfig`` scoped to ``tmp_path``.

    The BI-dispatch path only touches ``services.bi_tools`` and the derived
    ``dashboard_config``; all other sections (models, databases,
    agentic_nodes, storage) are left at sensible defaults so we do not pay
    the cost of the heavier ``real_agent_config`` fixture in
    ``tests/unit_tests/conftest.py``.
    """
    os.makedirs(os.path.join(str(tmp_path), "workspace"), exist_ok=True)
    return AgentConfig(
        nodes={},
        home=str(tmp_path),
        target="mock",
        models={
            "mock": {
                "type": "openai",
                "api_key": "mock-api-key",
                "model": "mock-model",
                "base_url": "http://localhost:0",
            },
        },
        services={
            "databases": {},
            "semantic_layer": {},
            "bi_tools": dict(bi_tools),
            "schedulers": {},
        },
        project_root=str(tmp_path / "workspace"),
        storage={},
        agentic_nodes={},
    )


@pytest.fixture
def cli_with_superset(tmp_path, bi_core_stub):
    """Real AgentConfig + Superset BI service + fake CLI harness."""
    agent_config = _build_agent_config(
        tmp_path,
        bi_tools={
            "superset": {
                "api_base_url": "http://superset.test/api",
                "username": "admin",
                "password": "admin",
                "type": "superset",
            },
        },
    )
    return _FakeCLI(agent_config)


@pytest.fixture
def cli_with_two_bi_services(tmp_path, bi_core_stub):
    agent_config = _build_agent_config(
        tmp_path,
        bi_tools={
            "superset": {
                "api_base_url": "http://superset.test/api",
                "username": "admin",
                "password": "admin",
                "type": "superset",
            },
            "grafana_ro": {
                "api_base_url": "http://grafana.test/api",
                "api_key": "token",
                "type": "grafana_ro",
            },
        },
    )
    return _FakeCLI(agent_config)


@pytest.fixture
def cli_without_services(tmp_path, bi_core_stub):
    agent_config = _build_agent_config(tmp_path, bi_tools={})
    return _FakeCLI(agent_config)


def _output(cli) -> str:
    return cli.console.file.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServicesListing:
    def test_services_lists_configured_entries(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.cmd_services("")
        out = _output(cli_with_superset)
        assert "superset" in out
        # Type column shows the human-readable label, not the raw agent.yml key.
        assert "BI platform" in out

    def test_multi_service_both_listed(self, cli_with_two_bi_services):
        cmd = ServiceCommands(cli_with_two_bi_services)
        cmd.cmd_services("")
        out = _output(cli_with_two_bi_services)
        assert "superset" in out
        assert "grafana_ro" in out

    def test_empty_services_prints_hint(self, cli_without_services):
        # No bi_tools / schedulers / semantic_layer configured.
        cmd = ServiceCommands(cli_without_services)
        cmd.cmd_services("")
        out = _output(cli_without_services)
        assert "No services configured" in out


class TestMethodListing:
    def test_full_adapter_lists_read_methods_only(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        handled = cmd.dispatch("/superset", "")
        assert handled is True
        out = _output(cli_with_superset)
        # Read methods advertised.
        for expected in ("list_dashboards", "get_dashboard", "list_charts", "get_chart", "list_datasets"):
            assert expected in out
        # get_chart_data advertised (full adapter overrides base).
        assert "get_chart_data" in out
        # list_bi_databases advertised because DatasetWriteMixin is present.
        assert "list_bi_databases" in out
        # No write methods leak into the listing.
        for blocked in ("create_dashboard", "create_chart", "delete_dashboard", "write_query"):
            assert blocked not in out

    def test_readonly_adapter_hides_gated_methods(self, cli_with_two_bi_services):
        cmd = ServiceCommands(cli_with_two_bi_services)
        cmd.dispatch("/grafana_ro", "")
        out = _output(cli_with_two_bi_services)
        assert "list_dashboards" in out
        assert "get_dashboard" in out
        # ReadOnlyStubAdapter has no DatasetWriteMixin → list_bi_databases must
        # not appear, and no get_chart_data method is defined on it.
        assert "get_chart_data" not in out
        assert "list_bi_databases" not in out


class TestInvocation:
    def test_list_dashboards_end_to_end(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.list_dashboards", "")
        out = _output(cli_with_superset)
        assert "Finance Overview" in out
        assert "Sales Overview" in out

    def test_list_dashboards_with_named_filter(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.list_dashboards", "--search=Sales")
        out = _output(cli_with_superset)
        assert "Sales Overview" in out
        assert "Finance Overview" not in out

    def test_get_dashboard_positional(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.get_dashboard", "1")
        out = _output(cli_with_superset)
        # Single-dict payloads render as a Field/Value table; both the
        # returned id and name should appear as cell contents.
        assert "Finance Overview" in out
        assert "1" in out

    def test_get_dashboard_missing_id_shows_schema(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.get_dashboard", "")
        out = _output(cli_with_superset)
        assert "Missing required argument" in out
        assert "dashboard_id" in out

    def test_help_flag_renders_schema(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.get_chart_data", "--help")
        out = _output(cli_with_superset)
        assert "parameters" in out.lower()
        assert "chart_id" in out

    def test_tool_error_rendered(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        # Adapter returns None → FuncToolResult.success=0 with "not found".
        cmd.dispatch("/superset.get_dashboard", "missing")
        out = _output(cli_with_superset)
        assert "not found" in out.lower()

    def test_int_coercion_for_limit(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.get_chart_data", "42 --limit=1")
        out = _output(cli_with_superset)
        # Single-dict payload renders as a K/V table. Limit=1 → one row
        # returned; ``row_count`` and ``1`` both appear as cell contents.
        assert "row_count" in out
        # Ensure the count really is 1, not 2 (the stub returns 2 without limit).
        assert "2024-02" not in out


class TestWriteBlockedAndUnknown:
    def test_write_method_is_blocked(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.create_dashboard", "--title=new")
        out = _output(cli_with_superset).lower()
        assert "write" in out or "read-only" in out or "privileged" in out

    def test_unknown_method_prints_hint(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        cmd.dispatch("/superset.no_such", "")
        out = _output(cli_with_superset)
        assert "Unknown method" in out or "no_such" in out


class TestMultiInstanceSamePlatform:
    """Two BI services pointing at the same adapter (``superset``) with
    distinct CLI aliases.

    Before the ``adapter_type`` split, ``init_dashboard`` required the
    YAML key to equal the ``type`` field — which meant two entries
    named ``superset_a`` / ``superset_b`` with ``type: superset`` were
    rejected at config load, and omitting ``type`` made the adapter
    lookup use the alias (``superset_a``) and fail. This test pins the
    now-supported multi-instance shape end-to-end.
    """

    @pytest.fixture
    def cli_with_two_supersets(self, tmp_path, bi_core_stub):
        agent_config = _build_agent_config(
            tmp_path,
            bi_tools={
                "superset_prod": {
                    "api_base_url": "http://prod.test/api",
                    "username": "admin",
                    "password": "admin",
                    "type": "superset",
                },
                "superset_staging": {
                    "api_base_url": "http://staging.test/api",
                    "username": "admin",
                    "password": "admin",
                    "type": "superset",
                },
            },
        )
        return _FakeCLI(agent_config)

    def test_services_lists_both_with_configured_status(self, cli_with_two_supersets):
        cmd = ServiceCommands(cli_with_two_supersets)
        cmd.cmd_services("")
        out = _output(cli_with_two_supersets)
        assert "superset_prod" in out
        assert "superset_staging" in out
        # Both should show as "configured" — adapter is the same registered
        # "superset" class, which the stub declares available.
        assert "missing adapter" not in out

    def test_both_aliases_route_to_same_adapter_class(self, cli_with_two_supersets):
        cmd = ServiceCommands(cli_with_two_supersets)

        cmd.dispatch("/superset_prod.list_dashboards", "")
        out = _output(cli_with_two_supersets)
        assert "Finance Overview" in out

        cli_with_two_supersets.console.file.seek(0)
        cli_with_two_supersets.console.file.truncate()

        cmd.dispatch("/superset_staging.list_dashboards", "")
        out = _output(cli_with_two_supersets)
        assert "Finance Overview" in out  # same stub data, no crosstalk error


class TestMissingAdapterEndToEnd:
    """Service configured in agent.yml, but the adapter package isn't installed.

    Without the probe / preflight, the failure surfaces as a cryptic
    ``DatusException`` from ``BIFuncTool._build_adapter`` at the first
    actual invocation. With preflight, ``.services`` explicitly reports
    ``missing adapter`` and ``.<service>.<method>`` prints an install hint.
    """

    @pytest.fixture
    def cli_with_unregistered_platform(self, tmp_path, bi_core_stub):
        # bi_core_stub registers 'superset' and 'grafana_ro'. We intentionally
        # configure a platform name NOT registered in the stub so the probe
        # fails the way it would on a real machine missing ``datus-bi-<x>``.
        agent_config = _build_agent_config(
            tmp_path,
            bi_tools={"tableau": {"api_base_url": "http://tableau.test", "type": "tableau"}},
        )
        return _FakeCLI(agent_config)

    def test_services_lists_missing_adapter_status(self, cli_with_unregistered_platform):
        cmd = ServiceCommands(cli_with_unregistered_platform)
        cmd.cmd_services("")
        out = _output(cli_with_unregistered_platform)
        assert "tableau" in out
        assert "missing adapter" in out

    def test_invocation_prints_install_hint(self, cli_with_unregistered_platform):
        cmd = ServiceCommands(cli_with_unregistered_platform)
        cmd.dispatch("/tableau.list_dashboards", "")
        out = _output(cli_with_unregistered_platform)
        assert "not installed" in out
        assert "datus-bi-" in out

    def test_bare_service_prints_install_hint(self, cli_with_unregistered_platform):
        cmd = ServiceCommands(cli_with_unregistered_platform)
        cmd.dispatch("/tableau", "")
        out = _output(cli_with_unregistered_platform)
        assert "not installed" in out
        # Method table is suppressed.
        assert "read methods" not in out


class TestMultiServiceRouting:
    def test_two_services_invoked_independently(self, cli_with_two_bi_services):
        cmd = ServiceCommands(cli_with_two_bi_services)

        cmd.dispatch("/superset.list_dashboards", "")
        out_a = _output(cli_with_two_bi_services)
        assert "Finance Overview" in out_a

        # Clear captured output.
        cli_with_two_bi_services.console.file.seek(0)
        cli_with_two_bi_services.console.file.truncate()

        cmd.dispatch("/grafana_ro.list_dashboards", "")
        out_b = _output(cli_with_two_bi_services)
        assert "Read Only" in out_b
        # The two adapters respond with different data — no cross-talk.
        assert "Finance Overview" not in out_b

    def test_unknown_service_not_handled(self, cli_with_superset):
        cmd = ServiceCommands(cli_with_superset)
        handled = cmd.dispatch("/mystery.foo", "")
        assert handled is False
