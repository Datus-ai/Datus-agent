# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.service_client``: registry + allow-list."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from datus.cli.service_client import READ_METHODS, ServiceClient, ServiceClientRegistry


def _fake_agent_config(bi_platforms=None, schedulers=None, semantic_layer=None):
    """Build a minimal ``agent_config``-like object for registry tests."""
    return SimpleNamespace(
        services=SimpleNamespace(
            bi_platforms=bi_platforms or {},
            schedulers=schedulers or {},
            semantic_layer=semantic_layer or {},
        ),
    )


class _FakeBITool:
    """Stand-in for BIFuncTool that doesn't touch the adapter registry."""

    def list_dashboards(self, search: str = ""):
        """List BI dashboards."""
        return {"success": 1, "result": [{"id": 1, "search": search}]}

    def create_dashboard(self, title: str, description: str = ""):
        """Write method — should be blocked by allow-list."""
        return {"success": 1, "result": {"id": 99, "title": title}}

    def get_dashboard(self, dashboard_id: str):
        """Get dashboard by id."""
        return {"success": 1, "result": {"id": dashboard_id}}


class TestReadMethodsAllowList:
    def test_allowlist_covers_three_service_types(self):
        assert set(READ_METHODS.keys()) == {"bi_platforms", "schedulers", "semantic_layer"}

    def test_no_write_methods_leak(self):
        write_prefixes = ("create_", "update_", "delete_", "write_", "submit_", "add_")
        for service_type, methods in READ_METHODS.items():
            for method in methods:
                for prefix in write_prefixes:
                    assert not method.startswith(prefix), f"{service_type}.{method} starts with write prefix '{prefix}'"


class TestServiceClient:
    def _make_client(self, tool=None):
        return ServiceClient(
            service_type="bi_platforms",
            service_name="superset",
            tool_instance=tool or _FakeBITool(),
            method_names=READ_METHODS["bi_platforms"],
        )

    def test_list_methods_sorted_with_docs(self):
        client = self._make_client()
        methods = client.list_methods()
        names = [m[0] for m in methods]
        # Only allow-listed methods that actually exist on the instance.
        assert "list_dashboards" in names
        assert "get_dashboard" in names
        # Allow-listed but missing on instance → excluded.
        assert "list_charts" not in names
        # Write method never appears even though it exists on the tool.
        assert "create_dashboard" not in names
        # Sorted alphabetically.
        assert names == sorted(names)
        # Doc is the first non-empty line of the docstring.
        list_dash_doc = dict(methods)["list_dashboards"]
        assert list_dash_doc == "List BI dashboards."

    def test_get_tool_blocks_write_methods(self):
        client = self._make_client()
        # A tool not in READ_METHODS is blocked.
        assert client.get_tool("create_dashboard") is None

    def test_get_tool_blocks_missing_methods(self):
        client = self._make_client()
        # Allow-listed but not implemented on this tool → None.
        assert client.get_tool("list_charts") is None

    def test_get_tool_wraps_and_caches(self):
        client = self._make_client()
        tool_a = client.get_tool("list_dashboards")
        tool_b = client.get_tool("list_dashboards")
        assert tool_a is not None
        assert tool_a is tool_b  # cached

    def test_has_method(self):
        client = self._make_client()
        assert client.has_method("list_dashboards") is True
        # write method → not allow-listed
        assert client.has_method("create_dashboard") is False
        # not in allow-list → False even though method exists
        assert client.has_method("get_dashboard") is True


class _AvailableToolsFilteredTool:
    """Stub that mimics BIFuncTool's capability-gated ``available_tools()``."""

    def __init__(self, advertised):
        self._advertised = advertised

    def list_dashboards(self):
        """List."""

    def get_dashboard(self):
        """Get."""

    def get_chart_data(self):
        """Chart data — may be hidden."""

    def list_bi_databases(self):
        """Databases — may be hidden."""

    def available_tools(self):
        return [SimpleNamespace(name=n) for n in self._advertised]


class TestServiceClientAvailableToolsFilter:
    """Even when a method is both allow-listed and present on the instance,
    it must also be advertised via ``available_tools()`` to appear."""

    def test_advertised_subset_exposes_only_intersection(self):
        tool = _AvailableToolsFilteredTool(advertised={"list_dashboards", "get_dashboard"})
        client = ServiceClient(
            service_type="bi_platforms",
            service_name="superset_ro",
            tool_instance=tool,
            method_names=READ_METHODS["bi_platforms"],
        )
        names = [m[0] for m in client.list_methods()]
        assert names == ["get_dashboard", "list_dashboards"]
        # get_chart_data / list_bi_databases exist on the instance but aren't
        # advertised (read-only BI adapter) → must not be callable.
        assert client.has_method("get_chart_data") is False
        assert client.get_tool("get_chart_data") is None
        assert client.get_tool("list_bi_databases") is None

    def test_fallback_when_no_available_tools(self):
        """Tools without ``available_tools`` fall back to allow-list ∩ hasattr."""

        class _Plain:
            def list_dashboards(self):
                """."""

        client = ServiceClient(
            service_type="bi_platforms",
            service_name="plain",
            tool_instance=_Plain(),
            method_names=READ_METHODS["bi_platforms"],
        )
        names = [m[0] for m in client.list_methods()]
        assert names == ["list_dashboards"]
        assert client.get_tool("list_dashboards") is not None

    def test_available_tools_exception_falls_back(self):
        class _Broken:
            def list_dashboards(self):
                """."""

            def available_tools(self):
                raise RuntimeError("bootstrap failure")

        client = ServiceClient(
            service_type="bi_platforms",
            service_name="broken",
            tool_instance=_Broken(),
            method_names=READ_METHODS["bi_platforms"],
        )
        names = [m[0] for m in client.list_methods()]
        # Falls back to allow-list ∩ hasattr — list_dashboards is present.
        assert names == ["list_dashboards"]


class _FakeSchedulerTool:
    def list_scheduler_jobs(self):
        """List scheduled jobs."""
        return {"success": 1, "result": []}


class _FakeSemanticTool:
    def list_metrics(self):
        """List available metrics."""
        return {"success": 1, "result": []}


class TestServiceClientRegistry:
    def test_empty_config_yields_no_services(self):
        registry = ServiceClientRegistry(_fake_agent_config())
        assert registry.list_services() == []
        assert registry.get("nothing") is None
        assert registry.has("nothing") is False

    def test_discover_lists_all_service_types(self):
        cfg = _fake_agent_config(
            bi_platforms={"superset": {"api_base_url": "x"}, "grafana": {"api_base_url": "y"}},
            schedulers={"airflow": {"type": "airflow"}},
            semantic_layer={"metricflow": {"datasource": "x"}},
        )
        always_available = {k: (lambda c, n: True) for k in ("bi_platforms", "schedulers", "semantic_layer")}
        with patch.dict("datus.cli.service_client._PROBES", always_available):
            registry = ServiceClientRegistry(cfg)
            rows = registry.list_services()
        names = {r[0] for r in rows}
        types = {r[0]: r[1] for r in rows}
        assert names == {"superset", "grafana", "airflow", "metricflow"}
        assert types["superset"] == "bi_platforms"
        assert types["airflow"] == "schedulers"
        assert types["metricflow"] == "semantic_layer"
        # All "configured" at list time — adapter available, client not yet built.
        assert all(r[2] == "configured" for r in rows)

    def test_lazy_construction_then_active_status(self):
        cfg = _fake_agent_config(bi_platforms={"superset": {}})
        factory = MagicMock(return_value=_FakeBITool())
        with (
            patch.dict("datus.cli.service_client._FACTORIES", {"bi_platforms": factory}),
            patch.dict("datus.cli.service_client._PROBES", {"bi_platforms": lambda c, n: True}),
        ):
            registry = ServiceClientRegistry(cfg)
            # Not built yet.
            factory.assert_not_called()
            # Before any get(), status is "configured" (adapter available, not built).
            assert registry.list_services()[0][2] == "configured"
            client = registry.get("superset")
            assert client is not None
            factory.assert_called_once()
            # Status flips to "active" once the client is cached.
            assert registry.list_services()[0][2] == "active"
            # Second get returns cached instance.
            assert registry.get("superset") is client
            factory.assert_called_once()

    def test_case_insensitive_lookup(self):
        cfg = _fake_agent_config(bi_platforms={"Superset": {}})
        factory = MagicMock(return_value=_FakeBITool())
        with patch.dict("datus.cli.service_client._FACTORIES", {"bi_platforms": factory}):
            registry = ServiceClientRegistry(cfg)
            assert registry.has("superset") is True
            assert registry.has("SUPERSET") is True
            client = registry.get("SUPERSET")
            assert client is not None
            # Original-case name preserved for display.
            assert client.service_name == "Superset"

    def test_factory_error_returns_none(self):
        cfg = _fake_agent_config(bi_platforms={"broken": {}})
        factory = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict("datus.cli.service_client._FACTORIES", {"bi_platforms": factory}):
            registry = ServiceClientRegistry(cfg)
            assert registry.get("broken") is None
            # Still listed — the name is configured; it just fails to build.
            assert registry.list_services()[0][0] == "broken"

    def test_scheduler_factory_wired(self):
        cfg = _fake_agent_config(schedulers={"airflow": {"type": "airflow"}})
        factory = MagicMock(return_value=_FakeSchedulerTool())
        with patch.dict("datus.cli.service_client._FACTORIES", {"schedulers": factory}):
            registry = ServiceClientRegistry(cfg)
            client = registry.get("airflow")
            assert client is not None
            assert client.service_type == "schedulers"
            factory.assert_called_once_with(cfg, "airflow")

    def test_semantic_factory_wired(self):
        cfg = _fake_agent_config(semantic_layer={"metricflow": {}})
        factory = MagicMock(return_value=_FakeSemanticTool())
        with patch.dict("datus.cli.service_client._FACTORIES", {"semantic_layer": factory}):
            registry = ServiceClientRegistry(cfg)
            client = registry.get("metricflow")
            assert client is not None
            assert client.service_type == "semantic_layer"
            factory.assert_called_once_with(cfg, "metricflow")

    def test_none_services_attribute_tolerated(self):
        """Agent configs that omit ``services`` entirely should not crash."""
        cfg = SimpleNamespace()  # no `services`
        registry = ServiceClientRegistry(cfg)
        assert registry.list_services() == []

    def test_cache_invalidated_on_datasource_switch(self):
        """After ``.database`` / ``.datasource`` switch, cached ``ServiceClient``s
        must be dropped — ``SemanticTools`` / ``BIFuncTool`` internalise the
        active datasource at construction time (MetricRAG, read_connector,
        adapter config resolution) and continuing to reuse them would run
        queries against the old datasource.
        """
        cfg = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={"superset": {}}, schedulers={}, semantic_layer={}),
            current_datasource="datasource_a",
        )
        factory = MagicMock(side_effect=lambda *_: _FakeBITool())
        with (
            patch.dict("datus.cli.service_client._FACTORIES", {"bi_platforms": factory}),
            patch.dict("datus.cli.service_client._PROBES", {"bi_platforms": lambda c, n: True}),
        ):
            registry = ServiceClientRegistry(cfg)
            c1 = registry.get("superset")
            assert c1 is not None
            factory.assert_called_once()

            # Same datasource → cached instance reused.
            assert registry.get("superset") is c1
            factory.assert_called_once()

            # User runs ``.database datasource_b`` — current_datasource mutates.
            cfg.current_datasource = "datasource_b"
            c2 = registry.get("superset")
            assert c2 is not c1
            assert factory.call_count == 2
            # list_services reflects the rebuild with the fresh client cached.
            statuses = {name: status for name, _type, status in registry.list_services()}
            assert statuses["superset"] == "active"

    def test_datasource_field_also_triggers_invalidation(self):
        """The ``datasource`` attribute is also part of the fingerprint."""
        cfg = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={"superset": {}}, schedulers={}, semantic_layer={}),
            current_datasource="shared",
            datasource="tenant_a",
        )
        factory = MagicMock(side_effect=lambda *_: _FakeBITool())
        with (
            patch.dict("datus.cli.service_client._FACTORIES", {"bi_platforms": factory}),
            patch.dict("datus.cli.service_client._PROBES", {"bi_platforms": lambda c, n: True}),
        ):
            registry = ServiceClientRegistry(cfg)
            registry.get("superset")
            cfg.datasource = "tenant_b"
            registry.get("superset")
            assert factory.call_count == 2

    def test_list_services_drops_to_configured_after_invalidation(self):
        cfg = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={"superset": {}}, schedulers={}, semantic_layer={}),
            current_datasource="a",
        )
        with (
            patch.dict(
                "datus.cli.service_client._FACTORIES",
                {"bi_platforms": MagicMock(side_effect=lambda *_: _FakeBITool())},
            ),
            patch.dict("datus.cli.service_client._PROBES", {"bi_platforms": lambda c, n: True}),
        ):
            registry = ServiceClientRegistry(cfg)
            registry.get("superset")
            assert registry.list_services()[0][2] == "active"

            # Datasource switch without a follow-up get() — status drops back to
            # "configured" (adapter still available, client cache cleared).
            cfg.current_datasource = "b"
            # list_services itself triggers invalidation.
            assert registry.list_services()[0][2] == "configured"

    def test_missing_adapter_status_beats_active_even_with_cached_client(self):
        """A cached ServiceClient does not imply the adapter is usable.

        ``dispatch`` has to build the wrapper to run the preflight probe,
        which puts the client in ``_clients``. Without the fix,
        ``.services`` would then report the service as ``active`` even
        though any invocation would fail with "adapter not installed".
        """
        cfg = _fake_agent_config(bi_platforms={"broken": {}})
        with patch.dict("datus.cli.service_client._PROBES", {"bi_platforms": lambda c, n: False}):
            registry = ServiceClientRegistry(cfg)
            # Simulate a prior dispatch that built the lightweight wrapper.
            registry._clients["broken"] = ServiceClient(
                service_type="bi_platforms",
                service_name="broken",
                tool_instance=_FakeBITool(),
                method_names=READ_METHODS["bi_platforms"],
            )
            rows = registry.list_services()
        assert rows[0] == ("broken", "bi_platforms", "missing adapter")

    def test_missing_adapter_status_when_probe_fails(self):
        """Probe failing → status is 'missing adapter'."""
        cfg = _fake_agent_config(bi_platforms={"superset": {}})
        with patch.dict("datus.cli.service_client._PROBES", {"bi_platforms": lambda c, n: False}):
            registry = ServiceClientRegistry(cfg)
            rows = registry.list_services()
        assert rows[0] == ("superset", "bi_platforms", "missing adapter")

    def test_adapter_available_is_cached_until_fingerprint_change(self):
        cfg = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={"superset": {}}, schedulers={}, semantic_layer={}),
            current_datasource="a",
        )
        probe = MagicMock(return_value=True)
        with patch.dict("datus.cli.service_client._PROBES", {"bi_platforms": probe}):
            registry = ServiceClientRegistry(cfg)
            assert registry.adapter_available("superset") is True
            assert registry.adapter_available("superset") is True
            # Cached → called once.
            assert probe.call_count == 1

            # Datasource switch → cache dropped → probe runs again.
            cfg.current_datasource = "b"
            assert registry.adapter_available("superset") is True
            assert probe.call_count == 2

    def test_adapter_available_false_for_unknown_service(self):
        registry = ServiceClientRegistry(_fake_agent_config())
        assert registry.adapter_available("not-configured") is False


class TestAdapterProbes:
    """Directly exercise the per-section adapter probes."""

    def test_bi_probe_returns_false_when_datus_bi_core_missing(self):
        from datus.cli.service_client import _probe_bi_adapter

        # Simulate missing package via sys.modules.
        with patch.dict("sys.modules", {"datus_bi_core": None}):
            assert _probe_bi_adapter(None, "superset") is False

    def test_bi_probe_returns_true_when_platform_registered(self):
        from datus.cli.service_client import _probe_bi_adapter

        stub_module = MagicMock()
        stub_module.adapter_registry.get.return_value = object()
        stub_module.adapter_registry.discover_adapters = MagicMock()
        with patch.dict("sys.modules", {"datus_bi_core": stub_module}):
            assert _probe_bi_adapter(None, "superset") is True

    def test_bi_probe_uses_adapter_type_from_dashboard_config(self):
        """Multi-instance: alias ``superset_prod`` with ``type: superset``
        must probe the ``superset`` adapter — not the alias."""
        from datus.cli.service_client import _probe_bi_adapter

        stub_module = MagicMock()
        stub_module.adapter_registry.get.return_value = object()
        stub_module.adapter_registry.discover_adapters = MagicMock()

        dash_cfg = SimpleNamespace(adapter_type="superset")
        agent_config = SimpleNamespace(dashboard_config={"superset_prod": dash_cfg})

        with patch.dict("sys.modules", {"datus_bi_core": stub_module}):
            assert _probe_bi_adapter(agent_config, "superset_prod") is True
            stub_module.adapter_registry.get.assert_called_with("superset")

    def test_bi_probe_returns_false_when_platform_unknown(self):
        from datus.cli.service_client import _probe_bi_adapter

        stub_module = MagicMock()
        stub_module.adapter_registry.get.return_value = None
        stub_module.adapter_registry.discover_adapters = MagicMock()
        with patch.dict("sys.modules", {"datus_bi_core": stub_module}):
            assert _probe_bi_adapter(None, "superset") is False

    def test_scheduler_probe_checks_platform_registration(self):
        """Platform adapter missing → probe returns False.

        ``datus-scheduler-core`` is a hard dep and always importable; the
        real-world gap is the platform-specific package (e.g.
        ``datus-scheduler-airflow``) that registers the adapter into
        ``SchedulerAdapterRegistry``.
        """
        from datus.cli.service_client import _probe_scheduler_adapter

        mock_registry = MagicMock()
        mock_registry.get_adapter_class.return_value = None
        mock_registry.has_adapter.return_value = None
        mock_registry.get.return_value = None
        cfg = SimpleNamespace(
            get_scheduler_config=lambda name: {"type": "airflow"},
        )
        with patch("datus.cli.service_client.SchedulerAdapterRegistry", mock_registry):
            assert _probe_scheduler_adapter(cfg, "airflow_local") is False

    def test_scheduler_probe_returns_true_when_platform_registered(self):
        from datus.cli.service_client import _probe_scheduler_adapter

        mock_registry = MagicMock()
        mock_registry.get_adapter_class.return_value = object()
        cfg = SimpleNamespace(
            get_scheduler_config=lambda name: {"type": "airflow"},
        )
        with patch("datus.cli.service_client.SchedulerAdapterRegistry", mock_registry):
            assert _probe_scheduler_adapter(cfg, "airflow_local") is True
            # The getter was asked about the platform from the config, not the
            # service name (which is only a user-chosen alias).
            mock_registry.get_adapter_class.assert_called_with("airflow")

    def test_probe_helper_defensive_on_exception(self):
        from datus.cli.service_client import _probe

        with patch.dict(
            "datus.cli.service_client._PROBES",
            {"bi_platforms": MagicMock(side_effect=RuntimeError("boom"))},
        ):
            assert _probe(None, "bi_platforms", "x") is False


class TestRegistryFactoriesWired:
    def test_registry_uses_expected_factories(self):
        """Registry honors the mocked factory for each service type."""
        cfg = _fake_agent_config(
            bi_platforms={"s1": {}},
            schedulers={"s2": {}},
            semantic_layer={"s3": {}},
        )
        bi_factory = MagicMock(return_value=MagicMock(spec=[]))
        sched_factory = MagicMock(return_value=MagicMock(spec=[]))
        sem_factory = MagicMock(return_value=MagicMock(spec=[]))
        with patch.dict(
            "datus.cli.service_client._FACTORIES",
            {"bi_platforms": bi_factory, "schedulers": sched_factory, "semantic_layer": sem_factory},
        ):
            registry = ServiceClientRegistry(cfg)
            registry.get("s1")
            registry.get("s2")
            registry.get("s3")
            bi_factory.assert_called_once()
            sched_factory.assert_called_once()
            sem_factory.assert_called_once()
