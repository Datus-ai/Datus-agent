# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_bi_picker`.

We mock both :class:`BootstrapBiApp` (so we don't drive a real prompt_toolkit
event loop) and the BI adapter (so we don't touch network IO). The picker's
state machine is exercised by feeding it a sequence of canned ``app.run``
return values and asserting the expected adapter / assembler interaction.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.bootstrap_bi_app import (
    BootstrapBiResult,
    BootstrapBiSelection,
    PendingAssemble,
    PendingFetch,
    ServiceEntry,
)
from datus.cli.bootstrap_bi_picker import BootstrapBiPicker

# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120, log_path=False)


def _agent_config(*, dashboard_config: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        db_type="duckdb",
        current_datasource="local",
        dashboard_config=dashboard_config or {},
        current_db_config=lambda *_a, **_k: None,
    )


def _service_entry() -> ServiceEntry:
    return ServiceEntry(name="superset_local", adapter_type="superset", api_base_url="http://localhost:8088")


# ─────────────────────────────────────────────────────────────────
# adapter discovery / cancellation paths
# ─────────────────────────────────────────────────────────────────


def test_run_raises_when_no_adapters_installed(console) -> None:
    """No adapter implementations on the path → fast path with a clear error."""
    picker = BootstrapBiPicker(_agent_config(), console)
    picker._adapter_registry = {}  # simulate no adapters discovered
    with pytest.raises(ValueError, match="No BI adapter implementations"):
        picker.run()


def test_run_raises_when_no_services_configured(console) -> None:
    picker = BootstrapBiPicker(_agent_config(dashboard_config={}), console)
    picker._adapter_registry = {"superset": MagicMock()}
    with pytest.raises(ValueError, match="No BI platforms configured"):
        picker.run()


def test_run_returns_none_when_user_cancels_at_service_view(console) -> None:
    picker = BootstrapBiPicker(_agent_config(), console)
    picker._adapter_registry = {"superset": MagicMock()}

    with patch("datus.cli.bootstrap_bi_picker.build_service_entries", return_value=[_service_entry()]):
        with patch.object(BootstrapBiPicker, "_run_app", return_value=None):
            assert picker.run() is None


# ─────────────────────────────────────────────────────────────────
# happy path: 4 stages with a fully-mocked adapter
# ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.list_dashboards.return_value = SimpleNamespace(
        items=[SimpleNamespace(id=1, name="Sales", url="http://localhost/d/1")]
    )
    adapter.get_dashboard_info.return_value = SimpleNamespace(
        id=1,
        name="Sales",
        description="Quarterly numbers",
    )
    adapter.list_charts.return_value = [
        SimpleNamespace(
            id=10,
            name="orders",
            title=None,
            query=SimpleNamespace(sql=["SELECT COUNT(*) FROM orders"]),
        ),
    ]
    adapter.get_chart.side_effect = lambda chart_id, dashboard_id: SimpleNamespace(
        id=chart_id,
        name=f"chart-{chart_id}",
        title=None,
        query=SimpleNamespace(sql=["SELECT COUNT(*) FROM orders"]),
    )
    adapter.list_datasets.return_value = []
    adapter.parse_dashboard_id.return_value = 99
    return adapter


def test_run_drives_four_stages_and_returns_plan(console, fake_adapter) -> None:
    auth = SimpleNamespace(name="superset", auth_type="LOGIN")
    picker = BootstrapBiPicker(
        _agent_config(dashboard_config={"superset": SimpleNamespace()}),
        console,
    )
    picker._adapter_registry = {"superset": MagicMock(return_value=fake_adapter)}

    # Returns from each ``app.run`` call.
    svc = _service_entry()
    run_returns = [
        PendingFetch(service=svc),
        BootstrapBiSelection(
            service=svc,
            dashboard_id=1,
            dashboard_name="Sales",
            dashboard_url="http://localhost/d/1",
            is_manual_url=False,
        ),
        PendingAssemble(chart_ref_indices=[0], chart_metrics_indices=[0]),
        BootstrapBiResult(
            service=svc,
            dashboard_id=1,
            dashboard_name="Sales",
            dashboard_url="http://localhost/d/1",
            is_manual_url=False,
            chart_ref_indices=[0],
            chart_metrics_indices=[0],
            table_indices=[0],
            pool_size=3,
        ),
    ]

    assembled = SimpleNamespace(tables=["orders"], reference_sqls=[], metric_sqls=[])
    with (
        patch("datus.cli.bootstrap_bi_picker.build_service_entries", return_value=[_service_entry()]),
        patch("datus.cli.bootstrap_bi_picker.adapter_registry") as adapter_registry_mock,
        patch.object(BootstrapBiPicker, "_resolve_auth_params", return_value=auth),
        patch.object(BootstrapBiPicker, "_run_app", side_effect=run_returns),
        patch("datus.cli.bootstrap_bi_picker.DashboardAssembler") as assembler_cls,
    ):
        adapter_registry_mock.get_metadata.return_value = SimpleNamespace(auth_type="LOGIN")
        assembler_cls.return_value.assemble.return_value = assembled

        plan = picker.run()

    assert plan.options.platform == "superset"
    assert plan.adapter is fake_adapter
    assert plan.dashboard.name == "Sales"
    assert plan.dashboard_id == 1
    assert plan.pool_size == 3
    assert plan.assembled.tables == ["orders"]  # filtered by user's selection
    fake_adapter.list_dashboards.assert_called_once()
    fake_adapter.list_charts.assert_called_once_with(1)
    fake_adapter.list_datasets.assert_called_once_with(1)


def test_run_closes_adapter_when_dashboard_view_cancelled(console, fake_adapter) -> None:
    auth = SimpleNamespace(name="superset", auth_type="LOGIN")
    picker = BootstrapBiPicker(
        _agent_config(dashboard_config={"superset": SimpleNamespace()}),
        console,
    )
    picker._adapter_registry = {"superset": MagicMock(return_value=fake_adapter)}

    run_returns = [PendingFetch(service=_service_entry()), None]  # cancel at DASHBOARD view
    with (
        patch("datus.cli.bootstrap_bi_picker.build_service_entries", return_value=[_service_entry()]),
        patch("datus.cli.bootstrap_bi_picker.adapter_registry") as adapter_registry_mock,
        patch.object(BootstrapBiPicker, "_resolve_auth_params", return_value=auth),
        patch.object(BootstrapBiPicker, "_run_app", side_effect=run_returns),
    ):
        adapter_registry_mock.get_metadata.return_value = SimpleNamespace(auth_type="LOGIN")
        assert picker.run() is None
    fake_adapter.close.assert_called_once()


def test_run_url_fallback_when_list_dashboards_fails(console, fake_adapter) -> None:
    """``list_dashboards`` raising should switch the App to URL_FALLBACK
    rather than aborting the whole flow."""
    fake_adapter.list_dashboards.side_effect = RuntimeError("network down")
    auth = SimpleNamespace(name="superset", auth_type="LOGIN")
    picker = BootstrapBiPicker(
        _agent_config(dashboard_config={"superset": SimpleNamespace()}),
        console,
    )
    picker._adapter_registry = {"superset": MagicMock(return_value=fake_adapter)}

    run_returns = [PendingFetch(service=_service_entry()), None]  # cancel after fallback
    fake_app = MagicMock()
    with (
        patch("datus.cli.bootstrap_bi_picker.build_service_entries", return_value=[_service_entry()]),
        patch("datus.cli.bootstrap_bi_picker.adapter_registry") as adapter_registry_mock,
        patch.object(BootstrapBiPicker, "_resolve_auth_params", return_value=auth),
        patch.object(BootstrapBiPicker, "_build_bootstrap_app", return_value=fake_app),
        patch.object(BootstrapBiPicker, "_run_app", side_effect=run_returns),
    ):
        adapter_registry_mock.get_metadata.return_value = SimpleNamespace(auth_type="LOGIN")
        picker.run()

    fake_app.force_url_fallback.assert_called_once()


def test_run_raises_when_dashboard_has_no_charts(console, fake_adapter) -> None:
    fake_adapter.list_charts.return_value = []
    auth = SimpleNamespace(name="superset", auth_type="LOGIN")
    picker = BootstrapBiPicker(
        _agent_config(dashboard_config={"superset": SimpleNamespace()}),
        console,
    )
    picker._adapter_registry = {"superset": MagicMock(return_value=fake_adapter)}

    svc = _service_entry()
    run_returns = [
        PendingFetch(service=svc),
        BootstrapBiSelection(
            service=svc,
            dashboard_id=1,
            dashboard_name="x",
            dashboard_url="http://x/d/1",
            is_manual_url=False,
        ),
    ]
    with (
        patch("datus.cli.bootstrap_bi_picker.build_service_entries", return_value=[_service_entry()]),
        patch("datus.cli.bootstrap_bi_picker.adapter_registry") as adapter_registry_mock,
        patch.object(BootstrapBiPicker, "_resolve_auth_params", return_value=auth),
        patch.object(BootstrapBiPicker, "_run_app", side_effect=run_returns),
    ):
        adapter_registry_mock.get_metadata.return_value = SimpleNamespace(auth_type="LOGIN")
        with pytest.raises(ValueError, match="No charts found"):
            picker.run()
    fake_adapter.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# helper-method coverage
# ─────────────────────────────────────────────────────────────────


def test_items_from_adapter_result_handles_items_attribute() -> None:
    result = SimpleNamespace(items=[1, 2, 3])
    assert BootstrapBiPicker._items_from_adapter_result(result) == [1, 2, 3]


def test_items_from_adapter_result_handles_iterable_directly() -> None:
    assert BootstrapBiPicker._items_from_adapter_result([1, 2]) == [1, 2]
    assert BootstrapBiPicker._items_from_adapter_result(None) == []


def test_lookup_dashboard_config_finds_case_insensitive() -> None:
    cfg = SimpleNamespace()
    out = BootstrapBiPicker._lookup_dashboard_config({"Superset": cfg}, "superset")
    assert out is cfg


def test_charts_to_rows_flags_aggregation() -> None:
    chart = SimpleNamespace(
        id=1,
        name="orders",
        title=None,
        query=SimpleNamespace(sql=["SELECT SUM(price) FROM o", "SELECT a FROM b"]),
    )
    rows = BootstrapBiPicker._charts_to_rows([chart])
    assert rows[0].sql_count == 2
    assert rows[0].has_aggregation is True


def test_charts_to_rows_handles_missing_query() -> None:
    chart = SimpleNamespace(id=1, name="x", title=None, query=None)
    rows = BootstrapBiPicker._charts_to_rows([chart])
    assert rows[0].sql_count == 0
    assert rows[0].has_aggregation is False


def test_resolve_default_table_context_prefers_cli_context(console) -> None:
    cli = SimpleNamespace(
        cli_context=SimpleNamespace(
            current_catalog="cat",
            current_db_name="db",
            current_schema="sch",
        )
    )
    picker = BootstrapBiPicker(_agent_config(), console, cli=cli)
    assert picker._resolve_default_table_context() == ("cat", "db", "sch")


def test_resolve_default_table_context_falls_back_to_db_config(console) -> None:
    db_cfg = SimpleNamespace(catalog="from_cfg", database="d", schema="s")
    agent_config = SimpleNamespace(
        db_type="duckdb",
        current_datasource="local",
        dashboard_config={},
        current_db_config=lambda *_a, **_k: db_cfg,
    )
    picker = BootstrapBiPicker(agent_config, console)
    assert picker._resolve_default_table_context() == ("from_cfg", "d", "s")
