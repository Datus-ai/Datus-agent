# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_bi_app`.

The Application is exercised by manipulating the view state directly and
invoking the action handlers — running a real prompt_toolkit Application
inside pytest is unreliable across CI environments.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from datus.cli.bootstrap_bi_app import (
    BootstrapBiApp,
    BootstrapBiResult,
    BootstrapBiSelection,
    ChartRow,
    DashboardEntry,
    PendingAssemble,
    PendingFetch,
    ServiceEntry,
    TableRow,
    _derive_url,
    _View,
    build_service_entries,
)


@pytest.fixture()
def console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120, log_path=False)


@pytest.fixture()
def services() -> list[ServiceEntry]:
    return [
        ServiceEntry(name="superset_local", adapter_type="superset", api_base_url="http://localhost:8088"),
        ServiceEntry(name="grafana_dev", adapter_type="grafana", api_base_url="https://grafana.dev"),
    ]


def _stub_app_exit(app: BootstrapBiApp) -> MagicMock:
    """Replace the underlying ``Application.exit`` with a no-op spy so we
    can introspect what would have been returned without driving the real
    prompt_toolkit event loop."""
    spy = MagicMock()
    app._app.exit = spy  # type: ignore[assignment]
    return spy


# ─────────────────────────────────────────────────────────────────
# build_service_entries
# ─────────────────────────────────────────────────────────────────


def test_build_service_entries_flattens_dashboard_config() -> None:
    cfg = {
        "superset_local": SimpleNamespace(adapter_type="superset", api_base_url="http://localhost:8088"),
        "grafana_dev": SimpleNamespace(adapter_type="grafana", api_base_url="https://grafana.dev"),
    }
    entries = build_service_entries(cfg)
    assert {e.name for e in entries} == {"superset_local", "grafana_dev"}
    superset = next(e for e in entries if e.name == "superset_local")
    assert superset.adapter_type == "superset"
    assert superset.api_base_url == "http://localhost:8088"


def test_build_service_entries_falls_back_to_name_when_type_missing() -> None:
    cfg = {"my_custom": SimpleNamespace(api_base_url="http://x")}
    [entry] = build_service_entries(cfg)
    assert entry.adapter_type == "my_custom"


def test_build_service_entries_handles_empty() -> None:
    assert build_service_entries({}) == []
    assert build_service_entries(None) == []


# ─────────────────────────────────────────────────────────────────
# _derive_url
# ─────────────────────────────────────────────────────────────────


def test_derive_url_appends_dashboard_path() -> None:
    assert _derive_url("http://host:8088", 42, "x") == "http://host:8088/superset/dashboard/42/"


def test_derive_url_returns_empty_when_inputs_missing() -> None:
    assert _derive_url("", 42, "x") == ""
    assert _derive_url("http://host", None, "x") == ""


# ─────────────────────────────────────────────────────────────────
# Service view → PendingFetch
# ─────────────────────────────────────────────────────────────────


def test_service_enter_exits_with_pending_fetch(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    spy = _stub_app_exit(app)
    app._service_cursor = 1

    app._on_service_enter()

    assert isinstance(app._result, PendingFetch)
    assert app._result.service.name == "grafana_dev"
    spy.assert_called_once()
    assert spy.call_args.kwargs["result"] is app._result


def test_service_enter_with_no_services_exits_none(console: Console) -> None:
    app = BootstrapBiApp(console, services=[])
    spy = _stub_app_exit(app)

    app._on_service_enter()

    assert app._result is None
    spy.assert_called_once_with(result=None)


# ─────────────────────────────────────────────────────────────────
# set_dashboards switches view based on result
# ─────────────────────────────────────────────────────────────────


def test_set_dashboards_with_items_enters_dashboard_view(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]

    app.set_dashboards([DashboardEntry(id=1, name="Sales", url="http://x/d/1")])

    assert app._view == _View.DASHBOARD
    assert app._error_message is None
    assert len(app._dashboards) == 1


def test_set_dashboards_empty_falls_back_to_url(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]

    app.set_dashboards([])

    assert app._view == _View.URL_FALLBACK
    assert app._error_message and "paste URL" in app._error_message


def test_force_url_fallback_with_error_message(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]

    app.force_url_fallback("connection refused")

    assert app._view == _View.URL_FALLBACK
    assert "connection refused" in (app._error_message or "")


# ─────────────────────────────────────────────────────────────────
# Dashboard view → BootstrapBiSelection
# ─────────────────────────────────────────────────────────────────


def test_dashboard_enter_emits_selection(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    spy = _stub_app_exit(app)
    app.set_dashboards(
        [
            DashboardEntry(id=10, name="Sales", url=""),
            DashboardEntry(id=11, name="Marketing", url="http://localhost:8088/dashboards/11"),
        ]
    )
    app._dashboard_cursor = 1

    app._on_dashboard_enter()

    assert isinstance(app._result, BootstrapBiSelection)
    assert app._result.dashboard_id == 11
    assert app._result.dashboard_url == "http://localhost:8088/dashboards/11"
    assert app._result.is_manual_url is False
    spy.assert_called_once()


def test_dashboard_enter_synthesises_url_when_missing(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    _stub_app_exit(app)
    app.set_dashboards([DashboardEntry(id=42, name="Ad-hoc")])

    app._on_dashboard_enter()

    assert isinstance(app._result, BootstrapBiSelection)
    assert app._result.dashboard_url == "http://localhost:8088/superset/dashboard/42/"


def test_dashboard_filter_narrows_visible_items(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    app.set_dashboards(
        [
            DashboardEntry(id=1, name="Sales overview"),
            DashboardEntry(id=2, name="Marketing funnel"),
            DashboardEntry(id=3, name="Sales by region"),
        ]
    )
    app._filter.text = "sales"

    visible = app._filtered_dashboards()

    assert {d.id for d in visible} == {1, 3}


# ─────────────────────────────────────────────────────────────────
# URL fallback view
# ─────────────────────────────────────────────────────────────────


def test_url_submit_with_valid_url(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    spy = _stub_app_exit(app)
    app.force_url_fallback()
    app._url_input.text = "http://localhost:8088/superset/dashboard/7/"

    app._on_url_submit()

    assert isinstance(app._result, BootstrapBiSelection)
    assert app._result.is_manual_url is True
    assert app._result.dashboard_url.endswith("/7/")
    spy.assert_called_once()


def test_url_submit_rejects_empty(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    spy = _stub_app_exit(app)
    app.force_url_fallback()
    app._url_input.text = ""

    app._on_url_submit()

    assert app._result is None
    spy.assert_not_called()
    assert "URL is required" in (app._error_message or "")


def test_url_submit_rejects_malformed(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    spy = _stub_app_exit(app)
    app.force_url_fallback()
    app._url_input.text = "not a url"

    app._on_url_submit()

    spy.assert_not_called()
    assert "Invalid URL" in (app._error_message or "")


# ─────────────────────────────────────────────────────────────────
# Back navigation
# ─────────────────────────────────────────────────────────────────


def test_back_from_dashboard_returns_to_service(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    app.set_dashboards([DashboardEntry(id=1, name="A")])

    app._on_back()

    assert app._view == _View.SERVICE


def test_back_from_url_fallback_returns_to_dashboard_when_available(
    console: Console, services: list[ServiceEntry]
) -> None:
    app = BootstrapBiApp(console, services)
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    app.set_dashboards([DashboardEntry(id=1, name="A")])
    # Simulate the user pressing 'm' to reach URL_FALLBACK on top of an
    # already-loaded dashboard list.
    app._view = _View.URL_FALLBACK

    app._on_back()

    assert app._view == _View.DASHBOARD


def test_back_from_service_exits(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    spy = _stub_app_exit(app)

    app._on_back()

    spy.assert_called_once_with(result=None)


# ─────────────────────────────────────────────────────────────────
# Chart multi-select (CHART_REF + CHART_METRICS)
# ─────────────────────────────────────────────────────────────────


def _seed_chart_view(app: BootstrapBiApp, services: list[ServiceEntry]) -> None:
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    app._dashboard_pick = BootstrapBiSelection(
        service=services[0],
        dashboard_id=1,
        dashboard_name="dash",
        dashboard_url="http://x",
        is_manual_url=False,
    )
    app.set_charts(
        [
            ChartRow(id=10, name="Sales overview", sql_count=1, has_aggregation=False),
            ChartRow(id=11, name="Total by region", sql_count=1, has_aggregation=True),
            ChartRow(id=12, name="Latest orders", sql_count=2, has_aggregation=False),
        ]
    )


def test_set_charts_pre_selects_all_for_ref_and_aggregates_for_metrics(
    console: Console, services: list[ServiceEntry]
) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)

    assert app._view == _View.CHART_REF
    assert app._chart_ref_selected == {0, 1, 2}
    # Only chart with has_aggregation=True is pre-selected for metrics.
    assert app._chart_metrics_selected == {1}


def test_chart_toggle_flips_membership(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)
    app._chart_cursor = 0

    app._on_chart_toggle()
    assert 0 not in app._chart_ref_selected

    app._on_chart_toggle()
    assert 0 in app._chart_ref_selected


def test_chart_select_all_and_none(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)
    app._view = _View.CHART_METRICS  # default has only {1}

    app._on_chart_select_all(True)
    assert app._chart_metrics_selected == {0, 1, 2}

    app._on_chart_select_all(False)
    assert app._chart_metrics_selected == set()


def test_chart_enter_advances_ref_to_metrics(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)

    app._on_chart_enter()

    assert app._view == _View.CHART_METRICS
    assert app._chart_cursor == 0
    assert app._error_message is None


def test_chart_enter_from_metrics_emits_pending_assemble(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)
    spy = _stub_app_exit(app)
    app._view = _View.CHART_METRICS

    app._on_chart_enter()

    assert isinstance(app._result, PendingAssemble)
    assert app._result.chart_ref_indices == [0, 1, 2]
    assert app._result.chart_metrics_indices == [1]
    spy.assert_called_once()


def test_chart_enter_from_metrics_blocks_when_nothing_selected(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)
    spy = _stub_app_exit(app)
    app._view = _View.CHART_METRICS
    app._chart_ref_selected.clear()
    app._chart_metrics_selected.clear()

    app._on_chart_enter()

    spy.assert_not_called()
    assert "at least one chart" in (app._error_message or "")


def test_chart_back_from_metrics_returns_to_ref(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)
    app._view = _View.CHART_METRICS

    app._on_back()

    assert app._view == _View.CHART_REF


# ─────────────────────────────────────────────────────────────────
# Table multi-select
# ─────────────────────────────────────────────────────────────────


def _seed_table_view(app: BootstrapBiApp, services: list[ServiceEntry]) -> None:
    app._app.layout.focus = MagicMock()  # type: ignore[assignment]
    app._selected_service = services[0]
    app._dashboard_pick = BootstrapBiSelection(
        service=services[0],
        dashboard_id=1,
        dashboard_name="dash",
        dashboard_url="http://x",
        is_manual_url=False,
    )
    app.set_tables([TableRow(name="db.tbl_a"), TableRow(name="db.tbl_b"), TableRow(name="db.tbl_c")])


def test_set_tables_pre_selects_all(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_table_view(app, services)

    assert app._view == _View.TABLE_REVIEW
    assert app._table_selected == {0, 1, 2}


def test_table_toggle_and_select_all(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_table_view(app, services)
    app._table_cursor = 1

    app._on_table_toggle()
    assert 1 not in app._table_selected

    app._on_table_select_all(False)
    assert app._table_selected == set()

    app._on_table_select_all(True)
    assert app._table_selected == {0, 1, 2}


def test_table_enter_advances_to_concurrency(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_table_view(app, services)

    app._on_table_enter()

    assert app._view == _View.CONCURRENCY


# ─────────────────────────────────────────────────────────────────
# Concurrency view + final BootstrapBiResult
# ─────────────────────────────────────────────────────────────────


def test_concurrency_enter_emits_full_result(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_chart_view(app, services)
    # Walk the same App through tables to mimic real flow.
    app._chart_ref_selected = {0, 2}
    app._chart_metrics_selected = {1}
    app.set_tables([TableRow(name="t0"), TableRow(name="t1")])
    app._table_selected = {0}
    app._view = _View.CONCURRENCY
    app._pool_cursor = 2  # → 5 threads
    spy = _stub_app_exit(app)

    app._on_concurrency_enter()

    assert isinstance(app._result, BootstrapBiResult)
    assert app._result.pool_size == app._pool_choices[2] == 5
    assert app._result.chart_ref_indices == [0, 2]
    assert app._result.chart_metrics_indices == [1]
    assert app._result.table_indices == [0]
    assert app._result.dashboard_id == 1
    spy.assert_called_once()


def test_concurrency_back_returns_to_table_review(console: Console, services: list[ServiceEntry]) -> None:
    app = BootstrapBiApp(console, services)
    _seed_table_view(app, services)
    app._view = _View.CONCURRENCY

    app._on_back()

    assert app._view == _View.TABLE_REVIEW


def test_concurrency_default_cursor_at_three_threads(console: Console, services: list[ServiceEntry]) -> None:
    """Mirrors the historical ``init_reference_sql`` default (pool_size=3)."""
    app = BootstrapBiApp(console, services)

    assert app._pool_choices[app._pool_cursor] == 3
