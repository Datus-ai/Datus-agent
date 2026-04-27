# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/bootstrap-bi`` picker TUI.

A single :mod:`prompt_toolkit` :class:`Application` that walks the user
through two views:

1. ``SERVICE`` — pick one of the BI platforms already configured in
   ``agent.yml`` (rendered from :attr:`AgentConfig.dashboard_config`).
2. ``DASHBOARD`` — after the outer caller fetched the dashboard list via
   ``adapter.list_dashboards()``, the user picks one from a filterable
   list.
3. ``URL_FALLBACK`` — if ``list_dashboards`` is unsupported, throws, or
   returns empty, the App switches to a single-line URL input. The user
   can also press ``m`` from ``DASHBOARD`` to enter a URL manually.

Slow IO (``adapter.list_dashboards``) is intentionally NOT performed
inside the Application: the caller drives two ``app.run()`` invocations
separated by an out-of-band fetch, mirroring the
:class:`datus.cli.service_config_app.ServiceConfigApp` pattern.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, render_tui_title_bar
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _View(Enum):
    SERVICE = "service"
    DASHBOARD = "dashboard"
    URL_FALLBACK = "url"
    CHART_REF = "chart_ref"
    CHART_METRICS = "chart_metrics"
    TABLE_REVIEW = "table_review"
    CONCURRENCY = "concurrency"


# Default thread-pool sizes shown in the CONCURRENCY view. Single-select.
_POOL_CHOICES: Tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class ServiceEntry:
    """Row in the SERVICE view."""

    name: str
    adapter_type: str
    api_base_url: str = ""


@dataclass
class DashboardEntry:
    """Row in the DASHBOARD view."""

    id: Any
    name: str
    url: str = ""


@dataclass
class ChartRow:
    """Row in the CHART_REF / CHART_METRICS multi-select views.

    A single chart yields ONE row that the user toggles independently for
    reference-SQL inclusion (``CHART_REF``) and metrics inclusion
    (``CHART_METRICS``).
    """

    id: Any
    name: str
    sql_count: int = 0
    has_aggregation: bool = False


@dataclass
class TableRow:
    """Row in the TABLE_REVIEW multi-select view."""

    name: str


@dataclass
class PendingFetch:
    """SERVICE → caller fetches dashboards then re-enters."""

    service: ServiceEntry


@dataclass
class BootstrapBiSelection:
    """DASHBOARD → caller fetches charts then re-enters with ``set_charts``."""

    service: ServiceEntry
    dashboard_id: Optional[Any] = None
    dashboard_name: str = ""
    dashboard_url: str = ""
    is_manual_url: bool = False


@dataclass
class PendingAssemble:
    """CHART_METRICS → caller runs assembler.assemble() then ``set_tables``."""

    chart_ref_indices: List[int]
    chart_metrics_indices: List[int]


@dataclass
class BootstrapBiResult:
    """CONCURRENCY → final outcome of the picker."""

    service: ServiceEntry
    dashboard_id: Optional[Any]
    dashboard_name: str
    dashboard_url: str
    is_manual_url: bool
    chart_ref_indices: List[int]
    chart_metrics_indices: List[int]
    table_indices: List[int]
    pool_size: int


def _derive_url(api_base_url: str, dashboard_id: Any, name: str) -> str:
    """Best-effort URL synthesis when adapter does not expose ``dashboard.url``.

    Falls back to ``{api_base_url}/superset/dashboard/{id}/`` for Superset
    style platforms; callers that need exact URLs should set
    ``DashboardEntry.url`` directly.
    """
    base = (api_base_url or "").rstrip("/")
    if not base or dashboard_id in (None, ""):
        return ""
    return f"{base}/superset/dashboard/{dashboard_id}/"


class BootstrapBiApp:
    """Single Application driving SERVICE → DASHBOARD / URL_FALLBACK views."""

    def __init__(
        self,
        console: Console,
        services: Sequence[ServiceEntry],
        *,
        status_message: Optional[str] = None,
    ) -> None:
        self._console = console
        self._services: List[ServiceEntry] = list(services)
        self._dashboards: List[DashboardEntry] = []
        self._dashboards_loaded: bool = False
        self._view: _View = _View.SERVICE
        self._error_message: Optional[str] = None
        self._status_message: Optional[str] = status_message

        self._service_cursor: int = 0
        self._dashboard_cursor: int = 0
        self._dashboard_offset: int = 0

        # Multi-select state for CHART_REF / CHART_METRICS / TABLE_REVIEW.
        self._charts: List[ChartRow] = []
        self._chart_cursor: int = 0
        self._chart_offset: int = 0
        self._chart_ref_selected: set[int] = set()
        self._chart_metrics_selected: set[int] = set()
        self._tables: List[TableRow] = []
        self._table_cursor: int = 0
        self._table_offset: int = 0
        self._table_selected: set[int] = set()
        # CONCURRENCY single-select state.
        self._pool_choices: Tuple[int, ...] = _POOL_CHOICES
        # Default cursor at index 1 → 3 threads (mirrors the historical
        # ``init_reference_sql`` default).
        self._pool_cursor: int = 1

        self._selected_service: Optional[ServiceEntry] = None
        self._dashboard_pick: Optional[BootstrapBiSelection] = None
        self._result: Any = None

        self._filter = TextArea(
            height=1,
            multiline=False,
            prompt="filter: ",
            focus_on_click=True,
        )
        self._url_input = TextArea(
            height=1,
            multiline=False,
            prompt="dashboard URL: ",
            focus_on_click=True,
        )

        term_height = shutil.get_terminal_size((120, 40)).lines
        # title(1) + sep(1) + body header(1) + footer(1) + buffer(3)
        self._max_visible: int = max(3, min(15, term_height - 7))

        self._app = self._build_application()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> Optional[Any]:
        try:
            self._result = None
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("BootstrapBiApp crashed: %s", exc)
            return None

    def set_dashboards(self, items: Sequence[DashboardEntry]) -> None:
        """Feed the dashboards fetched by the caller and switch to that view."""
        self._dashboards = list(items)
        self._dashboards_loaded = True
        self._dashboard_cursor = 0
        self._dashboard_offset = 0
        self._filter.text = ""
        if not self._dashboards:
            self._view = _View.URL_FALLBACK
            self._error_message = "No dashboards listed; paste URL instead."
            self._app.layout.focus(self._url_input)
        else:
            self._view = _View.DASHBOARD
            self._error_message = None
            try:
                self._app.layout.focus(self._filter)
            except Exception:
                pass

    def force_url_fallback(self, message: Optional[str] = None) -> None:
        """Switch to URL fallback view (e.g. when list_dashboards raised)."""
        self._view = _View.URL_FALLBACK
        self._error_message = message or "Dashboard listing failed; paste URL instead."
        self._dashboards_loaded = True
        try:
            self._app.layout.focus(self._url_input)
        except Exception:
            pass

    def set_charts(self, charts: Sequence[ChartRow]) -> None:
        """Feed the chart rows fetched by the caller; switch to CHART_REF view.

        ``CHART_REF`` is the first multi-select view; pressing Enter advances
        to ``CHART_METRICS``. The same list is reused for both views — only
        the selection sets differ.
        """
        self._charts = list(charts)
        self._chart_cursor = 0
        self._chart_offset = 0
        # By default include every chart for reference SQL; for metrics
        # auto-pre-select rows that look aggregation-friendly so the user
        # only has to confirm.
        self._chart_ref_selected = set(range(len(self._charts)))
        self._chart_metrics_selected = {i for i, c in enumerate(self._charts) if c.has_aggregation}
        self._view = _View.CHART_REF
        self._error_message = None

    def set_tables(self, tables: Sequence[TableRow]) -> None:
        """Feed the assembled tables; switch to TABLE_REVIEW view."""
        self._tables = list(tables)
        self._table_cursor = 0
        self._table_offset = 0
        self._table_selected = set(range(len(self._tables)))
        self._view = _View.TABLE_REVIEW
        self._error_message = None

    # ─────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────

    def _build_application(self) -> Application:
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Bootstrap BI")),
            height=1,
        )

        service_window = Window(
            content=FormattedTextControl(self._render_service_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )
        dashboard_window = Window(
            content=FormattedTextControl(self._render_dashboard_list, focusable=False),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )
        chart_window = Window(
            content=FormattedTextControl(self._render_chart_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )
        table_window = Window(
            content=FormattedTextControl(self._render_table_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )
        concurrency_window = Window(
            content=FormattedTextControl(self._render_concurrency_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )

        dashboard_body = HSplit(
            [
                self._filter,
                Window(height=1, char="\u2500"),
                dashboard_window,
            ]
        )
        url_body = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_url_header, focusable=False),
                    height=Dimension(min=1, max=2),
                ),
                self._url_input,
            ]
        )

        def _body() -> Any:
            if self._view == _View.DASHBOARD:
                return dashboard_body
            if self._view == _View.URL_FALLBACK:
                return url_body
            if self._view in (_View.CHART_REF, _View.CHART_METRICS):
                return chart_window
            if self._view == _View.TABLE_REVIEW:
                return table_window
            if self._view == _View.CONCURRENCY:
                return concurrency_window
            return service_window

        body = DynamicContainer(_body)

        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:bootstrap-bi.error", f"  {self._error_message or ''}")]),
                height=1,
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )
        status_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:bootstrap-bi.status", f"  {self._status_message or ''}")]),
                height=1,
            ),
            filter=Condition(lambda: bool(self._status_message)),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        root = HSplit(
            [
                title_bar,
                Window(height=1, char="\u2500"),
                body,
                error_window,
                status_window,
                Window(height=1, char="\u2500"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root, focused_element=None),
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    # ─────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────

    def _render_service_list(self) -> List[Tuple[str, str]]:
        if not self._services:
            return [
                ("class:bootstrap-bi.empty", "  No BI platforms configured.\n"),
                ("class:bootstrap-bi.hint", "  Run /services to add one, then retry /bootstrap-bi.\n"),
            ]
        self._service_cursor = max(0, min(self._service_cursor, len(self._services) - 1))
        lines: List[Tuple[str, str]] = [("bold", "  Pick a configured BI platform:\n")]
        for i, svc in enumerate(self._services):
            label = f"{svc.name:<22} {svc.adapter_type:<12} {svc.api_base_url}"
            if i == self._service_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append((CLR_CURRENT if False else "", f"    {label}\n"))
        return lines

    def _filtered_dashboards(self) -> List[DashboardEntry]:
        needle = (self._filter.text or "").strip().lower()
        if not needle:
            return self._dashboards
        return [d for d in self._dashboards if needle in str(d.name).lower() or needle in str(d.id).lower()]

    def _render_dashboard_list(self) -> List[Tuple[str, str]]:
        items = self._filtered_dashboards()
        total = len(items)
        if total == 0:
            return [("class:bootstrap-bi.empty", "  No dashboards match the filter.\n")]
        self._dashboard_cursor = max(0, min(self._dashboard_cursor, total - 1))
        if self._dashboard_cursor < self._dashboard_offset:
            self._dashboard_offset = self._dashboard_cursor
        elif self._dashboard_cursor >= self._dashboard_offset + self._max_visible:
            self._dashboard_offset = self._dashboard_cursor - self._max_visible + 1
        end = min(total, self._dashboard_offset + self._max_visible)
        lines: List[Tuple[str, str]] = []
        if end - self._dashboard_offset < total:
            lines.append(
                (
                    "class:bootstrap-bi.scroll",
                    f"  ({self._dashboard_offset + 1}-{end} of {total})\n",
                )
            )
        for idx in range(self._dashboard_offset, end):
            entry = items[idx]
            label = f"{str(entry.id):<8} {entry.name}"
            if idx == self._dashboard_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_url_header(self) -> List[Tuple[str, str]]:
        svc = self._selected_service.name if self._selected_service else ""
        return [
            ("bold", f"  Manual URL entry for service `{svc}`:\n"),
        ]

    def _selected_set_for_chart_view(self) -> set[int]:
        return self._chart_ref_selected if self._view == _View.CHART_REF else self._chart_metrics_selected

    def _render_chart_list(self) -> List[Tuple[str, str]]:
        total = len(self._charts)
        if total == 0:
            return [("class:bootstrap-bi.empty", "  No charts in this dashboard.\n")]
        purpose = "reference SQL" if self._view == _View.CHART_REF else "metrics"
        selected = self._selected_set_for_chart_view()
        self._chart_cursor = max(0, min(self._chart_cursor, total - 1))
        if self._chart_cursor < self._chart_offset:
            self._chart_offset = self._chart_cursor
        elif self._chart_cursor >= self._chart_offset + self._max_visible:
            self._chart_offset = self._chart_cursor - self._max_visible + 1
        end = min(total, self._chart_offset + self._max_visible)
        lines: List[Tuple[str, str]] = [
            ("bold", f"  Select charts for {purpose} ({len(selected)}/{total} selected):\n"),
        ]
        if end - self._chart_offset < total:
            lines.append(
                (
                    "class:bootstrap-bi.scroll",
                    f"  ({self._chart_offset + 1}-{end} of {total})\n",
                )
            )
        for idx in range(self._chart_offset, end):
            row = self._charts[idx]
            check = "[x]" if idx in selected else "[ ]"
            agg_tag = " (agg)" if row.has_aggregation else ""
            label = f"{check} {str(row.id):<6} {row.name}{agg_tag}"
            if idx == self._chart_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_table_list(self) -> List[Tuple[str, str]]:
        total = len(self._tables)
        if total == 0:
            return [("class:bootstrap-bi.empty", "  No tables to review.\n")]
        self._table_cursor = max(0, min(self._table_cursor, total - 1))
        if self._table_cursor < self._table_offset:
            self._table_offset = self._table_cursor
        elif self._table_cursor >= self._table_offset + self._max_visible:
            self._table_offset = self._table_cursor - self._max_visible + 1
        end = min(total, self._table_offset + self._max_visible)
        lines: List[Tuple[str, str]] = [
            (
                "bold",
                f"  Review tables to scope ({len(self._table_selected)}/{total} selected):\n",
            ),
        ]
        if end - self._table_offset < total:
            lines.append(
                (
                    "class:bootstrap-bi.scroll",
                    f"  ({self._table_offset + 1}-{end} of {total})\n",
                )
            )
        for idx in range(self._table_offset, end):
            row = self._tables[idx]
            check = "[x]" if idx in self._table_selected else "[ ]"
            label = f"{check} {row.name}"
            if idx == self._table_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_concurrency_list(self) -> List[Tuple[str, str]]:
        self._pool_cursor = max(0, min(self._pool_cursor, len(self._pool_choices) - 1))
        lines: List[Tuple[str, str]] = [
            ("bold", "  Pick a thread-pool size for parallel LLM calls:\n"),
        ]
        for idx, value in enumerate(self._pool_choices):
            label = f"{value} threads"
            if idx == self._pool_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.SERVICE:
            base = "  \u2191\u2193 navigate   \u21b5 select   Esc cancel"
        elif self._view == _View.DASHBOARD:
            base = "  type to filter   \u2191\u2193 navigate   \u21b5 select   m manual URL   Esc back"
        elif self._view in (_View.CHART_REF, _View.CHART_METRICS):
            base = "  \u2191\u2193 navigate   Space toggle   a all   n none   \u21b5 next   Esc back"
        elif self._view == _View.TABLE_REVIEW:
            base = "  \u2191\u2193 navigate   Space toggle   a all   n none   \u21b5 next   Esc back"
        elif self._view == _View.CONCURRENCY:
            base = "  \u2191\u2193 navigate   \u21b5 finish   Esc back"
        else:
            base = "  \u21b5 confirm   Esc back"
        return [("class:bootstrap-bi.hint", base)]

    # ─────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────

    def _on_service_enter(self) -> None:
        if not self._services:
            self._app.exit(result=None)
            return
        self._selected_service = self._services[self._service_cursor]
        self._result = PendingFetch(service=self._selected_service)
        self._app.exit(result=self._result)

    def _on_dashboard_enter(self) -> None:
        items = self._filtered_dashboards()
        if not items:
            return
        entry = items[self._dashboard_cursor]
        url = entry.url or _derive_url(
            self._selected_service.api_base_url if self._selected_service else "",
            entry.id,
            entry.name,
        )
        if not url:
            self._error_message = "Selected dashboard has no URL; switch to manual entry (m)."
            return
        self._dashboard_pick = BootstrapBiSelection(
            service=self._selected_service,  # type: ignore[arg-type]
            dashboard_id=entry.id,
            dashboard_name=entry.name,
            dashboard_url=url,
            is_manual_url=False,
        )
        self._result = self._dashboard_pick
        self._app.exit(result=self._result)

    def _on_url_submit(self) -> None:
        url = (self._url_input.text or "").strip()
        if not url:
            self._error_message = "URL is required."
            return
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            self._error_message = "Invalid URL."
            return
        self._dashboard_pick = BootstrapBiSelection(
            service=self._selected_service,  # type: ignore[arg-type]
            dashboard_id=None,
            dashboard_name="",
            dashboard_url=url,
            is_manual_url=True,
        )
        self._result = self._dashboard_pick
        self._app.exit(result=self._result)

    def _on_chart_toggle(self) -> None:
        if not self._charts:
            return
        selected = self._selected_set_for_chart_view()
        if self._chart_cursor in selected:
            selected.remove(self._chart_cursor)
        else:
            selected.add(self._chart_cursor)

    def _on_chart_select_all(self, value: bool) -> None:
        selected = self._selected_set_for_chart_view()
        if value:
            selected.clear()
            selected.update(range(len(self._charts)))
        else:
            selected.clear()

    def _on_chart_enter(self) -> None:
        """CHART_REF → CHART_METRICS; CHART_METRICS → exit PendingAssemble."""
        if self._view == _View.CHART_REF:
            self._view = _View.CHART_METRICS
            self._chart_cursor = 0
            self._chart_offset = 0
            self._error_message = None
            return
        # CHART_METRICS confirm → exit
        if not self._chart_ref_selected and not self._chart_metrics_selected:
            self._error_message = "Select at least one chart for reference SQL or metrics."
            return
        self._result = PendingAssemble(
            chart_ref_indices=sorted(self._chart_ref_selected),
            chart_metrics_indices=sorted(self._chart_metrics_selected),
        )
        self._app.exit(result=self._result)

    def _on_table_toggle(self) -> None:
        if not self._tables:
            return
        if self._table_cursor in self._table_selected:
            self._table_selected.remove(self._table_cursor)
        else:
            self._table_selected.add(self._table_cursor)

    def _on_table_select_all(self, value: bool) -> None:
        if value:
            self._table_selected.clear()
            self._table_selected.update(range(len(self._tables)))
        else:
            self._table_selected.clear()

    def _on_table_enter(self) -> None:
        """TABLE_REVIEW → CONCURRENCY view (always advance, even if empty)."""
        self._view = _View.CONCURRENCY
        self._error_message = None

    def _on_concurrency_enter(self) -> None:
        if self._dashboard_pick is None:
            self._app.exit(result=None)
            return
        pool = self._pool_choices[self._pool_cursor]
        self._result = BootstrapBiResult(
            service=self._dashboard_pick.service,
            dashboard_id=self._dashboard_pick.dashboard_id,
            dashboard_name=self._dashboard_pick.dashboard_name,
            dashboard_url=self._dashboard_pick.dashboard_url,
            is_manual_url=self._dashboard_pick.is_manual_url,
            chart_ref_indices=sorted(self._chart_ref_selected),
            chart_metrics_indices=sorted(self._chart_metrics_selected),
            table_indices=sorted(self._table_selected),
            pool_size=pool,
        )
        self._app.exit(result=self._result)

    def _on_back(self) -> None:
        if self._view == _View.URL_FALLBACK and self._dashboards:
            self._view = _View.DASHBOARD
            self._error_message = None
            try:
                self._app.layout.focus(self._filter)
            except Exception:
                pass
            return
        if self._view == _View.DASHBOARD:
            self._view = _View.SERVICE
            self._error_message = None
            self._dashboards_loaded = False
            return
        if self._view == _View.CHART_METRICS:
            self._view = _View.CHART_REF
            self._chart_cursor = 0
            self._chart_offset = 0
            self._error_message = None
            return
        if self._view == _View.CHART_REF:
            # Going back from CHART_REF returns control to the caller so
            # they can re-fetch a different dashboard if desired.
            self._app.exit(result=None)
            return
        if self._view == _View.TABLE_REVIEW:
            # Going back from TABLE_REVIEW would require re-fetching charts;
            # punt on that and just cancel.
            self._app.exit(result=None)
            return
        if self._view == _View.CONCURRENCY:
            self._view = _View.TABLE_REVIEW
            self._error_message = None
            return
        self._app.exit(result=None)

    # ─────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_service = Condition(lambda: self._view == _View.SERVICE)
        is_dashboard = Condition(lambda: self._view == _View.DASHBOARD)
        is_url = Condition(lambda: self._view == _View.URL_FALLBACK)

        # Service view ---------------------------------------------------
        @kb.add("up", filter=is_service)
        def _(event):
            if self._services:
                self._service_cursor = (self._service_cursor - 1) % len(self._services)
            self._error_message = None

        @kb.add("down", filter=is_service)
        def _(event):
            if self._services:
                self._service_cursor = (self._service_cursor + 1) % len(self._services)
            self._error_message = None

        @kb.add("enter", filter=is_service)
        def _(event):
            self._on_service_enter()

        @kb.add("escape", filter=is_service)
        def _(event):
            event.app.exit(result=None)

        # Dashboard view -------------------------------------------------
        @kb.add("up", filter=is_dashboard)
        def _(event):
            items = self._filtered_dashboards()
            if items:
                self._dashboard_cursor = (self._dashboard_cursor - 1) % len(items)
            self._error_message = None

        @kb.add("down", filter=is_dashboard)
        def _(event):
            items = self._filtered_dashboards()
            if items:
                self._dashboard_cursor = (self._dashboard_cursor + 1) % len(items)
            self._error_message = None

        @kb.add("pageup", filter=is_dashboard)
        def _(event):
            self._dashboard_cursor = max(0, self._dashboard_cursor - 10)

        @kb.add("pagedown", filter=is_dashboard)
        def _(event):
            items = self._filtered_dashboards()
            if items:
                self._dashboard_cursor = min(len(items) - 1, self._dashboard_cursor + 10)

        @kb.add("enter", filter=is_dashboard)
        def _(event):
            self._on_dashboard_enter()

        @kb.add("c-m", filter=is_dashboard)
        def _(event):  # noqa: F811 - prompt_toolkit treats Ctrl+M as Enter on some terminals
            self._on_dashboard_enter()

        @kb.add("escape", filter=is_dashboard)
        def _(event):
            self._on_back()

        @kb.add("m", filter=is_dashboard)
        def _(event):
            # Only trigger when filter input is not focused, otherwise
            # treat it as text input. Detect by checking the focused
            # control.
            try:
                focused = event.app.layout.current_control
                if focused is self._filter.control:
                    # Pass through as character into the filter buffer.
                    self._filter.buffer.insert_text("m")
                    return
            except Exception:
                pass
            self._view = _View.URL_FALLBACK
            self._error_message = None
            try:
                event.app.layout.focus(self._url_input)
            except Exception:
                pass

        # URL fallback ---------------------------------------------------
        @kb.add("enter", filter=is_url)
        def _(event):
            self._on_url_submit()

        @kb.add("escape", filter=is_url)
        def _(event):
            self._on_back()

        # Chart multi-select (CHART_REF + CHART_METRICS) -----------------
        is_chart = Condition(lambda: self._view in (_View.CHART_REF, _View.CHART_METRICS))

        @kb.add("up", filter=is_chart)
        def _(event):
            if self._charts:
                self._chart_cursor = (self._chart_cursor - 1) % len(self._charts)
            self._error_message = None

        @kb.add("down", filter=is_chart)
        def _(event):
            if self._charts:
                self._chart_cursor = (self._chart_cursor + 1) % len(self._charts)
            self._error_message = None

        @kb.add("pageup", filter=is_chart)
        def _(event):
            self._chart_cursor = max(0, self._chart_cursor - 10)

        @kb.add("pagedown", filter=is_chart)
        def _(event):
            if self._charts:
                self._chart_cursor = min(len(self._charts) - 1, self._chart_cursor + 10)

        @kb.add("space", filter=is_chart)
        def _(event):
            self._on_chart_toggle()

        @kb.add("a", filter=is_chart)
        def _(event):
            self._on_chart_select_all(True)

        @kb.add("n", filter=is_chart)
        def _(event):
            self._on_chart_select_all(False)

        @kb.add("enter", filter=is_chart)
        def _(event):
            self._on_chart_enter()

        @kb.add("escape", filter=is_chart)
        def _(event):
            self._on_back()

        # Table review ---------------------------------------------------
        is_table = Condition(lambda: self._view == _View.TABLE_REVIEW)

        @kb.add("up", filter=is_table)
        def _(event):
            if self._tables:
                self._table_cursor = (self._table_cursor - 1) % len(self._tables)
            self._error_message = None

        @kb.add("down", filter=is_table)
        def _(event):
            if self._tables:
                self._table_cursor = (self._table_cursor + 1) % len(self._tables)
            self._error_message = None

        @kb.add("pageup", filter=is_table)
        def _(event):
            self._table_cursor = max(0, self._table_cursor - 10)

        @kb.add("pagedown", filter=is_table)
        def _(event):
            if self._tables:
                self._table_cursor = min(len(self._tables) - 1, self._table_cursor + 10)

        @kb.add("space", filter=is_table)
        def _(event):
            self._on_table_toggle()

        @kb.add("a", filter=is_table)
        def _(event):
            self._on_table_select_all(True)

        @kb.add("n", filter=is_table)
        def _(event):
            self._on_table_select_all(False)

        @kb.add("enter", filter=is_table)
        def _(event):
            self._on_table_enter()

        @kb.add("escape", filter=is_table)
        def _(event):
            self._on_back()

        # Concurrency ----------------------------------------------------
        is_pool = Condition(lambda: self._view == _View.CONCURRENCY)

        @kb.add("up", filter=is_pool)
        def _(event):
            if self._pool_choices:
                self._pool_cursor = (self._pool_cursor - 1) % len(self._pool_choices)

        @kb.add("down", filter=is_pool)
        def _(event):
            if self._pool_choices:
                self._pool_cursor = (self._pool_cursor + 1) % len(self._pool_choices)

        @kb.add("enter", filter=is_pool)
        def _(event):
            self._on_concurrency_enter()

        @kb.add("escape", filter=is_pool)
        def _(event):
            self._on_back()

        return kb


def build_service_entries(dashboard_config: Any, adapter_registry_obj: Any = None) -> List[ServiceEntry]:
    """Flatten ``AgentConfig.dashboard_config`` into :class:`ServiceEntry` rows.

    Accepts the dict directly (mapping name → ``DashboardConfig``) and
    optionally an adapter registry to enrich ``adapter_type`` when the
    config does not pin it explicitly.
    """
    out: List[ServiceEntry] = []
    if not dashboard_config:
        return out
    items = dashboard_config.items() if hasattr(dashboard_config, "items") else []
    for name, cfg in items:
        adapter_type = getattr(cfg, "adapter_type", None) or getattr(cfg, "type", None) or name
        api_base_url = getattr(cfg, "api_base_url", "") or ""
        out.append(
            ServiceEntry(
                name=name,
                adapter_type=str(adapter_type),
                api_base_url=str(api_base_url),
            )
        )
    return out


__all__ = [
    "BootstrapBiApp",
    "BootstrapBiResult",
    "BootstrapBiSelection",
    "ChartRow",
    "DashboardEntry",
    "PendingAssemble",
    "PendingFetch",
    "ServiceEntry",
    "TableRow",
    "build_service_entries",
    "_derive_url",
]


# Suppress unused field warning — `_dashboards_loaded` is referenced via
# instance state inside multiple methods and exists for future state
# transitions.
_ = field  # type: ignore[name-defined]
