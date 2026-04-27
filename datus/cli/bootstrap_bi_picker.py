# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/bootstrap-bi`` picker driver.

The picker is a synchronous, multi-stage state machine: each user-facing
view in :class:`BootstrapBiApp` is interleaved with out-of-band BI
adapter IO (``list_dashboards`` / ``list_charts`` / ``list_datasets``
/ assemble). This driver runs the four stages in order and returns a
:class:`BootstrapBiPlan` ready for the streaming pipeline to consume.

Keeping the picker in its own module means the streams / commands stay
purely about generation and rendering — they never touch a BI adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

try:
    from datus_bi_core import (
        AuthParam,
        AuthType,
        BIAdapterBase,
        ChartInfo,
        DashboardInfo,
        adapter_registry,
    )
except ImportError:  # pragma: no cover - optional dependency
    AuthParam = AuthType = BIAdapterBase = ChartInfo = DashboardInfo = adapter_registry = None  # type: ignore[assignment,misc]

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
    build_service_entries,
)
from datus.configuration.agent_config import AgentConfig, DashboardConfig
from datus.tools.bi_tools.dashboard_assembler import (
    ChartSelection,
    DashboardAssembler,
    DashboardAssemblyResult,
)
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI


@dataclass(slots=True)
class DashboardCliOptions:
    """Adapter constructor options collected by the picker."""

    platform: str
    dashboard_url: str
    api_base_url: str
    auth_params: "AuthParam | None" = None
    dialect: Optional[str] = None


@dataclass
class BootstrapBiPlan:
    """Output of :meth:`BootstrapBiPicker.run` consumed by the streams pipeline.

    The caller takes ownership of ``adapter`` — it must call
    ``adapter.close()`` when done (typically inside a ``finally`` block
    in the slash command's outer scope).
    """

    options: DashboardCliOptions
    adapter: "BIAdapterBase"
    dashboard: "DashboardInfo"
    dashboard_id: Optional[Union[int, str]]
    chart_selections_ref: List[ChartSelection]
    chart_selections_metrics: List[ChartSelection]
    assembled: DashboardAssemblyResult
    pool_size: int


class BootstrapBiPicker:
    """Drive the four-stage TUI picker plus the IO between stages.

    Returns:
        :class:`BootstrapBiPlan` on success; ``None`` when the user
        cancels at any view (Esc / EOF / KeyboardInterrupt).

    Raises:
        ValueError: when there is no usable BI adapter installed or
            configured, or when adapter IO fails outside the picker's
            recoverable paths (e.g. ``get_dashboard_info``).
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        console: Console,
        *,
        cli: Optional["DatusCLI"] = None,
    ) -> None:
        self.agent_config = agent_config
        self.console = console
        self.cli = cli
        self._adapter_registry = self._discover_adapters()

    # ── public ────────────────────────────────────────────────────────

    def run(self) -> Optional[BootstrapBiPlan]:
        if not self._adapter_registry:
            raise ValueError(
                "No BI adapter implementations found. Install the Superset adapter with: pip install datus-bi-superset"
            )

        services = build_service_entries(getattr(self.agent_config, "dashboard_config", {}) or {})
        services = [svc for svc in services if svc.adapter_type in self._adapter_registry]
        if not services:
            raise ValueError("No BI platforms configured. Run /services and add a dashboard service first.")

        app = self._build_bootstrap_app(services)

        # ── Stage 1: SERVICE ──────────────────────────────────────────
        first = self._run_app(app)
        if not isinstance(first, PendingFetch):
            return None
        service = first.service

        metadata = adapter_registry.get_metadata(service.adapter_type)
        if metadata is None:
            raise ValueError(f"Missing BI adapter metadata for '{service.adapter_type}'")
        auth_param = self._resolve_auth_params(service.name, metadata.auth_type)
        if auth_param is None:
            raise ValueError(f"Service `{service.name}` has no saved credentials. Re-run /services to fix it.")

        options = DashboardCliOptions(
            platform=service.adapter_type,
            dashboard_url="",
            api_base_url=service.api_base_url,
            auth_params=auth_param,
            dialect=self.agent_config.db_type,
        )
        adapter = self._create_adapter(options)

        # ── Stage 2: DASHBOARD ────────────────────────────────────────
        try:
            dashboards_raw = self._items_from_adapter_result(adapter.list_dashboards(limit=200))
        except Exception as exc:
            logger.warning("list_dashboards failed: %s", exc, exc_info=True)
            app.force_url_fallback(f"Failed to list dashboards: {exc}")
        else:
            entries = self._dashboards_to_entries(dashboards_raw)
            app.set_dashboards(entries)

        second = self._run_app(app)
        if not isinstance(second, BootstrapBiSelection):
            self._safe_close(adapter)
            return None

        options.dashboard_url = second.dashboard_url

        if second.is_manual_url:
            dashboard, dashboard_id = self._confirm_dashboard(adapter, second.dashboard_url)
            if dashboard is None:
                self._safe_close(adapter)
                return None
        else:
            try:
                dashboard = adapter.get_dashboard_info(second.dashboard_id)
            except Exception as exc:
                logger.error("get_dashboard_info failed: %s", exc, exc_info=True)
                self._safe_close(adapter)
                raise ValueError(f"Failed to load dashboard: {exc}") from exc
            dashboard_id = second.dashboard_id

        # ── Stage 3: CHART_REF + CHART_METRICS ────────────────────────
        try:
            chart_metas = self._items_from_adapter_result(adapter.list_charts(dashboard_id))
        except Exception as exc:
            logger.error("list_charts failed: %s", exc, exc_info=True)
            self._safe_close(adapter)
            raise ValueError(f"Failed to load charts: {exc}") from exc
        if not chart_metas:
            self._safe_close(adapter)
            raise ValueError("No charts found in this dashboard.")

        chart_details = self._hydrate_charts(adapter, dashboard_id, chart_metas)
        app.set_charts(self._charts_to_rows(chart_details))

        third = self._run_app(app)
        if not isinstance(third, PendingAssemble):
            self._safe_close(adapter)
            return None

        chart_selections_ref = self._load_chart_selections(chart_details, third.chart_ref_indices)
        chart_selections_metrics = self._load_chart_selections(chart_details, third.chart_metrics_indices)

        # ── Stage 4: TABLE_REVIEW + CONCURRENCY ───────────────────────
        default_catalog, default_database, default_schema = self._resolve_default_table_context()
        assembler = DashboardAssembler(
            adapter,
            default_dialect=options.dialect,
            default_catalog=default_catalog,
            default_database=default_database,
            default_schema=default_schema,
        )

        try:
            datasets = self._items_from_adapter_result(adapter.list_datasets(dashboard_id))
        except Exception as exc:
            logger.error("list_datasets failed: %s", exc, exc_info=True)
            self._safe_close(adapter)
            raise ValueError(f"Failed to load datasets: {exc}") from exc

        datasets = assembler.hydrate_datasets(datasets, dashboard_id)

        assembled = assembler.assemble(dashboard, chart_selections_ref, chart_selections_metrics, datasets)

        table_rows = [TableRow(name=str(t)) for t in (assembled.tables or []) if t]
        app.set_tables(table_rows)
        fourth = self._run_app(app)
        if not isinstance(fourth, BootstrapBiResult):
            self._safe_close(adapter)
            return None

        kept_tables = [table_rows[i].name for i in fourth.table_indices if 0 <= i < len(table_rows)]
        assembled.tables = kept_tables

        return BootstrapBiPlan(
            options=options,
            adapter=adapter,
            dashboard=dashboard,
            dashboard_id=dashboard_id,
            chart_selections_ref=chart_selections_ref,
            chart_selections_metrics=chart_selections_metrics,
            assembled=assembled,
            pool_size=fourth.pool_size,
        )

    # ── adapter discovery / construction ─────────────────────────────

    def _discover_adapters(self) -> dict:
        if adapter_registry is None:
            return {}
        return adapter_registry.list_adapters()

    def _build_bootstrap_app(self, services: List[ServiceEntry]) -> BootstrapBiApp:
        return BootstrapBiApp(self.console, services)

    def _run_app(self, app: BootstrapBiApp) -> Any:
        tui_app = getattr(self.cli, "tui_app", None) if self.cli else None
        if tui_app is not None and hasattr(tui_app, "suspend_input"):
            with tui_app.suspend_input():
                return app.run()
        return app.run()

    def _create_adapter(self, options: DashboardCliOptions) -> "BIAdapterBase":
        adapter_cls = self._adapter_registry.get(options.platform)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported platform '{options.platform}'. "
                "Install the Superset adapter with: pip install datus-bi-superset"
            )
        return adapter_cls(
            api_base_url=options.api_base_url,
            auth_params=options.auth_params,
            dialect=self.agent_config.db_type,
        )

    @staticmethod
    def _safe_close(adapter: Any) -> None:
        try:
            adapter.close()
        except Exception:
            pass

    # ── auth resolution ──────────────────────────────────────────────

    def _resolve_auth_params(self, service_name: str, auth_type: "AuthType") -> Optional["AuthParam"]:
        configs = getattr(self.agent_config, "dashboard_config", None) or {}
        config = self._lookup_dashboard_config(configs, service_name)
        if config is None:
            return None

        username = (config.username or "").strip()
        password = (config.password or "").strip()
        api_key = (config.api_key or "").strip()
        extra = config.extra or {}

        auth_param = AuthParam()
        if auth_type == AuthType.LOGIN:
            if not username or not password:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=f"Dashboard auth config for '{service_name}' requires username and password.",
                )
            auth_param.username = username
            auth_param.password = password
            auth_param.extra = extra
        elif auth_type == AuthType.API_KEY:
            if not api_key:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=f"Dashboard auth config for '{service_name}' requires api_key.",
                )
            auth_param.api_key = api_key
            auth_param.extra = extra
        else:
            raise ValueError(f"Unsupported auth type '{auth_type}'.")
        return auth_param

    @staticmethod
    def _lookup_dashboard_config(configs: dict, service_name: str) -> Optional[DashboardConfig]:
        if service_name in configs:
            return configs[service_name]
        key = (service_name or "").strip().lower()
        if key in configs:
            return configs[key]
        for name, config in configs.items():
            if (name or "").strip().lower() == key:
                return config
        return None

    # ── adapter result helpers ───────────────────────────────────────

    @staticmethod
    def _items_from_adapter_result(result: Any) -> list[Any]:
        if result is None:
            return []
        items = getattr(result, "items", None)
        if items is not None and not callable(items):
            return list(items or [])
        return list(result)

    @staticmethod
    def _dashboards_to_entries(dashboards: Sequence[Any]) -> List[DashboardEntry]:
        entries: List[DashboardEntry] = []
        for item in dashboards:
            entry_id = getattr(item, "id", None)
            entry_name = getattr(item, "name", None) or getattr(item, "title", "") or str(entry_id or "")
            entry_url = getattr(item, "url", "") or ""
            entries.append(DashboardEntry(id=entry_id, name=str(entry_name), url=str(entry_url)))
        return entries

    def _confirm_dashboard(
        self, adapter: "BIAdapterBase", dashboard_url: str
    ) -> Tuple[Optional["DashboardInfo"], Optional[Union[int, str]]]:
        dashboard_id = adapter.parse_dashboard_id(dashboard_url)
        try:
            dashboard = adapter.get_dashboard_info(dashboard_id)
        except Exception as exc:
            parsed = urlparse(dashboard_url)
            safe_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            logger.error("Failed to load dashboard from %s", safe_url, exc_info=True)
            raise ValueError(f"Failed to load dashboard: {exc}") from exc
        return dashboard, dashboard_id

    def _hydrate_charts(
        self,
        adapter: "BIAdapterBase",
        dashboard_id: Union[int, str],
        chart_metas: Sequence["ChartInfo"],
    ) -> List["ChartInfo"]:
        charts: List[ChartInfo] = []
        for chart_meta in chart_metas:
            try:
                chart_detail = adapter.get_chart(chart_meta.id, dashboard_id)
            except Exception as exc:
                logger.warning("Failed to load chart %s: %s", chart_meta.id, exc)
                chart_detail = None
            charts.append(chart_detail or chart_meta)
        return charts

    @staticmethod
    def _charts_to_rows(charts: Sequence["ChartInfo"]) -> List[ChartRow]:
        import re

        agg_re = re.compile(r"\b(?:SUM|COUNT|AVG|MAX|MIN)\s*\(", re.IGNORECASE)
        rows: List[ChartRow] = []
        for chart in charts:
            sqls: list = []
            try:
                if chart.query is not None:
                    sqls = list(chart.query.sql or [])
            except Exception:
                sqls = []
            sql_blob = "\n".join(s for s in sqls if isinstance(s, str))
            rows.append(
                ChartRow(
                    id=getattr(chart, "id", None),
                    name=str(getattr(chart, "name", None) or getattr(chart, "title", "") or ""),
                    sql_count=len(sqls),
                    has_aggregation=bool(agg_re.search(sql_blob)) if sql_blob else False,
                )
            )
        return rows

    @staticmethod
    def _load_chart_selections(
        charts: Sequence["ChartInfo"],
        indices: Sequence[int],
    ) -> List[ChartSelection]:
        selections: List[ChartSelection] = []
        if not indices:
            return selections
        for idx in indices:
            chart = charts[idx]
            sqls = chart.query.sql or [] if chart.query else []
            selections.append(ChartSelection(chart=chart, sql_indices=list(range(len(sqls)))))
        return selections

    # ── default db context ───────────────────────────────────────────

    def _resolve_default_table_context(self) -> Tuple[str, str, str]:
        catalog = ""
        database = ""
        schema = ""

        cli_context = getattr(self.cli, "cli_context", None) if self.cli else None
        if cli_context:
            catalog = (cli_context.current_catalog or "").strip()
            database = (cli_context.current_db_name or "").strip()
            schema = (cli_context.current_schema or "").strip()

        if not (catalog and database and schema):
            try:
                db_config = self.agent_config.current_db_config(self.agent_config.current_datasource)
            except Exception:
                db_config = None
            if db_config:
                if not catalog:
                    catalog = db_config.catalog or ""
                if not database:
                    database = db_config.database or ""
                if not schema:
                    schema = db_config.schema or ""

        return catalog, database, schema


__all__ = ["BootstrapBiPicker", "BootstrapBiPlan", "DashboardCliOptions"]
