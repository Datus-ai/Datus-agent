# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""BIFuncTool: LLM function calling layer for BI adapters."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, trans_to_function_tool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig, DashboardConfig

_VALID_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")
_VALID_IF_EXISTS = {"replace", "append", "fail"}

logger = get_logger(__name__)


class BIFuncTool:
    """
    LLM function calling layer for BI adapters.

    Aligned with ``SemanticTools`` / ``SchedulerTools`` pattern: takes
    ``agent_config`` plus an optional service name and constructs the underlying
    BI adapter lazily via ``datus_bi_core.adapter_registry``.

    Dynamically exposes tools based on adapter capabilities:
    - All adapters: list_dashboards, get_dashboard, list_charts, get_chart, list_datasets
    - Supported adapters: get_chart_data
    - DashboardWriteMixin: create_dashboard, update_dashboard
    - ChartWriteMixin: create_chart, update_chart, add_chart_to_dashboard
    - DatasetWriteMixin: create_dataset, list_bi_databases
    - dataset_db set on the service config: write_query
    """

    def __init__(
        self,
        agent_config: Optional["AgentConfig"] = None,
        bi_service: Optional[str] = None,
        *,
        adapter: Any = None,
    ) -> None:
        self.agent_config = agent_config
        self.bi_service = bi_service
        self._adapter = adapter
        self._dash_cfg_resolved = False
        self._dash_cfg: Optional["DashboardConfig"] = None
        self._read_connector: Any = None
        self._read_connector_loaded = False
        self._write_engine = None
        self._dataset_db_id = None
        self._grafana_ds_uid = None

    # ------------------------------------------------------------------ #
    # Service resolution + lazy adapter construction
    # ------------------------------------------------------------------ #

    def _resolved_platform(self) -> Optional[str]:
        """Return the BI platform name to use.

        Preference order:
        1. Explicit ``bi_service`` passed to the constructor.
        2. Auto-pick when ``agent_config.dashboard_config`` has exactly one entry.
        3. Raise ``DatusException`` when multiple are configured and no
           ``bi_service`` disambiguates — mirrors ``SemanticTools`` behavior.
        """
        if self.bi_service:
            return self.bi_service
        if self.agent_config is None:
            return None
        dashboard_config = getattr(self.agent_config, "dashboard_config", {}) or {}
        if len(dashboard_config) == 1:
            return next(iter(dashboard_config))
        if len(dashboard_config) > 1:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"Multiple BI platforms configured ({list(dashboard_config.keys())}); "
                    "pass `bi_service` explicitly to disambiguate."
                ),
            )
        return None

    @property
    def _resolved_dash_cfg(self) -> Optional["DashboardConfig"]:
        """Lazily resolve the ``DashboardConfig`` for the selected service."""
        if not self._dash_cfg_resolved:
            self._dash_cfg_resolved = True
            platform = self._resolved_platform()
            if platform and self.agent_config is not None:
                dashboard_config = getattr(self.agent_config, "dashboard_config", {}) or {}
                self._dash_cfg = dashboard_config.get(platform)
        return self._dash_cfg

    @property
    def adapter(self) -> Any:
        """Lazily construct the underlying BI adapter on first access.

        Tests may short-circuit construction by passing ``adapter=`` to the
        constructor.
        """
        if self._adapter is None:
            self._adapter = self._build_adapter()
        return self._adapter

    def _build_adapter(self) -> Any:
        """Build the BI adapter from the resolved DashboardConfig.

        Encapsulates the logic previously inline in
        ``gen_dashboard_agentic_node._setup_bi_tools``.

        The adapter is looked up by ``DashboardConfig.adapter_type`` — not
        by the service alias — so a multi-instance deployment like
        ``services.bi_platforms.superset_prod: { type: superset, ... }`` targets
        the registered ``superset`` adapter while the service still appears
        under its unique alias for CLI / dashboard addressing. When
        ``adapter_type`` is empty (legacy single-instance configs that
        omitted ``type``) it falls back to the service alias, preserving
        existing behaviour.
        """
        platform = self._resolved_platform()
        dash_cfg = self._resolved_dash_cfg
        if not platform or dash_cfg is None:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"BI service '{platform}' not found in `agent.services.bi_platforms`. "
                    "Configure it or pass a pre-built adapter to BIFuncTool(adapter=...)."
                ),
            )

        adapter_type = getattr(dash_cfg, "adapter_type", "") or platform

        from datus_bi_core import AuthParam, adapter_registry

        adapter_registry.discover_adapters()
        adapter_cls = adapter_registry.get(adapter_type)
        if not adapter_cls:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(f"No BI adapter registered for type '{adapter_type}' (service alias: '{platform}')"),
            )

        # Derive dialect from dataset_db config (explicit > inferred from URI).
        dialect = ""
        if dash_cfg.dataset_db:
            dialect = dash_cfg.dataset_db.get("dialect", "")
            if not dialect:
                ds_uri = dash_cfg.dataset_db.get("uri", "")
                if ds_uri:
                    try:
                        from sqlalchemy.engine.url import make_url

                        dialect = make_url(ds_uri).get_backend_name()
                    except Exception:
                        pass

        return adapter_cls(
            api_base_url=dash_cfg.api_url,
            auth_params=AuthParam(
                username=dash_cfg.username,
                password=dash_cfg.password,
                api_key=dash_cfg.api_key,
                extra=dash_cfg.extra or {},
            ),
            dialect=dialect,
        )

    # ------------------------------------------------------------------ #
    # Derived config properties (sourced from DashboardConfig.dataset_db)
    # ------------------------------------------------------------------ #

    @property
    def dataset_db_uri(self) -> str:
        dash_cfg = self._resolved_dash_cfg
        if dash_cfg and dash_cfg.dataset_db:
            return dash_cfg.dataset_db.get("uri", "") or ""
        return ""

    @property
    def dataset_db_schema(self) -> str:
        dash_cfg = self._resolved_dash_cfg
        if dash_cfg and dash_cfg.dataset_db:
            return dash_cfg.dataset_db.get("schema", "") or ""
        return ""

    @property
    def datasource_name(self) -> str:
        dash_cfg = self._resolved_dash_cfg
        if dash_cfg and dash_cfg.dataset_db:
            return dash_cfg.dataset_db.get("datasource_name", "") or ""
        return ""

    @property
    def read_connector(self) -> Any:
        """Source DB connector for ``write_query`` (lazy).

        Loaded once on first access from ``db_manager_instance``. Tests that
        need to inject a mock can assign ``self._read_connector`` directly
        before first access — the property short-circuits when a non-None
        connector is already cached.
        """
        if self._read_connector is None and not self._read_connector_loaded:
            self._read_connector_loaded = True
            if self.agent_config is None:
                return None
            try:
                from datus.tools.db_tools.db_manager import db_manager_instance

                db_manager = db_manager_instance(self.agent_config.namespaces)
                current_db = getattr(self.agent_config, "current_datasource", "") or ""
                # Pick whichever connector the namespace carries rather than
                # assuming ``namespace == logic_db``. In the post-services
                # migration they happen to match, but ``first_conn_with_name``
                # stays correct if a namespace ever bundles multiple DBs again.
                _, self._read_connector = db_manager.first_conn_with_name(current_db)
            except Exception as exc:
                logger.warning(f"No source DB connector for write_query: {exc}")
                self._read_connector = None
        return self._read_connector

    # ------------------------------------------------------------------ #
    # Read operations (available on all adapters)
    # ------------------------------------------------------------------ #

    def list_dashboards(self, search: str = "", limit: int = 50, offset: int = 0) -> FuncToolResult:
        """List dashboards in the BI platform. Optionally filter by search keyword.

        Returns:
            FuncToolResult with result as FuncToolListResult:
              - items (List[Dict]): dashboard rows
              - total (int | None): upstream full count (Superset exposes it;
                Grafana /api/search doesn't → None)
              - has_more (bool | None): next-page hint
              - extra (dict | None): {"next_offset": int} when has_more is True

            Pagination: call again with offset=extra.next_offset until
            has_more is False.
        """
        if not hasattr(self.adapter, "list_dashboards"):
            return FuncToolResult(success=0, error="This adapter does not support list_dashboards")
        try:
            page = self.adapter.list_dashboards(search=search, limit=limit, offset=offset)
            return self._build_list_envelope(page, offset=offset, limit=limit)
        except Exception as exc:
            logger.warning(f"list_dashboards failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def get_dashboard(self, dashboard_id: str) -> FuncToolResult:
        """Get detailed information about a specific dashboard by its ID."""
        try:
            result = self.adapter.get_dashboard_info(dashboard_id)
            if result is None:
                return FuncToolResult(success=0, error=f"Dashboard {dashboard_id} not found")
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"get_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def list_charts(self, dashboard_id: str, limit: int = 50, offset: int = 0) -> FuncToolResult:
        """List all charts/panels in a dashboard.

        Returns a FuncToolListResult envelope; see list_dashboards for details.
        """
        try:
            page = self.adapter.list_charts(dashboard_id, limit=limit, offset=offset)
            return self._build_list_envelope(page, offset=offset, limit=limit)
        except Exception as exc:
            logger.warning(f"list_charts failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def get_chart(self, chart_id: str, dashboard_id: str = "") -> FuncToolResult:
        """Get detailed information about a specific chart or panel by its ID.

        For Grafana, pass dashboard_id because panels are scoped to a dashboard.
        """
        try:
            dashboard_arg = dashboard_id.strip() or None
            result = self.adapter.get_chart(chart_id, dashboard_id=dashboard_arg)
            if result is None:
                return FuncToolResult(success=0, error=f"Chart {chart_id} not found")
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"get_chart failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def get_chart_data(self, chart_id: str, dashboard_id: str = "", limit: int = 0) -> FuncToolResult:
        """Get backend query results for a specific chart.

        Supported on adapters that expose chart query execution.
        """
        if not self._supports_chart_data():
            return FuncToolResult(
                success=0,
                error="This adapter does not support get_chart_data",
            )

        try:
            dashboard_arg = dashboard_id.strip() or None
            limit_arg = None
            if limit not in (None, "", 0, "0"):
                limit_arg = int(limit)
                if limit_arg < 0:
                    return FuncToolResult(success=0, error="limit must be a non-negative integer")
            result = self.adapter.get_chart_data(chart_id, dashboard_id=dashboard_arg, limit=limit_arg)
            if result is None:
                return FuncToolResult(success=0, error=f"Chart {chart_id} data not found")
            return FuncToolResult(result=result.model_dump())
        except NotImplementedError as exc:
            logger.warning(f"get_chart_data unsupported: {exc}")
            return FuncToolResult(
                success=0,
                error="This adapter does not support get_chart_data",
            )
        except Exception as exc:
            logger.warning(f"get_chart_data failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def list_datasets(self, dashboard_id: str = "", limit: int = 50, offset: int = 0) -> FuncToolResult:
        """List datasets available in the BI platform.

        For Superset, pass dashboard_id to scope results. Returns a
        FuncToolListResult envelope; see list_dashboards for details.
        """
        try:
            page = self.adapter.list_datasets(dashboard_id, limit=limit, offset=offset)
            return self._build_list_envelope(page, offset=offset, limit=limit)
        except Exception as exc:
            logger.warning(f"list_datasets failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    @staticmethod
    def _build_list_envelope(page: Any, *, offset: int, limit: int) -> FuncToolResult:
        """Translate an adapter ``PaginatedResult[T]`` into ``FuncToolListResult``.

        * ``items`` = rows serialised via ``.model_dump()`` so the LLM / CLI
          sees plain dicts.
        * ``total`` comes from the adapter when the upstream reports it
          (Superset ``count``). ``None`` on platforms that don't
          (Grafana ``/api/search``).
        * ``has_more`` is exact when ``total`` is known; otherwise falls
          back to ``len(items) == limit`` as the "looks like another page"
          heuristic.
        * ``extra.next_offset`` is provided only when ``has_more`` is True,
          so consumers can paste it back verbatim.
        """
        rows = [r.model_dump() for r in page.items]
        total = page.total
        if total is not None:
            has_more: bool | None = offset + len(rows) < total
        elif limit > 0:
            has_more = len(rows) == limit
        else:
            has_more = None
        extra = {"next_offset": offset + len(rows)} if has_more else None
        return FuncToolResult(
            success=1,
            result=FuncToolListResult(items=rows, total=total, has_more=has_more, extra=extra).model_dump(),
        )

    # ------------------------------------------------------------------ #
    # Dashboard write operations (DashboardWriteMixin)
    # ------------------------------------------------------------------ #

    def create_dashboard(self, title: str, description: str = "") -> FuncToolResult:
        """Create a new empty dashboard with the given title."""
        try:
            from datus_bi_core.models import DashboardSpec

            spec = DashboardSpec(title=title, description=description)
            result = self.adapter.create_dashboard(spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"create_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def update_dashboard(self, dashboard_id: str, title: str = "", description: str = "") -> FuncToolResult:
        """Update an existing dashboard's title or description."""
        try:
            from datus_bi_core.models import DashboardSpec

            existing = self.adapter.get_dashboard_info(dashboard_id)
            if existing is None:
                return FuncToolResult(success=0, error=f"Dashboard {dashboard_id} not found")
            spec = DashboardSpec(
                title=title or existing.name,
                description=description or (existing.description or ""),
            )
            result = self.adapter.update_dashboard(dashboard_id, spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"update_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def delete_dashboard(self, dashboard_id: str) -> FuncToolResult:
        """Delete a dashboard by its ID."""
        try:
            success = self.adapter.delete_dashboard(dashboard_id)
            return FuncToolResult(result={"deleted": success, "dashboard_id": dashboard_id})
        except Exception as exc:
            logger.warning(f"delete_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    # ------------------------------------------------------------------ #
    # Chart write operations (ChartWriteMixin)
    # ------------------------------------------------------------------ #

    def create_chart(
        self,
        chart_type: str,
        title: str,
        dataset_id: str = "",
        x_axis: str = "",
        metrics: str = "",
        dimensions: str = "",
        dashboard_id: str = "",
        description: str = "",
        sql: str = "",
    ) -> FuncToolResult:
        """
        Create a new chart/panel.

        For Superset: requires dataset_id (create one first with create_dataset()).
        For Grafana: requires sql and dashboard_id. The datasource is auto-resolved from dataset_db config.

        Args:
            chart_type: Type of chart: bar, line, pie, table, big_number, scatter
            title: Chart title
            dataset_id: (Superset: required) Dataset ID from create_dataset()
            x_axis: Column name for x-axis or time column (for line/bar charts)
            metrics: Comma-separated metric expressions. Supported formats:
                     - "column_name" → defaults to SUM(column_name)
                     - "AVG(column_name)", "MAX(column_name)", "MIN(column_name)", "COUNT(column_name)"
                     Examples: "revenue,count" or "AVG(activity_count)" or "MAX(price),MIN(price)"
            dimensions: Comma-separated list of dimension/groupby column names
            dashboard_id: (Grafana: required) Dashboard ID to add the chart to
            description: Chart description
            sql: SQL query for the chart (Grafana: required, used directly in the panel)
        """
        try:
            from datus_bi_core.models import ChartSpec

            metrics_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
            dims_list = [d.strip() for d in dimensions.split(",") if d.strip()] if dimensions else None
            ds_id = int(dataset_id) if dataset_id.strip().isdigit() else None

            # Superset requires dataset_id; Grafana requires sql + dashboard_id
            if not ds_id and not sql:
                return FuncToolResult(
                    success=0,
                    error="Either dataset_id (Superset) or sql (Grafana) is required.",
                )

            extra = {}
            dash_id = dashboard_id.strip() or None
            # Auto-resolve Grafana datasource UID when sql is provided
            if sql:
                if not dash_id:
                    return FuncToolResult(
                        success=0,
                        error="dashboard_id is required when using sql (Grafana). Create a dashboard first.",
                    )
                ds_uid = self._resolve_grafana_datasource_uid()
                if ds_uid:
                    extra["datasource_uid"] = ds_uid

            spec = ChartSpec(
                chart_type=chart_type,
                title=title,
                description=description,
                dataset_id=ds_id,
                sql=sql or None,
                x_axis=x_axis or None,
                metrics=metrics_list,
                dimensions=dims_list,
                extra=extra,
            )
            result = self.adapter.create_chart(spec, dashboard_id=dash_id)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"create_chart failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def update_chart(
        self,
        chart_id: str,
        title: str = "",
        chart_type: str = "",
        sql: str = "",
        metrics: str = "",
        x_axis: str = "",
        description: str = "",
    ) -> FuncToolResult:
        """Update an existing chart's type, SQL, metrics, or title."""
        try:
            from datus_bi_core.models import ChartSpec

            metrics_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
            existing = self.adapter.get_chart(chart_id)
            if existing is None:
                return FuncToolResult(success=0, error=f"Chart {chart_id} not found")
            spec = ChartSpec(
                chart_type=chart_type or existing.chart_type,
                title=title or existing.name,
                description=description or getattr(existing, "description", "") or "",
                sql=sql or getattr(existing, "sql", None),
                x_axis=x_axis or getattr(existing, "x_axis", None),
                metrics=metrics_list or getattr(existing, "metrics", None),
            )
            result = self.adapter.update_chart(chart_id, spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"update_chart failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def add_chart_to_dashboard(self, chart_id: str, dashboard_id: str) -> FuncToolResult:
        """Add an existing chart to a dashboard."""
        try:
            success = self.adapter.add_chart_to_dashboard(dashboard_id, chart_id)
            return FuncToolResult(result={"success": success, "chart_id": chart_id, "dashboard_id": dashboard_id})
        except Exception as exc:
            logger.warning(f"add_chart_to_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def delete_chart(self, chart_id: str) -> FuncToolResult:
        """Delete a chart by its ID."""
        try:
            success = self.adapter.delete_chart(chart_id)
            return FuncToolResult(result={"deleted": success, "chart_id": chart_id})
        except Exception as exc:
            logger.warning(f"delete_chart failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    # ------------------------------------------------------------------ #
    # Dataset write operations (DatasetWriteMixin)
    # ------------------------------------------------------------------ #

    def create_dataset(self, name: str, database_id: str, sql: str = "", description: str = "") -> FuncToolResult:
        """
        Create a dataset in the BI platform.

        For a physical table (already exists in the DB): omit sql or leave it empty.
        For a virtual/SQL dataset: provide the sql SELECT query.

        Args:
            name: Dataset name (also used as the table name for physical datasets)
            database_id: The BI platform's database connection ID (use list_bi_databases() to find it)
            sql: Optional SELECT query for virtual datasets. Leave empty to register an existing physical table.
            description: Optional description
        """
        if not database_id or not database_id.strip().isdigit():
            return FuncToolResult(
                success=0,
                error="database_id must be a numeric ID. Use list_bi_databases() to find available database IDs.",
            )
        try:
            from datus_bi_core.models import DatasetSpec

            spec = DatasetSpec(name=name, sql=sql or None, database_id=int(database_id), description=description)
            result = self.adapter.create_dataset(spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"create_dataset failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def list_bi_databases(self) -> FuncToolResult:
        """List available database connections in the BI platform. Call this before create_dataset."""
        try:
            results = self.adapter.list_bi_databases()
            return FuncToolResult(result=results)
        except Exception as exc:
            logger.warning(f"list_bi_databases failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def delete_dataset(self, dataset_id: str) -> FuncToolResult:
        """Delete a dataset by its ID."""
        try:
            success = self.adapter.delete_dataset(dataset_id)
            return FuncToolResult(result={"deleted": success, "dataset_id": dataset_id})
        except Exception as exc:
            logger.warning(f"delete_dataset failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    # ------------------------------------------------------------------ #
    # Write query (source DB → dashboard DB)
    # ------------------------------------------------------------------ #

    def write_query(
        self,
        sql: str,
        table_name: str,
        if_exists: str = "replace",
    ) -> FuncToolResult:
        """
        Execute a SQL query on the source database (via the active connector) and write
        the result set to the dashboard's own database as a new table.

        This lets you materialise query results inside the BI platform's database so
        that Superset/Grafana can query them directly without touching the source DB.

        Args:
            sql: SELECT statement to run on the source (namespace) database.
            table_name: Target table name inside the dashboard database.
            if_exists: What to do if the table already exists: "replace" (default),
                       "append", or "fail".
        """
        if not self.dataset_db_uri:
            return FuncToolResult(success=0, error="dataset_db is not configured for this BI platform")
        if not _VALID_TABLE_NAME.match(table_name):
            return FuncToolResult(success=0, error="Invalid table_name: must match [a-zA-Z_][a-zA-Z0-9_]{0,62}")
        if if_exists not in _VALID_IF_EXISTS:
            return FuncToolResult(success=0, error=f"if_exists must be one of: {sorted(_VALID_IF_EXISTS)}")
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return FuncToolResult(success=0, error="Only SELECT/WITH queries are allowed in write_query")
        # Reject multi-statement SQL to prevent piggy-backed writes (e.g., "SELECT 1; DROP TABLE x")
        if ";" in sql_stripped.rstrip(";"):
            return FuncToolResult(success=0, error="Multi-statement SQL is not allowed in write_query")
        try:
            read_connector = self.read_connector
            if read_connector is None:
                return FuncToolResult(success=0, error="No source database connector available for write_query")

            from sqlalchemy import create_engine

            if self._write_engine is None:
                self._write_engine = create_engine(self.dataset_db_uri)

            result = read_connector.execute_query(sql, result_format="pandas")
            if not result.success:
                return FuncToolResult(success=0, error=result.error)

            df = result.sql_return
            schema = self.dataset_db_schema or None
            df.to_sql(table_name, self._write_engine, schema=schema, if_exists=if_exists, index=False)
            rows = len(df)
            result_data = {
                "table_name": table_name,
                "rows_written": rows,
                "schema": schema,
                "if_exists": if_exists,
            }
            database_id = self._resolve_dataset_db_id()
            if database_id is not None:
                result_data["database_id"] = database_id
            return FuncToolResult(result=result_data)
        except Exception as exc:
            logger.warning(f"write_query failed: {exc}")
            return FuncToolResult(success=0, error=f"write_query failed for table '{table_name}': {exc}")

    def _resolve_grafana_datasource_uid(self) -> Any:
        """Look up a pre-configured datasource in the BI platform and return its UID.

        Uses ``datasource_name`` (from agent.yml) to find the datasource.
        Falls back to matching by database name from ``dataset_db_uri``.
        """
        if self._grafana_ds_uid is not None:
            return self._grafana_ds_uid
        if not hasattr(self.adapter, "list_datasets"):
            return None
        try:
            datasets = self.adapter.list_datasets("")
            target_name = self.datasource_name
            # Try matching by configured datasource_name first
            if target_name:
                for ds in datasets:
                    name = ds.name if hasattr(ds, "name") else ""
                    if name == target_name:
                        ds_uid = (ds.extra or {}).get("grafana_ds", {}).get("uid") if hasattr(ds, "extra") else None
                        if ds_uid:
                            self._grafana_ds_uid = ds_uid
                            return ds_uid

            # Fallback: match by database name from dataset_db_uri
            if self.dataset_db_uri:
                from sqlalchemy.engine.url import make_url

                db_name = make_url(self.dataset_db_uri).database or ""
                if db_name:
                    for ds in datasets:
                        extra = (ds.extra or {}).get("grafana_ds", {}) if hasattr(ds, "extra") else {}
                        json_data = extra.get("jsonData", {})
                        if json_data.get("database") == db_name:
                            uid = extra.get("uid")
                            if uid:
                                self._grafana_ds_uid = uid
                                return uid
        except Exception as exc:
            logger.debug(f"Could not resolve Grafana datasource: {exc}")
        return None

    def _resolve_dataset_db_id(self) -> Any:
        """Look up the BI platform database ID that matches dataset_db by name.

        The database must be pre-registered in the BI platform (e.g. via Superset UI
        or admin scripts). This method only performs a lookup — it does not register.
        """
        if self._dataset_db_id is not None:
            return self._dataset_db_id
        try:
            from sqlalchemy.engine.url import make_url

            target_url = make_url(self.dataset_db_uri)
            target_db_name = target_url.database or ""
            if not target_db_name:
                return None

            databases = self.adapter.list_bi_databases()
            for db in databases:
                name = db.get("name", "") if isinstance(db, dict) else getattr(db, "name", "")
                if name == target_db_name:
                    db_id = db.get("id") if isinstance(db, dict) else getattr(db, "id", None)
                    self._dataset_db_id = db_id
                    return db_id

            logger.warning(
                f"Database '{target_db_name}' not found in BI platform. Please register it in the BI platform first."
            )
        except Exception as exc:
            logger.debug(f"Could not resolve dataset_db_id: {exc}")
        return None

    def _supports_chart_data(self) -> bool:
        adapter_method = getattr(type(self.adapter), "get_chart_data", None)
        if not callable(adapter_method):
            return False

        try:
            from datus_bi_core import BIAdapterBase

            base_method = getattr(BIAdapterBase, "get_chart_data", None)
        except Exception:
            base_method = None

        if not callable(base_method):
            return True
        return adapter_method is not base_method

    # ------------------------------------------------------------------ #
    # Dynamic tool registration
    # ------------------------------------------------------------------ #

    def available_tools(self) -> List[Tool]:
        """Return tools based on what capabilities the adapter supports."""
        # Try to import Mixin types from datus_bi_core
        try:
            from datus_bi_core import (
                ChartWriteMixin,
                DashboardWriteMixin,
                DatasetWriteMixin,
            )

            has_dash_write = isinstance(self.adapter, DashboardWriteMixin)
            has_chart_write = isinstance(self.adapter, ChartWriteMixin)
            has_dataset_write = isinstance(self.adapter, DatasetWriteMixin)
        except ImportError:
            # Fallback: check by method existence
            has_dash_write = hasattr(self.adapter, "create_dashboard")
            has_chart_write = hasattr(self.adapter, "create_chart")
            has_dataset_write = hasattr(self.adapter, "create_dataset")

        methods: List = [
            self.list_dashboards,
            self.get_dashboard,
            self.list_charts,
            self.get_chart,
            self.list_datasets,
        ]
        if self._supports_chart_data():
            methods.append(self.get_chart_data)

        if has_dash_write:
            methods += [self.create_dashboard, self.update_dashboard, self.delete_dashboard]
        if has_chart_write:
            methods += [self.create_chart, self.update_chart, self.add_chart_to_dashboard, self.delete_chart]
        if has_dataset_write:
            methods += [self.create_dataset, self.list_bi_databases, self.delete_dataset]

        if self.dataset_db_uri:
            methods.append(self.write_query)

        return [trans_to_function_tool(m) for m in methods]
