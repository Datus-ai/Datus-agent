# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""BIFuncTool: LLM function calling layer for BI adaptors."""

from __future__ import annotations

import re
from typing import Any, List

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

_VALID_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")
_VALID_IF_EXISTS = {"replace", "append", "fail"}

logger = get_logger(__name__)


class BIFuncTool:
    """
    LLM function calling layer for BI adaptors.

    Dynamically exposes tools based on adaptor capabilities:
    - All adaptors: list_dashboards, get_dashboard, list_charts, list_datasets
    - DashboardWriteMixin: create_dashboard, update_dashboard
    - ChartWriteMixin: create_chart, update_chart, add_chart_to_dashboard
    - DatasetWriteMixin: create_dataset, list_bi_databases
    - dataset_db_uri set: write_query (execute SQL on source DB and write result to dashboard DB)
    """

    def __init__(
        self, adaptor: Any, dataset_db_uri: str = "", dataset_db_schema: str = "", read_connector: Any = None
    ) -> None:
        self.adaptor = adaptor
        self._dataset_db_uri = dataset_db_uri
        self._dataset_db_schema = dataset_db_schema
        self._read_connector = read_connector
        self._write_engine = None  # lazy-initialized
        self._dataset_db_id = None  # lazy-resolved from BI platform

    # ------------------------------------------------------------------ #
    # Read operations (available on all adaptors)
    # ------------------------------------------------------------------ #

    def list_dashboards(self, search: str = "") -> FuncToolResult:
        """List dashboards in the BI platform. Optionally filter by search keyword."""
        try:
            if hasattr(self.adaptor, "list_dashboards"):
                results = self.adaptor.list_dashboards(search=search)
                return FuncToolResult(result=[r.model_dump() for r in results])
            return FuncToolResult(success=0, error="This adaptor does not support list_dashboards")
        except Exception as exc:
            logger.warning(f"list_dashboards failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def get_dashboard(self, dashboard_id: str) -> FuncToolResult:
        """Get detailed information about a specific dashboard by its ID."""
        try:
            result = self.adaptor.get_dashboard_info(dashboard_id)
            if result is None:
                return FuncToolResult(success=0, error=f"Dashboard {dashboard_id} not found")
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"get_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def list_charts(self, dashboard_id: str) -> FuncToolResult:
        """List all charts/panels in a dashboard."""
        try:
            results = self.adaptor.list_charts(dashboard_id)
            return FuncToolResult(result=[r.model_dump() for r in results])
        except Exception as exc:
            logger.warning(f"list_charts failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def list_datasets(self, dashboard_id: str = "") -> FuncToolResult:
        """List datasets available in the BI platform. For Superset, pass dashboard_id to scope results."""
        try:
            results = self.adaptor.list_datasets(dashboard_id)
            return FuncToolResult(result=[r.model_dump() for r in results])
        except Exception as exc:
            logger.warning(f"list_datasets failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    # ------------------------------------------------------------------ #
    # Dashboard write operations (DashboardWriteMixin)
    # ------------------------------------------------------------------ #

    def create_dashboard(self, title: str, description: str = "") -> FuncToolResult:
        """Create a new empty dashboard with the given title."""
        try:
            from datus_bi_core.models import DashboardSpec

            spec = DashboardSpec(title=title, description=description)
            result = self.adaptor.create_dashboard(spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"create_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def update_dashboard(self, dashboard_id: str, title: str = "", description: str = "") -> FuncToolResult:
        """Update an existing dashboard's title or description."""
        try:
            from datus_bi_core.models import DashboardSpec

            existing = self.adaptor.get_dashboard_info(dashboard_id)
            if existing is None:
                return FuncToolResult(success=0, error=f"Dashboard {dashboard_id} not found")
            spec = DashboardSpec(
                title=title or existing.name,
                description=description or (existing.description or ""),
            )
            result = self.adaptor.update_dashboard(dashboard_id, spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"update_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def delete_dashboard(self, dashboard_id: str) -> FuncToolResult:
        """Delete a dashboard by its ID."""
        try:
            success = self.adaptor.delete_dashboard(dashboard_id)
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
    ) -> FuncToolResult:
        """
        Create a new chart/panel. Requires a dataset — create one first with create_dataset().

        Args:
            chart_type: Type of chart: bar, line, pie, table, big_number, scatter
            title: Chart title
            dataset_id: (Required) Dataset ID from create_dataset()
            x_axis: Column name for x-axis or time column (for line/bar charts)
            metrics: Comma-separated metric expressions. Supported formats:
                     - "column_name" → defaults to SUM(column_name)
                     - "AVG(column_name)", "MAX(column_name)", "MIN(column_name)", "COUNT(column_name)"
                     Examples: "revenue,count" or "AVG(activity_count)" or "MAX(price),MIN(price)"
            dimensions: Comma-separated list of dimension/groupby column names
            dashboard_id: (Grafana: required) Dashboard ID to add the chart to
            description: Chart description
        """
        try:
            from datus_bi_core.models import ChartSpec

            metrics_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
            dims_list = [d.strip() for d in dimensions.split(",") if d.strip()] if dimensions else None
            ds_id = int(dataset_id) if dataset_id.strip().isdigit() else None
            if not ds_id:
                return FuncToolResult(
                    success=0,
                    error="dataset_id is required. Create a dataset first with create_dataset(), then use its ID here.",
                )
            spec = ChartSpec(
                chart_type=chart_type,
                title=title,
                description=description,
                dataset_id=ds_id,
                x_axis=x_axis or None,
                metrics=metrics_list,
                dimensions=dims_list,
            )
            dash_id = dashboard_id.strip() or None
            result = self.adaptor.create_chart(spec, dashboard_id=dash_id)
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
            existing = self.adaptor.get_chart(chart_id)
            if existing is None:
                return FuncToolResult(success=0, error=f"Chart {chart_id} not found")
            spec = ChartSpec(
                chart_type=chart_type or existing.chart_type,
                title=title or existing.name,
                description=description,
                sql=sql or None,
                x_axis=x_axis or None,
                metrics=metrics_list,
            )
            result = self.adaptor.update_chart(chart_id, spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"update_chart failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def add_chart_to_dashboard(self, chart_id: str, dashboard_id: str) -> FuncToolResult:
        """Add an existing chart to a dashboard."""
        try:
            success = self.adaptor.add_chart_to_dashboard(dashboard_id, chart_id)
            return FuncToolResult(result={"success": success, "chart_id": chart_id, "dashboard_id": dashboard_id})
        except Exception as exc:
            logger.warning(f"add_chart_to_dashboard failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def delete_chart(self, chart_id: str) -> FuncToolResult:
        """Delete a chart by its ID."""
        try:
            success = self.adaptor.delete_chart(chart_id)
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
            result = self.adaptor.create_dataset(spec)
            return FuncToolResult(result=result.model_dump())
        except Exception as exc:
            logger.warning(f"create_dataset failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def list_bi_databases(self) -> FuncToolResult:
        """List available database connections in the BI platform. Call this before create_dataset."""
        try:
            results = self.adaptor.list_bi_databases()
            return FuncToolResult(result=results)
        except Exception as exc:
            logger.warning(f"list_bi_databases failed: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def delete_dataset(self, dataset_id: str) -> FuncToolResult:
        """Delete a dataset by its ID."""
        try:
            success = self.adaptor.delete_dataset(dataset_id)
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
        if not self._dataset_db_uri:
            return FuncToolResult(success=0, error="dataset_db is not configured for this BI platform")
        if not _VALID_TABLE_NAME.match(table_name):
            return FuncToolResult(success=0, error="Invalid table_name: must match [a-zA-Z_][a-zA-Z0-9_]{0,62}")
        if if_exists not in _VALID_IF_EXISTS:
            return FuncToolResult(success=0, error=f"if_exists must be one of: {sorted(_VALID_IF_EXISTS)}")
        sql_stripped = sql.strip().upper()
        if not (sql_stripped.startswith("SELECT") or sql_stripped.startswith("WITH")):
            return FuncToolResult(success=0, error="Only SELECT/WITH queries are allowed in write_query")
        try:
            if self._read_connector is None:
                return FuncToolResult(success=0, error="No source database connector available for write_query")

            from sqlalchemy import create_engine

            if self._write_engine is None:
                self._write_engine = create_engine(self._dataset_db_uri)

            result = self._read_connector.execute_query(sql, result_format="pandas")
            if not result.success:
                return FuncToolResult(success=0, error=result.error)

            df = result.sql_return
            schema = self._dataset_db_schema or None
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

    def _resolve_dataset_db_id(self) -> Any:
        """Look up the BI platform database ID that matches dataset_db by name.

        The database must be pre-registered in the BI platform (e.g. via Superset UI
        or admin scripts). This method only performs a lookup — it does not register.
        """
        if self._dataset_db_id is not None:
            return self._dataset_db_id
        try:
            from sqlalchemy.engine.url import make_url

            target_url = make_url(self._dataset_db_uri)
            target_db_name = target_url.database or ""
            if not target_db_name:
                return None

            databases = self.adaptor.list_bi_databases()
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

    # ------------------------------------------------------------------ #
    # Dynamic tool registration
    # ------------------------------------------------------------------ #

    def available_tools(self) -> List[Tool]:
        """Return tools based on what capabilities the adaptor supports."""
        # Try to import Mixin types from datus_bi_core
        try:
            from datus_bi_core import (
                ChartWriteMixin,
                DashboardWriteMixin,
                DatasetWriteMixin,
                ListDashboardsMixin,
            )

            has_list = isinstance(self.adaptor, ListDashboardsMixin)
            has_dash_write = isinstance(self.adaptor, DashboardWriteMixin)
            has_chart_write = isinstance(self.adaptor, ChartWriteMixin)
            has_dataset_write = isinstance(self.adaptor, DatasetWriteMixin)
        except ImportError:
            # Fallback: check by method existence
            has_list = hasattr(self.adaptor, "list_dashboards")
            has_dash_write = hasattr(self.adaptor, "create_dashboard")
            has_chart_write = hasattr(self.adaptor, "create_chart")
            has_dataset_write = hasattr(self.adaptor, "create_dataset")

        methods: List = [self.get_dashboard, self.list_charts, self.list_datasets]

        if has_list:
            methods.insert(0, self.list_dashboards)

        if has_dash_write:
            methods += [self.create_dashboard, self.update_dashboard, self.delete_dashboard]
        if has_chart_write:
            methods += [self.create_chart, self.update_chart, self.add_chart_to_dashboard, self.delete_chart]
        if has_dataset_write:
            methods += [self.create_dataset, self.list_bi_databases, self.delete_dataset]

        if self._dataset_db_uri:
            methods.append(self.write_query)

        return [trans_to_function_tool(m) for m in methods]
