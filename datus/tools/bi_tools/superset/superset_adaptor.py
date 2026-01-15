# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, urlparse

import httpx

from datus.tools.bi_tools.base_adaptor import (
    AuthParam,
    AuthType,
    BIAdaptorBase,
    ChartInfo,
    ColumnInfo,
    DashboardInfo,
    DatasetInfo,
    DimensionDef,
    MetricDef,
    QuerySpec,
)
from datus.tools.bi_tools.superset.superset_util import build_query_context
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names

logger = get_logger(__name__)


class SupersetAdaptorError(RuntimeError):
    """Errors raised by the Superset adaptor."""


class SupersetAdaptor(BIAdaptorBase):
    """Adaptor that extracts chart SQL and metadata from Superset."""

    def __init__(
        self,
        api_base_url: str,
        auth_params: AuthParam,
        dialect: str,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(api_base_url, auth_params, dialect, timeout)

        self.auth_params = auth_params
        self._api_base = self._normalize_api_base(self.api_base_url)
        self._client = httpx.Client(
            timeout=timeout, verify=self.api_base_url.startswith("https://"), follow_redirects=True
        )
        self._owns_client = True

        self._auth_header_value: Optional[str] = None
        self._token_expiration: Optional[float] = None
        self._csrf_token: Optional[str] = None
        self._dataset_cache: Dict[str, DatasetInfo] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def platform_name(self) -> str:
        return "superset"

    def auth_type(self) -> AuthType:
        return AuthType.LOGIN

    def parse_dashboard_id(self, dashboard_url: str) -> Union[int, str]:
        stripped = (dashboard_url or "").strip()
        if not stripped:
            return ""

        if stripped.isdigit():
            return int(stripped)

        parsed = urlparse(stripped)
        if parsed.scheme and parsed.netloc:
            segments = [segment for segment in parsed.path.split("/") if segment]
            for segment in reversed(segments):
                if segment:
                    return segment

            query_params = parse_qs(parsed.query)
            for key in ("dashboard_id", "id"):
                values = query_params.get(key)
                if values:
                    return values[0]

        return stripped

    def get_dashboard_base_info(self, dashboard_url: str) -> DashboardInfo:
        dashboard_id = self.parse_dashboard_id(dashboard_url)
        dashboard_info = self.get_dashboard_info(dashboard_id)
        if dashboard_info is None:
            raise SupersetAdaptorError(f"Dashboard {dashboard_id} not found")
        return dashboard_info

    def get_dashboard_info(self, dashboard_id: Union[int, str]) -> Optional[DashboardInfo]:
        try:
            dashboard = self._get_dashboard(dashboard_id)
        except SupersetAdaptorError as exc:
            logger.warning(f"Failed to fetch dashboard {dashboard_id}: {exc}")
            raise exc

        name = dashboard.get("dashboard_title") or dashboard.get("title") or dashboard.get("slug") or str(dashboard_id)
        description = dashboard.get("description") or dashboard.get("description_markeddown")

        chart_ids: List[Union[int, str]] = []
        try:
            charts = self._get_dashboard_charts(dashboard_id)
            for chart_meta in charts:
                chart_id = self._extract_chart_id(chart_meta)
                if chart_id is not None:
                    chart_ids.append(chart_id)
        except SupersetAdaptorError as exc:
            logger.warning(f"Failed to fetch charts for dashboard {dashboard_id}: {exc}")

        if chart_ids:
            seen = set()
            deduped: List[Union[int, str]] = []
            for chart_id in chart_ids:
                if chart_id in seen:
                    continue
                seen.add(chart_id)
                deduped.append(chart_id)
            chart_ids = deduped

        return DashboardInfo(
            id=dashboard.get("id", dashboard_id),
            name=name,
            description=description,
            chart_ids=chart_ids,
            extra={"raw": dashboard},
        )

    def list_charts(self, dashboard_id: Union[int, str]) -> List[ChartInfo]:
        try:
            charts = self._get_dashboard_charts(dashboard_id)
        except SupersetAdaptorError as exc:
            logger.warning(f"Failed to list charts for dashboard {dashboard_id}: {exc}")
            return []

        results: List[ChartInfo] = []
        for chart_meta in charts:
            chart_id = self._extract_chart_id(chart_meta)
            if chart_id is None:
                logger.warning(f"Skip chart without chart_id: {chart_meta}")
                continue

            results.append(
                ChartInfo(
                    id=chart_id,
                    name=chart_meta.get("slice_name") or chart_meta.get("name") or str(chart_id),
                    description=self._chart_description(chart_meta, chart_detail=None),
                    chart_type=chart_meta.get("viz_type") or chart_meta.get("form_data", {}).get("viz_type"),
                )
            )
        return results

    def list_datasets(self, dashboard_id: Union[int, str]) -> List[DatasetInfo]:
        try:
            charts = self._get_dashboard_charts(dashboard_id)
        except SupersetAdaptorError as exc:
            logger.warning(f"Failed to list datasets for dashboard {dashboard_id}: {exc}")
            return []

        dataset_map: Dict[str, Dict[str, Any]] = {}
        for chart_meta in charts:
            form_data = _load_json_field(chart_meta.get("form_data")) or {}
            query_context = _load_json_field(chart_meta.get("query_context"))
            ref = self._extract_datasource_ref(
                form_data=form_data,
                chart_meta=chart_meta,
                query_context=query_context if isinstance(query_context, dict) else None,
            )
            if not ref:
                continue
            key = str(ref["id"])
            entry = dataset_map.setdefault(
                key,
                {
                    "ref": dict(ref),
                    "tables": [],
                    "metrics": [],
                    "dimensions": [],
                    "dialect": self.dialect,
                },
            )
            if ref.get("name") and not entry["ref"].get("name"):
                entry["ref"]["name"] = ref["name"]

            table_name = entry["ref"].get("name") or ref.get("name")

            dataset_block = chart_meta.get("dataset") if isinstance(chart_meta.get("dataset"), dict) else None
            if dataset_block:
                if not table_name:
                    table_name = dataset_block.get("table_name") or dataset_block.get("datasource_name")
                columns = self._parse_dataset_columns(dataset_block, table_name)
                entry["metrics"].extend(self._parse_dataset_metrics(dataset_block, table_name))
                entry["dimensions"].extend(self._parse_dataset_dimensions(dataset_block, table_name, columns))
                entry["tables"].extend(self._tables_from_sql(dataset_block.get("sql")))

            if isinstance(query_context, dict):
                query_metrics, query_dimensions = self._extract_query_metrics_dimensions(query_context, table_name)
                entry["metrics"].extend(query_metrics)
                entry["dimensions"].extend(query_dimensions)

            form_metrics, form_dimensions = self._extract_form_data_metrics_dimensions(form_data, table_name)
            entry["metrics"].extend(form_metrics)
            entry["dimensions"].extend(form_dimensions)

        datasets: List[DatasetInfo] = []
        for entry in dataset_map.values():
            ref = entry["ref"]
            dataset_id = ref["id"]
            dataset_name = ref.get("name") or str(dataset_id)
            tables = self._dedupe_tables(entry["tables"])
            metrics = self._dedupe_metrics(entry["metrics"])
            dimensions = self._dedupe_dimensions(entry["dimensions"])
            datasets.append(
                DatasetInfo(
                    id=dataset_id,
                    name=dataset_name,
                    dialect=entry["dialect"],
                    tables=tables or None,
                    metrics=metrics or None,
                    dimensions=dimensions or None,
                    extra={"datasource": ref},
                )
            )
        return datasets

    def get_chart(self, chart_id: Union[int, str], dashboard_id: Union[int, str, None] = None) -> Optional[ChartInfo]:
        try:
            chart_detail = self._get_chart(chart_id)
        except SupersetAdaptorError as exc:
            logger.warning(f"Failed to fetch chart {chart_id}: {exc}")
            return None
        dataset_info = chart_detail.get("dataset")
        if not isinstance(dataset_info, dict):
            dataset_info = None
        outer_form_data = _load_json_field(chart_detail.get("form_data"))
        if outer_form_data is None:
            outer_form_data = _load_json_field(chart_detail.get("params"))
        if not isinstance(outer_form_data, dict):
            outer_form_data = None

        slice_detail = chart_detail.get("slice") or chart_detail
        if slice_detail is not chart_detail:
            chart_detail = slice_detail

        slice_form_data = _load_json_field(slice_detail.get("form_data"))
        if slice_form_data is None:
            slice_form_data = _load_json_field(slice_detail.get("params"))
        if not isinstance(slice_form_data, dict):
            slice_form_data = None

        def _matches_chart(fd: Optional[Dict[str, Any]]) -> bool:
            if not isinstance(fd, dict):
                return False
            chart_key = slice_detail.get("slice_id") or slice_detail.get("id") or chart_id
            if chart_key is None:
                return False
            candidate = fd.get("slice_id") or fd.get("chart_id") or fd.get("id")
            if candidate is None:
                return False
            return str(candidate) == str(chart_key)

        form_data: Dict[str, Any] = {}
        if _matches_chart(outer_form_data):
            if slice_form_data:
                form_data.update(slice_form_data)
            form_data.update(outer_form_data)
        elif slice_form_data:
            form_data = slice_form_data
        elif outer_form_data:
            form_data = outer_form_data
        query_context = self._extract_query_context(chart_detail, form_data, dataset_info)

        if not dataset_info:
            datasource_ref = self._extract_datasource_ref(
                form_data=form_data, chart_detail=chart_detail, query_context=query_context
            )
            tables = self._resolve_tables(datasource_ref)
        else:
            datasource_ref = None
            tables = self._tables_from_sql(dataset_info.get("sql"))
        # IMPORTANT: avoid hard-binding metrics/dimensions to a physical table name.
        # Superset datasets can be virtual (SQL-based) and datasource names/aliases may drift.
        table_name_for_tagging: Optional[str] = None

        sqls: List[str] = []
        used_query_indexes: Optional[set[int]] = None
        try:
            sqls, used_query_indexes = self._collect_sql_from_chart(
                chart_id,
                query_context,
            )
        except SupersetAdaptorError as exc:
            logger.warning(f"Failed to fetch SQL for chart {chart_id}: {exc}")

        metrics: List[MetricDef] = []
        dimensions: List[DimensionDef] = []
        if query_context:
            metrics, dimensions = self._extract_query_metrics_dimensions(
                query_context, table_name_for_tagging, only_query_indexes=used_query_indexes
            )

        if sqls:
            for sql_text in sqls:
                tables.extend(self._tables_from_sql(sql_text))
        tables = self._dedupe_tables(tables)

        query_spec: Optional[QuerySpec] = None
        if query_context or sqls:
            payload: Dict[str, Any] = {}
            if query_context:
                payload["query_context"] = query_context
            if form_data:
                payload["form_data"] = form_data
            if datasource_ref:
                payload["datasource"] = datasource_ref

            query_spec = QuerySpec(
                kind="sql" if sqls else "semantic",
                payload=payload,
                sql=sqls,
                tables=tables or None,
                metrics=metrics or None,
                dimensions=dimensions or None,
            )

        return ChartInfo(
            id=chart_detail.get("id", chart_id),
            name=chart_detail.get("slice_name") or chart_detail.get("name") or str(chart_id),
            description=self._chart_description(None, chart_detail=chart_detail),
            query=query_spec,
            chart_type=chart_detail.get("viz_type") or chart_detail.get("chart_type") or form_data.get("viz_type"),
            extra={"raw": chart_detail, "datasource": datasource_ref},
        )

    def get_dataset(
        self, dataset_id: Union[int, str], dashboard_id: Union[int, str, None] = None
    ) -> Optional[DatasetInfo]:
        if dataset_id is None:
            return None
        cache_key = str(dataset_id)
        cached = self._dataset_cache.get(cache_key)
        if cached:
            return cached

        try:
            data = self._request_json("GET", f"dataset/{dataset_id}")
        except SupersetAdaptorError as exc:
            logger.warning(f"Failed to fetch dataset {dataset_id}: {exc}")
            return None

        dataset = data.get("result", data)
        resolved_id = dataset.get("id", dataset_id)
        name = (
            dataset.get("table_name")
            or dataset.get("datasource_name")
            or dataset.get("name")
            or dataset.get("verbose_name")
            or str(resolved_id)
        )
        description = dataset.get("description") or dataset.get("description_markeddown")

        table_name = dataset.get("table_name") or dataset.get("datasource_name") or name
        columns = self._parse_dataset_columns(dataset, table_name)
        metrics = self._parse_dataset_metrics(dataset, table_name)
        dimensions = self._parse_dataset_dimensions(dataset, table_name, columns)
        tables = self._tables_from_sql(dataset.get("sql"))
        tables = self._dedupe_tables(tables)

        extra = {
            "database": dataset.get("database"),
            "schema": dataset.get("schema"),
            "sql": dataset.get("sql"),
            "extra": dataset.get("extra"),
        }

        dataset_info = DatasetInfo(
            id=resolved_id,
            name=name,
            dialect=self.dialect,
            description=description,
            tables=tables or None,
            columns=columns or None,
            metrics=metrics or None,
            dimensions=dimensions or None,
            extra=extra,
        )
        self._dataset_cache[cache_key] = dataset_info
        return dataset_info

    def extract_dashboard_id(self, dashboard_url: str) -> Union[int, str]:
        return self.parse_dashboard_id(dashboard_url)

    def _chart_description(
        self, chart_meta: Optional[Dict[str, Any]], chart_detail: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        description = None
        if chart_meta:
            description = chart_meta.get("description") or chart_meta.get("description_markeddown")
        if not description and chart_detail:
            description = chart_detail.get("description") or chart_detail.get("description_markeddown")
        return description

    def _extract_query_context(
        self, chart_detail: Dict[str, Any], form_data: Dict[str, Any], dataset_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        raw_context = chart_detail.get("query_context")

        if raw_context:
            parsed_context = _load_json_field(raw_context)
        else:
            parsed_context = build_query_context(form_data=form_data)

        if isinstance(parsed_context, dict):
            _normalize_series_columns_in_query_context(parsed_context)
            return parsed_context
        return None

    def _collect_sql_from_chart(
        self,
        chart_id: Union[str, int],
        query_context: Optional[Dict[str, Any]],
    ) -> tuple[List[str], Optional[set[int]]]:
        """
        Collect SQL from chart using either chart/data API (new) or explore_json (legacy).

        API selection logic:
        1. If viz_type is a legacy chart type -> use explore_json directly
        2. If query_context is available -> try chart/data API first
        3. Fallback to explore_json if chart/data fails or returns empty

        :param chart_id: The chart identifier
        :param query_context: Query context for new charts (uses /api/v1/chart/data)
        :return: Tuple of (sql_list, used_query_indexes)
        """
        # For modern viz types, try chart/data API first
        sqls = []
        query_indexes = None
        if query_context:
            try:
                sqls, query_indexes = self._collect_sql_via_chart_data(chart_id, query_context)
            except SupersetAdaptorError as exc:
                logger.warning(f"chart/data failed for {chart_id}, trying explore_json: {exc}")

        if sqls:
            return sqls, query_indexes

        logger.debug(f"No sqls for chart {chart_id}")
        return [], None

    def _collect_sql_via_chart_data(
        self,
        chart_id: Union[str, int],
        query_context: Dict[str, Any],
    ) -> tuple[List[str], Optional[set[int]]]:
        """Collect SQL using /api/v1/chart/data endpoint (new API)."""
        payload = dict(query_context)
        _normalize_series_columns_in_query_context(payload)
        payload.setdefault("result_format", "json")
        payload.setdefault("result_type", "query")

        try:
            response_data = self._request_json("POST", "chart/data", json=payload)
        except SupersetAdaptorError as exc:
            raise SupersetAdaptorError(f"chart/data failed for {chart_id}: {exc}") from exc

        sqls: List[str] = []
        used_query_indexes: set[int] = set()
        results = response_data.get("result", [])

        for idx, block in enumerate(results):
            before = len(sqls)
            self._append_sql_from_block(block, sqls)
            if len(sqls) > before:
                used_query_indexes.add(idx)

        return sqls, (used_query_indexes or None)

    def _append_sql_from_block(self, block: Dict[str, Any], sqls: List[str]) -> None:
        if sql_text := block.get("query"):
            sqls.append(sql_text.strip())

    def _get_dashboard(self, dashboard_id: Union[str, int]) -> Dict[str, Any]:
        data = self._request_json("GET", f"dashboard/{dashboard_id}")
        if "result" in data and isinstance(data["result"], dict):
            return data["result"]
        return data

    def _get_dashboard_charts(self, dashboard_id: Union[str, int]) -> List[Dict[str, Any]]:
        data = self._request_json("GET", f"dashboard/{dashboard_id}/charts")
        charts = data.get("result", data)
        if not isinstance(charts, list):
            raise SupersetAdaptorError(f"Unexpected charts payload: {charts}")
        return charts

    def _get_chart(self, chart_id: Union[str, int]) -> Dict[str, Any]:
        """
        call `/api/v1/explore` ->
            result ->
                dataset
                form_data
                metadata
                slice ->
                    form_data
                    query_context
                    slice_name
        :param chart_id:
        :return:
        """
        data = self._request_json("GET", f"explore/?slice_id={chart_id}")
        chart = data.get("result", data)
        return chart

    def _normalize_api_base(self, api_base_url: str) -> str:
        base = (api_base_url or "").rstrip("/")
        if base.endswith("/api/v1"):
            return base
        return f"{base}/api/v1"

    def _extract_chart_id(self, chart_meta: Dict[str, Any]) -> Optional[Union[int, str]]:
        form_data = _load_json_field(chart_meta.get("form_data")) or {}
        chart_id = (
            form_data.get("slice_id")
            or chart_meta.get("slice_id")
            or chart_meta.get("chart_id")
            or chart_meta.get("id")
        )
        return chart_id

    def _extract_datasource_ref(
        self,
        form_data: Optional[Dict[str, Any]] = None,
        chart_meta: Optional[Dict[str, Any]] = None,
        chart_detail: Optional[Dict[str, Any]] = None,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        ds_id: Optional[Union[int, str]] = None
        ds_type: Optional[str] = None
        name: Optional[str] = None

        if query_context:
            datasource = query_context.get("datasource")
            if isinstance(datasource, dict):
                # parse dataset id
                ds_id = ds_id or _coerce_id(datasource.get("id"))
                ds_type = ds_type or datasource.get("type") or datasource.get("datasource_type")
                name = (
                    name or datasource.get("name") or datasource.get("datasource_name") or datasource.get("table_name")
                )

        if chart_detail:
            if not ds_id:
                ds_id = chart_detail.get("datasource_id") or chart_detail.get("dataset_id")
                ds_type = ds_type or chart_detail.get("datasource_type")
            if not name:
                dataset_block = chart_detail.get("dataset") if isinstance(chart_detail.get("dataset"), dict) else {}
                for candidate in (
                    chart_detail.get("datasource_name"),
                    chart_detail.get("table_name"),
                    dataset_block.get("datasource_name"),
                    dataset_block.get("table_name"),
                ):
                    if candidate:
                        name = candidate
                        break

        if chart_meta:
            ds_id = ds_id or chart_meta.get("datasource_id") or chart_meta.get("dataset_id")
            ds_type = ds_type or chart_meta.get("datasource_type")

        if form_data:
            ds_id = ds_id or form_data.get("datasource_id") or form_data.get("dataset_id")
            ds_type = ds_type or form_data.get("datasource_type")
            if not ds_id and "datasource" in form_data:
                parsed_id, parsed_type = _parse_datasource_value(form_data.get("datasource"))
                ds_id = parsed_id or ds_id
                ds_type = parsed_type or ds_type

        ds_id = _coerce_id(ds_id)
        if ds_id is None:
            return None

        return {"id": ds_id, "type": ds_type}

    def _resolve_tables(self, datasource_ref: Optional[Dict[str, Any]]) -> List[str]:
        if not datasource_ref:
            return []

        dataset_id = datasource_ref.get("id")
        dataset_info = self.get_dataset(dataset_id) if dataset_id is not None else None
        return list(dataset_info.tables or []) if dataset_info else []

    def _tables_from_sql(self, sql: Optional[str]) -> List[str]:
        if not sql:
            return []
        read_dialect = self.dialect
        return [name for name in extract_table_names(sql, read_dialect, ignore_empty=True) if name]

    def _extract_query_metrics_dimensions(
        self,
        query_context: Dict[str, Any],
        table_name: Optional[str],
        only_query_indexes: Optional[set[int]] = None,
    ) -> tuple[List[MetricDef], List[DimensionDef]]:
        metrics: List[MetricDef] = []
        dimensions: List[DimensionDef] = []

        queries = query_context.get("queries") or []
        if not isinstance(queries, list):
            return metrics, dimensions

        for idx, query in enumerate(queries):
            if only_query_indexes is not None and idx not in only_query_indexes:
                continue
            if not isinstance(query, dict):
                continue
            metrics.extend(self._metrics_from_query(query, table_name))
            dimensions.extend(self._dimensions_from_query(query, table_name))

        return self._dedupe_metrics(metrics), self._dedupe_dimensions(dimensions)

    def _extract_form_data_metrics_dimensions(
        self, form_data: Dict[str, Any], table_name: Optional[str]
    ) -> tuple[List[MetricDef], List[DimensionDef]]:
        metrics: List[MetricDef] = []
        dimensions: List[DimensionDef] = []

        raw_metrics = form_data.get("metrics")
        if raw_metrics is None and "metric" in form_data:
            raw_metrics = [form_data.get("metric")]
        if isinstance(raw_metrics, list):
            for metric in raw_metrics:
                metric_def = self._normalize_metric(metric, table_name, origin="chart")
                if metric_def:
                    metrics.append(metric_def)
        elif raw_metrics:
            metric_def = self._normalize_metric(raw_metrics, table_name, origin="chart")
            if metric_def:
                metrics.append(metric_def)

        raw_dimensions: List[Any] = []
        for key in ("groupby", "columns", "all_columns"):
            items = form_data.get(key)
            if isinstance(items, list):
                raw_dimensions.extend(items)
        for key in ("granularity", "granularity_sqla", "time_column", "time_col"):
            value = form_data.get(key)
            if value:
                raw_dimensions.append(value)

        for item in raw_dimensions:
            dim = self._normalize_dimension(item, table_name, origin="chart")
            if dim:
                dimensions.append(dim)

        return self._dedupe_metrics(metrics), self._dedupe_dimensions(dimensions)

    def _metrics_from_query(self, query: Dict[str, Any], table_name: Optional[str]) -> List[MetricDef]:
        metrics: List[MetricDef] = []
        for metric in query.get("metrics") or []:
            metric_def = self._normalize_metric(metric, table_name, origin="chart")
            if metric_def:
                metrics.append(metric_def)
        return metrics

    def _dimensions_from_query(self, query: Dict[str, Any], table_name: Optional[str]) -> List[DimensionDef]:
        dimensions: List[DimensionDef] = []
        for key in ("groupby", "columns"):
            items = query.get(key) or []
            if not isinstance(items, list):
                continue
            for item in items:
                dim = self._normalize_dimension(item, table_name, origin="chart")
                if dim:
                    dimensions.append(dim)

        time_column = query.get("time_column") or query.get("time_col")
        if time_column:
            dim = self._normalize_dimension(time_column, table_name, origin="chart")
            if dim:
                dimensions.append(dim)

        return dimensions

    def _normalize_metric(self, metric: Any, table_name: Optional[str], origin: str) -> Optional[MetricDef]:
        if isinstance(metric, str):
            name = metric.strip()
            if not name:
                return None
            return MetricDef(name=name, expression=name, table=table_name, origin=origin)

        if not isinstance(metric, dict):
            return None

        name = metric.get("label") or metric.get("metric_name") or metric.get("name")
        description = metric.get("description") or metric.get("verbose_name")

        expression = metric.get("expression") or metric.get("sqlExpression") or metric.get("sql_expression")
        if not expression and metric.get("expressionType") == "SIMPLE":
            aggregate = metric.get("aggregate") or metric.get("aggregation")
            column = metric.get("column") or {}
            column_name = None
            if isinstance(column, dict):
                column_name = column.get("column_name") or column.get("name")
            elif isinstance(column, str):
                column_name = column
            if aggregate and column_name:
                expression = f"{aggregate}({column_name})"

        if not expression and metric.get("expressionType") == "SQL":
            expression = metric.get("sqlExpression")

        if not name and expression:
            name = expression
        if not name:
            return None

        return MetricDef(
            name=str(name),
            expression=str(expression or name),
            table=table_name,
            description=description,
            origin=origin,
            extra=metric,
        )

    def _normalize_dimension(self, item: Any, table_name: Optional[str], origin: str) -> Optional[DimensionDef]:
        if isinstance(item, str):
            name = item.strip()
            if not name:
                return None
            return DimensionDef(name=name, table=table_name, origin=origin)

        if not isinstance(item, dict):
            return None

        name = item.get("column_name") or item.get("name") or item.get("label") or item.get("verbose_name")
        if not name:
            return None

        title = item.get("verbose_name") or item.get("label")
        data_type = item.get("type") or item.get("data_type") or item.get("datatype")
        description = item.get("description")

        return DimensionDef(
            name=str(name),
            title=title,
            data_type=data_type,
            table=table_name,
            description=description,
            origin=origin,
            extra=item,
        )

    def _dedupe_metrics(self, metrics: List[MetricDef]) -> List[MetricDef]:
        seen = set()
        unique: List[MetricDef] = []
        for metric in metrics:
            # Table binding is often ambiguous in Superset (virtual datasets / aliases), so dedupe by semantic identity.
            key = (metric.name, metric.expression)
            if key in seen:
                continue
            seen.add(key)
            unique.append(metric)
        return unique

    def _dedupe_dimensions(self, dimensions: List[DimensionDef]) -> List[DimensionDef]:
        seen = set()
        unique: List[DimensionDef] = []
        for dimension in dimensions:
            # Dedupe primarily by column name; table binding may be unstable for virtual datasets.
            key = (dimension.name,)
            if key in seen:
                continue
            seen.add(key)
            unique.append(dimension)
        return unique

    def _dedupe_tables(self, tables: List[str]) -> List[str]:
        seen = set()
        unique: List[str] = []
        for table in tables:
            if not table:
                continue
            name = str(table)
            if name in seen:
                continue
            seen.add(name)
            unique.append(name)
        return unique

    def _parse_dataset_columns(self, dataset: Dict[str, Any], table_name: Optional[str]) -> List[ColumnInfo]:
        columns: List[ColumnInfo] = []
        for column in dataset.get("columns") or []:
            if not isinstance(column, dict):
                continue
            name = column.get("column_name") or column.get("name")
            if not name:
                continue
            columns.append(
                ColumnInfo(
                    name=str(name),
                    data_type=column.get("type") or column.get("data_type"),
                    description=column.get("description"),
                    table=table_name,
                    extra=column,
                )
            )
        return columns

    def _parse_dataset_metrics(self, dataset: Dict[str, Any], table_name: Optional[str]) -> List[MetricDef]:
        metrics: List[MetricDef] = []
        for metric in dataset.get("metrics") or []:
            metric_def = self._normalize_metric(metric, table_name, origin="dataset")
            if metric_def:
                metrics.append(metric_def)
        return self._dedupe_metrics(metrics)

    def _parse_dataset_dimensions(
        self, dataset: Dict[str, Any], table_name: Optional[str], columns: List[ColumnInfo]
    ) -> List[DimensionDef]:
        raw_columns = dataset.get("columns") or []
        has_flags = any(
            isinstance(col, dict) and any(key in col for key in ("groupby", "filterable", "is_dttm"))
            for col in raw_columns
        )

        dimensions: List[DimensionDef] = []
        if raw_columns:
            for column in raw_columns:
                if not isinstance(column, dict):
                    continue
                if has_flags and not (column.get("groupby") or column.get("filterable") or column.get("is_dttm")):
                    continue
                dim = self._normalize_dimension(column, table_name, origin="dataset")
                if dim:
                    dimensions.append(dim)

        if not dimensions and columns:
            for column in columns:
                dimensions.append(
                    DimensionDef(
                        name=column.name,
                        title=column.name,
                        data_type=column.data_type,
                        table=column.table,
                        description=column.description,
                        origin="dataset",
                    )
                )

        return self._dedupe_dimensions(dimensions)

    def _request_json(self, method: str, endpoint: str, need_csrf_token: bool = False, **kwargs) -> Dict[str, Any]:
        response = self._request(method, endpoint, need_csrf_token=need_csrf_token, **kwargs)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise SupersetAdaptorError(f"Invalid JSON response for {endpoint}: {exc}") from exc

    def _request(
        self, method: str, endpoint: str, require_auth: bool = True, need_csrf_token: bool = False, **kwargs
    ) -> httpx.Response:
        url = f"{self._api_base}/{endpoint.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        if require_auth:
            self._ensure_authenticated()
            headers.update(self._auth_headers())
            if need_csrf_token:
                token = self._get_csrf_token()
                headers["X-Csrftoken"] = token

        try:
            response = self._client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise SupersetAdaptorError(
                f"Superset API {method} {endpoint} failed with {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SupersetAdaptorError(f"Superset API {method} {endpoint} failed: {exc}") from exc

    def _auth_headers(self) -> Dict[str, str]:
        if not self._auth_header_value:
            return {}
        return {"Authorization": self._auth_header_value}

    def _ensure_authenticated(self) -> None:
        if self._auth_header_value and self._token_expiration and time.time() < self._token_expiration:
            return
        self._authenticate()

    def _authenticate(self) -> None:
        payload = {
            "username": self.auth_params.username,
            "password": self.auth_params.password,
            "refresh": True,
            "provider": "db" if not self.auth_params.extra else self.auth_params.extra.get("provider") or "db",
        }
        try:
            response = self._request("POST", "security/login", require_auth=False, json=payload)
        except SupersetAdaptorError as exc:
            raise SupersetAdaptorError(f"Authentication failed: {exc}") from exc

        data = response.json()
        token_payload = data.get("result", data)
        access_token = token_payload.get("access_token")
        token_type = token_payload.get("token_type", "Bearer")
        expires_in = token_payload.get("expires_in")

        if not access_token:
            raise SupersetAdaptorError("Superset login response missing access_token")

        self._auth_header_value = f"{token_type} {access_token}".strip()
        if isinstance(expires_in, (int, float)) and expires_in > 0:
            self._token_expiration = time.time() + expires_in - 60
        else:
            self._token_expiration = time.time() + 3600

        # Clear CSRF token when re-authenticating
        self._csrf_token = None

    def _get_csrf_token(self) -> str:
        """Get CSRF token for legacy API endpoints like explore_json."""
        if self._csrf_token:
            return self._csrf_token

        self._ensure_authenticated()
        try:
            response = self._request_json("GET", "security/csrf_token/", need_csrf_token=False)
            self._csrf_token = response.get("result")
            if not self._csrf_token:
                raise SupersetAdaptorError("CSRF token not found in response")
            return self._csrf_token
        except SupersetAdaptorError as exc:
            raise SupersetAdaptorError(f"Failed to get CSRF token: {exc}") from exc


def _normalize_series_columns_in_query_context(query_context: Dict[str, Any]) -> None:
    queries = query_context.get("queries")
    if not isinstance(queries, list):
        return
    for query in queries:
        if not isinstance(query, dict):
            continue
        _normalize_series_columns_in_query(query)


def _normalize_series_columns_in_query(query: Dict[str, Any]) -> None:
    series_columns = query.get("series_columns")
    if series_columns is None:
        return

    def _ensure_list(value: Any) -> Optional[List[Any]]:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]

    def _column_name(item: Any) -> Optional[str]:
        if isinstance(item, dict):
            return item.get("column_name") or item.get("name") or item.get("label")
        if item is None:
            return None
        return str(item)

    def _dedupe(items: List[str]) -> List[str]:
        seen: set[str] = set()
        unique: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    series_list = _ensure_list(series_columns)
    if series_list is None:
        return

    series_names = _dedupe([name for item in series_list if (name := _column_name(item))])

    columns_list = _ensure_list(query.get("columns")) or []
    column_names = _dedupe([name for item in columns_list if (name := _column_name(item))])

    if series_names:
        for name in series_names:
            if name not in column_names:
                column_names.append(name)
        query["columns"] = column_names
    elif columns_list:
        query["columns"] = columns_list

    query["series_columns"] = series_names


def _parse_datasource_value(value: Any) -> tuple[Optional[Union[int, str]], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        return _coerce_id(value.get("id")), value.get("type") or value.get("datasource_type")
    if isinstance(value, int):
        return value, None
    if isinstance(value, str):
        if "__" in value:
            id_part, type_part = value.split("__", 1)
            return _coerce_id(id_part), type_part
        if value.isdigit():
            return int(value), None
    return None, None


def _coerce_id(value: Any) -> Optional[Union[int, str]]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _load_json_field(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.debug(f"Failed to decode JSON field: {value[:128]}")
    return None


SupersetAdaptor.register("superset", auth_type=AuthType.LOGIN, display_name="Superset")
