# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from datus.tools.bi_tools.base_adaptor import BIAdaptorBase, ChartInfo, DashboardInfo, DatasetInfo
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names, metadata_identifier, normalize_sql

logger = get_logger(__name__)


@dataclass(slots=True)
class SelectedSqlCandidate:
    chart_id: Union[int, str]
    chart_name: str
    description: Optional[str]
    sql: str
    index: int = 0

    def question(self) -> str:
        if self.chart_name and self.description:
            return f"{self.chart_name} - {self.description}"
        return self.chart_name or self.description or ""

    def to_payload(self) -> Dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "chart_name": self.chart_name,
            "description": self.description,
            "sql": self.sql,
            "index": self.index,
            "question": self.question(),
        }


@dataclass(slots=True)
class ChartSelection:
    chart: ChartInfo
    sql_indices: List[int] = field(default_factory=list)


@dataclass(slots=True)
class DashboardExtraction:
    dashboard_id: Union[int, str]
    dashboard: DashboardInfo
    charts: List[ChartInfo]
    datasets: List[DatasetInfo]

    def chart_map(self) -> Dict[str, ChartInfo]:
        return {str(chart.id): chart for chart in self.charts}

    def dataset_map(self) -> Dict[str, DatasetInfo]:
        return {str(dataset.id): dataset for dataset in self.datasets}


@dataclass(slots=True)
class DashboardAssemblyResult:
    dashboard: DashboardInfo
    charts: List[ChartInfo]
    datasets: List[DatasetInfo]
    reference_sqls: List[SelectedSqlCandidate]
    metric_sqls: List[SelectedSqlCandidate]
    tables: List[str]


class DashboardAssembler:
    def __init__(
        self,
        adaptor: BIAdaptorBase,
        default_dialect: Optional[str] = None,
        default_catalog: str = "",
        default_database: str = "",
        default_schema: str = "",
    ) -> None:
        self.adaptor = adaptor
        self.default_dialect = default_dialect
        self.default_catalog = (default_catalog or "").strip()
        self.default_database = (default_database or "").strip()
        self.default_schema = (default_schema or "").strip()

    def extract_dashboard(self, dashboard_url: str) -> DashboardExtraction:
        dashboard_id = self.adaptor.parse_dashboard_id(dashboard_url)
        dashboard = self.adaptor.get_dashboard_info(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard {dashboard_id} not found")

        charts = self._load_charts(dashboard_id, dashboard)
        datasets = self.adaptor.list_datasets(dashboard_id)

        return DashboardExtraction(
            dashboard_id=dashboard_id,
            dashboard=dashboard,
            charts=charts,
            datasets=datasets,
        )

    def hydrate_datasets(
        self, datasets: Sequence[DatasetInfo], dashboard_id: Union[int, str, None] = None
    ) -> List[DatasetInfo]:
        hydrated: List[DatasetInfo] = []
        for dataset in datasets:
            try:
                detail = self.adaptor.get_dataset(dataset.id, dashboard_id)
            except Exception as exc:
                logger.warning("Failed to fetch dataset %s: %s", dataset.id, exc)
                detail = None
            hydrated.append(detail or dataset)
        return hydrated

    def assemble(
        self,
        dashboard: DashboardInfo,
        chart_selections_ref: Sequence[ChartSelection],
        chart_selections_metrics: Sequence[ChartSelection],
        datasets: Sequence[DatasetInfo],
    ) -> DashboardAssemblyResult:
        selected_charts = [selection.chart for selection in chart_selections_ref]
        dataset_by_id = {str(dataset.id): dataset for dataset in datasets}
        dataset_by_name = {dataset.name: dataset for dataset in datasets if dataset.name}

        tables: List[str] = []
        reference_sqls: List[SelectedSqlCandidate] = self._parse_to_sql_candidate(
            chart_selections_ref, dataset_by_id, dataset_by_name, tables
        )
        metric_sqls: List[SelectedSqlCandidate] = self._parse_to_sql_candidate(
            chart_selections_metrics, dataset_by_id, dataset_by_name, tables
        )

        tables = self._dedupe_tables(tables)

        return DashboardAssemblyResult(
            dashboard=dashboard,
            charts=selected_charts,
            datasets=list(datasets),
            reference_sqls=reference_sqls,
            metric_sqls=metric_sqls,
            tables=tables,
        )

    def _parse_to_sql_candidate(
        self,
        chart_selections: Sequence[ChartSelection],
        dataset_by_id: Dict[str, DatasetInfo],
        dataset_by_name: Dict[str, DatasetInfo],
        tables: List[str],
    ) -> List[SelectedSqlCandidate]:
        result: List[SelectedSqlCandidate] = []
        for selection in chart_selections:
            chart = selection.chart
            query = chart.query
            dataset = self._resolve_chart_dataset(chart, dataset_by_id, dataset_by_name)
            sqls = list(query.sql or []) if query else []
            indices = selection.sql_indices or list(range(len(sqls)))
            indices = [idx for idx in indices if 0 <= idx < len(sqls)]

            if query:
                if query.tables:
                    tables.extend(self._normalize_tables(query.tables, dataset=dataset))

            for idx in indices:
                sql_text = sqls[idx]
                result.append(
                    SelectedSqlCandidate(
                        chart_id=chart.id,
                        chart_name=chart.name,
                        description=chart.description,
                        sql=sql_text,
                        index=idx,
                    )
                )
                raw_tables = self._tables_from_sql(sql_text)
                tables.extend(self._normalize_tables(raw_tables, dataset=dataset))
        return result

    def _load_charts(self, dashboard_id: Union[int, str], dashboard: DashboardInfo) -> List[ChartInfo]:
        chart_metas = self.adaptor.list_charts(dashboard_id)
        chart_meta_map = {str(chart.id): chart for chart in chart_metas}

        chart_ids = list(dashboard.chart_ids or [])
        if not chart_ids:
            chart_ids = [chart.id for chart in chart_metas]

        charts: List[ChartInfo] = []
        for chart_id in chart_ids:
            try:
                chart = self.adaptor.get_chart(chart_id, dashboard_id)
            except Exception as exc:
                logger.warning("Failed to fetch chart %s: %s", chart_id, exc)
                chart = None
            if chart is None:
                fallback = chart_meta_map.get(str(chart_id))
                if fallback:
                    charts.append(fallback)
                continue
            charts.append(chart)
        return charts

    def _resolve_chart_dataset(
        self,
        chart: ChartInfo,
        dataset_by_id: Dict[str, DatasetInfo],
        dataset_by_name: Dict[str, DatasetInfo],
    ) -> Optional[DatasetInfo]:
        datasource: Optional[Dict[str, Any]] = None
        if chart.query and isinstance(chart.query.payload, dict):
            payload = chart.query.payload
            datasource = payload.get("datasource") if isinstance(payload.get("datasource"), dict) else None
        if not datasource and isinstance(chart.extra, dict):
            extra_source = chart.extra.get("datasource")
            datasource = extra_source if isinstance(extra_source, dict) else None

        if datasource:
            ds_id = datasource.get("id")
            if ds_id is not None:
                return dataset_by_id.get(str(ds_id))
            ds_name = datasource.get("name")
            if ds_name:
                return dataset_by_name.get(ds_name)
        return None

    def _tables_from_sql(self, sql: str) -> List[str]:
        if not sql:
            return []
        table_names = extract_table_names(sql, self.default_dialect, ignore_empty=True)
        return [name for name in table_names if name]

    def _normalize_tables(self, tables: Iterable[str], dataset: Optional[DatasetInfo] = None) -> List[str]:
        catalog_name, database_name, schema_name = self._resolve_table_context(dataset)
        normalized: List[str] = []
        for table in tables:
            raw = (table or "").strip()
            if not raw:
                continue
            if "." in raw:
                normalized.append(raw)
                continue
            if not self._can_qualify_table(self.default_dialect, catalog_name, database_name, schema_name):
                normalized.append(raw)
                continue
            normalized.append(
                metadata_identifier(
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                    table_name=raw,
                    dialect=self.default_dialect,
                )
            )
        return normalized

    def _resolve_table_context(self, dataset: Optional[DatasetInfo]) -> tuple[str, str, str]:
        catalog_name = self.default_catalog
        database_name = self.default_database
        schema_name = self.default_schema

        if dataset and isinstance(dataset.extra, dict):
            extra = dataset.extra
            schema_name = str(extra.get("schema") or schema_name or "").strip()
            catalog_name = str(extra.get("catalog") or catalog_name or "").strip()

            extra_db = extra.get("database")
            database_name = self._extract_database_name(extra_db) or database_name

        return catalog_name, database_name, schema_name

    def _extract_database_name(self, value: Any) -> str:
        if isinstance(value, dict):
            name = value.get("database_name") or value.get("name") or ""
            return str(name).strip()
        if isinstance(value, str):
            return value.strip()
        return ""

    def _can_qualify_table(
        self, dialect: Optional[str], catalog_name: str, database_name: str, schema_name: str
    ) -> bool:
        normalized = (dialect or "").strip().lower()
        if normalized in (DBType.MYSQL, DBType.STARROCKS, DBType.SQLITE):
            return bool(database_name)
        if normalized in (DBType.POSTGRES, DBType.POSTGRESQL, DBType.ORACLE, DBType.DUCKDB):
            return bool(database_name and schema_name)
        if normalized == DBType.SNOWFLAKE:
            return bool(database_name and schema_name)
        return bool(database_name) or bool(schema_name)

    def _dedupe_tables(self, tables: Iterable[str]) -> List[str]:
        deduped: List[str] = []
        for table in tables:
            name = (table or "").strip()
            if not name:
                continue
            matched_index = None
            table_parts = split_table_parts(name)
            for idx, existing in enumerate(deduped):
                existing_parts = split_table_parts(existing)
                if parts_match(table_parts, existing_parts):
                    matched_index = idx
                    break

            if matched_index is None:
                deduped.append(name)
                continue

            deduped[matched_index] = self._prefer_table_name(deduped[matched_index], name)

        return deduped

    def _prefer_table_name(self, left: str, right: str) -> str:
        left_parts = split_table_parts(left)
        right_parts = split_table_parts(right)
        if len(right_parts) > len(left_parts):
            return right
        return left


def split_table_parts(name: str) -> List[str]:
    parts = [part.strip('`"[]') for part in normalize_sql(name).split(".") if part.strip('`"[]')]
    return [part.lower() for part in parts if part]


def parts_match(left: List[str], right: List[str]) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) < len(right):
        return right[-len(left) :] == left
    if len(right) < len(left):
        return left[-len(right) :] == right
    return False
