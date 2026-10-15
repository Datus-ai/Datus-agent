# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Fake dosi-engine binding for unit tests.

Importable directly by runtime test modules so
they never depend on the fake being present in ``sys.modules`` at import
time. The conftest installs :func:`build_fake_module` into
``sys.modules["dosi_engine"]`` only for the duration of each test (and
restores the original afterward), so the fake never leaks into a same-process
integration run. Row shapes match the ``dosi::list`` machine contract.
"""

from __future__ import annotations

import copy
import types
from typing import Any, ClassVar, Dict, List, Optional

METRIC_ROWS = [
    {
        "name": "order_count",
        "kind": "aggregate",
        "datasets": ["orders"],
        "measures": ["order_count"],
        "time_dimension": "orders.order_date",
        "description": "Number of orders",
    },
    {
        "name": "revenue",
        "kind": "aggregate",
        "datasets": ["orders"],
        "measures": ["revenue"],
        "time_dimension": "orders.order_date",
        "description": "Total order amount",
    },
    {
        "name": "running_revenue",
        "kind": "aggregate",
        "datasets": ["orders"],
        "measures": ["running_revenue"],
        # Exercise the dataset-level time-dimension fallback.
        "time_dimension": None,
        "description": "Running revenue",
    },
]

DIMENSION_ROWS = [
    {
        "name": "orders.status",
        "is_time": False,
        "is_dimension": True,
        "source": "inferred",
        "description": "Order status",
    },
    {
        "name": "orders.order_date",
        "is_time": True,
        "is_dimension": True,
        "source": "inferred:time",
        "time_granularity": "day",
        "description": "Order date",
    },
    {
        "name": "customers.region",
        "is_time": False,
        "is_dimension": True,
        "source": "inferred",
        "description": None,
    },
    {
        "name": "orders.amount",
        "is_time": False,
        "is_dimension": False,
        "source": "inferred:measure",
        "description": None,
    },
    {
        "name": "orders.order_id",
        "is_time": False,
        "is_dimension": False,
        "source": "inferred:primary_key",
        "description": None,
    },
]

METRIC_DIMENSION_ROWS = [
    {
        "name": "metric_time",
        "is_time": True,
        "is_dimension": True,
        "source": "inferred:time",
        "time_granularity": None,
        "description": None,
    },
    DIMENSION_ROWS[1],
    DIMENSION_ROWS[0],
    *DIMENSION_ROWS[2:],
]

DATASET_ROWS = [
    {
        "name": "orders",
        "source": "main.orders",
        "primary_key": ["order_id"],
        "fields": 5,
        "time_dimensions": ["order_date"],
        "primary_time_dimension": "order_date",
    },
    {
        "name": "customers",
        "source": "main.customers",
        "primary_key": ["customer_id"],
        "fields": 3,
        "time_dimensions": [],
    },
]

EXECUTE_RESULT = {
    "dialect": "duckdb",
    "sql": "SELECT status, COUNT(*) AS order_count FROM main.orders GROUP BY status",
    "columns": ["status", "order_count"],
    "rows": [{"status": "paid", "order_count": 2}],
    "row_count": 1,
}

ATTRIBUTE_RESULT = {
    "metric": "revenue",
    "strategy": "term_wise",
    "total_change": {
        "baseline_value": 100.0,
        "current_value": 150.0,
        "delta": 50.0,
        "pct_change": 50.0,
    },
    "dimension_ranking": [
        {
            "dimension": "orders.status",
            "score": 1.0,
            "non_additive": False,
            "truncated": False,
        }
    ],
    "selected_dimensions": ["orders.status"],
    "top_dimension_values": [
        {
            "dimension": "orders.status",
            "value": "paid",
            "baseline_value": 100.0,
            "current_value": 150.0,
            "delta": 50.0,
            "contribution_pct": 100.0,
            "segment_kind": "normal",
            "drill_down": {"where_sql": "orders.status = 'paid'"},
        }
    ],
    "per_dimension": {
        "orders.status": {
            "values": [
                {
                    "dimension": "orders.status",
                    "value": "paid",
                    "baseline_value": 100.0,
                    "current_value": 150.0,
                    "delta": 50.0,
                    "contribution_pct": 100.0,
                    "segment_kind": "normal",
                    "drill_down": {"where_sql": "orders.status = 'paid'"},
                }
            ],
            "score": 1.0,
            "non_additive": False,
            "truncated": False,
            "reconciliation": {
                "baseline_residual": 0.0,
                "current_residual": 0.0,
                "passed": True,
            },
        }
    },
    "comparison_metadata": {
        "baseline": {"start": "2026-01-01", "end": "2026-01-08"},
        "current": {"start": "2026-01-08", "end": "2026-01-15"},
        "baseline_days": 7,
        "current_days": 7,
        "equal_length_windows": True,
        "time_dimension": "orders.order_date",
        "queries_executed": 2,
    },
    "warnings": [],
}


class OsiError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "internal",
        metrics=(),
        candidates=(),
        hint: Optional[str] = None,
        suggested_retry: Optional[Dict[str, Any]] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.metrics = list(metrics)
        self.candidates = list(candidates)
        self.hint = hint
        self.suggested_retry = suggested_retry
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "metrics": self.metrics,
            "candidates": self.candidates,
            "hint": self.hint,
            "suggested_retry": self.suggested_retry,
            "detail": self.detail,
        }


class ModelError(OsiError):
    def __init__(self, message, *, issues=(), compile_errors=(), **kwargs):
        super().__init__(message, **kwargs)
        self.issues = list(issues)
        self.compile_errors = list(compile_errors)


class QueryError(OsiError):
    pass


class NotComputableError(OsiError):
    pass


class ExecuteError(OsiError):
    pass


class FakeEngine:
    """Records constructor and call arguments; returns canned rows.

    Program a per-instance failure via ``fail_with`` (raised by compile and
    execute). Every constructed instance is appended to ``instances``.
    """

    instances: ClassVar[List["FakeEngine"]] = []
    time_axis_metrics: ClassVar[set[str]] = {"running_revenue"}

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_text: Optional[str] = None,
        connections: Optional[Dict[str, Dict[str, Any]]] = None,
        pool_size: int = 8,
    ) -> None:
        self.model_path = model_path
        self.model_text = model_text
        self.connections = connections
        self.pool_size = pool_size
        self.compile_calls: List[Dict[str, Any]] = []
        self.dimension_calls: List[Optional[str]] = []
        self.execute_calls: List[Dict[str, Any]] = []
        self.attribute_calls: List[Dict[str, Any]] = []
        self.lineage_calls: List[Dict[str, Any]] = []
        self.fail_with: Optional[Exception] = None
        FakeEngine.instances.append(self)

    def datasets(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in DATASET_ROWS]

    def metrics(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in METRIC_ROWS]

    def dimensions(self, metric: Optional[str] = None) -> List[Dict[str, Any]]:
        self.dimension_calls.append(metric)
        rows = METRIC_DIMENSION_ROWS if metric is not None else DIMENSION_ROWS
        return [dict(r) for r in rows]

    def lineage(self, redact_sql: bool = False) -> Dict[str, Any]:
        """Shape-faithful miniature of the dosi lineage graph contract v2."""
        self.lineage_calls.append({"redact_sql": redact_sql})
        source = "<redacted>" if redact_sql else "main.orders"
        return {
            "version": 2,
            "model": "fake_model",
            "mode": "datus",
            "layers": [],
            "nodes": [
                {
                    "id": "dataset:orders",
                    "kind": "dataset",
                    "detail": {"source": source},
                }
            ],
            "edges": [],
        }

    def compile(self, query, dialect=None, connection=None, pretty=False):
        self.compile_calls.append(
            {
                "query": query,
                "dialect": dialect,
                "connection": connection,
                "pretty": pretty,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        group_by = query.get("group_by") or []
        has_time_axis = any(
            item.get("grain") and item.get("field") in {"metric_time", "orders.order_date"} for item in group_by
        )
        if any(metric in self.time_axis_metrics for metric in query.get("metrics") or []) and not has_time_axis:
            raise QueryError(
                "window metric requires its time axis in group_by with a grain",
                code="unknown_dimension",
                hint="add metric_time:<grain> to group_by",
            )
        return {
            "dialect": dialect or "duckdb",
            "sql": "SELECT 1 AS compiled",
            "outputs": [{"name": metric, "type": "metric"} for metric in query.get("metrics") or []],
        }

    def explain(self, query) -> str:
        return "ScanDataset orders"

    def execute(self, query, dialect=None, connection=None, timeout_secs=None, db_path=None):
        self.execute_calls.append(
            {
                "query": query,
                "dialect": dialect,
                "connection": connection,
                "timeout_secs": timeout_secs,
                "db_path": db_path,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        return {
            **EXECUTE_RESULT,
            "outputs": [{"name": metric, "type": "metric"} for metric in query.get("metrics") or []],
        }

    def attribute(
        self,
        request,
        dialect=None,
        connection=None,
        timeout_secs=None,
        db_path=None,
    ):
        self.attribute_calls.append(
            {
                "request": request,
                "dialect": dialect,
                "connection": connection,
                "timeout_secs": timeout_secs,
                "db_path": db_path,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        return copy.deepcopy(ATTRIBUTE_RESULT)


# Ontology payloads follow the Dosi MCP ``describe`` / ``paths`` JSON, which the
# Python binding returns verbatim (aviation example, trimmed).
ONTOLOGY_CATALOG = {
    "ontology": "Aviation",
    "description": "Flights between airports, with delays and alerts.",
    "concepts": [
        {"name": "Airport", "identity": ["code"], "out": 0, "in": 4, "props": 4, "root": True},
        {"name": "Flight", "identity": ["id"], "out": 5, "in": 3, "props": 11, "root": True},
        {"name": "Delay", "identity": ["id"], "out": 1, "in": 0, "props": 4, "root": True},
        {"name": "FlightAlert", "identity": ["id"], "out": 1, "in": 0, "props": 7, "root": True},
    ],
}

ONTOLOGY_CONCEPTS = {
    "FlightAlert": {
        "concept": "FlightAlert",
        "identity": ["id"],
        "root": "flight_alerts",
        "properties": [
            {"path": "FlightAlert.id", "type_name": "AlertId", "base": "String"},
            {
                "path": "FlightAlert.status",
                "type_name": "AlertStatus",
                "base": "String",
                "domain": ["OPEN", "RESOLVED"],
            },
            {"path": "FlightAlert.created_at", "type_name": "DateTime", "base": "DateTime", "is_time": True},
        ],
        "links": [{"name": "flight", "role": "flight", "to": "Flight", "multiplicity": "ManyToOne", "no_join": False}],
        "incoming": [],
        "groupable_metrics": [{"group_by": "FlightAlert.id", "metrics": ["alert_count"]}],
        "time_dimensions_reachable": ["FlightAlert.created_at", "FlightAlert.flight.scheduled_departure"],
    },
    "Delay": {
        "concept": "Delay",
        "identity": ["id"],
        "root": "delays",
        "properties": [{"path": "Delay.minutes", "type_name": "NrMinutes", "base": "Integer"}],
        "links": [{"name": "flight", "role": "flight", "to": "Flight", "multiplicity": "ManyToOne", "no_join": False}],
        "incoming": [],
        "groupable_metrics": [{"group_by": "Delay.id", "metrics": ["delayed_flights"]}],
        "time_dimensions_reachable": ["Delay.flight.scheduled_departure"],
    },
}

ONTOLOGY_PATHS = {
    "paths": [
        {"path": "Flight.arrives_at", "roles": ["destination"], "hops": 1, "multiplicity": "ManyToOne"},
        {"path": "Flight.departs_from", "roles": ["origin"], "hops": 1, "multiplicity": "ManyToOne"},
    ],
    "equivalences": [
        {
            "a": "Flight.departs_from",
            "b": "Flight.follows_route.origin",
            "declared_by": "Flight.follows_route.origin == origin",
        }
    ],
    "max_hops": 3,
}

SELECT_RESULT = {
    "dialect": "duckdb",
    "sql": "SELECT flight_alerts.alert_id AS alert_id FROM main.flight_alerts AS flight_alerts",
    "columns": ["alert_id", "priority"],
    "rows": [{"alert_id": "A007", "priority": "P1"}, {"alert_id": "A011", "priority": "P1"}],
    "row_count": 2,
    "requested_columns": [
        {"requested": "FlightAlert.id", "column": "alert_id"},
        {"requested": "FlightAlert.priority", "column": "priority"},
    ],
}


class FakeOntologyEngine(FakeEngine):
    """A binding Engine that also answers ontology calls (``describe``/``paths``/``select``)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_text: Optional[str] = None,
        connections: Optional[Dict[str, Dict[str, Any]]] = None,
        pool_size: int = 8,
        *,
        ontology_path: Optional[str] = None,
        ontology_text: Optional[str] = None,
        mapping: Optional[str] = None,
    ) -> None:
        super().__init__(model_path=model_path, model_text=model_text, connections=connections, pool_size=pool_size)
        self.ontology_path = ontology_path
        self.mapping = mapping
        self.describe_calls: List[Optional[str]] = []
        self.paths_calls: List[tuple] = []
        self.select_calls: List[Dict[str, Any]] = []
        self.compile_select_calls: List[Dict[str, Any]] = []

    @property
    def has_ontology(self) -> bool:
        return self.ontology_path is not None

    def ontology_info(self) -> Optional[Dict[str, Any]]:
        if not self.ontology_path:
            return None
        return {"name": "Aviation", "description": "", "mapping": self.mapping, "source_path": self.ontology_path}

    def _require_ontology(self) -> None:
        if not self.ontology_path:
            raise QueryError("load an ontology to describe concepts", code="ontology_required")

    def describe(self, name: Optional[str] = None) -> Dict[str, Any]:
        self.describe_calls.append(name)
        if name is None:
            if not self.ontology_path:
                return {"model": "m", "datasets": [], "metrics": []}
            return copy.deepcopy(ONTOLOGY_CATALOG)
        self._require_ontology()
        if name not in ONTOLOGY_CONCEPTS:
            raise QueryError(f"unknown concept {name!r}", code="unknown_concept", candidates=sorted(ONTOLOGY_CONCEPTS))
        return copy.deepcopy(ONTOLOGY_CONCEPTS[name])

    def paths(self, source: str, target: str) -> Dict[str, Any]:
        self.paths_calls.append((source, target))
        self._require_ontology()
        return copy.deepcopy(ONTOLOGY_PATHS)

    def compile_select(self, query, dialect=None, connection=None, pretty=False, dws_mode=None):
        self.compile_select_calls.append({"query": query, "dialect": dialect, "connection": connection})
        if self.fail_with is not None:
            raise self.fail_with
        return {
            "dialect": dialect or "duckdb",
            "sql": SELECT_RESULT["sql"],
            "outputs": [],
            "requested_columns": copy.deepcopy(SELECT_RESULT["requested_columns"]),
        }

    def select(self, query, dialect=None, connection=None, timeout_secs=None, db_path=None):
        self.select_calls.append(
            {"query": query, "dialect": dialect, "connection": connection, "timeout_secs": timeout_secs}
        )
        if self.fail_with is not None:
            raise self.fail_with
        return copy.deepcopy(SELECT_RESULT)


def default_validate(model_text: str, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
    names = metric_names or [row["name"] for row in METRIC_ROWS]
    compiled_metrics = [dict(row) for row in METRIC_ROWS if row["name"] in names]
    return {
        "valid": True,
        "issues": [],
        "compile_errors": [],
        "compiled_metrics": compiled_metrics,
        "compiled_metric_digests": {row["name"]: f"sha256:{row['name']}" for row in compiled_metrics},
        "contract_digest": "sha256:" + "a" * 64,
    }


def build_fake_module(*, ontology: bool = False) -> types.ModuleType:
    """Build a module object mirroring the real ``dosi_engine`` surface.

    Not installed into ``sys.modules`` here — the conftest installs it per
    test and restores the previous entry, so it never shadows a real wheel
    outside unit tests.
    """
    module = types.ModuleType("dosi_engine")
    module.__dosi_fake__ = True
    module.Engine = FakeOntologyEngine if ontology else FakeEngine
    module.validate = default_validate
    module.SPEC_VERSION = "0.2.0.dev0"
    module.DATUS_EXT_VERSION = "1.2"
    module.DATUS_EXT = {
        "version": "1.2",
        "keys": [
            {"key": "time_dimension"},
            {"key": "time_granularity"},
            {"key": "window"},
        ],
    }
    module.DIALECTS = [
        "duckdb",
        "starrocks",
        "clickhouse",
        "doris",
        "tidb",
        "trino",
        "postgres",
        "mysql",
        "snowflake",
        "bigquery",
        "databricks",
        "redshift",
    ]
    module.CONNECTION_TYPES = [*module.DIALECTS, "sqlite"]
    module.OsiError = OsiError
    module.ModelError = ModelError
    module.QueryError = QueryError
    module.NotComputableError = NotComputableError
    module.ExecuteError = ExecuteError
    return module
