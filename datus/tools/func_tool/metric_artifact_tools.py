# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Compile saved metric recipes and execute them through the enforced read path."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from datus.schemas.analysis_artifacts import QueryBrief, SubjectRefs
from datus.schemas.gen_visual_dashboard_models import QueryTemplateMetaFile, TemplateParamDecl
from datus.schemas.gen_visual_report_models import QueryColumnMeta, QueryResultFile
from datus.schemas.metric_artifact_query import MetricQueryFile, MetricQueryRequest, ParamRef
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.async_utils import run_async
from datus.utils.exceptions import DatusException, ErrorCode


def _bound(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, ParamRef):
        if value.param not in params:
            raise ValueError(f"missing metric query parameter {value.param!r}")
        return params[value.param]
    return value


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("filter numbers must be finite")
        return str(value)
    if isinstance(value, str):
        if "\\" in value or "\x00" in value:
            raise ValueError("filter strings cannot contain backslashes or NUL")
        return "'" + value.replace("'", "''") + "'"
    raise ValueError("filter values must be strings, finite numbers or booleans")


def metric_query_arguments(query: MetricQueryRequest, params: dict[str, Any]) -> dict[str, Any]:
    """Only bind values; identifiers and operators come from the saved schema."""
    clauses = []
    operators = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    for item in query.filters:
        value = _bound(item.value, params)
        if isinstance(item.value, ParamRef) and value is None:
            continue  # An omitted optional slicer imposes no predicate.
        if item.op in {"is_null", "is_not_null"}:
            clauses.append(f"{item.dimension} IS {'NOT ' if item.op == 'is_not_null' else ''}NULL")
        elif item.op in {"in", "not_in"}:
            if not isinstance(value, list):
                raise ValueError(f"{item.op} requires an array")
            if not value:
                clauses.append("1 = 0" if item.op == "in" else "1 = 1")
            else:
                clauses.append(
                    f"{item.dimension} {'NOT IN' if item.op == 'not_in' else 'IN'} ({', '.join(map(_literal, value))})"
                )
        elif value is None:
            if item.op not in {"eq", "ne"}:
                raise ValueError("null only supports eq/ne filters")
            clauses.append(f"{item.dimension} IS {'NOT ' if item.op == 'ne' else ''}NULL")
        else:
            clauses.append(f"{item.dimension} {operators[item.op]} {_literal(value)}")
    return {
        "metrics": [query.metric.name],
        "dimensions": query.dimensions,
        "time_start": _bound(query.time_start, params),
        "time_end": _bound(query.time_end, params),
        "time_granularity": query.time_granularity,
        "where": " AND ".join(f"({clause})" for clause in clauses) or None,
        "order_by": query.order_by or None,
        "limit": query.limit,
        "params": {
            key: _bound(value, params)
            for key, value in query.metric_params.items()
            if not isinstance(value, ParamRef) or _bound(value, params) is not None
        },
        "dry_run": True,
    }


def execute_metric_artifact_query(
    query: MetricQueryRequest,
    *,
    semantic_tools,
    db_tool,
    project_root: Path,
    params: dict[str, Any] | None = None,
    policy_context: dict[str, Any] | None = None,
    saved: MetricQueryFile | None = None,
) -> tuple[QueryResultFile, dict[str, Any]]:
    """The engine owns calculation; DBFuncTool owns warehouse read enforcement."""
    from datus.tools.func_tool.report_artifact_tools import _infer_column_type, _normalize_value
    from datus.utils.artifact_insight import query_lineage

    runtime = semantic_tools.runtime
    if runtime is None:
        raise ValueError("metric runtime unavailable")
    binding = runtime.artifact_metric_binding(query.metric.name)
    model_path = Path(binding["model_path"]).resolve()
    # Absolute Studio paths are never persisted or accepted from a viewer.
    relative_path = model_path.relative_to(project_root.resolve()).as_posix()
    if saved and (
        saved.model_path != relative_path
        or saved.model_revision != binding["model_revision"]
        or saved.datasource != binding["datasource"]
    ):
        raise ValueError(
            "METRIC_MODEL_CHANGED: saved semantic model or datasource differs; regenerate or republish the query"
        )
    if saved:
        # Published execution uses the pinned model and captured identity, not
        # mutable subject-index availability. Compilation validates dimensions.
        detail = saved.metric_detail
    else:
        detail_result = semantic_tools.get_metric(name=query.metric.name, path=query.metric.path)
        detail = detail_result.result
        if (
            not detail_result.success
            or not isinstance(detail, dict)
            or detail.get("name") != query.metric.name
            or detail.get("path") != query.metric.path
        ):
            raise ValueError("metric subject identity could not be resolved unambiguously")
        allowed_dimensions = {item.get("name") for item in detail.get("dimensions", []) if isinstance(item, dict)}
        if any(item.dimension not in allowed_dimensions for item in query.filters):
            raise ValueError("filter dimension is not queryable for this metric")
        from datus.agent.node.visual_artifact._insight_snapshot import _metric_match

        match = _metric_match(run_async(runtime.lineage_graph()), query.metric.name)
        if match is not None:
            detail = {**detail, "definition": match[1].get("detail", {})}
    values = {name: None for name in query.parameter_names()}
    values.update(params or {})
    compiled = run_async(runtime.query_metrics(**metric_query_arguments(query, values)))
    sql = (compiled.metadata or {}).get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("metric engine returned no compiled SQL")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != binding["model_revision"]:
        raise ValueError("semantic model changed while compiling; retry the metric query")
    datasource = binding["datasource"]
    connector = db_tool._get_connector(datasource)
    oversize = db_tool.guard_estimated_rows(sql, connector)
    if oversize is not None:
        raise ValueError(oversize.error)
    result = db_tool.execute_read_enforced(
        sql, connector, datasource=datasource, result_format="arrow", policy_context=policy_context
    )
    if not result.success:
        if getattr(result, "error_code", None) == ErrorCode.POLICY_DENIED.code:
            raise DatusException(ErrorCode.POLICY_DENIED, result.error)
        raise ValueError(result.error or "metric query execution failed")
    # Arrow retains every projected column even when no rows match. Compiler
    # outputs describe metric aliases, not the complete grouped projection.
    table = result.sql_return
    if not hasattr(table, "column_names") or not hasattr(table, "to_pylist"):
        raise ValueError("metric execution must return an Arrow result with projection columns")
    names = list(table.column_names)
    raw_rows = table.to_pylist()
    if not isinstance(raw_rows, list) or any(not isinstance(row, dict) for row in raw_rows):
        raise ValueError("metric execution must return complete dictionary rows")
    if len(raw_rows) > query.limit:
        raise ValueError("metric execution exceeded the saved row limit")
    rows = [{key: _normalize_value(value) for key, value in row.items()} for row in raw_rows]
    if not names:
        raise ValueError("metric engine returned no output columns")
    columns = [
        QueryColumnMeta(name=name, type=_infer_column_type([row.get(name) for row in rows[:200]])) for name in names
    ]
    from datus.tools.func_tool._visual_artifact_helpers import utc_now_iso

    payload = QueryResultFile(
        executed_at=utc_now_iso(),
        datasource=datasource,
        row_count=len(rows),
        columns=columns,
        rows=rows,
        sql=sql,
        source={"kind": "metric", "metric": query.metric.model_dump()},
    )
    tables = query_lineage(sql, None, datasource).tables
    return payload, {
        "datasource": datasource,
        "model_path": relative_path,
        "model_revision": binding["model_revision"],
        "model_snapshot": model_path.read_bytes().decode("utf-8"),
        "metric_detail": detail,
        "metric_tables": tables,
        "captured_at": payload.executed_at,
        "generated_sql": sql,
    }


def execute_published_metric_query(query: MetricQueryFile, *, runtime_config, db_tool, params, policy_context):
    """Compile only the artifact's frozen model, including standalone dashboards."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    from datus.tools.semantic_tools.dosi import DosiRuntime

    if query.model_snapshot is None:
        raise ValueError("METRIC_MODEL_SNAPSHOT_MISSING: republish a metric query with its semantic model snapshot")
    with TemporaryDirectory(prefix="datus-metric-query-") as directory:
        root = Path(directory)
        model = root / query.model_path
        model.parent.mkdir(parents=True)
        model.write_bytes(query.model_snapshot.encode("utf-8"))
        config = runtime_config.model_copy(
            update={
                "semantic_model_path": str(model),
                "semantic_models_path": None,
                "datasource": query.datasource,
                "connection": query.datasource,
            }
        )
        return execute_metric_artifact_query(
            query,
            semantic_tools=SimpleNamespace(runtime=DosiRuntime(config)),
            db_tool=db_tool,
            project_root=root,
            params=params,
            policy_context=policy_context,
            saved=query,
        )


class MetricArtifactToolsMixin:
    """Common save implementation for report snapshots and dashboard recipes."""

    def _validate_metric_queries(self, kind: str) -> str | None:
        """Check complete tool-produced bundles before declaring render success."""
        try:
            metric_revisions: dict[tuple[tuple[str, ...], str], str] = {}
            for path in self.queries_dir.glob("*.metric.json"):
                saved = MetricQueryFile.model_validate_json(path.read_text(encoding="utf-8"))
                identity = (tuple(saved.metric.path), saved.metric.name)
                previous_revision = metric_revisions.setdefault(identity, saved.model_revision)
                if previous_revision != saved.model_revision:
                    raise ValueError(
                        "one artifact cannot mix model revisions of the same metric; resave all its queries"
                    )
                name = path.name.removesuffix(".metric.json")
                if saved.name != name or any(
                    (self.queries_dir / f"{name}{suffix}").exists() for suffix in (".sql", ".sql.j2")
                ):
                    raise ValueError(f"{name}: metric identity or exclusive source mismatch")
                brief = QueryBrief.model_validate_json(
                    (self.queries_dir / f"{name}.brief.json").read_text(encoding="utf-8")
                )
                if brief.name != name or brief.uses.metrics != [saved.metric]:
                    raise ValueError(f"{name}: metric attribution mismatch")
                if kind == "report":
                    result = QueryResultFile.model_validate_json(
                        (self.queries_dir / f"{name}.json").read_text(encoding="utf-8")
                    )
                    if (
                        saved.parameter_names()
                        or result.datasource != saved.datasource
                        or result.sql != saved.generated_sql
                        or (result.source or {}).get("kind") != "metric"
                        or (result.source or {}).get("metric") != saved.metric.model_dump()
                    ):
                        raise ValueError(f"{name}: metric result does not match its recipe")
                else:
                    meta = QueryTemplateMetaFile.model_validate_json(
                        (self.queries_dir / f"{name}.params.json").read_text(encoding="utf-8")
                    )
                    if (
                        meta.slug != name
                        or meta.datasource != saved.datasource
                        or saved.parameter_names() != {p.name for p in meta.params}
                    ):
                        raise ValueError(f"{name}: metric parameters do not match its recipe")
        except (OSError, ValueError) as exc:
            return f"Invalid saved metric query: {exc}"
        return None

    def save_metric_query(
        self, name: str, query: dict[str, Any], goal: str, hypothesis: str, caveats: str = ""
    ) -> FuncToolResult:
        """Save a report using ONE existing metric, never authored SQL.

        query: metric {path, name}, dimensions, time_start/time_end,
        time_granularity, filters [{dimension, op, value}], order_by, limit,
        metric_params. Use get_metric to inspect capabilities first.
        Returns complete-result data_ref, columns and a small row preview.
        """
        return self._save_metric_artifact(name, query, goal, hypothesis, caveats, None, None)

    def save_metric_query_template(
        self,
        name: str,
        query: dict[str, Any],
        goal: str,
        hypothesis: str,
        params: list[dict[str, Any]],
        sample_params: dict[str, Any],
        caveats: str = "",
    ) -> FuncToolResult:
        """Save a live dashboard query using ONE metric.

        Same query fields as save_metric_query. Bind value fields with
        {"param": "start_date"}; params declares name/type/required; sample_params
        supplies typed trial values. Metric names and dimensions cannot be bound.
        """
        return self._save_metric_artifact(name, query, goal, hypothesis, caveats, params, sample_params)

    def _save_metric_artifact(self, name, query, goal, hypothesis, caveats, params, sample_params) -> FuncToolResult:
        from datus.api.services.dashboard_service import _validate_params
        from datus.tools.func_tool._visual_artifact_helpers import upsert_manifest_after_save
        from datus.tools.func_tool.report_artifact_tools import _atomic_write_text

        kind = "dashboard" if params is not None else "report"
        tool_name = "save_metric_query_template" if params is not None else "save_metric_query"
        unbound = self._require_active(tool_name)
        if unbound is not None:
            return unbound
        if self._semantic_tools is None:
            return FuncToolResult(
                success=0, error="semantic tools unavailable; inspect metric configuration before saving"
            )
        try:
            request = MetricQueryRequest.model_validate(query)
            declarations = [TemplateParamDecl.model_validate(item) for item in params or []]
            if len({item.name for item in declarations}) != len(declarations):
                raise ValueError("duplicate parameter declaration")
            if request.parameter_names() != {item.name for item in declarations}:
                raise ValueError("declared parameters must match the recipe's value bindings")
            values = _validate_params(declarations, sample_params or {})
            brief = QueryBrief(
                name=name, hypothesis=hypothesis.strip(), uses=SubjectRefs(metrics=[request.metric]), caveats=caveats
            )
            # Validate the identity and metadata before touching the warehouse.
            if not goal.strip():
                raise ValueError("goal must be non-empty")
            payload, binding = execute_metric_artifact_query(
                request,
                semantic_tools=self._semantic_tools,
                db_tool=self._db_func_tool,
                project_root=self._project_root,
                params=values,
            )
            saved = MetricQueryFile(**request.model_dump(), name=name, goal=goal.strip(), **binding)
            files = {
                f"{name}.metric.json": saved.model_dump_json(indent=2),
                f"{name}.brief.json": json.dumps(brief.model_dump(exclude_none=True), ensure_ascii=False, indent=2),
            }
            if kind == "report":
                files[f"{name}.json"] = payload.model_dump_json(indent=2)
            else:
                meta = QueryTemplateMetaFile(
                    slug=name,
                    description=goal.strip(),
                    datasource=payload.datasource,
                    params=declarations,
                    columns=payload.columns,
                    sample_params=values,
                    sample_row_count=payload.row_count,
                    saved_at=payload.executed_at,
                )
                files[f"{name}.params.json"] = meta.model_dump_json(indent=2)
            if sum(len(value.encode("utf-8")) for value in files.values()) > 20 * 1024 * 1024:
                raise ValueError("metric query files exceed the 20 MB limit; aggregate or lower limit")
            for filename, content in files.items():
                _atomic_write_text(self.queries_dir / filename, content + "\n")
            for suffix in (".sql", ".sql.j2"):
                (self.queries_dir / f"{name}{suffix}").unlink(missing_ok=True)
            artifact_dir = self.dashboard_dir if kind == "dashboard" else self.report_dir
            warning = upsert_manifest_after_save(
                artifact_dir / "manifest.json", datasource=payload.datasource, timestamp=payload.executed_at
            )
            return FuncToolResult(
                result={
                    "name": name,
                    "source_kind": "metric",
                    "metric_path": (self.queries_dir / f"{name}.metric.json")
                    .relative_to(self._project_root)
                    .as_posix(),
                    "data_ref": f"queries/{name}",
                    "columns": [column.model_dump() for column in payload.columns],
                    "row_count": payload.row_count,
                    "preview_rows": payload.rows[:3],
                    "manifest_warning": warning,
                }
            )
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Metric query was not saved: {exc}")
