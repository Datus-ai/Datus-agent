# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Aggregate saved artifact files without executing SQL, JSX or an LLM.

The same reader serves working copies and immutable publication bundles.
Malformed optional files degrade individually; never repair old snapshots by
looking up today's semantic model.
"""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, ValidationError

from datus.schemas.analysis_artifacts import Insight, QueryBrief, SuggestedQuestion
from datus.schemas.artifact_insight import (
    ArtifactBlock,
    ArtifactInsight,
    InsightQuery,
    MetricSnapshot,
    QueryLineage,
    ReferenceSqlSnapshot,
)
from datus.schemas.artifact_manifest import ArtifactManifest
from datus.schemas.gen_visual_dashboard_models import QueryTemplateMetaFile
from datus.schemas.gen_visual_report_models import QueryResultFile, extract_query_slug
from datus.schemas.key_tables_schema import KeyTablesSchemaFile
from datus.schemas.metric_artifact_query import MetricQueryFile


def artifact_revision(manifest: dict, files: dict[str, str]) -> str:
    payload = json.dumps([manifest, sorted(files.items())], sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    return hashlib.sha256(payload.encode()).hexdigest()


def query_revision(files: dict[str, str]) -> str:
    return artifact_revision({}, {p: s for p, s in files.items() if p.startswith("queries/")})


def _read(files: dict[str, str], path: str, model: type[BaseModel], warnings: list[str]):
    if path not in files:
        return None

    try:
        return model.model_validate_json(files[path])
    except (ValueError, ValidationError):
        warnings.append(path)
        return None


def _read_list(files: dict[str, str], path: str, model: type[BaseModel], warnings: list[str]) -> list:
    if path not in files:
        return []

    try:
        raw = json.loads(files[path])
        if not isinstance(raw, list):
            raise ValueError("Expected a list")
    except (ValueError, TypeError):
        warnings.append(path)
        return []

    result = []
    for item in raw:
        try:
            result.append(model.model_validate(item))
        except (ValueError, TypeError):
            warnings.append(path)

    return result


def query_lineage(
    sql: str | None, template: QueryTemplateMetaFile | None, datasource: str | None, *, is_template: bool = False
) -> QueryLineage:
    from sqlglot import exp, parse
    from sqlglot.optimizer.scope import traverse_scope

    origin = "sample_parameters" if template or is_template else "saved_sql"
    result = QueryLineage(status="unavailable", origin=origin, datasource=datasource)
    if not sql or (is_template and template is None):
        return result

    try:
        if template:
            from datus.tools.func_tool.dashboard_artifact_tools import render_dashboard_template

            sql = render_dashboard_template(sql, template.params, template.sample_params)

        trees = parse(sql)
        if not trees or any(tree is None or not isinstance(tree, exp.Query) for tree in trees):
            return result

        tables = set()
        incomplete = False
        for tree in trees:
            for scope in traverse_scope(tree):
                # UNNEST and lateral table functions are scopes, not exp.Table.
                if isinstance(scope.expression, exp.UDTF):
                    incomplete = True
                for source in scope.sources.values():
                    if isinstance(source, exp.Table):
                        parts = source.parts
                        if parts and all(isinstance(part, exp.Identifier) and part.name for part in parts):
                            tables.add(".".join(part.name for part in parts))
                        else:
                            # SQLGlot also wraps function calls in exp.Table.
                            incomplete = True

        # A template's trial parameter branch cannot prove every runtime branch.
        result.status = "partial" if template or incomplete else "parsed"
        result.tables = sorted(tables)
    except Exception:
        # Unsupported SQL/dialects and dynamic identifiers must remain unknown.
        pass

    return result


def _blocks(files: dict[str, str], query_names: set[str]) -> list[ArtifactBlock]:
    from datus.tools.func_tool._visual_artifact_cards import (
        BLOCK_HANDLE_OPEN_RE,
        CHART_CARD_OPEN_RE,
        JS_NON_CODE_RE,
        SPREAD_ATTR_RE,
        parse_card_attributes,
    )

    blocks: dict[str, ArtifactBlock] = {}
    duplicates: set[str] = set()
    attr = re.compile(r"""\b(chartId|handleId|title|name|kind|chartType|sqlId)\s*=\s*['"]([^'"]+)['"]""")

    for path, source in sorted(files.items()):
        if not path.startswith("render/") or not path.endswith((".jsx", ".js")):
            continue

        non_code = [(m.start(), m.end()) for m in JS_NON_CODE_RE.finditer(source)]

        # Use the same static declaration grammar as validate_render; only literal
        # bindings are shown. Dynamic/spread wrappers remain explicitly partial.
        for pattern, id_attr in ((CHART_CARD_OPEN_RE, "chartId"), (BLOCK_HANDLE_OPEN_RE, "handleId")):
            for match in pattern.finditer(source):
                if any(start <= match.start() < end for start, end in non_code):
                    continue

                attrs = match.group(1)
                if SPREAD_ATTR_RE.search(attrs):
                    continue

                props = parse_card_attributes(attrs)
                values = dict(attr.findall(props.literals))
                block_id = values.get(id_attr)
                if not block_id or not re.fullmatch(r"[a-z0-9_]{1,64}", block_id):
                    continue
                if block_id in blocks:
                    duplicates.add(block_id)
                    continue

                blocks[block_id] = ArtifactBlock(
                    id=block_id,
                    title=values.get("title") or values.get("name") or block_id,
                    kind=values.get("chartType") or values.get("kind") or "chart",
                    query_ids=list(
                        dict.fromkeys(
                            slug
                            for ref in [values.get("sqlId", ""), *(props.query_ids or [])]
                            if (slug := extract_query_slug(ref)) in query_names
                        )
                    ),
                    source_path=path,
                )

    return [b for key, b in blocks.items() if key not in duplicates]


def build_artifact_insight(manifest: dict, files: dict[str, str]) -> ArtifactInsight:
    parsed_manifest = ArtifactManifest.model_validate(manifest)
    warnings: list[str] = []
    queries = []
    suffix = ".sql.j2" if parsed_manifest.kind == "dashboard" else ".sql"
    names = {p[len("queries/") : -len(suffix)] for p in files if p.startswith("queries/") and p.endswith(suffix)}

    # Retain queries with missing SQL rather than silently dropping them.
    names.update(
        p[len("queries/") : -len(".brief.json")]
        for p in files
        if p.startswith("queries/") and p.endswith(".brief.json")
    )

    # Sidecars survive partial saves / manual SQL deletion too.
    for path in files:
        if path.startswith("queries/") and path.endswith(".json"):
            name = path[len("queries/") : -len(".json")]
            for ending in (".brief", ".params", ".metric"):
                name = name.removesuffix(ending)
            names.add(name)

    for name in sorted(names):
        if not re.fullmatch(r"[a-z0-9_]{1,64}", name):
            continue

        base = f"queries/{name}"
        sql = files.get(base + suffix)
        metric = _read(files, base + ".metric.json", MetricQueryFile, warnings)
        brief = _read(files, base + ".brief.json", QueryBrief, warnings)
        template = (
            _read(files, base + ".params.json", QueryTemplateMetaFile, warnings)
            if parsed_manifest.kind == "dashboard"
            else None
        )
        result = _read(files, base + ".json", QueryResultFile, warnings) if parsed_manifest.kind == "report" else None

        if brief is not None and brief.name != name:
            warnings.append(base + ".brief.json")
            brief = None
        if template is not None and template.slug != name:
            warnings.append(base + ".params.json")
            template = None

        if metric and (
            metric.name != name
            or sql is not None
            or (
                result
                and (
                    (result.source or {}).get("kind") != "metric"
                    or (
                        (result.source or {}).get("metric") is not None
                        and result.source["metric"] != metric.metric.model_dump()
                    )
                )
            )
        ):
            warnings.append(base + ".metric.json:source_mismatch")
            metric = None
        generated_sql = metric.generated_sql if metric else sql
        lineage = query_lineage(
            generated_sql,
            None if metric else template,
            metric.datasource if metric else result.datasource if result else template.datasource if template else None,
            is_template=parsed_manifest.kind == "dashboard" and metric is None,
        )
        if metric:
            lineage.origin = "metric_sample" if template else "metric_query"
            if template and lineage.status == "parsed":
                lineage.status = "partial"
        goal = (
            metric.goal
            if metric
            else template.description
            if template
            else (sql.splitlines()[0][3:] if sql and sql.startswith("-- ") else None)
        )
        queries.append(
            InsightQuery(
                name=name,
                source_kind="metric" if metric else "sql",
                metric_query=metric,
                goal=goal,
                sql=generated_sql,
                brief=brief,
                result=result,
                template=template,
                lineage=lineage,
            )
        )

    metric_details = []
    if "analysis/metric_snapshots.json" in files:
        try:
            raw = json.loads(files["analysis/metric_snapshots.json"])
            if raw["query_revision"] != query_revision(files):
                warnings.append("analysis/metric_snapshots.json:stale")
            else:
                metric_details = [MetricSnapshot.model_validate(m) for m in raw["metrics"]]
        except (ValueError, TypeError, KeyError):
            warnings.append("analysis/metric_snapshots.json")

    # Execution-time metric definitions survive render edits and finalization.
    # Do not overwrite them with today's catalog for a referenced SQL query.
    native_snapshots = {}
    for query in queries:
        if query.metric_query:
            metric = query.metric_query
            key = (tuple(metric.metric.path), metric.metric.name)
            native_snapshots[key] = MetricSnapshot(
                origin="metric_execution",
                ref=metric.metric,
                captured_at=metric.captured_at,
                status="captured",
                detail=metric.metric_detail,
                tables=metric.metric_tables,
                lineage_status="resolved",
            )
    metric_details = [m for m in metric_details if (tuple(m.ref.path), m.ref.name) not in native_snapshots]
    metric_details.extend(native_snapshots.values())

    reference_sql_details = []
    reference_path = "analysis/reference_sql_snapshots.json"
    if reference_path in files:
        try:
            raw = json.loads(files[reference_path])
            if raw["query_revision"] != query_revision(files):
                warnings.append(f"{reference_path}:stale")
            else:
                reference_sql_details = [ReferenceSqlSnapshot.model_validate(s) for s in raw["reference_sql"]]
        except (ValueError, TypeError, KeyError):
            warnings.append(reference_path)

    blocks = _blocks(files, names)

    return ArtifactInsight(
        source_revision=artifact_revision(manifest, files),
        manifest=parsed_manifest,
        queries=queries,
        blocks=blocks,
        blocks_status="partial" if blocks else "unavailable",
        metric_details=metric_details,
        reference_sql_details=reference_sql_details,
        key_tables_schema=_read(files, "analysis/key_tables_schema.json", KeyTablesSchemaFile, warnings),
        insights=_read_list(files, "analysis/insights.json", Insight, warnings)
        if parsed_manifest.kind == "report"
        else [],
        suggested_questions=_read_list(files, "analysis/suggested_questions.json", SuggestedQuestion, warnings),
        warnings=sorted(set(warnings)),
    )
