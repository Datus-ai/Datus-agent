# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Best-effort semantic definition snapshots; never execute business queries."""

import json
from datetime import datetime, timezone
from pathlib import Path

from datus.schemas.analysis_artifacts import SubjectRefs
from datus.schemas.artifact_insight import MetricSnapshot, ReferenceSqlSnapshot
from datus.schemas.artifact_manifest import ArtifactKind
from datus.utils.artifact_files import iter_artifact_files
from datus.utils.artifact_insight import query_revision
from datus.utils.async_utils import run_async


def _metric_match(graphs: list[dict], name: str):
    matches = [
        (graph, node)
        for graph in graphs
        for node in graph.get("nodes", [])
        if node.get("kind") in {"metric_atomic", "metric_derived"} and node.get("name") == name
    ]

    if len(matches) != 1:
        return None

    return matches[0]


def metric_tables(graphs: list[dict], name: str) -> list[str] | None:
    match = _metric_match(graphs, name)

    if match is None:
        return None

    graph, metric = match
    nodes = {n["id"]: n for n in graph["nodes"]}
    upstream: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        # Joins describe available relationships, not required dependencies.
        if edge.get("kind") in {"reads_table", "aggregates", "derive_base", "window_base", "compose_member"}:
            upstream.setdefault(edge["target"], []).append(edge["source"])

    pending, seen, tables = [metric["id"]], set(), set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        node = nodes.get(key)
        if not node:
            return None
        if node["kind"] == "physical_table":
            tables.add(node.get("detail", {}).get("table") or node["name"])
        pending.extend(upstream.get(key, []))

    return sorted(tables)


def bake_metric_snapshots(artifact_dir: Path, refs, semantic_tools, *, artifact_kind: ArtifactKind) -> str | None:
    """Capture the local model at finalization, tied to the saved query bytes.

    A render-only edit reuses the original definition snapshots. A model lookup
    failure writes explicit unavailability rather than retaining stale metadata.
    """
    from datus.tools.func_tool.report_artifact_tools import _atomic_write_text

    target = artifact_dir / "analysis" / "metric_snapshots.json"

    try:
        files = {
            p.relative_to(artifact_dir).as_posix(): p.read_text(encoding="utf-8")
            for p in iter_artifact_files(artifact_dir, artifact_kind, queries_only=True)
        }
        revision = query_revision(files)

        if target.is_file():
            try:
                if json.loads(target.read_text(encoding="utf-8"))["query_revision"] == revision:
                    return None
            except (ValueError, KeyError, TypeError):
                pass

        graphs = []
        if refs.metrics and semantic_tools is not None:
            try:
                if semantic_tools.runtime is not None:
                    graphs = run_async(semantic_tools.runtime.lineage_graph())
            except Exception:
                pass

        captured_at = datetime.now(timezone.utc).isoformat()
        snapshots = []
        for ref in refs.metrics:
            detail = None
            if semantic_tools is not None:
                try:
                    result = semantic_tools.get_metric(name=ref.name, path=ref.path)
                    if result.success and isinstance(result.result, dict):
                        candidate = result.result
                        if candidate.get("name") == ref.name and candidate.get("path") == ref.path:
                            detail = dict(candidate)
                            match = _metric_match(graphs, ref.name)
                            if match is not None:
                                detail["definition"] = match[1].get("detail", {})
                except Exception:
                    pass

            tables = metric_tables(graphs, ref.name) if detail else None
            snapshots.append(
                MetricSnapshot(
                    ref=ref,
                    captured_at=captured_at,
                    status="captured" if detail else "unavailable",
                    detail=detail,
                    tables=tables or [],
                    lineage_status="resolved" if tables is not None else "unavailable",
                ).model_dump()
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            target, json.dumps({"query_revision": revision, "metrics": snapshots}, ensure_ascii=False, indent=2)
        )
    except Exception as exc:
        return f"metric snapshots unavailable: {type(exc).__name__}"

    return None


def bake_reference_sql_snapshots(
    artifact_dir: Path, refs: SubjectRefs, context_search_tools, *, artifact_kind: ArtifactKind
) -> str | None:
    """Freeze referenced SQL at generation time, without running SQL or an LLM."""
    from datus.tools.func_tool.report_artifact_tools import _atomic_write_text

    target = artifact_dir / "analysis" / "reference_sql_snapshots.json"

    try:
        files = {
            p.relative_to(artifact_dir).as_posix(): p.read_text(encoding="utf-8")
            for p in iter_artifact_files(artifact_dir, artifact_kind, queries_only=True)
        }
        revision = query_revision(files)

        if target.is_file():
            try:
                if json.loads(target.read_text(encoding="utf-8"))["query_revision"] == revision:
                    return None
            except (ValueError, KeyError, TypeError):
                pass

        captured_at = datetime.now(timezone.utc).isoformat()
        snapshots = []
        for ref in refs.reference_sql:
            sql, summary = None, None
            if context_search_tools is not None:
                try:
                    result = context_search_tools.get_reference_sql(subject_path=ref.path, name=ref.name)
                    if result.success and isinstance(result.result, dict):
                        candidate = result.result
                        text = candidate.get("sql")
                        if candidate.get("name") == ref.name and isinstance(text, str) and text.strip():
                            sql = text
                            summary = candidate.get("summary") if isinstance(candidate.get("summary"), str) else None
                except Exception:
                    pass

            snapshots.append(
                ReferenceSqlSnapshot(
                    ref=ref,
                    captured_at=captured_at,
                    status="captured" if sql is not None else "unavailable",
                    sql=sql,
                    summary=summary,
                ).model_dump()
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            target,
            json.dumps({"query_revision": revision, "reference_sql": snapshots}, ensure_ascii=False, indent=2),
        )
    except Exception as exc:
        return f"reference SQL snapshots unavailable: {type(exc).__name__}"

    return None
