# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Async generators driving each step of ``/bootstrap-bi``.

Every ``stream_bi_*`` returns ``AsyncGenerator[ActionHistory, None]`` so
the unified :class:`InlineStreamingContext` daemon renders the pipeline
identically to ``/bootstrap``. Cross-stream state (qualified table list,
collected reference SQL paths, semantic-model validation result, metric
names) is threaded through a shared :class:`BiBuildState` instance —
this keeps ``ActionHistory.output`` clean of build artifacts.

The reference-SQL step delegates to ``bootstrap_streams.stream_reference_sql``
(per-item parallel ``SqlSummaryAgenticNode`` invocations) and then
post-processes each generated YAML file to extract the subject-path
identifiers needed by ``ScopedContext``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, List, Optional, Sequence

import yaml

from datus.cli.bootstrap_bi_subagents import (
    normalize_identifier,
    parse_subject_path_for_metrics,
    write_chart_sql_files,
    write_metrics_csv,
)
from datus.cli.bootstrap_streams import _run_helper_with_actions, _run_helper_with_events
from datus.cli.bootstrap_subagent import as_task_subagent, message_action
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionStatus,
)
from datus.schemas.batch_events import BatchEvent
from datus.tools.bi_tools.dashboard_assembler import SelectedSqlCandidate
from datus.utils.loggings import get_logger
from datus.utils.reference_paths import quote_path_segment

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Shared mutable state
# ─────────────────────────────────────────────────────────────────────


@dataclass
class BiBuildState:
    """Shared state threaded between streams (closure injection).

    Streams write into this dataclass instead of returning values, so the
    coordinator (`BootstrapBiCommands._run_plan`) can branch on the
    aggregated state (e.g. skip metrics when ``semantic_ok`` is False).
    """

    table_names: List[str] = field(default_factory=list)
    ref_sqls: List[str] = field(default_factory=list)
    semantic_ok: bool = False
    metrics: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# stream_bi_metadata — async DB crawl
# ─────────────────────────────────────────────────────────────────────


async def stream_bi_metadata(
    agent_config: AgentConfig,
    *,
    table_names: List[str],
    pool_size: int = 4,
) -> AsyncGenerator[ActionHistory, None]:
    """Crawl schema for the qualified ``table_names`` via ``init_local_schema_async``.

    The async helper does not currently support filtering by table list —
    it crawls the configured datasource end-to-end. We log the requested
    scope on entry so the operator can see what's expected.
    """
    from datus.storage.schema_metadata import create_metadata_rag
    from datus.storage.schema_metadata.local_init import init_local_schema_async
    from datus.tools.db_tools.db_manager import db_manager_instance

    if not table_names:
        yield message_action("No tables in scope; skipping metadata crawl.", status=ActionStatus.FAILED)
        return

    yield message_action(f"Crawling metadata for {len(table_names)} table(s)…")

    metadata_store = create_metadata_rag(agent_config)
    db_manager = db_manager_instance(agent_config.datasource_configs)

    async def _factory(emit: Callable[[BatchEvent], None]):
        return await init_local_schema_async(
            table_lineage_store=metadata_store,
            agent_config=agent_config,
            db_manager=db_manager,
            build_mode="incremental",
            table_type="full",
            pool_size=pool_size,
            emit=emit,
        )

    try:
        async for action in _run_helper_with_events(_factory, function_name="schema_crawl"):
            yield action
    except Exception as exc:
        logger.error("Metadata crawl failed: %s", exc, exc_info=True)
        yield message_action(f"Metadata crawl failed: {exc}", status=ActionStatus.FAILED)
        return

    yield message_action("Metadata crawl finished.")


# ─────────────────────────────────────────────────────────────────────
# stream_bi_reference_sql — wraps bootstrap.stream_reference_sql
# ─────────────────────────────────────────────────────────────────────


def _collect_ref_sqls_from_summary_files(
    summary_files: Sequence[str],
    agent_config: AgentConfig,
) -> List[str]:
    """Read each generated SQL summary YAML and build dotted reference paths.

    Mirrors the logic from the original ``BiDashboardCommands._gen_reference_sqls``
    (subject_tree + quoted name → dotted path) and dedupes the result.
    """
    sql_summary_root = agent_config.path_manager.sql_summary_path()
    seen: set[str] = set()
    out: List[str] = []
    for relative in summary_files:
        if not relative:
            continue
        candidate = Path(relative)
        if candidate.is_absolute():
            path = candidate
        elif len(candidate.parts) >= 2 and candidate.parts[0:2] == ("subject", "sql_summaries"):
            path = sql_summary_root.joinpath(*candidate.parts[2:])
        else:
            path = sql_summary_root / candidate
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning("Failed to read summary YAML %s: %s", path, exc)
            continue
        subject_tree = (doc.get("subject_tree") or "").strip()
        name = (doc.get("name") or "").strip()
        parts = [p.strip() for p in subject_tree.split("/") if p.strip()]
        if name:
            parts.append(quote_path_segment(name))
        if not parts:
            continue
        key = ".".join(parts)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


async def stream_bi_reference_sql(
    agent_config: AgentConfig,
    *,
    reference_sqls: Sequence[SelectedSqlCandidate],
    platform: str,
    dashboard_name: str,
    pool_size: int,
    state: BiBuildState,
) -> AsyncGenerator[ActionHistory, None]:
    """Materialize chart SQLs to disk then drive the per-item indexing pipeline.

    1. Write all selected chart SQLs into a single ``.sql`` file under
       ``dashboard_path/{platform}/``.
    2. Delegate to :func:`bootstrap_streams.stream_reference_sql` which runs
       :class:`SqlSummaryAgenticNode` per item with a ``pool_size`` semaphore.
    3. Observe each yielded ``sql_summary_response`` action — its output
       carries the relative path of the generated YAML, which we post-
       process into ``state.ref_sqls`` once the inner stream finishes.
    """
    from datus.cli.bootstrap_streams import stream_reference_sql

    if not reference_sqls:
        yield message_action("No reference SQL selected; skipping.")
        return

    sql_dir = write_chart_sql_files(
        reference_sqls,
        platform=platform,
        dashboard_name=dashboard_name,
        agent_config=agent_config,
    )
    if sql_dir is None:
        yield message_action("Reference SQL: no SQL written; skipping.", status=ActionStatus.FAILED)
        return

    yield message_action(f"Wrote {len(reference_sqls)} chart SQL(s) to `{sql_dir}`.")

    subject_tree_hint = f"{platform}/{normalize_identifier(dashboard_name or '', max_words=3, fallback='dashboard')}"
    extra_instructions = (
        f"IMPORTANT: All SQL summaries from this batch MUST use the SAME subject_tree classification. "
        f'Suggested subject_tree: "{subject_tree_hint}". '
        f"You may adjust the classification based on SQL content, but ensure consistency across all items."
    )

    summary_files: List[str] = []
    async for action in stream_reference_sql(
        agent_config,
        datasource=getattr(agent_config, "current_datasource", "") or "",
        sql_dir=str(sql_dir),
        pool_size=pool_size,
        build_mode="incremental",
        subject_tree=[subject_tree_hint],
        extra_instructions=extra_instructions,
    ):
        # ``sql_summary_response`` is the per-item terminal action emitted
        # inside each per-item subagent group; its output carries the
        # generated YAML path that we'll need for ScopedContext.
        if (
            action.action_type == "sql_summary_response"
            and isinstance(action.output, dict)
            and action.output.get("success")
        ):
            sf = action.output.get("sql_summary_file")
            if sf:
                summary_files.append(sf)
        yield action

    if summary_files:
        collected = _collect_ref_sqls_from_summary_files(summary_files, agent_config)
        state.ref_sqls.extend(collected)
        yield message_action(f"Collected {len(collected)} reference SQL identifier(s).")
    else:
        yield message_action("No SQL summaries generated.", status=ActionStatus.FAILED)


# ─────────────────────────────────────────────────────────────────────
# stream_bi_semantic_model — unified semantic authoring
# ─────────────────────────────────────────────────────────────────────


def _validate_semantic_model_sync(agent_config: AgentConfig, *, scope: str = "all") -> tuple[bool, Optional[str]]:
    """Run :class:`SemanticTools.validate_semantic` synchronously.

    Returns ``(ok, error_message)``. Adapter-loading or runtime errors are
    captured and surfaced as ``error_message`` rather than propagating, so
    the coordinator can yield a single failure message and gate metrics.
    Use ``scope="semantic_model"`` before metric generation so the expected
    no-metrics validation issue does not block the metrics step.
    """
    try:
        from datus.tools.func_tool.semantic_tools import SemanticTools
    except Exception as exc:
        return False, f"semantic_tools unavailable: {exc}"

    try:
        from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type

        adapter_type = resolve_semantic_adapter_type(agent_config)

        tools = SemanticTools(agent_config=agent_config, adapter_type=adapter_type)
        if not tools.adapter:
            return False, f"Semantic adapter not available. Install with: pip install datus-semantic-{adapter_type}"
        result = tools.validate_semantic(scope=scope)
        if not result.success:
            return False, result.error or "Semantic validation failed"
        return True, None
    except Exception as exc:
        logger.warning("Semantic model validation check failed: %s", exc)
        return False, f"Validation check failed: {exc}"


async def stream_bi_semantic_model(
    agent_config: AgentConfig,
    *,
    sqls: Sequence[SelectedSqlCandidate],
    platform: str,
    dashboard_name: str,
    state: BiBuildState,
) -> AsyncGenerator[ActionHistory, None]:
    """Author Dosi semantic models and metrics in one unified workflow."""
    if not sqls:
        yield message_action("No SQL queries for semantic model; skipping.", status=ActionStatus.FAILED)
        return

    csv_path = write_metrics_csv(
        sqls,
        platform=platform,
        dashboard_name=dashboard_name,
        agent_config=agent_config,
    )

    from datus.agent.node.semantic_authoring import ensure_semantic_agent_available
    from datus.storage.semantic_model.semantic_modeling_init import init_success_story_semantic_modeling_async

    try:
        ensure_semantic_agent_available("semantic_modeling", agent_config)
    except Exception as exc:
        state.semantic_ok = False
        yield message_action(str(exc), status=ActionStatus.FAILED)
        return

    subject_tree_hint = [
        platform,
        normalize_identifier(dashboard_name or "", max_words=3, fallback="dashboard"),
    ]
    captured: dict[str, Any] = {}

    async def _factory(emit: Callable[[BatchEvent], None], on_action: Callable[[ActionHistory], None]):
        ok, err, result = await init_success_story_semantic_modeling_async(
            agent_config,
            str(csv_path),
            subject_tree_hint,
            emit,
            build_mode="incremental",
            action_callback=on_action,
        )
        captured["ok"] = ok
        captured["err"] = err
        captured["result"] = result
        return ok

    async def _inner(_mgr: ActionHistoryManager):
        async for action in _run_helper_with_actions(_factory, function_name="semantic_modeling"):
            yield action

    async for action in as_task_subagent(
        subagent_type="semantic_modeling",
        description=dashboard_name or "<dashboard>",
        inner_factory=_inner,
    ):
        yield action

    if not captured.get("ok"):
        state.semantic_ok = False
        return

    ok, err = await asyncio.to_thread(_validate_semantic_model_sync, agent_config)
    state.semantic_ok = ok
    if ok:
        result = captured.get("result") or {}
        semantic_files = result.get("semantic_models", []) if isinstance(result, dict) else []
        metrics = _collect_metrics_from_semantic_models(semantic_files, agent_config)
        state.metrics.extend(metrics)
        yield message_action(f"Semantic layer validated; collected {len(metrics)} metric identifier(s).")
    else:
        yield message_action(f"Semantic layer validation failed: {err}", status=ActionStatus.FAILED)


# ─────────────────────────────────────────────────────────────────────
# Metric identifier collection and compatibility alias
# ─────────────────────────────────────────────────────────────────────


def _collect_metrics_from_semantic_models(
    semantic_files: Sequence[str],
    agent_config: AgentConfig,
) -> List[str]:
    """Resolve each generated semantic-model YAML and pull metric identifiers.

    Reuses :func:`generation_hooks.resolve_kb_sandbox_path` to keep metric
    paths inside the project's KB sandbox. Returns dotted ``subject.metric``
    identifiers ready for ``ScopedContext.metrics``.
    """
    from datus.cli.generation_hooks import resolve_kb_sandbox_path
    from datus.tools.semantic_tools.osi_document import load_osi_document

    knowledge_base_dir = str(agent_config.path_manager.subject_dir)
    out: set[str] = set()
    for relative in semantic_files:
        if not relative:
            continue
        resolved = resolve_kb_sandbox_path(relative, "metric", knowledge_base_dir)
        if not resolved:
            logger.warning("Skipping metric file outside sandbox: %s", relative)
            continue
        try:
            with open(resolved, "r", encoding="utf-8") as fh:
                docs = list(yaml.safe_load_all(fh))
        except Exception as exc:
            logger.warning("Failed to load semantic model YAML %s: %s", resolved, exc)
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            semantic_models = doc.get("semantic_model") or []
            if isinstance(semantic_models, dict):
                semantic_models = [semantic_models]
            for model in semantic_models:
                if not isinstance(model, dict) or not model.get("name"):
                    continue
                model_name = str(model["name"])
                try:
                    parsed = load_osi_document(resolved, model_name)
                except Exception as exc:
                    logger.warning("Failed to parse OSI/Dosi semantic model %s in %s: %s", model_name, resolved, exc)
                    continue
                for metric in parsed.metrics:
                    if not metric.name:
                        continue
                    subject_path = metric.subject_path or ["Metrics", metric.dataset or model_name]
                    quoted_path = [quote_path_segment(part) for part in subject_path]
                    out.add(".".join([*quoted_path, quote_path_segment(metric.name)]))

            meta = doc.get("metric") or {}
            name = meta.get("name")
            tags = meta.get("locked_metadata", {}).get("tags", [])
            subject_tree = parse_subject_path_for_metrics(tags)
            if name and subject_tree:
                out.add(f"{subject_tree}.{quote_path_segment(name)}")
    return list(out)


async def stream_bi_metrics(
    agent_config: AgentConfig,
    *,
    sqls: Sequence[SelectedSqlCandidate],
    platform: str,
    dashboard_name: str,
    state: BiBuildState,
) -> AsyncGenerator[ActionHistory, None]:
    """Compatibility alias; metrics are authored by ``semantic_modeling``."""
    del agent_config, sqls, platform, dashboard_name, state
    yield message_action("Metrics are authored by semantic_modeling; no separate metrics step is required.")


__all__ = [
    "BiBuildState",
    "stream_bi_metadata",
    "stream_bi_reference_sql",
    "stream_bi_semantic_model",
    "stream_bi_metrics",
]
