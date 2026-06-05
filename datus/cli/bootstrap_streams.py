# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Async generators that drive each ``/bootstrap`` panel.

Every ``stream_*`` returns ``AsyncGenerator[ActionHistory, None]`` so the
:class:`InlineStreamingContext` daemon can render every step uniformly.
The two reference-SQL / reference-template panels expose **per-item
parallelism**: each SQL summary invocation is its own ``task`` subagent
group, with N groups streaming concurrently in the UI.

Each stream function:
* takes ``datasource`` and pins ``agent_config.current_datasource`` to it
  before invoking any storage / DB code, so the user's tab selection
  drives connection scope reliably regardless of REPL state;
* takes ``build_mode`` (``"overwrite"`` | ``"incremental"``) and forwards
  it verbatim to the underlying ``init_*_async`` helper.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from datus.cli.bootstrap_subagent import as_task_subagent, message_action
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.batch_events import BatchEvent, BatchStage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────


_DONE = object()


def _set_current_datasource(agent_config: AgentConfig, datasource: str) -> None:
    """Pin ``agent_config.current_datasource`` so storage classes scope correctly.

    ``SemanticModelRAG`` / ``MetricRAG`` / ``ExtKnowledgeRAG`` all read
    this attribute at construction time, so callers must set it before
    instantiating any RAG. We avoid mutating it when ``datasource`` is
    falsy to keep the existing REPL state intact.
    """
    if datasource:
        agent_config.current_datasource = datasource


def _status_equals(action: ActionHistory, expected: ActionStatus) -> bool:
    return action.status == expected or action.status == expected.value


def _completed_subagent_description(action: ActionHistory, subagent_type: str) -> Optional[str]:
    """Return the task description for terminal bootstrap subagent actions."""
    if action.action_type != "task" or not str(action.action_id or "").startswith("complete_"):
        return None
    payload = action.input if isinstance(action.input, dict) else {}
    if payload.get("type") != subagent_type:
        return None
    description = payload.get("description")
    return str(description) if description is not None else None


async def merge_streams(
    *streams: AsyncGenerator[ActionHistory, None],
) -> AsyncGenerator[ActionHistory, None]:
    """Interleave N action streams via a shared queue."""
    if not streams:
        return

    queue: "asyncio.Queue[Any]" = asyncio.Queue()

    async def _feed(stream: AsyncGenerator[ActionHistory, None]) -> None:
        try:
            async for action in stream:
                await queue.put(action)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("merge_streams producer raised: %s", exc, exc_info=True)
        finally:
            await queue.put(_DONE)

    tasks = [asyncio.create_task(_feed(s)) for s in streams]
    remaining = len(tasks)

    try:
        while remaining > 0:
            item = await queue.get()
            if item is _DONE:
                remaining -= 1
                continue
            yield item
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _batch_event_to_action(event: BatchEvent, *, function_name: str = "batch") -> Optional[ActionHistory]:
    """Translate a :class:`BatchEvent` into a depth=1 :class:`ActionHistory`."""
    stage_value = event.stage.value if isinstance(event.stage, BatchStage) else str(event.stage)

    if stage_value == BatchStage.TASK_VALIDATED.value and event.total_items:
        return ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type=function_name,
            messages=f"validated {event.total_items} item(s)",
            input_data={"function_name": function_name},
            status=ActionStatus.SUCCESS,
        )
    if stage_value == BatchStage.TASK_COMPLETED.value:
        text = f"completed {event.completed_items or 0}/{event.total_items or 0}"
        if event.failed_items:
            text += f" ({event.failed_items} failed)"
        return ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type=function_name,
            messages=text,
            input_data={"function_name": function_name},
            status=ActionStatus.SUCCESS if not event.failed_items else ActionStatus.FAILED,
        )
    if stage_value == BatchStage.TASK_FAILED.value:
        return ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type=function_name,
            messages=event.error or "task failed",
            input_data={"function_name": function_name},
            status=ActionStatus.FAILED,
        )
    if stage_value == BatchStage.ITEM_FAILED.value and event.error:
        return ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type=function_name,
            messages=event.error,
            input_data={"function_name": function_name},
            status=ActionStatus.FAILED,
        )
    return None


async def _run_helper_with_events(
    helper_coro_factory: Callable[[Callable[[BatchEvent], None]], Awaitable[Any]],
    *,
    function_name: str = "batch",
) -> AsyncGenerator[ActionHistory, None]:
    """Run an existing ``init_*_async`` helper while streaming its BatchEvents."""
    queue: "asyncio.Queue[Any]" = asyncio.Queue()

    def _emit(event: BatchEvent) -> None:
        queue.put_nowait(event)

    helper_done: Tuple[bool, Optional[Exception]] = (False, None)

    async def _runner() -> None:
        nonlocal helper_done
        try:
            await helper_coro_factory(_emit)
            helper_done = (True, None)
        except Exception as exc:  # pragma: no cover - propagated below
            helper_done = (True, exc)
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(_runner())
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, BatchEvent):
                action = _batch_event_to_action(item, function_name=function_name)
                if action is not None:
                    yield action
    finally:
        await task

    _, exc = helper_done
    if exc is not None:
        raise exc


async def _run_helper_with_actions(
    helper_coro_factory: Callable[
        [Callable[[BatchEvent], None], Callable[[ActionHistory], None]],
        Awaitable[Any],
    ],
    *,
    function_name: str = "batch",
) -> AsyncGenerator[ActionHistory, None]:
    """Run an ``init_*_async`` helper that supports BOTH ``emit`` and ``action_callback``.

    Yields:
      * native :class:`ActionHistory` entries the helper's underlying
        :meth:`AgenticNode.execute_stream` produces (high detail, fine-grained
        LLM tool calls / assistant chunks);
      * translated :class:`BatchEvent` markers for coarse milestones
        (``task_started`` / ``task_validated`` / ``task_completed``).

    Native and translated entries are interleaved in arrival order via a
    single shared queue. Used by ``stream_semantic_model`` /
    ``stream_metrics`` / ``stream_knowledge`` / ``stream_reference_template``
    so the user sees actual node activity inside the ``task(...)`` subagent
    group, not just BatchEvent counters.
    """
    queue: "asyncio.Queue[Any]" = asyncio.Queue()

    def _emit(event: BatchEvent) -> None:
        queue.put_nowait(("event", event))

    def _on_action(action: ActionHistory) -> None:
        queue.put_nowait(("action", action))

    helper_done: Tuple[bool, Optional[Exception]] = (False, None)

    async def _runner() -> None:
        nonlocal helper_done
        try:
            await helper_coro_factory(_emit, _on_action)
            helper_done = (True, None)
        except Exception as exc:  # pragma: no cover - propagated below
            helper_done = (True, exc)
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(_runner())
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            kind, payload = item
            if kind == "action":
                yield payload
            elif kind == "event":
                translated = _batch_event_to_action(payload, function_name=function_name)
                if translated is not None:
                    yield translated
    finally:
        await task

    _, exc = helper_done
    if exc is not None:
        raise exc


# ─────────────────────────────────────────────────────────────────────
# stream_metadata — no LLM
# ─────────────────────────────────────────────────────────────────────


async def stream_metadata(
    agent_config: AgentConfig,
    *,
    datasource: str,
    build_mode: str = "incremental",
) -> AsyncGenerator[ActionHistory, None]:
    """Crawl the live database schema into the metadata RAG.

    Pure DB I/O — no AgenticNode invocation. ``table_type`` and
    ``pool_size`` are kept at sensible defaults; the simplified ``/bootstrap``
    Schema tab does not expose them.
    """
    from datus.storage.schema_metadata.local_init import init_local_schema_async
    from datus.storage.schema_metadata.store import SchemaWithValueRAG
    from datus.tools.db_tools.db_manager import db_manager_instance

    if not datasource:
        yield message_action("Schema: --datasource is required, skipping.", status=ActionStatus.FAILED)
        return

    _set_current_datasource(agent_config, datasource)

    yield message_action(f"Crawling schema from datasource `{datasource}` (mode={build_mode})…")

    metadata_store = SchemaWithValueRAG(agent_config)
    db_manager = db_manager_instance(agent_config.datasource_configs)

    async def _factory(emit):
        return await init_local_schema_async(
            table_lineage_store=metadata_store,
            agent_config=agent_config,
            db_manager=db_manager,
            build_mode=build_mode,
            table_type="full",
            pool_size=4,
            emit=emit,
        )

    try:
        async for action in _run_helper_with_events(_factory, function_name="schema_crawl"):
            yield action
    except Exception as exc:
        yield message_action(f"Schema crawl failed: {exc}", status=ActionStatus.FAILED)
        return

    yield message_action("Schema crawl finished.")


# ─────────────────────────────────────────────────────────────────────
# stream_reference_sql — per-item parallel subagents
# ─────────────────────────────────────────────────────────────────────


async def stream_reference_sql(
    agent_config: AgentConfig,
    *,
    datasource: str,
    sql_dir: str,
    pool_size: int = 3,
    build_mode: str = "incremental",
    subject_tree: Optional[List[str]] = None,
    extra_instructions: Optional[str] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Index every SQL file under ``sql_dir`` as a reference SQL.

    Each item is an independent ``task(gen_sql_summary, …)`` subagent
    group; up to ``pool_size`` groups stream concurrently. ``extra_instructions``
    is appended to the per-item ``user_message`` so callers can enforce
    cross-batch constraints (e.g. ``/bootstrap-bi`` pins all summaries to
    a single ``subject_tree``).
    """
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
    from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
    from datus.storage.reference_sql.init_utils import exists_reference_sql, gen_reference_sql_id
    from datus.storage.reference_sql.reference_sql_init import _sync_reference_sql_provenance
    from datus.storage.reference_sql.sql_file_processor import process_sql_files
    from datus.storage.reference_sql.store import ReferenceSqlRAG

    if not datasource:
        yield message_action("Reference SQL: --datasource is required, skipping.", status=ActionStatus.FAILED)
        return
    if not sql_dir:
        yield message_action("Reference SQL: --sql_dir is required, skipping.", status=ActionStatus.FAILED)
        return

    _set_current_datasource(agent_config, datasource)

    storage = ReferenceSqlRAG(agent_config)
    yield message_action(f"Discovering SQL files under `{sql_dir}` (mode={build_mode})…")

    valid_items, invalid_items = process_sql_files(sql_dir)
    if invalid_items:
        yield message_action(
            f"Skipping {len(invalid_items)} invalid SQL items.",
            status=ActionStatus.FAILED,
        )

    if build_mode == "overwrite":
        logger.info(
            "[overwrite] Wiping reference_sql store for datasource '%s' before re-population",
            agent_config.current_datasource,
        )
        storage.truncate()
        yield message_action("Reference SQL: cleared existing entries (overwrite mode).")

    if build_mode == "incremental":
        existing_ids = exists_reference_sql(storage, build_mode)
        valid_items = [it for it in valid_items if gen_reference_sql_id(it.get("sql", "")) not in existing_ids]

    if not valid_items:
        yield message_action("No new reference SQL items to process.")
        return

    yield message_action(
        f"Processing {len(valid_items)} SQL item(s) with concurrency={pool_size}.",
    )

    semaphore = asyncio.Semaphore(max(1, pool_size))

    def _build_inner(item: dict) -> Callable[[ActionHistoryManager], AsyncGenerator[ActionHistory, None]]:
        async def _factory(mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
            async with semaphore:
                node = SqlSummaryAgenticNode(
                    node_name="gen_sql_summary",
                    agent_config=agent_config,
                    execution_mode="workflow",
                    build_mode=build_mode,
                    subject_tree=subject_tree,
                )
                user_message = "Analyze and summarize this SQL query"
                if extra_instructions:
                    user_message = f"{user_message}\n\n## Additional Instructions\n{extra_instructions}"
                node.input = SqlSummaryNodeInput(
                    user_message=user_message,
                    sql_query=item.get("sql"),
                )
                async for action in node.execute_stream(mgr):
                    yield action

        return _factory

    def _item_description(item: dict) -> str:
        return str(item.get("filepath") or "<sql>")

    streams = [
        as_task_subagent(
            subagent_type="gen_sql_summary",
            description=_item_description(item),
            inner_factory=_build_inner(item),
        )
        for item in valid_items
    ]

    completed_by_description: Dict[str, ActionStatus] = {}
    failed_descriptions: Set[str] = set()

    async for action in merge_streams(*streams):
        description = _completed_subagent_description(action, "gen_sql_summary")
        if description is not None:
            if _status_equals(action, ActionStatus.FAILED):
                completed_by_description[description] = ActionStatus.FAILED
                failed_descriptions.add(description)
            elif _status_equals(action, ActionStatus.SUCCESS) and description not in failed_descriptions:
                completed_by_description[description] = ActionStatus.SUCCESS
        yield action

    successful_items = [
        item for item in valid_items if completed_by_description.get(_item_description(item)) == ActionStatus.SUCCESS
    ]
    provenance_entries = _sync_reference_sql_provenance(agent_config, successful_items)
    if provenance_entries:
        yield message_action(f"Synced {provenance_entries} reference SQL provenance row(s).")

    storage.after_init()
    yield message_action(f"Indexed {len(successful_items)} reference SQL item(s).")


# ─────────────────────────────────────────────────────────────────────
# stream_reference_template — same shape as reference_sql
# ─────────────────────────────────────────────────────────────────────


async def stream_reference_template(
    agent_config: AgentConfig,
    *,
    datasource: str,
    template_dir: str,
    pool_size: int = 3,
    build_mode: str = "incremental",
    subject_tree: Optional[List[str]] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Reference-template equivalent of :func:`stream_reference_sql`."""
    from datus.storage.reference_template.reference_template_init import init_reference_template_async
    from datus.storage.reference_template.store import ReferenceTemplateRAG

    if not datasource:
        yield message_action("Reference Template: --datasource is required, skipping.", status=ActionStatus.FAILED)
        return
    if not template_dir:
        yield message_action("Reference Template: --template_dir is required, skipping.", status=ActionStatus.FAILED)
        return

    _set_current_datasource(agent_config, datasource)

    storage = ReferenceTemplateRAG(agent_config)
    yield message_action(f"Indexing templates under `{template_dir}` (mode={build_mode})…")

    async def _factory(emit, on_action):
        return await init_reference_template_async(
            storage=storage,
            global_config=agent_config,
            template_dir=template_dir,
            validate_only=False,
            build_mode=build_mode,
            pool_size=pool_size,
            subject_tree=subject_tree,
            emit=emit,
            action_callback=on_action,
        )

    async def _inner(_mgr):
        async for action in _run_helper_with_actions(_factory, function_name="gen_template_summary"):
            yield action

    async for action in as_task_subagent(
        subagent_type="reference_template",
        description=template_dir,
        inner_factory=_inner,
    ):
        yield action


# ─────────────────────────────────────────────────────────────────────
# stream_semantic_model — single batched subagent
# ─────────────────────────────────────────────────────────────────────


async def stream_semantic_model(
    agent_config: AgentConfig,
    *,
    datasource: str,
    success_story: str,
    build_mode: str = "incremental",
) -> AsyncGenerator[ActionHistory, None]:
    """Generate the semantic model from a success-story CSV.

    Wraps :class:`GenSemanticModelAgenticNode` as one
    ``task(gen_semantic_model, …)`` subagent group. ``build_mode`` is
    forwarded to ``init_success_story_semantic_model_async``: in
    ``incremental`` mode the helper short-circuits without an LLM call
    when the store is already populated.
    """
    if not datasource:
        yield message_action("Semantic Model: --datasource is required, skipping.", status=ActionStatus.FAILED)
        return
    if not success_story:
        yield message_action("Semantic Model: --success_story is required, skipping.", status=ActionStatus.FAILED)
        return

    _set_current_datasource(agent_config, datasource)

    from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

    async def _factory(emit, on_action):
        return await init_success_story_semantic_model_async(
            agent_config,
            success_story,
            emit,
            build_mode=build_mode,
            action_callback=on_action,
        )

    async def _inner(_mgr):
        async for action in _run_helper_with_actions(_factory, function_name="gen_semantic_model"):
            yield action

    async for action in as_task_subagent(
        subagent_type="gen_semantic_model",
        description=f"{success_story} (mode={build_mode})",
        inner_factory=_inner,
    ):
        yield action


# ─────────────────────────────────────────────────────────────────────
# stream_metrics — single batched subagent
# ─────────────────────────────────────────────────────────────────────


async def stream_metrics(
    agent_config: AgentConfig,
    *,
    datasource: str,
    success_story: str,
    pool_size: int = 5,
    build_mode: str = "incremental",
    subject_tree: Optional[List[str]] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Wrap :class:`GenMetricsAgenticNode` invocation as a ``gen_metrics`` task group.

    For metrics, ``pool_size`` controls SQL queries per generation batch. The
    name is kept for CLI/TUI parity with other bootstrap components.
    """
    if not datasource:
        yield message_action("Metrics: --datasource is required, skipping.", status=ActionStatus.FAILED)
        return
    if not success_story:
        yield message_action("Metrics: --success_story is required, skipping.", status=ActionStatus.FAILED)
        return

    _set_current_datasource(agent_config, datasource)

    from datus.storage.metric.metric_init import init_success_story_metrics_async

    async def _factory(emit, on_action):
        return await init_success_story_metrics_async(
            agent_config=agent_config,
            success_story=success_story,
            subject_tree=subject_tree,
            emit=emit,
            build_mode=build_mode,
            action_callback=on_action,
            batch_size=max(1, int(pool_size)),
        )

    async def _inner(_mgr):
        async for action in _run_helper_with_actions(_factory, function_name="gen_metrics"):
            yield action

    async for action in as_task_subagent(
        subagent_type="gen_metrics",
        description=f"{success_story} (mode={build_mode})",
        inner_factory=_inner,
    ):
        yield action


# ─────────────────────────────────────────────────────────────────────
# stream_knowledge — LLM-driven generation from a success story
# ─────────────────────────────────────────────────────────────────────


async def stream_knowledge(
    agent_config: AgentConfig,
    *,
    datasource: str,
    success_story: str,
    pool_size: int = 4,  # noqa: ARG001 - reserved for future per-row parallelism
    build_mode: str = "incremental",
    subject_tree: Optional[List[str]] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """LLM-driven external knowledge generation from a success-story CSV.

    The legacy CSV-direct-import path (``ext_knowledge_csv``) was dropped
    in the simplified ``/bootstrap`` form: users now convert any direct
    knowledge CSV into a success story before invoking this stream.
    """
    del pool_size  # reserved for per-row parallelism; current impl is sequential
    if not datasource:
        yield message_action("Knowledge: --datasource is required, skipping.", status=ActionStatus.FAILED)
        return
    if not success_story:
        yield message_action("Knowledge: --success_story is required, skipping.", status=ActionStatus.FAILED)
        return

    _set_current_datasource(agent_config, datasource)

    from datus.storage.ext_knowledge.ext_knowledge_init import init_success_story_knowledge_async

    async def _factory(_emit, on_action):
        ok, err = await init_success_story_knowledge_async(
            agent_config=agent_config,
            success_story=success_story,
            subject_tree=subject_tree,
            build_mode=build_mode,
            action_callback=on_action,
        )
        if not ok and err:
            # Surface the helper's error string as a final FAILED action so
            # ``as_task_subagent`` flips the group's overall status.
            on_action(
                ActionHistory.create_action(
                    role=ActionRole.TOOL,
                    action_type="gen_ext_knowledge",
                    messages=err,
                    input_data={"function_name": "gen_ext_knowledge"},
                    status=ActionStatus.FAILED,
                )
            )

    async def _inner(_mgr):
        async for action in _run_helper_with_actions(_factory, function_name="gen_ext_knowledge"):
            yield action

    async for action in as_task_subagent(
        subagent_type="gen_ext_knowledge",
        description=f"{success_story} (mode={build_mode})",
        inner_factory=_inner,
    ):
        yield action


__all__ = [
    "merge_streams",
    "stream_metadata",
    "stream_reference_sql",
    "stream_reference_template",
    "stream_semantic_model",
    "stream_metrics",
    "stream_knowledge",
]
