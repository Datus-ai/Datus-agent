"""Service for knowledge base bootstrap with SSE streaming."""

import asyncio
import os
import types
from datetime import datetime
from typing import AsyncGenerator, Optional

from datus.api.models.kb_models import (
    BootstrapDocInput,
    BootstrapKbEvent,
    BootstrapKbInput,
    KbComponent,
)
from datus.configuration.agent_config import AgentConfig
from datus.schemas.batch_events import BatchEvent, BatchStage
from datus.storage.ext_knowledge.ext_knowledge_init import (
    init_ext_knowledge,
    init_success_story_knowledge,
)
from datus.storage.ext_knowledge.store import ExtKnowledgeRAG
from datus.storage.metric.metric_init import init_success_story_metrics
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql import ReferenceSqlRAG
from datus.storage.reference_sql.reference_sql_init import init_reference_sql
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.schema_metadata.local_init import init_local_schema
from datus.storage.semantic_model.semantic_model_init import (
    init_success_story_semantic_model,
)
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Sentinel signalling that a component is done pushing events
_COMPONENT_DONE = object()


class KbService:
    """Wraps knowledge base bootstrap logic with streaming support."""

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def bootstrap_stream(
        self,
        request: BootstrapKbInput,
        stream_id: str,
        cancel_event: asyncio.Event,
        project_root: str,
    ) -> AsyncGenerator[BootstrapKbEvent, None]:
        """Run bootstrap components sequentially, yielding SSE events."""

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        summary: dict[str, dict] = {}

        for comp_name in request.components:
            if cancel_event.is_set():
                yield self._make_event(
                    stream_id,
                    comp_name,
                    BatchStage.TASK_FAILED,
                    error="Cancelled by user",
                )
                break
            # Run the sync init in a background thread.
            # Stream BatchEvents from the queue in real-time while the thread runs.
            future = loop.run_in_executor(
                None,
                self._run_component,
                request,
                comp_name,
                queue,
                loop,
                cancel_event,
                project_root,
            )

            # Consume events as they arrive until the thread signals completion
            result = None
            component_error = None
            while True:
                if future.done() and queue.empty():
                    # Thread finished and queue drained
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No event yet — check if thread is still alive
                    if future.done():
                        break
                    continue

                if item is _COMPONENT_DONE:
                    break
                yield self._batch_event_to_sse(stream_id, comp_name, item)

            # Collect the thread result
            try:
                result = await future
            except Exception as exc:
                logger.exception(f"Component {comp_name} failed")
                component_error = exc

            # Drain any remaining events
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is _COMPONENT_DONE:
                    break
                yield self._batch_event_to_sse(stream_id, comp_name, item)

            # Final per-component event
            if component_error:
                yield self._make_event(stream_id, comp_name, BatchStage.TASK_FAILED, error=str(component_error))
                summary[comp_name] = {"status": "failed", "message": str(component_error)}
            else:
                status = result.get("status", "success") if isinstance(result, dict) else "success"
                message = result.get("message", "") if isinstance(result, dict) else str(result)
                error_msg = result.get("error") if isinstance(result, dict) else None

                if status == "failed":
                    yield self._make_event(
                        stream_id, comp_name, BatchStage.TASK_FAILED, message=message, error=error_msg
                    )
                else:
                    yield self._make_event(
                        stream_id,
                        comp_name,
                        BatchStage.TASK_COMPLETED,
                        message=message,
                        payload=result if isinstance(result, dict) else None,
                    )
                summary[comp_name] = (
                    result if isinstance(result, dict) else {"status": "success", "message": str(result)}
                )

        # Final stream-end event
        yield self._make_event(
            stream_id,
            "all",
            BatchStage.TASK_COMPLETED,
            message="Bootstrap complete",
            payload={"components": summary},
        )

    # ------------------------------------------------------------------
    # Per-component dispatch (runs in a worker thread)
    # ------------------------------------------------------------------

    def _run_component(
        self,
        request: BootstrapKbInput,
        component: str,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        cancel_event: asyncio.Event,
        project_root: str,
    ) -> dict:
        config = self.agent_config
        strategy = request.strategy
        pool_size = 1
        dir_path = config.rag_storage_path()

        # Thread-safe emit that pushes BatchEvent into the async queue
        def emit(event: BatchEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        # Build a SimpleNamespace that mimics argparse.Namespace for init functions
        args = self._build_args(request, project_root)

        subject_tree = request.subject_tree

        try:
            if component == KbComponent.METADATA:
                return self._init_metadata(config, strategy, pool_size, dir_path, args, emit)

            elif component == KbComponent.SEMANTIC_MODEL:
                return self._init_semantic_model(config, strategy, dir_path, args, emit)

            elif component == KbComponent.METRICS:
                return self._init_metrics(config, strategy, dir_path, args, subject_tree, emit)

            elif component == KbComponent.EXT_KNOWLEDGE:
                return self._init_ext_knowledge(config, strategy, pool_size, dir_path, args, subject_tree)

            elif component == KbComponent.REFERENCE_SQL:
                return self._init_reference_sql(config, strategy, pool_size, dir_path, args, subject_tree, emit)

            else:
                return {"status": "failed", "message": f"Unknown component: {component}"}
        finally:
            # Signal the event-loop drain loop that this component is done
            loop.call_soon_threadsafe(queue.put_nowait, _COMPONENT_DONE)

    # ------------------------------------------------------------------
    # Component init methods
    # ------------------------------------------------------------------

    def _init_metadata(
        self,
        config: AgentConfig,
        strategy: str,
        pool_size: int,
        dir_path: str,
        args: types.SimpleNamespace,
        emit=None,
    ) -> dict:
        if strategy == "check":
            store = SchemaWithValueRAG(config)
            return {
                "status": "success",
                "message": f"metadata already built, schema_size={store.get_schema_size()}, "
                f"value_size={store.get_value_size()}",
            }

        if strategy == "overwrite":
            store = SchemaWithValueRAG(config)
            store.truncate()
            logger.info("Truncated schema metadata tables for overwrite")

        store = SchemaWithValueRAG(config)
        db_manager = DBManager(config.datasource_configs)
        init_local_schema(
            store,
            config,
            db_manager,
            strategy,
            table_type=args.schema_linking_type,
            init_catalog_name=args.catalog,
            init_database_name=args.database_name,
            pool_size=pool_size,
            emit=emit,
        )
        return {
            "status": "success",
            "message": f"metadata bootstrap completed, schema_size={store.get_schema_size()}, "
            f"value_size={store.get_value_size()}",
        }

    def _init_semantic_model(
        self,
        config: AgentConfig,
        strategy: str,
        dir_path: str,
        args: types.SimpleNamespace,
        emit,
    ) -> dict:
        successful, error_message = init_success_story_semantic_model(config, args.success_story, emit=emit)
        if successful:
            rag = SemanticModelRAG(config)
            return {
                "status": "success",
                "message": f"semantic_model bootstrap completed, semantic_object_count={rag.get_size()}",
                "error": error_message,
            }
        return {"status": "failed", "message": error_message}

    def _init_metrics(
        self,
        config: AgentConfig,
        strategy: str,
        dir_path: str,
        args: types.SimpleNamespace,
        subject_tree: Optional[list],
        emit,
    ) -> dict:
        successful, error_message, _ = init_success_story_metrics(config, args.success_story, subject_tree, emit=emit)
        if successful:
            rag = MetricRAG(config)
            return {
                "status": "success",
                "message": f"metrics bootstrap completed, metrics_count={rag.get_metrics_size()}",
                "error": error_message,
            }
        return {"status": "failed", "message": error_message}

    def _init_ext_knowledge(
        self,
        config: AgentConfig,
        strategy: str,
        pool_size: int,
        dir_path: str,
        args: types.SimpleNamespace,
        subject_tree: Optional[list],
    ) -> dict:
        rag = ExtKnowledgeRAG(config)

        if hasattr(args, "ext_knowledge") and args.ext_knowledge:
            init_ext_knowledge(rag.store, args, build_mode=strategy, pool_size=pool_size)
        elif hasattr(args, "success_story") and args.success_story:
            successful, error_message = init_success_story_knowledge(config, args.success_story, subject_tree)
            if not successful:
                return {"status": "failed", "message": error_message}

        return {
            "status": "success",
            "message": f"ext_knowledge bootstrap completed, knowledge_size={rag.store.table_size()}",
        }

    def _init_reference_sql(
        self,
        config: AgentConfig,
        strategy: str,
        pool_size: int,
        dir_path: str,
        args: types.SimpleNamespace,
        subject_tree: Optional[list],
        emit,
    ) -> dict:
        store = ReferenceSqlRAG(config)
        result = init_reference_sql(
            store,
            config,
            args.sql_dir,
            validate_only=False,
            build_mode=strategy,
            pool_size=pool_size,
            subject_tree=subject_tree,
            emit=emit,
        )
        return result if isinstance(result, dict) else {"status": "success", "message": str(result)}

    # ------------------------------------------------------------------
    # Platform document bootstrap
    # ------------------------------------------------------------------

    async def bootstrap_doc_stream(
        self,
        request: BootstrapDocInput,
        stream_id: str,
        cancel_event: asyncio.Event,
    ) -> AsyncGenerator[BootstrapKbEvent, None]:
        """Run platform doc bootstrap, yielding SSE events."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        component = "platform_doc"

        future = loop.run_in_executor(
            None,
            self._run_doc_init,
            request,
            queue,
            loop,
            cancel_event,
        )

        # Drain queue (same pattern as bootstrap_stream)
        while True:
            if future.done() and queue.empty():
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if future.done():
                    break
                continue
            if item is _COMPONENT_DONE:
                break
            yield self._batch_event_to_sse(stream_id, component, item)

        # Collect result
        result = None
        component_error = None
        try:
            result = await future
        except Exception as exc:
            logger.exception("Platform doc bootstrap failed")
            component_error = exc

        # Drain remaining events
        while not queue.empty():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is _COMPONENT_DONE:
                break
            yield self._batch_event_to_sse(stream_id, component, item)

        # Final event
        if component_error:
            yield self._make_event(stream_id, component, BatchStage.TASK_FAILED, error=str(component_error))
        elif result and result.success:
            yield self._make_event(
                stream_id,
                component,
                BatchStage.TASK_COMPLETED,
                message=f"Processed {result.total_docs} docs, {result.total_chunks} chunks",
                payload={
                    "platform": result.platform,
                    "version": result.version,
                    "total_docs": result.total_docs,
                    "total_chunks": result.total_chunks,
                    "duration_seconds": result.duration_seconds,
                    "errors": result.errors,
                },
            )
        else:
            error_msg = "; ".join(result.errors) if result and result.errors else "Unknown error"
            yield self._make_event(stream_id, component, BatchStage.TASK_FAILED, error=error_msg)

    def _run_doc_init(
        self,
        request: BootstrapDocInput,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        cancel_event: asyncio.Event,
    ):
        """Sync worker for platform doc bootstrap (runs in executor thread)."""
        from datus.configuration.agent_config import DocumentConfig
        from datus.storage.document.doc_init import init_platform_docs

        config = self.agent_config
        platform = request.platform

        # Resolve DocumentConfig: YAML base + API overrides
        base_cfg = config.document_configs.get(platform, DocumentConfig())
        merged_cfg = self._merge_doc_overrides(base_cfg, request)

        # Thread-safe emit bridging to async queue
        def emit(event: BatchEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        # Cancel bridge: asyncio.Event -> sync callable
        def cancel_check() -> bool:
            return cancel_event.is_set()

        try:
            return init_platform_docs(
                platform=platform,
                cfg=merged_cfg,
                build_mode=request.build_mode,
                pool_size=request.pool_size,
                emit=emit,
                cancel_check=cancel_check,
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _COMPONENT_DONE)

    @staticmethod
    def _merge_doc_overrides(base, request: BootstrapDocInput):
        """Overlay non-None API request fields onto the YAML-based DocumentConfig."""
        import dataclasses

        field_map = {
            "source_type": "type",
            "source": "source",
            "version": "version",
            "github_ref": "github_ref",
            "github_token": "github_token",
            "paths": "paths",
            "chunk_size": "chunk_size",
            "max_depth": "max_depth",
            "include_patterns": "include_patterns",
            "exclude_patterns": "exclude_patterns",
        }
        overrides = {}
        for req_field, cfg_field in field_map.items():
            val = getattr(request, req_field, None)
            if val is not None:
                overrides[cfg_field] = val
        return dataclasses.replace(base, **overrides)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_args(request: BootstrapKbInput, project_root: str) -> types.SimpleNamespace:
        """Create a SimpleNamespace mimicking argparse.Namespace for datus init functions."""
        # Resolve relative paths against the project root
        success_story = os.path.join(project_root, request.success_story) if request.success_story else None
        sql_dir = os.path.join(project_root, request.sql_dir) if request.sql_dir else None
        ext_knowledge = os.path.join(project_root, request.ext_knowledge) if request.ext_knowledge else None

        return types.SimpleNamespace(
            success_story=success_story,
            sql_dir=sql_dir,
            ext_knowledge=ext_knowledge,
            schema_linking_type=request.schema_linking_type,
            catalog=request.catalog or "",
            database_name=request.database_name or "",
            pool_size=1,
            kb_update_strategy=request.strategy,
            validate_only=False,
        )

    @staticmethod
    def _make_event(
        stream_id: str,
        component: str,
        stage: BatchStage,
        message: Optional[str] = None,
        error: Optional[str] = None,
        progress: Optional[dict] = None,
        payload: Optional[dict] = None,
    ) -> BootstrapKbEvent:
        return BootstrapKbEvent(
            stream_id=stream_id,
            component=component,
            stage=stage.value if isinstance(stage, BatchStage) else stage,
            message=message,
            error=error,
            progress=progress,
            payload=payload,
            timestamp=datetime.now().isoformat(),
        )

    @staticmethod
    def _batch_event_to_sse(stream_id: str, component: str, event: BatchEvent) -> BootstrapKbEvent:
        """Convert a datus BatchEvent into our SSE envelope."""
        progress = None
        if event.total_items is not None:
            progress = {
                "total": event.total_items,
                "completed": event.completed_items or 0,
                "failed": event.failed_items or 0,
            }
        return BootstrapKbEvent(
            stream_id=stream_id,
            component=component,
            stage=event.stage if isinstance(event.stage, str) else event.stage.value,
            message=event.message,
            error=event.error,
            progress=progress,
            payload=event.payload,
            timestamp=event.timestamp.isoformat() if event.timestamp else datetime.now().isoformat(),
        )
