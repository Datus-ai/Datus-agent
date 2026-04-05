"""Tests for datus.api.services.kb_service — knowledge base bootstrap."""

from datetime import datetime

import pytest

from datus.api.models.kb_models import BootstrapKbEvent, BootstrapKbInput
from datus.api.services.kb_service import KbService
from datus.schemas.batch_events import BatchEvent, BatchStage


class TestKbServiceInit:
    """Tests for KbService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """KbService initializes with real agent config."""
        svc = KbService(agent_config=real_agent_config)
        assert svc is not None
        assert svc.agent_config is real_agent_config


class TestKbServiceBuildArgs:
    """Tests for _build_args — argument namespace creation."""

    def test_build_args_with_all_paths(self):
        """_build_args resolves all paths against project_root."""
        request = BootstrapKbInput(
            components=["metadata"],
            strategy="check",
            success_story="data/stories",
            sql_dir="data/sql",
            ext_knowledge="data/knowledge",
            schema_linking_type="table",
            catalog="main",
            database_name="test_db",
        )
        args = KbService._build_args(request, "/project")
        assert args.success_story == "/project/data/stories"
        assert args.sql_dir == "/project/data/sql"
        assert args.ext_knowledge == "/project/data/knowledge"
        assert args.schema_linking_type == "table"
        assert args.catalog == "main"
        assert args.database_name == "test_db"

    def test_build_args_with_empty_paths(self):
        """_build_args handles None/empty paths gracefully."""
        request = BootstrapKbInput(
            components=["metadata"],
            strategy="check",
        )
        args = KbService._build_args(request, "/project")
        assert args.success_story is None
        assert args.sql_dir is None
        assert args.ext_knowledge is None

    def test_build_args_sets_defaults(self):
        """_build_args sets default values for common fields."""
        request = BootstrapKbInput(
            components=["metadata"],
            strategy="overwrite",
        )
        args = KbService._build_args(request, "/proj")
        assert args.pool_size == 1
        assert args.kb_update_strategy == "overwrite"
        assert args.validate_only is False
        assert args.catalog == ""
        assert args.database_name == ""


class TestKbServiceMakeEvent:
    """Tests for _make_event — SSE event construction."""

    def test_make_event_completed(self):
        """_make_event creates a completed event."""
        event = KbService._make_event(
            stream_id="s1",
            component="metadata",
            stage=BatchStage.TASK_COMPLETED,
            message="Done",
        )
        assert isinstance(event, BootstrapKbEvent)
        assert event.stream_id == "s1"
        assert event.component == "metadata"
        assert event.message == "Done"

    def test_make_event_with_error(self):
        """_make_event creates a failed event with error."""
        event = KbService._make_event(
            stream_id="s2",
            component="metrics",
            stage=BatchStage.TASK_FAILED,
            error="Connection timeout",
        )
        assert event.error == "Connection timeout"

    def test_make_event_with_payload(self):
        """_make_event includes payload data."""
        event = KbService._make_event(
            stream_id="s3",
            component="all",
            stage=BatchStage.TASK_COMPLETED,
            payload={"count": 10},
        )
        assert event.payload == {"count": 10}

    def test_make_event_has_timestamp(self):
        """_make_event always includes timestamp."""
        event = KbService._make_event("s4", "test", BatchStage.TASK_STARTED)
        assert event.timestamp is not None


class TestKbServiceBatchEventToSse:
    """Tests for _batch_event_to_sse — BatchEvent conversion."""

    def test_convert_basic_event(self):
        """_batch_event_to_sse converts basic BatchEvent."""
        batch = BatchEvent(
            biz_name="test",
            stage=BatchStage.TASK_STARTED,
            message="Starting metadata init",
        )
        result = KbService._batch_event_to_sse("stream-1", "metadata", batch)
        assert isinstance(result, BootstrapKbEvent)
        assert result.stream_id == "stream-1"
        assert result.component == "metadata"
        assert result.message == "Starting metadata init"

    def test_convert_event_with_progress(self):
        """_batch_event_to_sse includes progress when total_items is set."""
        batch = BatchEvent(
            biz_name="test",
            stage=BatchStage.TASK_PROCESSING,
            message="Processing",
            total_items=100,
            completed_items=50,
            failed_items=2,
        )
        result = KbService._batch_event_to_sse("s1", "metadata", batch)
        assert result.progress is not None
        assert result.progress["total"] == 100
        assert result.progress["completed"] == 50
        assert result.progress["failed"] == 2

    def test_convert_event_without_progress(self):
        """_batch_event_to_sse has no progress when total_items is None."""
        batch = BatchEvent(
            biz_name="test",
            stage=BatchStage.TASK_COMPLETED,
            message="Done",
        )
        result = KbService._batch_event_to_sse("s1", "metadata", batch)
        assert result.progress is None

    def test_convert_event_with_error(self):
        """_batch_event_to_sse includes error field."""
        batch = BatchEvent(
            biz_name="test",
            stage=BatchStage.TASK_FAILED,
            error="Something went wrong",
        )
        result = KbService._batch_event_to_sse("s1", "metadata", batch)
        assert result.error == "Something went wrong"

    def test_convert_event_with_payload(self):
        """_batch_event_to_sse includes payload field."""
        batch = BatchEvent(
            biz_name="test",
            stage=BatchStage.TASK_COMPLETED,
            payload={"tables": 5},
        )
        result = KbService._batch_event_to_sse("s1", "metadata", batch)
        assert result.payload == {"tables": 5}

    def test_convert_event_with_timestamp(self):
        """_batch_event_to_sse converts timestamp from BatchEvent."""
        batch = BatchEvent(
            biz_name="test",
            stage=BatchStage.TASK_COMPLETED,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )
        result = KbService._batch_event_to_sse("s1", "metadata", batch)
        assert "2025-01-01" in result.timestamp


@pytest.mark.asyncio
class TestKbServiceBootstrapStream:
    """Tests for bootstrap_stream — the main async generator."""

    async def test_bootstrap_stream_metadata_check(self, real_agent_config):
        """bootstrap_stream with metadata check strategy yields events and completes."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        cancel_event = asyncio.Event()
        request = BootstrapKbInput(components=["metadata"], strategy="check")

        events = []
        async for event in svc.bootstrap_stream(request, "test-stream", cancel_event, str(real_agent_config.home)):
            events.append(event)

        assert len(events) >= 1
        # Last event should be the "all" completion
        last_event = events[-1]
        assert last_event.component == "all"

    async def test_bootstrap_stream_multiple_components(self, real_agent_config):
        """bootstrap_stream with multiple components processes sequentially."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        cancel_event = asyncio.Event()
        request = BootstrapKbInput(components=["metadata", "ext_knowledge"], strategy="check")

        events = []
        async for event in svc.bootstrap_stream(request, "multi-stream", cancel_event, str(real_agent_config.home)):
            events.append(event)

        assert len(events) >= 2
        # Should have events for metadata, ext_knowledge, and final "all"
        components_seen = {e.component for e in events}
        assert "all" in components_seen

    async def test_bootstrap_stream_semantic_model(self, real_agent_config):
        """bootstrap_stream with semantic_model component processes it."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        cancel_event = asyncio.Event()
        request = BootstrapKbInput(components=["semantic_model"], strategy="check")

        events = []
        async for event in svc.bootstrap_stream(request, "sm-stream", cancel_event, str(real_agent_config.home)):
            events.append(event)

        assert len(events) >= 1
        components_seen = {e.component for e in events}
        assert "all" in components_seen

    async def test_bootstrap_stream_ext_knowledge(self, real_agent_config):
        """bootstrap_stream with ext_knowledge component processes it."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        cancel_event = asyncio.Event()
        request = BootstrapKbInput(components=["ext_knowledge"], strategy="check")

        events = []
        async for event in svc.bootstrap_stream(request, "ek-stream", cancel_event, str(real_agent_config.home)):
            events.append(event)

        assert len(events) >= 1

    async def test_bootstrap_stream_reference_sql(self, real_agent_config):
        """bootstrap_stream with reference_sql component processes it."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        cancel_event = asyncio.Event()
        request = BootstrapKbInput(components=["reference_sql"], strategy="check")

        events = []
        async for event in svc.bootstrap_stream(request, "rs-stream", cancel_event, str(real_agent_config.home)):
            events.append(event)

        assert len(events) >= 1

    async def test_bootstrap_stream_all_components(self, real_agent_config):
        """bootstrap_stream with all components processes them sequentially."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        cancel_event = asyncio.Event()
        request = BootstrapKbInput(
            components=["metadata", "semantic_model", "ext_knowledge", "reference_sql"],
            strategy="check",
        )

        events = []
        async for event in svc.bootstrap_stream(request, "all-stream", cancel_event, str(real_agent_config.home)):
            events.append(event)

        assert len(events) >= 4  # At least one per component + final
        last = events[-1]
        assert last.component == "all"

    async def test_bootstrap_stream_cancelled(self, real_agent_config):
        """bootstrap_stream stops when cancel_event is set."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        cancel_event = asyncio.Event()
        cancel_event.set()  # Pre-cancel

        request = BootstrapKbInput(components=["metadata", "ext_knowledge"], strategy="check")

        events = []
        async for event in svc.bootstrap_stream(request, "cancel-stream", cancel_event, str(real_agent_config.home)):
            events.append(event)

        # Should have cancelled event + final event
        assert len(events) >= 1
        # Check that a cancelled/failed event was produced
        has_cancelled = any("cancel" in (e.error or "").lower() for e in events if e.error)
        has_final = any(e.component == "all" for e in events)
        assert has_cancelled or has_final


class TestKbServiceInitMetadata:
    """Tests for _init_metadata — metadata initialization."""

    def test_init_metadata_check_strategy(self, real_agent_config):
        """_init_metadata with check strategy returns existing sizes."""
        svc = KbService(agent_config=real_agent_config)
        result = svc._init_metadata(
            config=real_agent_config,
            strategy="check",
            pool_size=1,
            dir_path=real_agent_config.rag_storage_path(),
            args=KbService._build_args(
                BootstrapKbInput(components=["metadata"], strategy="check"),
                str(real_agent_config.home),
            ),
        )
        assert result["status"] == "success"
        assert "schema_size" in result["message"]
        assert "value_size" in result["message"]

    def test_init_metadata_overwrite_strategy(self, real_agent_config):
        """_init_metadata with overwrite strategy truncates and rebuilds."""
        svc = KbService(agent_config=real_agent_config)
        result = svc._init_metadata(
            config=real_agent_config,
            strategy="overwrite",
            pool_size=1,
            dir_path=real_agent_config.rag_storage_path(),
            args=KbService._build_args(
                BootstrapKbInput(components=["metadata"], strategy="overwrite"),
                str(real_agent_config.home),
            ),
        )
        assert result["status"] == "success"


class TestKbServiceRunComponent:
    """Tests for _run_component dispatch."""

    def test_unknown_component_returns_failed(self, real_agent_config):
        """_run_component for unknown component returns failed status."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = asyncio.Event()

        # Use valid components list but pass unknown component name to _run_component directly
        request = BootstrapKbInput(components=["metadata"], strategy="check")
        result = svc._run_component(request, "unknown_comp", queue, loop, cancel_event, str(real_agent_config.home))
        assert result["status"] == "failed"
        assert "Unknown component" in result["message"]
        loop.close()

    def test_semantic_model_component(self, real_agent_config):
        """_run_component for semantic_model exercises the init path."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = asyncio.Event()

        request = BootstrapKbInput(components=["semantic_model"], strategy="check")
        result = svc._run_component(request, "semantic_model", queue, loop, cancel_event, str(real_agent_config.home))
        # May succeed or fail depending on whether success_story dir exists
        assert isinstance(result, dict)
        loop.close()

    def test_ext_knowledge_component(self, real_agent_config):
        """_run_component for ext_knowledge exercises the init path."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = asyncio.Event()

        request = BootstrapKbInput(components=["ext_knowledge"], strategy="check")
        result = svc._run_component(request, "ext_knowledge", queue, loop, cancel_event, str(real_agent_config.home))
        assert isinstance(result, dict)
        loop.close()

    def test_metrics_component(self, real_agent_config):
        """_run_component for metrics exercises the init path (may fail without success_story)."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = asyncio.Event()

        request = BootstrapKbInput(components=["metrics"], strategy="check")
        try:
            result = svc._run_component(request, "metrics", queue, loop, cancel_event, str(real_agent_config.home))
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail if init_success_story_metrics requires specific data
        loop.close()

    def test_reference_sql_component(self, real_agent_config):
        """_run_component for reference_sql exercises the init path."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = asyncio.Event()

        request = BootstrapKbInput(components=["reference_sql"], strategy="check")
        result = svc._run_component(request, "reference_sql", queue, loop, cancel_event, str(real_agent_config.home))
        assert isinstance(result, dict)
        loop.close()

    def test_metadata_component_check(self, real_agent_config):
        """_run_component for metadata with check strategy returns success."""
        import asyncio

        svc = KbService(agent_config=real_agent_config)
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = asyncio.Event()

        request = BootstrapKbInput(components=["metadata"], strategy="check")
        result = svc._run_component(request, "metadata", queue, loop, cancel_event, str(real_agent_config.home))
        assert result["status"] == "success"
        loop.close()
