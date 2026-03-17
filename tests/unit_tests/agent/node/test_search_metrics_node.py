# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SearchMetricsNode.

CI-level: zero external deps, zero network, zero API keys.
MetricRAG is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.search_metrics_node import SearchMetricsNode
from datus.schemas.action_history import ActionStatus
from datus.schemas.node_models import Metric, SQLContext, SqlTask
from datus.schemas.search_metrics_node_models import SearchMetricsInput, SearchMetricsResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(rag_path="/tmp/nonexistent_rag"):
    cfg = MagicMock()
    cfg.search_metrics_rate = "fast"
    cfg.rag_storage_path.return_value = rag_path
    cfg.agentic_nodes = {}
    cfg.permissions_config = None
    cfg.skills_config = None
    return cfg


def _make_sql_task():
    return SqlTask(
        database_type="sqlite",
        database_name="test_db",
        task="Find total revenue",
        subject_path=["Finance"],
    )


def _make_input():
    return SearchMetricsInput(
        input_text="Find total revenue",
        sql_task=_make_sql_task(),
        database_type="sqlite",
        matching_rate="fast",
    )


def _make_node(agent_config=None):
    cfg = agent_config or _make_agent_config()
    node = SearchMetricsNode(
        node_id="search_metrics_1",
        description="Search metrics",
        node_type="search_metrics",
        agent_config=cfg,
    )
    node.input = _make_input()
    return node


def _make_workflow(metrics=None, reflection_round=0):
    wf = MagicMock()
    wf.task = _make_sql_task()
    wf.context.metrics = metrics or []
    wf.context.sql_contexts = []
    wf.reflection_round = reflection_round
    return wf


def _make_metric(name="revenue_total"):
    m = MagicMock(spec=Metric)
    m.name = name
    return m


def _make_metrics_result(success=True, metrics=None, count=0, error=None):
    result = MagicMock(spec=SearchMetricsResult)
    result.success = success
    result.metrics = metrics or []
    result.metrics_count = count
    result.error = error
    result.sql_task = _make_sql_task()
    return result


# ---------------------------------------------------------------------------
# TestSearchMetricsNodeInit
# ---------------------------------------------------------------------------


class TestSearchMetricsNodeInit:
    def test_node_creates(self):
        node = _make_node()
        assert node.id == "search_metrics_1"
        assert node.description == "Search metrics"
        assert node._store is None

    def test_store_property_creates_metric_rag(self):
        """Accessing .store creates MetricRAG lazily."""
        node = _make_node()
        with patch("datus.agent.node.search_metrics_node.MetricRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            store = node.store

        mock_rag.assert_called_once_with(node.agent_config)
        assert store is mock_rag.return_value

    def test_store_property_caches(self):
        """Accessing .store twice returns same instance."""
        node = _make_node()
        mock_rag_instance = MagicMock()
        node._store = mock_rag_instance
        # Should not create a new one
        assert node.store is mock_rag_instance


# ---------------------------------------------------------------------------
# TestSetupInput
# ---------------------------------------------------------------------------


class TestSetupInputSearchMetrics:
    def test_setup_input_builds_metrics_input(self):
        node = _make_node()
        wf = _make_workflow()
        result = node.setup_input(wf)

        assert result["success"] is True
        assert isinstance(node.input, SearchMetricsInput)
        assert node.input.input_text == "Find total revenue"

    def test_setup_input_escalates_rate_with_reflection(self):
        """reflection_round escalates matching_rate."""
        cfg = _make_agent_config()
        cfg.search_metrics_rate = "fast"
        node = _make_node(agent_config=cfg)
        wf = _make_workflow(reflection_round=1)  # fast -> medium
        node.setup_input(wf)

        assert node.input.matching_rate == "medium"

    def test_setup_input_reflection_caps_at_slow(self):
        """reflection_round beyond bounds caps at 'slow'."""
        cfg = _make_agent_config()
        cfg.search_metrics_rate = "slow"
        node = _make_node(agent_config=cfg)
        wf = _make_workflow(reflection_round=5)
        node.setup_input(wf)

        assert node.input.matching_rate == "slow"

    def test_setup_input_passes_sql_contexts(self):
        node = _make_node()
        sql_ctx = SQLContext(sql_query="SELECT 1")
        wf = _make_workflow()
        wf.context.sql_contexts = [sql_ctx]
        node.setup_input(wf)

        assert node.input.sql_contexts == [sql_ctx]


# ---------------------------------------------------------------------------
# TestUpdateContext
# ---------------------------------------------------------------------------


class TestUpdateContextSearchMetrics:
    def test_update_context_sets_metrics_when_empty(self):
        node = _make_node()
        metrics = [_make_metric("revenue")]
        result = _make_metrics_result(success=True, metrics=metrics, count=1)
        node.result = result

        wf = _make_workflow()
        update_result = node.update_context(wf)

        assert update_result["success"] is True
        assert wf.context.metrics == metrics

    def test_update_context_skips_when_metrics_already_set(self):
        node = _make_node()
        existing_metrics = [_make_metric("existing")]
        new_metrics = [_make_metric("revenue")]
        result = _make_metrics_result(success=True, metrics=new_metrics, count=1)
        node.result = result

        wf = _make_workflow(metrics=existing_metrics)
        node.update_context(wf)

        # Should not overwrite
        assert wf.context.metrics == existing_metrics

    def test_update_context_handles_exception(self):
        node = _make_node()
        node.result = _make_metrics_result()

        wf = _make_workflow()
        # Make the metrics assignment raise
        wf.context.metrics = MagicMock()
        wf.context.metrics.__len__ = MagicMock(side_effect=RuntimeError("context error"))

        result = node.update_context(wf)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# TestGetBadResult
# ---------------------------------------------------------------------------


class TestGetBadResult:
    def test_get_bad_result_structure(self):
        node = _make_node()
        result = node.get_bad_result("some error")

        assert result.success is False
        assert result.error == "some error"
        assert result.metrics == []
        assert result.metrics_count == 0


# ---------------------------------------------------------------------------
# TestExecuteSearchMetrics
# ---------------------------------------------------------------------------


class TestExecuteSearchMetrics:
    def test_execute_fallback_when_no_rag_path(self):
        """When RAG path doesn't exist, return bad result."""
        cfg = _make_agent_config(rag_path="/nonexistent/path/xxx")
        node = _make_node(agent_config=cfg)

        result = node._execute_search_metrics()

        assert result.success is False

    def test_execute_returns_result_when_rag_exists(self, tmp_path):
        """When RAG path exists and search succeeds, return result."""
        cfg = _make_agent_config(rag_path=str(tmp_path))
        node = _make_node(agent_config=cfg)

        good_result = _make_metrics_result(success=True, metrics=[_make_metric()], count=1)
        with patch.object(node, "_search_metrics", return_value=good_result):
            result = node._execute_search_metrics()

        assert result.success is True
        assert result.metrics_count == 1

    def test_execute_returns_bad_when_search_fails(self, tmp_path):
        """When search result is not successful, return bad result."""
        cfg = _make_agent_config(rag_path=str(tmp_path))
        node = _make_node(agent_config=cfg)

        bad_result = _make_metrics_result(success=False, error="no data")
        with patch.object(node, "_search_metrics", return_value=bad_result):
            result = node._execute_search_metrics()

        assert result.success is False

    def test_execute_handles_exception(self, tmp_path):
        """When search raises, return bad result."""
        cfg = _make_agent_config(rag_path=str(tmp_path))
        node = _make_node(agent_config=cfg)

        with patch.object(node, "_search_metrics", side_effect=RuntimeError("search error")):
            result = node._execute_search_metrics()

        assert result.success is False

    def test_execute_sets_result(self, tmp_path):
        cfg = _make_agent_config(rag_path=str(tmp_path))
        node = _make_node(agent_config=cfg)

        good_result = _make_metrics_result(success=True, metrics=[_make_metric()], count=1)
        with patch.object(node, "_search_metrics", return_value=good_result):
            node.execute()

        assert node.result is not None


# ---------------------------------------------------------------------------
# TestSearchMetrics (private method)
# ---------------------------------------------------------------------------


class TestSearchMetricsInternal:
    def test_search_metrics_calls_store(self):
        node = _make_node()
        raw_metrics = [
            {
                "name": "revenue",
                "description": "Total revenue",
            }
        ]
        mock_store = MagicMock()
        mock_store.search_metrics.return_value = raw_metrics
        node._store = mock_store

        with patch.object(Metric, "from_dict", side_effect=lambda d: MagicMock(spec=Metric)):
            result = node._search_metrics()

        mock_store.search_metrics.assert_called_once()
        assert result.success is True
        assert result.metrics_count == 1


# ---------------------------------------------------------------------------
# TestExecuteStream
# ---------------------------------------------------------------------------


class TestExecuteStreamSearchMetrics:
    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        """execute_stream yields at least one action."""
        node = _make_node()
        good_result = _make_metrics_result(success=True, metrics=[], count=0)

        with patch.object(node, "_execute_search_metrics", return_value=good_result):
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert len(actions) >= 1

    @pytest.mark.asyncio
    async def test_execute_stream_final_action_success(self):
        """Final action is SUCCESS when metrics search succeeds."""
        node = _make_node()
        good_result = _make_metrics_result(success=True, count=2)

        with patch.object(node, "_execute_search_metrics", return_value=good_result):
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_final_action_failed_on_error(self):
        """Final action is FAILED when metrics search returns failure."""
        node = _make_node()
        bad_result = _make_metrics_result(success=False, error="no rag")

        with patch.object(node, "_execute_search_metrics", return_value=bad_result):
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_stream_stores_result(self):
        """execute_stream stores result on self.result."""
        node = _make_node()
        good_result = _make_metrics_result(success=True)

        with patch.object(node, "_execute_search_metrics", return_value=good_result):
            async for _ in node.execute_stream():
                pass

        assert node.result is good_result

    @pytest.mark.asyncio
    async def test_execute_stream_action_type_metrics_search(self):
        """Yielded action has action_type='metrics_search'."""
        node = _make_node()
        good_result = _make_metrics_result(success=True)

        with patch.object(node, "_execute_search_metrics", return_value=good_result):
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        action_types = [a.action_type for a in actions]
        assert "metrics_search" in action_types

    @pytest.mark.asyncio
    async def test_execute_stream_propagates_exception(self):
        """execute_stream re-raises exceptions."""
        node = _make_node()

        with patch.object(node, "_execute_search_metrics", side_effect=RuntimeError("fatal error")):
            with pytest.raises(RuntimeError, match="fatal error"):
                async for _ in node.execute_stream():
                    pass
