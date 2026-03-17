# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/generation_hooks.py — GenerationHooks.

All external dependencies are mocked. Tests cover:
- Initialization
- on_tool_end routing
- _extract_filepaths_from_result
- _extract_metric_generation_result
- _process_single_file (file not found, empty, already processed, happy path)
- _handle_sql_summary_result
- _is_sql_summary_tool_call / _is_ext_knowledge_tool_call
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.cli.generation_hooks import GenerationCancelledException, GenerationHooks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broker():
    b = MagicMock()
    b.request = AsyncMock()
    return b


@pytest.fixture
def agent_config():
    cfg = MagicMock()
    cfg.home = "/tmp/datus_test"
    cfg.current_namespace = "test_ns"
    return cfg


@pytest.fixture
def hooks(broker, agent_config):
    with patch("datus.cli.generation_hooks.get_path_manager"):
        return GenerationHooks(broker=broker, agent_config=agent_config)


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


class TestGenerationHooksInit:
    def test_init_sets_broker(self, broker, agent_config):
        h = GenerationHooks(broker=broker, agent_config=agent_config)
        assert h.broker is broker

    def test_init_sets_agent_config(self, broker, agent_config):
        h = GenerationHooks(broker=broker, agent_config=agent_config)
        assert h.agent_config is agent_config

    def test_init_empty_processed_files(self, broker, agent_config):
        h = GenerationHooks(broker=broker, agent_config=agent_config)
        assert h.processed_files == set()

    def test_init_no_config(self, broker):
        h = GenerationHooks(broker=broker)
        assert h.agent_config is None


# ---------------------------------------------------------------------------
# Tests: on_tool_end routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnToolEnd:
    async def test_routes_end_semantic_model_generation(self, hooks):
        hooks._handle_end_semantic_model_generation = AsyncMock()
        tool = MagicMock()
        tool.name = "end_semantic_model_generation"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_end_semantic_model_generation.assert_awaited_once_with("result")

    async def test_routes_end_metric_generation(self, hooks):
        hooks._handle_end_metric_generation = AsyncMock()
        tool = MagicMock()
        tool.name = "end_metric_generation"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_end_metric_generation.assert_awaited_once_with("result")

    async def test_routes_write_file_sql_summary(self, hooks):
        hooks._handle_sql_summary_result = AsyncMock()
        hooks._is_sql_summary_tool_call = MagicMock(return_value=True)
        tool = MagicMock()
        tool.name = "write_file"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_sql_summary_result.assert_awaited_once()

    async def test_routes_write_file_ext_knowledge(self, hooks):
        hooks._handle_ext_knowledge_result = AsyncMock()
        hooks._is_sql_summary_tool_call = MagicMock(return_value=False)
        hooks._is_ext_knowledge_tool_call = MagicMock(return_value=True)
        tool = MagicMock()
        tool.name = "write_file"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_ext_knowledge_result.assert_awaited_once()

    async def test_unrelated_tool_does_nothing(self, hooks):
        hooks._handle_end_semantic_model_generation = AsyncMock()
        hooks._handle_end_metric_generation = AsyncMock()
        tool = MagicMock()
        tool.name = "some_other_tool"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_end_semantic_model_generation.assert_not_called()
        hooks._handle_end_metric_generation.assert_not_called()

    async def test_tool_name_via_dunder_name(self, hooks):
        """Handles tools that use __name__ instead of .name attribute."""
        hooks._handle_end_semantic_model_generation = AsyncMock()
        tool = MagicMock(spec=[])  # no .name attribute
        tool.__name__ = "end_semantic_model_generation"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_end_semantic_model_generation.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: on_start / on_tool_start / on_handoff / on_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStubHooks:
    async def test_on_start(self, hooks):
        await hooks.on_start(MagicMock(), MagicMock())  # no exception

    async def test_on_tool_start(self, hooks):
        await hooks.on_tool_start(MagicMock(), MagicMock(), MagicMock())

    async def test_on_handoff(self, hooks):
        await hooks.on_handoff(MagicMock(), MagicMock(), MagicMock())

    async def test_on_end(self, hooks):
        await hooks.on_end(MagicMock(), MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# Tests: _extract_filepaths_from_result
# ---------------------------------------------------------------------------


class TestExtractFilepaths:
    def test_from_dict_with_files(self, hooks):
        result = {"result": {"semantic_model_files": ["/a/b.yaml", "/c/d.yaml"]}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == ["/a/b.yaml", "/c/d.yaml"]

    def test_from_dict_no_files(self, hooks):
        result = {"result": {}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == []

    def test_from_object_with_result(self, hooks):
        r = MagicMock()
        r.result = {"semantic_model_files": ["/x/y.yaml"]}
        r.success = True
        paths = hooks._extract_filepaths_from_result(r)
        assert paths == ["/x/y.yaml"]

    def test_from_none_returns_empty(self, hooks):
        paths = hooks._extract_filepaths_from_result(None)
        assert paths == []

    def test_dict_with_empty_list(self, hooks):
        result = {"result": {"semantic_model_files": []}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == []


# ---------------------------------------------------------------------------
# Tests: _extract_metric_generation_result
# ---------------------------------------------------------------------------


class TestExtractMetricGenerationResult:
    def test_from_dict(self, hooks):
        result = {
            "result": {
                "metric_file": "/m/metric.yaml",
                "semantic_model_file": "/s/sem.yaml",
                "metric_sqls": {"revenue": "SELECT SUM(amount) FROM orders"},
            }
        }
        mf, smf, sqls = hooks._extract_metric_generation_result(result)
        assert mf == "/m/metric.yaml"
        assert smf == "/s/sem.yaml"
        assert sqls == {"revenue": "SELECT SUM(amount) FROM orders"}

    def test_from_object(self, hooks):
        r = MagicMock()
        r.result = {"metric_file": "/m.yaml", "semantic_model_file": "", "metric_sqls": {}}
        r.success = True
        mf, smf, sqls = hooks._extract_metric_generation_result(r)
        assert mf == "/m.yaml"

    def test_invalid_result_returns_empty(self, hooks):
        mf, smf, sqls = hooks._extract_metric_generation_result("not a dict or obj")
        assert mf == ""
        assert smf == ""
        assert sqls == {}


# ---------------------------------------------------------------------------
# Tests: _process_single_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessSingleFile:
    async def test_file_not_found(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        await hooks._process_single_file("/nonexistent/file.yaml")
        hooks._get_sync_confirmation.assert_not_called()

    async def test_empty_file_skipped(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # empty
            path = f.name
        try:
            await hooks._process_single_file(path)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_already_processed_skipped(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            path = f.name
        hooks.processed_files.add(path)
        try:
            await hooks._process_single_file(path)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_happy_path_calls_confirmation(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            path = f.name
        try:
            await hooks._process_single_file(path)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()
        assert path in hooks.processed_files


# ---------------------------------------------------------------------------
# Tests: _handle_sql_summary_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleSqlSummaryResult:
    async def test_no_file_path_returns_early(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        result = {"result": "some unrelated message"}
        await hooks._handle_sql_summary_result(result)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_file_not_exists_returns_early(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        result = {"result": "File written successfully: /nonexistent/path.sql"}
        await hooks._handle_sql_summary_result(result)
        hooks._get_sync_confirmation.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _handle_end_semantic_model_generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleEndSemanticModelGeneration:
    async def test_no_file_paths_logs_warning(self, hooks):
        hooks._process_single_file = AsyncMock()
        result = {"result": {}}  # no semantic_model_files
        await hooks._handle_end_semantic_model_generation(result)
        hooks._process_single_file.assert_not_called()

    async def test_with_file_paths_processes_each(self, hooks):
        hooks._process_single_file = AsyncMock()
        result = {"result": {"semantic_model_files": ["/a.yaml", "/b.yaml"]}}
        await hooks._handle_end_semantic_model_generation(result)
        assert hooks._process_single_file.await_count == 2

    async def test_cancelled_exception_absorbed(self, hooks):
        hooks._process_single_file = AsyncMock(side_effect=GenerationCancelledException)
        result = {"result": {"semantic_model_files": ["/a.yaml"]}}
        await hooks._handle_end_semantic_model_generation(result)  # should not raise


# ---------------------------------------------------------------------------
# Tests: _handle_end_metric_generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleEndMetricGeneration:
    async def test_no_metric_file_returns_early(self, hooks):
        hooks._process_single_file = AsyncMock()
        result = {"result": {"metric_file": "", "semantic_model_file": "", "metric_sqls": {}}}
        await hooks._handle_end_metric_generation(result)
        hooks._process_single_file.assert_not_called()

    async def test_with_metric_no_semantic(self, hooks):
        hooks._process_single_file = AsyncMock()
        result = {
            "result": {
                "metric_file": "/m.yaml",
                "semantic_model_file": "",
                "metric_sqls": {},
            }
        }
        await hooks._handle_end_metric_generation(result)
        hooks._process_single_file.assert_awaited_once()

    async def test_with_metric_and_semantic(self, hooks):
        hooks._process_metric_with_semantic_model = AsyncMock()
        result = {
            "result": {
                "metric_file": "/m.yaml",
                "semantic_model_file": "/s.yaml",
                "metric_sqls": {},
            }
        }
        with patch("datus.cli.generation_hooks.get_path_manager"):
            await hooks._handle_end_metric_generation(result)
        hooks._process_metric_with_semantic_model.assert_awaited_once()

    async def test_cancelled_exception_absorbed(self, hooks):
        hooks._process_single_file = AsyncMock(side_effect=GenerationCancelledException)
        result = {"result": {"metric_file": "/m.yaml", "semantic_model_file": "", "metric_sqls": {}}}
        await hooks._handle_end_metric_generation(result)  # should not raise
