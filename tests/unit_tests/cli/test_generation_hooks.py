# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/generation_hooks.py — GenerationHooks.

All external dependencies are mocked. Tests cover:
- Initialization
- on_tool_end routing
- _extract_filepaths_from_result
- _process_single_file (file not found, empty, already processed, happy path)
- _handle_sql_summary_result
- _is_sql_summary_tool_call / _is_ext_knowledge_tool_call
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.cli.execution_state import InteractionCancelled
from datus.cli.generation_hooks import (
    GenerationCancelledException,
    GenerationHooks,
    make_kb_path_normalizer,
    normalize_kb_relative_path,
    resolve_kb_sandbox_path,
)
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broker():
    b = MagicMock()
    b.request = AsyncMock()
    return b


@pytest.fixture
def agent_config(tmp_path):
    kb_home = tmp_path / "kb"
    (kb_home / "semantic_models" / "test_ns").mkdir(parents=True, exist_ok=True)
    (kb_home / "sql_summaries" / "test_ns").mkdir(parents=True, exist_ok=True)
    (kb_home / "ext_knowledge" / "test_ns").mkdir(parents=True, exist_ok=True)
    cfg = MagicMock()
    cfg.home = str(tmp_path)
    cfg.current_database = "test_ns"
    cfg.current_namespace = "test_ns"
    cfg.db_type = "sqlite"
    cfg.path_manager = MagicMock()
    cfg.path_manager.semantic_model_path.return_value = kb_home / "semantic_models" / "test_ns"
    # Real value so _resolve_path's realpath/commonpath containment check works.
    cfg.path_manager.knowledge_base_home = kb_home
    return cfg


@pytest.fixture
def hooks(broker, agent_config):
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
        tool = MagicMock()
        tool.name = "some_other_tool"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_end_semantic_model_generation.assert_not_called()

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
    def test_from_dict_with_files(self, hooks, agent_config):
        # Absolute paths inside the configured knowledge_base_home pass the
        # containment check and are returned normpath'd.
        sem_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "semantic_models" / "test_ns"
        paths_in = [str(sem_dir / "a.yaml"), str(sem_dir / "b.yaml")]
        result = {"result": {"semantic_model_files": paths_in}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == paths_in

    def test_from_dict_drops_paths_outside_kb_home(self, hooks):
        """Absolute paths outside knowledge_base_home must be filtered out."""
        result = {"result": {"semantic_model_files": ["/etc/passwd", "/a/b.yaml"]}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == []

    def test_from_dict_no_files(self, hooks):
        result = {"result": {}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == []

    def test_from_object_with_result(self, hooks, agent_config):
        sem_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "semantic_models" / "test_ns"
        inside = str(sem_dir / "x.yaml")
        r = MagicMock()
        r.result = {"semantic_model_files": [inside]}
        r.success = True
        paths = hooks._extract_filepaths_from_result(r)
        assert paths == [inside]

    def test_from_none_returns_empty(self, hooks):
        paths = hooks._extract_filepaths_from_result(None)
        assert paths == []

    def test_dict_with_empty_list(self, hooks):
        result = {"result": {"semantic_model_files": []}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == []


class TestResolvePath:
    """
    Tests for ``GenerationHooks._resolve_path``.

    The resolver joins relative paths against ``knowledge_base_home`` after
    routing them through ``normalize_kb_relative_path`` — so a naked filename
    written by the LLM (e.g. ``orders.yml``) lands at
    ``{kb_home}/{type_subdir}/{namespace}/orders.yml``, matching where the
    FilesystemFuncTool actually wrote the file.
    """

    def _make_hooks(self, broker, kb="/ws", namespace="ns_a"):
        cfg = MagicMock()
        cfg.current_namespace = namespace
        cfg.path_manager = MagicMock()
        cfg.path_manager.knowledge_base_home = Path(kb)
        return GenerationHooks(broker=broker, agent_config=cfg), cfg

    def test_absolute_path_outside_kb_rejected(self, broker):
        """Absolute paths that escape knowledge_base_home must be returned as an
        empty string so downstream ``os.path.exists`` / ``open`` never sees
        them — fail closed, no arbitrary file disclosure."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("/etc/passwd", "semantic") == ""
        assert h._resolve_path("/abs/path/to/file.yml", "semantic") == ""

    def test_absolute_path_inside_kb_home_is_normpathed(self, broker, tmp_path):
        """Absolute paths that resolve inside knowledge_base_home are accepted."""
        kb = tmp_path / "kb"
        (kb / "semantic_models" / "ns_a").mkdir(parents=True)
        inside = kb / "semantic_models" / "ns_a" / "orders.yml"
        inside.write_text("x")
        h, _ = self._make_hooks(broker, kb=str(kb))
        resolved = h._resolve_path(str(inside), "semantic")
        assert os.path.realpath(resolved) == os.path.realpath(str(inside))

    def test_relative_joined_for_semantic(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("orders.yml", "semantic") == "/ws/semantic_models/ns_a/orders.yml"

    def test_relative_joined_for_sql_summary(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("q_001.yaml", "sql_summary") == "/ws/sql_summaries/ns_a/q_001.yaml"

    def test_relative_joined_for_ext_knowledge(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("gmv.yaml", "ext_knowledge") == "/ws/ext_knowledge/ns_a/gmv.yaml"

    def test_nested_relative_joined(self, broker):
        h, _ = self._make_hooks(broker)
        assert (
            h._resolve_path("metrics/orders_metrics.yml", "semantic")
            == "/ws/semantic_models/ns_a/metrics/orders_metrics.yml"
        )

    def test_already_prefixed_path_passes_through(self, broker):
        """LLM that includes the {subdir}/{namespace}/ prefix must not be double-prefixed."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("semantic_models/ns_a/orders.yml", "semantic") == "/ws/semantic_models/ns_a/orders.yml"

    def test_other_namespace_subdir_passes_through(self, broker):
        """Explicit cross-namespace authoring is preserved."""
        h, _ = self._make_hooks(broker)
        assert (
            h._resolve_path("semantic_models/other_db/orders.yml", "semantic")
            == "/ws/semantic_models/other_db/orders.yml"
        )

    def test_empty_path_returns_unchanged(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("", "semantic") == ""

    def test_unknown_kind_resolves_against_kb_home_root(self, broker):
        """Unknown kind: normalizer adds no prefix, but path still rooted at kb_home."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("orders.yml", "unknown") == "/ws/orders.yml"

    def test_no_agent_config_leaves_relative_unchanged(self, broker):
        h = GenerationHooks(broker=broker, agent_config=None)
        assert h._resolve_path("orders.yml", "semantic") == "orders.yml"

    def test_rejects_traversal_escape(self, broker):
        """`../../etc/passwd` resolves outside knowledge_base_home and must be rejected."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("../../etc/passwd", "semantic") == ""

    def test_allows_traversal_that_stays_inside_kb_home(self, broker):
        """A path whose normpath stays under knowledge_base_home is allowed."""
        h, _ = self._make_hooks(broker)
        # `metrics/../orders.yml` → prepend → `semantic_models/ns_a/metrics/../orders.yml`
        # → normpath under /ws → `/ws/semantic_models/ns_a/orders.yml`
        assert h._resolve_path("metrics/../orders.yml", "semantic") == "/ws/semantic_models/ns_a/orders.yml"

    def test_rejects_symlink_that_escapes_kb_home(self, broker, tmp_path):
        """A symlink inside the KB whose target is outside must be rejected."""
        kb_home = tmp_path / "kb"
        sub = kb_home / "semantic_models" / "ns_a"
        sub.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.yml").write_text("x")
        (sub / "leak.yml").symlink_to(outside / "secret.yml")

        h, _ = self._make_hooks(broker, kb=str(kb_home))
        # Textually the path looks inside kb_home, but realpath dereferences the
        # symlink to /…/outside/secret.yml which escapes the workspace root.
        assert h._resolve_path("leak.yml", "semantic") == ""

    def test_uses_current_namespace_at_call_time(self, broker):
        """Sub-agent switches change current_namespace; resolution must follow."""
        h, cfg = self._make_hooks(broker, namespace="ns_a")
        assert h._resolve_path("orders.yml", "semantic") == "/ws/semantic_models/ns_a/orders.yml"

        cfg.current_namespace = "ns_b"
        assert h._resolve_path("orders.yml", "semantic") == "/ws/semantic_models/ns_b/orders.yml"

    def test_extract_filepaths_resolves_relative_entries(self, broker):
        h, _ = self._make_hooks(broker)
        # The relative entry resolves inside kb_home; the escaping absolute entry
        # is dropped so downstream processing never sees it.
        result = {"result": {"semantic_model_files": ["orders.yml", "/abs/customers.yml"]}}
        paths = h._extract_filepaths_from_result(result)
        assert paths == ["/ws/semantic_models/ns_a/orders.yml"]


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

    async def test_with_file_paths_processes_each(self, hooks, agent_config):
        hooks._process_single_file = AsyncMock()
        sem_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "semantic_models" / "test_ns"
        result = {"result": {"semantic_model_files": [str(sem_dir / "a.yaml"), str(sem_dir / "b.yaml")]}}
        await hooks._handle_end_semantic_model_generation(result)
        assert hooks._process_single_file.await_count == 2

    async def test_cancelled_exception_absorbed(self, hooks):
        hooks._process_single_file = AsyncMock(side_effect=GenerationCancelledException)
        result = {"result": {"semantic_model_files": ["/a.yaml"]}}
        await hooks._handle_end_semantic_model_generation(result)  # should not raise


@pytest.fixture
def hooks_no_config(broker):
    return GenerationHooks(broker=broker, agent_config=None)


# ---------------------------------------------------------------------------
# Tests: GenerationCancelledException
# ---------------------------------------------------------------------------


class TestGenerationCancelledException:
    def test_is_exception(self):
        exc = GenerationCancelledException("cancelled")
        assert isinstance(exc, Exception)
        assert str(exc) == "cancelled"


# ---------------------------------------------------------------------------
# Tests: _is_sql_summary_tool_call
# ---------------------------------------------------------------------------


class TestIsSqlSummaryToolCall:
    def test_returns_true_for_sql_summary(self, hooks):
        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"file_type": "sql_summary"})
        assert hooks._is_sql_summary_tool_call(ctx) is True

    def test_returns_false_for_other_type(self, hooks):
        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"file_type": "semantic"})
        assert hooks._is_sql_summary_tool_call(ctx) is False

    def test_returns_false_for_no_tool_arguments(self, hooks):
        ctx = MagicMock(spec=[])  # no tool_arguments attribute
        assert hooks._is_sql_summary_tool_call(ctx) is False

    def test_returns_false_for_empty_tool_arguments(self, hooks):
        ctx = MagicMock()
        ctx.tool_arguments = ""
        assert hooks._is_sql_summary_tool_call(ctx) is False

    def test_returns_false_for_invalid_json(self, hooks):
        ctx = MagicMock()
        ctx.tool_arguments = "not-json"
        assert hooks._is_sql_summary_tool_call(ctx) is False


# ---------------------------------------------------------------------------
# Tests: _is_ext_knowledge_tool_call
# ---------------------------------------------------------------------------


class TestIsExtKnowledgeToolCall:
    def test_returns_true_for_ext_knowledge(self, hooks):
        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"file_type": "ext_knowledge"})
        assert hooks._is_ext_knowledge_tool_call(ctx) is True

    def test_returns_false_for_sql_summary(self, hooks):
        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"file_type": "sql_summary"})
        assert hooks._is_ext_knowledge_tool_call(ctx) is False

    def test_returns_false_for_no_attribute(self, hooks):
        ctx = MagicMock(spec=[])
        assert hooks._is_ext_knowledge_tool_call(ctx) is False

    def test_returns_false_for_invalid_json(self, hooks):
        ctx = MagicMock()
        ctx.tool_arguments = "{"
        assert hooks._is_ext_knowledge_tool_call(ctx) is False


# ---------------------------------------------------------------------------
# Tests: _handle_sql_summary_result - additional branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleSqlSummaryResultExtended:
    async def test_result_object_with_no_match(self, hooks):
        """result.result doesn't match expected pattern -> early return."""
        hooks._get_sync_confirmation = AsyncMock()
        result = MagicMock()
        result.result = "Some unrelated message"
        await hooks._handle_sql_summary_result(result)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_result_object_file_written_but_not_exists(self, hooks):
        """result.result matches pattern but file doesn't exist -> early return."""
        hooks._get_sync_confirmation = AsyncMock()
        result = MagicMock()
        result.result = "File written successfully: /nonexistent/path.yaml"
        await hooks._handle_sql_summary_result(result)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_already_processed_skipped(self, hooks):
        """File already in processed_files -> skipped."""
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_sql\nsql: SELECT 1\n")
            path = f.name
        hooks.processed_files.add(path)
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_sql_summary_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_happy_path_calls_confirmation(self, hooks, agent_config):
        """File exists with content -> confirmation called."""
        hooks._get_sync_confirmation = AsyncMock()
        sql_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "sql_summaries" / "test_ns"
        path_obj = sql_dir / "q_happy.yaml"
        path_obj.write_text("name: test_sql\nsql: SELECT 1\n")
        path = str(path_obj)
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_sql_summary_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()
        assert path in hooks.processed_files

    async def test_reference_sql_file_written_pattern(self, hooks, agent_config):
        """'Reference SQL file written successfully:' pattern is also matched."""
        hooks._get_sync_confirmation = AsyncMock()
        sql_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "sql_summaries" / "test_ns"
        path_obj = sql_dir / "q_ref.yaml"
        path_obj.write_text("name: test_sql\nsql: SELECT 1\n")
        path = str(path_obj)
        try:
            result = {"result": f"Reference SQL file written successfully: {path}"}
            await hooks._handle_sql_summary_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: _handle_ext_knowledge_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleExtKnowledgeResult:
    async def test_no_match_returns_early(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        result = {"result": "unrelated message"}
        await hooks._handle_ext_knowledge_result(result)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_file_not_exists_returns_early(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        result = {"result": "File written successfully: /nonexistent/ext.yaml"}
        await hooks._handle_ext_knowledge_result(result)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_happy_path_calls_confirmation(self, hooks, agent_config):
        hooks._get_sync_confirmation = AsyncMock()
        ext_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "ext_knowledge" / "test_ns"
        path_obj = ext_dir / "ext_happy.yaml"
        path_obj.write_text("key: value\n")
        path = str(path_obj)
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_ext_knowledge_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()
        assert path in hooks.processed_files

    async def test_ext_knowledge_file_written_pattern(self, hooks, agent_config):
        hooks._get_sync_confirmation = AsyncMock()
        ext_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "ext_knowledge" / "test_ns"
        path_obj = ext_dir / "ext_pattern.yaml"
        path_obj.write_text("key: value\n")
        path = str(path_obj)
        try:
            result = {"result": f"External knowledge file written successfully: {path}"}
            await hooks._handle_ext_knowledge_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()

    async def test_already_processed_skipped(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            path = f.name
        hooks.processed_files.add(path)
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_ext_knowledge_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_empty_file_returns_early(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_ext_knowledge_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_not_called()

    async def test_result_object_with_match(self, hooks, agent_config):
        hooks._get_sync_confirmation = AsyncMock()
        ext_dir = Path(str(agent_config.path_manager.knowledge_base_home)) / "ext_knowledge" / "test_ns"
        path_obj = ext_dir / "ext_match.yaml"
        path_obj.write_text("key: value\n")
        path = str(path_obj)
        try:
            result = MagicMock()
            result.result = f"File written successfully: {path}"
            await hooks._handle_ext_knowledge_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: _get_sync_confirmation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetSyncConfirmation:
    async def test_choice_yes_calls_sync_and_callback(self, hooks):
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("y", callback))
        hooks._sync_to_storage = AsyncMock(return_value="Synced!")

        await hooks._get_sync_confirmation(
            yaml_content="key: val",
            file_path="/tmp/test.yaml",
            yaml_type="semantic",
        )

        hooks._sync_to_storage.assert_awaited_once()
        callback.assert_awaited_once()
        args = callback.call_args[0][0]
        assert "Synced!" in args

    async def test_choice_no_calls_callback_with_file_only_message(self, hooks):
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("n", callback))

        await hooks._get_sync_confirmation(
            yaml_content="key: val",
            file_path="/tmp/test.yaml",
            yaml_type="semantic",
        )

        callback.assert_awaited_once()
        args = callback.call_args[0][0]
        assert "/tmp/test.yaml" in args

    async def test_interaction_cancelled_raises_generation_cancelled(self, hooks):
        hooks.broker.request = AsyncMock(side_effect=InteractionCancelled())

        with pytest.raises(GenerationCancelledException):
            await hooks._get_sync_confirmation(
                yaml_content="key: val",
                file_path="/tmp/test.yaml",
                yaml_type="semantic",
            )

    async def test_with_prebuilt_display_content(self, hooks):
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("n", callback))

        await hooks._get_sync_confirmation(
            yaml_content="key: val",
            file_path="/tmp/test.yaml",
            yaml_type="sql_summary",
            display_content="## Pre-built header\n```yaml\nkey: val\n```\n",
        )
        # Should not raise
        callback.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: _sync_to_storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSyncToStorage:
    async def test_no_agent_config_returns_error_string(self, hooks_no_config):
        result = await hooks_no_config._sync_to_storage("/tmp/file.yaml", "semantic")
        assert "Error" in result
        assert "configuration not available" in result

    async def test_invalid_yaml_type_returns_error(self, hooks):
        result = await hooks._sync_to_storage("/tmp/file.yaml", "unknown_type")
        assert "Error" in result
        assert "Invalid yaml_type" in result

    async def test_semantic_type_calls_sync_semantic(self, hooks):
        mock_result = {"success": True, "message": "3 objects synced"}
        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db", return_value=mock_result):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "semantic")
        assert "Successfully synced" in result

    async def test_semantic_type_sync_failure(self, hooks):
        mock_result = {"success": False, "error": "YAML parse error"}
        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db", return_value=mock_result):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "semantic")
        assert "Sync failed" in result
        assert "YAML parse error" in result

    async def test_sql_summary_type_calls_sync_reference(self, hooks):
        mock_result = {"success": True, "message": "SQL synced"}
        with patch("datus.cli.generation_hooks.GenerationHooks._sync_reference_sql_to_db", return_value=mock_result):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "sql_summary")
        assert "Successfully synced" in result

    async def test_ext_knowledge_type_calls_sync(self, hooks):
        mock_result = {"success": True, "message": "Ext knowledge synced"}
        with patch("datus.cli.generation_hooks.GenerationHooks._sync_ext_knowledge_to_db", return_value=mock_result):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "ext_knowledge")
        assert "Successfully synced" in result

    async def test_sql_summary_type_calls_sync_reference_sql(self, hooks):
        """sql_summary type delegates to _sync_reference_sql_to_db."""
        mock_result = {"success": True, "message": "SQL synced via reference_sql"}
        with patch("datus.cli.generation_hooks.GenerationHooks._sync_reference_sql_to_db", return_value=mock_result):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "sql_summary")
        assert "Successfully synced" in result

    async def test_exception_returns_error_string(self, hooks):
        with patch(
            "datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db",
            side_effect=RuntimeError("disk full"),
        ):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "semantic")
        assert "error" in result.lower() or "Sync error" in result


# ---------------------------------------------------------------------------
# Tests: _sync_reference_sql_to_db
# ---------------------------------------------------------------------------


class TestSyncReferenceTemplateToDb:
    def test_valid_template_yaml(self, tmp_path):
        import yaml

        from datus.cli.generation_hooks import GenerationHooks

        yaml_file = tmp_path / "tpl.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "sql": "SELECT * FROM t WHERE x = '{{val}}'",
                    "name": "test_reference_sql",
                    "summary": "Test reference sql",
                    "search_text": "test reference sql val",
                    "subject_tree": "Sales/Revenue",
                    "tags": "test",
                }
            )
        )

        mock_config = MagicMock()

        with (
            patch("datus.cli.generation_hooks.ReferenceSqlRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_sql.init_utils.exists_reference_sql",
                return_value=set(),
            ),
            patch(
                "datus.storage.reference_sql.init_utils.gen_reference_sql_id",
                return_value="new_id",
            ),
        ):
            mock_rag = mock_rag_cls.return_value
            mock_rag.upsert_batch = MagicMock()

            result = GenerationHooks._sync_reference_sql_to_db(str(yaml_file), mock_config)

        assert result["success"] is True
        assert "Synced" in result["message"]

    def test_missing_sql_field(self, tmp_path):
        import yaml

        from datus.cli.generation_hooks import GenerationHooks

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml.dump({"name": "no_sql"}))

        result = GenerationHooks._sync_reference_sql_to_db(str(yaml_file), MagicMock())
        assert result["success"] is False
        assert "No reference_sql data" in result["error"]

    def test_duplicate_skipped(self, tmp_path):
        import yaml

        from datus.cli.generation_hooks import GenerationHooks

        yaml_file = tmp_path / "dup.yaml"
        yaml_file.write_text(yaml.dump({"sql": "SELECT 1", "name": "dup", "summary": "x", "search_text": "x"}))

        mock_config = MagicMock()
        with (
            patch("datus.cli.generation_hooks.ReferenceSqlRAG"),
            patch(
                "datus.storage.reference_sql.init_utils.exists_reference_sql",
                return_value={"existing_id"},
            ),
            patch(
                "datus.storage.reference_sql.init_utils.gen_reference_sql_id",
                return_value="existing_id",
            ),
        ):
            result = GenerationHooks._sync_reference_sql_to_db(str(yaml_file), mock_config)

        assert result["success"] is True
        assert "already exists" in result["message"]


# ---------------------------------------------------------------------------
# Tests: _process_metric_with_semantic_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessMetricWithSemanticModel:
    async def test_semantic_missing_tries_metric_alone(self, hooks):
        hooks._process_single_file = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as mf:
            mf.write("metric: revenue\n")
            metric_path = mf.name
        try:
            await hooks._process_metric_with_semantic_model(
                semantic_model_file="/nonexistent/sem.yaml",
                metric_file=metric_path,
            )
        finally:
            os.unlink(metric_path)
        hooks._process_single_file.assert_awaited_once_with(metric_path, metric_sqls=None)

    async def test_metric_missing_tries_semantic_alone(self, hooks):
        hooks._process_single_file = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("data_source:\n  name: orders\n")
            sem_path = sf.name
        try:
            await hooks._process_metric_with_semantic_model(
                semantic_model_file=sem_path,
                metric_file="/nonexistent/metric.yaml",
            )
        finally:
            os.unlink(sem_path)
        hooks._process_single_file.assert_awaited_once_with(sem_path)

    async def test_both_missing_does_nothing(self, hooks):
        hooks._process_single_file = AsyncMock()
        await hooks._process_metric_with_semantic_model(
            semantic_model_file="/nonexistent/sem.yaml",
            metric_file="/nonexistent/metric.yaml",
        )
        hooks._process_single_file.assert_not_called()

    async def test_both_already_processed_skipped(self, hooks):
        hooks._get_sync_confirmation_for_pair = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("data_source:\n  name: orders\n")
            sem_path = sf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as mf:
            mf.write("metric: revenue\n")
            metric_path = mf.name
        hooks.processed_files.add(sem_path)
        hooks.processed_files.add(metric_path)
        try:
            await hooks._process_metric_with_semantic_model(sem_path, metric_path)
        finally:
            os.unlink(sem_path)
            os.unlink(metric_path)
        hooks._get_sync_confirmation_for_pair.assert_not_called()

    async def test_happy_path_calls_confirmation_for_pair(self, hooks):
        hooks._get_sync_confirmation_for_pair = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("data_source:\n  name: orders\n")
            sem_path = sf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as mf:
            mf.write("metric: revenue\n")
            metric_path = mf.name
        try:
            await hooks._process_metric_with_semantic_model(sem_path, metric_path)
        finally:
            os.unlink(sem_path)
            os.unlink(metric_path)
        hooks._get_sync_confirmation_for_pair.assert_awaited_once()
        assert sem_path in hooks.processed_files
        assert metric_path in hooks.processed_files

    async def test_empty_content_returns_early(self, hooks):
        hooks._get_sync_confirmation_for_pair = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("")
            sem_path = sf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as mf:
            mf.write("metric: revenue\n")
            metric_path = mf.name
        try:
            await hooks._process_metric_with_semantic_model(sem_path, metric_path)
        finally:
            os.unlink(sem_path)
            os.unlink(metric_path)
        hooks._get_sync_confirmation_for_pair.assert_not_called()

    async def test_confirmation_error_propagates(self, hooks):
        """Exception in _get_sync_confirmation_for_pair propagates to caller."""
        hooks._get_sync_confirmation_for_pair = AsyncMock(side_effect=RuntimeError("broker down"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("data_source:\n  name: orders\n")
            sem_path = sf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as mf:
            mf.write("metric: revenue\n")
            metric_path = mf.name
        try:
            with pytest.raises(RuntimeError, match="broker down"):
                await hooks._process_metric_with_semantic_model(sem_path, metric_path)
        finally:
            os.unlink(sem_path)
            os.unlink(metric_path)

    async def test_read_error_propagates(self, hooks, tmp_path):
        """Unreadable file raises OSError."""
        sem_dir = tmp_path / "not_a_file"
        sem_dir.mkdir()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as mf:
            mf.write("metric: revenue\n")
            metric_path = mf.name
        try:
            # sem_dir is a directory, open() will raise
            with pytest.raises(IsADirectoryError):
                await hooks._process_metric_with_semantic_model(str(sem_dir), metric_path)
        finally:
            os.unlink(metric_path)


# ---------------------------------------------------------------------------
# Tests: _get_sync_confirmation_for_pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetSyncConfirmationForPair:
    async def test_accept_syncs_both_files(self, hooks):
        """Choosing 'y' calls _sync_semantic_and_metric once."""
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("y", callback))
        hooks._sync_semantic_and_metric = AsyncMock(return_value="Synced OK")

        await hooks._get_sync_confirmation_for_pair(
            semantic_model_file="/tmp/sem.yaml",
            metric_file="/tmp/met.yaml",
        )

        hooks._sync_semantic_and_metric.assert_awaited_once_with("/tmp/sem.yaml", "/tmp/met.yaml", None)
        callback.assert_awaited_once()

    async def test_reject_skips_sync(self, hooks):
        """Choosing 'n' does not call _sync_semantic_and_metric."""
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("n", callback))
        hooks._sync_semantic_and_metric = AsyncMock()

        await hooks._get_sync_confirmation_for_pair(
            semantic_model_file="/tmp/sem.yaml",
            metric_file="/tmp/met.yaml",
        )

        hooks._sync_semantic_and_metric.assert_not_called()
        callback.assert_awaited_once()

    async def test_interaction_cancelled_raises_generation_cancelled(self, hooks):
        """InteractionCancelled is wrapped in GenerationCancelledException."""
        hooks.broker.request = AsyncMock(side_effect=InteractionCancelled())

        with pytest.raises(GenerationCancelledException, match="User interrupted"):
            await hooks._get_sync_confirmation_for_pair(
                semantic_model_file="/tmp/sem.yaml",
                metric_file="/tmp/met.yaml",
            )

    async def test_display_content_includes_both_files(self, hooks):
        """Request content includes both file names when display_content is pre-built."""
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("n", callback))

        display = (
            "## Generated Semantic Model: sem.yaml\n\n"
            "*Path: /tmp/sem.yaml*\n\n"
            "```yaml\ndata_source:\n  name: orders\n```\n\n"
            "---\n\n"
            "## Generated Metric: met.yaml\n\n"
            "*Path: /tmp/met.yaml*\n\n"
            "```yaml\nmetric: revenue\n```\n"
        )

        await hooks._get_sync_confirmation_for_pair(
            semantic_model_file="/tmp/sem.yaml",
            metric_file="/tmp/met.yaml",
            display_content=display,
        )

        request_content = hooks.broker.request.call_args[1].get("contents") or hooks.broker.request.call_args[0][0]
        content_str = str(request_content)
        assert "sem.yaml" in content_str
        assert "met.yaml" in content_str


# ---------------------------------------------------------------------------
# Tests: _parse_subject_tree_from_tags (static method)
# ---------------------------------------------------------------------------


class TestParseSubjectTreeFromTags:
    def test_valid_tag_returns_path(self):
        tags = ["subject_tree: Finance/Revenue/Q1"]
        result = GenerationHooks._parse_subject_tree_from_tags(tags)
        assert result == ["Finance", "Revenue", "Q1"]

    def test_no_subject_tree_tag_returns_none(self):
        tags = ["some_tag", "another_tag"]
        result = GenerationHooks._parse_subject_tree_from_tags(tags)
        assert result is None

    def test_empty_list_returns_none(self):
        result = GenerationHooks._parse_subject_tree_from_tags([])
        assert result is None

    def test_none_returns_none(self):
        result = GenerationHooks._parse_subject_tree_from_tags(None)
        assert result is None

    def test_non_list_returns_none(self):
        result = GenerationHooks._parse_subject_tree_from_tags("not a list")
        assert result is None

    def test_single_component_path(self):
        tags = ["subject_tree: Finance"]
        result = GenerationHooks._parse_subject_tree_from_tags(tags)
        assert result == ["Finance"]

    def test_tag_with_extra_whitespace(self):
        tags = ["subject_tree:  Sales / Marketing "]
        result = GenerationHooks._parse_subject_tree_from_tags(tags)
        assert result == ["Sales", "Marketing"]

    def test_non_string_tag_ignored(self):
        tags = [42, None, "subject_tree: Finance/Revenue"]
        result = GenerationHooks._parse_subject_tree_from_tags(tags)
        assert result == ["Finance", "Revenue"]


# ---------------------------------------------------------------------------
# Tests: _sync_semantic_to_db — boolean coercion
# ---------------------------------------------------------------------------


class TestSyncSemanticToDbBooleanCoercion:
    """Verify that YAML fields like create_metric and is_partition are coerced to bool.

    Root cause: YAML values like ``1.0`` or ``1`` are not Python bools.
    When table-kind rows and column-kind rows share a DataFrame, missing
    fields become NaN → pandas promotes bool columns to float64 →
    PostgreSQL rejects ``double precision`` for a ``boolean`` column.
    """

    @staticmethod
    def _build_yaml(create_metric_value, is_partition_value):
        import yaml

        doc = {
            "data_source": {
                "name": "test_table",
                "description": "Test table",
                "sql_table": "db.test_table",
                "measures": [
                    {
                        "name": "total_amount",
                        "description": "Total amount",
                        "agg": "SUM",
                        "expr": "amount",
                        "create_metric": create_metric_value,
                    }
                ],
                "dimensions": [
                    {
                        "name": "created_at",
                        "type": "TIME",
                        "description": "Creation time",
                        "expr": "created_at",
                        "type_params": {
                            "is_primary": True,
                            "time_granularity": "DAY",
                        },
                        "is_partition": is_partition_value,
                    }
                ],
            }
        }
        return yaml.safe_dump(doc, allow_unicode=True)

    @pytest.mark.parametrize(
        "create_metric_val,is_partition_val",
        [
            (1.0, 1.0),
            (1, 1),
            (True, True),
            ("yes", "yes"),
        ],
    )
    def test_boolean_fields_are_coerced(self, agent_config, create_metric_val, is_partition_val, tmp_path):
        yaml_content = self._build_yaml(create_metric_val, is_partition_val)
        yaml_file = tmp_path / "test_semantic.yml"
        yaml_file.write_text(yaml_content)

        # Configure agent_config mock to have required db_config attributes
        db_config = MagicMock()
        db_config.catalog = ""
        db_config.database = "test_db"
        db_config.schema = "public"
        db_config.db_type = "postgresql"
        agent_config.current_db_config.return_value = db_config
        agent_config.namespaces = ["test_ns"]

        captured_semantic = []
        captured_metric = []

        def fake_upsert_semantic(objects):
            captured_semantic.extend(objects)

        def fake_upsert_metric(objects):
            captured_metric.extend(objects)

        mock_semantic_rag = MagicMock()
        mock_semantic_rag.upsert_batch = fake_upsert_semantic
        mock_metric_rag = MagicMock()
        mock_metric_rag.upsert_batch = fake_upsert_metric

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=mock_metric_rag),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(yaml_file),
                agent_config=agent_config,
            )

        assert result["success"], f"Sync failed: {result.get('error')}"

        # Find the measure row (has create_metric) and dimension row (has is_partition)
        measure_rows = [o for o in captured_semantic if o.get("agg") == "SUM"]
        dim_rows = [o for o in captured_semantic if o.get("is_dimension") is True]

        assert len(measure_rows) == 1, f"Expected 1 measure row, got {len(measure_rows)}"
        assert len(dim_rows) == 1, f"Expected 1 dimension row, got {len(dim_rows)}"

        # Core assertion: create_metric must be Python bool, not float/int/str
        assert measure_rows[0]["create_metric"] is True
        assert type(measure_rows[0]["create_metric"]) is bool

        # Core assertion: is_partition must be Python bool
        assert dim_rows[0]["is_partition"] is True
        assert type(dim_rows[0]["is_partition"]) is bool

        # Also verify table-kind row has bool defaults (not NaN)
        table_rows = [o for o in captured_semantic if o.get("is_dimension") is False and o.get("is_measure") is False]
        assert len(table_rows) >= 1
        assert type(table_rows[0]["create_metric"]) is bool
        assert type(table_rows[0]["is_partition"]) is bool


# ---------------------------------------------------------------------------
# Tests: _sync_semantic_to_db actionable error for empty metric-only sync
# ---------------------------------------------------------------------------


class TestSyncSemanticToDbMetricOnlyDiagnostic:
    """Metric-only syncs (e.g. end_metric_generation) emit a tailored error
    message when the file has no `metric:` blocks, so the LLM can self-correct
    instead of getting a generic "No valid objects found to sync"."""

    def test_metric_only_sync_with_no_metric_blocks_returns_actionable_error(self, agent_config, tmp_path):
        empty_metric = tmp_path / "frpm_metrics.yml"
        empty_metric.write_text(
            "# Generated metric documentation\n\n## Summary\n\n- avg_percent_eligible_free_ages_5_17\n"
        )

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG"),
            patch("datus.cli.generation_hooks.MetricRAG"),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(empty_metric),
                agent_config=agent_config,
                include_semantic_objects=False,
                include_metrics=True,
            )

        assert result["success"] is False
        assert "no `metric:` YAML blocks" in result["error"]
        assert "create_metric: true" in result["error"]
        assert str(empty_metric) in result["error"]

    def test_combined_sync_keeps_generic_error_when_both_missing(self, agent_config, tmp_path):
        """A combined sync (semantic + metrics) with neither still uses the
        original generic message; the new diagnostic only fires for the
        metrics-only branch."""
        empty = tmp_path / "empty.yml"
        empty.write_text("# nothing useful\n")

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG"),
            patch("datus.cli.generation_hooks.MetricRAG"),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(empty),
                agent_config=agent_config,
                include_semantic_objects=True,
                include_metrics=True,
            )

        assert result["success"] is False
        assert result["error"] == "No data_source or metrics found in YAML file"


# ---------------------------------------------------------------------------
# Tests: _get_base_dir edge cases (resolver missing / exception)
# ---------------------------------------------------------------------------


class TestGetBaseDirEdgeCases:
    def test_returns_none_when_resolver_attr_is_none(self, broker):
        """path_manager exists but the named resolver attribute is None."""
        cfg = MagicMock()
        cfg.current_namespace = "ns"
        cfg.path_manager = MagicMock(spec=[])  # no attrs → getattr returns None
        h = GenerationHooks(broker=broker, agent_config=cfg)
        assert h._get_base_dir("semantic") is None

    def test_returns_none_when_resolver_raises(self, broker):
        """Exceptions raised by the resolver are caught and return None."""
        cfg = MagicMock()
        cfg.current_namespace = "ns"
        cfg.path_manager = MagicMock()
        cfg.path_manager.semantic_model_path = MagicMock(side_effect=RuntimeError("boom"))
        h = GenerationHooks(broker=broker, agent_config=cfg)
        assert h._get_base_dir("semantic") is None


class TestResolvePathCommonpathValueError:
    def test_returns_empty_when_commonpath_raises_value_error(self, broker):
        """When os.path.commonpath raises ValueError (e.g. mixed drives), we
        can't verify containment, so the resolver must fail closed by
        returning an empty string (not the original path)."""
        cfg = MagicMock()
        cfg.current_namespace = "ns"
        cfg.path_manager = MagicMock()
        cfg.path_manager.knowledge_base_home = Path("/ws")
        h = GenerationHooks(broker=broker, agent_config=cfg)
        with patch("datus.cli.generation_hooks.os.path.commonpath", side_effect=ValueError("mixed drives")):
            assert h._resolve_path("orders.yml", "semantic") == ""


# ---------------------------------------------------------------------------
# Tests: _handle_end_metric_generation resolves relative paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleEndMetricGeneration:
    async def test_missing_metric_file_warns_and_returns(self, hooks):
        hooks._extract_metric_generation_result = MagicMock(return_value=(None, None, {}))
        hooks._process_single_file = AsyncMock()
        hooks._process_metric_with_semantic_model = AsyncMock()
        await hooks._handle_end_metric_generation({"result": {}})
        hooks._process_single_file.assert_not_awaited()
        hooks._process_metric_with_semantic_model.assert_not_awaited()

    async def test_resolves_relative_paths_via_resolve_path(self, hooks):
        """Relative metric_file and semantic_model_file must be resolved through _resolve_path."""
        hooks._extract_metric_generation_result = MagicMock(
            return_value=("metrics/orders.yml", "semantic/orders.yml", {"m": "SELECT 1"})
        )
        hooks._process_metric_with_semantic_model = AsyncMock()
        hooks._resolve_path = MagicMock(side_effect=lambda p, k: f"/ws/sm/{p}" if p else p)

        await hooks._handle_end_metric_generation({"result": {"metric_file": "metrics/orders.yml"}})

        hooks._resolve_path.assert_any_call("metrics/orders.yml", "semantic")
        hooks._resolve_path.assert_any_call("semantic/orders.yml", "semantic")
        hooks._process_metric_with_semantic_model.assert_awaited_once_with(
            "/ws/sm/semantic/orders.yml", "/ws/sm/metrics/orders.yml", {"m": "SELECT 1"}
        )

    async def test_no_semantic_model_falls_back_to_single_file(self, hooks):
        hooks._extract_metric_generation_result = MagicMock(return_value=("metrics/orders.yml", None, {"m": "SQL"}))
        hooks._process_single_file = AsyncMock()
        hooks._resolve_path = MagicMock(side_effect=lambda p, k: f"/ws/sm/{p}" if p else p)

        await hooks._handle_end_metric_generation({"result": {}})

        hooks._process_single_file.assert_awaited_once_with("/ws/sm/metrics/orders.yml", metric_sqls={"m": "SQL"})

    async def test_cancelled_exception_absorbed(self, hooks):
        hooks._extract_metric_generation_result = MagicMock(return_value=("m.yml", None, {}))
        hooks._resolve_path = MagicMock(side_effect=lambda p, k: p)
        hooks._process_single_file = AsyncMock(side_effect=GenerationCancelledException("user-cancel"))
        await hooks._handle_end_metric_generation({"result": {}})  # must not raise

    async def test_unexpected_exception_absorbed(self, hooks):
        hooks._extract_metric_generation_result = MagicMock(side_effect=RuntimeError("boom"))
        await hooks._handle_end_metric_generation({"result": {}})  # must not raise


# ---------------------------------------------------------------------------
# Tests: normalize_kb_relative_path (pure function)
# ---------------------------------------------------------------------------


class TestNormalizeKbRelativePath:
    def test_prepends_when_prefix_missing(self):
        assert (
            normalize_kb_relative_path("orders.yaml", "semantic", "school_db")
            == "semantic_models/school_db/orders.yaml"
        )

    def test_prepends_for_sql_summary(self):
        assert (
            normalize_kb_relative_path("q_001.yaml", "sql_summary", "school_db") == "sql_summaries/school_db/q_001.yaml"
        )

    def test_prepends_for_ext_knowledge(self):
        assert (
            normalize_kb_relative_path("notes.yaml", "ext_knowledge", "school_db")
            == "ext_knowledge/school_db/notes.yaml"
        )

    def test_metric_kind_co_locates_with_semantic_models(self):
        """metrics live under semantic_models/{db}/metrics/ — same root as semantic."""
        assert (
            normalize_kb_relative_path("metrics/orders_metrics.yaml", "metric", "school_db")
            == "semantic_models/school_db/metrics/orders_metrics.yaml"
        )

    def test_idempotent_when_prefix_already_correct(self):
        already = "semantic_models/school_db/orders.yaml"
        assert normalize_kb_relative_path(already, "semantic", "school_db") == already

    def test_passes_through_paths_in_other_namespaces(self):
        path = "semantic_models/other_db/orders.yaml"
        assert normalize_kb_relative_path(path, "semantic", "school_db") == path

    def test_passes_through_paths_in_other_kinds(self):
        path = "sql_summaries/school_db/q_001.yaml"
        assert normalize_kb_relative_path(path, "semantic", "school_db") == path

    def test_absolute_paths_unchanged(self):
        assert normalize_kb_relative_path("/abs/path/orders.yaml", "semantic", "school_db") == "/abs/path/orders.yaml"

    def test_empty_path_unchanged(self):
        assert normalize_kb_relative_path("", "semantic", "school_db") == ""

    def test_dot_path_unchanged(self):
        assert normalize_kb_relative_path(".", "semantic", "school_db") == "."

    def test_parent_traversal_unchanged(self):
        assert normalize_kb_relative_path("../../etc/passwd", "semantic", "school_db") == "../../etc/passwd"

    def test_unknown_kind_unchanged(self):
        assert normalize_kb_relative_path("orders.yaml", "unknown", "school_db") == "orders.yaml"

    def test_missing_namespace_unchanged(self):
        assert normalize_kb_relative_path("orders.yaml", "semantic", None) == "orders.yaml"


# ---------------------------------------------------------------------------
# Tests: make_kb_path_normalizer factory
# ---------------------------------------------------------------------------


class _StubCfg:
    """Minimal agent_config stand-in for normalizer factory tests."""

    def __init__(self, ns: str):
        self.current_namespace = ns


class TestMakeKbPathNormalizer:
    def test_uses_default_kind_when_file_type_missing(self):
        normalizer = make_kb_path_normalizer(_StubCfg("db"), default_kind="semantic")
        assert normalizer("orders.yaml", None) == "semantic_models/db/orders.yaml"

    def test_file_type_overrides_default_kind(self):
        normalizer = make_kb_path_normalizer(_StubCfg("db"), default_kind="semantic")
        assert normalizer("q_001.yaml", "sql_summary") == "sql_summaries/db/q_001.yaml"

    def test_file_type_aliases_recognized(self):
        normalizer = make_kb_path_normalizer(_StubCfg("db"), default_kind=None)
        assert normalizer("orders.yaml", "semantic_model") == "semantic_models/db/orders.yaml"
        assert normalizer("metrics/x.yaml", "metric") == "semantic_models/db/metrics/x.yaml"
        assert normalizer("notes.yaml", "ext_knowledge") == "ext_knowledge/db/notes.yaml"

    def test_namespace_resolved_at_call_time(self):
        """Sub-agent switches mid-session must be honored — closure rebinds each call."""
        cfg = _StubCfg("ns_x")
        normalizer = make_kb_path_normalizer(cfg, default_kind="semantic")
        assert normalizer("orders.yaml", None) == "semantic_models/ns_x/orders.yaml"
        cfg.current_namespace = "ns_y"
        assert normalizer("orders.yaml", None) == "semantic_models/ns_y/orders.yaml"

    def test_strict_kind_rejects_cross_kind_write(self):
        """Mutating ops (strict_kind=True) must reject writes to peer kinds' subdirs."""
        normalizer = make_kb_path_normalizer(_StubCfg("db"), default_kind="semantic")
        # Read-lax: cross-kind reads still allowed.
        assert normalizer("sql_summaries/db/q.yaml", None) == "sql_summaries/db/q.yaml"
        # Write-strict: the same cross-kind path is refused.
        with pytest.raises(ValueError, match="Write to 'sql_summaries/' is not allowed"):
            normalizer("sql_summaries/db/q.yaml", None, strict_kind=True)

    def test_strict_kind_ignores_file_type_override(self):
        """In strict mode, file_type cannot be used to switch kinds."""
        normalizer = make_kb_path_normalizer(_StubCfg("db"), default_kind="semantic")
        # Without strict: file_type override is honored.
        assert normalizer("q.yaml", "sql_summary") == "sql_summaries/db/q.yaml"
        # With strict: override is ignored; default_kind wins.
        assert normalizer("q.yaml", "sql_summary", strict_kind=True) == "semantic_models/db/q.yaml"

    def test_strict_kind_rejects_cross_namespace_prefixed_write(self):
        """Even within the same kind, an explicit prefix pointing at another
        namespace must be rejected by a mutating op — otherwise a node whose
        ``current_namespace`` is ``db`` could overwrite ``other_db``'s KB by
        emitting ``semantic_models/other_db/orders.yml`` verbatim.

        Rationale: with ``FilesystemFuncTool``'s ``root_path`` widened to
        ``knowledge_base_home`` (so reads can browse peer namespaces), the
        strict normalizer is the last line of defence for write/edit ops.
        """
        normalizer = make_kb_path_normalizer(_StubCfg("db"), default_kind="semantic")
        # Read-lax: explicit cross-namespace prefix is honored so the LLM can
        # browse peer namespaces.
        assert normalizer("semantic_models/other_db/orders.yml", None) == "semantic_models/other_db/orders.yml"
        # Write-strict: the same path must be refused.
        with pytest.raises(ValueError, match="other_db"):
            normalizer("semantic_models/other_db/orders.yml", None, strict_kind=True)

    def test_strict_kind_allows_correct_namespace_prefix(self):
        """Own-namespace prefix is still accepted in strict mode."""
        normalizer = make_kb_path_normalizer(_StubCfg("db"), default_kind="semantic")
        assert normalizer("semantic_models/db/orders.yml", None, strict_kind=True) == "semantic_models/db/orders.yml"


# ---------------------------------------------------------------------------
# Tests: hook + tool agreement — _resolve_path finds files written via the
# same normalizer regardless of whether the LLM emitted a naked filename.
# ---------------------------------------------------------------------------


class TestHookAndToolPathAgreement:
    def test_resolve_path_finds_naked_file_after_normalized_write(self, tmp_path, real_agent_config):
        """FilesystemFuncTool writes orders.yml → hook resolves 'orders.yml' to the same on-disk path."""
        kb_root = Path(str(real_agent_config.path_manager.knowledge_base_home))

        tool = FilesystemFuncTool(
            root_path=str(kb_root),
            path_normalizer=make_kb_path_normalizer(real_agent_config, default_kind="semantic"),
        )
        write_result = tool.write_file("orders.yml", "id: orders\n", file_type="semantic_model")
        assert write_result.success == 1

        hooks = GenerationHooks(broker=None, agent_config=real_agent_config)
        resolved = hooks._resolve_path("orders.yml", "semantic")

        on_disk = kb_root / "semantic_models" / real_agent_config.current_namespace / "orders.yml"
        assert os.path.realpath(resolved) == os.path.realpath(str(on_disk))
        assert Path(resolved).is_file()

    def test_extract_filepaths_resolves_relative_entries_against_kb_home(self, real_agent_config):
        """end_semantic_model_generation payloads with bare filenames resolve correctly."""
        kb_root = Path(str(real_agent_config.path_manager.knowledge_base_home))
        target = kb_root / "semantic_models" / real_agent_config.current_namespace / "orders.yml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data\n")

        hooks = GenerationHooks(broker=None, agent_config=real_agent_config)
        paths = hooks._extract_filepaths_from_result({"result": {"semantic_model_files": ["orders.yml"]}})
        assert len(paths) == 1
        assert os.path.realpath(paths[0]) == os.path.realpath(str(target))


# ---------------------------------------------------------------------------
# Tests: resolve_kb_sandbox_path — used by workflow-mode _save_to_db() helpers
# ---------------------------------------------------------------------------


class _SandboxCfg:
    """Minimal agent_config stand-in for resolve_kb_sandbox_path tests."""

    def __init__(self, ns: str):
        self.current_namespace = ns


class TestResolveKbSandboxPath:
    def test_empty_path_returns_none(self, tmp_path):
        assert resolve_kb_sandbox_path("", "sql_summary", _SandboxCfg("db"), str(tmp_path)) is None

    def test_bare_filename_is_prefixed_under_sandbox(self, tmp_path):
        kb = tmp_path
        resolved = resolve_kb_sandbox_path("q_001.yaml", "sql_summary", _SandboxCfg("db"), str(kb))
        assert resolved == os.path.normpath(str(kb / "sql_summaries" / "db" / "q_001.yaml"))

    def test_fully_prefixed_relative_path_passes_through(self, tmp_path):
        kb = tmp_path
        resolved = resolve_kb_sandbox_path("sql_summaries/db/q.yaml", "sql_summary", _SandboxCfg("db"), str(kb))
        assert resolved == os.path.normpath(str(kb / "sql_summaries" / "db" / "q.yaml"))

    def test_absolute_path_inside_sandbox_accepted(self, tmp_path):
        kb = tmp_path
        (kb / "sql_summaries" / "db").mkdir(parents=True)
        inside = kb / "sql_summaries" / "db" / "q.yaml"
        inside.write_text("x")
        resolved = resolve_kb_sandbox_path(str(inside), "sql_summary", _SandboxCfg("db"), str(kb))
        assert os.path.realpath(resolved) == os.path.realpath(str(inside))

    def test_absolute_path_outside_sandbox_rejected(self, tmp_path):
        """A fabricated absolute path outside the sandbox must be refused so
        _save_to_db never syncs an arbitrary on-disk file."""
        assert resolve_kb_sandbox_path("/etc/passwd", "sql_summary", _SandboxCfg("db"), str(tmp_path)) is None

    def test_cross_kind_prefix_rejected(self, tmp_path):
        """Workflow returning ``ext_knowledge/db/foo.yaml`` from a
        sql_summary node must be refused — the prompt-compliant output here
        is restricted to ``sql_summaries/``."""
        assert (
            resolve_kb_sandbox_path("ext_knowledge/db/foo.yaml", "sql_summary", _SandboxCfg("db"), str(tmp_path))
            is None
        )

    def test_cross_namespace_prefix_rejected(self, tmp_path):
        """sql_summaries/other_db/q.yaml is inside the kind but outside the
        current namespace's sandbox → rejected (no cross-namespace writes)."""
        assert (
            resolve_kb_sandbox_path("sql_summaries/other_db/q.yaml", "sql_summary", _SandboxCfg("db"), str(tmp_path))
            is None
        )

    def test_traversal_escape_rejected(self, tmp_path):
        """``../../etc/passwd`` resolves outside the sandbox → rejected."""
        assert resolve_kb_sandbox_path("../../etc/passwd", "sql_summary", _SandboxCfg("db"), str(tmp_path)) is None

    def test_unknown_kind_no_containment_check(self, tmp_path):
        """For an unknown kind we cannot compute a sandbox — fall back to
        just normalizing against knowledge_base_dir."""
        resolved = resolve_kb_sandbox_path("foo.yaml", "unknown", _SandboxCfg("db"), str(tmp_path))
        assert resolved == os.path.normpath(str(tmp_path / "foo.yaml"))

    def test_missing_namespace_no_containment_check(self, tmp_path):
        """Without a namespace we cannot compute the {kind}/{ns}/ sandbox so
        containment is skipped, matching normalize_kb_relative_path semantics."""
        resolved = resolve_kb_sandbox_path("foo.yaml", "sql_summary", _SandboxCfg(None), str(tmp_path))
        assert resolved == os.path.normpath(str(tmp_path / "foo.yaml"))

    def test_commonpath_value_error_fails_closed(self, tmp_path):
        """Simulate os.path.commonpath raising (e.g. mixed drives on
        Windows) — the resolver must fail closed with None."""
        with patch("datus.cli.generation_hooks.os.path.commonpath", side_effect=ValueError("mixed drives")):
            assert resolve_kb_sandbox_path("q.yaml", "sql_summary", _SandboxCfg("db"), str(tmp_path)) is None
