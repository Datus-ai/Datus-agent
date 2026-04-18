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
    normalize_kb_relative_path,
    resolve_kb_sandbox_path,
)
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool  # noqa: F401

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
    # After the storage refactor, KB content lives under {project_root}/subject/
    # without per-namespace subdirectories.
    subject_dir = tmp_path / "subject"
    (subject_dir / "semantic_models").mkdir(parents=True, exist_ok=True)
    (subject_dir / "sql_summaries").mkdir(parents=True, exist_ok=True)
    (subject_dir / "ext_knowledge").mkdir(parents=True, exist_ok=True)
    cfg = MagicMock()
    cfg.home = str(tmp_path)
    cfg.current_database = "test_ns"
    cfg.current_namespace = "test_ns"
    cfg.db_type = "sqlite"
    cfg.path_manager = MagicMock()
    cfg.path_manager.semantic_model_path.return_value = subject_dir / "semantic_models"
    cfg.path_manager.sql_summary_path.return_value = subject_dir / "sql_summaries"
    cfg.path_manager.ext_knowledge_path.return_value = subject_dir / "ext_knowledge"
    # Real value so _resolve_path's realpath/commonpath containment check works.
    cfg.path_manager.subject_dir = subject_dir
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
        # Absolute paths inside the subject directory pass the containment
        # check and are returned normpath'd.
        sem_dir = Path(str(agent_config.path_manager.subject_dir)) / "semantic_models"
        paths_in = [str(sem_dir / "a.yaml"), str(sem_dir / "b.yaml")]
        result = {"result": {"semantic_model_files": paths_in}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == paths_in

    def test_from_dict_drops_paths_outside_subject(self, hooks):
        """Absolute paths outside the subject directory must be filtered out."""
        result = {"result": {"semantic_model_files": ["/etc/passwd", "/a/b.yaml"]}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == []

    def test_from_dict_no_files(self, hooks):
        result = {"result": {}}
        paths = hooks._extract_filepaths_from_result(result)
        assert paths == []

    def test_from_object_with_result(self, hooks, agent_config):
        sem_dir = Path(str(agent_config.path_manager.subject_dir)) / "semantic_models"
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

    The resolver joins relative paths against the project ``subject/`` directory
    after routing them through ``normalize_kb_relative_path`` — so a naked
    filename written by the LLM (e.g. ``orders.yml``) lands at
    ``{subject_dir}/{type_subdir}/orders.yml``, matching where the
    FilesystemFuncTool actually wrote the file.
    """

    def _make_hooks(self, broker, subject="/ws"):
        cfg = MagicMock()
        cfg.path_manager = MagicMock()
        cfg.path_manager.subject_dir = Path(subject)
        return GenerationHooks(broker=broker, agent_config=cfg), cfg

    def test_absolute_path_outside_subject_rejected(self, broker):
        """Absolute paths outside subject_dir are rejected (fail closed)."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("/etc/passwd", "semantic") == ""
        assert h._resolve_path("/abs/path/to/file.yml", "semantic") == ""

    def test_absolute_path_inside_subject_is_normpathed(self, broker, tmp_path):
        """Absolute paths that resolve inside subject_dir are accepted."""
        subject = tmp_path / "subject"
        (subject / "semantic_models").mkdir(parents=True)
        inside = subject / "semantic_models" / "orders.yml"
        inside.write_text("x")
        h, _ = self._make_hooks(broker, subject=str(subject))
        resolved = h._resolve_path(str(inside), "semantic")
        assert os.path.realpath(resolved) == os.path.realpath(str(inside))

    def test_relative_joined_for_semantic(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("orders.yml", "semantic") == "/ws/semantic_models/orders.yml"

    def test_relative_joined_for_sql_summary(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("q_001.yaml", "sql_summary") == "/ws/sql_summaries/q_001.yaml"

    def test_relative_joined_for_ext_knowledge(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("gmv.yaml", "ext_knowledge") == "/ws/ext_knowledge/gmv.yaml"

    def test_nested_relative_joined(self, broker):
        h, _ = self._make_hooks(broker)
        assert (
            h._resolve_path("metrics/orders_metrics.yml", "semantic")
            == "/ws/semantic_models/metrics/orders_metrics.yml"
        )

    def test_already_prefixed_path_passes_through(self, broker):
        """LLM that includes the ``{subdir}/`` prefix must not be double-prefixed."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("semantic_models/orders.yml", "semantic") == "/ws/semantic_models/orders.yml"

    def test_empty_path_returns_unchanged(self, broker):
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("", "semantic") == ""

    def test_unknown_kind_resolves_against_subject_root(self, broker):
        """Unknown kind: normalizer adds no prefix, but path still rooted at subject_dir."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("orders.yml", "unknown") == "/ws/orders.yml"

    def test_no_agent_config_leaves_relative_unchanged(self, broker):
        h = GenerationHooks(broker=broker, agent_config=None)
        assert h._resolve_path("orders.yml", "semantic") == "orders.yml"

    def test_rejects_traversal_escape(self, broker):
        """``../../etc/passwd`` resolves outside subject_dir and must be rejected."""
        h, _ = self._make_hooks(broker)
        assert h._resolve_path("../../etc/passwd", "semantic") == ""

    def test_allows_traversal_that_stays_inside_subject(self, broker):
        """A path whose normpath stays under subject_dir is allowed."""
        h, _ = self._make_hooks(broker)
        # ``metrics/../orders.yml`` → prepend → ``semantic_models/metrics/../orders.yml``
        # → normpath under /ws → ``/ws/semantic_models/orders.yml``
        assert h._resolve_path("metrics/../orders.yml", "semantic") == "/ws/semantic_models/orders.yml"

    def test_rejects_symlink_that_escapes_subject(self, broker, tmp_path):
        """A symlink inside the KB whose target is outside must be rejected."""
        subject = tmp_path / "subject"
        sub = subject / "semantic_models"
        sub.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.yml").write_text("x")
        (sub / "leak.yml").symlink_to(outside / "secret.yml")

        h, _ = self._make_hooks(broker, subject=str(subject))
        assert h._resolve_path("leak.yml", "semantic") == ""

    def test_extract_filepaths_resolves_relative_entries(self, broker):
        h, _ = self._make_hooks(broker)
        # The relative entry resolves inside subject_dir; the escaping absolute entry
        # is dropped so downstream processing never sees it.
        result = {"result": {"semantic_model_files": ["orders.yml", "/abs/customers.yml"]}}
        paths = h._extract_filepaths_from_result(result)
        assert paths == ["/ws/semantic_models/orders.yml"]


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
        sem_dir = Path(str(agent_config.path_manager.subject_dir)) / "semantic_models"
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
        sql_dir = Path(str(agent_config.path_manager.subject_dir)) / "sql_summaries"
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
        sql_dir = Path(str(agent_config.path_manager.subject_dir)) / "sql_summaries"
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
        ext_dir = Path(str(agent_config.path_manager.subject_dir)) / "ext_knowledge"
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
        ext_dir = Path(str(agent_config.path_manager.subject_dir)) / "ext_knowledge"
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
        ext_dir = Path(str(agent_config.path_manager.subject_dir)) / "ext_knowledge"
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
# Tests: _sync_reference_sql_to_db / _sync_reference_template_to_db
# ---------------------------------------------------------------------------


class TestSyncReferenceSqlToDb:
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


class TestSyncReferenceTemplateToDb:
    def test_valid_template_yaml(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "tpl.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "sql": "SELECT * FROM t WHERE x = '{{val}}'",
                    "name": "test_reference_template",
                    "summary": "Test reference template",
                    "search_text": "test reference template val",
                    "subject_tree": "Sales/Revenue",
                    "comment": "Helpful template",
                    "tags": "test",
                }
            )
        )

        mock_config = MagicMock()

        with (
            patch("datus.storage.reference_template.store.ReferenceTemplateRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_template.init_utils.exists_reference_templates",
                return_value=set(),
            ),
            patch(
                "datus.storage.reference_template.init_utils.gen_reference_template_id",
                return_value="new_tpl_id",
            ),
            patch(
                "datus.storage.reference_template.template_file_processor.extract_template_parameters",
                return_value=[{"name": "val", "type": "string"}],
            ),
        ):
            mock_rag = mock_rag_cls.return_value
            mock_rag.upsert_batch = MagicMock()

            result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), mock_config)

        assert result["success"] is True
        assert "Synced reference template" in result["message"]
        mock_rag.upsert_batch.assert_called_once()
        stored = mock_rag.upsert_batch.call_args.args[0][0]
        assert stored == {
            "id": "new_tpl_id",
            "name": "test_reference_template",
            "template": "SELECT * FROM t WHERE x = '{{val}}'",
            "parameters": json.dumps([{"name": "val", "type": "string"}]),
            "comment": "Helpful template",
            "summary": "Test reference template",
            "search_text": "test reference template val",
            "filepath": str(yaml_file),
            "subject_path": ["Sales", "Revenue"],
            "tags": "test",
        }

    def test_missing_sql_field(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml.dump({"name": "no_sql"}))

        result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), MagicMock())

        assert result["success"] is False
        assert "No reference_template data" in result["error"]

    def test_blank_sql_returns_error(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "blank.yaml"
        yaml_file.write_text(yaml.dump({"sql": "   ", "name": "blank"}))

        result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), MagicMock())

        assert result["success"] is False
        assert "non-empty string" in result["error"]

    def test_duplicate_skipped(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "dup.yaml"
        yaml_file.write_text(yaml.dump({"sql": "SELECT 1", "name": "dup_tpl"}))

        mock_config = MagicMock()
        with (
            patch("datus.storage.reference_template.store.ReferenceTemplateRAG"),
            patch(
                "datus.storage.reference_template.init_utils.exists_reference_templates",
                return_value={"existing_tpl_id"},
            ),
            patch(
                "datus.storage.reference_template.init_utils.gen_reference_template_id",
                return_value="existing_tpl_id",
            ),
        ):
            result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), mock_config)

        assert result["success"] is True
        assert "already exists" in result["message"]

    def test_storage_error_returns_failure(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "boom.yaml"
        yaml_file.write_text(yaml.dump({"sql": "SELECT 1", "name": "boom_tpl"}))

        mock_config = MagicMock()
        with (
            patch("datus.storage.reference_template.store.ReferenceTemplateRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_template.init_utils.exists_reference_templates",
                return_value=set(),
            ),
            patch(
                "datus.storage.reference_template.init_utils.gen_reference_template_id",
                return_value="boom_id",
            ),
            patch(
                "datus.storage.reference_template.template_file_processor.extract_template_parameters",
                return_value=[],
            ),
        ):
            mock_rag = mock_rag_cls.return_value
            mock_rag.upsert_batch.side_effect = RuntimeError("boom")

            result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), mock_config)

        assert result["success"] is False
        assert result["error"] == "boom"


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
        cfg.path_manager = MagicMock()
        cfg.path_manager.subject_dir = Path("/ws")
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
        assert normalize_kb_relative_path("orders.yaml", "semantic") == "semantic_models/orders.yaml"

    def test_prepends_for_sql_summary(self):
        assert normalize_kb_relative_path("q_001.yaml", "sql_summary") == "sql_summaries/q_001.yaml"

    def test_prepends_for_ext_knowledge(self):
        assert normalize_kb_relative_path("notes.yaml", "ext_knowledge") == "ext_knowledge/notes.yaml"

    def test_metric_kind_co_locates_with_semantic_models(self):
        """metrics live under semantic_models/metrics/ — same root as semantic."""
        assert (
            normalize_kb_relative_path("metrics/orders_metrics.yaml", "metric")
            == "semantic_models/metrics/orders_metrics.yaml"
        )

    def test_idempotent_when_prefix_already_correct(self):
        already = "semantic_models/orders.yaml"
        assert normalize_kb_relative_path(already, "semantic") == already

    def test_passes_through_paths_in_other_kinds(self):
        path = "sql_summaries/q_001.yaml"
        assert normalize_kb_relative_path(path, "semantic") == path

    def test_absolute_paths_unchanged(self):
        assert normalize_kb_relative_path("/abs/path/orders.yaml", "semantic") == "/abs/path/orders.yaml"

    def test_empty_path_unchanged(self):
        assert normalize_kb_relative_path("", "semantic") == ""

    def test_dot_path_unchanged(self):
        assert normalize_kb_relative_path(".", "semantic") == "."

    def test_parent_traversal_unchanged(self):
        assert normalize_kb_relative_path("../../etc/passwd", "semantic") == "../../etc/passwd"

    def test_unknown_kind_unchanged(self):
        assert normalize_kb_relative_path("orders.yaml", "unknown") == "orders.yaml"


# ---------------------------------------------------------------------------
# Tests: hook + tool agreement — _resolve_path finds files written by the
# tool (naked filename path was a normalizer concern; with normalizer gone
# the LLM writes the full prefix, so the hook resolver must keep returning
# the same absolute path regardless of which form the caller uses).
# ---------------------------------------------------------------------------


class TestHookAndToolPathAgreement:
    def test_resolve_path_finds_file_written_with_full_prefix(self, tmp_path, real_agent_config):
        """FilesystemFuncTool writes subject/semantic_models/orders.yml → hook resolves the same on-disk path."""
        subject_root = Path(str(real_agent_config.path_manager.subject_dir))
        project_root = subject_root.parent

        tool = FilesystemFuncTool(
            root_path=str(project_root),
            current_node="gen_semantic_model",
        )
        write_result = tool.write_file("subject/semantic_models/orders.yml", "id: orders\n")
        assert write_result.success == 1

        hooks = GenerationHooks(broker=None, agent_config=real_agent_config)
        # Hook's legacy resolver still accepts naked filenames via
        # normalize_kb_relative_path; the resolver path is decoupled from
        # the fs tool.
        resolved = hooks._resolve_path("orders.yml", "semantic")

        on_disk = subject_root / "semantic_models" / "orders.yml"
        assert os.path.realpath(resolved) == os.path.realpath(str(on_disk))
        assert Path(resolved).is_file()

    def test_extract_filepaths_resolves_relative_entries_against_subject(self, real_agent_config):
        """end_semantic_model_generation payloads with bare filenames resolve correctly."""
        subject_root = Path(str(real_agent_config.path_manager.subject_dir))
        target = subject_root / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data\n")

        hooks = GenerationHooks(broker=None, agent_config=real_agent_config)
        paths = hooks._extract_filepaths_from_result({"result": {"semantic_model_files": ["orders.yml"]}})
        assert len(paths) == 1
        assert os.path.realpath(paths[0]) == os.path.realpath(str(target))


# ---------------------------------------------------------------------------
# Tests: resolve_kb_sandbox_path — used by workflow-mode _save_to_db() helpers
# ---------------------------------------------------------------------------


class TestResolveKbSandboxPath:
    def test_empty_path_returns_none(self, tmp_path):
        assert resolve_kb_sandbox_path("", "sql_summary", str(tmp_path)) is None

    def test_bare_filename_is_prefixed_under_sandbox(self, tmp_path):
        kb = tmp_path
        resolved = resolve_kb_sandbox_path("q_001.yaml", "sql_summary", str(kb))
        assert resolved == os.path.normpath(str(kb / "sql_summaries" / "q_001.yaml"))

    def test_fully_prefixed_relative_path_passes_through(self, tmp_path):
        kb = tmp_path
        resolved = resolve_kb_sandbox_path("sql_summaries/q.yaml", "sql_summary", str(kb))
        assert resolved == os.path.normpath(str(kb / "sql_summaries" / "q.yaml"))

    def test_absolute_path_inside_sandbox_accepted(self, tmp_path):
        kb = tmp_path
        (kb / "sql_summaries").mkdir(parents=True)
        inside = kb / "sql_summaries" / "q.yaml"
        inside.write_text("x")
        resolved = resolve_kb_sandbox_path(str(inside), "sql_summary", str(kb))
        assert os.path.realpath(resolved) == os.path.realpath(str(inside))

    def test_absolute_path_outside_sandbox_rejected(self, tmp_path):
        """A fabricated absolute path outside the sandbox must be refused so
        _save_to_db never syncs an arbitrary on-disk file."""
        assert resolve_kb_sandbox_path("/etc/passwd", "sql_summary", str(tmp_path)) is None

    def test_cross_kind_prefix_rejected(self, tmp_path):
        """Workflow returning ``ext_knowledge/foo.yaml`` from a
        sql_summary node must be refused — the prompt-compliant output here
        is restricted to ``sql_summaries/``."""
        assert resolve_kb_sandbox_path("ext_knowledge/foo.yaml", "sql_summary", str(tmp_path)) is None

    def test_traversal_escape_rejected(self, tmp_path):
        """``../../etc/passwd`` resolves outside the sandbox → rejected."""
        assert resolve_kb_sandbox_path("../../etc/passwd", "sql_summary", str(tmp_path)) is None

    def test_unknown_kind_no_containment_check(self, tmp_path):
        """For an unknown kind we cannot compute a sandbox — fall back to
        just normalizing against knowledge_base_dir."""
        resolved = resolve_kb_sandbox_path("foo.yaml", "unknown", str(tmp_path))
        assert resolved == os.path.normpath(str(tmp_path / "foo.yaml"))

    def test_commonpath_value_error_fails_closed(self, tmp_path):
        """Simulate os.path.commonpath raising (e.g. mixed drives on
        Windows) — the resolver must fail closed with None."""
        with patch("datus.cli.generation_hooks.os.path.commonpath", side_effect=ValueError("mixed drives")):
            assert resolve_kb_sandbox_path("q.yaml", "sql_summary", str(tmp_path)) is None
