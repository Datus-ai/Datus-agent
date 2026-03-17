# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Extended unit tests for datus/cli/generation_hooks.py.

Covers additional uncovered lines:
- _is_sql_summary_tool_call / _is_ext_knowledge_tool_call
- _handle_sql_summary_result (full happy path with temp files)
- _handle_ext_knowledge_result (happy path)
- _get_sync_confirmation (yes/no choices, InteractionCancelled)
- _get_sync_confirmation_for_pair (yes/no choices, InteractionCancelled)
- _sync_to_storage (no config, invalid type, semantic, sql_summary, ext_knowledge)
- _process_metric_with_semantic_model (missing files, both exist, already processed)
- _parse_subject_tree_from_tags (static)
- GenerationCancelledException
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.cli.execution_state import InteractionCancelled
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
    cfg.db_type = "sqlite"
    return cfg


@pytest.fixture
def hooks(broker, agent_config):
    return GenerationHooks(broker=broker, agent_config=agent_config)


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

    async def test_happy_path_calls_confirmation(self, hooks):
        """File exists with content -> confirmation called."""
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_sql\nsql: SELECT 1\n")
            path = f.name
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_sql_summary_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()
        assert path in hooks.processed_files

    async def test_reference_sql_file_written_pattern(self, hooks):
        """'Reference SQL file written successfully:' pattern is also matched."""
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_sql\nsql: SELECT 1\n")
            path = f.name
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

    async def test_happy_path_calls_confirmation(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            path = f.name
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_ext_knowledge_result(result)
        finally:
            os.unlink(path)
        hooks._get_sync_confirmation.assert_awaited_once()
        assert path in hooks.processed_files

    async def test_ext_knowledge_file_written_pattern(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            path = f.name
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

    async def test_result_object_with_match(self, hooks):
        hooks._get_sync_confirmation = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            path = f.name
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
# Tests: _get_sync_confirmation_for_pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetSyncConfirmationForPair:
    async def test_choice_yes_calls_sync_pair(self, hooks):
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("y", callback))
        hooks._sync_semantic_and_metric = AsyncMock(return_value="PairSynced!")

        await hooks._get_sync_confirmation_for_pair(
            semantic_model_file="/tmp/sem.yaml",
            metric_file="/tmp/met.yaml",
        )

        hooks._sync_semantic_and_metric.assert_awaited_once()
        callback.assert_awaited_once()

    async def test_choice_no_calls_callback_with_file_names(self, hooks):
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("n", callback))

        await hooks._get_sync_confirmation_for_pair(
            semantic_model_file="/tmp/sem.yaml",
            metric_file="/tmp/met.yaml",
        )

        callback.assert_awaited_once()
        args = callback.call_args[0][0]
        assert "/tmp/sem.yaml" in args
        assert "/tmp/met.yaml" in args

    async def test_interaction_cancelled_raises(self, hooks):
        hooks.broker.request = AsyncMock(side_effect=InteractionCancelled())

        with pytest.raises(GenerationCancelledException):
            await hooks._get_sync_confirmation_for_pair(
                semantic_model_file="/tmp/sem.yaml",
                metric_file="/tmp/met.yaml",
            )


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

    async def test_exception_returns_error_string(self, hooks):
        with patch(
            "datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db",
            side_effect=RuntimeError("disk full"),
        ):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "semantic")
        assert "error" in result.lower() or "Sync error" in result


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
