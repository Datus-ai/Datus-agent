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
    cfg.path_manager = MagicMock()
    cfg.path_manager.semantic_model_path.return_value = Path("/tmp/datus_test/semantic_models/test_ns")
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

    async def test_reference_template_type_calls_sync(self, hooks):
        mock_result = {"success": True, "message": "Template synced"}
        with patch(
            "datus.cli.generation_hooks.GenerationHooks._sync_reference_template_to_db", return_value=mock_result
        ):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "reference_template")
        assert "Successfully synced" in result

    async def test_exception_returns_error_string(self, hooks):
        with patch(
            "datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db",
            side_effect=RuntimeError("disk full"),
        ):
            result = await hooks._sync_to_storage("/tmp/file.yaml", "semantic")
        assert "error" in result.lower() or "Sync error" in result


# ---------------------------------------------------------------------------
# Tests: _sync_reference_template_to_db
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
                    "name": "test_template",
                    "summary": "Test template",
                    "search_text": "test template val",
                    "subject_tree": "Sales/Revenue",
                    "tags": "test",
                }
            )
        )

        mock_config = MagicMock()
        mock_config.current_namespace = "test_ns"

        with (
            patch("datus.storage.reference_template.store.ReferenceTemplateRAG") as mock_rag_cls,
            patch("datus.storage.reference_template.init_utils.exists_reference_templates", return_value=set()),
        ):
            mock_rag = mock_rag_cls.return_value
            mock_rag.upsert_batch = MagicMock()

            result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), mock_config)

        assert result["success"] is True
        assert "Synced" in result["message"]

    def test_missing_sql_field(self, tmp_path):
        import yaml

        from datus.cli.generation_hooks import GenerationHooks

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml.dump({"name": "no_sql"}))

        result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), MagicMock())
        assert result["success"] is False
        assert "No reference_template data" in result["error"]

    def test_duplicate_skipped(self, tmp_path):
        import yaml

        from datus.cli.generation_hooks import GenerationHooks

        yaml_file = tmp_path / "dup.yaml"
        yaml_file.write_text(yaml.dump({"sql": "SELECT 1", "name": "dup", "summary": "x", "search_text": "x"}))

        mock_config = MagicMock()
        with (
            patch("datus.storage.reference_template.store.ReferenceTemplateRAG"),
            patch(
                "datus.storage.reference_template.init_utils.exists_reference_templates",
                return_value={"existing_id"},
            ),
            patch(
                "datus.storage.reference_template.init_utils.gen_reference_template_id",
                return_value="existing_id",
            ),
        ):
            result = GenerationHooks._sync_reference_template_to_db(str(yaml_file), mock_config)

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
        """Choosing 'y' calls _sync_to_storage for both files."""
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("y", callback))
        hooks._sync_to_storage = AsyncMock(return_value="Synced OK")

        await hooks._get_sync_confirmation_for_pair(
            "/tmp/sem.yaml",
            "data_source:\n  name: orders\n",
            "/tmp/met.yaml",
            "metric: revenue\n",
        )

        assert hooks._sync_to_storage.await_count == 2
        hooks._sync_to_storage.assert_any_await("/tmp/sem.yaml", "semantic")
        hooks._sync_to_storage.assert_any_await("/tmp/met.yaml", "semantic")
        callback.assert_awaited_once()

    async def test_reject_skips_sync(self, hooks):
        """Choosing 'n' does not call _sync_to_storage."""
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("n", callback))
        hooks._sync_to_storage = AsyncMock()

        await hooks._get_sync_confirmation_for_pair(
            "/tmp/sem.yaml",
            "data_source:\n  name: orders\n",
            "/tmp/met.yaml",
            "metric: revenue\n",
        )

        hooks._sync_to_storage.assert_not_called()
        callback.assert_awaited_once()

    async def test_interaction_cancelled_raises_generation_cancelled(self, hooks):
        """InteractionCancelled is wrapped in GenerationCancelledException."""
        hooks.broker.request = AsyncMock(side_effect=InteractionCancelled())

        with pytest.raises(GenerationCancelledException, match="User interrupted"):
            await hooks._get_sync_confirmation_for_pair(
                "/tmp/sem.yaml",
                "sem content",
                "/tmp/met.yaml",
                "met content",
            )

    async def test_display_content_includes_both_files(self, hooks):
        """Request content includes both semantic model and metric YAML."""
        callback = AsyncMock()
        hooks.broker.request = AsyncMock(return_value=("n", callback))

        await hooks._get_sync_confirmation_for_pair(
            "/tmp/sem.yaml",
            "data_source:\n  name: orders\n",
            "/tmp/met.yaml",
            "metric: revenue\n",
        )

        request_content = hooks.broker.request.call_args[1].get("contents") or hooks.broker.request.call_args[0][0]
        content_str = str(request_content)
        assert "sem.yaml" in content_str
        assert "met.yaml" in content_str
        assert "data_source" in content_str
        assert "revenue" in content_str


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
