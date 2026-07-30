# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/generation_hooks.py — GenerationHooks.

All external dependencies are mocked. Tests cover:
- Initialization
- on_tool_end routing
- _extract_filepaths_from_result
- _handle_sql_summary_result
- _is_sql_summary_tool_call
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datus_db_core import connector_registry

from datus.cli.generation_hooks import (
    GenerationCancelledException,
    GenerationHooks,
    normalize_kb_relative_path,
    resolve_kb_sandbox_path,
)
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool  # noqa: F401
from datus.tools.func_tool.generation_evidence import GenerationEvidence

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
    # without per-datasource subdirectories.
    subject_dir = tmp_path / "subject"
    (subject_dir / "semantic_models").mkdir(parents=True, exist_ok=True)
    (subject_dir / "sql_summaries").mkdir(parents=True, exist_ok=True)
    cfg = MagicMock()
    cfg.home = str(tmp_path)
    cfg.current_datasource = "test_ns"
    cfg.current_datasource = "test_ns"
    cfg.db_type = "sqlite"
    cfg.path_manager = MagicMock()
    cfg.path_manager.semantic_model_path.return_value = subject_dir / "semantic_models"
    cfg.path_manager.sql_summary_path.return_value = subject_dir / "sql_summaries"
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
    async def test_routes_write_file_sql_summary(self, hooks):
        hooks._handle_sql_summary_result = AsyncMock()
        hooks._is_sql_summary_tool_call = MagicMock(return_value=True)
        tool = MagicMock()
        tool.name = "write_file"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_sql_summary_result.assert_awaited_once()

    async def test_unrelated_tool_does_nothing(self, hooks):
        tool = MagicMock()
        tool.name = "some_other_tool"
        assert await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result") is None

    async def test_tool_name_via_dunder_name(self, hooks):
        """Handles tools that use __name__ instead of .name attribute."""
        hooks._handle_sql_summary_result = AsyncMock()
        hooks._is_sql_summary_tool_call = MagicMock(return_value=True)
        tool = MagicMock(spec=[])  # no .name attribute
        tool.__name__ = "write_file"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result")
        hooks._handle_sql_summary_result.assert_awaited_once()

    async def test_records_validate_semantic_success(self, broker, agent_config):
        evidence = GenerationEvidence()
        hooks = GenerationHooks(broker=broker, agent_config=agent_config, generation_evidence=evidence)
        tool = MagicMock()
        tool.name = "validate_semantic"
        result = FuncToolResult(success=1, result={"valid": True, "issues": []})

        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, result)

        assert evidence.validation_passed is True

    async def test_records_query_metrics_dry_run_success(self, broker, agent_config):
        evidence = GenerationEvidence()
        hooks = GenerationHooks(broker=broker, agent_config=agent_config, generation_evidence=evidence)
        tool = MagicMock()
        tool.name = "query_metrics"
        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"metrics": ["revenue"], "dry_run": True})
        result = FuncToolResult(
            success=1,
            result={
                "columns": [],
                "data": [],
                "metadata": {"sql": "SELECT SUM(revenue) AS revenue FROM orders"},
            },
        )

        await hooks.on_tool_end(ctx, MagicMock(), tool, result)

        assert evidence.metric_dry_run_passed is True
        assert evidence.metric_sqls == {"revenue": "SELECT SUM(revenue) AS revenue FROM orders"}


# ---------------------------------------------------------------------------
# Tests: on_start / on_tool_start / on_handoff / on_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStubHooks:
    async def test_on_start(self, hooks):
        result = await hooks.on_start(MagicMock(), MagicMock())
        assert result is None

    async def test_on_tool_start(self, hooks):
        result = await hooks.on_tool_start(MagicMock(), MagicMock(), MagicMock())
        assert result is None

    async def test_on_handoff(self, hooks):
        result = await hooks.on_handoff(MagicMock(), MagicMock(), MagicMock())
        assert result is None

    async def test_on_end(self, hooks):
        result = await hooks.on_end(MagicMock(), MagicMock(), MagicMock())
        assert result is None


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


# ---------------------------------------------------------------------------
# Tests: _handle_sql_summary_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleSqlSummaryResult:
    async def test_no_file_path_returns_early(self, hooks):
        hooks._sync_generated_file = AsyncMock()
        result = {"result": "some unrelated message"}
        await hooks._handle_sql_summary_result(result)
        hooks._sync_generated_file.assert_not_called()

    async def test_file_not_exists_returns_early(self, hooks):
        hooks._sync_generated_file = AsyncMock()
        result = {"result": "File written successfully: /nonexistent/path.sql"}
        await hooks._handle_sql_summary_result(result)
        hooks._sync_generated_file.assert_not_called()


# ---------------------------------------------------------------------------
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
# Tests: _handle_sql_summary_result - additional branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleSqlSummaryResultExtended:
    async def test_result_object_with_no_match(self, hooks):
        """result.result doesn't match expected pattern -> early return."""
        hooks._sync_generated_file = AsyncMock()
        result = MagicMock()
        result.result = "Some unrelated message"
        await hooks._handle_sql_summary_result(result)
        hooks._sync_generated_file.assert_not_called()

    async def test_result_object_file_written_but_not_exists(self, hooks):
        """result.result matches pattern but file doesn't exist -> early return."""
        hooks._sync_generated_file = AsyncMock()
        result = MagicMock()
        result.result = "File written successfully: /nonexistent/path.yaml"
        await hooks._handle_sql_summary_result(result)
        hooks._sync_generated_file.assert_not_called()

    async def test_already_processed_skipped(self, hooks):
        """File already in processed_files -> skipped."""
        hooks._sync_generated_file = AsyncMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_sql\nsql: SELECT 1\n")
            path = f.name
        hooks.processed_files.add(path)
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_sql_summary_result(result)
        finally:
            os.unlink(path)
        hooks._sync_generated_file.assert_not_called()

    async def test_happy_path_auto_syncs(self, hooks, agent_config):
        """File exists with content -> sync called."""
        hooks._sync_generated_file = AsyncMock()
        sql_dir = Path(str(agent_config.path_manager.subject_dir)) / "sql_summaries"
        path_obj = sql_dir / "q_happy.yaml"
        path_obj.write_text("name: test_sql\nsql: SELECT 1\n")
        path = str(path_obj)
        try:
            result = {"result": f"File written successfully: {path}"}
            await hooks._handle_sql_summary_result(result)
        finally:
            os.unlink(path)
        hooks._sync_generated_file.assert_awaited_once()
        assert path in hooks.processed_files

    async def test_reference_sql_file_written_pattern(self, hooks, agent_config):
        """'Reference SQL file written successfully:' pattern is also matched."""
        hooks._sync_generated_file = AsyncMock()
        sql_dir = Path(str(agent_config.path_manager.subject_dir)) / "sql_summaries"
        path_obj = sql_dir / "q_ref.yaml"
        path_obj.write_text("name: test_sql\nsql: SELECT 1\n")
        path = str(path_obj)
        try:
            result = {"result": f"Reference SQL file written successfully: {path}"}
            await hooks._handle_sql_summary_result(result)
        finally:
            os.unlink(path)
        hooks._sync_generated_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: _sync_generated_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSyncGeneratedFile:
    async def test_auto_sync_calls_sync(self, hooks):
        hooks.broker.request = AsyncMock()
        hooks._sync_to_storage = AsyncMock(return_value="Synced!")

        await hooks._sync_generated_file(
            yaml_content="key: val",
            file_path="/tmp/test.yaml",
            yaml_type="semantic",
        )

        hooks._sync_to_storage.assert_awaited_once()
        hooks.broker.request.assert_not_awaited()

    async def test_auto_sync_ignores_deprecated_display_content(self, hooks):
        hooks.broker.request = AsyncMock()
        hooks._sync_to_storage = AsyncMock(return_value="Synced!")

        await hooks._sync_generated_file(
            yaml_content="key: val",
            file_path="/tmp/test.yaml",
            yaml_type="sql_summary",
            display_content="## Pre-built header\n```yaml\nkey: val\n```\n",
        )

        hooks._sync_to_storage.assert_awaited_once_with("/tmp/test.yaml", "sql_summary", metric_sqls=None)
        hooks.broker.request.assert_not_awaited()

    async def test_sync_error_propagates(self, hooks):
        hooks._sync_to_storage = AsyncMock(side_effect=RuntimeError("sync failed"))

        with pytest.raises(RuntimeError, match="sync failed"):
            await hooks._sync_generated_file(
                yaml_content="key: val",
                file_path="/tmp/test.yaml",
                yaml_type="semantic",
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
        assert hooks.generation_evidence.semantic_kb_sync_passed is True

    async def test_metric_type_calls_sync_metric(self, hooks):
        mock_result = {"success": True, "message": "1 metric synced"}
        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db", return_value=mock_result) as sync:
            result = await hooks._sync_to_storage("/tmp/metric.yaml", "metric", metric_sqls={"m": "SQL"})
        assert "Successfully synced" in result
        assert hooks.generation_evidence.metric_kb_sync_passed is True
        assert hooks.generation_evidence.semantic_kb_sync_passed is False
        sync.assert_called_once_with(
            "/tmp/metric.yaml",
            hooks.agent_config,
            include_semantic_objects=False,
            include_metrics=True,
            metric_sqls={"m": "SQL"},
            original_yaml_path="/tmp/metric.yaml",
            replace_metric_artifact=False,
        )

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
        assert hooks.generation_evidence.generic_kb_sync_passed is True

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
        assert "**Sync error:** disk full" in result


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
        agent_config.datasource_configs = ["test_ns"]

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


def test_sync_qualified_database_table_clears_stale_schema(agent_config, tmp_path, monkeypatch):
    yaml_file = tmp_path / "orders.yml"
    yaml_file.write_text(
        """
data_source:
  name: orders
  sql_table: project_a.orders
  dimensions:
    - name: status
      type: CATEGORICAL
      expr: status
""",
        encoding="utf-8",
    )
    db_config = MagicMock()
    db_config.catalog = ""
    db_config.database = "project_a"
    db_config.schema = "stale_schema"
    agent_config.current_db_config.return_value = db_config
    agent_config.db_type = "flexdb"

    def parse_identifier(identifier):
        parts = identifier.split(".")
        return {
            "catalog_name": "",
            "database_name": parts[0] if len(parts) > 1 else "",
            "schema_name": parts[1] if len(parts) == 3 else "",
            "table_name": parts[-1],
        }

    snapshot_attrs = ("_connectors", "_metadata", "_capabilities", "_uri_builders", "_context_resolvers")
    snapshots = {attr: getattr(connector_registry, attr).copy() for attr in snapshot_attrs}
    mock_semantic_rag = MagicMock()
    try:
        connector_registry.register_handlers("flexdb", capabilities={"database", "schema"})
        monkeypatch.setattr(
            connector_registry,
            "get_identifier_parser",
            lambda dialect: parse_identifier if dialect == "flexdb" else None,
            raising=False,
        )
        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=MagicMock()),
        ):
            result = GenerationHooks._sync_semantic_to_db(str(yaml_file), agent_config)
    finally:
        for attr, saved in snapshots.items():
            live = getattr(connector_registry, attr)
            live.clear()
            live.update(saved)

    assert result["success"], result.get("error")
    rows = mock_semantic_rag.upsert_batch.call_args.args[0]
    table_row = next(row for row in rows if row["kind"] == "table")
    assert table_row["fq_name"] == "project_a.orders"
    assert table_row["schema_name"] == ""


# ---------------------------------------------------------------------------
# Tests: _sync_semantic_to_db — metric reference normalization
# ---------------------------------------------------------------------------


class TestSyncSemanticToDbMetricReferenceNormalization:
    def test_sync_replaces_rows_for_same_yaml_artifact(self, agent_config, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
data_source:
  name: orders
  sql_table: public.orders
  dimensions:
    - name: status
      type: CATEGORICAL
      expr: status
  measures:
    - name: revenue
      agg: SUM
      expr: amount
---
metric:
  name: total_revenue
  type: measure_proxy
  type_params:
    measure: revenue
""",
            encoding="utf-8",
        )
        db_config = MagicMock()
        db_config.catalog = ""
        db_config.database = "test_db"
        db_config.schema = "public"
        agent_config.current_db_config.return_value = db_config
        agent_config.project_name = "unit-test-project"

        mock_semantic_rag = MagicMock()
        mock_metric_rag = MagicMock()
        mock_profile_rag = MagicMock()

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=mock_metric_rag),
            patch("datus.cli.generation_hooks.TableSemanticProfileRAG", return_value=mock_profile_rag),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(yaml_file),
                agent_config=agent_config,
                original_yaml_path=str(yaml_file),
            )

        assert result["success"], f"Sync failed: {result.get('error')}"
        mock_semantic_rag.delete_artifact_rows.assert_not_called()
        mock_semantic_rag.upsert_batch.assert_called_once()
        mock_semantic_rag.delete_artifact_rows_except.assert_called_once()
        mock_profile_rag.delete_artifact_rows.assert_not_called()
        mock_profile_rag.upsert_batch.assert_called_once()
        mock_profile_rag.delete_artifact_rows_except.assert_called_once()
        mock_metric_rag.delete_artifact_rows.assert_not_called()
        mock_metric_rag.upsert_batch.assert_called_once()
        mock_metric_rag.delete_artifact_rows_except.assert_called_once()

    def test_sync_replacement_restores_snapshots_when_later_store_fails(self, agent_config, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
data_source:
  name: orders
  sql_table: public.orders
  dimensions:
    - name: status
      type: CATEGORICAL
      expr: status
  measures:
    - name: revenue
      agg: SUM
      expr: amount
---
metric:
  name: total_revenue
  type: measure_proxy
  type_params:
    measure: revenue
""",
            encoding="utf-8",
        )
        db_config = MagicMock()
        db_config.catalog = ""
        db_config.database = "test_db"
        db_config.schema = "public"
        agent_config.current_db_config.return_value = db_config
        agent_config.project_name = "unit-test-project"

        mock_semantic_rag = MagicMock()
        mock_semantic_rag.list_artifact_rows.return_value = [{"id": "old-semantic"}]
        mock_metric_rag = MagicMock()
        mock_metric_rag.list_artifact_rows.return_value = [{"id": "old-metric"}]
        mock_metric_rag.upsert_batch.side_effect = RuntimeError("metric write failed")
        mock_profile_rag = MagicMock()
        mock_profile_rag.list_artifact_rows.return_value = [{"id": "old-profile"}]

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=mock_metric_rag),
            patch("datus.cli.generation_hooks.TableSemanticProfileRAG", return_value=mock_profile_rag),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(yaml_file),
                agent_config=agent_config,
                original_yaml_path=str(yaml_file),
            )

        assert result["success"] is False
        assert "metric write failed" in result["error"]
        mock_semantic_rag.delete_artifact_rows.assert_not_called()
        mock_metric_rag.delete_artifact_rows.assert_not_called()
        mock_profile_rag.delete_artifact_rows.assert_not_called()
        mock_semantic_rag.restore_artifact_rows.assert_called_once_with(str(yaml_file), [{"id": "old-semantic"}])
        mock_profile_rag.restore_artifact_rows.assert_called_once_with(str(yaml_file), [{"id": "old-profile"}])
        mock_metric_rag.restore_artifact_rows.assert_called_once_with(str(yaml_file), [{"id": "old-metric"}])

    def test_metric_partial_sync_does_not_replace_whole_yaml_artifact(self, agent_config, tmp_path):
        yaml_file = tmp_path / "orders_metrics.yml"
        yaml_file.write_text(
            """
metric:
  name: total_revenue
  type: measure_proxy
  type_params:
    measure: revenue
""",
            encoding="utf-8",
        )
        db_config = MagicMock()
        db_config.catalog = ""
        db_config.database = "test_db"
        db_config.schema = "public"
        agent_config.current_db_config.return_value = db_config

        mock_semantic_rag = MagicMock()
        mock_metric_rag = MagicMock()

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=mock_metric_rag),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(yaml_file),
                agent_config=agent_config,
                include_semantic_objects=False,
                include_metrics=True,
                original_yaml_path=str(yaml_file),
                replace_metric_artifact=False,
            )

        assert result["success"], f"Sync failed: {result.get('error')}"
        mock_metric_rag.delete_artifact_rows.assert_not_called()
        mock_metric_rag.delete_artifact_rows_except.assert_not_called()
        mock_metric_rag.list_artifact_rows.assert_called_once_with(str(yaml_file))
        mock_metric_rag.upsert_batch.assert_called_once()

    def test_metric_partial_sync_restores_metric_rows_on_later_failure(self, agent_config, tmp_path):
        yaml_file = tmp_path / "orders_metrics.yml"
        yaml_file.write_text(
            """
metric:
  name: total_revenue
  type: measure_proxy
  type_params:
    measure: revenue
""",
            encoding="utf-8",
        )
        db_config = MagicMock()
        db_config.catalog = ""
        db_config.database = "test_db"
        db_config.schema = "public"
        agent_config.current_db_config.return_value = db_config

        mock_semantic_rag = MagicMock()
        mock_metric_rag = MagicMock()
        mock_metric_rag.list_artifact_rows.return_value = [{"id": "old-metric"}]
        mock_metric_rag.create_indices.side_effect = RuntimeError("index failed")

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=mock_metric_rag),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(yaml_file),
                agent_config=agent_config,
                include_semantic_objects=False,
                include_metrics=True,
                original_yaml_path=str(yaml_file),
                replace_metric_artifact=False,
            )

        assert result["success"] is False
        assert "index failed" in result["error"]
        mock_metric_rag.delete_artifact_rows_except.assert_not_called()
        mock_metric_rag.restore_artifact_rows.assert_called_once_with(str(yaml_file), [{"id": "old-metric"}])

    def test_metric_id_includes_subject_path_to_avoid_same_name_collision(self, agent_config, tmp_path):
        yaml_file = tmp_path / "metrics.yml"
        yaml_file.write_text(
            """
metric:
  name: average_gross_order_value
  description: "Commerce AOV"
  type: simple
  locked_metadata:
    tags:
      - "subject_tree: Commerce/Orders/Average_Order_Value"
---
metric:
  name: average_gross_order_value
  description: "Finance AOV"
  type: simple
  locked_metadata:
    tags:
      - "subject_tree: Finance/Orders/Average_Order_Value"
""",
            encoding="utf-8",
        )

        captured_metric = []
        mock_semantic_rag = MagicMock()
        mock_semantic_rag.upsert_batch = MagicMock()
        mock_metric_rag = MagicMock()
        mock_metric_rag.upsert_batch = lambda objects: captured_metric.extend(objects)
        db_config = MagicMock()
        db_config.catalog = ""
        db_config.database = "test_db"
        db_config.schema = ""
        agent_config.current_db_config.return_value = db_config

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=mock_metric_rag),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(yaml_file),
                agent_config=agent_config,
                include_semantic_objects=False,
                include_metrics=True,
            )

        assert result["success"], f"Sync failed: {result.get('error')}"
        assert len(captured_metric) == 2
        assert captured_metric[0]["id"] == "metric:average_gross_order_value"
        assert captured_metric[1]["id"] == "metric:average_gross_order_value"
        assert captured_metric[0]["subject_path"] == ["Commerce", "Orders", "Average_Order_Value"]
        assert captured_metric[1]["subject_path"] == ["Finance", "Orders", "Average_Order_Value"]

    def test_measure_proxy_nested_measure_is_stored_as_string(self, agent_config, tmp_path):
        yaml_file = tmp_path / "metrics.yml"
        yaml_file.write_text(
            """
data_source:
  name: orders
  sql_table: public.orders
  measures:
    - name: order_count
      agg: COUNT
      expr: "1"
  dimensions:
    - name: status
      type: CATEGORICAL
      expr: status
---
metric:
  name: completed_order_count
  description: "Completed orders"
  type: measure_proxy
  type_params:
    measure:
      name: order_count
      constraint: "status = 'completed'"
""",
            encoding="utf-8",
        )

        captured_metric = []
        mock_semantic_rag = MagicMock()
        mock_semantic_rag.upsert_batch = MagicMock()
        mock_metric_rag = MagicMock()
        mock_metric_rag.upsert_batch = lambda objects: captured_metric.extend(objects)
        db_config = MagicMock()
        db_config.catalog = ""
        db_config.database = "test_db"
        db_config.schema = ""
        agent_config.current_db_config.return_value = db_config

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG", return_value=mock_semantic_rag),
            patch("datus.cli.generation_hooks.MetricRAG", return_value=mock_metric_rag),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(yaml_file),
                agent_config=agent_config,
                include_semantic_objects=False,
                include_metrics=True,
            )

        assert result["success"], f"Sync failed: {result.get('error')}"
        assert len(captured_metric) == 1
        assert captured_metric[0]["base_measures"] == ["order_count"]
        assert captured_metric[0]["measure_expr"] == "order_count WHERE status = 'completed'"


# ---------------------------------------------------------------------------
# Tests: _sync_semantic_to_db actionable error for empty metric-only sync
# ---------------------------------------------------------------------------


class TestSyncSemanticToDbMetricOnlyDiagnostic:
    """Metric-only syncs (e.g. publish_metrics) emit a tailored error
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

    def test_metric_only_sync_with_unnamed_metric_blocks_returns_actionable_error(self, agent_config, tmp_path):
        unnamed_metric = tmp_path / "unnamed_metrics.yml"
        unnamed_metric.write_text("metric:\n  description: missing name\n  type: measure_proxy\n")

        with (
            patch("datus.cli.generation_hooks.SemanticModelRAG"),
            patch("datus.cli.generation_hooks.MetricRAG"),
        ):
            result = GenerationHooks._sync_semantic_to_db(
                file_path=str(unnamed_metric),
                agent_config=agent_config,
                include_semantic_objects=False,
                include_metrics=True,
            )

        assert result["success"] is False
        assert "non-empty `metric.name`" in result["error"]
        assert str(unnamed_metric) in result["error"]

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
        cfg.current_datasource = "ns"
        cfg.path_manager = MagicMock(spec=[])  # no attrs → getattr returns None
        h = GenerationHooks(broker=broker, agent_config=cfg)
        assert h._get_base_dir("semantic") is None

    def test_returns_none_when_resolver_raises(self, broker):
        """Exceptions raised by the resolver are caught and return None."""
        cfg = MagicMock()
        cfg.current_datasource = "ns"
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
# Tests: normalize_kb_relative_path (pure function)
# ---------------------------------------------------------------------------


class TestNormalizeKbRelativePath:
    def test_prepends_when_prefix_missing(self):
        assert normalize_kb_relative_path("orders.yaml", "semantic") == "semantic_models/orders.yaml"

    def test_prepends_for_sql_summary(self):
        assert normalize_kb_relative_path("q_001.yaml", "sql_summary") == "sql_summaries/q_001.yaml"

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
