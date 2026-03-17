# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Additional unit tests for datus/cli/interactive_init.py.

Covers branches not exercised by test_interactive_init.py:
- InteractiveInit._configure_workspace: success and failure
- InteractiveInit._display_summary: smoke test
- InteractiveInit._display_completion: smoke test
- InteractiveInit._configure_llm: empty api_key returns False
- parse_subject_tree: None, empty string, comma-separated
- _format_reference_sql_line: short and long strings
- ReferenceSqlStreamHandler.handle_event: each BatchStage
- do_init_sql_and_log_result: nonexistent dir, dir with no sql files
- overwrite_sql_and_log_result: exception handling
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from datus.cli.interactive_init import (
    InteractiveInit,
    ReferenceSqlStreamHandler,
    _format_reference_sql_line,
    do_init_sql_and_log_result,
    overwrite_sql_and_log_result,
    parse_subject_tree,
)


def _make_console():
    return Console(file=io.StringIO(), no_color=True)


# ---------------------------------------------------------------------------
# parse_subject_tree
# ---------------------------------------------------------------------------


class TestParseSubjectTree:
    def test_none_returns_none(self):
        assert parse_subject_tree(None) is None

    def test_empty_string_returns_none(self):
        assert parse_subject_tree("") is None

    def test_single_item(self):
        result = parse_subject_tree("Finance")
        assert result == ["Finance"]

    def test_comma_separated(self):
        result = parse_subject_tree("Finance, Revenue, Q1")
        assert result == ["Finance", "Revenue", "Q1"]

    def test_strips_whitespace(self):
        result = parse_subject_tree("  a , b  ,  c ")
        assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _format_reference_sql_line
# ---------------------------------------------------------------------------


class TestFormatReferenceSqlLine:
    def test_short_sql_returned_as_is(self):
        sql = "SELECT 1"
        result = _format_reference_sql_line(sql)
        assert result == "SELECT 1"

    def test_long_sql_truncated(self):
        sql = "SELECT " + "a" * 100
        result = _format_reference_sql_line(sql, max_length=20)
        assert len(result) <= 23  # 20 chars + "..."
        assert result.endswith("...")

    def test_empty_string_returns_unknown(self):
        result = _format_reference_sql_line("")
        assert result == "unknown_sql"

    def test_multiline_condensed(self):
        sql = "SELECT\n  a,\n  b\nFROM t"
        result = _format_reference_sql_line(sql)
        assert "\n" not in result
        assert "SELECT" in result


# ---------------------------------------------------------------------------
# ReferenceSqlStreamHandler
# ---------------------------------------------------------------------------


class TestReferenceSqlStreamHandler:
    def _make_handler(self):
        output_mgr = MagicMock()
        handler = ReferenceSqlStreamHandler(output_mgr)
        return handler, output_mgr

    def _make_event(self, stage, **kwargs):
        pass

        event = MagicMock()
        event.stage = stage
        event.payload = kwargs.get("payload", {})
        event.total_items = kwargs.get("total_items", 0)
        event.group_id = kwargs.get("group_id", None)
        event.completed_items = kwargs.get("completed_items", 0)
        event.failed_items = kwargs.get("failed_items", 0)
        event.error = kwargs.get("error", None)
        return event

    def test_task_started_does_nothing(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.TASK_STARTED)
        handler.handle_event(event)
        output_mgr.start.assert_not_called()

    def test_task_validated_with_invalid_items(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.TASK_VALIDATED, payload={"valid_items": 5, "invalid_items": 2})
        handler.handle_event(event)
        output_mgr.add_message.assert_called_once()

    def test_task_validated_all_valid(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.TASK_VALIDATED, payload={"valid_items": 10, "invalid_items": 0})
        handler.handle_event(event)
        output_mgr.add_message.assert_called_once()

    def test_task_processing_starts_progress(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.TASK_PROCESSING, total_items=5)
        handler.handle_event(event)
        output_mgr.start.assert_called_once_with(total_items=5, description="Initializing reference SQL")

    def test_group_started(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.GROUP_STARTED, payload={"filepath": "/path/to/file.sql"}, total_items=3)
        handler.handle_event(event)
        output_mgr.start_task.assert_called_once()

    def test_group_completed(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.GROUP_COMPLETED)
        handler.handle_event(event)
        output_mgr.complete_task.assert_called_once_with(success=True)

    def test_item_started(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.ITEM_STARTED, payload={"filepath": "/f.sql", "sql": "SELECT 1"})
        handler.handle_event(event)
        output_mgr.add_message.assert_called_once()

    def test_item_completed_advances_progress(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.ITEM_COMPLETED)
        handler.handle_event(event)
        output_mgr.update_progress.assert_called_once_with(advance=1.0)

    def test_item_failed_logs_error(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.ITEM_FAILED, error="Processing failed")
        handler.handle_event(event)
        output_mgr.error.assert_called_once()
        output_mgr.update_progress.assert_called_once_with(advance=1.0)

    def test_task_completed_all_success(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.TASK_COMPLETED, completed_items=10, failed_items=0)
        handler.handle_event(event)
        output_mgr.success.assert_called_once()

    def test_task_completed_with_failures(self):
        from datus.schemas.batch_events import BatchStage

        handler, output_mgr = self._make_handler()
        event = self._make_event(BatchStage.TASK_COMPLETED, completed_items=8, failed_items=2)
        handler.handle_event(event)
        output_mgr.warning.assert_called_once()


# ---------------------------------------------------------------------------
# InteractiveInit._configure_workspace
# ---------------------------------------------------------------------------


class TestConfigureWorkspace:
    def test_success_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=Path(tmpdir))
            workspace = str(Path(tmpdir) / "workspace")

            with patch("datus.cli.interactive_init.Prompt.ask", return_value=workspace):
                result = init._configure_workspace()

            assert result is True
            assert Path(workspace).exists()
            assert init.workspace_path == workspace

    def test_failure_on_permission_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=Path(tmpdir))
            workspace = str(Path(tmpdir) / "workspace")

            with patch("datus.cli.interactive_init.Prompt.ask", return_value=workspace):
                with patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")):
                    with patch("datus.cli.interactive_init.print_rich_exception"):
                        result = init._configure_workspace()

            assert result is False


# ---------------------------------------------------------------------------
# InteractiveInit._display_summary and _display_completion
# ---------------------------------------------------------------------------


class TestDisplayMethods:
    def test_display_summary_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)
            init.config["agent"]["target"] = "openai"
            init.config["agent"]["models"]["openai"] = {
                "type": "openai",
                "model": "gpt-4.1",
                "api_key": "key",
                "base_url": "https://api.openai.com/v1",
            }
            init.namespace_name = "test_ns"
            init.workspace_path = "/tmp/workspace"
            # Should not raise
            init._display_summary()

    def test_display_completion_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)
            init.namespace_name = "test_ns"
            # Should not raise
            init._display_completion()


# ---------------------------------------------------------------------------
# InteractiveInit._configure_llm: empty api_key
# ---------------------------------------------------------------------------


class TestConfigureLLM:
    def test_empty_api_key_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            with patch("datus.cli.interactive_init.Prompt.ask", return_value="openai"):
                with patch("datus.cli.interactive_init.getpass", return_value=""):
                    result = init._configure_llm()

            assert result is False


# ---------------------------------------------------------------------------
# do_init_sql_and_log_result: edge cases
# ---------------------------------------------------------------------------


class TestDoInitSqlAndLogResult:
    def test_nonexistent_dir_prints_error(self):
        console = _make_console()
        mock_config = MagicMock()

        do_init_sql_and_log_result(mock_config, "/nonexistent/path/12345", None, console)

        output = console.file.getvalue()
        assert "No sql files found" in output or "sql files" in output.lower()

    def test_empty_sql_dir_prints_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = _make_console()
            mock_config = MagicMock()

            do_init_sql_and_log_result(mock_config, tmpdir, None, console)

            output = console.file.getvalue()
            assert "No sql files found" in output or "sql files" in output.lower()

    def test_non_sql_file_extension_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-sql file
            f = Path(tmpdir) / "data.csv"
            f.write_text("a,b,c")

            console = _make_console()
            mock_config = MagicMock()

            # Pass the file directly (not a .sql file)
            do_init_sql_and_log_result(mock_config, str(f), None, console)

            output = console.file.getvalue()
            # Should print error about non-sql extension
            assert ".sql" in output or "sql" in output.lower()


# ---------------------------------------------------------------------------
# overwrite_sql_and_log_result: exception propagation
# ---------------------------------------------------------------------------


class TestOverwriteSqlAndLogResult:
    def test_exception_is_caught_and_printed(self):
        console = _make_console()

        with patch(
            "datus.configuration.agent_config_loader.load_agent_config", side_effect=RuntimeError("config error")
        ), patch("datus.cli.interactive_init.print_rich_exception") as mock_print_exc:
            overwrite_sql_and_log_result(
                namespace_name="test_ns",
                sql_dir="/some/dir",
                config_path="/path/to/agent.yml",
                console=console,
            )

        # Exception should be caught and reported via print_rich_exception
        mock_print_exc.assert_called_once()
