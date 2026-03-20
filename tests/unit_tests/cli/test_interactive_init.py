import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from rich.console import Console

from datus.cli.interactive_init import (
    InteractiveInit,
    ReferenceSqlStreamHandler,
    _format_reference_sql_line,
    do_init_sql_and_log_result,
    overwrite_sql_and_log_result,
    parse_subject_tree,
)


class TestInit:
    """N4: Init configuration and connectivity tests."""

    def test_llm_config_probe_success(self):
        """N4-01a: LLM connectivity probe succeeds when model returns a response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            init.config["agent"]["target"] = "openai"
            init.config["agent"]["models"]["openai"] = {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-api-key-123",
                "model": "gpt-4.1",
            }

            # Mock the underlying LLM model class so _test_llm_connectivity
            # exercises its real logic (config parsing, model creation, generate call)
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = "Hello!"

            mock_module = MagicMock()
            mock_module.OpenAIModel.return_value = mock_model_instance
            with patch.dict("sys.modules", {"datus.models.openai_model": mock_module}):
                success, error_msg = init._test_llm_connectivity()

            assert success is True, f"LLM probe should succeed, got error: {error_msg}"
            assert error_msg == "", "Error message should be empty on success"

    def test_llm_config_probe_failure(self):
        """N4-01b: LLM connectivity probe fails when model raises an exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            init.config["agent"]["target"] = "openai"
            init.config["agent"]["models"]["openai"] = {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "bad-key",
                "model": "gpt-4.1",
            }

            mock_model_instance = MagicMock()
            mock_model_instance.generate.side_effect = ConnectionError("Connection refused")

            mock_module = MagicMock()
            mock_module.OpenAIModel.return_value = mock_model_instance
            with patch.dict("sys.modules", {"datus.models.openai_model": mock_module}):
                success, error_msg = init._test_llm_connectivity()

            assert success is False, "LLM probe should fail with connection error"
            assert "Connection refused" in error_msg, f"Error should mention reason, got: {error_msg}"

    def test_llm_config_unsupported_type(self):
        """N4-01c: Unsupported model type returns failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            init.config["agent"]["target"] = "unsupported_provider"
            init.config["agent"]["models"]["unsupported_provider"] = {
                "type": "unsupported_provider",
                "base_url": "https://example.com",
                "api_key": "key",
                "model": "model",
            }

            success, error_msg = init._test_llm_connectivity()

            assert success is False, "Should fail for unsupported model type"
            assert "Unsupported" in error_msg, f"Error should mention unsupported type, got: {error_msg}"

    def test_config_file_generation(self):
        """N4-03: Configuration file generation and validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            # Set up config directory
            conf_dir = Path(tmpdir) / ".datus" / "conf"
            conf_dir.mkdir(parents=True, exist_ok=True)
            init.conf_dir = conf_dir

            # Configure all sections
            init.config["agent"]["target"] = "deepseek"
            init.config["agent"]["models"]["deepseek"] = {
                "type": "deepseek",
                "base_url": "https://api.deepseek.com",
                "api_key": "test-key-456",
                "model": "deepseek-chat",
            }
            init.config["agent"]["namespace"]["test_ns"] = {
                "type": "duckdb",
                "name": "test_ns",
                "uri": "duckdb:///test.db",
            }
            init.config["agent"]["storage"]["workspace_root"] = str(Path(tmpdir) / "workspace")
            init.namespace_name = "test_ns"

            # Save configuration
            result = init._save_configuration()
            assert result is True, "Configuration save should succeed"

            # Verify file exists
            config_path = conf_dir / "agent.yml"
            assert config_path.exists(), "agent.yml should be created"

            # Load and validate the saved config
            with open(config_path, "r") as f:
                saved_config = yaml.safe_load(f)

            assert saved_config["agent"]["target"] == "deepseek", "Saved config should have correct target"
            assert "deepseek" in saved_config["agent"]["models"], (
                "Saved config should have deepseek model configuration"
            )
            assert saved_config["agent"]["models"]["deepseek"]["model"] == "deepseek-chat", (
                "Saved config should have correct model name"
            )
            assert "test_ns" in saved_config["agent"]["namespace"], "Saved config should have namespace configuration"
            assert saved_config["agent"]["namespace"]["test_ns"]["type"] == "duckdb", (
                "Saved namespace should have correct db type"
            )
            assert "workspace_root" in saved_config["agent"]["storage"], (
                "Saved config should have workspace_root in storage"
            )

    def test_optional_component_init(self):
        """N4-04: Optional component initialization (metadata and reference SQL)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)
            init.namespace_name = "test_ns"
            init.workspace_path = str(Path(tmpdir) / "workspace")

            # Create workspace directory
            Path(init.workspace_path).mkdir(parents=True, exist_ok=True)

            # Mock Confirm.ask to decline all optional setup
            with patch("datus.cli.interactive_init.Confirm.ask") as mock_confirm:
                mock_confirm.return_value = False
                init._optional_setup(str(Path(tmpdir) / "agent.yml"))

                # Verify Confirm.ask was called for metadata and reference SQL
                assert mock_confirm.call_count >= 1, "Should prompt for at least one optional component"

            # Test with metadata accepted but reference SQL declined
            with (
                patch("datus.cli.interactive_init.Confirm.ask") as mock_confirm,
                patch("datus.cli.interactive_init.init_metadata_and_log_result") as mock_metadata,
            ):
                mock_confirm.side_effect = [True, False]
                init._optional_setup(str(Path(tmpdir) / "agent.yml"))

                mock_metadata.assert_called_once_with(
                    "test_ns",
                    str(Path(tmpdir) / "agent.yml"),
                    init.console,
                )

            # Test with both accepted
            with (
                patch("datus.cli.interactive_init.Confirm.ask") as mock_confirm,
                patch("datus.cli.interactive_init.Prompt.ask") as mock_prompt,
                patch("datus.cli.interactive_init.init_metadata_and_log_result") as mock_metadata,
                patch("datus.cli.interactive_init.overwrite_sql_and_log_result") as mock_sql,
            ):
                default_sql_dir = str(Path(init.workspace_path) / "reference_sql")
                mock_confirm.side_effect = [True, True]
                mock_prompt.return_value = default_sql_dir

                init._optional_setup(str(Path(tmpdir) / "agent.yml"))

                assert mock_metadata.call_count == 1, "Metadata init should be called when accepted"
                assert mock_sql.call_count == 1, "SQL init should be called when accepted"


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

    def test_kimi_k25_sets_temperature_and_top_p_in_config(self):
        """kimi-k2.5 requires temperature=1.0 and top_p=0.95; verify they are stored in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = "Hello!"
            mock_module = MagicMock()
            mock_module.KimiModel.return_value = mock_model_instance

            with (
                patch(
                    "datus.cli.interactive_init.Prompt.ask",
                    side_effect=["kimi", "https://api.moonshot.cn/v1", "kimi-k2.5"],
                ),
                patch("datus.cli.interactive_init.getpass", return_value="test-key"),
                patch.dict("sys.modules", {"datus.models.kimi_model": mock_module}),
            ):
                result = init._configure_llm()

            assert result is True
            kimi_config = init.config["agent"]["models"]["kimi"]
            assert kimi_config["temperature"] == 1.0, "kimi-k2.5 should have temperature=1.0"
            assert kimi_config["top_p"] == 0.95, "kimi-k2.5 should have top_p=0.95"

    def test_kimi_k25_passes_params_to_model_config(self):
        """Verify _test_llm_connectivity passes temperature/top_p to ModelConfig for kimi-k2.5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            init.config["agent"]["target"] = "kimi"
            init.config["agent"]["models"]["kimi"] = {
                "type": "kimi",
                "base_url": "https://api.moonshot.cn/v1",
                "api_key": "test-key",
                "model": "kimi-k2.5",
                "temperature": 1.0,
                "top_p": 0.95,
            }

            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = "Hello!"
            mock_module = MagicMock()
            mock_module.KimiModel.return_value = mock_model_instance

            with patch.dict("sys.modules", {"datus.models.kimi_model": mock_module}):
                success, error_msg = init._test_llm_connectivity()

            assert success is True
            # Verify ModelConfig was created with correct temperature and top_p
            model_config = mock_module.KimiModel.call_args.kwargs["model_config"]
            assert model_config.temperature == 1.0, "ModelConfig should have temperature=1.0"
            assert model_config.top_p == 0.95, "ModelConfig should have top_p=0.95"

    def test_non_kimi_model_has_no_param_overrides(self):
        """Non-kimi models should not get temperature/top_p overrides in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = "Hello!"
            mock_module = MagicMock()
            mock_module.OpenAIModel.return_value = mock_model_instance

            with (
                patch(
                    "datus.cli.interactive_init.Prompt.ask",
                    side_effect=["openai", "https://api.openai.com/v1", "gpt-4.1"],
                ),
                patch("datus.cli.interactive_init.getpass", return_value="test-key"),
                patch.dict("sys.modules", {"datus.models.openai_model": mock_module}),
            ):
                result = init._configure_llm()

            assert result is True
            openai_config = init.config["agent"]["models"]["openai"]
            assert "temperature" not in openai_config, "OpenAI models should not have temperature override"
            assert "top_p" not in openai_config, "OpenAI models should not have top_p override"

    def test_qwen3_coder_plus_sets_temperature_and_top_p(self):
        """qwen3-coder-plus requires temperature=1.0 and top_p=0.95; verify they are stored in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = "Hello!"
            mock_module = MagicMock()
            mock_module.OpenAIModel.return_value = mock_model_instance

            with (
                patch(
                    "datus.cli.interactive_init.Prompt.ask",
                    side_effect=["qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-coder-plus"],
                ),
                patch("datus.cli.interactive_init.getpass", return_value="test-key"),
                patch.dict("sys.modules", {"datus.models.openai_model": mock_module}),
            ):
                result = init._configure_llm()

            assert result is True
            qwen_config = init.config["agent"]["models"]["qwen"]
            assert qwen_config["temperature"] == 1.0, "qwen3-coder-plus should have temperature=1.0"
            assert qwen_config["top_p"] == 0.95, "qwen3-coder-plus should have top_p=0.95"


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

        with (
            patch(
                "datus.configuration.agent_config_loader.load_agent_config", side_effect=RuntimeError("config error")
            ),
            patch("datus.cli.interactive_init.print_rich_exception") as mock_print_exc,
        ):
            overwrite_sql_and_log_result(
                namespace_name="test_ns",
                sql_dir="/some/dir",
                config_path="/path/to/agent.yml",
                console=console,
            )

        # Exception should be caught and reported via print_rich_exception
        mock_print_exc.assert_called_once()
