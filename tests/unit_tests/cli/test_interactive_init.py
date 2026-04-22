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
        """N4-01a: LLM connectivity probe succeeds when the staged probe returns a response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            init._pending_probe = {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-api-key-123",
                "model": "gpt-4.1",
                "auth_type": "api_key",
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

            init._pending_probe = {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "bad-key",
                "model": "gpt-4.1",
                "auth_type": "api_key",
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

            init._pending_probe = {
                "type": "unsupported_provider",
                "base_url": "https://example.com",
                "api_key": "key",
                "model": "model",
                "auth_type": "api_key",
            }

            success, error_msg = init._test_llm_connectivity()

            assert success is False, "Should fail for unsupported model type"
            assert "Unsupported" in error_msg, f"Error should mention unsupported type, got: {error_msg}"

    def test_llm_config_probe_without_pending_probe_fails(self):
        """Calling ``_test_llm_connectivity`` without a staged probe returns a clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)
            success, error_msg = init._test_llm_connectivity()
            assert success is False
            assert "pending" in error_msg.lower()

    def test_config_file_generation(self, monkeypatch):
        """N4-03: agent.yml + project-level .datus/config.yml round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ``save_project_override`` writes relative to CWD, so pin it to
            # the tmpdir for the duration of the test.
            monkeypatch.chdir(tmpdir)
            init = InteractiveInit(user_home=tmpdir)

            conf_dir = Path(tmpdir) / ".datus" / "conf"
            conf_dir.mkdir(parents=True, exist_ok=True)
            init.conf_dir = conf_dir

            init.config["agent"]["providers"]["deepseek"] = {
                "api_key": "test-key-456",
                "base_url": "https://api.deepseek.com",
                "auth_type": "api_key",
            }
            init.config["agent"]["services"]["datasources"]["test_ns"] = {
                "type": "duckdb",
                "name": "test_ns",
                "uri": "duckdb:///test.db",
            }
            init.config["agent"]["storage"]["workspace_root"] = str(Path(tmpdir) / "workspace")
            init.datasource_name = "test_ns"
            from datus.configuration.project_config import ProjectTarget

            init._pending_target = ProjectTarget(provider="deepseek", model="deepseek-chat")

            result = init._save_configuration()
            assert result is True, "Configuration save should succeed"

            config_path = conf_dir / "agent.yml"
            assert config_path.exists(), "agent.yml should be created"

            with open(config_path, "r") as f:
                saved_config = yaml.safe_load(f)

            assert "deepseek" in saved_config["agent"]["providers"]
            assert saved_config["agent"]["providers"]["deepseek"]["api_key"] == "test-key-456"
            assert "target" not in saved_config["agent"], "Global target should no longer be written"
            assert "models" not in saved_config["agent"], "Init wizard must not write agent.models"
            assert "test_ns" in saved_config["agent"]["services"]["datasources"]

            project_cfg_path = Path(tmpdir) / ".datus" / "config.yml"
            assert project_cfg_path.exists(), "Project-level .datus/config.yml should be created"
            with open(project_cfg_path, "r") as f:
                project_cfg = yaml.safe_load(f)
            assert project_cfg["target"] == {"provider": "deepseek", "model": "deepseek-chat"}

    def test_optional_component_init(self):
        """N4-04: Optional component initialization (metadata and reference SQL)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)
            init.datasource_name = "test_ns"
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
                    with patch("datus.cli.interactive_init.print_rich_exception") as mock_print_exc:
                        result = init._configure_workspace()

            assert result is False
            mock_print_exc.assert_called_once()


# ---------------------------------------------------------------------------
# InteractiveInit._display_summary and _display_completion
# ---------------------------------------------------------------------------


class TestDisplayMethods:
    def test_display_summary_smoke(self):
        from datus.configuration.project_config import ProjectTarget

        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)
            init.console = Console(file=io.StringIO(), no_color=True)
            init._pending_target = ProjectTarget(provider="openai", model="gpt-4.1")
            init.config["agent"]["providers"]["openai"] = {
                "api_key": "key",
                "base_url": "https://api.openai.com/v1",
                "auth_type": "api_key",
            }
            init.datasource_name = "test_ns"
            init.workspace_path = "/tmp/workspace"
            init._display_summary()
            output = init.console.file.getvalue()
            assert "openai" in output
            assert "gpt-4.1" in output

    def test_display_completion_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)
            init.console = Console(file=io.StringIO(), no_color=True)
            init.datasource_name = "test_ns"
            init._display_completion()
            assert "datus" in init.console.file.getvalue().lower()


# ---------------------------------------------------------------------------
# InteractiveInit._configure_llm: empty api_key
# ---------------------------------------------------------------------------


class TestConfigureLLM:
    def test_empty_api_key_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            with (
                patch("datus.cli.interactive_init.select_choice", return_value="openai"),
                patch("datus.cli.interactive_init.getpass", return_value=""),
            ):
                result = init._configure_llm()

            assert result is False

    def test_api_key_provider_writes_provider_level_credentials(self):
        """``_configure_llm`` persists the credential under ``agent.providers`` and stages the target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = "Hello!"
            mock_module = MagicMock()
            mock_module.KimiModel.return_value = mock_model_instance

            with (
                patch("datus.cli.interactive_init.select_choice", side_effect=["kimi", "kimi-k2.5"]),
                patch("datus.cli.interactive_init.Prompt.ask", return_value="https://api.moonshot.cn/v1"),
                patch("datus.cli.interactive_init.getpass", return_value="test-key"),
                patch.dict("sys.modules", {"datus.models.kimi_model": mock_module}),
            ):
                result = init._configure_llm()

            assert result is True
            providers = init.config["agent"]["providers"]
            assert providers["kimi"]["api_key"] == "test-key"
            assert providers["kimi"]["base_url"] == "https://api.moonshot.cn/v1"
            assert providers["kimi"]["auth_type"] == "api_key"
            assert init._pending_target.provider == "kimi"
            assert init._pending_target.model == "kimi-k2.5"
            # The init wizard never writes legacy agent.models entries.
            assert "models" not in init.config["agent"]

    def test_api_key_provider_probe_failure_drops_pending_target(self):
        """Probe failure must leave no staged target so the caller can retry cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            mock_model_instance = MagicMock()
            mock_model_instance.generate.side_effect = ConnectionError("blocked")
            mock_module = MagicMock()
            mock_module.OpenAIModel.return_value = mock_model_instance

            with (
                patch("datus.cli.interactive_init.select_choice", side_effect=["openai", "gpt-4.1"]),
                patch("datus.cli.interactive_init.Prompt.ask", return_value="https://api.openai.com/v1"),
                patch("datus.cli.interactive_init.getpass", return_value="test-key"),
                patch.dict("sys.modules", {"datus.models.openai_model": mock_module}),
            ):
                result = init._configure_llm()

            assert result is False
            assert init._pending_target is None
            assert "openai" not in init.config["agent"]["providers"]


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
                datasource_name="test_ns",
                sql_dir="/some/dir",
                config_path="/path/to/agent.yml",
                console=console,
            )

        # Exception should be caught and reported via print_rich_exception
        mock_print_exc.assert_called_once()


class TestConfigureLLMSubscriptionOAuth:
    """Subscription / OAuth providers delegate to ``provider_auth_flows``."""

    def test_subscription_success_writes_provider_section(self):
        """A successful helper result populates ``agent.providers`` + pending target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            provider_info = {
                "type": "claude",
                "base_url": "https://api.anthropic.com",
                "default_model": "claude-sonnet-4-6",
                "models": ["claude-sonnet-4-6"],
                "auth_type": "subscription",
            }

            helper_return = {
                "model": "claude-sonnet-4-6",
                "api_key": "sk-ant-oat01-test",
                "auth_type": "subscription",
                "base_url": "https://api.anthropic.com",
                "type": "claude",
            }

            with (
                patch("datus.cli.interactive_init.select_choice", return_value="claude_subscription"),
                patch("datus.cli.interactive_init.configure_claude_subscription", return_value=helper_return),
            ):
                # The catalog is loaded inline; pre-seed it so the provider
                # under test exists in the choice dict the test feeds.
                with patch.object(
                    init, "_load_provider_catalog", return_value={"providers": {"claude_subscription": provider_info}}
                ):
                    result = init._configure_llm()

            assert result is True
            assert init.config["agent"]["providers"]["claude_subscription"] == {
                "api_key": "sk-ant-oat01-test",
                "base_url": "https://api.anthropic.com",
                "auth_type": "subscription",
            }
            assert init._pending_target.provider == "claude_subscription"
            assert init._pending_target.model == "claude-sonnet-4-6"

    def test_oauth_helper_failure_returns_false(self):
        """If the OAuth helper returns None (login failed), the flow aborts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            provider_info = {
                "type": "codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "default_model": "gpt-5.3-codex",
                "models": ["gpt-5.3-codex"],
                "auth_type": "oauth",
            }

            with (
                patch("datus.cli.interactive_init.select_choice", return_value="codex"),
                patch("datus.cli.interactive_init.configure_codex_oauth", return_value=None),
                patch.object(init, "_load_provider_catalog", return_value={"providers": {"codex": provider_info}}),
            ):
                result = init._configure_llm()

            assert result is False
            assert init._pending_target is None
            assert "codex" not in init.config["agent"]["providers"]
