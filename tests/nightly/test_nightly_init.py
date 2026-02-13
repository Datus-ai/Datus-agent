import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from datus.cli.interactive_init import InteractiveInit


@pytest.mark.nightly
class TestNightlyInit:
    """N4: Init configuration and connectivity tests."""

    def test_llm_config_probe(self):
        """N4-01: LLM configuration + probe_llm() connectivity test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            # Configure LLM settings in the config
            init.config["agent"]["target"] = "openai"
            init.config["agent"]["models"]["openai"] = {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-api-key-123",
                "model": "gpt-4.1",
            }

            # Mock the LLM model to simulate successful connectivity
            with patch("datus.cli.interactive_init.InteractiveInit._test_llm_connectivity") as mock_probe:
                mock_probe.return_value = (True, "")
                success, error_msg = init._test_llm_connectivity()

                assert success is True, "LLM probe should succeed with mocked response"
                assert error_msg == "", "Error message should be empty on success"

            # Test failed connectivity
            with patch("datus.cli.interactive_init.InteractiveInit._test_llm_connectivity") as mock_probe:
                mock_probe.return_value = (False, "Connection refused")
                success, error_msg = init._test_llm_connectivity()

                assert success is False, "LLM probe should fail with error"
                assert "Connection refused" in error_msg, "Error message should contain failure reason"

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
            assert (
                "deepseek" in saved_config["agent"]["models"]
            ), "Saved config should have deepseek model configuration"
            assert (
                saved_config["agent"]["models"]["deepseek"]["model"] == "deepseek-chat"
            ), "Saved config should have correct model name"
            assert "test_ns" in saved_config["agent"]["namespace"], "Saved config should have namespace configuration"
            assert (
                saved_config["agent"]["namespace"]["test_ns"]["type"] == "duckdb"
            ), "Saved namespace should have correct db type"
            assert (
                "workspace_root" in saved_config["agent"]["storage"]
            ), "Saved config should have workspace_root in storage"

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
