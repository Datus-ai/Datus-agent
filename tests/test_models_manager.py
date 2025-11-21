#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Test cases for ModelsManager.

This module tests the models management functionality including
list, add, delete, and set-target operations.
"""

from unittest.mock import patch

import pytest

from datus.cli.models_manager import ModelsManager
from datus.configuration.agent_config import ModelConfig


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file for testing."""
    config_file = tmp_path / "agent.yml"
    config_content = """agent:
  target: test_model
  models:
    test_model:
      type: openai
      base_url: https://api.openai.com/v1
      api_key: sk-test-key
      model: gpt-4o-mini
      save_llm_trace: false
      enable_thinking: false
    another_model:
      type: claude
      base_url: https://api.anthropic.com
      api_key: sk-ant-test-key
      model: claude-haiku-4-5
      save_llm_trace: false
      enable_thinking: false
"""
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def models_manager(temp_config_file):
    """Create a ModelsManager instance with test configuration."""
    return ModelsManager(temp_config_file)


@pytest.fixture
def mock_console():
    """Mock the console to capture output."""
    with patch("datus.cli.models_manager.console") as mock_console:
        yield mock_console


@pytest.fixture
def mock_prompt():
    """Mock the rich prompt for user input."""
    with patch("datus.cli.models_manager.Prompt.ask") as mock_ask, patch(
        "datus.cli.models_manager.getpass"
    ) as mock_getpass, patch("datus.cli.models_manager.Confirm.ask") as mock_confirm:
        # Default mocks
        mock_ask.return_value = "new_model"
        mock_getpass.return_value = "test_api_key"
        mock_confirm.return_value = False

        yield mock_ask, mock_getpass, mock_confirm


class TestModelsManagerInit:
    """Test cases for ModelsManager initialization."""

    def test_init_success(self, models_manager):
        """Test successful initialization with valid config."""
        assert models_manager.config_loaded is True
        assert len(models_manager.models) == 2
        assert models_manager.target == "test_model"
        assert "test_model" in models_manager.models
        assert "another_model" in models_manager.models

    def test_init_file_not_found(self):
        """Test initialization with non-existent config file."""
        # ConfigurationManager will use default path if file not found
        # So we need to test with a truly invalid path
        with patch("datus.cli.models_manager.configuration_manager") as mock_config_manager:
            from datus.utils.exceptions import DatusException, ErrorCode

            mock_config_manager.side_effect = DatusException(ErrorCode.COMMON_FILE_NOT_FOUND)

            manager = ModelsManager("/non/existent/path.yml")
            assert manager.config_loaded is False

    def test_init_models_loaded_correctly(self, models_manager):
        """Test that models are loaded with correct attributes."""
        test_model = models_manager.models["test_model"]

        assert isinstance(test_model, ModelConfig)
        assert test_model.type == "openai"
        assert test_model.base_url == "https://api.openai.com/v1"
        assert test_model.model == "gpt-4o-mini"
        assert test_model.save_llm_trace is False
        assert test_model.enable_thinking is False


class TestModelsManagerList:
    """Test cases for ModelsManager.list method."""

    def test_list_with_models(self, models_manager, capsys):
        """Test listing models when some are configured."""
        result = models_manager.list()

        assert result == 0
        captured = capsys.readouterr()
        assert "Configured models:" in captured.out
        assert "Target model:" in captured.out
        assert "test_model" in captured.out
        assert "another_model" in captured.out
        assert "Type: openai" in captured.out
        assert "Type: claude" in captured.out

    def test_list_no_models(self, models_manager, capsys):
        """Test listing models when none are configured."""
        models_manager.models = {}

        result = models_manager.list()

        assert result == 0
        captured = capsys.readouterr()
        assert "No models configured." in captured.out

    def test_list_api_key_hidden(self, models_manager, capsys):
        """Test that API keys are hidden in list output."""
        models_manager.list()

        captured = capsys.readouterr()
        assert "sk-test-key" not in captured.out
        assert "sk-ant-test-key" not in captured.out
        assert "********" in captured.out

    def test_list_shows_active_model(self, models_manager, capsys):
        """Test that active/target model is marked."""
        models_manager.list()

        captured = capsys.readouterr()
        # Rich formatting adds ANSI codes, so check for "active" text
        assert "active" in captured.out


class TestModelsManagerAdd:
    """Test cases for ModelsManager.add method."""

    def test_add_model_empty_name(self, models_manager, mock_prompt, mock_console):
        """Test adding model with empty name."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = ""

        result = models_manager.add()

        assert result == 1
        mock_console.print.assert_called_with("❌ Model name cannot be empty")

    def test_add_model_invalid_name(self, models_manager, mock_prompt, mock_console):
        """Test adding model with invalid characters in name."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "invalid/name"

        result = models_manager.add()

        assert result == 1
        assert any("cannot contain" in str(call) for call in mock_console.print.call_args_list)

    def test_add_model_already_exists(self, models_manager, mock_prompt, mock_console):
        """Test adding model that already exists."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "test_model"  # Already exists

        result = models_manager.add()

        assert result == 1
        mock_console.print.assert_called_with("❌ Model 'test_model' already exists")

    def test_add_openai_model_success(self, models_manager, mock_prompt, mock_console):
        """Test successfully adding an OpenAI model."""
        mock_ask, mock_getpass, mock_confirm = mock_prompt

        # Need to provide base_url explicitly since empty string triggers validation
        mock_ask.side_effect = [
            "new_openai",  # model name
            "openai",  # model type
            "https://api.openai.com/v1",  # base URL
            "gpt-4o",  # model identifier
        ]
        mock_getpass.return_value = "sk-new-key"
        mock_confirm.side_effect = [False, False, False]  # save_trace, thinking, set_target

        with patch.object(models_manager, "_save_configuration", return_value=True):
            result = models_manager.add()

        assert result == 0
        assert "new_openai" in models_manager.models
        assert models_manager.models["new_openai"].type == "openai"
        assert models_manager.models["new_openai"].api_key == "sk-new-key"
        mock_console.print.assert_called_with("✔ Model 'new_openai' added successfully")

    def test_add_claude_model_success(self, models_manager, mock_prompt, mock_console):
        """Test successfully adding a Claude model."""
        mock_ask, mock_getpass, mock_confirm = mock_prompt

        mock_ask.side_effect = [
            "new_claude",  # model name
            "claude",  # model type
            "https://api.anthropic.com",  # base URL
            "claude-haiku-4-5",  # model identifier
        ]
        mock_getpass.return_value = "sk-ant-new-key"
        mock_confirm.side_effect = [False, False, False]

        with patch.object(models_manager, "_save_configuration", return_value=True):
            result = models_manager.add()

        assert result == 0
        assert "new_claude" in models_manager.models
        assert models_manager.models["new_claude"].type == "claude"
        assert models_manager.models["new_claude"].model == "claude-haiku-4-5"

    def test_add_model_with_thinking_enabled(self, models_manager, mock_prompt, mock_console):
        """Test adding model with thinking mode enabled."""
        mock_ask, mock_getpass, mock_confirm = mock_prompt

        mock_ask.side_effect = [
            "thinking_model",
            "deepseek",
            "https://api.deepseek.com",
            "deepseek-chat",
        ]
        mock_getpass.return_value = "test-key"
        mock_confirm.side_effect = [False, True, False]  # enable thinking

        with patch.object(models_manager, "_save_configuration", return_value=True):
            result = models_manager.add()

        assert result == 0
        assert models_manager.models["thinking_model"].enable_thinking is True

    def test_add_model_set_as_target(self, models_manager, mock_prompt, mock_console):
        """Test adding model and setting it as target."""
        mock_ask, mock_getpass, mock_confirm = mock_prompt

        mock_ask.side_effect = [
            "new_target",
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
        ]
        mock_getpass.return_value = "test-key"
        mock_confirm.side_effect = [False, False, True]  # set as target

        with patch.object(models_manager, "_save_configuration", return_value=True):
            result = models_manager.add()

        assert result == 0
        assert models_manager.target == "new_target"

    def test_add_model_empty_api_key(self, models_manager, mock_prompt, mock_console):
        """Test adding model with empty API key."""
        mock_ask, mock_getpass, _ = mock_prompt

        mock_ask.side_effect = [
            "new_model",
            "openai",
            "https://api.openai.com/v1",
        ]
        mock_getpass.return_value = ""  # Empty API key

        result = models_manager.add()

        assert result == 1
        mock_console.print.assert_called_with("❌ API Key cannot be empty")

    def test_add_model_save_failed(self, models_manager, mock_prompt, mock_console):
        """Test adding model when save fails."""
        mock_ask, mock_getpass, mock_confirm = mock_prompt

        mock_ask.side_effect = [
            "new_model",
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
        ]
        mock_getpass.return_value = "test-key"
        mock_confirm.side_effect = [False, False, False]

        with patch.object(models_manager, "_save_configuration", return_value=False):
            result = models_manager.add()

        assert result == 1
        mock_console.print.assert_called_with("❌ Failed to save configuration")


class TestModelsManagerDelete:
    """Test cases for ModelsManager.delete method."""

    def test_delete_no_models(self, models_manager, mock_console):
        """Test deleting when no models exist."""
        models_manager.models = {}

        result = models_manager.delete()

        assert result == 1
        mock_console.print.assert_called_with("❌ No models configured to delete")

    def test_delete_empty_name(self, models_manager, mock_prompt, mock_console):
        """Test deleting with empty model name."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = ""

        result = models_manager.delete()

        assert result == 1
        mock_console.print.assert_called_with("❌ Model name cannot be empty")

    def test_delete_non_existent_model(self, models_manager, mock_prompt, mock_console):
        """Test deleting model that doesn't exist."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "non_existent"

        result = models_manager.delete()

        assert result == 1
        mock_console.print.assert_called_with("❌ Model 'non_existent' does not exist")

    def test_delete_only_model(self, models_manager, mock_prompt, mock_console):
        """Test deleting the only model (should fail)."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "test_model"

        # Keep only one model
        models_manager.models = {"test_model": models_manager.models["test_model"]}
        models_manager.target = "test_model"

        result = models_manager.delete()

        assert result == 1
        mock_console.print.assert_called_with("❌ Cannot delete the only model. Add another model first.")

    def test_delete_cancelled(self, models_manager, mock_prompt, mock_console):
        """Test deleting when user cancels confirmation."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "another_model"
        mock_confirm.return_value = False  # User cancels

        result = models_manager.delete()

        assert result == 1
        mock_console.print.assert_called_with("❌ Model deletion cancelled")

    def test_delete_non_target_model_success(self, models_manager, mock_prompt, mock_console):
        """Test successfully deleting a non-target model."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "another_model"
        mock_confirm.return_value = True

        with patch.object(models_manager, "_save_configuration", return_value=True):
            result = models_manager.delete()

        assert result == 0
        assert "another_model" not in models_manager.models
        assert models_manager.target == "test_model"  # Target unchanged
        mock_console.print.assert_called_with("✔ Model 'another_model' deleted successfully")

    def test_delete_target_model_success(self, models_manager, mock_prompt, mock_console):
        """Test successfully deleting the target model."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.side_effect = [
            "test_model",  # model to delete
            "another_model",  # new target
        ]
        mock_confirm.return_value = True

        with patch.object(models_manager, "_save_configuration", return_value=True):
            result = models_manager.delete()

        assert result == 0
        assert "test_model" not in models_manager.models
        assert models_manager.target == "another_model"  # Target changed

    def test_delete_save_failed(self, models_manager, mock_prompt, mock_console):
        """Test deleting when save fails."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "another_model"
        mock_confirm.return_value = True

        with patch.object(models_manager, "_save_configuration", return_value=False):
            result = models_manager.delete()

        assert result == 1
        mock_console.print.assert_called_with("❌ Failed to save configuration after deletion")


class TestModelsManagerSetTarget:
    """Test cases for ModelsManager.set_target method."""

    def test_set_target_no_models(self, models_manager, mock_console):
        """Test setting target when no models exist."""
        models_manager.models = {}

        result = models_manager.set_target()

        assert result == 1
        mock_console.print.assert_called_with("❌ No models configured. Add a model first.")

    def test_set_target_same_as_current(self, models_manager, mock_prompt, mock_console):
        """Test setting target to the same model."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "test_model"  # Same as current

        result = models_manager.set_target()

        assert result == 0
        mock_console.print.assert_called_with("Target model is already 'test_model'")

    def test_set_target_success(self, models_manager, mock_prompt, mock_console):
        """Test successfully changing target model."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "another_model"

        with patch.object(models_manager, "_save_configuration", return_value=True):
            result = models_manager.set_target()

        assert result == 0
        assert models_manager.target == "another_model"
        mock_console.print.assert_called_with("✔ Target model set to 'another_model' successfully")

    def test_set_target_save_failed(self, models_manager, mock_prompt, mock_console):
        """Test setting target when save fails."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "another_model"

        with patch.object(models_manager, "_save_configuration", return_value=False):
            result = models_manager.set_target()

        assert result == 1
        mock_console.print.assert_called_with("❌ Failed to save configuration")


class TestModelsManagerRun:
    """Test cases for ModelsManager.run method."""

    def test_run_list_command(self, models_manager):
        """Test running list command."""
        with patch.object(models_manager, "list", return_value=0) as mock_list:
            result = models_manager.run("list")

            assert result == 0
            mock_list.assert_called_once()

    def test_run_add_command(self, models_manager):
        """Test running add command."""
        with patch.object(models_manager, "add", return_value=0) as mock_add:
            result = models_manager.run("add")

            assert result == 0
            mock_add.assert_called_once()

    def test_run_delete_command(self, models_manager):
        """Test running delete command."""
        with patch.object(models_manager, "delete", return_value=0) as mock_delete:
            result = models_manager.run("delete")

            assert result == 0
            mock_delete.assert_called_once()

    def test_run_set_target_command(self, models_manager):
        """Test running set-target command."""
        with patch.object(models_manager, "set_target", return_value=0) as mock_set_target:
            result = models_manager.run("set-target")

            assert result == 0
            mock_set_target.assert_called_once()

    def test_run_unknown_command(self, models_manager, mock_console):
        """Test running unknown command."""
        result = models_manager.run("unknown")

        assert result == 1
        mock_console.print.assert_called_with("❌ Unknown command: unknown")

    def test_run_config_not_loaded(self):
        """Test running command when config is not loaded."""
        with patch("datus.cli.models_manager.configuration_manager") as mock_config_manager:
            from datus.utils.exceptions import DatusException, ErrorCode

            mock_config_manager.side_effect = DatusException(ErrorCode.COMMON_FILE_NOT_FOUND)

            manager = ModelsManager("/non/existent/path.yml")
            result = manager.run("list")

            assert result == 1


class TestModelsManagerSaveConfiguration:
    """Test cases for ModelsManager._save_configuration method."""

    def test_save_configuration_success(self, models_manager, mock_console):
        """Test successful configuration save."""
        with patch.object(models_manager.config_manager, "update", return_value=True):
            result = models_manager._save_configuration()

            assert result is True
            assert any("Configuration saved" in str(call) for call in mock_console.print.call_args_list)

    def test_save_configuration_exception(self, models_manager, mock_console):
        """Test configuration save with exception."""
        with patch.object(models_manager.config_manager, "update", side_effect=Exception("Save error")):
            result = models_manager._save_configuration()

            assert result is False
            assert any("Failed to save configuration" in str(call) for call in mock_console.print.call_args_list)


class TestValidationFunctions:
    """Test cases for validation helper functions."""

    def test_validate_model_name_empty(self):
        """Test validation of empty model name."""
        from datus.cli.models_manager import _validate_model_name

        valid, error = _validate_model_name("")
        assert valid is False
        assert "cannot be empty" in error

    def test_validate_model_name_invalid_chars(self):
        """Test validation of model name with invalid characters."""
        from datus.cli.models_manager import _validate_model_name

        invalid_names = ["model/name", "model:name", "model name", "model*name"]
        for name in invalid_names:
            valid, error = _validate_model_name(name)
            assert valid is False
            assert "cannot contain" in error

    def test_validate_model_name_valid(self):
        """Test validation of valid model name."""
        from datus.cli.models_manager import _validate_model_name

        valid, error = _validate_model_name("valid_model_name")
        assert valid is True
        assert error == ""

    def test_validate_url_empty(self):
        """Test validation of empty URL."""
        from datus.cli.models_manager import _validate_url

        valid, error = _validate_url("")
        assert valid is False
        assert "cannot be empty" in error

    def test_validate_url_invalid_protocol(self):
        """Test validation of URL with invalid protocol."""
        from datus.cli.models_manager import _validate_url

        valid, error = _validate_url("ftp://example.com")
        assert valid is False
        assert "must start with http://" in error

    def test_validate_url_valid(self):
        """Test validation of valid URLs."""
        from datus.cli.models_manager import _validate_url

        valid_urls = ["https://api.openai.com", "http://localhost:8080"]
        for url in valid_urls:
            valid, error = _validate_url(url)
            assert valid is True
            assert error == ""
