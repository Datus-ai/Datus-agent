from unittest.mock import patch

import pytest

from datus.cli.namespace_manager import NamespaceManager
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from tests.conftest import TEST_CONF_DIR


@pytest.fixture
def agent_config(tmp_path) -> AgentConfig:
    # ``home=tmp_path`` pins every derived path inside the pytest-managed
    # tmp dir. The session-level ``_isolate_project_cwd`` autouse fixture
    # already chdir-s into tmp_path, so the yml's relative paths never
    # resolve under the repo root.
    return load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"), home=str(tmp_path), reload=True)


@pytest.fixture
def config_path(tmp_path):
    """Create a temporary config file path for testing."""
    import shutil

    # Copy test config to temp location
    test_config = TEST_CONF_DIR / "agent.yml"
    temp_config = tmp_path / "agent.yml"
    shutil.copy(test_config, temp_config)
    return str(temp_config)


@pytest.fixture
def mock_console():
    """Mock the console to capture output."""
    with patch("datus.cli.namespace_manager.console") as mock_console:
        yield mock_console


@pytest.fixture
def mock_detect_db_connectivity():
    """Mock the database connectivity test."""
    with patch("datus.cli.namespace_manager.detect_db_connectivity") as mock_test:
        mock_test.return_value = (True, "")
        yield mock_test


@pytest.fixture
def mock_save_configuration():
    """Mock the save configuration method."""
    with patch("datus.cli.namespace_manager.NamespaceManager._save_configuration") as mock_save:
        mock_save.return_value = True
        yield mock_save


@pytest.fixture
def mock_prompt():
    """Mock the rich prompt for user input."""
    with (
        patch("datus.cli.namespace_manager.Prompt.ask") as mock_ask,
        patch("datus.cli.namespace_manager.getpass") as mock_getpass,
        patch("datus.cli.namespace_manager.Confirm.ask") as mock_confirm,
    ):
        # Default mocks
        mock_ask.return_value = "non_exsited_namespace"
        mock_getpass.return_value = "test_password"
        mock_confirm.return_value = True

        yield mock_ask, mock_getpass, mock_confirm


class TestNamespaceManagerAdd:
    """Test cases for NamespaceManager.add method.

    These tests are marked as acceptance because they depend on:
    - Real connector registry and adapter metadata
    - Interactive prompting with variable number of fields per adapter
    - Database connectivity tests
    """

    def test_add_namespace_empty_name(self, config_path, mock_prompt, mock_console):
        """Test adding namespace with empty name."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = ""  # Empty namespace name

        nm = NamespaceManager(config_path)
        result = nm.add()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Namespace name cannot be empty")

    def test_add_namespace_already_exists(self, config_path, mock_prompt, mock_console):
        """Test adding namespace that already exists."""
        mock_ask, _, _ = mock_prompt
        # Use the project_name key which is the only key in compat namespaces dict
        nm = NamespaceManager(config_path)
        ns_key = list(nm.agent_config.namespaces.keys())[0]
        mock_ask.return_value = ns_key

        result = nm.add()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with(f"❌ Namespace '{ns_key}' already exists")

    def test_add_duckdb_namespace_success(
        self, config_path, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
    ):
        """Test successfully adding a DuckDB namespace."""
        mock_ask, _, _ = mock_prompt

        # Mock the sequence of user inputs for DuckDB
        mock_ask.side_effect = [
            "test_duckdb",  # namespace name
            "duckdb",  # database type
            "/path/to/test.db",  # connection string
        ]

        nm = NamespaceManager(config_path)
        result = nm.add()

        assert result == 0  # 0 means success
        # After add, the new entry should be in services.datasources keyed by the requested logical name.
        db_names = set(nm.agent_config.services.datasources.keys())
        assert "test_duckdb" in db_names, f"Expected 'test_duckdb' in services.datasources, got: {db_names}"
        mock_console.print.assert_any_call("✔ Database connection test successful\n")
        mock_console.print.assert_any_call("✔ Namespace 'test_duckdb' added successfully")


class TestNamespaceManagerList:
    """Test cases for NamespaceManager.list method."""

    def test_list_no_namespaces(self, config_path):
        """Test listing namespaces when none are configured."""
        nm = NamespaceManager(config_path)
        # Clear all databases (namespaces is a compat property, so clear the source)
        nm.agent_config.services.datasources.clear()

        result = nm.list()

        assert result == 0

    def test_list_with_namespaces(self, config_path):
        """Test listing namespaces when some are configured."""
        nm = NamespaceManager(config_path)
        # The config already has namespaces (bird_sqlite, etc.)

        result = nm.list()

        assert result == 0


class TestNamespaceManagerDelete:
    """Test cases for NamespaceManager.delete method.

    These tests are marked as acceptance because they depend on real configuration file operations.
    """

    def test_delete_namespace_not_exists(self, config_path, mock_prompt, mock_console):
        """Test deleting namespace when it doesn't exist."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "non_exsited_namespace"

        nm = NamespaceManager(config_path)
        result = nm.delete()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Namespace 'non_exsited_namespace' does not exist")

    def test_delete_namespace_empty_name(self, config_path, mock_prompt, mock_console):
        """Test deleting namespace with empty name."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = ""

        nm = NamespaceManager(config_path)
        result = nm.delete()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Namespace name cannot be empty")

    def test_delete_namespace_cancelled(self, config_path, mock_prompt, mock_console):
        """Test deleting namespace when user cancels confirmation."""
        mock_ask, _, mock_confirm = mock_prompt
        nm = NamespaceManager(config_path)
        ns_key = list(nm.agent_config.namespaces.keys())[0]
        mock_ask.return_value = ns_key
        mock_confirm.return_value = False  # User cancels

        result = nm.delete()

        assert result == 1  # 1 means failure (cancelled)
        mock_console.print.assert_called_with("❌ Namespace deletion cancelled")

    def test_delete_namespace_success(self, config_path, mock_prompt, mock_save_configuration, mock_console):
        """Test successfully deleting a namespace."""
        mock_ask, _, mock_confirm = mock_prompt
        nm = NamespaceManager(config_path)
        ns_key = list(nm.agent_config.namespaces.keys())[0]
        mock_ask.return_value = ns_key
        mock_confirm.return_value = True  # User confirms

        result = nm.delete()

        assert result == 0  # 0 means success
        mock_console.print.assert_called_with(f"✔ Namespace '{ns_key}' deleted successfully")

    def test_delete_namespace_save_failed(self, config_path, mock_prompt, mock_save_configuration, mock_console):
        """Test deleting namespace when save fails."""
        mock_ask, _, mock_confirm = mock_prompt
        nm = NamespaceManager(config_path)
        ns_key = list(nm.agent_config.namespaces.keys())[0]
        mock_ask.return_value = ns_key
        mock_confirm.return_value = True

        mock_save_configuration.return_value = False  # Save fails

        result = nm.delete()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Failed to save configuration after deletion")
