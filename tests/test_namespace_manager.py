from unittest.mock import patch

import pytest

from datus.cli.namespace_manager import NamespaceManager
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config()


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
    with patch("datus.cli.namespace_manager.Prompt.ask") as mock_ask, patch(
        "datus.cli.namespace_manager.getpass"
    ) as mock_getpass, patch("datus.cli.namespace_manager.Confirm.ask") as mock_confirm:
        # Default mocks
        mock_ask.return_value = "non_exsited_namespace"
        mock_getpass.return_value = "test_password"
        mock_confirm.return_value = True

        yield mock_ask, mock_getpass, mock_confirm


class TestNamespaceManagerAdd:
    """Test cases for NamespaceManager.add method."""

    def test_add_namespace_empty_name(self, agent_config, mock_prompt, mock_console):
        """Test adding namespace with empty name."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = ""  # Empty namespace name

        result = NamespaceManager.add(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Namespace name cannot be empty")

    def test_add_namespace_already_exists(self, agent_config, mock_prompt, mock_console):
        """Test adding namespace that already exists."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "existing_namespace"

        # Add existing namespace
        agent_config.namespaces["existing_namespace"] = {}

        result = NamespaceManager.add(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Namespace 'existing_namespace' already exists")

    def test_add_starrocks_namespace_success(
        self, agent_config, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
    ):
        """Test successfully adding a StarRocks namespace."""
        mock_ask, mock_getpass, _ = mock_prompt

        # Mock the sequence of user inputs for StarRocks
        mock_ask.side_effect = [
            "test_starrocks",  # namespace name
            "starrocks",  # database type
            "127.0.0.1",  # host
            "9030",  # port
            "test_user",  # username
            "test_db",  # database
        ]

        result = NamespaceManager.add(agent_config)

        assert result is True
        assert "test_starrocks" in agent_config.namespaces
        mock_console.print.assert_any_call("✔ Database connection test successful\n")
        mock_console.print.assert_any_call("✔ Namespace 'test_starrocks' added successfully")

    def test_add_snowflake_namespace_success(
        self, agent_config, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
    ):
        """Test successfully adding a Snowflake namespace."""
        mock_ask, mock_getpass, _ = mock_prompt

        # Mock the sequence of user inputs for Snowflake
        mock_ask.side_effect = [
            "test_snowflake",  # namespace name
            "snowflake",  # database type
            "test_user",  # username
            "test_account",  # account
            "test_warehouse",  # warehouse
            "test_db",  # database
            "test_schema",  # schema
        ]

        result = NamespaceManager.add(agent_config)

        assert result is True
        assert "test_snowflake" in agent_config.namespaces
        mock_console.print.assert_any_call("✔ Database connection test successful\n")
        mock_console.print.assert_any_call("✔ Namespace 'test_snowflake' added successfully")

    def test_add_duckdb_namespace_success(
        self, agent_config, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
    ):
        """Test successfully adding a DuckDB namespace."""
        mock_ask, _, _ = mock_prompt

        # Mock the sequence of user inputs for DuckDB
        mock_ask.side_effect = [
            "test_duckdb",  # namespace name
            "duckdb",  # database type
            "/path/to/test.db",  # connection string
        ]

        result = NamespaceManager.add(agent_config)

        assert result is True
        assert "test_duckdb" in agent_config.namespaces
        mock_console.print.assert_any_call("✔ Database connection test successful\n")
        mock_console.print.assert_any_call("✔ Namespace 'test_duckdb' added successfully")

    def test_add_namespace_db_connection_failed(
        self, agent_config, mock_prompt, mock_detect_db_connectivity, mock_console
    ):
        """Test adding namespace when database connection test fails."""
        mock_ask, mock_getpass, _ = mock_prompt

        # Mock the sequence of user inputs for StarRocks
        mock_ask.side_effect = [
            "test_failed",  # namespace name
            "starrocks",  # database type
            "127.0.0.1",  # host
            "9030",  # port
            "test_user",  # username
            "test_db",  # database
        ]

        # Mock connection test failure
        mock_detect_db_connectivity.return_value = (False, "Connection refused")

        result = NamespaceManager.add(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Database connectivity test failed: Connection refused\n")

    def test_add_namespace_save_config_failed(
        self, agent_config, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
    ):
        """Test adding namespace when configuration save fails."""
        mock_ask, mock_getpass, _ = mock_prompt

        # Mock the sequence of user inputs for StarRocks
        mock_ask.side_effect = [
            "test_save_failed",  # namespace name
            "starrocks",  # database type
            "127.0.0.1",  # host
            "9030",  # port
            "test_user",  # username
            "test_db",  # database
        ]

        # Mock save configuration failure
        mock_save_configuration.return_value = False

        result = NamespaceManager.add(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Failed to save configuration")


class TestNamespaceManagerList:
    """Test cases for NamespaceManager.list method."""

    def test_list_no_namespaces(self, agent_config, capsys):
        """Test listing namespaces when none are configured."""
        agent_config.namespaces = {}

        result = NamespaceManager.list(agent_config)

        assert result == 0
        captured = capsys.readouterr()
        assert "No namespace configured." in captured.out

    def test_list_with_namespaces(self, agent_config, capsys):
        """Test listing namespaces when some are configured."""
        from datus.configuration.agent_config import DbConfig

        # Create test namespace with database config
        db_config = DbConfig(type="duckdb", uri="duckdb:://test.db", database="test_db")
        agent_config.namespaces = {"test_namespace": {"test_db": db_config}}

        result = NamespaceManager.list(agent_config)

        assert result == 0
        captured = capsys.readouterr()
        assert "Configured namespaces:" in captured.out
        assert "Namespace: test_namespace" in captured.out
        assert "Type: duckdb" in captured.out
        assert "URI: duckdb:://test.db" in captured.out
        assert "Database: test_db" in captured.out


class TestNamespaceManagerDelete:
    """Test cases for NamespaceManager.delete method."""

    def test_delete_namespace_not_exists(self, agent_config, mock_prompt, mock_console):
        """Test deleting namespace when none exist."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = "non_exsited_namespace"

        result = NamespaceManager.delete(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Namespace 'non_exsited_namespace' does not exist")

    def test_delete_namespace_empty_name(self, agent_config, mock_prompt, mock_console):
        """Test deleting namespace with empty name."""
        mock_ask, _, _ = mock_prompt
        mock_ask.return_value = ""

        result = NamespaceManager.delete(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Namespace name cannot be empty")

    def test_delete_namespace_cancelled(self, agent_config, mock_prompt, mock_console):
        """Test deleting namespace when user cancels confirmation."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "test_namespace"
        mock_confirm.return_value = False  # User cancels

        agent_config.namespaces = {"test_namespace": {}}

        result = NamespaceManager.delete(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Namespace deletion cancelled")

    def test_delete_namespace_success(self, agent_config, mock_prompt, mock_save_configuration, mock_console):
        """Test successfully deleting a namespace."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "test_namespace"
        mock_confirm.return_value = True  # User confirms

        agent_config.namespaces = {"test_namespace": {}}

        result = NamespaceManager.delete(agent_config)

        assert result is True
        assert "test_namespace" not in agent_config.namespaces
        mock_console.print.assert_called_with("✔ Namespace 'test_namespace' deleted successfully")

    def test_delete_namespace_save_failed(self, agent_config, mock_prompt, mock_save_configuration, mock_console):
        """Test deleting namespace when save fails."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "test_namespace"
        mock_confirm.return_value = True

        agent_config.namespaces = {"test_namespace": {}}
        mock_save_configuration.return_value = False  # Save fails

        result = NamespaceManager.delete(agent_config)

        assert result is False
        mock_console.print.assert_called_with("❌ Failed to save configuration after deletion")
