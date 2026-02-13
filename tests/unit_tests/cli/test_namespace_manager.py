from unittest.mock import patch

import pytest

from datus.cli.namespace_manager import NamespaceManager
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from tests.conftest import TEST_CONF_DIR


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"), reload=True)


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
        mock_ask.return_value = "bird_sqlite"  # Use existing namespace from test config

        nm = NamespaceManager(config_path)
        result = nm.add()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Namespace 'bird_sqlite' already exists")

    @pytest.mark.skip(reason="starrocks adapter not installed in test environment")
    def test_add_starrocks_namespace_success(
        self, config_path, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
    ):
        """Test successfully adding a StarRocks namespace."""
        mock_ask, mock_getpass, _ = mock_prompt

        # Mock the sequence of user inputs for StarRocks
        # StarRocks fields: host, port, username, password, catalog, database, charset, autocommit, timeout_seconds
        mock_ask.side_effect = [
            "test_starrocks",  # namespace name
            "starrocks",  # database type
            "127.0.0.1",  # host
            "9030",  # port
            "test_user",  # username
            # password is prompted via getpass, not mock_ask
            "default_catalog",  # catalog
            "test_db",  # database
            "utf8mb4",  # charset
            "True",  # autocommit
            "30",  # timeout_seconds
        ]

        nm = NamespaceManager(config_path)
        result = nm.add()

        assert result == 0  # 0 means success
        assert "test_starrocks" in nm.agent_config.namespaces
        mock_console.print.assert_any_call("✔ Database connection test successful\n")
        mock_console.print.assert_any_call("✔ Namespace 'test_starrocks' added successfully")

    @pytest.mark.skip(reason="Snowflake adapter not installed in test environment")
    def test_add_snowflake_namespace_success(
        self, config_path, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
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

        nm = NamespaceManager(config_path)
        result = nm.add()

        assert result == 0  # 0 means success
        assert "test_snowflake" in nm.agent_config.namespaces
        mock_console.print.assert_any_call("✔ Database connection test successful\n")
        mock_console.print.assert_any_call("✔ Namespace 'test_snowflake' added successfully")

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
        assert "test_duckdb" in nm.agent_config.namespaces
        mock_console.print.assert_any_call("✔ Database connection test successful\n")
        mock_console.print.assert_any_call("✔ Namespace 'test_duckdb' added successfully")

    @pytest.mark.skip(reason="starrocks adapter not installed in test environment")
    def test_add_namespace_db_connection_failed(
        self, config_path, mock_prompt, mock_detect_db_connectivity, mock_console
    ):
        """Test adding namespace when database connection test fails."""
        mock_ask, mock_getpass, _ = mock_prompt

        # Mock the sequence of user inputs for StarRocks
        # StarRocks fields: host, port, username, password, catalog, database, charset, autocommit, timeout_seconds
        mock_ask.side_effect = [
            "test_failed",  # namespace name
            "starrocks",  # database type
            "127.0.0.1",  # host
            "9030",  # port
            "test_user",  # username
            # password is prompted via getpass, not mock_ask
            "default_catalog",  # catalog
            "test_db",  # database
            "utf8mb4",  # charset
            "True",  # autocommit
            "30",  # timeout_seconds
        ]

        # Mock connection test failure
        mock_detect_db_connectivity.return_value = (False, "Connection refused")

        nm = NamespaceManager(config_path)
        result = nm.add()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Database connectivity test failed: Connection refused\n")

    @pytest.mark.skip(reason="starrocks adapter not installed in test environment")
    def test_add_namespace_save_config_failed(
        self, config_path, mock_prompt, mock_detect_db_connectivity, mock_save_configuration, mock_console
    ):
        """Test adding namespace when configuration save fails."""
        mock_ask, mock_getpass, _ = mock_prompt

        # Mock the sequence of user inputs for StarRocks
        # StarRocks fields: host, port, username, password, catalog, database, charset, autocommit, timeout_seconds
        mock_ask.side_effect = [
            "test_save_failed",  # namespace name
            "starrocks",  # database type
            "127.0.0.1",  # host
            "9030",  # port
            "test_user",  # username
            # password is prompted via getpass, not mock_ask
            "default_catalog",  # catalog
            "test_db",  # database
            "utf8mb4",  # charset
            "True",  # autocommit
            "30",  # timeout_seconds
        ]

        # Mock save configuration failure
        mock_save_configuration.return_value = False

        nm = NamespaceManager(config_path)
        result = nm.add()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Failed to save configuration")


class TestNamespaceManagerList:
    """Test cases for NamespaceManager.list method."""

    def test_list_no_namespaces(self, config_path):
        """Test listing namespaces when none are configured."""
        nm = NamespaceManager(config_path)
        # Clear all namespaces
        nm.agent_config.namespaces = {}

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
        mock_ask.return_value = "bird_sqlite"  # Use existing namespace
        mock_confirm.return_value = False  # User cancels

        nm = NamespaceManager(config_path)
        result = nm.delete()

        assert result == 1  # 1 means failure (cancelled)
        mock_console.print.assert_called_with("❌ Namespace deletion cancelled")

    def test_delete_namespace_success(self, config_path, mock_prompt, mock_save_configuration, mock_console):
        """Test successfully deleting a namespace."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "bird_sqlite"  # Use existing namespace
        mock_confirm.return_value = True  # User confirms

        nm = NamespaceManager(config_path)
        result = nm.delete()

        assert result == 0  # 0 means success
        assert "bird_sqlite" not in nm.agent_config.namespaces
        mock_console.print.assert_called_with("✔ Namespace 'bird_sqlite' deleted successfully")

    def test_delete_namespace_save_failed(self, config_path, mock_prompt, mock_save_configuration, mock_console):
        """Test deleting namespace when save fails."""
        mock_ask, _, mock_confirm = mock_prompt
        mock_ask.return_value = "bird_sqlite"  # Use existing namespace
        mock_confirm.return_value = True

        mock_save_configuration.return_value = False  # Save fails

        nm = NamespaceManager(config_path)
        result = nm.delete()

        assert result == 1  # 1 means failure
        mock_console.print.assert_called_with("❌ Failed to save configuration after deletion")
