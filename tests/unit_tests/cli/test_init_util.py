from unittest.mock import MagicMock, patch

from datus.cli.init_util import detect_db_connectivity


class TestDatabaseConnectivity:
    """Test cases for database connectivity testing."""

    def test_detect_db_connectivity_starrocks_success(self):
        """Test database connectivity test for StarRocks success."""
        config_data = {
            "type": "starrocks",
            "name": "test_namespace",
            "host": "127.0.0.1",
            "port": 9030,
            "username": "test_user",
            "password": "test_password",
            "database": "test_db",
            "catalog": "default_catalog",
        }

        with patch("datus.tools.db_tools.db_manager.DBManager") as mock_db_manager:
            # Mock successful connection test
            mock_connector = MagicMock()
            mock_connector.test_connection.return_value = True
            mock_db_manager.return_value.get_conn.return_value = mock_connector

            success, error_msg = detect_db_connectivity("starrocks", config_data)

            assert success is True
            assert error_msg == ""

    def test_detect_db_connectivity_failure(self):
        """Test database connectivity test failure."""
        config_data = {
            "type": "starrocks",
            "name": "test_namespace",
            "host": "127.0.0.1",
            "port": 9030,
            "username": "test_user",
            "password": "test_password",
            "database": "test_db",
        }

        with patch("datus.tools.db_tools.db_manager.DBManager") as mock_db_manager:
            # Mock connection test failure
            mock_connector = MagicMock()
            mock_connector.test_connection.return_value = False
            mock_db_manager.return_value.get_conn.return_value = mock_connector

            success, error_msg = detect_db_connectivity("starrocks", config_data)

            assert success is False
            assert error_msg == "Connection test failed"

    def test_detect_db_connectivity_exception(self):
        """Test database connectivity test with exception."""
        config_data = {
            "type": "starrocks",
            "name": "test_namespace",
            "host": "127.0.0.1",
            "port": 9030,
            "username": "test_user",
            "password": "test_password",
            "database": "test_db",
        }

        with (
            patch("datus.tools.db_tools.db_manager.DBManager") as mock_db_manager,
            patch("datus.cli.init_util.logger") as mock_logger,
        ):
            # Mock exception during connection test
            mock_db_manager.side_effect = Exception("Connection refused")

            success, error_msg = detect_db_connectivity("starrocks", config_data)

            assert success is False
            assert "Connection refused" in error_msg
            mock_logger.error.assert_called()
