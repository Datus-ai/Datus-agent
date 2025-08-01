"""
Test cases for SQLAlchemy connector exception handling.
Tests the mapping of SQLAlchemy exceptions to specific Datus ErrorCode values.
"""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import (
    DatabaseError,
    DataError,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    TimeoutError,
)

from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.exceptions import DatusException, ErrorCode


class TestSQLAlchemyConnectorExceptions:
    """Test suite for SQLAlchemy connector exception handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.connector = SQLAlchemyConnector("sqlite:///:memory:")

    # Connection-related exception tests

    def test_connection_timeout_exception(self):
        """Test mapping OperationalError with timeout to DB_CONNECTION_TIMEOUT."""
        # Mock the engine creation to raise OperationalError
        with patch("datus.tools.db_tools.sqlalchemy_connector.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = OperationalError("", "", "Connection timeout occurred")

            with pytest.raises(DatusException) as exc_info:
                self.connector.connect()
            assert exc_info.value.code == ErrorCode.DB_CONNECTION_TIMEOUT
            assert "Connection timeout" in str(exc_info.value)

    def test_authentication_failed_exception(self):
        """Test mapping OperationalError with authentication failure to DB_AUTHENTICATION_FAILED."""
        # Mock the engine creation to raise OperationalError
        with patch("datus.tools.db_tools.sqlalchemy_connector.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = OperationalError("", "", "Access denied for user 'testuser'@'localhost'")

            with pytest.raises(DatusException) as exc_info:
                self.connector.connect()

            assert exc_info.value.code == ErrorCode.DB_AUTHENTICATION_FAILED
            assert "Access denied" in str(exc_info.value)

    def test_permission_denied_exception(self):
        """Test mapping OperationalError with permission denied to DB_PERMISSION_DENIED."""
        # Mock the engine creation to raise OperationalError
        with patch("datus.tools.db_tools.sqlalchemy_connector.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = OperationalError("", "", "Permission denied for relation 'test_table'")

            with pytest.raises(DatusException) as exc_info:
                self.connector.connect()

            assert exc_info.value.code == ErrorCode.DB_PERMISSION_DENIED
            assert "Permission denied" in str(exc_info.value)

    def test_connection_refused_exception(self):
        """Test mapping OperationalError with connection refused to DB_CONNECTION_FAILED."""
        # Mock the engine creation to raise OperationalError
        with patch("datus.tools.db_tools.sqlalchemy_connector.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = OperationalError("", "", "Connection refused")

            with pytest.raises(DatusException) as exc_info:
                self.connector.connect()

            assert exc_info.value.code == ErrorCode.DB_CONNECTION_FAILED
            assert "Connection refused" in str(exc_info.value)

    def test_interface_error_exception(self):
        """Test mapping InterfaceError to appropriate error code."""
        # Mock the engine creation to raise InterfaceError
        with patch("datus.tools.db_tools.sqlalchemy_connector.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = InterfaceError("", "", "Could not connect to server")

            with pytest.raises(DatusException) as exc_info:
                self.connector.connect()

            assert exc_info.value.code == ErrorCode.DB_CONNECTION_FAILED

    # SQL Syntax and Programming error tests

    def test_sql_syntax_error_exception(self):
        """Test mapping ProgrammingError with syntax error to DB_QUERY_SYNTAX_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = ProgrammingError("", "", 'syntax error at or near "SELEC"')
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELEC * FROM test")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_SYNTAX_ERROR
                assert "syntax error" in str(exc_info.value)

    def test_table_not_found_exception(self):
        """Test mapping ProgrammingError with table not found to DB_EXECUTION_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = ProgrammingError("", "", 'relation "nonexistent_table" does not exist')
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELECT * FROM nonexistent_table")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    def test_column_not_found_exception(self):
        """Test mapping ProgrammingError with column not found to DB_EXECUTION_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = ProgrammingError("", "", 'column "nonexistent_column" does not exist')
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELECT nonexistent_column FROM test")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    def test_schema_not_found_exception(self):
        """Test mapping ProgrammingError with schema not found to DB_EXECUTION_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = ProgrammingError("", "", 'schema "nonexistent_schema" does not exist')
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELECT * FROM nonexistent_schema.test")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    # Constraint violation tests

    def test_primary_key_violation_exception(self):
        """Test mapping IntegrityError with primary key violation to DB_CONSTRAINT_VIOLATION."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = IntegrityError(
                    "", "", 'duplicate key value violates unique constraint "test_pkey"'
                )
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.insert("INSERT INTO test (id) VALUES (1)")

                assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION
                assert "duplicate key" in str(exc_info.value)

    def test_foreign_key_violation_exception(self):
        """Test mapping IntegrityError with foreign key violation to DB_CONSTRAINT_VIOLATION."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = IntegrityError(
                    "", "", 'insert or update on table "test" violates foreign key constraint'
                )
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.insert("INSERT INTO test (foreign_id) VALUES (999)")

                assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION
                assert "foreign key constraint" in str(exc_info.value)

    def test_unique_constraint_violation_exception(self):
        """Test mapping IntegrityError with unique constraint violation to DB_CONSTRAINT_VIOLATION."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = IntegrityError(
                    "", "", 'duplicate key value violates unique constraint "test_unique"'
                )
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.insert("INSERT INTO test (unique_col) VALUES ('duplicate')")

                assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION
                assert "unique constraint" in str(exc_info.value)

    def test_not_null_violation_exception(self):
        """Test mapping IntegrityError with not null violation to DB_CONSTRAINT_VIOLATION."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = IntegrityError(
                    "", "", 'null value in column "required_col" violates not-null constraint'
                )
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.insert("INSERT INTO test (required_col) VALUES (NULL)")

                assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION
                assert "not-null constraint" in str(exc_info.value)

    def test_general_constraint_violation_exception(self):
        """Test mapping general IntegrityError to DB_CONSTRAINT_VIOLATION."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = IntegrityError("", "", "some other constraint violation")
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.insert("INSERT INTO test VALUES (1)")

                assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION

    # Timeout and other error tests

    def test_query_timeout_exception(self):
        """Test mapping TimeoutError to DB_QUERY_TIMEOUT."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = TimeoutError("Query execution timeout")
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELECT * FROM large_table")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_TIMEOUT
                assert "timeout" in str(exc_info.value)

    def test_database_error_exception(self):
        """Test mapping DatabaseError to DB_QUERY_EXECUTION_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = DatabaseError("", "", "database error")
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELECT * FROM test")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR
                assert "database error" in str(exc_info.value)

    def test_data_error_exception(self):
        """Test mapping DataError to DB_QUERY_EXECUTION_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = DataError("", "", "numeric value out of range")
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("INSERT INTO test (num) VALUES (9999999999)")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    def test_internal_error_exception(self):
        """Test mapping InternalError to DB_QUERY_EXECUTION_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = InternalError("", "", "internal database error")
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELECT * FROM test")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    def test_not_supported_error_exception(self):
        """Test mapping NotSupportedError to DB_QUERY_EXECUTION_ERROR."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = NotSupportedError("", "", "feature not supported")
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELECT * FROM test")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    # Error message extraction tests

    def test_extract_table_name_from_error(self):
        """Test extraction of table name from error messages."""
        error_message = 'relation "test_table" does not exist'
        table_name = self.connector._extract_table_name_from_error(error_message)
        assert table_name == "test_table"

        error_message = "Table 'test.abc' doesn't exist"
        table_name = self.connector._extract_table_name_from_error(error_message)
        assert table_name == "test.abc"

    def test_extract_column_name_from_error(self):
        """Test extraction of column name from error messages."""
        error_message = 'column "test_column" does not exist'
        column_name = self.connector._extract_column_name_from_error(error_message)
        assert column_name == "test_column"

    def test_extract_schema_name_from_error(self):
        """Test extraction of schema name from error messages."""
        error_message = 'schema "test_schema" does not exist'
        schema_name = self.connector._extract_schema_name_from_error(error_message)
        assert schema_name == "test_schema"
        error_message = "Unknown database 'abc'"
        schema_name = self.connector._extract_schema_name_from_error(error_message)
        assert schema_name == "abc"

    def test_extract_table_name_with_quotes(self):
        """Test extraction of table name with mixed quotes."""
        error_message = "relation 'test_table' does not exist"
        table_name = self.connector._extract_table_name_from_error(error_message)
        assert table_name == "test_table"

    def test_extract_no_table_name(self):
        """Test extraction when no table name is found."""
        error_message = "some generic error message"
        table_name = self.connector._extract_table_name_from_error(error_message)
        assert table_name is None

    # Integration tests with actual SQLite database

    def test_actual_connection_failure(self):
        """Test actual connection failure with invalid database."""
        connector = SQLAlchemyConnector("sqlite:///nonexistent/path/to/database.db")

        with pytest.raises(DatusException) as exc_info:
            connector.connect()

        assert exc_info.value.code == ErrorCode.DB_CONNECTION_FAILED

    def test_actual_sql_syntax_error(self):
        """Test actual SQL syntax error."""
        with patch.object(self.connector, "connect"):
            with patch.object(self.connector, "connection") as mock_connection:
                mock_execute = Mock()
                mock_execute.side_effect = ProgrammingError("", "", 'near "SELEC": syntax error')
                mock_connection.execute = mock_execute

                with pytest.raises(DatusException) as exc_info:
                    self.connector.execute_query("SELEC * FROM test")

                assert exc_info.value.code == ErrorCode.DB_EXECUTION_SYNTAX_ERROR

    def test_insert_with_constraint_violation_in_memory(self):
        """Test insert with constraint violation using in-memory SQLite."""
        # This test uses an actual in-memory SQLite database
        connector = SQLAlchemyConnector("sqlite:///:memory:")

        # Create a table with a primary key
        connector.connect()
        connector.execute_ddl("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        connector.insert("INSERT INTO test (id) VALUES (1)")

        # Attempt to insert duplicate primary key
        with pytest.raises(DatusException) as exc_info:
            connector.insert("INSERT INTO test (id) VALUES (1)")

        # The actual SQLite error will be mapped appropriately
        assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION
