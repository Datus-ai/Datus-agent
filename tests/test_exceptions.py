"""
Integration tests for SQLAlchemy connector exception handling.
Tests real database scenarios with SQLite.
"""

import os
import tempfile

import pytest

from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.exceptions import DatusException, ErrorCode


class TestIntegrationExceptions:
    """Integration tests with real SQLite database."""

    def test_sqlite_connection_failure(self):
        """Test connection failure with invalid SQLite path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "nonexistent", "database.db")
            connector = SQLAlchemyConnector(f"sqlite:///{invalid_path}")

            with pytest.raises(DatusException) as exc_info:
                connector.connect()

            # SQLite connection errors should be mapped to DB_CONNECTION_FAILED
            assert exc_info.value.code == ErrorCode.DB_CONNECTION_FAILED

    def test_sqlite_table_not_found(self):
        """Test actual table not found error."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        with pytest.raises(DatusException) as exc_info:
            connector.execute_query("SELECT * FROM nonexistent_table")

        assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    def test_sqlite_column_not_found(self):
        """Test actual column not found error."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Create a table
        connector.execute_query("CREATE TABLE test_table (id INTEGER, name TEXT)")

        with pytest.raises(DatusException) as exc_info:
            connector.execute_query("SELECT nonexistent_column FROM test_table")

        assert exc_info.value.code == ErrorCode.DB_EXECUTION_ERROR

    def test_sqlite_syntax_error(self):
        """Test actual SQL syntax error."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        with pytest.raises(DatusException) as exc_info:
            connector.execute_query("SELEC * FROM test_table")

        assert exc_info.value.code == ErrorCode.DB_EXECUTION_SYNTAX_ERROR

    def test_sqlite_primary_key_violation(self):
        """Test actual primary key violation."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Create table with primary key
        connector.execute_query("CREATE TABLE test_pk (id INTEGER PRIMARY KEY)")
        connector.execute_query("INSERT INTO test_pk (id) VALUES (1)")

        with pytest.raises(DatusException) as exc_info:
            connector.insert("INSERT INTO test_pk (id) VALUES (1)")

        assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION

    def test_sqlite_unique_constraint_violation(self):
        """Test actual unique constraint violation."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Create table with unique constraint
        connector.execute_query("CREATE TABLE test_unique (email TEXT UNIQUE)")
        connector.execute_query("INSERT INTO test_unique (email) VALUES ('test@example.com')")

        with pytest.raises(DatusException) as exc_info:
            connector.insert("INSERT INTO test_unique (email) VALUES ('test@example.com')")

        assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION

    def test_sqlite_not_null_violation(self):
        """Test actual not null constraint violation."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Create table with not null constraint
        connector.execute_query("CREATE TABLE test_notnull (name TEXT NOT NULL)")

        with pytest.raises(DatusException) as exc_info:
            connector.insert("INSERT INTO test_notnull (name) VALUES (NULL)")

        assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION

    def test_sqlite_foreign_key_violation(self):
        """Test actual foreign key violation."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Enable foreign key constraints
        connector.execute_query("PRAGMA foreign_keys = ON")

        # Create tables with foreign key
        connector.execute_query("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connector.execute_query("CREATE TABLE child (parent_id INTEGER, FOREIGN KEY (parent_id) REFERENCES parent(id))")

        with pytest.raises(DatusException) as exc_info:
            connector.insert("INSERT INTO child (parent_id) VALUES (999)")

        assert exc_info.value.code == ErrorCode.DB_CONSTRAINT_VIOLATION

    def test_successful_operations_do_not_raise_exceptions(self):
        """Test that successful operations don't raise exceptions."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Create table
        connector.execute_query("CREATE TABLE test_success (id INTEGER, name TEXT)")

        # Insert data
        result = connector.insert("INSERT INTO test_success (id, name) VALUES (1, 'test')")
        assert result[1] == 1  # rowcount should be 1

        # Query data
        df = connector.execute_query("SELECT * FROM test_success")
        assert len(df) == 1
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "test"

    def test_update_operations(self):
        """Test update operations with exception handling."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Create table and insert data
        connector.execute_query("CREATE TABLE test_update (id INTEGER, value INTEGER)")
        connector.insert("INSERT INTO test_update (id, value) VALUES (1, 100)")

        # Successful update
        rows = connector.update("UPDATE test_update SET value = 200 WHERE id = 1")
        assert rows == 1

        # Update non-existent record (should succeed but return 0 rows)
        rows = connector.update("UPDATE test_update SET value = 300 WHERE id = 999")
        assert rows == 0

    def test_delete_operations(self):
        """Test delete operations with exception handling."""
        connector = SQLAlchemyConnector("sqlite:///:memory:")
        connector.connect()

        # Create table and insert data
        connector.execute_query("CREATE TABLE test_delete (id INTEGER)")
        connector.insert("INSERT INTO test_delete (id) VALUES (1)")

        # Successful delete
        rows = connector.delete("DELETE FROM test_delete WHERE id = 1")
        assert rows == 1

        # Delete non-existent record (should succeed but return 0 rows)
        rows = connector.delete("DELETE FROM test_delete WHERE id = 999")
        assert rows == 0
