"""
Integration tests for SQLAlchemy connector exception handling.
Tests real database scenarios with SQLite.
"""

import os
import tempfile

import pytest

from datus.tools.db_tools import SQLiteConnector
from datus.utils.exceptions import DatusException, ErrorCode


class TestIntegrationExceptions:
    """Integration tests with real SQLite database."""

    def test_sqlite_connection_failure(self):
        """Test connection failure with invalid SQLite path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "nonexistent", "database.db")
            connector = SQLiteConnector(f"sqlite:///{invalid_path}")

            with pytest.raises(DatusException) as exc_info:
                connector.test_connection()
            # SQLite connection errors should be mapped to DB_CONNECTION_FAILED
            assert exc_info.value.code == ErrorCode.DB_CONNECTION_FAILED

    def test_sqlite_table_not_found(self):
        """Test actual table not found error."""
        connector = SQLiteConnector("sqlite:///:memory:")

        result = connector.execute_query("SELECT * FROM nonexistent_table")
        assert not result.success
        assert ErrorCode.DB_EXECUTION_ERROR.code in result.error

    def test_sqlite_column_not_found(self):
        """Test actual column not found error."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Create a table
        connector.execute_ddl("CREATE TABLE test_table (id INTEGER, name TEXT)")

        result = connector.execute_query("SELECT nonexistent_column FROM test_table")

        assert not result.success
        assert ErrorCode.DB_EXECUTION_ERROR.code in result.error

    def test_sqlite_syntax_error(self):
        """Test actual SQL syntax error."""
        connector = SQLiteConnector("sqlite:///:memory:")

        result = connector.execute_query("SELEC * FROM test_table")
        assert not result.success
        assert ErrorCode.DB_EXECUTION_SYNTAX_ERROR.code in result.error

    def test_sqlite_primary_key_violation(self):
        """Test actual primary key violation."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Create table with primary key
        connector.execute_ddl("CREATE TABLE test_pk (id INTEGER PRIMARY KEY)")
        connector.execute_insert("INSERT INTO test_pk (id) VALUES (1)")

        res = connector.execute_insert("INSERT INTO test_pk (id) VALUES (1)")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_unique_constraint_violation(self):
        """Test actual unique constraint violation."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Create table with unique constraint
        connector.execute_ddl("CREATE TABLE test_unique (email TEXT UNIQUE)")
        connector.execute_insert("INSERT INTO test_unique (email) VALUES ('test@example.com')")

        res = connector.execute_insert("INSERT INTO test_unique (email) VALUES ('test@example.com')")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_not_null_violation(self):
        """Test actual not null constraint violation."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Create table with not null constraint
        connector.execute_ddl("CREATE TABLE test_notnull (name TEXT NOT NULL)")

        res = connector.execute_insert("INSERT INTO test_notnull (name) VALUES (NULL)")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_foreign_key_violation(self):
        """Test actual foreign key violation."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Enable foreign key constraints
        connector.execute_ddl("PRAGMA foreign_keys = ON")

        # Create tables with foreign key
        connector.execute_ddl("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connector.execute_ddl("CREATE TABLE child (parent_id INTEGER, FOREIGN KEY (parent_id) REFERENCES parent(id))")

        res = connector.execute_insert("INSERT INTO child (parent_id) VALUES (999)")
        assert res.success is False

        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_successful_operations_do_not_raise_exceptions(self):
        """Test that successful operations don't raise exceptions."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Create table
        connector.execute_ddl("CREATE TABLE test_success (id INTEGER, name TEXT)")

        # Insert data
        result = connector.execute_insert("INSERT INTO test_success (id, name) VALUES (1, 'test')")
        assert result.sql_return == "1"  # rowcount should be 1

        # Query data
        df = connector.execute_pandas("SELECT * FROM test_success").sql_return
        assert len(df) == 1
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "test"

    def test_update_operations(self):
        """Test update operations with exception handling."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Create table and insert data
        connector.execute_ddl("CREATE TABLE test_update (id INTEGER, value INTEGER)")
        connector.execute_insert("INSERT INTO test_update (id, value) VALUES (1, 100)")

        # Successful update
        res = connector.execute_update("UPDATE test_update SET value = 200 WHERE id = 1")
        assert res.row_count == 1

        # Update non-existent record (should succeed but return 0 rows)
        res = connector.execute_update("UPDATE test_update SET value = 300 WHERE id = 999")
        assert res.row_count == 0

    def test_delete_operations(self):
        """Test delete operations with exception handling."""
        connector = SQLiteConnector("sqlite:///:memory:")

        # Create table and insert data
        connector.execute_ddl("CREATE TABLE test_delete (id INTEGER)")
        connector.execute_insert("INSERT INTO test_delete (id) VALUES (1)")

        # Successful delete
        res = connector.execute_delete("DELETE FROM test_delete WHERE id = 1")
        assert res.row_count == 1

        # Delete non-existent record (should succeed but return 0 rows)
        res = connector.execute_delete("DELETE FROM test_delete WHERE id = 999")
        assert res.row_count == 0
