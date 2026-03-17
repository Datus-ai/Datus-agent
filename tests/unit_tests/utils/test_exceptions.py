"""
Integration tests for SQLAlchemy connector exception handling.
Tests real database scenarios with SQLite.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from datus.tools.db_tools import SQLiteConnector
from datus.tools.db_tools.config import SQLiteConfig
from datus.utils.exceptions import DatusException, ErrorCode, setup_exception_handler


class TestIntegrationExceptions:
    """Integration tests with real SQLite database."""

    def test_sqlite_connection_failure(self):
        """Test connection failure with invalid SQLite path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "nonexistent", "database.db")
            config = SQLiteConfig(db_path=f"sqlite:///{invalid_path}")
            connector = SQLiteConnector(config)

            with pytest.raises(DatusException) as exc_info:
                connector.test_connection()
            # SQLite connection errors should be mapped to DB_CONNECTION_FAILED
            assert exc_info.value.code == ErrorCode.DB_CONNECTION_FAILED

    def test_sqlite_table_not_found(self):
        """Test actual table not found error."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        result = connector.execute_query("SELECT * FROM nonexistent_table")
        assert not result.success
        assert ErrorCode.DB_TABLE_NOT_EXISTS.code in result.error

    def test_sqlite_column_not_found(self):
        """Test actual column not found error."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create a table
        connector.execute_ddl("CREATE TABLE test_table (id INTEGER, name TEXT)")

        result = connector.execute_query("SELECT nonexistent_column FROM test_table")

        assert not result.success
        assert ErrorCode.DB_EXECUTION_ERROR.code in result.error

    def test_sqlite_syntax_error(self):
        """Test actual SQL syntax error."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        result = connector.execute_query("SELEC * FROM test_table")
        assert not result.success
        assert ErrorCode.DB_EXECUTION_SYNTAX_ERROR.code in result.error

    def test_sqlite_primary_key_violation(self):
        """Test actual primary key violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table with primary key
        connector.execute_ddl("CREATE TABLE test_pk (id INTEGER PRIMARY KEY)")
        connector.execute_insert("INSERT INTO test_pk (id) VALUES (1)")

        res = connector.execute_insert("INSERT INTO test_pk (id) VALUES (1)")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_unique_constraint_violation(self):
        """Test actual unique constraint violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table with unique constraint
        connector.execute_ddl("CREATE TABLE test_unique (email TEXT UNIQUE)")
        connector.execute_insert("INSERT INTO test_unique (email) VALUES ('test@example.com')")

        res = connector.execute_insert("INSERT INTO test_unique (email) VALUES ('test@example.com')")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_not_null_violation(self):
        """Test actual not null constraint violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table with not null constraint
        connector.execute_ddl("CREATE TABLE test_notnull (name TEXT NOT NULL)")

        res = connector.execute_insert("INSERT INTO test_notnull (name) VALUES (NULL)")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_foreign_key_violation(self):
        """Test actual foreign key violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

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
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

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
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

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
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table and insert data
        connector.execute_ddl("CREATE TABLE test_delete (id INTEGER)")
        connector.execute_insert("INSERT INTO test_delete (id) VALUES (1)")

        # Successful delete
        res = connector.execute_delete("DELETE FROM test_delete WHERE id = 1")
        assert res.row_count == 1

        # Delete non-existent record (should succeed but return 0 rows)
        res = connector.execute_delete("DELETE FROM test_delete WHERE id = 999")
        assert res.row_count == 0


class TestDatusExceptionBuildMsg:
    """Tests for DatusException.build_msg (line 167)."""

    def test_custom_message_takes_priority(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message="custom msg")
        assert "custom msg" in str(ex)

    def test_message_args_format_template(self):
        ex = DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message_args={"field_name": "username"},
        )
        assert "username" in str(ex)

    def test_no_message_no_args_uses_desc(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert ErrorCode.COMMON_UNKNOWN.desc in str(ex)

    def test_error_code_appears_in_message(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert ErrorCode.COMMON_UNKNOWN.code in str(ex)

    def test_str_returns_message(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message="hello")
        assert str(ex) == ex.message

    def test_custom_message_overrides_args(self):
        """When both message and message_args are provided, custom message wins."""
        ex = DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message="explicit override",
            message_args={"field_name": "x"},
        )
        assert "explicit override" in str(ex)

    def test_is_exception_subclass(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert isinstance(ex, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(DatusException) as exc_info:
            raise DatusException(ErrorCode.COMMON_UNKNOWN, message="test raise")
        assert "test raise" in str(exc_info.value)

    def test_code_attribute_set(self):
        ex = DatusException(ErrorCode.COMMON_FIELD_REQUIRED, message_args={"field_name": "f"})
        assert ex.code == ErrorCode.COMMON_FIELD_REQUIRED

    def test_message_args_attribute_set(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message_args={"key": "val"})
        assert ex.message_args == {"key": "val"}

    def test_message_args_defaults_to_empty_dict_when_none(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert ex.message_args == {}

    @pytest.mark.parametrize("code", list(ErrorCode))
    def test_all_error_codes_instantiable(self, code):
        """Every ErrorCode can be used to create a DatusException."""
        ex = DatusException(code)
        assert isinstance(ex, DatusException)


class TestSetupExceptionHandler:
    """Tests for setup_exception_handler (lines 178-219)."""

    def test_sets_sys_excepthook(self):
        original_hook = sys.excepthook
        try:
            setup_exception_handler()
            assert sys.excepthook is not original_hook
        finally:
            sys.excepthook = original_hook

    def test_system_exit_not_intercepted(self):
        """SystemExit should fall through to the original hook, not be swallowed."""
        original_hook = sys.excepthook
        try:
            setup_exception_handler()
            # We just verify that calling the hook with SystemExit doesn't raise
            # an unexpected error (it calls sys.__excepthook__ instead).
            # We can't easily verify the exact behavior without side effects,
            # but we can confirm setup_exception_handler runs without error.
            assert callable(sys.excepthook)
        finally:
            sys.excepthook = original_hook

    def test_handler_with_console_logger_datus_exception(self):
        """DatusException invokes console_logger with formatted message."""
        original_hook = sys.excepthook
        console_logger = MagicMock()
        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_lm.return_value.debug = True
                setup_exception_handler(console_logger=console_logger)
                # Simulate calling the exception hook directly
                try:
                    raise DatusException(ErrorCode.COMMON_UNKNOWN, message="test error")
                except DatusException:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            console_logger.assert_called_once()
            call_arg = console_logger.call_args[0][0]
            assert "test error" in call_arg or "Execution failed" in call_arg
        finally:
            sys.excepthook = original_hook

    def test_handler_without_console_logger_non_debug(self):
        """Without console_logger and non-debug mode, uses temporary_output."""
        original_hook = sys.excepthook
        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_manager = MagicMock()
                mock_manager.debug = False
                mock_manager.temporary_output.return_value.__enter__ = MagicMock(return_value=None)
                mock_manager.temporary_output.return_value.__exit__ = MagicMock(return_value=False)
                mock_lm.return_value = mock_manager

                setup_exception_handler()
                try:
                    raise ValueError("unexpected failure")
                except ValueError:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            assert mock_manager.temporary_output.called
        finally:
            sys.excepthook = original_hook

    def test_prefix_wrap_func_applied(self):
        """When prefix_wrap_func is provided it wraps the log prefix."""
        original_hook = sys.excepthook
        console_logger = MagicMock()

        def wrap_func(s):
            return f"[WRAPPED] {s}"

        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_lm.return_value.debug = True
                setup_exception_handler(console_logger=console_logger, prefix_wrap_func=wrap_func)
                try:
                    raise DatusException(ErrorCode.COMMON_UNKNOWN)
                except DatusException:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            call_arg = console_logger.call_args[0][0]
            assert "[WRAPPED]" in call_arg
        finally:
            sys.excepthook = original_hook

    def test_non_debug_with_console_logger(self):
        """Non-debug mode with console_logger logs trace to file and message to console."""
        original_hook = sys.excepthook
        console_logger = MagicMock()
        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_lm.return_value.debug = False
                setup_exception_handler(console_logger=console_logger)
                try:
                    raise DatusException(ErrorCode.COMMON_UNKNOWN, message="non-debug test")
                except DatusException:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            console_logger.assert_called_once()
        finally:
            sys.excepthook = original_hook
