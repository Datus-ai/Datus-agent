"""
Test cases for DBFuncTool compressor model_name initialization and execute_ddl.
"""

from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool


class TestDBFuncToolCompressorModelName:
    """Verify that DBFuncTool uses agent_config's model name for DataCompressor."""

    def test_compressor_uses_agent_config_model(self):
        """When agent_config is provided, compressor should use its active model name."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        mock_config = Mock()
        mock_config.active_model.return_value.model = "claude-sonnet-4"

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_connector, agent_config=mock_config)

        assert tool.compressor.model_name == "claude-sonnet-4"

    def test_compressor_defaults_without_agent_config(self):
        """When agent_config is None, compressor should fall back to gpt-3.5-turbo."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG"),
            patch("datus.tools.func_tool.database.SemanticModelRAG"),
        ):
            tool = DBFuncTool(mock_connector)

        assert tool.compressor.model_name == "gpt-3.5-turbo"


class TestDBFuncToolExecuteDDL:
    """Tests for DBFuncTool.execute_ddl method."""

    def _make_tool(self, connector):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector)

    def test_execute_ddl_success(self):
        """Test successful DDL execution."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock()
        ddl_result.success = True
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 1
        assert result.result["message"] == "DDL executed successfully"
        assert result.result["sql"] == "CREATE TABLE test (id INT)"

    def test_execute_ddl_failure(self):
        """Test DDL execution returning error."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock()
        ddl_result.success = False
        ddl_result.error = "table already exists"
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 0
        assert "table already exists" in result.error

    def test_execute_ddl_unsupported(self):
        """Test DDL on connector without execute_ddl support."""
        mock_connector = Mock(spec=[])  # No attributes at all
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases = Mock(return_value=[])

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 0
        assert "does not support DDL" in result.error

    def test_execute_ddl_not_in_available_tools(self):
        """Verify that execute_ddl is NOT in the default available_tools() list."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        tool = self._make_tool(mock_connector)
        tool_names = [t.name for t in tool.available_tools()]

        assert "execute_ddl" not in tool_names

    def test_execute_ddl_exception_handling(self):
        """Test DDL execution when connector raises an exception."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_ddl.side_effect = RuntimeError("connection lost")

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 0
        assert "connection lost" in result.error


class TestExecuteDDLStatementValidation:
    """Tests for execute_ddl SQL statement type validation."""

    def _make_tool(self, connector=None):
        if connector is None:
            connector = Mock()
            connector.dialect = "sqlite"
            connector.get_databases.return_value = []
            ddl_result = Mock()
            ddl_result.success = True
            connector.execute_ddl.return_value = ddl_result
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector)

    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE TABLE test (id INT)",
            "CREATE TABLE IF NOT EXISTS test (id INT)",
            "CREATE TABLE test AS SELECT * FROM other",
            "  CREATE TABLE test (id INT)",
            "ALTER TABLE test ADD COLUMN name TEXT",
            "DROP TABLE test",
            "DROP TABLE IF EXISTS test",
            "CREATE VIEW v AS SELECT 1",
            "DROP VIEW v",
            "CREATE OR REPLACE VIEW v AS SELECT 1",
            "CREATE TEMPORARY TABLE tmp AS SELECT 1",
            "CREATE TEMP TABLE tmp (id INT)",
        ],
    )
    def test_allowed_ddl_statements(self, sql):
        """Allowed DDL statement types should pass validation."""
        tool = self._make_tool()
        result = tool.execute_ddl(sql)
        assert result.success == 1

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM users",
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE users SET name='x'",
            "DELETE FROM users",
            "TRUNCATE TABLE users",
            "GRANT ALL ON users TO public",
            "CREATE OR REPLACE FUNCTION test() RETURNS void",
            "CREATE PROCEDURE test() BEGIN END",
        ],
    )
    def test_rejected_non_ddl_statements(self, sql):
        """Non-DDL statements should be rejected."""
        tool = self._make_tool()
        result = tool.execute_ddl(sql)
        assert result.success == 0
        assert "Only DDL statements are allowed" in result.error

    def test_rejected_multi_statement(self):
        """Multi-statement SQL should be rejected."""
        tool = self._make_tool()
        result = tool.execute_ddl("CREATE TABLE t1 (id INT); DROP TABLE users")
        assert result.success == 0
        assert "Multi-statement" in result.error

    def test_rejected_empty_sql(self):
        """Empty SQL should be rejected."""
        tool = self._make_tool()
        result = tool.execute_ddl("   ")
        assert result.success == 0
        assert "Empty SQL" in result.error

    def test_sql_comments_stripped(self):
        """SQL comments should be stripped before validation."""
        tool = self._make_tool()
        result = tool.execute_ddl("-- comment\nCREATE TABLE test (id INT)")
        assert result.success == 1
