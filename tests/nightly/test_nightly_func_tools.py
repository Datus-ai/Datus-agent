"""
Nightly integration tests for function tools (N11).

Tests cover DB tools (read_query failure, scoped tables)
and Context tools (metrics, reference SQL, semantic objects) with real configurations.
"""

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import DBFuncTool, db_function_tool_instance_multi
from tests.conftest import load_acceptance_config


@pytest.mark.nightly
class TestNightlyDBFuncToolErrors:
    """N11-07: read_query failure scenarios with real SSB SQLite database."""

    @pytest.fixture
    def ssb_config(self):
        return load_acceptance_config(namespace="ssb_sqlite", home="tests")

    @pytest.fixture
    def ssb_db_tool(self, ssb_config):
        return db_function_tool_instance_multi(ssb_config)

    def test_read_query_nonexistent_table(self, ssb_db_tool):
        """N11-07a: read_query with nonexistent table returns meaningful error."""
        result = ssb_db_tool.read_query("SELECT * FROM nonexistent_xyz_table")

        assert result.success == 0, "Should fail for nonexistent table"
        assert result.error is not None, "Error message should not be None"
        assert len(result.error) > 10, f"Error message should be descriptive, got: {result.error}"

    def test_read_query_invalid_sql_syntax(self, ssb_db_tool):
        """N11-07b: read_query with completely invalid SQL returns error."""
        result = ssb_db_tool.read_query("COMPLETELY INVALID SQL STATEMENT")

        assert result.success == 0, "Should fail for invalid SQL"
        assert result.error is not None, "Error message should not be None"
        assert len(result.error) > 0, "Error message should not be empty"


@pytest.mark.nightly
class TestNightlyScopedTables:
    """N11-09: Scoped tables filtering with real SSB SQLite database."""

    @pytest.fixture
    def ssb_config(self):
        return load_acceptance_config(namespace="ssb_sqlite", home="tests")

    @pytest.fixture
    def scoped_db_tool(self, ssb_config):
        """Create DBFuncTool with scoped_tables limited to customer and lineorder."""
        db_manager = db_manager_instance(ssb_config.namespaces)
        return DBFuncTool(
            db_manager,
            agent_config=ssb_config,
            default_database=ssb_config.current_database,
            scoped_tables=["customer", "lineorder"],
        )

    def test_list_tables_respects_scope(self, scoped_db_tool):
        """N11-09a: list_tables only returns tables within scoped_tables."""
        result = scoped_db_tool.list_tables()

        assert result.success == 1, f"list_tables should succeed, got error: {result.error}"
        table_names = [t["name"] for t in result.result]

        assert "customer" in table_names, "customer should be in scoped results"
        assert "lineorder" in table_names, "lineorder should be in scoped results"
        assert "supplier" not in table_names, "supplier should be filtered out by scope"
        assert "part" not in table_names, "part should be filtered out by scope"
        assert "date" not in table_names, "date should be filtered out by scope"

    def test_describe_table_blocked_by_scope(self, ssb_config):
        """N11-09b: describe_table rejects tables outside scoped_tables."""
        db_manager = db_manager_instance(ssb_config.namespaces)
        scoped_tool = DBFuncTool(
            db_manager,
            agent_config=ssb_config,
            default_database=ssb_config.current_database,
            scoped_tables=["customer"],
        )

        # Allowed table should work
        allowed_result = scoped_tool.describe_table("customer")
        assert (
            allowed_result.success == 1
        ), f"describe_table for scoped table should succeed, got: {allowed_result.error}"
        assert "columns" in allowed_result.result, "Should have columns in result"

        # Blocked table should fail
        blocked_result = scoped_tool.describe_table("supplier")
        assert blocked_result.success == 0, "describe_table for out-of-scope table should fail"
        assert blocked_result.error is not None, "Should have error message"


@pytest.mark.nightly
class TestNightlySearchTable:
    """N11-08: search_table RAG functionality."""

    def test_search_table_available_tools(self, agent_config: AgentConfig):
        """N11-08: Verify search_table presence in available_tools depends on has_schema."""
        db_tool = db_function_tool_instance_multi(agent_config)

        tools = db_tool.available_tools()
        tool_names = {tool.name for tool in tools}

        # search_table should be in available_tools only if schema RAG exists
        if hasattr(db_tool, "has_schema") and db_tool.has_schema:
            assert "search_table" in tool_names, "search_table should be available when schema RAG exists"
        else:
            assert "search_table" not in tool_names, "search_table should not be available without schema RAG"


@pytest.mark.nightly
class TestNightlyContextSearchTools:
    """N11-13 to N11-16: ContextSearchTools with bird_school configuration."""

    @pytest.fixture
    def ctx_tools(self, agent_config: AgentConfig):
        return ContextSearchTools(agent_config)

    def test_search_metrics(self, ctx_tools):
        """N11-13: search_metrics returns structured results."""
        assert ctx_tools.has_metrics is True, "bird_school should have metrics data"

        result = ctx_tools.search_metrics("school")

        assert result.success == 1, f"search_metrics should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) > 0, "Should find at least one metric matching 'school'"

        # Verify result structure
        first = result.result[0]
        assert "name" in first, "Each metric should have a 'name' field"

    def test_get_metrics(self, ctx_tools):
        """N11-13b: get_metrics retrieves specific metric details."""
        assert ctx_tools.has_metrics is True, "bird_school should have metrics data"

        # First search to get a valid subject_path and name
        search_result = ctx_tools.search_metrics("school")
        assert search_result.success == 1 and len(search_result.result) > 0, "Need search results to test get_metrics"

        first = search_result.result[0]
        subject_path = first.get("subject_path", [])
        name = first.get("name", "")

        assert subject_path and name, f"Search result should have subject_path and name, got: {first}"

        get_result = ctx_tools.get_metrics(subject_path=subject_path, name=name)

        assert get_result.success == 1, f"get_metrics should succeed, got error: {get_result.error}"
        assert get_result.result is not None, "Should return metric details"

    def test_search_reference_sql(self, ctx_tools):
        """N11-14: search_reference_sql returns list of SQL queries."""
        assert ctx_tools.has_reference_sql is True, "bird_school should have reference SQL data"

        result = ctx_tools.search_reference_sql("school")

        assert result.success == 1, f"search_reference_sql should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) > 0, "Should find at least one reference SQL matching 'school'"

        # Verify structure
        first = result.result[0]
        assert (
            "name" in first or "sql" in first
        ), f"Each result should have 'name' or 'sql', got keys: {list(first.keys())}"

    def test_get_reference_sql(self, ctx_tools):
        """N11-15: get_reference_sql retrieves specific SQL details."""
        assert ctx_tools.has_reference_sql is True, "bird_school should have reference SQL data"

        search_result = ctx_tools.search_reference_sql("school")
        assert (
            search_result.success == 1 and len(search_result.result) > 0
        ), "Need search results to test get_reference_sql"

        first = search_result.result[0]
        subject_path = first.get("subject_path", [])
        name = first.get("name", "")

        assert subject_path and name, f"Search result should have subject_path and name, got: {first}"

        get_result = ctx_tools.get_reference_sql(subject_path=subject_path, name=name)

        assert get_result.success == 1, f"get_reference_sql should succeed, got error: {get_result.error}"
        assert get_result.result is not None, "Should return SQL details"

    def test_search_semantic_objects_availability(self, ctx_tools):
        """N11-16: Verify search_semantic_objects availability and behavior."""
        # Test that the flag is a boolean
        assert isinstance(ctx_tools.has_semantic_objects, bool), "has_semantic_objects should be a boolean"

        if ctx_tools.has_semantic_objects:
            result = ctx_tools.search_semantic_objects("school")
            assert result.success == 1, f"search_semantic_objects should succeed when data exists, got: {result.error}"
            assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        else:
            # Verify it's correctly not in available tools
            tool_names = {t.name for t in ctx_tools.available_tools()}
            assert (
                "search_semantic_objects" not in tool_names
            ), "search_semantic_objects should not be available without data"
