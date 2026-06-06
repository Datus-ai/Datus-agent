import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.context_search import ContextSearchTools

pytestmark = pytest.mark.nightly


class TestContextSearchTools:
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
        assert "name" in first or "sql" in first, (
            f"Each result should have 'name' or 'sql', got keys: {list(first.keys())}"
        )

    def test_get_reference_sql(self, ctx_tools):
        """N11-15: get_reference_sql retrieves specific SQL details."""
        assert ctx_tools.has_reference_sql is True, "bird_school should have reference SQL data"

        search_result = ctx_tools.search_reference_sql("school")
        assert search_result.success == 1 and len(search_result.result) > 0, (
            "Need search results to test get_reference_sql"
        )

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
            assert "search_semantic_objects" not in tool_names, (
                "search_semantic_objects should not be available without data"
            )
