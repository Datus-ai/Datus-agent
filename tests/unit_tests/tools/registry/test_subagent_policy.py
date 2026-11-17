"""Type policies intersect registered capabilities without conflating API access."""

import pytest

from datus.tools.registry.capabilities import TOOL_CAPABILITIES, tool_capability
from datus.tools.registry.contracts import ToolCapability
from datus.tools.registry.subagent_policy import builtin_tools, resolve_tools, tool_reference


def test_builder_and_explorer_share_read_only_semantic_tools():
    explorer = {tool for tool in builtin_tools("analytics_explorer") if tool.startswith("semantic_tools.")}
    builder = {tool for tool in builtin_tools("analytics_builder") if tool.startswith("semantic_tools.")}
    assert builder == explorer
    assert builder
    assert all(tool_capability(tool).access == "read" for tool in builder)
    assert "semantic_tools.validate_semantic" not in builder
    assert "filesystem_tools.write_file" in builtin_tools("analytics_builder")
    assert "filesystem_tools.write_file" not in builtin_tools("analytics_explorer")


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder", "gen_sql", "ask_metrics"])
def test_semantic_catalog_matches_the_node_provider(kind):
    from datus.tools.func_tool.semantic_tools import SemanticTools

    available = set(SemanticTools.all_tools_name())
    if kind.startswith("analytics_"):
        available.add("get_query_metrics_result")
    catalog = set(tool_reference(kind)["tool_types"]["semantic_tools"]["tools"])
    assert catalog <= available
    assert "query_metrics" in catalog


def test_legacy_sql_keeps_the_full_semantic_provider():
    from datus.tools.func_tool.semantic_tools import SemanticTools

    assert set(tool_reference("gen_sql")["tool_types"]["semantic_tools"]["tools"]) == set(
        SemanticTools.all_tools_name()
    )


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
def test_unknown_or_changed_contract_does_not_gain_read_access(monkeypatch, kind):
    monkeypatch.setitem(TOOL_CAPABILITIES, "semantic_tools.query_metrics", ToolCapability("write"))
    monkeypatch.delitem(TOOL_CAPABILITIES, "semantic_tools.get_metric")
    candidates = ["semantic_tools.query_metrics", "semantic_tools.get_metric", "semantic_tools.get_dataset"]
    assert resolve_tools(kind, candidates, configured="semantic_tools.*") == ["semantic_tools.get_dataset"]
    catalog = tool_reference(kind)["tool_types"]["semantic_tools"]["tools"]
    assert "query_metrics" not in catalog
    assert "get_metric" not in catalog
    assert "get_dataset" in catalog


def test_direct_call_eligibility_does_not_gate_agent_tools():
    tool = "semantic_tools.validate_semantic"
    assert tool_capability(tool).direct_call is False
    assert resolve_tools("gen_sql", [tool]) == [tool]
    assert resolve_tools("analytics_builder", [tool]) == []


@pytest.mark.parametrize(
    "configured", ["filesystem_tools.read_file", [" filesystem_tools.read_file ", ""], ("filesystem_tools.read_file",)]
)
def test_provider_and_host_scope_both_restrict_the_type(configured):
    available = ["filesystem_tools.read_file", "filesystem_tools.write_file", "filesystem_tools.delete_file"]
    assert resolve_tools("analytics_builder", available, configured=configured) == ["filesystem_tools.read_file"]
    assert resolve_tools("analytics_builder", [], configured=configured) == []
    assert resolve_tools("analytics_builder", available, configured=" ") == []


def test_legacy_fast_types_keep_defaults_and_gain_file_tools():
    from datus.api.services.agent_service import SUBAGENT_TOOL_REFERENCE

    for kind in ("gen_sql", "ask_metrics"):
        tools = set(builtin_tools(kind))
        assert set(SUBAGENT_TOOL_REFERENCE[kind]["default_tools"]) <= tools
        assert {f"filesystem_tools.{method}" for method in ("read_file", "write_file", "edit_file")} <= tools


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
def test_analytics_context_catalog_is_explicit_and_read_only(monkeypatch, kind):
    expected = {
        "list_subject_tree",
        "search_metrics",
        "search_reference_sql",
        "get_reference_sql",
        "search_semantic_objects",
    }
    reference = tool_reference(kind)
    assert set(reference["tool_types"]["context_search_tools"]["tools"]) == expected
    selected = {f"context_search_tools.{name}" for name in expected}
    assert selected <= set(reference["default_tools"])
    assert all(tool_capability(tool).access == "read" for tool in selected)
    future = "context_search_tools.future_reader"
    monkeypatch.setitem(TOOL_CAPABILITIES, future, ToolCapability("read"))
    assert set(resolve_tools(kind, [*selected, future], configured="context_search_tools.*")) == selected
