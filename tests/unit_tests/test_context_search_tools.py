"""Tests for ContextSearchTools."""

from unittest.mock import Mock, patch

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool import ContextSearchTools
from datus.tools.func_tool.base import FuncToolResult

METRIC_ENTRIES = [
    {"subject_path": ["Sales", "Revenue", "Monthly"], "name": "monthly_sales"},
    {"subject_path": ["Sales", "Revenue", "Quarterly"], "name": "quarterly_sales"},
]

SQL_ENTRIES = [
    {"subject_path": ["Sales", "Revenue", "Monthly"], "name": "sales_query"},
    {"subject_path": ["Support", "Tickets", "Escalations"], "name": "support_query"},
]


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    config = Mock(spec=AgentConfig)
    config.rag_storage_path.return_value = "/tmp/test_rag_storage"
    config.sub_agent_config.return_value = None
    return config


def _build_tree_structure(entries: list) -> dict:
    """Build a tree structure from subject path entries.
    Returns format: {"name": {"node_id": id, "children": {...}}}
    """
    tree = {}
    for entry in entries:
        current = tree
        for part in entry["subject_path"]:
            if part not in current:
                current[part] = {"node_id": None, "children": {}}
            current = current[part]["children"]
    return tree


@pytest.fixture
def build_context_tools(mock_agent_config):
    def _builder(metric_cfg=None, sql_cfg=None):
        metric_cfg = metric_cfg or {}
        sql_cfg = sql_cfg or {}

        # Create mock SubjectTreeStore
        mock_subject_tree = Mock()
        mock_subject_tree.find_or_create_path = Mock()

        metric_rag = Mock()
        metric_entries = metric_cfg.get("entries", [])
        metric_rag.search_all_metrics.return_value = metric_entries
        metric_rag.search_metrics.return_value = metric_cfg.get("search_return", [])
        metric_rag.get_metrics_size.return_value = metric_cfg.get("size", len(metric_entries))
        if "search_all_side_effect" in metric_cfg:
            metric_rag.search_all_metrics.side_effect = metric_cfg["search_all_side_effect"]
        if "search_metrics_side_effect" in metric_cfg:
            metric_rag.search_metrics.side_effect = metric_cfg["search_metrics_side_effect"]

        sql_rag = Mock()
        sql_entries = sql_cfg.get("entries", [])
        sql_rag.search_all_reference_sql.return_value = sql_entries
        sql_rag.search_reference_sql.return_value = sql_cfg.get("search_return", [])
        sql_rag.get_reference_sql_size.return_value = sql_cfg.get("size", len(sql_entries))
        if "search_all_side_effect" in sql_cfg:
            sql_rag.search_all_reference_sql.side_effect = sql_cfg["search_all_side_effect"]
        if "search_sql_side_effect" in sql_cfg:
            sql_rag.search_reference_sql.side_effect = sql_cfg["search_sql_side_effect"]

        # Set up get_tree_structure to return tree from all entries
        all_entries = metric_entries + sql_entries
        mock_subject_tree.get_tree_structure.return_value = _build_tree_structure(all_entries)

        with (
            patch("datus.tools.func_tool.context_search.MetricRAG", return_value=metric_rag),
            patch("datus.tools.func_tool.context_search.SemanticModelRAG", return_value=Mock()),
            patch("datus.tools.func_tool.context_search.ReferenceSqlRAG", return_value=sql_rag),
            patch("datus.tools.func_tool.context_search.SubjectTreeStore", return_value=mock_subject_tree),
        ):
            tools = ContextSearchTools(mock_agent_config)
        return tools, metric_rag, sql_rag, mock_subject_tree

    return _builder


def test_available_tools_with_metrics_and_sql(build_context_tools):
    tools, _, _, _ = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES, "search_return": [{"name": "monthly_sales"}]},
        sql_cfg={"entries": SQL_ENTRIES, "search_return": [{"name": "sales_query"}]},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_subject_tree", "search_metrics", "search_reference_sql"}


def test_available_tools_metrics_only(build_context_tools):
    tools, _, _, _ = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES, "search_return": [{"name": "monthly_sales"}]},
        sql_cfg={"entries": [], "size": 0},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_subject_tree", "search_metrics"}


def test_available_tools_sql_only(build_context_tools):
    tools, _, _, _ = build_context_tools(
        metric_cfg={"entries": [], "size": 0},
        sql_cfg={"entries": SQL_ENTRIES, "search_return": [{"name": "sales_query"}]},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_subject_tree", "search_reference_sql"}


def test_list_domain_layers_tree_combined(build_context_tools):
    tools, _, _, _ = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES},
        sql_cfg={"entries": SQL_ENTRIES},
    )

    result = tools.list_subject_tree()
    assert isinstance(result, FuncToolResult)
    assert result.success == 1
    assert result.result == {
        "Sales": {
            "Revenue": {
                "Monthly": {},
                "Quarterly": {},
            }
        },
        "Support": {"Tickets": {"Escalations": {}}},
    }


def test_collect_metrics_entries_handles_exception(build_context_tools):
    # Set size > 0 so that _show_metrics() returns True and the method is called
    tools, metric_rag, _, _ = build_context_tools(
        metric_cfg={"entries": [], "size": 1, "search_all_side_effect": RuntimeError("metrics offline")}
    )

    entries = tools._collect_metrics_entries()
    assert entries == []
    metric_rag.search_all_metrics.assert_called_once()


def test_collect_sql_entries_handles_exception(build_context_tools):
    # Set size > 0 so that _show_sql() returns True and the method is called
    tools, _, sql_rag, _ = build_context_tools(
        sql_cfg={"entries": [], "size": 1, "search_all_side_effect": RuntimeError("sql offline")}
    )

    entries = tools._collect_sql_entries()
    assert entries == []
    sql_rag.search_all_reference_sql.assert_called_once()


def test_search_metrics_passes_filters(build_context_tools):
    tools, metric_rag, _, _ = build_context_tools(
        metric_cfg={
            "entries": METRIC_ENTRIES,
            "search_return": [{"name": "monthly_sales"}],
        }
    )

    result = tools.search_metrics(
        query_text="revenue",
        subject_path=["Sales", "Revenue", "Monthly"],
        top_n=3,
    )

    assert result.success == 1
    metric_rag.search_metrics.assert_called_once_with(
        query_text="revenue",
        subject_path=["Sales", "Revenue", "Monthly"],
        top_n=3,
    )


def test_search_metrics_handles_failure(build_context_tools):
    tools, metric_rag, _, _ = build_context_tools(
        metric_cfg={
            "entries": METRIC_ENTRIES,
            "search_metrics_side_effect": Exception("metric search failed"),
        }
    )

    result = tools.search_metrics("revenue")
    assert result.success == 0
    assert "metric search failed" in (result.error or "")
    metric_rag.search_metrics.assert_called_once()


def test_search_historical_sql(build_context_tools):
    tools, _, sql_rag, _ = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES},
        sql_cfg={
            "entries": SQL_ENTRIES,
            "search_return": [{"name": "sales_query", "sql": "SELECT * FROM sales"}],
        },
    )

    result = tools.search_reference_sql("sales report", subject_path=["Sales", "Revenue"], top_n=2)
    assert result.success == 1
    sql_rag.search_reference_sql.assert_called_once_with(
        query_text="sales report",
        subject_path=["Sales", "Revenue"],
        top_n=2,
        selected_fields=["name", "sql", "comment", "summary", "tags"],
    )


def test_search_historical_sql_handles_failure(build_context_tools):
    tools, _, sql_rag, _ = build_context_tools(
        sql_cfg={
            "entries": SQL_ENTRIES,
            "search_sql_side_effect": Exception("sql search failed"),
        }
    )

    result = tools.search_reference_sql("sales report")
    assert result.success == 0
    assert "sql search failed" in (result.error or "")
    sql_rag.search_reference_sql.assert_called_once()
