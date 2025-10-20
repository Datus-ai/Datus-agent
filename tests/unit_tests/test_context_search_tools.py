"""Tests for ContextSearchTools."""

from unittest.mock import Mock, patch

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.context_search import ContextSearchTools
from datus.tools.tools import FuncToolResult

METRIC_ENTRIES = [
    {"domain": "Sales", "layer1": "Revenue", "layer2": "Monthly", "name": "monthly_sales"},
    {"domain": "Sales", "layer1": "Revenue", "layer2": "Quarterly", "name": "quarterly_sales"},
]

SQL_ENTRIES = [
    {"domain": "Sales", "layer1": "Revenue", "layer2": "Monthly", "name": "sales_query"},
    {"domain": "Support", "layer1": "Tickets", "layer2": "Escalations", "name": "support_query"},
]


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    config = Mock(spec=AgentConfig)
    config.rag_storage_path.return_value = "/tmp/test_rag_storage"
    return config


@pytest.fixture
def build_context_tools(mock_agent_config):
    def _builder(metric_cfg=None, sql_cfg=None):
        metric_cfg = metric_cfg or {}
        sql_cfg = sql_cfg or {}

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
        sql_rag.search_all_sql_history.return_value = sql_entries
        sql_rag.search_sql_history_by_summary.return_value = sql_cfg.get("search_return", [])
        sql_rag.get_sql_history_size.return_value = sql_cfg.get("size", len(sql_entries))
        if "search_all_side_effect" in sql_cfg:
            sql_rag.search_all_sql_history.side_effect = sql_cfg["search_all_side_effect"]
        if "search_sql_side_effect" in sql_cfg:
            sql_rag.search_sql_history_by_summary.side_effect = sql_cfg["search_sql_side_effect"]

        with patch("datus.tools.context_search.SemanticMetricsRAG", return_value=metric_rag), patch(
            "datus.tools.context_search.SqlHistoryRAG", return_value=sql_rag
        ):
            tools = ContextSearchTools(mock_agent_config)
        return tools, metric_rag, sql_rag

    return _builder


def test_available_tools_with_metrics_and_sql(build_context_tools):
    tools, _, _ = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES, "search_return": [{"name": "monthly_sales"}]},
        sql_cfg={"entries": SQL_ENTRIES, "search_return": [{"name": "sales_query"}]},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_domain_layers_tree", "search_metrics", "search_reference_sql"}


def test_available_tools_metrics_only(build_context_tools):
    tools, _, _ = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES, "search_return": [{"name": "monthly_sales"}]},
        sql_cfg={"entries": [], "size": 0},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_domain_layers_tree", "search_metrics"}


def test_available_tools_sql_only(build_context_tools):
    tools, _, _ = build_context_tools(
        metric_cfg={"entries": [], "size": 0},
        sql_cfg={"entries": SQL_ENTRIES, "search_return": [{"name": "sales_query"}]},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_domain_layers_tree", "search_reference_sql"}


def test_list_domain_layers_tree_combined(build_context_tools):
    tools, _, _ = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES},
        sql_cfg={"entries": SQL_ENTRIES},
    )

    result = tools.list_domain_layers_tree()
    assert isinstance(result, FuncToolResult)
    assert result.success == 1
    assert result.result == {
        "Sales": {
            "Revenue": {
                "Monthly": {"metrics_size": 1, "sql_size": 1},
                "Quarterly": {"metrics_size": 1},
            }
        },
        "Support": {"Tickets": {"Escalations": {"sql_size": 1}}},
    }


def test_list_domain_layers_tree_handles_blank_taxonomy(build_context_tools):
    tools, _, _ = build_context_tools(
        metric_cfg={
            "entries": [
                {"domain": None, "layer1": "  ", "layer2": None, "name": "undefined_metric"},
            ]
        },
        sql_cfg={
            "entries": [
                {"domain": "", "layer1": None, "layer2": "Ops", "name": "ops_sql"},
            ]
        },
    )

    result = tools.list_domain_layers_tree()
    assert result.success == 1
    assert result.result == {"": {"": {"": {"metrics_size": 1}, "Ops": {"sql_size": 1}}}}


def test_list_domain_layers_tree_value_error(build_context_tools):
    tools, _, _ = build_context_tools(metric_cfg={"entries": []}, sql_cfg={"entries": []})

    with patch.object(tools, "_collect_metrics_entries", side_effect=ValueError("taxonomy failure")), patch.object(
        tools, "_collect_sql_entries", return_value=[]
    ):
        result = tools.list_domain_layers_tree()

    assert result.success == 0
    assert "taxonomy failure" in (result.error or "")


def test_collect_metrics_entries_handles_exception(build_context_tools):
    tools, metric_rag, _ = build_context_tools(
        metric_cfg={"entries": [], "search_all_side_effect": RuntimeError("metrics offline")}
    )

    entries = tools._collect_metrics_entries()
    assert entries == []
    metric_rag.search_all_metrics.assert_called_once()


def test_collect_sql_entries_handles_exception(build_context_tools):
    tools, _, sql_rag = build_context_tools(
        sql_cfg={"entries": [], "search_all_side_effect": RuntimeError("sql offline")}
    )

    entries = tools._collect_sql_entries()
    assert entries == []
    sql_rag.search_all_sql_history.assert_called_once()


def test_search_metrics_passes_filters(build_context_tools):
    tools, metric_rag, _ = build_context_tools(
        metric_cfg={
            "entries": METRIC_ENTRIES,
            "search_return": [{"name": "monthly_sales"}],
        }
    )

    result = tools.search_metrics(
        query_text="revenue",
        domain="Sales",
        layer1="Revenue",
        layer2="Monthly",
        top_n=3,
    )

    assert result.success == 1
    metric_rag.search_metrics.assert_called_once_with(
        query_text="revenue",
        domain="Sales",
        layer1="Revenue",
        layer2="Monthly",
        top_n=3,
    )


def test_search_metrics_handles_failure(build_context_tools):
    tools, metric_rag, _ = build_context_tools(
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
    tools, _, sql_rag = build_context_tools(
        metric_cfg={"entries": METRIC_ENTRIES},
        sql_cfg={
            "entries": SQL_ENTRIES,
            "search_return": [{"name": "sales_query", "sql": "SELECT * FROM sales"}],
        },
    )

    result = tools.search_reference_sql("sales report", domain="Sales", layer1="Revenue", top_n=2)
    assert result.success == 1
    sql_rag.search_sql_history_by_summary.assert_called_once_with(
        query_text="sales report", domain="Sales", layer1="Revenue", layer2="", top_n=2
    )


def test_search_historical_sql_handles_failure(build_context_tools):
    tools, _, sql_rag = build_context_tools(
        sql_cfg={
            "entries": SQL_ENTRIES,
            "search_sql_side_effect": Exception("sql search failed"),
        }
    )

    result = tools.search_reference_sql("sales report")
    assert result.success == 0
    assert "sql search failed" in (result.error or "")
    sql_rag.search_sql_history_by_summary.assert_called_once()
