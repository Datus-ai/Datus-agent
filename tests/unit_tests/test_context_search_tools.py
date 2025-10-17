"""Tests for ContextSearchTools."""

from unittest.mock import Mock, patch

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.context_search import ContextSearchTools
from datus.tools.tools import FuncToolResult


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    config = Mock(spec=AgentConfig)
    config.rag_storage_path.return_value = "/tmp/test_rag_storage"
    return config


@pytest.fixture
def context_tools_env(mock_agent_config):
    metric_rag = Mock()
    metric_rag.search_all_metrics.return_value = [
        {"domain": "Sales", "layer1": "Revenue", "layer2": "Monthly", "name": "monthly_sales"},
        {"domain": "Sales", "layer1": "Revenue", "layer2": "Quarterly", "name": "quarterly_sales"},
    ]
    metric_rag.search_metrics.return_value = [{"name": "monthly_sales", "sql_query": "SELECT 1"}]
    metric_rag.get_metrics_size.return_value = 2

    sql_rag = Mock()
    sql_rag.search_all_sql_history.return_value = [
        {"domain": "Sales", "layer1": "Revenue", "layer2": "Monthly", "name": "sales_query"},
        {"domain": "Support", "layer1": "Tickets", "layer2": "Escalations", "name": "support_query"},
    ]
    sql_rag.search_sql_history_by_summary.return_value = [{"name": "sales_query", "sql": "SELECT * FROM sales"}]
    sql_rag.get_sql_history_size.return_value = 2

    with patch("datus.tools.context_search.SemanticMetricsRAG", return_value=metric_rag), patch(
        "datus.tools.context_search.SqlHistoryRAG", return_value=sql_rag
    ):
        tools = ContextSearchTools(mock_agent_config)
        yield {"tools": tools, "metric_rag": metric_rag, "sql_rag": sql_rag}


def test_available_tools(context_tools_env):
    tools = context_tools_env["tools"].available_tools()
    tool_names = {tool.name for tool in tools}
    expected = {
        "list_domain_layers_tree",
        "search_metrics",
        "search_historical_sql",
    }
    assert expected.issubset(tool_names)


def test_list_domain_layers_tree_combined(context_tools_env):
    result = context_tools_env["tools"].list_domain_layers_tree()
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


def test_search_metrics_passes_filters(context_tools_env):
    tools = context_tools_env["tools"]
    metric_rag = context_tools_env["metric_rag"]

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


def test_search_historical_sql(context_tools_env):
    tools = context_tools_env["tools"]
    sql_rag = context_tools_env["sql_rag"]

    result = tools.search_reference_sql("sales report", domain="Sales", layer1="Revenue", top_n=2)
    assert result.success == 1
    sql_rag.search_sql_history_by_summary.assert_called_once_with(
        query_text="sales report", domain="Sales", layer1="Revenue", layer2="", top_n=2
    )
