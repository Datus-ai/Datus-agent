# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from unittest.mock import MagicMock, Mock, patch

import pytest

from datus.configuration.node_type import NodeType
from datus.tools.func_tool.base import FuncToolResult


def _fake_function_tool(method):
    tool = Mock()
    tool.name = getattr(method, "__name__", None) or getattr(method, "_mock_name", "tool")
    return tool


def _semantic_tools(adapter=True):
    tools = Mock()
    tools.adapter = Mock() if adapter else None
    tools._adapter_unavailable_message.return_value = "semantic adapter missing"
    for name in ("list_metrics", "get_dimensions", "query_metrics", "validate_semantic", "attribution_analyze"):
        setattr(tools, name, Mock(name=name))
    return tools


def _context_tools(tree):
    tools = Mock()
    tools.list_subject_tree.return_value = FuncToolResult(result=tree)
    for name in ("search_metrics", "get_metrics"):
        setattr(tools, name, Mock(name=name))
    return tools


def _make_node(real_agent_config, *, tree, adapter=True, node_config=None, node_name="ask_metrics", input_data=None):
    from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

    if node_config is not None:
        real_agent_config.agentic_nodes[node_name] = node_config

    semantic_tools = _semantic_tools(adapter=adapter)
    context_tools = _context_tools(tree)
    with (
        patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", return_value=semantic_tools),
        patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=context_tools),
        patch("datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool", side_effect=_fake_function_tool),
    ):
        node = AskMetricsAgenticNode(
            node_id="ask_metrics_test",
            description="Ask metrics",
            node_type=NodeType.TYPE_ASK_METRICS,
            input_data=input_data,
            agent_config=real_agent_config,
            node_name=node_name,
        )
    return node, semantic_tools, context_tools


class TestAskMetricsAgenticNode:
    def test_small_subject_tree_in_prompt_and_no_list_subject_tree_tool(self, real_agent_config, mock_llm_create):
        tree = {
            "Sales": {
                "Orders": {
                    "metrics": ["order_count", "revenue"],
                    "reference_sql": ["orders_revenue.sql"],
                    "knowledge": ["orders business rules"],
                    "reference_template": ["orders_template"],
                }
            }
        }

        node, _, _ = _make_node(real_agent_config, tree=tree)

        tool_names = [tool.name for tool in node.tools]
        assert tool_names == [
            "search_metrics",
            "get_metrics",
            "list_metrics",
            "get_dimensions",
            "query_metrics",
            "attribution_analyze",
        ]
        assert node.subject_tree_mode == "full"
        assert node.subject_tree_metric_entries == [
            {"path": ["Sales", "Orders"], "metrics": ["order_count", "revenue"]}
        ]

        prompt = node._get_system_prompt()
        assert "The complete metric subject entries are available below" in prompt
        assert "order_count" in prompt
        subject_tree_prompt = node.subject_tree_prompt
        assert "order_count" in subject_tree_prompt
        assert "orders_revenue.sql" not in prompt
        assert "orders business rules" not in prompt
        assert "orders_template" not in prompt
        assert "reference_sql" not in subject_tree_prompt
        assert "knowledge" not in subject_tree_prompt
        assert "reference_template" not in subject_tree_prompt
        assert "do not call `list_subject_tree`" in prompt
        assert "do not call `search_metrics` for metrics that match these entries directly" in prompt
        assert "When the subject tree gives a direct metric/path match" in prompt
        assert "`offset_window` metadata" in prompt
        assert 'dimensions=["metric_time__month"]' in prompt
        assert "query the complete metric bundle" in prompt
        assert "Previous-period aliases inside a derived metric are calculation inputs" in prompt
        assert "first period" in prompt
        assert "do not add a `where` filter that enumerates dimension values" in prompt
        assert "Query complete metric results by default" in prompt
        assert "Do not pass `limit` just to preview data" in prompt
        assert "also pass `order_by`" in prompt
        assert "full result is cached" in prompt
        assert node.subject_tree_prompt_limit == 100

    def test_reference_date_is_injected_into_runtime_context(self, real_agent_config, mock_llm_create):
        from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput

        node, _, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count"]}}},
            input_data=AskMetricsNodeInput(
                user_message="How many activities started in June?",
                reference_date="2025-12-31",
            ),
        )

        prompt = node._get_system_prompt()

        assert "Current context:" in prompt
        assert "Current date: 2025-12-31" in prompt
        assert "Available datasources:" not in prompt
        assert "Current sql files root directory:" not in prompt

    def test_large_subject_tree_exposes_list_subject_tree_tool(self, real_agent_config, mock_llm_create):
        tree = {
            "Domain": {
                f"Subject_{idx}": {
                    "metrics": [f"metric_{idx}"],
                    "reference_sql": [f"reference_{idx}.sql"],
                }
                for idx in range(101)
            }
        }

        node, _, _ = _make_node(real_agent_config, tree=tree)

        tool_names = [tool.name for tool in node.tools]
        assert "list_subject_tree" in tool_names
        assert node.subject_tree_mode == "partial"
        assert len(node.subject_tree_metric_entries) == 101

        prompt = node._get_system_prompt()
        assert "The metric subject tree has 101 entries" in prompt
        assert "use the excerpt for direct metric matching" in prompt
        assert "total_entries" in prompt
        assert "reference_0.sql" not in prompt

        result = node.list_subject_tree()
        assert result.success == 1
        assert result.result["total_entries"] == 101
        assert result.result["entries"][0] == {"path": ["Domain", "Subject_0"], "metrics": ["metric_0"]}
        assert "reference_sql" not in result.result["entries"][0]

    def test_subject_tree_entries_without_metrics_are_not_prompted(self, real_agent_config, mock_llm_create):
        tree = {
            "Sales": {
                "OnlyKnowledge": {"reference_sql": ["sales.sql"], "knowledge": ["sales rules"]},
                "Orders": {"metrics": ["order_count"]},
            }
        }

        node, _, _ = _make_node(real_agent_config, tree=tree)

        assert node.subject_tree_mode == "full"
        assert node.subject_tree_metric_entries == [{"path": ["Sales", "Orders"], "metrics": ["order_count"]}]

        prompt = node._get_system_prompt()
        assert "order_count" in prompt
        assert "OnlyKnowledge" not in prompt
        assert "sales.sql" not in prompt
        assert "sales rules" not in prompt

    def test_subject_tree_prompt_limit_is_configurable(self, real_agent_config, mock_llm_create):
        tree = {"Domain": {f"Subject_{idx}": {"metrics": [f"metric_{idx}"]} for idx in range(3)}}

        node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={"subject_tree_prompt_limit": 2},
        )

        tool_names = [tool.name for tool in node.tools]
        assert node.subject_tree_prompt_limit == 2
        assert node.subject_tree_mode == "partial"
        assert "list_subject_tree" in tool_names

        prompt = node._get_system_prompt()
        assert "The first 2 entries are shown below" in prompt
        assert "metric_0" in prompt
        assert "metric_1" in prompt
        assert "metric_2" not in prompt

    def test_final_result_selection_tool_is_opt_in(self, real_agent_config, mock_llm_create):
        tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}

        default_node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_name="default_final_selection_agent",
        )
        strict_node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={"require_final_result_selection": True},
            node_name="strict_final_selection_agent",
        )

        assert "select_final_metric_result" not in [tool.name for tool in default_node.tools]
        assert "select_final_metric_result" not in default_node._get_system_prompt()
        assert "select_final_metric_result" in [tool.name for tool in strict_node.tools]
        assert "select_final_metric_result" in strict_node._get_system_prompt()

    def test_invalid_subject_tree_prompt_limit_uses_default(self, real_agent_config, mock_llm_create):
        tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}

        text_node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={"subject_tree_prompt_limit": "invalid"},
            node_name="custom_metric_agent",
        )
        bool_node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={"subject_tree_prompt_limit": True},
            node_name="another_metric_agent",
        )

        assert text_node.subject_tree_prompt_limit == text_node.SUBJECT_TREE_PROMPT_LIMIT
        assert bool_node.subject_tree_prompt_limit == bool_node.SUBJECT_TREE_PROMPT_LIMIT

    def test_tools_follow_custom_configuration(self, real_agent_config, mock_llm_create):
        tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}

        node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={
                "type": "ask_metrics",
                "tools": "context_search_tools.search_metrics, semantic_tools.query_metrics",
            },
            node_name="custom_metric_agent",
        )

        assert [tool.name for tool in node.tools] == ["search_metrics", "query_metrics"]

    def test_tools_config_accepts_list_and_invalid_config_falls_back(self, real_agent_config, mock_llm_create):
        tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}

        list_node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={"tools": ["semantic_tools.query_metrics", "context_search_tools.search_metrics"]},
            node_name="custom_metric_agent",
        )
        invalid_node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={"tools": {"semantic_tools": ["query_metrics"]}},
            node_name="another_metric_agent",
        )

        assert [tool.name for tool in list_node.tools] == ["query_metrics", "search_metrics"]
        assert [tool.name for tool in invalid_node.tools] == [
            "search_metrics",
            "get_metrics",
            "list_metrics",
            "get_dimensions",
            "query_metrics",
            "attribution_analyze",
        ]

    def test_query_metrics_expands_period_over_period_bundle(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count_mom_delta"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={
                "items": [
                    {"name": "order_count", "metadata": {"metric_kind": "measure_proxy"}},
                    {
                        "name": "previous_month_order_count",
                        "metadata": {
                            "expr": "previous_month_order_count",
                            "inputs": [
                                {
                                    "name": "order_count",
                                    "alias": "previous_month_order_count",
                                    "offset_window": "1 month",
                                }
                            ],
                        },
                    },
                    {
                        "name": "order_count_previous_month",
                        "metadata": {
                            "expr": "order_count_previous_month",
                            "inputs": [
                                {
                                    "name": "order_count",
                                    "alias": "order_count_previous_month",
                                    "offset_window": "1 month",
                                }
                            ],
                        },
                    },
                    {
                        "name": "order_count_mom_delta",
                        "metadata": {
                            "inputs": [
                                {"name": "order_count"},
                                {
                                    "name": "order_count",
                                    "alias": "previous_month_order_count",
                                    "offset_window": "1 month",
                                },
                            ]
                        },
                    },
                ],
                "has_more": False,
            }
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        result = node.query_metrics(
            metrics=["order_count", "order_count_mom_delta"],
            dimensions=["customer_segment", "metric_time__month"],
            time_start="2025-04-01",
            time_end="2025-10-31",
            time_granularity="month",
            order_by=["metric_time__month", "customer_segment"],
        )

        assert result.success == 1
        semantic_tools.query_metrics.assert_called_once_with(
            metrics=[
                "order_count",
                "previous_month_order_count",
                "order_count_previous_month",
                "order_count_mom_delta",
            ],
            dimensions=["customer_segment", "metric_time__month"],
            path=None,
            time_start="2025-04-01",
            time_end="2025-10-31",
            time_granularity="month",
            where=None,
            limit=None,
            order_by=["metric_time__month", "customer_segment"],
            dry_run=False,
        )

    def test_query_metrics_does_not_invent_missing_previous_period_metric(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count_mom_delta"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={
                "items": [
                    {"name": "order_count", "metadata": {}},
                    {
                        "name": "order_count_mom_delta",
                        "metadata": {
                            "inputs": [
                                {"name": "order_count"},
                                {
                                    "name": "order_count",
                                    "alias": "previous_month_order_count",
                                    "offset_window": "1 month",
                                },
                            ]
                        },
                    },
                ],
                "has_more": False,
            }
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(metrics=["order_count_mom_delta"])

        semantic_tools.query_metrics.assert_called_once_with(
            metrics=["order_count", "order_count_mom_delta"],
            dimensions=None,
            path=None,
            time_start=None,
            time_end=None,
            time_granularity=None,
            where=None,
            limit=None,
            order_by=None,
            dry_run=False,
        )

    def test_query_metrics_rejects_missing_metrics_without_raising(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count"]}}},
        )

        result = node.query_metrics()

        assert result.success == 0
        assert "at least one metric name" in result.error
        semantic_tools.query_metrics.assert_not_called()

    def test_query_metrics_does_not_expand_previous_period_metric_alone(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["previous_month_order_count"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={
                "items": [
                    {"name": "order_count", "metadata": {}},
                    {
                        "name": "previous_month_order_count",
                        "metadata": {
                            "inputs": [
                                {
                                    "name": "order_count",
                                    "alias": "previous_month_order_count",
                                    "offset_window": "1 month",
                                }
                            ]
                        },
                    },
                ],
                "has_more": False,
            }
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(metrics=["previous_month_order_count"])

        semantic_tools.query_metrics.assert_called_once_with(
            metrics=["previous_month_order_count"],
            dimensions=None,
            path=None,
            time_start=None,
            time_end=None,
            time_granularity=None,
            where=None,
            limit=None,
            order_by=None,
            dry_run=False,
        )

    def test_query_metrics_does_not_expand_plain_derived_metric(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["delivery_activity_ratio"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={
                "items": [
                    {"name": "order_count", "metadata": {}},
                    {"name": "delivery_order_count", "metadata": {}},
                    {
                        "name": "delivery_activity_ratio",
                        "metadata": {
                            "inputs": [
                                {"name": "delivery_order_count"},
                                {"name": "order_count"},
                            ]
                        },
                    },
                ],
                "has_more": False,
            }
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(metrics=["delivery_activity_ratio"])

        semantic_tools.query_metrics.assert_called_once_with(
            metrics=["delivery_activity_ratio"],
            dimensions=None,
            path=None,
            time_start=None,
            time_end=None,
            time_granularity=None,
            where=None,
            limit=None,
            order_by=None,
            dry_run=False,
        )

    def test_query_metrics_expands_rolling_average_metric_bundle(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Commerce": {"Orders": {"metrics": ["moving_3_month_order_count_avg"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={
                "items": [
                    {
                        "name": "order_count",
                        "measures": ["orders.order_id"],
                        "metadata": {
                            "dataset": "orders",
                            "expr": "COUNT(DISTINCT order_id)",
                            "metric_kind": "aggregate",
                            "measure": "order_id",
                        },
                    },
                    {
                        "name": "moving_window_month_count",
                        "metadata": {
                            "dataset": "orders",
                            "time_dimension": "metric_time__month",
                            "window": "3 months",
                            "window_aggregation": "row_count",
                            "metric_kind": "cumulative",
                        },
                    },
                    {
                        "name": "moving_3_month_order_count_avg",
                        "measures": ["orders.order_id"],
                        "metadata": {
                            "dataset": "orders",
                            "time_dimension": "metric_time__month",
                            "window": "3 months",
                            "window_aggregation": "avg",
                            "metric_kind": "cumulative",
                            "expr": "COUNT(DISTINCT order_id)",
                            "measure": "order_id",
                        },
                    },
                ],
                "has_more": False,
            }
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(metrics=["moving_3_month_order_count_avg"], dimensions=[])

        semantic_tools.query_metrics.assert_called_once_with(
            metrics=["order_count", "moving_window_month_count", "moving_3_month_order_count_avg"],
            dimensions=["metric_time__month"],
            path=None,
            time_start=None,
            time_end=None,
            time_granularity="month",
            where=None,
            limit=None,
            order_by=["metric_time__month"],
            dry_run=False,
        )

    def test_query_metrics_infers_window_grain_from_metric_time_dimension(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Commerce": {"Orders": {"metrics": ["running_order_count"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={
                "items": [
                    {
                        "name": "order_count",
                        "measures": ["orders.order_id"],
                        "metadata": {
                            "dataset": "orders",
                            "expr": "COUNT(DISTINCT order_id)",
                            "metric_kind": "aggregate",
                            "measure": "order_id",
                        },
                    },
                    {
                        "name": "running_order_count",
                        "metadata": {
                            "dataset": "orders",
                            "time_dimension": "metric_time__month",
                            "window_aggregation": "sum",
                            "metric_kind": "cumulative",
                        },
                    },
                ],
                "has_more": False,
            }
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(metrics=["running_order_count"], dimensions=[])

        semantic_tools.query_metrics.assert_called_once_with(
            metrics=["running_order_count"],
            dimensions=["metric_time__month"],
            path=None,
            time_start=None,
            time_end=None,
            time_granularity="month",
            where=None,
            limit=None,
            order_by=["metric_time__month"],
            dry_run=False,
        )

    def test_query_metrics_expands_running_extrema_metric_bundle(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={
                "Commerce": {
                    "Orders": {
                        "metrics": [
                            "average_order_amount",
                            "running_min_average_order_amount",
                            "running_max_average_order_amount",
                        ]
                    }
                }
            },
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={
                "items": [
                    {
                        "name": "average_order_amount",
                        "metadata": {
                            "dataset": "orders",
                            "expr": "AVG(order_amount)",
                            "metric_kind": "aggregate",
                            "measure": "order_amount",
                        },
                    },
                    {
                        "name": "running_min_average_order_amount",
                        "metadata": {
                            "dataset": "orders",
                            "grain_to_date": "month",
                            "time_dimension": "metric_time__month",
                            "window_aggregation": "min",
                            "metric_kind": "cumulative",
                            "expr": "AVG(order_amount)",
                            "measure": "order_amount",
                        },
                    },
                    {
                        "name": "running_max_average_order_amount",
                        "metadata": {
                            "dataset": "orders",
                            "grain_to_date": "month",
                            "time_dimension": "metric_time__month",
                            "window_aggregation": "max",
                            "metric_kind": "cumulative",
                            "expr": "AVG(order_amount)",
                            "measure": "order_amount",
                        },
                    },
                ],
                "has_more": False,
            }
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(
            metrics=["running_min_average_order_amount", "running_max_average_order_amount"],
            dimensions=["metric_time__month"],
        )

        semantic_tools.query_metrics.assert_called_once_with(
            metrics=[
                "average_order_amount",
                "running_min_average_order_amount",
                "running_max_average_order_amount",
            ],
            dimensions=["metric_time__month"],
            path=None,
            time_start=None,
            time_end=None,
            time_granularity="month",
            where=None,
            limit=None,
            order_by=["metric_time__month"],
            dry_run=False,
        )

    def test_query_metrics_passes_limit_through(self, real_agent_config, mock_llm_create):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={"items": [{"name": "order_count", "metadata": {}}], "has_more": False}
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(
            metrics=["order_count"],
            dimensions=["ac_channel"],
            limit=10,
        )

        semantic_tools.query_metrics.assert_called_once()
        assert semantic_tools.query_metrics.call_args.kwargs["limit"] == 10

    def test_query_metrics_keeps_joined_dimension_where_semantics(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={"items": [{"name": "order_count", "metadata": {}}], "has_more": False}
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(
            metrics=["order_count"],
            dimensions=["dimension_key__display_name"],
            where="region = 'east'",
        )

        semantic_tools.query_metrics.assert_called_once()
        assert semantic_tools.query_metrics.call_args.kwargs["where"] == "region = 'east'"

    def test_query_metrics_passes_join_controls(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={"items": [{"name": "order_count", "metadata": {}}], "has_more": False}
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(result={"columns": [], "data": []})

        node.query_metrics(
            metrics=["order_count"],
            dimensions=["dimension_key__display_name"],
            join_policy="dimension_preserving",
            zero_fill=True,
        )

        semantic_tools.query_metrics.assert_called_once()
        assert semantic_tools.query_metrics.call_args.kwargs["join_policy"] == "dimension_preserving"
        assert semantic_tools.query_metrics.call_args.kwargs["zero_fill"] is True

    def test_query_metrics_aliases_joined_dimension_display_columns(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node, semantic_tools, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["order_count"]}}},
        )
        semantic_tools.list_metrics.return_value = FuncToolResult(
            result={"items": [{"name": "order_count", "metadata": {}}], "has_more": False}
        )
        semantic_tools.query_metrics.return_value = FuncToolResult(
            result={
                "columns": ["dimension_key__display_name", "order_count"],
                "data": {
                    "compressed_data": "dimension_key__display_name,order_count\nknown,1\n",
                    "original_rows": 1,
                },
                "metadata": {},
            }
        )

        result = node.query_metrics(metrics=["order_count"], dimensions=["dimension_key__display_name"])

        assert result.success == 1
        assert result.result["columns"] == ["display_name", "order_count"]
        assert result.result["metadata"]["_display_column_aliases"] == {"dimension_key__display_name": "display_name"}
        assert result.result["data"]["compressed_data"].startswith("display_name,order_count")

    def test_update_context_defaults_to_last_query_metrics_result(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node, _, _ = _make_node(real_agent_config, tree={})

        def query_action(rows, *, limit):
            return {
                "action_type": "query_metrics",
                "status": "success",
                "input": {"arguments": f'{{"limit": {limit}}}'},
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": [
                                "metric_time__month",
                                "customer_segment",
                                "order_count",
                                "previous_month_order_count",
                                "order_count_mom_delta",
                            ],
                            "data": {
                                "original_rows": rows,
                                "compressed_data": f"rows={rows}",
                            },
                            "metadata": {},
                        },
                    }
                },
            }

        node.result = Mock(
            action_history=[
                query_action(21, limit=100),
                query_action(3, limit=5),
            ]
        )
        workflow = Mock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result == {"success": True, "message": "query_metrics result captured"}
        assert len(workflow.context.sql_contexts) == 1
        assert workflow.context.sql_contexts[0].sql_return == "rows=3"
        assert workflow.context.sql_contexts[0].row_count == 3

    def test_update_context_required_selection_uses_selected_result_id(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node, _, _ = _make_node(
            real_agent_config,
            tree={},
            node_config={"require_final_result_selection": True},
            node_name="strict_ask_metrics",
        )

        def query_action(result_id, rows):
            return {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "result_id": result_id,
                            "columns": ["metric_time__month", "order_count"],
                            "data": {
                                "original_rows": rows,
                                "compressed_data": f"rows={rows}",
                            },
                            "metadata": {},
                        },
                    }
                },
            }

        node.result = Mock(
            action_history=[
                query_action("query_metrics:1", 21),
                query_action("query_metrics:2", 3),
                {
                    "action_type": "select_final_metric_result",
                    "status": "success",
                    "output": {
                        "raw_output": {
                            "success": 1,
                            "result": {"result_id": "query_metrics:1"},
                        }
                    },
                },
            ]
        )
        workflow = Mock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result == {"success": True, "message": "query_metrics result captured"}
        assert workflow.context.sql_contexts[0].sql_return == "rows=21"
        assert workflow.context.sql_contexts[0].row_count == 21

    def test_update_context_required_selection_fails_without_selection(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node, _, _ = _make_node(
            real_agent_config,
            tree={},
            node_config={"require_final_result_selection": True},
            node_name="strict_ask_metrics_missing_selection",
        )
        node.result = Mock(
            action_history=[
                {
                    "action_type": "query_metrics",
                    "status": "success",
                    "output": {
                        "raw_output": {
                            "success": 1,
                            "result": {
                                "result_id": "query_metrics:1",
                                "columns": ["order_count"],
                                "data": {"original_rows": 1, "compressed_data": "order_count\n1"},
                                "metadata": {},
                            },
                        }
                    },
                }
            ]
        )
        workflow = Mock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result == {"success": False, "message": "final query_metrics result was not selected"}
        assert workflow.context.sql_contexts == []

    def test_unsupported_tool_patterns_are_ignored(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["revenue"]}}},
            node_config={"tools": "unknown_tools.*"},
            node_name="custom_metric_agent",
        )

        node.semantic_tools = object()
        node._setup_specific_tool_method("semantic_tools", "missing_method")

        assert node.tools == []

    def test_custom_configuration_can_add_tools_outside_default_surface(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}
        db_tools = Mock()
        db_tools.read_query = Mock(name="read_query")
        db_tools.to_function_tool.side_effect = _fake_function_tool
        # Plain-string category + name list so the registry scanner picks the
        # mock up (Mock auto-attributes are excluded by the isinstance(str)
        # guard in ``AgenticNode._iter_tool_groups``).
        db_tools.permission_category = "db_tools"
        db_tools.all_tools_name = lambda: ["read_query"]
        # Model available_tools() explicitly instead of relying on Mock
        # auto-creation, so the double mirrors the full production contract
        # (permission_category + available_tools() + all_tools_name()).
        db_tools.available_tools.return_value = [_fake_function_tool(db_tools.read_query)]
        date_parsing_tools = Mock()
        date_parsing_tools.parse_temporal_expressions = Mock(name="parse_temporal_expressions")
        date_parsing_tools.permission_category = "date_parsing_tools"
        date_parsing_tools.all_tools_name = lambda: ["parse_temporal_expressions"]
        date_parsing_tools.available_tools.return_value = [
            _fake_function_tool(date_parsing_tools.parse_temporal_expressions)
        ]

        with (
            patch("datus.agent.node.ask_metrics_agentic_node.DBFuncTool", return_value=db_tools),
            patch("datus.agent.node.ask_metrics_agentic_node.DateParsingTools", return_value=date_parsing_tools),
        ):
            node, _, _ = _make_node(
                real_agent_config,
                tree=tree,
                node_config={
                    "type": "ask_metrics",
                    "tools": "db_tools.read_query, date_parsing_tools.parse_temporal_expressions",
                },
                node_name="custom_metric_agent",
            )

        assert [tool.name for tool in node.tools] == ["read_query", "parse_temporal_expressions"]
        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry.get("read_query") == "db_tools"
        assert registry.get("parse_temporal_expressions") == "date_parsing_tools"

    def test_setup_tools_handles_context_search_failure(self, real_agent_config, mock_llm_create):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        semantic_tools = _semantic_tools(adapter=True)
        with (
            patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", side_effect=RuntimeError("rag down")),
            patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", return_value=semantic_tools),
            patch("datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool", side_effect=_fake_function_tool),
        ):
            node = AskMetricsAgenticNode(
                node_id="ask_metrics_test",
                description="Ask metrics",
                node_type=NodeType.TYPE_ASK_METRICS,
                agent_config=real_agent_config,
                node_name="ask_metrics",
            )

        assert node.context_search_tools is None
        assert node.subject_tree_mode == "none"
        assert [tool.name for tool in node.tools] == [
            "list_metrics",
            "get_dimensions",
            "query_metrics",
            "attribution_analyze",
        ]

    def test_setup_tools_handles_semantic_setup_failure(self, real_agent_config, mock_llm_create):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        context_tools = _context_tools({"Sales": {"Orders": {"metrics": ["revenue"]}}})
        with (
            patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=context_tools),
            patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", side_effect=RuntimeError("bad adapter")),
            patch("datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool", side_effect=_fake_function_tool),
        ):
            node = AskMetricsAgenticNode(
                node_id="ask_metrics_test",
                description="Ask metrics",
                node_type=NodeType.TYPE_ASK_METRICS,
                agent_config=real_agent_config,
                node_name="ask_metrics",
            )

        assert node.startup_error == "Semantic adapter unavailable: bad adapter"
        assert node.tools == []

    def test_configured_list_subject_tree_only_exposed_for_partial_tree(self, real_agent_config, mock_llm_create):
        small_tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}
        small_node, _, _ = _make_node(
            real_agent_config,
            tree=small_tree,
            node_config={"type": "ask_metrics", "tools": "context_search_tools.list_subject_tree"},
            node_name="custom_metric_agent",
        )
        assert "list_subject_tree" not in [tool.name for tool in small_node.tools]

        large_tree = {"Domain": {f"Subject_{idx}": {"metrics": [f"metric_{idx}"]} for idx in range(3)}}
        large_node, _, _ = _make_node(
            real_agent_config,
            tree=large_tree,
            node_config={
                "type": "ask_metrics",
                "tools": "context_search_tools.list_subject_tree",
                "subject_tree_prompt_limit": 2,
            },
            node_name="custom_metric_agent",
        )
        assert [tool.name for tool in large_node.tools] == ["list_subject_tree"]

    def test_custom_prompt_template_and_version_are_used(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["revenue"]}}},
            node_config={"type": "ask_metrics", "prompt_version": "2.0"},
            node_name="custom_metric_agent",
        )

        prompt_manager = Mock()
        prompt_manager.render_template.return_value = "custom ask metrics prompt"
        with patch("datus.prompts.prompt_manager.get_prompt_manager", return_value=prompt_manager):
            prompt = node._get_system_prompt()

        assert "custom ask metrics prompt" in prompt
        prompt_manager.render_template.assert_called_once()
        call_kwargs = prompt_manager.render_template.call_args.kwargs
        assert call_kwargs["template_name"] == "custom_metric_agent_system"
        assert call_kwargs["version"] == "2.0"

    def test_custom_prompt_falls_back_to_builtin_template(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["revenue"]}}},
            node_config={"type": "ask_metrics", "prompt_version": "2.0"},
            node_name="custom_metric_agent",
        )

        prompt_manager = Mock()
        prompt_manager.render_template.side_effect = [FileNotFoundError("missing"), "builtin ask metrics prompt"]
        with patch("datus.prompts.prompt_manager.get_prompt_manager", return_value=prompt_manager):
            prompt = node._get_system_prompt()

        assert "builtin ask metrics prompt" in prompt
        assert prompt_manager.render_template.call_args_list[0].kwargs["template_name"] == "custom_metric_agent_system"
        assert prompt_manager.render_template.call_args_list[1].kwargs["template_name"] == "ask_metrics_system"
        assert prompt_manager.render_template.call_args_list[1].kwargs["version"] == "2.0"

    @pytest.mark.asyncio
    async def test_adapter_unavailable_fails_before_llm(self, real_agent_config, mock_llm_create):
        from datus.utils.exceptions import DatusException, ErrorCode

        node, _, _ = _make_node(real_agent_config, tree={}, adapter=False)

        assert node.tools == []
        assert node.startup_error == "semantic adapter missing"
        with pytest.raises(DatusException, match="ask_metrics is unavailable") as exc_info:
            await node._before_stream(Mock())
        assert exc_info.value.code == ErrorCode.COMMON_CONFIG_ERROR

    def test_success_result_reports_sorted_tools_used(self, real_agent_config, mock_llm_create):
        from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus

        node, _, _ = _make_node(real_agent_config, tree={})
        action_history_manager = ActionHistoryManager()
        for action_type in ("query_metrics", "get_dimensions", "query_metrics"):
            action_history_manager.add_action(
                ActionHistory.create_action(
                    role=ActionRole.TOOL,
                    action_type=action_type,
                    messages="",
                    input_data={},
                    output_data={},
                    status=ActionStatus.SUCCESS,
                )
            )

        result = node._build_success_result(
            Mock(
                response_content="done",
                last_successful_output=None,
                action_history_manager=action_history_manager,
            )
        )

        assert result.execution_stats["tools_used"] == ["get_dimensions", "query_metrics"]

    def test_success_result_uses_last_output_fallback_and_stringifies(self, real_agent_config, mock_llm_create):
        from datus.schemas.action_history import ActionHistoryManager

        node, _, _ = _make_node(real_agent_config, tree={})

        result = node._build_success_result(
            Mock(
                response_content="",
                last_successful_output={"response": {"value": 10}},
                action_history_manager=ActionHistoryManager(),
            )
        )

        assert result.response == "{'value': 10}"
        assert result.markdown_report == "{'value': 10}"

    def test_mounted_tool_surface_is_narrow(self, real_agent_config, mock_llm_create):
        """The mounted tool surface stays the DEFAULT_TOOLS whitelist.

        The registry intentionally registers each class's full name surface
        (superset, name -> category lookup only); narrowness is enforced by
        ``node.tools``, so assert on the mounted names instead.
        """
        tree = {"Sales": {"metrics": ["revenue"]}}
        node, _, _ = _make_node(real_agent_config, tree=tree)

        assert {tool.name for tool in node.tools} == {
            "list_metrics",
            "get_dimensions",
            "query_metrics",
            "attribution_analyze",
            "search_metrics",
            "get_metrics",
        }

    def test_extract_subject_tree_ignores_non_dict_nodes(self):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        assert AskMetricsAgenticNode._extract_subject_tree_metric_entries(["not", "a", "tree"]) == []


class TestUpdateContext:
    """Tests for AskMetricsAgenticNode.update_context."""

    def _make_node_with_result(self, action_history):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        node = MagicMock(spec=AskMetricsAgenticNode)
        node.result = MagicMock()
        node.result.action_history = action_history
        node.require_final_result_selection = False
        node._select_last_query_metrics_action = AskMetricsAgenticNode._select_last_query_metrics_action
        node._selected_final_result_id_from_actions = AskMetricsAgenticNode._selected_final_result_id_from_actions
        node._select_query_metrics_action_by_result_id = AskMetricsAgenticNode._select_query_metrics_action_by_result_id
        node._query_result_payload = AskMetricsAgenticNode._query_result_payload
        node._column_aliases_from_metadata = AskMetricsAgenticNode._column_aliases_from_metadata
        node._apply_column_aliases_to_columns = AskMetricsAgenticNode._apply_column_aliases_to_columns
        node._apply_column_aliases_to_csv = AskMetricsAgenticNode._apply_column_aliases_to_csv
        node.update_context = AskMetricsAgenticNode.update_context.__get__(node)
        return node

    def test_captures_compressed_data(self):
        actions = [
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": ["metric_time__month", "order_count"],
                            "data": {
                                "compressed_data": "index,metric_time__month,order_count\n0,2025-06,100",
                                "original_rows": 1,
                                "is_compressed": False,
                            },
                            "metadata": {"sql": "SELECT COUNT(*) FROM t"},
                        },
                    }
                },
            }
        ]
        node = self._make_node_with_result(actions)
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1
        ctx = workflow.context.sql_contexts[0]
        assert "order_count" in ctx.sql_return
        assert ctx.sql_query == "SELECT COUNT(*) FROM t"
        assert ctx.row_count == 1

    def test_captures_full_cached_data_before_compressed_preview(self):
        actions = [
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": ["metric_time__month", "order_count"],
                            "data": {
                                "compressed_data": (
                                    "metric_time__month,order_count\n2025-01-01,1\n...,...\n2025-12-01,12"
                                ),
                                "original_rows": 21,
                                "is_compressed": True,
                                "compression_type": "rows",
                            },
                            "metadata": {"_full_result_cache_key": "query_metrics:1"},
                        },
                    }
                },
            }
        ]
        node = self._make_node_with_result(actions)
        node.semantic_tools = MagicMock()
        node.semantic_tools.get_cached_query_metrics_result.return_value = {
            "csv": "metric_time__month,order_count\n2025-01-01,1\n2025-02-01,2\n",
            "row_count": 21,
        }
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        ctx = workflow.context.sql_contexts[0]
        assert "2025-02-01,2" in ctx.sql_return
        assert "..." not in ctx.sql_return
        assert ctx.row_count == 21

    def test_captures_full_cached_data_with_display_column_aliases(self):
        actions = [
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": ["display_key__display_name", "order_count"],
                            "data": {
                                "compressed_data": "display_name,order_count\nknown,1\n",
                                "original_rows": 1,
                            },
                            "metadata": {
                                "_full_result_cache_key": "query_metrics:1",
                                "_display_column_aliases": {"display_key__display_name": "display_name"},
                            },
                        },
                    }
                },
            }
        ]
        node = self._make_node_with_result(actions)
        node.semantic_tools = MagicMock()
        node.semantic_tools.get_cached_query_metrics_result.return_value = {
            "csv": "display_key__display_name,order_count\nknown,1\n",
            "row_count": 1,
        }
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        ctx = workflow.context.sql_contexts[0]
        assert ctx.sql_return.startswith("display_name,order_count")
        assert "display_key__display_name" not in ctx.sql_return

    def test_captures_list_data(self):
        actions = [
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": ["ac_channel", "count"],
                            "data": [["ch1", 10], ["ch2", 20]],
                            "metadata": {},
                        },
                    }
                },
            }
        ]
        node = self._make_node_with_result(actions)
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        ctx = workflow.context.sql_contexts[0]
        assert "ch1" in ctx.sql_return
        assert ctx.row_count == 2
        assert ctx.sql_query == ""

    def test_captures_zero_row_result(self):
        actions = [
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": ["metric_time__month", "order_count"],
                            "data": [],
                            "metadata": {"sql": "SELECT month, count FROM t WHERE 1 = 0"},
                        },
                    }
                },
            }
        ]
        node = self._make_node_with_result(actions)
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1
        ctx = workflow.context.sql_contexts[0]
        assert ctx.sql_return == "metric_time__month,order_count\r\n"
        assert ctx.row_count == 0
        assert ctx.sql_query == "SELECT month, count FROM t WHERE 1 = 0"

    def test_query_result_helpers_handle_defensive_branches(self):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": {"a": 1}}}) == {"a": 1}
        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": '{"a": 1}'}}) == {"a": 1}
        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": "not-json"}}) == {}
        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": "[1]"}}) == {}
        assert AskMetricsAgenticNode._query_action_arguments({}) == {}

        assert AskMetricsAgenticNode._query_result_payload({"output": "not-dict"}) is None
        assert AskMetricsAgenticNode._query_result_payload({"output": {"raw_output": {"success": 0}}}) is None
        assert (
            AskMetricsAgenticNode._query_result_payload({"output": {"raw_output": {"success": 1, "result": []}}})
            is None
        )
        assert AskMetricsAgenticNode._query_result_columns({"columns": "x"}) == []
        assert AskMetricsAgenticNode._query_result_columns({"columns": ["x", None, ""]}) == ["x", "None"]
        assert AskMetricsAgenticNode._query_result_row_count({"data": {"original_rows": "bad"}}) == 0
        assert AskMetricsAgenticNode._query_result_row_count({"data": [["a"], ["b"]]}) == 2
        assert AskMetricsAgenticNode._query_result_row_count({"data": "not-list"}) == 0
        assert AskMetricsAgenticNode._query_result_id({"result_id": " rid "}) == "rid"
        assert AskMetricsAgenticNode._query_result_id({"metadata": {"_full_result_cache_key": " cache "}}) == "cache"
        assert AskMetricsAgenticNode._query_result_id({"metadata": {}}) is None

        assert AskMetricsAgenticNode._select_last_query_metrics_action(["bad"]) is None
        assert AskMetricsAgenticNode._selected_final_result_id_from_actions(["bad"]) is None
        assert AskMetricsAgenticNode._select_query_metrics_action_by_result_id(["bad"], "missing") is None

    def test_skips_failed_actions(self):
        actions = [
            {
                "action_type": "query_metrics",
                "status": "failed",
                "output": {"error": "some error"},
            }
        ]
        node = self._make_node_with_result(actions)
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        with patch(
            "datus.agent.node.agentic_node.AgenticNode.update_context",
            return_value={"success": True},
        ):
            node.update_context(workflow)

        assert len(workflow.context.sql_contexts) == 0

    def test_picks_last_successful_call(self):
        actions = [
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": ["x"],
                            "data": {"compressed_data": "index,x\n0,first", "original_rows": 1},
                            "metadata": {},
                        },
                    }
                },
            },
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "columns": ["x"],
                            "data": {"compressed_data": "index,x\n0,last", "original_rows": 1},
                            "metadata": {},
                        },
                    }
                },
            },
        ]
        node = self._make_node_with_result(actions)
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        node.update_context(workflow)

        assert "last" in workflow.context.sql_contexts[0].sql_return

    def test_no_query_metrics_falls_back_to_super(self):
        actions = [
            {"action_type": "list_metrics", "status": "success", "output": {}},
        ]
        node = self._make_node_with_result(actions)
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        with patch(
            "datus.agent.node.agentic_node.AgenticNode.update_context",
            return_value={"success": True, "message": "parent called"},
        ) as parent_update:
            result = node.update_context(workflow)

        parent_update.assert_called_once()
        assert result["message"] == "parent called"
