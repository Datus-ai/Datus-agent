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


def _make_node(real_agent_config, *, tree, adapter=True, node_config=None, node_name="ask_metrics"):
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
        assert "query the base metric together with the derived metric" in prompt
        assert "first period" in prompt
        assert node.subject_tree_prompt_limit == 100

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
        date_parsing_tools = Mock()
        date_parsing_tools.parse_temporal_expressions = Mock(name="parse_temporal_expressions")

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
        mapping = node._tool_category_map()
        assert [tool.name for tool in mapping["db_tools"]] == ["read_query"]
        assert [tool.name for tool in mapping["date_parsing_tools"]] == ["parse_temporal_expressions"]

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

    def test_tool_category_map_is_narrow(self, real_agent_config, mock_llm_create):
        tree = {"Sales": {"metrics": ["revenue"]}}
        node, _, _ = _make_node(real_agent_config, tree=tree)

        mapping = node._tool_category_map()
        assert {tool.name for tool in mapping["semantic_tools"]} == {
            "list_metrics",
            "get_dimensions",
            "query_metrics",
            "attribution_analyze",
        }
        assert {tool.name for tool in mapping["context_search_tools"]} == {
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
                            "columns": ["metric_time__month", "activity_count"],
                            "data": {
                                "compressed_data": "index,metric_time__month,activity_count\n0,2025-06,100",
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
        assert "activity_count" in ctx.sql_return
        assert ctx.sql_query == "SELECT COUNT(*) FROM t"
        assert ctx.row_count == 1

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
