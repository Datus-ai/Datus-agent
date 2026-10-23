# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from datus.configuration.node_type import NodeType
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.query_result_text import parse_query_result_text


def _query_action(text, *, action_type="query_metrics", **extra):
    """A successful query action whose result is the text an executed query returns."""
    return {
        "action_type": action_type,
        "status": "success",
        "output": {"raw_output": {"success": 1, "result": text}},
        **extra,
    }


def _fake_function_tool(method, **kwargs):
    tool = Mock()
    tool.name = getattr(method, "__name__", None) or getattr(method, "_mock_name", "tool")
    tool.excluded_params = kwargs.get("excluded_params")
    return tool


def _semantic_tools():
    tools = Mock()
    for name in (
        "list_metrics",
        "get_metric",
        "get_dataset",
        "query_metrics",
        "validate_semantic",
        "attribution_analyze",
    ):
        setattr(tools, name, Mock(name=name))
    tools.available_tools.return_value = [
        _fake_function_tool(tools.list_metrics),
        _fake_function_tool(tools.get_metric),
        _fake_function_tool(tools.get_dataset),
        _fake_function_tool(tools.query_metrics),
        _fake_function_tool(tools.validate_semantic),
        _fake_function_tool(tools.attribution_analyze),
    ]
    return tools


def _context_tools(tree, semantic_objects=True):
    tools = Mock()
    tools.list_subject_tree.return_value = FuncToolResult(result=tree)
    for name in ("search_metrics", "search_semantic_objects"):
        setattr(tools, name, Mock(name=name))
    tools.has_semantic_objects = semantic_objects
    return tools


def _make_node(real_agent_config, *, tree, node_config=None, node_name="ask_metrics", input_data=None):
    from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

    if node_config is not None:
        real_agent_config.agentic_nodes[node_name] = node_config

    semantic_tools = _semantic_tools()
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
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_patterns, expected_names",
        [
            (
                "semantic_tools.list_metrics,semantic_tools.query_metrics",
                {"list_metrics", "query_metrics"},
            ),
            (
                "semantic_tools.*",
                {
                    "list_metrics",
                    "get_metric",
                    "get_dataset",
                    "query_metrics",
                    "validate_semantic",
                    "attribution_analyze",
                },
            ),
        ],
    )
    @pytest.mark.parametrize("params", [None, {"regions": ["APAC", "EMEA"], "threshold": 100}])
    async def test_registered_query_metrics_executes_with_parameter_bindings(
        self, real_agent_config, mock_llm_create, tool_patterns, expected_names, params
    ):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
        from datus.tools.func_tool.semantic_tools import SemanticTools
        from datus.tools.semantic_tools.models import QueryResult

        real_agent_config.agentic_nodes["ask_metrics"] = {"tools": tool_patterns}
        semantic_tools = SemanticTools(real_agent_config)
        runtime = Mock()
        runtime.query_metrics = AsyncMock(return_value=QueryResult(columns=["revenue"], data=[{"revenue": 100}]))
        semantic_tools._runtime = runtime
        with (
            patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", return_value=semantic_tools),
            patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=_context_tools({})),
        ):
            node = AskMetricsAgenticNode(
                node_id="ask_metrics_schema_test",
                description="Ask metrics",
                node_type=NodeType.TYPE_ASK_METRICS,
                agent_config=real_agent_config,
            )

        tools = {tool.name: tool for tool in node.tools}
        assert set(tools) == expected_names
        assert tools["list_metrics"].strict_json_schema is True
        query_tool = tools["query_metrics"]
        assert query_tool.strict_json_schema is False

        arguments = {"metrics": ["revenue"]}
        if params is not None:
            arguments["params"] = params
        result = await query_tool.on_invoke_tool(None, json.dumps(arguments))

        assert result["success"] == 1
        assert parse_query_result_text(result["result"]).columns == ["revenue"]
        assert parse_query_result_text(result["result"]).row_count == 1
        assert runtime.query_metrics.await_count == 1
        assert runtime.query_metrics.call_args.kwargs["metrics"] == ["revenue"]
        assert runtime.query_metrics.call_args.kwargs.get("params") == params

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
            "list_metrics",
            "get_metric",
            "get_dataset",
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
        # list_subject_tree is mounted only for a partial tree.
        assert "list_subject_tree" not in prompt
        assert "do not call `search_metrics` for metrics that match these entries directly" in prompt
        assert "When the subject tree gives a direct metric/path match" in prompt
        assert "dedicated catalog metric" in prompt
        assert "## Attribution Playbook" in prompt
        assert "reuse its `drill_down.where_sql`" in prompt
        assert "Read detailed contributions from `per_dimension" in prompt
        assert "`query_metrics.time_end` is inclusive" not in prompt
        assert "If `strategy` is `unsupported`" in prompt
        assert "`non_additive` or `truncated`" in prompt
        assert "failed, was non-additive, or was truncated" in prompt
        assert "`dimension_analysis_failed`" in prompt
        assert "Do not invent helper metrics" in prompt
        assert "do not add a `where` filter that enumerates dimension values" in prompt
        assert "Never pass empty JSON values" in prompt
        assert "request those metrics in one `query_metrics(metrics=[...])` call" in prompt
        assert "query all requested catalog metrics together" in prompt
        assert "do not add sibling display/name/label dimensions" in prompt
        assert "Call `get_dataset` for every dataset the query reads" in prompt
        assert "extra.time_dimension" not in prompt
        assert "extra.time_granularities" not in prompt
        assert "advisory defaults, not an exhaustive allowlist" in prompt
        assert "for Dosi validation" in prompt
        assert node.subject_tree_prompt_limit == 100

    def test_prompt_leaves_tool_usage_to_tool_descriptions(self, real_agent_config, mock_llm_create):
        """Every tool rule is sent once: what a tool's description states, the
        system prompt does not repeat."""
        node, _, _ = _make_node(real_agent_config, tree={"Sales": {"Orders": {"metrics": ["revenue"]}}})
        prompt = " ".join(node._get_system_prompt().split())
        # The node mounts mocks here; the descriptions the model reads come from these docstrings.
        tools = {
            name: " ".join(inspect.getdoc(getattr(SemanticTools, name)).split())
            for name in ("get_metric", "get_dataset", "query_metrics")
        }

        for rule, tool in (
            ("Do not pass limit just to preview data", "query_metrics"),
            ("also pass order_by", "query_metrics"),
            ("full result is cached", "query_metrics"),
            ("never the same start and end", "query_metrics"),
            ("never repeated in `where`", "query_metrics"),
            ("follow its `suggested_retry`", "query_metrics"),
            ("finest first (the first is", "get_metric"),
            ("dimensions every query of it must group by", "get_metric"),
            ("issue several calls in parallel", "get_metric"),
            ("issue the calls in", "get_dataset"),
        ):
            assert rule in tools[tool], rule
            assert rule.replace("`", "") not in prompt.replace("`", ""), rule
        assert "## Placing Conditions" not in prompt
        assert "Pass every name under `required`" not in prompt
        assert "OSI half-open `[start, end)` semantics" not in prompt

    def test_prompt_names_only_mounted_tools(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["revenue"]}}},
            node_config={
                "tools": "semantic_tools.list_metrics, semantic_tools.get_metric, semantic_tools.query_metrics"
            },
        )
        assert {tool.name for tool in node.tools} == {"list_metrics", "get_metric", "query_metrics"}

        prompt = node._get_system_prompt()

        for absent in ("attribution", "Attribution", "search_metrics", "search_semantic_objects", "list_subject_tree"):
            assert absent not in prompt, absent
        assert "`query_metrics` is the only tool that may produce metric results." in prompt
        assert "Use them as the routing catalog for direct metric matching." in prompt

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

    def test_dosi_prompt_uses_metric_time_as_input_not_suffixed_output_alias(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["running_revenue"]}}},
        )

        prompt = node._get_system_prompt()

        assert "Never pass a suffixed output name such as `metric_time__day` in `dimensions`" in prompt
        assert "the query input remains `metric_time` plus `time_granularity`" in prompt
        assert "Order time-axis window results by their grained time result" in prompt
        assert "Order value-oriented window results" in prompt

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

    def test_mcp_servers_attached_from_node_config(self, real_agent_config, mock_llm_create):
        """A custom agent's ``mcp`` selection is attached by the AgenticNode base."""
        tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}
        mock_server = MagicMock()

        with patch(
            "datus.agent.node.agentic_node.AgenticNode._setup_mcp_server_from_config",
            return_value=mock_server,
        ) as mock_setup:
            node, _, _ = _make_node(
                real_agent_config,
                tree=tree,
                node_config={"type": "ask_metrics", "mcp": "test_server"},
                node_name="mcp_metric_agent",
            )

        mock_setup.assert_called_once_with("test_server")
        assert node.mcp_servers == {"test_server": mock_server}

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
            "list_metrics",
            "get_metric",
            "get_dataset",
            "query_metrics",
            "attribution_analyze",
        ]

    def test_update_context_defaults_to_last_query_metrics_result(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node, _, _ = _make_node(real_agent_config, tree={})

        def query_action(rows, *, limit):
            return _query_action(
                f"result_id: query_metrics:{rows}\nrows: {rows}\n\nmetric_time__month,order_count\nrows={rows}",
                input={"arguments": f'{{"limit": {limit}}}'},
            )

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
        assert workflow.context.sql_contexts[0].sql_return == "metric_time__month,order_count\nrows=3"
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
            return _query_action(f"result_id: {result_id}\nrows: {rows}\n\nmetric_time__month,order_count\nrows={rows}")

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
        assert workflow.context.sql_contexts[0].sql_return == "metric_time__month,order_count\nrows=21"
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

    def test_db_tools_use_input_database(self, real_agent_config, mock_llm_create):
        from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput

        node, _, _ = _make_node(
            real_agent_config,
            tree={"Sales": {"Orders": {"metrics": ["revenue"]}}},
            node_config={"type": "ask_metrics", "tools": "db_tools.read_query"},
            input_data=AskMetricsNodeInput(user_message="Show revenue", database="california_schools"),
        )

        assert node.db_func_tool._default_database == "california_schools"

    def test_setup_tools_refreshes_db_tools_for_changed_input_database(self, real_agent_config, mock_llm_create):
        from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput

        tree = {"Sales": {"Orders": {"metrics": ["revenue"]}}}
        node, _, _ = _make_node(
            real_agent_config,
            tree=tree,
            node_config={"type": "ask_metrics", "tools": "db_tools.read_query"},
        )
        original_db_tool = node.db_func_tool

        node.input = AskMetricsNodeInput(user_message="Show revenue", database="california_schools")
        with (
            patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", return_value=_semantic_tools()),
            patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=_context_tools(tree)),
            patch("datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool", side_effect=_fake_function_tool),
        ):
            node.setup_tools()

        assert node.db_func_tool is not original_db_tool
        assert node.db_func_tool._default_database == "california_schools"

    def test_setup_tools_handles_context_search_failure(self, real_agent_config, mock_llm_create):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        semantic_tools = _semantic_tools()
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
            "get_metric",
            "get_dataset",
            "query_metrics",
            "attribution_analyze",
        ]

    def test_setup_tools_handles_semantic_setup_failure(self, real_agent_config, mock_llm_create):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        context_tools = _context_tools({"Sales": {"Orders": {"metrics": ["revenue"]}}})
        with (
            patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=context_tools),
            patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", side_effect=RuntimeError("bad runtime")),
            patch("datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool", side_effect=_fake_function_tool),
        ):
            node = AskMetricsAgenticNode(
                node_id="ask_metrics_test",
                description="Ask metrics",
                node_type=NodeType.TYPE_ASK_METRICS,
                agent_config=real_agent_config,
                node_name="ask_metrics",
            )

        assert node.startup_error == "Dosi runtime unavailable: bad runtime"
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

    def test_success_result_reports_sorted_tools_used(self, real_agent_config, mock_llm_create):
        from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus

        node, _, _ = _make_node(real_agent_config, tree={})
        action_history_manager = ActionHistoryManager()
        for action_type in ("query_metrics", "get_metric", "query_metrics"):
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

        assert result.execution_stats["tools_used"] == ["get_metric", "query_metrics"]

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
            "get_metric",
            "get_dataset",
            "query_metrics",
            "attribution_analyze",
            "search_metrics",
        }

    @pytest.mark.parametrize("synced", [True, False])
    def test_semantic_object_search_is_mounted_only_with_objects_to_search(
        self, real_agent_config, mock_llm_create, synced
    ):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        real_agent_config.agentic_nodes["ask_metrics"] = {
            "tools": "context_search_tools.search_metrics,context_search_tools.search_semantic_objects"
        }
        semantic_tools = _semantic_tools()
        context_tools = _context_tools({"Sales": {"metrics": ["revenue"]}}, semantic_objects=synced)
        with (
            patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", return_value=semantic_tools),
            patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=context_tools),
            patch(
                "datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool",
                side_effect=_fake_function_tool,
            ),
        ):
            node = AskMetricsAgenticNode(
                node_id="ask_metrics_test",
                description="Ask metrics",
                node_type=NodeType.TYPE_ASK_METRICS,
                agent_config=real_agent_config,
                node_name="ask_metrics",
            )

        names = [tool.name for tool in node.tools]
        assert "search_metrics" in names
        assert ("search_semantic_objects" in names) is synced

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
        node._query_result_id = AskMetricsAgenticNode._query_result_id
        node.update_context = AskMetricsAgenticNode.update_context.__get__(node)
        return node

    def test_captures_the_preview_when_the_cache_no_longer_holds_the_result(self):
        actions = [_query_action("result_id: query_metrics:1\nrows: 1\n\nmetric_time__month,order_count\n2025-06,100")]
        node = self._make_node_with_result(actions)
        node.semantic_tools = MagicMock()
        node.semantic_tools.get_cached_query_metrics_result.return_value = None
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1
        ctx = workflow.context.sql_contexts[0]
        assert ctx.sql_return == "metric_time__month,order_count\n2025-06,100"
        assert ctx.sql_query == ""
        assert ctx.row_count == 1

    def test_captures_full_cached_data_before_the_preview(self):
        actions = [
            _query_action(
                "result_id: query_metrics:1\nrows: 21\n\nmetric_time__month,order_count\n2025-01-01,1\n"
                "... 19 rows omitted\n2025-12-01,12"
            )
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
        node.semantic_tools.get_cached_query_metrics_result.assert_called_once_with("query_metrics:1")
        ctx = workflow.context.sql_contexts[0]
        assert "2025-02-01,2" in ctx.sql_return
        assert "omitted" not in ctx.sql_return
        assert ctx.row_count == 21

    def test_captures_zero_row_result(self):
        actions = [_query_action("result_id: query_metrics:1\nrows: 0\n\nmetric_time__month,order_count")]
        node = self._make_node_with_result(actions)
        node.semantic_tools = MagicMock()
        node.semantic_tools.get_cached_query_metrics_result.return_value = {
            "csv": "metric_time__month,order_count\r\n",
            "row_count": 0,
        }
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1
        ctx = workflow.context.sql_contexts[0]
        assert ctx.sql_return == "metric_time__month,order_count\r\n"
        assert ctx.row_count == 0

    def test_a_dry_run_result_is_not_captured(self):
        """A dry run returns compiled SQL, not rows: it is never the answer."""
        actions = [
            {
                "action_type": "query_metrics",
                "status": "success",
                "output": {"raw_output": {"success": 1, "result": {"columns": ["sql"], "data": {}, "metadata": {}}}},
            }
        ]
        node = self._make_node_with_result(actions)
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        with patch(
            "datus.agent.node.agentic_node.AgenticNode.update_context",
            return_value={"success": True, "message": "parent called"},
        ):
            result = node.update_context(workflow)

        assert result["message"] == "parent called"
        assert workflow.context.sql_contexts == []

    def test_query_result_helpers_handle_defensive_branches(self):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": {"a": 1}}}) == {"a": 1}
        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": '{"a": 1}'}}) == {"a": 1}
        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": "not-json"}}) == {}
        assert AskMetricsAgenticNode._query_action_arguments({"input": {"arguments": "[1]"}}) == {}
        assert AskMetricsAgenticNode._query_action_arguments({}) == {}

        assert AskMetricsAgenticNode._query_result_payload({"output": "not-dict"}) is None
        assert AskMetricsAgenticNode._query_result_payload({"output": {"raw_output": {"success": 0}}}) is None
        assert AskMetricsAgenticNode._query_result_id({"result_id": " rid "}) == "rid"
        assert AskMetricsAgenticNode._query_result_id({"result_id": None}) is None
        assert AskMetricsAgenticNode._query_result_id("result_id: rid\nrows: 1\n\nx\n1") == "rid"
        assert AskMetricsAgenticNode._query_result_id("rows: 1\n\nx\n1") is None
        assert AskMetricsAgenticNode._query_result_id("not a result") is None
        assert AskMetricsAgenticNode._query_result_id(None) is None

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
            _query_action("result_id: query_metrics:1\nrows: 1\n\nx\nfirst"),
            _query_action("result_id: query_metrics:2\nrows: 1\n\nx\nlast"),
        ]
        node = self._make_node_with_result(actions)
        node.semantic_tools = None
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
