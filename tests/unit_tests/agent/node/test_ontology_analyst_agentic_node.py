# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from datus.configuration.node_type import NodeType
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.exceptions import DatusException

CATALOG = {
    "ontology": "Aviation",
    "concepts": [
        {"name": "Airport", "identity": ["code"], "root": True},
        {"name": "Flight", "identity": ["id"], "root": True},
        {"name": "FlightAlert", "identity": ["id"], "root": True},
    ],
}


def _fake_function_tool(method):
    tool = Mock()
    tool.name = getattr(method, "__name__", None) or getattr(method, "_mock_name", "tool")
    return tool


def _runtime(*, has_ontology=True, unpaired=(), catalog=None, datasource="flights_db"):
    runtime = Mock()
    runtime.datasource = datasource
    runtime.has_ontology.return_value = has_ontology
    runtime.unpaired_ontologies.return_value = list(unpaired)
    runtime.describe = AsyncMock(return_value=CATALOG if catalog is None else catalog)
    return runtime


def _semantic_tools(runtime):
    tools = Mock()
    tools.runtime = runtime
    tools._runtime_unavailable_message.return_value = "Dosi runtime unavailable: bad yaml"
    names = (
        "list_metrics",
        "get_metric",
        "query_metrics",
        "attribution_analyze",
        "describe_concept",
        "find_concept_paths",
        "select_records",
    )
    for name in names:
        setattr(tools, name, Mock(name=name))
    tools.available_tools.return_value = [_fake_function_tool(getattr(tools, name)) for name in names]
    return tools


def _context_tools():
    tools = Mock()
    tools.list_subject_tree.return_value = FuncToolResult(result={})
    tools.search_metrics = Mock(name="search_metrics")
    return tools


def _make_node(
    real_agent_config,
    *,
    runtime=None,
    supports=True,
    node_config=None,
    node_name="ontology_analyst",
    semantic_tools_error=None,
):
    from datus.agent.node.ontology_analyst_agentic_node import OntologyAnalystAgenticNode

    if node_config is not None:
        real_agent_config.agentic_nodes[node_name] = node_config
    runtime = _runtime() if runtime is None else runtime
    semantic_tools = _semantic_tools(runtime)
    semantic_tools_factory = Mock(return_value=semantic_tools, side_effect=semantic_tools_error)
    with (
        patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools", semantic_tools_factory),
        patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=_context_tools()),
        patch("datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool", side_effect=_fake_function_tool),
        patch("datus.agent.node.ontology_analyst_agentic_node.binding_supports_ontology", return_value=supports),
    ):
        node = OntologyAnalystAgenticNode(
            node_id="ontology_analyst_test",
            description="Ontology analyst",
            node_type=NodeType.TYPE_ONTOLOGY_ANALYST,
            agent_config=real_agent_config,
            node_name=node_name,
        )
    return node, semantic_tools_factory, runtime


class TestSetup:
    def test_default_tools_and_ontology_enabled_semantic_tools(self, real_agent_config, mock_llm_create):
        node, factory, _ = _make_node(real_agent_config)

        assert node.startup_error is None
        assert [tool.name for tool in node.tools] == [
            "describe_concept",
            "find_concept_paths",
            "select_records",
            "list_metrics",
            "get_metric",
            "query_metrics",
            "attribution_analyze",
            "search_metrics",
        ]
        kwargs = factory.call_args.kwargs
        assert kwargs["ontology_enabled"] is True
        assert kwargs["ontology_mapping"] is None

    def test_ontology_mapping_comes_from_agent_yml(self, real_agent_config, mock_llm_create):
        _, factory, _ = _make_node(real_agent_config, node_config={"ontology_mapping": " departures "})
        assert factory.call_args.kwargs["ontology_mapping"] == "departures"

    def test_catalog_is_loaded_into_the_prompt(self, real_agent_config, mock_llm_create):
        node, _, runtime = _make_node(real_agent_config)

        runtime.describe.assert_awaited_once_with()
        assert node.concept_catalog_mode == "full"
        assert node.concept_catalog_count == 3
        prompt = node._get_system_prompt()
        assert "## Concept Catalog" in prompt
        assert '"FlightAlert"' in prompt
        assert "describe_concept" in prompt and "find_concept_paths" in prompt and "select_records" in prompt

    def test_large_catalog_prompts_names_only(self, real_agent_config, mock_llm_create):
        catalog = {"concepts": [{"name": f"C{i}", "identity": ["id"], "root": True} for i in range(5)]}
        node, _, _ = _make_node(
            real_agent_config,
            runtime=_runtime(catalog=catalog),
            node_config={"concept_catalog_prompt_limit": 3},
        )
        assert node.concept_catalog_mode == "partial"
        excerpt = json.loads(node.concept_catalog_prompt)
        assert excerpt == {"concept_names": ["C0", "C1", "C2", "C3", "C4"], "total_concepts": 5}
        assert "only their names are listed" in node._get_system_prompt()

    @pytest.mark.parametrize("raw_limit", [0, -1, "many", True])
    def test_invalid_catalog_limit_uses_default(self, real_agent_config, mock_llm_create, raw_limit):
        node, _, _ = _make_node(real_agent_config, node_config={"concept_catalog_prompt_limit": raw_limit})
        assert node.concept_catalog_mode == "full"

    def test_multi_ontology_catalog_counts_every_concept(self, real_agent_config, mock_llm_create):
        catalog = {"ontologies": [CATALOG, {"concepts": [{"name": "Store"}]}]}
        node, _, _ = _make_node(real_agent_config, runtime=_runtime(catalog=catalog))
        assert node.concept_catalog_count == 4


class TestStartupErrors:
    """ontology_analyst never degrades: each missing prerequisite becomes a startup error."""

    @staticmethod
    def _missing(node, *fragments):
        """Fragments absent from the startup error; an empty error misses all of them."""
        error = node.startup_error or ""
        return [fragment for fragment in fragments if fragment not in error]

    def test_binding_without_ontology_support(self, real_agent_config, mock_llm_create):
        node, _, runtime = _make_node(real_agent_config, supports=False)
        assert self._missing(node, "binding cannot describe ontologies") == []
        runtime.describe.assert_not_called()

    def test_runtime_unavailable(self, real_agent_config, mock_llm_create):
        runtime = Mock()
        from datus.agent.node.ontology_analyst_agentic_node import OntologyAnalystAgenticNode

        with (
            patch("datus.agent.node.ask_metrics_agentic_node.SemanticTools") as factory,
            patch("datus.agent.node.ask_metrics_agentic_node.ContextSearchTools", return_value=_context_tools()),
            patch("datus.agent.node.ask_metrics_agentic_node.trans_to_function_tool", side_effect=_fake_function_tool),
            patch("datus.agent.node.ontology_analyst_agentic_node.binding_supports_ontology", return_value=True),
        ):
            tools = _semantic_tools(runtime)
            tools.runtime = None
            factory.return_value = tools
            node = OntologyAnalystAgenticNode(
                node_id="n",
                description="d",
                node_type=NodeType.TYPE_ONTOLOGY_ANALYST,
                agent_config=real_agent_config,
            )
        assert self._missing(node, "bad yaml") == []

    def test_no_ontology_points_to_ask_metrics(self, real_agent_config, mock_llm_create):
        node, _, runtime = _make_node(real_agent_config, runtime=_runtime(has_ontology=False))
        expected_dir = str(real_agent_config.path_manager.ontology_path("flights_db"))
        assert self._missing(node, "no ontology for datasource 'flights_db'", expected_dir, "ask_metrics") == []
        runtime.describe.assert_not_called()

    def test_unpaired_ontology_names_the_file(self, real_agent_config, mock_llm_create):
        runtime = _runtime(has_ontology=False, unpaired=["/p/subject/semantic_models/ontology/flights_db/hr.yaml"])
        node, _, _ = _make_node(real_agent_config, runtime=runtime)
        assert self._missing(node, "hr.yaml", "pair with no semantic model file") == []

    def test_ontology_load_failure(self, real_agent_config, mock_llm_create):
        runtime = _runtime()
        runtime.has_ontology.side_effect = RuntimeError("[ontology_mapping_unresolved] pick one with --mapping")
        node, _, _ = _make_node(real_agent_config, runtime=runtime)
        assert self._missing(node, "ontology_mapping_unresolved") == []

    def test_describe_failure(self, real_agent_config, mock_llm_create):
        runtime = _runtime()
        runtime.describe = AsyncMock(side_effect=RuntimeError("engine exploded"))
        node, _, _ = _make_node(real_agent_config, runtime=runtime)
        assert self._missing(node, "failed to describe the ontology", "engine exploded") == []

    def test_semantic_tools_failure_keeps_the_parent_error(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(real_agent_config, semantic_tools_error=RuntimeError("no dosi"))
        assert self._missing(node, "Dosi runtime unavailable: no dosi") == []

    @pytest.mark.asyncio
    async def test_startup_error_raises_on_first_turn_with_node_name(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(real_agent_config, runtime=_runtime(has_ontology=False))
        with pytest.raises(DatusException, match="ontology_analyst is unavailable: no ontology"):
            await node._before_stream(Mock())


class TestPromptAndResult:
    def test_custom_prompt_falls_back_to_ontology_template(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(real_agent_config, node_config={"type": "ontology_analyst"}, node_name="flight_ops")
        prompt_manager = Mock()
        prompt_manager.render_template.side_effect = [FileNotFoundError("missing"), "builtin ontology prompt"]
        with patch("datus.prompts.prompt_manager.get_prompt_manager", return_value=prompt_manager):
            prompt = node._get_system_prompt()
        assert "builtin ontology prompt" in prompt
        calls = prompt_manager.render_template.call_args_list
        assert calls[0].kwargs["template_name"] == "flight_ops_system"
        assert calls[1].kwargs["template_name"] == "ontology_analyst_system"

    def test_missing_builtin_template_is_not_masked(self, real_agent_config, mock_llm_create):
        node, _, _ = _make_node(real_agent_config)
        prompt_manager = Mock()
        prompt_manager.render_template.side_effect = FileNotFoundError("missing")
        with patch("datus.prompts.prompt_manager.get_prompt_manager", return_value=prompt_manager):
            with pytest.raises(FileNotFoundError):
                node._get_system_prompt()
        assert prompt_manager.render_template.call_count == 1

    @staticmethod
    def _action(action_type, rows, sql):
        return {
            "action_type": action_type,
            "status": "success",
            "input": {"arguments": "{}"},
            "output": {
                "raw_output": {
                    "success": 1,
                    "result": {
                        "columns": ["alert_id", "priority"],
                        "data": [["A007", "P1"]] * rows,
                        "metadata": {"sql": sql},
                    },
                }
            },
        }

    @pytest.mark.parametrize("last", ["select_records", "query_metrics"])
    def test_update_context_captures_the_last_result_of_either_tool(self, real_agent_config, mock_llm_create, last):
        node, _, _ = _make_node(real_agent_config)
        first = "query_metrics" if last == "select_records" else "select_records"
        node.result = Mock(
            action_history=[self._action(first, 5, "SELECT first"), self._action(last, 2, "SELECT last")]
        )
        workflow = Mock()
        workflow.context.sql_contexts = []

        assert node.update_context(workflow)["success"] is True
        [captured] = workflow.context.sql_contexts
        assert captured.row_count == 2
        assert captured.sql_query == "SELECT last"

    def test_success_result_uses_ontology_result_class(self, real_agent_config, mock_llm_create):
        from datus.schemas.action_history import ActionHistoryManager
        from datus.schemas.ontology_analyst_agentic_node_models import OntologyAnalystNodeResult

        node, _, _ = _make_node(real_agent_config)
        ctx = Mock()
        ctx.response_content = "answer"
        ctx.action_history_manager = ActionHistoryManager()
        result = node._build_success_result(ctx)
        assert isinstance(result, OntologyAnalystNodeResult)
        assert result.response == "answer"
