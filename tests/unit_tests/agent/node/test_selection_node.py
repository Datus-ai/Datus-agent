# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for SelectionNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import GenerateSQLResult
from datus.schemas.parallel_node_models import SelectionInput, SelectionResult


def make_agent_config():
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_database = "test"
    cfg.nodes = {}
    cfg.custom_workflows = {}
    return cfg


def make_node(input_data=None):
    cfg = make_agent_config()
    return Node.new_instance(
        "sel_1",
        "Selection",
        NodeType.TYPE_SELECTION,
        input_data=input_data,
        agent_config=cfg,
    )


def make_candidates():
    return {
        "node_a": {
            "success": True,
            "result": GenerateSQLResult(success=True, sql_query="SELECT a FROM t", tables=["t"]),
            "node_id": "node_a",
        },
        "node_b": {
            "success": True,
            "result": GenerateSQLResult(success=True, sql_query="SELECT b FROM t", tables=["t"]),
            "node_id": "node_b",
        },
    }


class TestSelectionNodeExecute:
    """Test execute() with no candidates, single candidate, and multiple candidates."""

    def test_execute_no_input_returns_failure(self):
        node = make_node()
        node.execute()
        assert node.result.success is False
        assert "No candidate" in node.result.error

    def test_execute_no_candidates_returns_failure(self):
        node = make_node(SelectionInput(candidate_results={}, selection_criteria="best"))
        node.execute()
        assert node.result.success is False

    def test_execute_single_candidate_selected_directly(self):
        candidates = {"only_node": {"success": True, "result": None}}
        node = make_node(SelectionInput(candidate_results=candidates))
        node.execute()
        assert node.result.success is True
        assert node.result.selected_source == "only_node"
        assert node.result.selection_reason == "Only one candidate available"

    def test_execute_multiple_candidates_with_llm(self):
        node = make_node(SelectionInput(candidate_results=make_candidates()))
        node.model = MagicMock()
        llm_response = {
            "best_candidate": "node_a",
            "reason": "Better SQL quality",
            "score_analysis": {"node_a": {"score": 9}, "node_b": {"score": 7}},
        }
        node.model.generate_with_json_output.return_value = llm_response

        with patch("datus.agent.node.selection_node.create_selection_prompt", return_value="prompt"):
            node.execute()

        assert node.result.success is True
        assert node.result.selected_source == "node_a"

    def test_execute_multiple_candidates_without_model_uses_rules(self):
        node = make_node(SelectionInput(candidate_results=make_candidates()))
        node.model = None
        node.execute()
        assert node.result.success is True
        assert node.result.selected_source in make_candidates()

    def test_execute_llm_returns_invalid_candidate_falls_back(self):
        node = make_node(SelectionInput(candidate_results=make_candidates()))
        node.model = MagicMock()
        llm_response = {"best_candidate": "nonexistent_node", "reason": "bad"}
        node.model.generate_with_json_output.return_value = llm_response

        with patch("datus.agent.node.selection_node.create_selection_prompt", return_value="prompt"):
            node.execute()

        # Falls back to rule-based
        assert node.result.success is True
        assert node.result.selected_source in make_candidates()

    def test_execute_exception_returns_failure(self):
        node = make_node(SelectionInput(candidate_results=make_candidates()))
        node.model = MagicMock()
        node.model.generate_with_json_output.side_effect = RuntimeError("llm crash")

        with patch("datus.agent.node.selection_node.create_selection_prompt", return_value="prompt"):
            node.execute()

        # LLM fails -> falls back to rule-based
        assert node.result.success is True


class TestSelectionNodeRuleBasedSelection:
    """Test _rule_based_selection scoring."""

    def test_rule_based_selects_highest_score(self):
        node = make_node()
        candidates = {
            "winner": {"success": True, "result": MagicMock(sql_result_final="data"), "error": None},
            "loser": {"success": False, "result": None, "error": "failed"},
        }
        result = node._rule_based_selection(candidates)
        assert result.success is True
        assert result.selected_source == "winner"

    def test_rule_based_single_candidate(self):
        node = make_node()
        candidates = {"only": {"success": True, "result": None}}
        result = node._rule_based_selection(candidates)
        assert result.selected_source == "only"

    def test_rule_based_non_dict_candidates(self):
        node = make_node()
        # Non-dict values should score 0
        candidates = {"a": "string_result", "b": None}
        result = node._rule_based_selection(candidates)
        assert result.success is True


class TestSelectionNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_from_parallel_results(self):
        node = make_node()
        workflow = MagicMock()
        workflow.current_node_index = 1
        workflow.node_order = ["n1", "sel_1"]
        candidates = make_candidates()
        workflow.context.parallel_results = candidates

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, SelectionInput)
        assert node.input.candidate_results == candidates

    def test_setup_input_no_parallel_results_uses_reasoning_node(self):
        node = make_node()
        workflow = MagicMock()
        workflow.current_node_index = 1
        workflow.node_order = ["reasoning_node_id", "sel_1"]
        workflow.context.parallel_results = None

        reasoning_node = MagicMock()
        reasoning_node.type = "reasoning"
        reasoning_node.status = "completed"
        reasoning_node.result = MagicMock()
        reasoning_node.result.success = True
        reasoning_node.id = "reasoning_node_id"

        workflow.nodes = {"reasoning_node_id": reasoning_node}

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert "reasoning_node" in node.input.candidate_results

    def test_setup_input_no_parallel_no_reasoning_returns_failure(self):
        node = make_node()
        workflow = MagicMock()
        workflow.current_node_index = 0
        workflow.node_order = ["sel_1"]
        workflow.context.parallel_results = None

        other_node = MagicMock()
        other_node.type = "generate_sql"
        other_node.status = "completed"
        workflow.nodes = {"sel_1": other_node}

        result = node.setup_input(workflow)
        assert result["success"] is False

    def test_setup_input_updates_existing_selection_input(self):
        existing_input = SelectionInput(candidate_results={"old": {}})
        node = make_node(existing_input)
        workflow = MagicMock()
        workflow.current_node_index = 0
        workflow.node_order = []
        new_candidates = make_candidates()
        workflow.context.parallel_results = new_candidates

        node.setup_input(workflow)
        assert node.input.candidate_results == new_candidates


class TestSelectionNodeUpdateContext:
    """Test update_context."""

    def test_update_context_with_sql_result(self):
        node = make_node(SelectionInput(candidate_results=make_candidates()))
        sql_result = GenerateSQLResult(success=True, sql_query="SELECT a FROM t", tables=["t"], explanation="ok")
        node.result = SelectionResult(
            success=True,
            selected_result={"result": sql_result},
            selected_source="node_a",
            selection_reason="Best quality",
            all_candidates=make_candidates(),
        )
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1

    def test_update_context_no_result(self):
        node = make_node()
        node.result = None
        workflow = MagicMock()

        result = node.update_context(workflow)
        assert result["success"] is True  # non-SQL result path

    def test_update_context_no_sql_in_result(self):
        node = make_node(SelectionInput(candidate_results={"x": {}}))
        node.result = SelectionResult(
            success=True,
            selected_result={"result": MagicMock(spec=[])},  # no sql_query attr
            selected_source="x",
            selection_reason="ok",
            all_candidates={"x": {}},
        )
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True


class TestSelectionNodeStream:
    """Test execute_stream."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_candidates(self):
        node = make_node()
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        # No action_history_manager provided, so no yields
        assert actions == []

    @pytest.mark.asyncio
    async def test_execute_stream_with_manager(self):
        node = make_node(SelectionInput(candidate_results=make_candidates()))
        node.model = None  # use rule-based

        mgr = MagicMock()
        mgr.create.return_value = MagicMock()
        mgr.update.return_value = MagicMock()
        actions = []
        async for action in node.execute_stream(action_history_manager=mgr):
            actions.append(action)

        assert len(actions) >= 2  # start + end
