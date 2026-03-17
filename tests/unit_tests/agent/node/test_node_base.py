# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for Node base class (node.py) - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.base import BaseResult
from datus.schemas.node_models import ExecuteSQLResult, GenerateSQLResult
from datus.schemas.schema_linking_node_models import SchemaLinkingResult


def make_mock_agent_config():
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_namespace = "test"
    cfg.nodes = {}
    cfg.custom_workflows = {}
    return cfg


class TestNodeInstantiation:
    """Test Node.new_instance factory for all registered types."""

    def setup_method(self):
        self.agent_config = make_mock_agent_config()

    def test_new_instance_schema_linking(self):
        node = Node.new_instance("n1", "desc", NodeType.TYPE_SCHEMA_LINKING, agent_config=self.agent_config)
        assert node.id == "n1"
        assert node.type == NodeType.TYPE_SCHEMA_LINKING
        assert node.status == "pending"

    def test_new_instance_generate_sql(self):
        node = Node.new_instance("n2", "desc", NodeType.TYPE_GENERATE_SQL, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_GENERATE_SQL

    def test_new_instance_execute_sql(self):
        node = Node.new_instance("n3", "desc", NodeType.TYPE_EXECUTE_SQL, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_EXECUTE_SQL

    def test_new_instance_reasoning(self):
        node = Node.new_instance("n4", "desc", NodeType.TYPE_REASONING, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_REASONING

    def test_new_instance_output(self):
        node = Node.new_instance("n5", "desc", NodeType.TYPE_OUTPUT, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_OUTPUT

    def test_new_instance_fix(self):
        node = Node.new_instance("n6", "desc", NodeType.TYPE_FIX, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_FIX

    def test_new_instance_reflect(self):
        node = Node.new_instance("n7", "desc", NodeType.TYPE_REFLECT, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_REFLECT

    def test_new_instance_begin(self):
        node = Node.new_instance("n8", "desc", NodeType.TYPE_BEGIN, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_BEGIN

    def test_new_instance_parallel(self):
        node = Node.new_instance("n9", "desc", NodeType.TYPE_PARALLEL, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_PARALLEL

    def test_new_instance_selection(self):
        node = Node.new_instance("n10", "desc", NodeType.TYPE_SELECTION, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_SELECTION

    def test_new_instance_subworkflow(self):
        node = Node.new_instance("n11", "desc", NodeType.TYPE_SUBWORKFLOW, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_SUBWORKFLOW

    def test_new_instance_date_parser(self):
        node = Node.new_instance("n12", "desc", NodeType.TYPE_DATE_PARSER, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_DATE_PARSER

    def test_new_instance_hitl(self):
        node = Node.new_instance("n13", "desc", NodeType.TYPE_HITL, agent_config=self.agent_config)
        assert node.type == NodeType.TYPE_HITL

    def test_new_instance_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid node type"):
            Node.new_instance("nx", "desc", "nonexistent_type", agent_config=self.agent_config)

    def test_node_invalid_type_in_constructor(self):
        """Node constructor validates type against ACTION_TYPES and CONTROL_TYPES."""
        from datus.agent.node.generate_sql_node import GenerateSQLNode

        with pytest.raises(ValueError, match="Invalid node type"):
            GenerateSQLNode("nx", "desc", "bad_type")


class TestNodeLifecycle:
    """Test node lifecycle: start, complete, fail."""

    def setup_method(self):
        self.agent_config = make_mock_agent_config()

    def _make_node(self):
        return Node.new_instance("life_test", "Test", NodeType.TYPE_SCHEMA_LINKING, agent_config=self.agent_config)

    def test_initial_state(self):
        node = self._make_node()
        assert node.status == "pending"
        assert node.result is None
        assert node.start_time is None
        assert node.end_time is None
        assert node.dependencies == []
        assert node.metadata == {}

    def test_start(self):
        node = self._make_node()
        node.start()
        assert node.status == "running"
        assert node.start_time is not None

    def test_complete_success(self):
        node = self._make_node()
        result = BaseResult(success=True)
        node.complete(result)
        assert node.status == "completed"
        assert node.result == result
        assert node.end_time is not None

    def test_complete_failure(self):
        node = self._make_node()
        result = BaseResult(success=False, error="oops")
        node.complete(result)
        assert node.status == "failed"

    def test_fail_with_error(self):
        node = self._make_node()
        node.fail("some error")
        assert node.status == "failed"
        assert node.result.error == "some error"
        assert node.end_time is not None

    def test_fail_without_error(self):
        node = self._make_node()
        node.fail()
        assert node.status == "failed"
        assert node.result is None

    def test_add_dependency(self):
        node = self._make_node()
        node.add_dependency("dep_a")
        node.add_dependency("dep_b")
        assert "dep_a" in node.dependencies
        assert "dep_b" in node.dependencies
        assert len(node.dependencies) == 2

    def test_add_dependency_no_duplicates(self):
        node = self._make_node()
        node.add_dependency("dep_a")
        node.add_dependency("dep_a")
        assert len(node.dependencies) == 1

    def test_to_dict(self):
        node = self._make_node()
        d = node.to_dict()
        assert d["id"] == "life_test"
        assert d["type"] == NodeType.TYPE_SCHEMA_LINKING
        assert d["status"] == "pending"
        assert "dependencies" in d
        assert "metadata" in d

    def test_to_dict_with_result(self):
        node = self._make_node()
        result = BaseResult(success=True)
        node.complete(result)
        d = node.to_dict()
        assert d["status"] == "completed"


class TestNodeRun:
    """Test Node.run() method through a mock concrete node."""

    def setup_method(self):
        self.agent_config = make_mock_agent_config()

    def test_run_calls_execute_and_completes(self):
        """Node.run() should call _initialize, start, execute and complete."""
        node = Node.new_instance("run_test", "Test", NodeType.TYPE_SCHEMA_LINKING, agent_config=self.agent_config)
        # Patch _initialize to avoid LLM setup, execute to return a success result
        mock_result = SchemaLinkingResult(
            success=True, table_schemas=[], table_values=[], schema_count=0, value_count=0
        )
        node._initialize = MagicMock()
        node.execute = MagicMock(side_effect=lambda: setattr(node, "result", mock_result))

        result = node.run()
        node._initialize.assert_called_once()
        node.execute.assert_called_once()
        assert node.status == "completed"
        assert result.success is True

    def test_run_fails_on_exception(self):
        node = Node.new_instance("run_fail", "Test", NodeType.TYPE_SCHEMA_LINKING, agent_config=self.agent_config)
        node._initialize = MagicMock()
        node.execute = MagicMock(side_effect=RuntimeError("boom"))

        result = node.run()
        assert node.status == "failed"
        assert result is not None

    def test_run_fails_when_result_is_none(self):
        node = Node.new_instance("run_none", "Test", NodeType.TYPE_GENERATE_SQL, agent_config=self.agent_config)
        node._initialize = MagicMock()
        node.execute = MagicMock()  # Does not set node.result

        node.run()
        # result is None → node should fail
        assert node.status == "failed"


class TestNodeInitialize:
    """Test Node._initialize() model setup."""

    def setup_method(self):
        self.agent_config = make_mock_agent_config()

    def test_initialize_with_model_already_set_as_llm(self):
        """If model is already an LLMBaseModel, _initialize is a no-op."""
        from datus.models.base import LLMBaseModel

        node = Node.new_instance("init_test", "Test", NodeType.TYPE_GENERATE_SQL, agent_config=self.agent_config)
        mock_model = MagicMock(spec=LLMBaseModel)
        node.model = mock_model

        with patch("datus.agent.node.node.LLMBaseModel.create_model") as mock_create:
            node._initialize()
            mock_create.assert_not_called()

    def test_initialize_with_model_name_string(self):
        """If model is a string, _initialize creates the model via LLMBaseModel.create_model."""
        node = Node.new_instance("init_str", "Test", NodeType.TYPE_GENERATE_SQL, agent_config=self.agent_config)
        node.model = "gpt-4"

        mock_llm = MagicMock()
        mock_llm.model_config.type = "openai"
        mock_llm.model_config.model = "gpt-4"
        mock_llm.model_config.save_llm_trace = False

        with patch("datus.agent.node.node.LLMBaseModel.create_model", return_value=mock_llm) as mock_create:
            node._initialize()
            mock_create.assert_called_once_with(model_name="gpt-4", agent_config=self.agent_config)
            assert node.model == mock_llm

    def test_initialize_from_agent_config_nodes(self):
        """If model is None, use agent_config.nodes to find model name."""
        node_config = MagicMock()
        node_config.model = "claude-3"
        node_config.input = None
        self.agent_config.nodes = {NodeType.TYPE_GENERATE_SQL: node_config}

        node = Node.new_instance("init_cfg", "Test", NodeType.TYPE_GENERATE_SQL, agent_config=self.agent_config)

        mock_llm = MagicMock()
        mock_llm.model_config.type = "anthropic"
        mock_llm.model_config.model = "claude-3"
        mock_llm.model_config.save_llm_trace = False

        with patch("datus.agent.node.node.LLMBaseModel.create_model", return_value=mock_llm):
            node._initialize()
            assert node.model == mock_llm


class TestNodeFromDict:
    """Test Node.from_dict round-trip."""

    def setup_method(self):
        self.agent_config = make_mock_agent_config()

    def test_from_dict_schema_linking(self):
        node_dict = {
            "id": "fd_test",
            "description": "From dict test",
            "type": NodeType.TYPE_SCHEMA_LINKING,
            "input": None,
            "status": "pending",
            "result": None,
            "start_time": None,
            "end_time": None,
            "dependencies": [],
            "metadata": {},
        }
        node = Node.from_dict(node_dict, agent_config=self.agent_config)
        assert node.id == "fd_test"
        assert node.type == NodeType.TYPE_SCHEMA_LINKING

    def test_from_dict_restores_status_and_result(self):
        node_dict = {
            "id": "fd2",
            "description": "test",
            "type": NodeType.TYPE_EXECUTE_SQL,
            "input": {"sql_query": "SELECT 1", "database_name": "db"},
            "status": "completed",
            "result": {
                "success": True,
                "error": None,
                "sql_query": "SELECT 1",
                "row_count": 1,
                "sql_return": "1",
                "result_format": "",
                "action_history": None,
                "execution_stats": None,
            },
            "start_time": 1000.0,
            "end_time": 1001.0,
            "dependencies": ["dep_x"],
            "metadata": {"key": "val"},
        }
        node = Node.from_dict(node_dict, agent_config=self.agent_config)
        assert node.status == "completed"
        assert node.dependencies == ["dep_x"]
        assert node.metadata == {"key": "val"}
        assert isinstance(node.result, ExecuteSQLResult)

    def test_from_dict_with_generate_sql_result(self):
        node_dict = {
            "id": "fd3",
            "description": "test",
            "type": NodeType.TYPE_GENERATE_SQL,
            "input": None,
            "status": "completed",
            "result": {
                "success": True,
                "error": None,
                "sql_query": "SELECT 1",
                "tables": ["t1"],
                "explanation": "test",
                "action_history": None,
                "execution_stats": None,
            },
            "start_time": None,
            "end_time": None,
            "dependencies": [],
            "metadata": {},
        }
        node = Node.from_dict(node_dict, agent_config=self.agent_config)
        assert isinstance(node.result, GenerateSQLResult)

    def test_from_dict_handles_corrupt_input_gracefully(self):
        """from_dict should not raise even if result dict is malformed."""
        node_dict = {
            "id": "fd_bad",
            "description": "test",
            "type": NodeType.TYPE_GENERATE_SQL,
            "input": {"invalid_key": "oops"},
            "status": "failed",
            "result": {"garbage": "data"},
            "start_time": None,
            "end_time": None,
            "dependencies": [],
            "metadata": {},
        }
        # Should not raise
        node = Node.from_dict(node_dict, agent_config=self.agent_config)
        assert node.id == "fd_bad"
