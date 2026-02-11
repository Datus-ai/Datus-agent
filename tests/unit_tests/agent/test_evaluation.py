from unittest.mock import MagicMock, patch

import pytest

from datus.agent.evaluate import evaluate_result
from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask
from datus.utils.constants import DBType


class TestEvaluation:
    """Test suite for the evaluation module."""

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent config."""
        config = MagicMock(spec=AgentConfig)
        config.namespaces = {"test_db": {"type": "duckdb", "database": ":memory:"}}
        return config

    def test_evaluate_successful_result(self, mock_agent_config):
        """Test evaluation of a successful result."""
        # Create a mock workflow with a SQL task
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="Find employees with salary over $50,000",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        with patch.object(Workflow, "_init_tools", return_value=None):
            workflow = Workflow("test_workflow", task=sql_task, agent_config=mock_agent_config)
            workflow.tools = []

            # Create nodes
            gen_node = Node.new_instance(
                "task1",
                "Generate SQL",
                NodeType.TYPE_GENERATE_SQL,
            )
            exec_node = Node.new_instance(
                "task2",
                "Execute SQL",
                NodeType.TYPE_EXECUTE_SQL,
            )

            workflow.add_node(gen_node)
            workflow.add_node(exec_node)

            # Set current node to first node
            workflow.current_node_index = 0
            gen_node.result = "SELECT * FROM employees WHERE salary > 50000;"
            gen_node.status = "completed"

            # Mock setup_input to return success
            with patch("datus.agent.evaluate.setup_node_input", return_value={"success": True, "suggestions": []}):
                # Evaluate the result
                evaluation = evaluate_result(gen_node, workflow)

                # Check the evaluation
                assert evaluation["success"] is True

    def test_evaluate_failed_result(self, mock_agent_config):
        """Test evaluation of a failed result."""
        # Create a mock workflow
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="Find employees with salary over $50,000",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        with patch.object(Workflow, "_init_tools", return_value=None):
            workflow = Workflow("test_workflow", task=sql_task, agent_config=mock_agent_config)
            workflow.tools = []

            gen_node = Node.new_instance(
                "task1",
                "Generate SQL",
                NodeType.TYPE_GENERATE_SQL,
            )
            exec_node = Node.new_instance(
                "task2",
                "Execute SQL",
                NodeType.TYPE_EXECUTE_SQL,
            )

            workflow.add_node(gen_node)
            workflow.add_node(exec_node)
            workflow.current_node_index = 0

            gen_node.result = "SELECT * FORM employees WHERE salary > 50000;"
            gen_node.status = "failed"

            # Mock setup_input to return failure
            with patch(
                "datus.agent.evaluate.setup_node_input",
                return_value={"success": False, "suggestions": ["Fix SQL syntax error: 'FORM' should be 'FROM'"]},
            ):
                evaluation = evaluate_result(gen_node, workflow)

                assert evaluation["success"] is False
                assert len(evaluation["suggestions"]) == 1
                assert "Fix SQL syntax error" in evaluation["suggestions"][0]

    def test_evaluate_empty_result(self, mock_agent_config):
        """Test evaluation of an empty result."""
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="Find employees with salary over $50,000",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        with patch.object(Workflow, "_init_tools", return_value=None):
            workflow = Workflow("test_workflow", task=sql_task, agent_config=mock_agent_config)
            workflow.tools = []

            gen_node = Node.new_instance(
                "task1",
                "Generate SQL",
                NodeType.TYPE_GENERATE_SQL,
            )
            exec_node = Node.new_instance(
                "task2",
                "Execute SQL",
                NodeType.TYPE_EXECUTE_SQL,
            )

            workflow.add_node(gen_node)
            workflow.add_node(exec_node)
            workflow.current_node_index = 0

            gen_node.result = ""
            gen_node.status = "failed"

            with patch(
                "datus.agent.evaluate.setup_node_input",
                return_value={"success": False, "suggestions": ["No SQL query was generated"]},
            ):
                evaluation = evaluate_result(gen_node, workflow)

                assert evaluation["success"] is False
                assert len(evaluation["suggestions"]) == 1
                assert "No SQL query was generated" in evaluation["suggestions"][0]

    def test_evaluate_error_result(self, mock_agent_config):
        """Test evaluation of a result containing an error message."""
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="SELECT * FROM employees WHERE salary > 50000;",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        with patch.object(Workflow, "_init_tools", return_value=None):
            workflow = Workflow("test_workflow", task=sql_task, agent_config=mock_agent_config)
            workflow.tools = []

            exec_node = Node.new_instance(
                "task1",
                "Execute SQL",
                NodeType.TYPE_EXECUTE_SQL,
            )
            output_node = Node.new_instance(
                "task2",
                "Output",
                NodeType.TYPE_OUTPUT,
            )

            workflow.add_node(exec_node)
            workflow.add_node(output_node)
            workflow.current_node_index = 0

            exec_node.result = {"error": "Table 'employees' not found"}
            exec_node.status = "failed"

            with patch(
                "datus.agent.evaluate.setup_node_input",
                return_value={"success": False, "suggestions": ["Create or verify the 'employees' table exists"]},
            ):
                evaluation = evaluate_result(exec_node, workflow)

                assert evaluation["success"] is False
                assert len(evaluation["suggestions"]) == 1
                assert "Create or verify" in evaluation["suggestions"][0]

    def test_evaluate_different_task_types(self, mock_agent_config):
        """Test evaluation of results from different task types."""
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="Find employees with salary over $50,000",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        with patch.object(Workflow, "_init_tools", return_value=None):
            workflow = Workflow("test_workflow", task=sql_task, agent_config=mock_agent_config)
            workflow.tools = []

            # Test SQL generation task
            sql_gen_task = Node.new_instance(
                "task1",
                "Generate SQL",
                NodeType.TYPE_GENERATE_SQL,
            )
            sql_gen_task.result = "SELECT * FROM employees WHERE salary > 50000;"
            sql_gen_task.status = "completed"

            # Test SQL execution task
            sql_exec_task = Node.new_instance(
                "task2",
                "Execute SQL",
                NodeType.TYPE_EXECUTE_SQL,
            )
            sql_exec_task.result = [{"id": 1, "name": "John Doe", "salary": 75000}]
            sql_exec_task.status = "completed"

            # Test schema linking task
            schema_task = Node.new_instance(
                "task3",
                "Schema Linking",
                NodeType.TYPE_SCHEMA_LINKING,
            )
            schema_task.result = {
                "schemas": ["employees"],
                "columns": ["id", "name", "salary"],
            }
            schema_task.status = "completed"

            output_node = Node.new_instance("task_output", "Output", NodeType.TYPE_OUTPUT)

            workflow.add_node(sql_gen_task)
            workflow.add_node(sql_exec_task)
            workflow.add_node(schema_task)
            workflow.add_node(output_node)

            # Mock setup_input to always return success
            with patch("datus.agent.evaluate.setup_node_input", return_value={"success": True, "suggestions": []}):
                # Evaluate each task
                workflow.current_node_index = 0
                sql_gen_eval = evaluate_result(sql_gen_task, workflow)

                workflow.current_node_index = 1
                sql_exec_eval = evaluate_result(sql_exec_task, workflow)

                workflow.current_node_index = 2
                schema_eval = evaluate_result(schema_task, workflow)

                # Check that each evaluation was successful
                assert sql_gen_eval["success"] is True
                assert sql_exec_eval["success"] is True
                assert schema_eval["success"] is True
