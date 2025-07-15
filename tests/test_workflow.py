from datus.agent.node import Node
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.schemas.node_models import SqlTask
from datus.utils.constants import LLMProvider


class TestNode:
    # Save workflow for debugging
    WORKFLOW_SAVE_PATH = "tests/test_workflow.yaml"

    """Test suite for the Node class."""

    def test_node_initialization(self):
        """Test that a Node initializes with the correct attributes."""
        node = Node.new_instance(
            node_id="test_node",
            description="Test node",
            node_type="sql_generation",
            input_data={"query": "SELECT * FROM users"},
        )

        assert node.id == "test_node"
        assert node.description == "Test node"
        assert node.type == "sql_generation"
        assert node.input == {"query": "SELECT * FROM users"}
        assert node.status == "pending"
        assert node.result is None
        assert node.start_time is None
        assert node.end_time is None
        assert node.dependencies == []
        assert node.metadata == {}

    def test_node_state_transitions(self):
        """Test node state transitions (start, complete, fail)."""
        node = Node.new_instance("test_node", "Test node", "sql_generation")

        # Test start transition
        node.start()
        assert node.status == "running"
        assert node.start_time is not None

        # Test complete transition
        result = {"data": [1, 2, 3]}
        node.complete(result)
        assert node.status == "completed"
        assert node.result == result
        assert node.end_time is not None

        # Test fail transition
        node = Node.new_instance("test_node", "Test node", "sql_generation")
        error_msg = "Test error"
        node.fail(error_msg)
        assert node.status == "failed"
        assert node.result == {"error": error_msg}
        assert node.end_time is not None

    def test_node_dependencies(self):
        """Test adding dependencies to a node."""
        node = Node.new_instance("test_node", "Test node", "sql_generation")

        # Test adding dependencies
        node.add_dependency("dep_1")
        node.add_dependency("dep_2")
        assert "dep_1" in node.dependencies
        assert "dep_2" in node.dependencies

        # Test duplicate dependency
        node.add_dependency("dep_1")
        assert len(node.dependencies) == 2

    def test_node_to_dict(self):
        """Test converting a node to dictionary representation."""
        node = Node.new_instance(
            node_id="test_node",
            description="Test node",
            node_type="sql_generation",
            input_data={"query": "SELECT * FROM users"},
        )

        node_dict = node.to_dict()
        assert node_dict["id"] == "test_node"
        assert node_dict["description"] == "Test node"
        assert node_dict["type"] == "sql_generation"
        assert node_dict["status"] == "pending"
        assert node_dict["result"] is None
        assert node_dict["dependencies"] == []
        assert node_dict["metadata"] == {}

    def test_control_node_types(self):
        """Test initialization of different control node types."""
        # Test evaluate node type
        evaluate_node = Node.new_instance(
            node_id="eval_node",
            description="Evaluation node",
            node_type=Node.TYPE_EVALUATE,
            input_data={"result": "test_result"},
        )
        assert evaluate_node.type == Node.TYPE_EVALUATE

        # Test HITL node type
        hitl_node = Node.new_instance(
            node_id="hitl_node",
            description="HITL node",
            node_type=Node.TYPE_HITL,
            input_data={"prompt": "test_prompt"},
        )
        assert hitl_node.type == Node.TYPE_HITL

        # Test subplan node type
        subplan_node = Node.new_instance(
            node_id="subplan_node",
            description="Subplan node",
            node_type=Node.TYPE_SUBPLAN,
            input_data={"plan": "test_plan"},
        )
        assert subplan_node.type == Node.TYPE_SUBPLAN

    def test_node_run_tool_type(self, mocker):
        """Test running a tool-type node."""
        # Mock the tool execution
        mock_tool = mocker.MagicMock()
        mock_tool.execute.return_value = {"result": "success"}

        # Mock the get_tool function
        mocker.patch("tools.get_tool", return_value=mock_tool)

        # Create and run a tool-type node
        node = Node.new_instance(
            node_id="tool_node",
            description="Tool node",
            node_type="sql_generation",
            input_data={"query": "SELECT * FROM users"},
        )

        result = node.run()
        assert result == {"result": "success"}
        assert node.status == "completed"
        mock_tool.execute.assert_called_once_with({"query": "SELECT * FROM users"})

    def test_node_run_evaluate_type(self, mocker):
        """Test running an evaluate-type node."""
        # Mock the evaluate_result function
        mock_evaluate = mocker.patch(
            "datus.agent.evaluate.evaluate_result",
            return_value={"evaluation": "passed"},
        )

        # Create and run an evaluate node
        node = Node.new_instance(
            node_id="eval_node",
            description="Evaluation node",
            node_type=Node.TYPE_EVALUATE,
            input_data={"result": "test_result"},
        )

        result = node.run()
        assert result == {"evaluation": "passed"}
        assert node.status == "completed"
        mock_evaluate.assert_called_once_with({"result": "test_result"})

    def test_node_run_subplan_type(self, mocker):
        """Test running a subplan-type node."""
        # Mock a workflow
        mock_workflow = mocker.MagicMock()
        mock_workflow.run.return_value = {"status": "completed"}
        mock_workflow.__class__.__name__ = "Workflow"

        # Create and run a subplan node
        node = Node.new_instance(
            node_id="subplan_node",
            description="Subplan node",
            node_type=Node.TYPE_SUBPLAN,
            input_data=mock_workflow,
        )

        result = node.run()
        assert result == {"status": "completed"}
        assert node.status == "completed"
        mock_workflow.run.assert_called_once()

    def test_node_run_failure(self):
        """Test node run failure scenarios."""
        # Test failure when tool is not found
        node = Node.new_instance(
            node_id="invalid_node",
            description="Invalid node",
            node_type="invalid_type",
            input_data={},
        )

        result = node.run()
        assert node.status == "failed"
        assert result["error"] == "Tool not found: invalid_type"

        # Test failure with invalid subplan input
        node = Node.new_instance(
            node_id="invalid_subplan",
            description="Invalid subplan",
            node_type=Node.TYPE_SUBPLAN,
            input_data="invalid_workflow",
        )

        result = node.run()
        assert node.status == "failed"
        assert result["error"] == "Invalid sub-workflow input"

    def test_sql_generation_node(self, mocker):
        """Test SQL generation node with schema linking output."""
        # Use real DeepSeek model for testing
        # from tools.llms_tools.generate_sql import GenerateSQLTool
        # sql_tool = GenerateSQLTool()

        # Create input data from schema linking output
        input_data = {
            "success": True,
            "similar_schemas": [
                {
                    "table_name": "supplier",
                    "definition": "CREATE TABLE supplier (\ns_suppkey INT PRIMARY KEY,s_name TEXT,s_address TEXT,"
                    "s_city TEXT,s_nation TEXT,s_region TEXT,s_phone TEXT\n);",
                    "database_name": "main",
                    "created_at": "2025-03-28T02:28:28.900973",
                    "_distance": 1.4497300386428833,
                }
            ],
            "count": 1,
            "llm": LLMProvider.DEEPSEEK,
        }

        # Create and run SQL generation node
        node = Node.new_instance(
            node_id="sql_gen",
            description="SQL Generation",
            node_type="sql_generation",
            input_data=input_data,
        )

        # Test node initialization
        assert node.id == "sql_gen"
        assert node.type == "sql_generation"
        assert node.input == input_data
        assert node.status == "pending"

        # Run the node and verify results
        result = node.run()
        assert "sql" in result
        assert isinstance(result["sql"], str)
        assert node.status == "completed"


class TestWorkflow:
    # Save workflow for debugging
    WORKFLOW_SAVE_PATH = "tests/test_workflow.yaml"

    """Test suite for the Workflow class."""

    def test_create_default_workflow(self):
        """Test the default workflow creation with core nodes using plan._create_default_workflow."""

        # Initialize with specified parameters
        task = (
            "Calculate the total annual revenue from 1992 to 1997 for each "
            "combination of customer city and supplier city, where the customer is "
            "located in the U.S. cities 'UNITED KI1' or 'UNITED KI5', "
            "and the supplier is also located in these two cities. "
            "The results should be sorted in ascending order by year and, "
            "within each year, in descending order by revenue."
        )

        # Create workflow using plan.py's method
        workflow = Workflow("default_workflow", "Test default workflow")
        workflow_gen = generate_workflow(SqlTask(task=task), "reflection")
        nodes = workflow_gen.nodes
        for node in nodes:
            workflow.add_node(node)

        # Verify core nodes exist
        expected_nodes = [
            ("node_0", "Beginning of the workflow", "start"),
            ("node_1", "Understand the query and find related schemas", "schema_linking"),
            ("node_2", "Generate SQL query", "generate_sql"),
            ("node_3", "Execute SQL query", "execute_sql"),
            ("node_5", "Return the results to the user", "output"),
        ]

        # Check node properties
        for node_id, description, node_type in expected_nodes:
            node = workflow.get_node(node_id)
            assert node is not None
            assert node.description == description
            assert node.type == node_type

        # Verify workflow metadata
        assert workflow.status == "pending"
        assert workflow.current_node_index == 0
        assert len(workflow.nodes) == 5

        # workflow.save(self.WORKFLOW_SAVE_PATH)

    # def test_e2e_workflow(self):
    #    """Test the end-to-end workflow execution."""
    #    # Initialize with specified parameters
    #    task = "Calculate the total annual revenue from 1992 to 1997 for each combination of
    #    customer city and supplier city, where the customer is located in the U.S.
    #    cities 'UNITED KI1' or 'UNITED KI5', and the supplier is also located in these two cities.
    #    The results should be sorted in ascending order by year and,
    #    within each year, in descending order by revenue."
    #
    #    # Create workflow using plan.py's method
    #    workflow = Workflow("nl2sql_workflow", "Test a e2e workflow")
    #    nodes = _create_default_workflow(task)
    #    for node in nodes:
    #        workflow.add_node(node)

    def test_workflow_save_load(self, tmp_path):
        """Test saving and loading a workflow with multiple nodes."""
        # Create a workflow with multiple nodes
        workflow = Workflow(name="test_workflow", task=SqlTask(task="Test workflow save/load"), db_conn="test_conn")

        node1 = Node.new_instance(
            node_id="node1",
            description="First node",
            node_type="sql_generation",
            input_data={"query": "SELECT * FROM users"},
        )

        node2 = Node.new_instance(
            node_id="node2",
            description="Second node",
            node_type="execute_sql",
            input_data={"sql": "SELECT * FROM users"},
        )

        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.move_node("node2", 0)
        workflow.status = "running"
        workflow.current_node_index = 1
        workflow.metadata = {"test_key": "test_value"}

        # Save workflow to temporary file
        save_path = tmp_path / "test_workflow.yaml"
        # save_path = "./tests/test_workflow.yaml"
        workflow.save(save_path)

        # Load workflow from file
        loaded_workflow = Workflow.load(save_path)

        # Verify all properties are correctly restored
        assert loaded_workflow.name == "test_workflow"
        assert loaded_workflow.description == "Test workflow save/load"
        assert loaded_workflow.db_conn == "test_conn"
        assert loaded_workflow.status == "running"
        assert loaded_workflow.current_node_index == 1
        assert loaded_workflow.metadata == {"test_key": "test_value"}

        # Verify nodes are correctly restored
        assert len(loaded_workflow.nodes) == 2
        assert "node1" in loaded_workflow.nodes
        assert "node2" in loaded_workflow.nodes
        assert loaded_workflow.node_order == ["node2", "node1"]

        # Verify node properties
        loaded_node1 = loaded_workflow.nodes["node1"]
        assert loaded_node1.description == "First node"
        assert loaded_node1.type == "sql_generation"
        assert loaded_node1.input == {"query": "SELECT * FROM users"}

        loaded_node2 = loaded_workflow.nodes["node2"]
        assert loaded_node2.description == "Second node"
        assert loaded_node2.type == "execute_sql"
        assert loaded_node2.input == {"sql": "SELECT * FROM users"}
