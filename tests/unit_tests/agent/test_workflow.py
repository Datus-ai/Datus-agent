from datus.agent.node import Node
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask


class TestNode:
    """Test suite for the Node class."""

    def test_node_initialization(self):
        """Test that a Node initializes with the correct attributes."""
        node = Node.new_instance(
            node_id="test_node",
            description="Test node",
            node_type=NodeType.TYPE_GENERATE_SQL,
            input_data=None,
        )

        assert node.id == "test_node"
        assert node.description == "Test node"
        assert node.type == NodeType.TYPE_GENERATE_SQL
        assert node.input is None
        assert node.status == "pending"
        assert node.result is None
        assert node.start_time is None
        assert node.end_time is None
        assert node.dependencies == []
        assert node.metadata == {}

    def test_node_state_transitions(self):
        """Test node state transitions (start, complete, fail)."""
        from datus.schemas.node_models import BaseResult

        node = Node.new_instance("test_node", "Test node", NodeType.TYPE_GENERATE_SQL)

        # Test start transition
        node.start()
        assert node.status == "running"
        assert node.start_time is not None

        # Test complete transition
        result = BaseResult(success=True)
        node.complete(result)
        assert node.status == "completed"
        assert node.result == result
        assert node.end_time is not None

        # Test fail transition
        node = Node.new_instance("test_node", "Test node", NodeType.TYPE_GENERATE_SQL)
        error_msg = "Test error"
        node.fail(error_msg)
        assert node.status == "failed"
        assert node.result.success is False
        assert node.result.error == error_msg
        assert node.end_time is not None

    def test_node_dependencies(self):
        """Test adding dependencies to a node."""
        node = Node.new_instance("test_node", "Test node", NodeType.TYPE_GENERATE_SQL)

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
            node_type=NodeType.TYPE_GENERATE_SQL,
            input_data=None,
        )

        node_dict = node.to_dict()
        assert node_dict["id"] == "test_node"
        assert node_dict["description"] == "Test node"
        assert node_dict["type"] == NodeType.TYPE_GENERATE_SQL
        assert node_dict["status"] == "pending"
        assert node_dict["result"] is None
        assert node_dict["dependencies"] == []
        assert node_dict["metadata"] == {}

    def test_control_node_types(self):
        """Test initialization of different control node types."""
        # Test HITL node type
        hitl_node = Node.new_instance(
            node_id="hitl_node",
            description="HITL node",
            node_type=NodeType.TYPE_HITL,
            input_data=None,
        )
        assert hitl_node.type == NodeType.TYPE_HITL

        # Test reflect node type
        reflect_node = Node.new_instance(
            node_id="reflect_node",
            description="Reflect node",
            node_type=NodeType.TYPE_REFLECT,
            input_data=None,
        )
        assert reflect_node.type == NodeType.TYPE_REFLECT

        # Test subworkflow node type
        subworkflow_node = Node.new_instance(
            node_id="subworkflow_node",
            description="Subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            input_data=None,
        )
        assert subworkflow_node.type == NodeType.TYPE_SUBWORKFLOW

    def test_node_run_failure(self):
        """Test node run failure scenarios."""
        import pytest

        # Test failure when node type is invalid - should raise ValueError during creation
        with pytest.raises(ValueError, match="Invalid node type"):
            Node.new_instance(
                node_id="invalid_node",
                description="Invalid node",
                node_type="invalid_type",
                input_data=None,
            )


class TestWorkflow:
    """Test suite for the Workflow class."""

    def test_create_default_workflow(self, real_agent_config):
        """Test the default workflow creation with core nodes using plan._create_default_workflow."""

        # Initialize with specified parameters
        task_text = (
            "Calculate the total annual revenue from 1992 to 1997 for each "
            "combination of customer city and supplier city, where the customer is "
            "located in the U.S. cities 'UNITED KI1' or 'UNITED KI5', "
            "and the supplier is also located in these two cities. "
            "The results should be sorted in ascending order by year and, "
            "within each year, in descending order by revenue."
        )

        # Create workflow using plan.py's method
        task = SqlTask(task=task_text)
        workflow = generate_workflow(task, "reflection", agent_config=real_agent_config)

        # Verify core nodes exist
        expected_nodes = [
            ("node_0", "Beginning of the workflow", NodeType.TYPE_BEGIN),
            ("node_1", "Understand the query and find related schemas", NodeType.TYPE_SCHEMA_LINKING),
            ("node_2", "Generate SQL query", NodeType.TYPE_GENERATE_SQL),
            ("node_3", "Execute SQL query", NodeType.TYPE_EXECUTE_SQL),
            ("node_4", "evaluation and self-reflection", NodeType.TYPE_REFLECT),
            ("node_5", "Return the results to the user", NodeType.TYPE_OUTPUT),
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
        assert len(workflow.nodes) == 6

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

    def test_workflow_save_load(self, tmp_path, real_agent_config):
        """Test saving and loading a workflow with multiple nodes."""
        # Create a workflow with multiple nodes
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test workflow save/load"), agent_config=real_agent_config
        )

        node1 = Node.new_instance(
            node_id="node1",
            description="First node",
            node_type=NodeType.TYPE_GENERATE_SQL,
            input_data=None,
        )

        node2 = Node.new_instance(
            node_id="node2",
            description="Second node",
            node_type=NodeType.TYPE_EXECUTE_SQL,
            input_data=None,
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
        workflow.save(str(save_path))

        # Load workflow from file
        loaded_workflow = Workflow.load(str(save_path), agent_config=real_agent_config)

        # Verify all properties are correctly restored
        assert loaded_workflow.name == "test_workflow"
        assert loaded_workflow.task.task == "Test workflow save/load"
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
        assert loaded_node1.type == NodeType.TYPE_GENERATE_SQL

        loaded_node2 = loaded_workflow.nodes["node2"]
        assert loaded_node2.description == "Second node"
        assert loaded_node2.type == NodeType.TYPE_EXECUTE_SQL
