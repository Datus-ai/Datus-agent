from unittest.mock import MagicMock, patch

from datus.agent.node import Node
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow


class TestPlanning:
    """Test suite for the planning module."""

    def test_generate_workflow(self):
        """Test generating a workflow from a task description."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "workflow": {
                "name": "SQL Query Workflow",
                "description": "Convert natural language to SQL and execute",
                "tasks": [
                    {
                        "id": "task1",
                        "description": "Parse the query",
                        "type": "query_processing",
                        "dependencies": [],
                    },
                    {
                        "id": "task2",
                        "description": "Generate SQL",
                        "type": "sql_generation",
                        "dependencies": ["task1"],
                    },
                    {
                        "id": "task3",
                        "description": "Execute SQL",
                        "type": "sql_execution",
                        "dependencies": ["task2"],
                    },
                ],
            }
        }

        # Call generate_workflow
        with patch(
            "agent.planning.plan.generate_workflow",
            return_value=Workflow("SQL Query Workflow", "Convert natural language to SQL and execute"),
        ):
            workflow = generate_workflow(
                model=mock_model,
                task="Find employees with salary over $50,000",
                use_plan=True,
            )

            # Check that the workflow was created
            assert workflow is not None
            assert workflow.name == "SQL Query Workflow"
            assert workflow.description == "Convert natural language to SQL and execute"

    def test_generate_workflow_with_plan(self):
        """Test generating a workflow with planning enabled."""
        # Create a mock model
        mock_model = MagicMock()

        # Mock the generate_with_json_output method to return a workflow plan
        mock_model.generate_with_json_output.return_value = {
            "workflow": {
                "name": "SQL Query Workflow",
                "description": "Convert natural language to SQL and execute",
                "tasks": [
                    {
                        "id": "task1",
                        "description": "Analyze query intent",
                        "type": "query_processing",
                        "dependencies": [],
                    },
                    {
                        "id": "task2",
                        "description": "Identify database entities",
                        "type": "query_processing",
                        "dependencies": ["task1"],
                    },
                    {
                        "id": "task3",
                        "description": "Generate SQL query",
                        "type": "sql_generation",
                        "dependencies": ["task2"],
                    },
                    {
                        "id": "task4",
                        "description": "Execute SQL query",
                        "type": "sql_execution",
                        "dependencies": ["task3"],
                    },
                    {
                        "id": "task5",
                        "description": "Format results",
                        "type": "data_validation",
                        "dependencies": ["task4"],
                    },
                ],
            }
        }

        # Create a function to simulate the behavior of generate_workflow
        def mock_generate_workflow(model, task, use_plan):
            # Create a new workflow
            workflow = Workflow("SQL Query Workflow", "Convert natural language to SQL and execute")

            # Add tasks to the workflow based on the mock response
            task1 = Node.new_instance("task1", "Analyze query intent", "query_processing")
            task2 = Node.new_instance("task2", "Identify database entities", "query_processing")
            task3 = Node.new_instance("task3", "Generate SQL query", "sql_generation")
            task4 = Node.new_instance("task4", "Execute SQL query", "sql_execution")
            task5 = Node.new_instance("task5", "Format results", "data_validation")

            # Add dependencies
            task2.add_dependency("task1")
            task3.add_dependency("task2")
            task4.add_dependency("task3")
            task5.add_dependency("task4")

            # Add tasks to workflow
            workflow.add_task(task1)
            workflow.add_task(task2)
            workflow.add_task(task3)
            workflow.add_task(task4)
            workflow.add_task(task5)

            return workflow

        # Patch the generate_workflow function
        with patch("datus.agent.plan.generate_workflow", side_effect=mock_generate_workflow):
            # Call generate_workflow
            workflow = generate_workflow(
                model=mock_model,
                task="Find employees with salary over $50,000",
                use_plan=True,
            )

            # Check that the workflow was created with the correct structure
            assert workflow is not None
            assert workflow.name == "SQL Query Workflow"
            assert len(workflow.tasks) == 5
            assert "task1" in workflow.tasks
            assert "task5" in workflow.tasks

            # Check task dependencies
            assert workflow.tasks["task2"].dependencies == ["task1"]
            assert workflow.tasks["task3"].dependencies == ["task2"]
            assert workflow.tasks["task4"].dependencies == ["task3"]
            assert workflow.tasks["task5"].dependencies == ["task4"]

    def test_generate_workflow_without_plan(self):
        """Test generating a workflow without planning enabled."""
        # Create a mock model
        mock_model = MagicMock()

        # Create a function to simulate the behavior of generate_workflow without planning
        def mock_generate_workflow(model, task, use_plan):
            # Create a new workflow with a simpler structure
            workflow = Workflow("Simple SQL Workflow", "Convert natural language to SQL and execute")

            # Add basic tasks to the workflow
            task1 = Node.new_instance("task1", "Generate SQL", "sql_generation", task)
            task2 = Node.new_instance("task2", "Execute SQL", "sql_execution")

            # Add dependency
            task2.add_dependency("task1")

            # Add tasks to workflow
            workflow.add_task(task1)
            workflow.add_task(task2)

            return workflow

        # Patch the generate_workflow function
        with patch("agent.planning.plan.generate_workflow", side_effect=mock_generate_workflow):
            # Call generate_workflow with use_plan=False
            workflow = generate_workflow(
                model=mock_model,
                task="Find employees with salary over $50,000",
                use_plan=False,
            )

            # Check that the workflow was created with the correct structure
            assert workflow is not None
            assert workflow.name == "Simple SQL Workflow"
            assert len(workflow.tasks) == 2
            assert "task1" in workflow.tasks
            assert "task2" in workflow.tasks

            # Check task dependencies
            assert workflow.tasks["task2"].dependencies == ["task1"]

            # Check that the input was passed to the first task
            assert workflow.tasks["task1"].input == "Find employees with salary over $50,000"
