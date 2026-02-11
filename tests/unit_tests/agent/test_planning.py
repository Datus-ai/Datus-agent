from unittest.mock import MagicMock, patch

import pytest

from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.node_models import SqlTask
from datus.utils.constants import DBType


class TestPlanning:
    """Test suite for the planning module."""

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent config."""
        config = MagicMock(spec=AgentConfig)
        config.namespaces = {"test_db": {"type": "duckdb", "database": ":memory:"}}
        config.custom_workflows = {}
        config.workflow_plan = "reflection"
        config.schema_linking_rate = "fast"
        return config

    def test_generate_workflow(self, mock_agent_config):
        """Test generating a workflow from a task description."""
        # Create a SQL task
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="Find employees with salary over $50,000",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        # Use a lambda that sets tools to []
        def mock_init_tools(self):
            self.tools = []

        with patch.object(Workflow, "_init_tools", mock_init_tools):
            # Call generate_workflow with the default plan_type
            workflow = generate_workflow(
                task=sql_task,
                plan_type="reflection",
                agent_config=mock_agent_config,
            )

            # Check that the workflow was created
            assert workflow is not None
            assert workflow.name == "SQL Query Workflow (reflection)"
            assert workflow.task == sql_task

    def test_generate_workflow_with_plan(self, mock_agent_config):
        """Test generating a workflow with planning enabled."""
        # Create a SQL task
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="Find employees with salary over $50,000",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        def mock_init_tools(self):
            self.tools = []

        with patch.object(Workflow, "_init_tools", mock_init_tools):
            # Call generate_workflow with a valid builtin plan type
            workflow = generate_workflow(
                task=sql_task,
                plan_type="reflection",
                agent_config=mock_agent_config,
            )

            # Check that the workflow was created with the correct structure
            assert workflow is not None
            assert workflow.name == "SQL Query Workflow (reflection)"
            assert len(workflow.nodes) > 0
            # Reflection workflow should have multiple nodes
            assert len(workflow.nodes) >= 3

    def test_generate_workflow_without_plan(self, mock_agent_config):
        """Test generating a workflow without planning (using fixed strategy)."""
        # Create a SQL task
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="Find employees with salary over $50,000",
            catalog_name="",
            database_name="test_db",
            schema_name="",
        )

        # Mock _init_tools to avoid database initialization
        def mock_init_tools(self):
            self.tools = []

        with patch.object(Workflow, "_init_tools", mock_init_tools):
            # Call generate_workflow with the 'fixed' plan type (simpler workflow)
            workflow = generate_workflow(
                task=sql_task,
                plan_type="fixed",
                agent_config=mock_agent_config,
            )

            # Check that the workflow was created with the correct structure
            assert workflow is not None
            assert workflow.name == "SQL Query Workflow (fixed)"
            assert len(workflow.nodes) >= 2
            # The workflow should contain at least begin and some action nodes
            assert "node_0" in workflow.nodes  # Begin node
