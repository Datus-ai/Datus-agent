import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def mock_args():
    """Create a mock arguments object for testing."""
    args = argparse.Namespace(
        model="deepseek-v3",
        temperature=0.5,
        top_p=0.9,
        max_tokens=2500,
        task="Select all employees who earn more than $50,000",
        task_type="local",
        db_path="test_db.sqlite",
        schema_path="test_schema.sql",
        plan=True,
        max_steps=20,
        human_in_loop=False,
        output_dir="test_output",
    )
    return args


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.generate.return_value = "Generated text response"
    model.generate_with_json_output.return_value = {"result": "success"}
    model.generate_sql.return_value = "SELECT * FROM employees WHERE salary > 50000;"
    return model


# @pytest.fixture
# def sample_workflow():
#     """Create a sample workflow for testing."""
#     from datus.agent.workflow import Node, Workflow

#     workflow = Workflow("Test Workflow", "A workflow for testing")

#     # Add some tasks to the workflow
#     task1 = Node(
#         "task1",
#         "Parse the query",
#         "query_processing",
#         "Select all employees who earn more than $50,000",
#     )
#     task2 = Node("task2", "Generate SQL", "sql_generation", "Parsed query data")
#     task3 = Node(
#         "task3",
#         "Execute SQL",
#         "sql_execution",
#         "SELECT * FROM employees WHERE salary > 50000;",
#     )

#     workflow.add_task(task1)
#     workflow.add_task(task2)
#     workflow.add_task(task3)

#     return workflow


@pytest.fixture
def sample_database_schema():
    """Create a sample database schema for testing."""
    return """
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL
    );

    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        budget REAL NOT NULL
    );
    """


@pytest.fixture
def sample_database_data():
    """Create sample database data for testing."""
    return [
        {
            "id": 1,
            "name": "John Doe",
            "department": "Engineering",
            "salary": 75000,
            "hire_date": "2020-01-15",
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "department": "Marketing",
            "salary": 65000,
            "hire_date": "2019-05-20",
        },
        {
            "id": 3,
            "name": "Bob Johnson",
            "department": "Engineering",
            "salary": 85000,
            "hire_date": "2018-11-10",
        },
        {
            "id": 4,
            "name": "Alice Brown",
            "department": "HR",
            "salary": 45000,
            "hire_date": "2021-03-01",
        },
        {
            "id": 5,
            "name": "Charlie Wilson",
            "department": "Marketing",
            "salary": 55000,
            "hire_date": "2020-07-30",
        },
    ]


def load_acceptance_config(namespace: str = "snowflake", home: str = "") -> AgentConfig:
    return load_agent_config(
        config="tests/conf/agent.yml", namespace=namespace, home=home, reload=True, force=True, yes=True
    )
