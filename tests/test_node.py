import glob
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pytest
import yaml
from agents import Tool
from pydantic import BaseModel, ValidationError

from datus.agent.node import Node
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.base import BaseResult
from datus.schemas.compare_node_models import CompareInput, CompareResult
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.schemas.node_models import (
    ExecuteSQLInput,
    ExecuteSQLResult,
    GenerateSQLInput,
    GenerateSQLResult,
    ReflectionInput,
    SQLContext,
    SqlTask,
)
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.schemas.search_metrics_node_models import SearchMetricsInput, SearchMetricsResult
from datus.tools.func_tool import db_function_tools
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)


@pytest.fixture
def generate_sql_input():
    """load idata from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "GenerateSQLInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def schema_linking_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "SchemaLinkingInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def execute_sql_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "ExecuteSQLInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def reflection_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "ReflectionInput.yaml"
    with open(yaml_path, "r") as f:
        r_input = yaml.safe_load(f)
        return r_input


@pytest.fixture
def output_input() -> Dict[str, Any]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "OutputInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def reasoning_input() -> Dict[str, Any]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "ReasoningInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def doc_search_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "DocSearchInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def search_metrics_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "SearchMetricsInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# @pytest.fixture
# def mock_generate_sql_input() -> Dict[str, Any]:
#    return {
#        "database_type": "sqlite",
#        "table_schemas": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
#        "input_text": "Query all data from users table",
#        "context": []
#    }


@pytest.fixture
def agent_config() -> AgentConfig:
    agent_config = load_acceptance_config(namespace="bird_sqlite")  # FIXME Modify it according to your configuration
    return agent_config


@pytest.fixture
def function_tools(agent_config: AgentConfig) -> List[Tool]:
    return db_function_tools(agent_config)


def save_to_yaml(content: BaseModel, filename: str):
    """Save a Pydantic model instance to a YAML file"""
    with open(filename, "w") as f:
        yaml.dump(content.to_dict(), f, allow_unicode=True)


def init_metricflow_db() -> None:
    db_dir = Path(__file__).parent / "data" / "datus_metricflow_db"
    db_path = db_dir / "duck.db"
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
    csv_path: Path = Path(__file__).parent / "data/metricflow_csv" / "*.csv"
    print(f"Creating db_path: {db_path}")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE SCHEMA IF NOT EXISTS mf_demo;")
    csv_files = glob.glob(str(csv_path))
    for csv_file in csv_files:
        full_file_name = Path(csv_file).name
        file_name = full_file_name.split(".")[0]
        table_name = f"mf_demo.{file_name}"
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}', header=TRUE)")
        print(f"finish the import of {csv_file}")
    conn.close()


class TestNode:
    """Test suite for Node class"""

    def setup_method(self) -> None:
        init_metricflow_db()

    def test_node_initialization(self, agent_config):
        """Test node initialization"""
        node = Node.new_instance(
            node_id="test_node",
            description="Test Node",
            node_type=NodeType.TYPE_SCHEMA_LINKING,
            agent_config=agent_config,
        )

        assert node.id == "test_node"
        assert node.description == "Test Node"
        assert node.type == NodeType.TYPE_SCHEMA_LINKING
        assert node.status == "pending"
        assert node.result is None
        assert node.dependencies == []

    def test_node_state_transitions(self, agent_config):
        """Test node state transitions"""
        node = Node.new_instance("test_node", "Test Node", NodeType.TYPE_SCHEMA_LINKING, agent_config=agent_config)

        # Test start state
        node.start()
        assert node.status == "running"
        assert node.start_time is not None

        # Test complete state
        result = BaseResult(success=True, error=None)
        node.complete(result)
        assert node.status == "completed"
        assert node.result == result
        assert node.end_time is not None

        # Test fail state
        node = Node.new_instance("test_node", "Test Node", NodeType.TYPE_SCHEMA_LINKING, agent_config=agent_config)
        error_msg = "Test error"
        node.fail(error_msg)
        assert node.status == "failed"
        assert node.result == BaseResult(success=False, error=error_msg)

    def test_schema_linking_node(self, schema_linking_input, agent_config: AgentConfig):
        """Test schema linking node"""
        # Take first test case from the list
        for inputs in schema_linking_input:
            test_case = inputs["input"]
            if "namespace" in test_case:
                agent_config.current_namespace = test_case["namespace"]
                del test_case["namespace"]
            node = Node.new_instance(
                node_id="schema_link",
                description="Schema Linking",
                node_type=NodeType.TYPE_SCHEMA_LINKING,
                input_data=SchemaLinkingInput(**test_case),
                agent_config=agent_config,
            )
            assert node.type == NodeType.TYPE_SCHEMA_LINKING
            assert isinstance(node.input, SchemaLinkingInput)
            assert node.input.input_text == test_case["input_text"]
            result = node.run()
            assert isinstance(result, SchemaLinkingResult)
            print(f"result is {type(result)}, {result.success}, {result.schema_count}")
            assert isinstance(result, SchemaLinkingResult)
            assert result.success
            assert result.schema_count > 0

    def test_schema_linking_fallback(self, agent_config: AgentConfig):
        """Test schema linking node with fallback"""
        agent_config.current_namespace = "bird_sqlite"
        agent_config.rag_base_path = "/tmp/test_data"
        node = Node.new_instance(
            node_id="schema_link",
            description="Schema Linking",
            node_type=NodeType.TYPE_SCHEMA_LINKING,
            input_data=SchemaLinkingInput(
                input_text="",
                database_type=DBType.SQLITE,
                catalog_name="",
                database_name="california_schools",
                schema_name="",
            ),
            agent_config=agent_config,
        )
        res = node.run()
        assert isinstance(res, SchemaLinkingResult)
        assert res.success is True
        assert res.schema_count > 0
        assert res.value_count == 0

    @pytest.mark.acceptance
    def test_generation_node(self, generate_sql_input, agent_config, function_tools: List[Tool]):
        """Test SQL generation node with real DeepSeek model and Snowflake database"""
        try:
            # Create table schema from input data
            input_data = GenerateSQLInput(**generate_sql_input[0]["input"])

            # Create node instance for testing
            node = Node.new_instance(
                node_id="gen_sql_test",
                description="Generate SQL Test",
                node_type=NodeType.TYPE_GENERATE_SQL,
                input_data=input_data,
                agent_config=agent_config,
                tools=function_tools,
            )

            # Verify initial node configuration
            assert node.type == NodeType.TYPE_GENERATE_SQL
            assert isinstance(node.input, GenerateSQLInput)
            assert node.input.sql_task.task == generate_sql_input[0]["input"]["sql_task"]["task"]
            assert node.input.database_type == DBType.SQLITE
            assert len(node.input.table_schemas) == 3
            assert node.input.table_schemas[0].table_name == "schools"

            #            # Test error handling for missing dependencies
            #            with pytest.raises(Exception) as exc_info:
            #                node.run(db_conn=None, model=None)
            #            assert "SQL generation model not provided" in str(exc_info.value)

            # Test validation error for invalid input
            with pytest.raises(ValidationError):
                GenerateSQLInput(**{"invalid": "data"})

            # Execute node with valid dependencies
            result = node.run()
            logger.debug(f"Generation node result: {result.to_str()}")

            # Verify execution results
            assert node.status == "completed", f"Node execution failed with status: {node.status}"
            assert result.success is True, f"Node execution failed: {result}"
            assert isinstance(result, GenerateSQLResult), "Result type mismatch"
            assert len(result.sql_query) > 0, "Empty SQL query generated"
            assert isinstance(result.tables, list), "Tables result is not a list"
            assert len(result.tables) > 0, "No tables in result"

            # Test error state handling
            node.fail("Test error")
            assert node.status == "failed"
            assert node.result["error"] == "Test error"

        except Exception as e:
            logger.error(f"Generation node test failed: {str(e)}")
            raise

    def test_node_dependencies(self, agent_config):
        """Test node dependencies"""
        node = Node.new_instance("test_node", "Test Node", NodeType.TYPE_SCHEMA_LINKING, agent_config=agent_config)

        # Add dependencies
        node.add_dependency("dep_1")
        node.add_dependency("dep_2")

        assert "dep_1" in node.dependencies
        assert "dep_2" in node.dependencies
        assert len(node.dependencies) == 2

        # Test duplicate dependency
        node.add_dependency("dep_1")
        assert len(node.dependencies) == 2

    def test_invalid_node_type(self, agent_config):
        """Test invalid node type"""
        with pytest.raises(ValueError) as exc_info:
            Node.new_instance("test_node", "Test Node", "invalid_type", agent_config=agent_config)
        assert "Invalid node type" in str(exc_info.value)

    def test_reflection_init(self):
        """Test reflection input initialization with list of SQL contexts"""
        # Create test data
        task = SqlTask(task="test task", database_type="snowlfake", database_name="test_db")

        sql_contexts = [
            SQLContext(
                sql_query="SELECT * FROM test",
                explanation="test explanation",
                sql_return="test result",
                row_count=1,
                reflection_strategy="SUCCESS",
                reflection_explanation="test explanation",
            ),
            SQLContext(
                sql_query="SELECT count(*) FROM test",
                explanation="count explanation",
                sql_return="10",
                row_count=1,
                reflection_strategy="SCHEMA_LINKING",
                reflection_explanation="need schema linking",
            ),
        ]

        # Create reflection input
        reflect_input = ReflectionInput(task_description=task, sql_context=sql_contexts)

        # Verify initialization
        assert reflect_input.task_description.task == "test task"
        assert len(reflect_input.sql_context) == 2
        assert reflect_input.sql_context[0].sql_query == "SELECT * FROM test"
        assert reflect_input.sql_context[1]
        # save_to_yaml(reflection_input, "ReflectionInput.yaml")

    # @pytest.mark.acceptance
    def test_reasoning_node_snowflake(self, reasoning_input, agent_config):
        """Test reasoning node with real DeepSeek model and Snowflake database"""
        try:
            agent_config.current_namespace = "snowflake"
            # Initialize reasoning input
            # reasoning_input = generate_sql_input  # the same for simple test
            input_data = ReasoningInput(**reasoning_input[0]["input"])

            # Create node instance for testing
            node = Node.new_instance(
                node_id="reasoning_test",
                description="Reasoning SQL Test",
                node_type=NodeType.TYPE_REASONING,
                input_data=input_data,
                agent_config=agent_config,
                tools=db_function_tools(agent_config),
            )

            # Verify initial node configuration
            assert node.type == NodeType.TYPE_REASONING
            assert isinstance(node.input, ReasoningInput)
            assert node.input.sql_task.task == reasoning_input[0]["input"]["sql_task"]["task"]
            assert node.input.database_type == reasoning_input[0]["input"]["database_type"]

            # Test validation error for invalid input
            with pytest.raises(ValidationError):
                ReasoningInput(**{"invalid": "data"})

            # Execute node with valid dependencies
            result = node.run()
            logger.info(f"Reasoning node result: {result.to_str()}")
            # Verify execution results
            assert isinstance(result, ReasoningResult), "Result type mismatch"

            assert node.status == "completed", f"Node execution failed with status: {node.status}"
            assert result.success is True, f"Node execution failed: {result}"
            assert isinstance(result.sql_contexts, list), "sql_contexts is not a list"
            # assert len(result.sql_contexts) > 0, "No SQL contexts returned"

            # Test error state handling
            node.fail("Test error")
            assert node.status == "failed"
            assert node.result["error"] == "Test error"

        except Exception as e:
            logger.error(f"Reasoning node test failed: {str(e)}")
            raise

    def test_reasoning_node(self, agent_config, function_tools: List[Tool]):
        """Test reasoning node with SSB SQLite database using revenue calculation task"""
        try:
            agent_config.current_namespace = "ssb_sqlite"

            # Create simple ReasoningInput with revenue calculation task
            input_data = ReasoningInput(
                contexts=[],
                data_details=[],
                table_schemas=[],
                metrics=[],
                sql_task=SqlTask(
                    id="revenue_test",
                    task=(
                        "Total revenue for January 1994 where discount was between 4 and 6 and "
                        "quantity sold was between 26 and 35"
                    ),
                    database_type="sqlite",
                    database_name="SSB",
                    output_dir="output/test",
                ),
                database_type="sqlite",
                external_knowledge="",
                prompt_version="",
            )

            # Create node instance for testing
            node = Node.new_instance(
                node_id="reasoning_test",
                description="Reasoning SQL Test",
                node_type=NodeType.TYPE_REASONING,
                input_data=input_data,
                agent_config=agent_config,
                tools=function_tools,
            )

            # Verify initial node configuration
            assert node.type == NodeType.TYPE_REASONING
            assert isinstance(node.input, ReasoningInput)

            # Execute node
            result = node.run()
            logger.debug(f"Reasoning node result: {result.to_str()}")

            # Simple assertions - just check it works and produces SQL
            assert isinstance(result, ReasoningResult), "Result type mismatch"
            assert result.success is True, f"Node execution failed: {result}"
            assert node.status == "completed", f"Node execution failed with status: {node.status}"

        except Exception as e:
            logger.error(f"Simple reasoning node test failed: {str(e)}")
            raise

    @pytest.mark.acceptance
    def test_reflection_node(self, reflection_input, agent_config, function_tools: List[Tool]):
        """Test reflection node with test case[0] from YAML"""
        try:
            # Create reflection input data
            index = 0
            input_data = reflection_input[index]["input"]
            # expected_result = reflection_input[index]["result"]

            logger.debug(f"raw input: {input_data}")

            # Parse input components
            task_description = SqlTask.from_dict(input_data["task_description"])
            contexts = [SQLContext(**context_data) for context_data in input_data["sql_context"]]

            # Create reflection input
            reflection_input = ReflectionInput(task_description=task_description, sql_context=contexts)

            # Create reflection node
            node = Node.new_instance(
                node_id=f"reflection_test_{index}",
                description="Reflection Analysis",
                node_type=NodeType.TYPE_REFLECT,
                input_data=reflection_input,
                agent_config=agent_config,
                tools=function_tools,
            )

            # Validate node type and input
            assert node.type == NodeType.TYPE_REFLECT
            assert isinstance(node.input, ReflectionInput)

            # Run reflection node
            result = node.run()
            logger.debug(f"Reflection node result: {result}")

        except Exception as e:
            logger.error(f"Reflection node test failed: {str(e)}")
            raise

    @pytest.mark.acceptance
    def test_execution_node(self, execute_sql_input, agent_config, function_tools: List[Tool]):
        """Test SQL execution node with Snowflake database"""
        try:
            agent_config.current_namespace = "bird_sqlite"
            # Create execution input from test data
            test_cases = [0, 1]
            for test_case_num in test_cases:
                # Create execution input from test data
                exec_input = execute_sql_input[test_case_num]["input"]
                input_data = ExecuteSQLInput(**exec_input)

                # Create node instance for testing
                node = Node.new_instance(
                    node_id="execute_sql_test",
                    description="Execute SQL Test",
                    node_type=NodeType.TYPE_EXECUTE_SQL,
                    input_data=input_data,
                    agent_config=agent_config,
                    tools=function_tools,
                )

                # Verify initial node configuration
                assert node.type == NodeType.TYPE_EXECUTE_SQL
                assert isinstance(node.input, ExecuteSQLInput)
                assert node.input.sql_query == exec_input["sql_query"]

                # Test validation error for invalid input
                with pytest.raises(ValidationError):
                    ExecuteSQLInput(**{"invalid": "data"})

                # Execute node with valid database connection
                result = node.run()
                logger.debug(f"Execution node result: {result}")

                # Verify execution results
                assert node.status == "completed", f"Node execution failed with status: {node.status}"
                assert isinstance(result, ExecuteSQLResult), "Result type mismatch"
                assert result.success is True, f"Node execution failed: {result}"
                assert result.sql_return is not None, "Execution result is empty"
                assert result.row_count is not None, "Execution explanation is empty"

        except Exception as e:
            logger.error(f"Execution node test failed: {str(e)}")
            raise

    def test_doc_search_node(self, doc_search_input, agent_config):
        """Test document node"""
        try:
            # Create doc search input from test data
            for case in doc_search_input:
                input_data = DocSearchInput(**case["doc_search"])
                node = Node.new_instance(
                    node_id="doc_search_test",
                    description="Doc Search Test",
                    node_type=NodeType.TYPE_DOC_SEARCH,
                    input_data=input_data,
                    agent_config=agent_config,
                )
                result = node.run()
                logger.debug(f"Doc search node result: {result}")
                assert node.status == "completed", f"Node execution failed with status: {node.status}"
                assert isinstance(result, DocSearchResult), "Result type mismatch"
                assert result.success is True, f"Node execution failed: {result}"
        except Exception as e:
            logger.error(f"Doc search node test failed: {str(e)}")
            raise

    def test_search_metrics_node(self, search_metrics_input, agent_config: AgentConfig):
        """Test schema linking node"""
        # Take first test case from the list
        try:
            for case in search_metrics_input:
                input_data = SearchMetricsInput(**case["input"])
                node = Node.new_instance(
                    node_id="search_metrics",
                    description="Search Metrics",
                    node_type=NodeType.TYPE_SEARCH_METRICS,
                    input_data=input_data,
                    agent_config=agent_config,
                )
                assert node.type == NodeType.TYPE_SEARCH_METRICS
                assert isinstance(node.input, SearchMetricsInput)
                result = node.run()
                logger.debug(f"Search metrics node result: {result}")
                assert node.status == "completed", f"Node execution failed with status: {node.status}"
                assert isinstance(result, SearchMetricsResult), "Result type mismatch"
                assert result.success is True, f"Node execution failed: {result}"
        except Exception as e:
            logger.error(f"Search metrics node test failed: {str(e)}")
            raise

    # @pytest.mark.acceptance
    def test_compare_node(self, agent_config: AgentConfig, function_tools: List[Tool]):
        """Test compare node with real california_schools data"""
        try:
            # Create test SQL task
            sql_task = SqlTask(
                task=(
                    "Please list the phone numbers of the direct charter-funded schools "
                    "that are opened after 2000/1/1."
                ),
                database_type="sqlite",
                database_name="california_schools",
            )

            # Create test SQL context
            sql_context = SQLContext(
                sql_query=(
                    "SELECT Phone FROM schools WHERE Charter = 1 AND FundingType = 'Directly funded' "
                    "AND OpenDate > '2000-01-01' AND Phone IS NOT NULL ORDER BY OpenDate"
                ),
                explanation="Query to get phone numbers of direct charter-funded schools opened after 2000/1/1",
                sql_return="Phone numbers result",
                row_count=5,
            )

            # Create compare input with expected SQL
            input_data = CompareInput(
                sql_task=sql_task,
                sql_context=sql_context,
                expectation=(
                    "SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode "
                    "WHERE T1.Charter Funding Type = 'Directly funded' AND T1.Charter School (Y/N) = 1 "
                    "AND T2.OpenDate > '2000-01-01'"
                ),
            )

            # Create compare node
            node = Node.new_instance(
                node_id="compare_test",
                description="Compare SQL Test",
                node_type=NodeType.TYPE_COMPARE,
                input_data=input_data,
                agent_config=agent_config,
                tools=db_function_tools,
            )

            # Verify initial node configuration
            assert node.type == NodeType.TYPE_COMPARE
            assert isinstance(node.input, CompareInput)
            assert (
                node.input.sql_task.task
                == "Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1."
            )
            assert "SELECT Phone FROM schools WHERE Charter = 1" in node.input.sql_context.sql_query
            assert "SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2" in node.input.expectation

            # Test validation error for invalid input
            with pytest.raises(ValidationError):
                CompareInput(**{"invalid": "data"})

            # Execute node
            result = node.run()
            logger.info(f"Compare node result: {result}")

            # Verify execution results
            assert node.status == "completed", f"Node execution failed with status: {node.status}"
            assert isinstance(result, CompareResult), "Result type mismatch"
            assert result.success is True, f"Node execution failed: {result}"
            assert len(result.explanation) > 0, "Empty explanation"
            assert len(result.suggest) > 0, "Empty suggestions"

            # Test that explanation contains meaningful content
            assert (
                "Charter" in result.explanation or "charter" in result.explanation
            ), "Explanation should mention charter schools"

            # Test that suggestions contain actionable advice
            assert (
                "JOIN" in result.suggest or "join" in result.suggest or "table" in result.suggest
            ), "Suggestions should mention JOIN or table differences"

            # Print results for manual inspection
            print("\n=== Compare Node Test Results ===")
            print(f"Explanation: {result.explanation}")
            print(f"Suggestions: {result.suggest}")
            print(f"Success: {result.success}")
            print("=====================================\n")

        except Exception as e:
            logger.error(f"Compare node test failed: {str(e)}")
            raise

    @pytest.mark.acceptance
    def test_compare_with_mcp_node(self, agent_config, function_tools: List[Tool]):
        """Test compare node with MCP streaming for enhanced database analysis"""
        try:
            # Create test SQL task
            sql_task = SqlTask(
                task=(
                    "Please list the phone numbers of the direct charter-funded schools"
                    " that are opened after 2000/1/1."
                ),
                database_type="sqlite",
                database_name="california_schools",
            )

            # Create test SQL context with current query
            sql_context = SQLContext(
                sql_query=(
                    "SELECT Phone FROM schools WHERE Charter = 1 AND FundingType = 'Directly funded' "
                    "AND OpenDate > '2000-01-01' AND Phone IS NOT NULL ORDER BY OpenDate"
                ),
                explanation="Query to get phone numbers of direct charter-funded schools opened after 2000/1/1",
                sql_return="Phone numbers result",
                row_count=5,
            )

            # Create compare input with expected SQL that uses proper joins
            input_data = CompareInput(
                sql_task=sql_task,
                sql_context=sql_context,
                expectation=(
                    "SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode"
                    " WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`Charter School (Y/N)` = 1"
                    " AND T2.OpenDate > '2000-01-01'"
                ),
            )

            # Create compare node
            node = Node.new_instance(
                node_id="compare_mcp_test",
                description="Compare SQL MCP Test",
                node_type=NodeType.TYPE_COMPARE,
                input_data=input_data,
                agent_config=agent_config,
                tools=function_tools,
            )

            # Verify initial node configuration
            assert node.type == NodeType.TYPE_COMPARE
            assert isinstance(node.input, CompareInput)
            assert (
                node.input.sql_task.task
                == "Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1."
            )
            assert "SELECT Phone FROM schools WHERE Charter = 1" in node.input.sql_context.sql_query
            assert "SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2" in node.input.expectation

            # Test MCP streaming method exists
            assert hasattr(node, "_compare_sql_stream"), "Node should have MCP streaming method"

            # Execute node (standard execution first)
            result = node.run()
            logger.debug(f"Compare MCP node result: {result}")

            # Verify execution results
            assert node.status == "completed", f"Node execution failed with status: {node.status}"
            assert isinstance(result, CompareResult), "Result type mismatch"
            assert result.success is True, f"Node execution failed: {result}"
            assert len(result.explanation) > 0, "Empty explanation"
            # assert len(result.suggest) > 0, "Empty suggestions"

            # Should identify key differences between single table vs JOIN approach
            explanation_lower = result.explanation.lower()
            assert (
                "join" in explanation_lower or "table" in explanation_lower or "frpm" in explanation_lower
            ), "Should identify table structure differences"

            # Suggestions should be actionable and database-informed
            if result.suggest:
                suggest_lower = result.suggest.lower()
                assert (
                    "join" in suggest_lower or "table" in suggest_lower or "modify" in suggest_lower
                ), "Should provide actionable database-informed suggestions"

            # Print results for manual inspection
            print("\n=== Compare MCP Node Test Results ===")
            print(f"Explanation: {result.explanation}")
            print(f"Suggestions: {result.suggest}")
            print(f"Success: {result.success}")
            print("==========================================\n")

        except Exception as e:
            logger.error(f"Compare MCP node test failed: {str(e)}")
            raise
