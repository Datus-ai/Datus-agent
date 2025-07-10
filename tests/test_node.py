from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from datus.agent.node import Node
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.configuration.node_type import NodeType
from datus.schemas.base import BaseResult
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput, GenerateMetricsResult
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
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
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.tools.llms_tools.llms import LLMTool
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
def generate_metrics_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "GenerateMetricsInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def generate_semantic_model_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = Path(__file__).parent / "data" / "GenerateSemanticModelInput.yaml"
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
    agent_config = load_acceptance_config(namespace="local_duckdb")  # FIXME Modify it according to your configuration
    return agent_config


@pytest.fixture
def search_metrics_agent_config() -> AgentConfig:
    # conf = Path(__file__).parent.parent / "conf" / "agent.yml"
    agent_config = load_agent_config(namespace="local_duckdb")  # FIXME Modify it according to your configuration
    return agent_config


@pytest.fixture
def db_manager(agent_config: AgentConfig) -> DBManager:
    return db_manager_instance(agent_config.namespaces)


@pytest.fixture
def sql_connector(db_manager: DBManager):
    """Create real Snowflake database connection"""
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Use connection info from environment variables
    db = db_manager.get_conn("snowflake", db_type="snowflake")

    res = db.test_connection()
    logger.debug(f"connection test {res}")

    yield db

    # Close connection after test
    db.close()


@pytest.fixture
def llm_tools():
    llm_tool = LLMTool(load_agent_config()["deep_seek_ark"])

    logger.info("Testing LLM tool connectivity")
    try:
        response = llm_tool.test("Hello, can you hear me?")
        logger.info(f"LLM tool test successful {response}")
        return llm_tool
    except Exception as e:
        logger.error(f"LLM tool test failed: {str(e)}")
        raise pytest.skip(f"LLM tool not available: {str(e)}")


def save_to_yaml(content: BaseModel, filename: str):
    """Save a Pydantic model instance to a YAML file"""
    with open(filename, "w") as f:
        yaml.dump(content.to_dict(), f)


class TestNode:
    """Test suite for Node class"""

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
        agent_config.current_namespace = "local_sqlite2"
        node = Node.new_instance(
            node_id="schema_link",
            description="Schema Linking",
            node_type=NodeType.TYPE_SCHEMA_LINKING,
            input_data=SchemaLinkingInput(
                input_text="",
                database_type="sqlite",
                catalog_name="",
                database_name="",
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
    def test_generation_node(self, generate_sql_input, agent_config):
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
            )

            # Verify initial node configuration
            assert node.type == NodeType.TYPE_GENERATE_SQL
            assert isinstance(node.input, GenerateSQLInput)
            assert node.input.sql_task.task == generate_sql_input[0]["input"]["sql_task"]["task"]
            assert node.input.database_type == "sqlite"
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

    @pytest.mark.acceptance
    def test_reasoning_node(self, reasoning_input, agent_config):
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
            logger.debug(f"Reasoning node result: {result.to_str()}")
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

    @pytest.mark.acceptance
    def test_reflection_node(self, reflection_input, agent_config):
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
    def test_execution_node(self, execute_sql_input, sql_connector, agent_config):
        """Test SQL execution node with Snowflake database"""
        try:
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
                )

                # Verify initial node configuration
                assert node.type == NodeType.TYPE_EXECUTE_SQL
                assert isinstance(node.input, ExecuteSQLInput)
                assert node.input.sql_query == exec_input["sql_query"]

                # Test validation error for invalid input
                with pytest.raises(ValidationError):
                    ExecuteSQLInput(**{"invalid": "data"})

                print(f"execute sql {exec_input['sql_query']}", sql_connector)
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

    def test_generate_semantic_model_node(self, generate_semantic_model_input, agent_config):
        """Test generate semantic model node"""
        try:
            # Create generate semantic model input from test data
            for case in generate_semantic_model_input:
                input_data = GenerateSemanticModelInput(**case["input"])
                node = Node.new_instance(
                    node_id="generate_semantic_model_test",
                    description="Generate semantic model test",
                    node_type=NodeType.TYPE_GENERATE_SEMANTIC_MODEL,
                    input_data=input_data,
                    agent_config=agent_config,
                )
                result = node.run()
                logger.debug(f"Generate semantic model node result: {result}")
                assert node.status == "completed", f"Node execution failed with status: {node.status}"
                assert isinstance(result, GenerateSemanticModelResult), "Result type mismatch"
                assert result.success is True, f"Node execution failed: {result}"
        except Exception as e:
            logger.error(f"Generate semantic model node test failed: {str(e)}")
            raise

    def test_generate_metrics_node(self, generate_metrics_input, agent_config):
        """Test generate metrics node"""
        try:
            # Create generate metrics input from test data
            for case in generate_metrics_input:
                input_data = GenerateMetricsInput(**case["input"])
                node = Node.new_instance(
                    node_id="generate_metrics_test",
                    description="Generate Metrics Test",
                    node_type=NodeType.TYPE_GENERATE_METRICS,
                    input_data=input_data,
                    agent_config=agent_config,
                )
                result = node.run()
                logger.debug(f"Generate metrics node result: {result}")
                assert node.status == "completed", f"Node execution failed with status: {node.status}"
                assert isinstance(result, GenerateMetricsResult), "Result type mismatch"
                assert result.success is True, f"Node execution failed: {result}"
                assert len(result.metrics) > 0, "Metrics is empty"
                assert len(result.sql_queries) > 0, "SQL queries is empty"
        except Exception as e:
            logger.error(f"Generate metrics node test failed: {str(e)}")
            raise

    def test_search_metrics_node(self, search_metrics_input, search_metrics_agent_config: AgentConfig):
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
                    agent_config=search_metrics_agent_config,
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
