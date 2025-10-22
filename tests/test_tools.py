import os
import sys
from pathlib import Path

import pytest
import yaml

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
from datus.schemas.reason_sql_node_models import ReasoningInput
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata.store import SchemaStorage
from datus.tools.lineage_graph_tools import SchemaLineageTool
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_logger(__name__)


class TestLLMsTools:
    """Test suite for LLMs tools functionality."""

    @pytest.fixture
    def global_config(self) -> AgentConfig:
        return load_agent_config(namespace="bird_sqlite")

    @pytest.fixture
    def llm(self, global_config: AgentConfig):
        return LLMBaseModel.create_model(agent_config=global_config)

    @pytest.fixture
    def test_data(self):
        # load data YAML from files
        yaml_path = Path(__file__).parent / "data" / "GenerateSQLInput.yaml"
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def rag_storage(self) -> SchemaStorage:
        """Create a temporary lineage tool instance"""

        # FIXME Modify it according to your configuration
        test_db_path = Path(__file__).parent.parent / "data/datus_db_bird_sqlite"
        storage = SchemaStorage(str(test_db_path), embedding_model=get_db_embedding_model())
        return storage

    def test_reasoning_sql(self, llm: LLMBaseModel, test_data):
        """Test basic tool execution."""
        # using test data from YAML
        input_data = ReasoningInput(**test_data[0]["input"])
        print(input_data)
        from datus.tools.llms_tools.reasoning_sql import reasoning_sql_with_mcp

        result = reasoning_sql_with_mcp(model=llm, input_data=input_data, tools=[], tool_config={"max_turns": 10})
        assert result is not None, "Tool execution should return a result"


class TestLineageTools:
    """Test suite for lineage graph functionality"""

    @pytest.fixture
    def agent_config(self) -> AgentConfig:
        return load_agent_config(**{"namespace": "bird_sqlite"})

    @pytest.fixture
    def setup_lineage_tool(self, agent_config: AgentConfig):
        """Create a temporary lineage tool instance"""

        test_db_path = agent_config.rag_storage_path()
        logger.debug(f"Test db path: {test_db_path}")
        tool = SchemaLineageTool(agent_config=agent_config)
        yield tool
        # Cleanup test database
        # import shutil
        # if os.path.exists(test_db_path):
        #    shutil.rmtree(test_db_path)

    @pytest.fixture
    def test_data(self):
        """Load test data from YAML file"""
        yaml_path = Path(__file__).parent / "data/SchemaLinkingInput.yaml"
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    # @pytest.mark.acceptance
    def test_search(self, setup_lineage_tool: SchemaLineageTool, test_data):
        """Test store and search functionality
        Need to init spider snowflake dataset first and set the db_path to
        "data/datus_db_{namespace}"
        """
        # Use test data from YAML
        input_data = test_data[0]["input"]  # use first test data

        # Convert input data to SchemaLinkingInput model
        input_model = SchemaLinkingInput(**input_data)

        # Store schema with input from YAML
        result = setup_lineage_tool.execute(input_model)
        assert isinstance(result, SchemaLinkingResult), f"Expected SchemaLinkingResult, got {type(result)}"
        assert result.success is True, f"Schema storage failed: {result}"

        # Search similar schemas using the same input text
        search_params = SchemaLinkingInput(
            input_text=input_data["input_text"],
            matching_rate=input_data["matching_rate"],
            database_type=input_data["database_type"],
            database_name=input_data["database_name"],
        )
        search_result = setup_lineage_tool.execute(search_params)
        # logger.debug(f"Search result: {search_result}")

        # Verify result type and content
        assert isinstance(
            search_result, SchemaLinkingResult
        ), f"Expected SchemaLinkingResult, got {type(search_result)}"
        assert search_result.success is True, f"Schema search failed: {search_result}"
        assert search_result.schema_count > 0, "Invalid schema count"
        assert search_result.value_count > 0, "Invalid value count"
        # assert len(search_result.table_schemas) == input_data["top_n"], \
        #     f"Expected {input_data['top_n']} results, got {len(search_result.table_schemas)}"

    def test_invalid_schema_input(self, setup_lineage_tool):
        """Test invalid input handling"""
        # Test missing required parameter
        with pytest.raises(ValueError):
            setup_lineage_tool.execute(SchemaLinkingInput())

        # Test invalid input_text type
        with pytest.raises(ValueError):
            setup_lineage_tool.execute(
                SchemaLinkingInput(
                    input_text=123,
                    matching_rate="fast",
                    database_type=DBType.SQLITE,
                    database_name="test",
                )
            )

        # Test invalid top_n parameter
        with pytest.raises(ValueError):
            setup_lineage_tool.execute(
                SchemaLinkingInput(
                    input_text="CREATE TABLE test (id INTEGER)",
                    matching_rate="abc",
                    database_type=DBType.SQLITE,
                    database_name="test",
                )
            )

    def test_get_table_and_values2(self, setup_lineage_tool):
        """Test get table and values functionality"""
        # Use test data from YAML
        input_data = {
            "database_type": DBType.SNOWFLAKE,
            "database_name": "GLOBAL_WEATHER__CLIMATE_DATA_FOR_BI",
            "table_names": ["GLOBAL_WEATHER__CLIMATE_DATA_FOR_BI.STANDARD_TILE.HISTORY_DAY"],
        }
        schemas, values = setup_lineage_tool.get_table_and_values(
            input_data["database_name"], input_data["table_names"]
        )

        logger.debug(f"Result schemas: {schemas}")
        assert len(schemas) == 1, "Invalid schema count"
        assert len(values) == 1, "Invalid value count"

    # def test_search_tables_with_llm(self, setup_lineage_tool, test_data, llm):
    #    """Test table search functionality with llm"""
    #    pass
