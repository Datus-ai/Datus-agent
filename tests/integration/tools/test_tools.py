import os
import sys
from pathlib import Path

import pytest
import yaml

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.reason_sql_node_models import ReasoningInput
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata.store import SchemaStorage
from datus.tools.lineage_graph_tools import SchemaLineageTool
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from tests.conftest import TEST_DATA_DIR
from tests.unit_tests.mock_llm_model import build_simple_response

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_logger(__name__)


class TestLLMsTools:
    """Test suite for LLMs tools functionality."""

    @pytest.fixture
    def test_data(self):
        # load data YAML from files
        yaml_path = TEST_DATA_DIR / "GenerateSQLInput.yaml"
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def rag_storage(self) -> SchemaStorage:
        """Create a temporary lineage tool instance"""

        storage = SchemaStorage(embedding_model=get_db_embedding_model())
        return storage

    def test_reasoning_sql(self, mock_llm_create, test_data):
        """Test basic tool execution."""
        # Configure MockLLMModel response - reasoning_sql_with_mcp expects JSON with 'sql' field
        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    content=(
                        '{"sql": "SELECT Zip FROM frpm WHERE frpm.`Charter School (Y/N)` = 1", '
                        '"tables": ["frpm"], "explanation": "Query charter schools zip codes"}'
                    )
                ),
            ]
        )

        # Using test data from YAML
        input_data = ReasoningInput(**test_data[0]["input"])
        from datus.tools.llms_tools.reasoning_sql import reasoning_sql_with_mcp

        # Use mock_llm_create directly (it's already a MockLLMModel instance)
        result = reasoning_sql_with_mcp(
            model=mock_llm_create, input_data=input_data, tools=[], tool_config={"max_turns": 10}
        )
        assert result is not None, "Tool execution should return a result"


class TestLineageTools:
    """Test suite for lineage graph functionality"""

    @pytest.fixture
    def agent_config(self) -> AgentConfig:
        # Use test config file which has bird_sqlite namespace defined
        test_conf_path = Path(__file__).parent.parent.parent / "conf" / "agent.yml"
        return load_agent_config(config=str(test_conf_path), namespace="bird_sqlite")

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
        yaml_path = TEST_DATA_DIR / "SchemaLinkingInput.yaml"
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

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
