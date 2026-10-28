import os
import sys
from pathlib import Path

import pytest
import yaml

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.tools.lineage_graph_tools import SchemaLineageTool
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from tests.conftest import TEST_DATA_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_logger(__name__)


class TestLineageTools:
    """Test suite for lineage graph functionality"""

    @pytest.fixture
    def agent_config(self, tmp_path, monkeypatch) -> AgentConfig:
        # Use test config file which has bird_sqlite datasource defined.
        # The yml points ``home`` at the relative path ``.datus_test_data``,
        # which ``Path.resolve()`` anchors to the current cwd. Switch cwd to
        # ``tmp_path`` so every derived path (path_manager, storage data_dir,
        # etc.) lands inside the pytest-managed tmp dir — safe under xdist and
        # free of repo-root pollution.
        monkeypatch.chdir(tmp_path)
        test_conf_path = Path(__file__).parent.parent.parent / "conf" / "agent.yml"
        return load_agent_config(
            config=str(test_conf_path),
            datasource="bird_sqlite",
            home=str(tmp_path),
        )

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
