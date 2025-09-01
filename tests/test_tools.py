import os
import sys
from pathlib import Path

import pytest
import yaml

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
from datus.schemas.node_models import GenerateSQLInput
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata.store import SchemaStorage
from datus.tools.db_tools.snowflake_connector import SnowflakeConnector
from datus.tools.db_tools.starrocks_connector import StarRocksConnector
from datus.tools.lineage_graph_tools import SchemaLineageTool
from datus.tools.llms_tools import LLMTool
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
    def setup_tool(self, llm):
        return LLMTool(model=llm)

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

    def test_tool_has_required_functions(self, setup_tool: LLMTool):
        """Test that the tool has required functions."""
        tool = setup_tool
        assert hasattr(tool, "generate_sql"), "Tool should have generate_sql function"
        assert hasattr(tool, "match_schema"), "Tool should have match_schema function"

    # @pytest.mark.acceptance
    def test_generate_sql(self, setup_tool, test_data):
        """Test basic tool execution."""
        tool = setup_tool
        # using test data from YAML
        input_data = GenerateSQLInput(**test_data[0]["input"])
        result = tool.generate_sql(input_data)
        assert result is not None, "Tool execution should return a result"

    # @pytest.mark.acceptance
    def test_schema_linking(self, setup_tool: LLMTool, rag_storage: SchemaStorage):
        # TODO: fix this test
        """Test schema linking by llm"""
        input_data = SchemaLinkingInput(
            input_text="Describe the information about rulings for card named 'Sublime Epiphany' with number 74s.",
            matching_rate="fast",
            database_type=DBType.SQLITE,
            database_name="card_games",
        )
        match_result = setup_tool.match_schema(input_data=input_data, rag_storage=rag_storage)
        logger.info(f"match result: {len(match_result.table_schemas)} schemas, {len(match_result.table_values)} values")
        assert len(match_result.table_schemas) > 0

    def test_reasoning_sql(self, setup_tool, test_data):
        """Test basic tool execution."""
        tool = setup_tool
        # using test data from YAML
        input_data = GenerateSQLInput(**test_data[0]["input"])
        print(input_data)
        result = tool.generate_sql(input_data)
        assert result is not None, "Tool execution should return a result"


class TestDBTools:
    """Test suite for database tools functionality"""

    @pytest.fixture
    def starrocks_connector(self):
        """Create a StarRocks connector instance"""
        import os

        from dotenv import load_dotenv

        from datus.tools.db_tools import StarRocksConnector

        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

        connector = StarRocksConnector(
            host=os.getenv("STARROCKS_HOST", ""),
            port=os.getenv("STARROCKS_PORT", ""),
            user=os.getenv("STARROCKS_USER", ""),
            password=os.getenv("STARROCKS_PASSWORD", ""),
            database=os.getenv("STARROCKS_DATABASE", ""),
        )
        yield connector
        connector.close()

    def test_sr_connection(self, starrocks_connector: StarRocksConnector):
        """Test database connection functionality"""
        result = starrocks_connector.test_connection()
        assert result is True

    def test_execute_query_sr(self, starrocks_connector: StarRocksConnector):
        """Test SQL query execution functionality"""
        # Test simple query
        result = starrocks_connector.execute({"sql_query": "SELECT 1"})
        assert result.success is True
        assert result.row_count > 0
        assert result.error is None

        result = starrocks_connector.execute({"sql_query": "SELECT count(*) from part"})
        assert result.success is True
        assert result.row_count > 0
        assert result.error is None

        logger.debug(f"select count(*) from part result: {result}")

        result = starrocks_connector.get_tables()
        assert len(result) > 0
        assert "part" in result
        logger.debug(f"get_tables result: {result}")

        result = starrocks_connector.get_schema("part")
        logger.debug(f"get_schema result: {result}")

        result = starrocks_connector.get_sample_rows("ssb_1", ["part"])
        assert len(result) > 0
        logger.debug(f"get_sample_rows result: {result}")

        result = starrocks_connector.get_tables_with_ddl("ssb_1")
        assert len(result) > 0
        logger.debug(f"get_tables_with_ddl ssb_1 result: {len(result)} {result[0].get('table_name')}")

        result = starrocks_connector.get_tables_with_ddl("ssb")
        assert len(result) > 0
        logger.debug(f"get_tables_with_ddl ssb result: {len(result)} {result[0].get('table_name')}")

    @pytest.fixture
    def snowflake_connector(self):
        """Create a Snowflake connector instance"""
        import os

        from dotenv import load_dotenv

        from datus.tools.db_tools import SnowflakeConnector

        # Load environment variables
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

        # Initialize connector with environment variables
        connector = SnowflakeConnector(
            account=os.getenv("SNOWFLAKE_ACCOUNT", "RSRSBDK-YDB67606"),
            user=os.getenv("SNOWFLAKE_USER", "") or os.getenv("SNOWFLAKE_USERNAME", ""),
            password=os.getenv("SNOWFLAKE_PASSWORD", ""),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", ""),
            database=os.getenv("SNOWFLAKE_DATABASE", "GOOGLE_TRNEDS"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        )

        yield connector

        # Clean up connection after tests
        connector.close()

    def test_connection(self, snowflake_connector):
        """Test database connection functionality"""
        result = snowflake_connector.test_connection()
        assert result["success"] is True
        assert result["message"] == "Connection successful"

    # @pytest.mark.acceptance
    def test_execute_query(self, snowflake_connector):
        """Test SQL query execution functionality"""
        # Test simple query
        result = snowflake_connector.execute({"sql_query": "SELECT 1"})
        assert result.success is True
        assert result.row_count > 0
        assert result.error is None

    def test_execute_invalid_query(self, snowflake_connector):
        """Test error handling for invalid SQL queries"""
        result = snowflake_connector.execute({"sql_query": "SELECT * FROM non_existent_table"})
        assert result.success is True  # Note: Implementation returns True even for SQL errors
        assert result.error is not None
        assert "non_existent_table" in result.error.lower()

    def test_execute_multiple_queries(self, snowflake_connector):
        """Test execution of multiple SQL queries"""
        queries = [
            "SELECT 1",
            "SELECT CURRENT_TIMESTAMP()",
            "SELECT * from CRYPTO.CRYPTO_BITCOIN_CASH.TRANSACTIONS limit 5",
        ]
        results = snowflake_connector.execute_queries(queries)
        assert len(results) == 3
        for result in results:
            assert result.success is True
            assert result.error is None
        logger.debug(results)

    def test_validate_input(self, snowflake_connector):
        """Test input parameter validation"""
        # Test valid input
        valid_input = {"sql_query": "SELECT 1", "params": {"param1": "value1"}}
        snowflake_connector.validate_input(valid_input)

        # Test invalid params type
        invalid_input = {"sql_query": "SELECT 1", "params": 123}
        with pytest.raises(ValueError, match="params must be dict or Sequence"):
            snowflake_connector.validate_input(invalid_input)

    def test_execute_query_arrow(self, snowflake_connector):
        """Test SQL query execution with Arrow format"""
        # Test simple query with Arrow format
        result = snowflake_connector.do_execute_arrow({"sql_query": "SELECT 1"})
        assert result.success is True
        assert result.row_count > 0
        assert result.error is None
        assert result.result_format == "arrow"

        # Verify we can convert Arrow data to pandas DataFrame
        import pandas as pd
        import pyarrow as pa

        df = result.sql_return
        if isinstance(df, pa.lib.Table):
            df = df.to_pandas()

        assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert df.shape[0] > 0, "DataFrame should have at least one row"

    # @pytest.mark.acceptance
    def test_execute_multiple_queries_arrow(self, snowflake_connector: SnowflakeConnector):
        """Test execution of multiple SQL queries with Arrow format"""
        queries = [
            "SELECT 1",
            "SELECT CURRENT_TIMESTAMP()",
            "SELECT * from CRYPTO.CRYPTO_BITCOIN_CASH.TRANSACTIONS limit 5",
        ]
        results = snowflake_connector.execute_queries_arrow(queries)
        assert len(results) == 3

        for result in results:
            assert result.success is True
            assert result.error is None
            assert result.result_format == "arrow"

            # Verify each result can be processed as a pandas DataFrame
            import pandas as pd
            import pyarrow

            df = result.sql_return
            if isinstance(df, pyarrow.lib.Table):
                df = df.to_pandas()
            assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"

        # Verify the third query returns the expected number of rows
        assert results[2].row_count == 5, "Should return exactly 5 rows for the third query"

        # Test data manipulation with pandas
        df = results[2].sql_return
        assert len(df.columns) > 0, "DataFrame should have columns"
        if isinstance(df, pyarrow.lib.Table):
            df = df.to_pandas()
        # Log sample data for debugging
        logger.debug(f"Sample data from Arrow result: {df.head(2)}")


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

    def test_get_table_and_values(self, agent_config: AgentConfig):
        """Test get table and values functionality"""
        agent_config.current_namespace = "snowflake"
        setup_lineage_tool = SchemaLineageTool(agent_config=agent_config)
        # Use test data from YAML
        input_data = {
            "database_type": DBType.SNOWFLAKE,
            "database_name": "ETHEREUM_BLOCKCHAIN",
            "table_names": [
                "ETHEREUM_BLOCKCHAIN.ETHEREUM_BLOCKCHAIN.TRACES",
                "ETHEREUM_BLOCKCHAIN.ETHEREUM_BLOCKCHAIN.TRANSACTIONS",
                "ETHEREUM_BLOCKCHAIN.ETHEREUM_BLOCKCHAIN.BLOCKS",
            ],
        }
        schemas, values = setup_lineage_tool.get_table_and_values(
            database_name=input_data["database_name"], table_names=input_data["table_names"]
        )

        logger.debug(f"Result schemas: {schemas}")
        assert len(schemas) == 3, "Invalid schema count"
        assert len(values) == 3, "Invalid value count"
        agent_config.current_namespace = "bird_sqlite"

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
