import os

import pytest
from dotenv import load_dotenv
from langsmith import traceable

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.claude_model import ClaudeModel
from datus.tools.mcp_server import MCPServer
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class TestClaudeModel:
    """Test suite for the ClaudeModel class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method."""
        load_dotenv()

        config = load_agent_config()
        model_config = config.active_model()

        # Initialize the model with default parameters
        self.model = ClaudeModel(
            model_config=model_config,
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
        )

    @pytest.mark.asyncio
    @traceable(name="test_with_mcp")
    async def test_with_mcp(self):
        try:
            # Create model instance for MCP testing
            model = ClaudeModel(model_config=self.model.model_config)

            # Define Snowflake expert instructions
            instructions = """You are a snowflake expert. Your task is to:
1. Understand the user's question about data analysis
2. Generate appropriate SQL queries
3. Execute the queries using the provided tools
4. Present the results in a clear and concise manner
*Enclose all column names in double quotes to comply with Snowflake syntax requirements
and avoid grammar errors.* When referencing table names in Snowflake SQL,
you must include both the database_name and schema_name.
output format: {
  "sql": "SELECT * FROM database.schema.table LIMIT 10",
  "result": "Results here...",
  "explanation": "Explanation here..."
}
"""
            # Test query for Ethereum transactions
            question = (
                "database_type='snowflake' task='how many eth transactions "
                "in the last 7 days?' database_name='ETHEREUM_BLOCKCHAIN' "
                "schema_name='ETHEREUM_BLOCKCHAIN' table_name='TRANSACTIONS'"
            )

            mcp_server = MCPServer.get_snowflake_mcp_server()

            # Get MCP directory from environment
            directory = os.environ.get("SNOWFLAKE_MCP_DIR")
            if not directory:
                pytest.fail("Missing required environment variable: SNOWFLAKE_MCP_DIR")

            # Execute MCP generation with specified parameters
            result = await model.generate_with_mcp(
                prompt=question,
                output_type=str,
                mcp_servers={"snowflake": mcp_server},
                instruction=instructions,
            )

            # Log and validate MCP response
            logger.debug(f"\nReceived response: {result.get('content', '')}")
            logger.debug(f"\nSQL contexts: {result.get('sql_contexts', '')}")

            assert result is not None, "Response should not be None"

        except Exception as e:
            pytest.fail(f"Failed to get response from API: {str(e)}")
