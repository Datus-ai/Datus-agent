import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv
from langsmith import traceable

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.qwen_model import QwenModel
from datus.tools.mcp_server import MCPServer
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

set_tracing_disabled(True)


class TestQwenModel:
    """Test suite for the DeepSeekModel class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method."""
        load_dotenv()

        config = load_agent_config()
        config.current_namespace = "snowflake"
        model_config = config.active_model()

        # Initialize the model with default parameters
        self.model = QwenModel(model_config=model_config)

    def test_basic_chat(self):
        """Test basic chat functionality with real API calls and logging."""
        try:
            # Call generate method with a basic chat prompt
            result = self.model.generate("Hello", temperature=0.5, top_p=0.9, max_tokens=1000)

            # Verify the response is not empty and has expected properties
            assert result is not None, "Response should not be None"
            assert isinstance(result, str), "Response should be a string"
            assert len(result) > 0, "Response should not be empty"

            # Log the successful response
            print(f"\nReceived response: {result}")

        except Exception as e:
            pytest.fail(f"Failed to get response from API: {str(e)}")

    def test_json_chat(self):
        try:
            # Test JSON output generation capability
            result = self.model.generate_with_json_output("Hello, how are you, response in json format")

            # Validate JSON response properties
            assert result is not None, "Response should not be None"
            assert isinstance(result, dict), "Response should be a dict"
            assert len(result) > 0, "Response should not be empty"

            print(f"\nReceived response: {result}")

        except Exception as e:
            pytest.fail(f"Failed to get response from API: {str(e)}")

    def test_system_prompt(self):
        try:
            # Test system prompt with JSON output
            result = self.model.generate_with_json_output(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant., response in json format, "
                        "like {'question': 'xxx', 'answer': 'xxx'}",
                    },
                    {"role": "user", "content": "Hello, how many r's in the strawbeery?"},
                ]
            )

            # Validate system prompt response
            assert result is not None, "Response should not be None"
            assert isinstance(result, dict), "Response should be a dict"
            assert len(result) > 0, "Response should not be empty"

            print(f"\nReceived response: {result}")

        except Exception as e:
            pytest.fail(f"Failed to get response from API: {str(e)}")

    @pytest.mark.asyncio
    @traceable(name="test_with_mcp")
    async def test_with_mcp(self):
        try:
            # Create model instance for MCP testing
            model = QwenModel(model_config=self.model.model_config)

            # Define Snowflake expert instructions
            instructions = """You are a snowflake expert. Your task is to:
            1. Understand the user's question about data analysis
            2. Generate appropriate SQL queries
            3. Execute the queries using the provided tools
            4. Present the results in a clear and concise manner
                *Enclose all column names in double quotes to comply with Snowflake syntax
                requirements and avoid grammar errors.* When referencing table names
                in Snowflake SQL, you must include both the database_name and schema_name.
                output format: {
                    "sql": "SELECT * FROM database.schema.table LIMIT 10",
                    "result": "Results here...",
                    "explanation": "Explanation here..."
                }
            """
            # Test query for Ethereum transactions
            question = (
                "database_type='snowflake' task='how many eth transactions in the last 7 days?' "
                "database_name='ETHEREUM_BLOCKCHAIN' schema_name='ETHEREUM_BLOCKCHAIN' "
                "table_name='TRANSACTIONS'"
            )

            mcp_server = MCPServer.get_snowflake_mcp_server()

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

            # final_result = ReasoningSQLResponse(**result.get('content', {}))
            # logger.debug(f"\nFinal result: {final_result.sql}
            # {final_result.result} {final_result.explanation}")
            assert result is not None, "Response should not be None"

        except Exception as e:
            pytest.fail(f"Failed to get response from API: {str(e)}")

    @pytest.mark.asyncio
    @traceable(name="test_with_mcp")
    async def test_with_mcp_starrocks(self):
        try:
            # Create model instance for MCP testing
            model = QwenModel(model_config=self.model.model_config)

            # Define StarRocks expert instructions
            instructions = """You are a StarRocks expert. Your task is to:
            1. Understand the user's question about data analysis
            2. Generate appropriate SQL queries for StarRocks
            3. Execute the queries using the provided tools
            4. Present the results in a clear and concise manner
                *Follow StarRocks SQL syntax requirements.* When referencing table names
                in StarRocks SQL, use appropriate database and table naming conventions.
                output format: {
                    "sql": "SELECT * FROM database.table LIMIT 10",
                    "result": "Results here...",
                    "explanation": "Explanation here..."
                }
            """
            # Test query for Ethereum transactions
            question = (
                "database_type='starrocks' task='Calculate gross profit (revenue - supply cost) by year and"
                " customer nation for orders where customers and suppliers are in the Americas and parts are"
                " manufactured by 'MFGR#1' or 'MFGR#2', sorted by year and nation?' "
            )

            mcp_server = MCPServer.get_starrocks_mcp_server()

            # Execute MCP generation with specified parameters
            result = await model.generate_with_mcp(
                prompt=question,
                output_type=str,
                mcp_servers={"starrocks": mcp_server},
                instruction=instructions,
            )

            # Log and validate MCP response
            logger.debug(f"\nReceived response: {result.get('content', '')}")
            logger.debug(f"\nSQL contexts: {result.get('sql_contexts', '')}")

            # final_result = ReasoningSQLResponse(**result.get('content', {}))
            # logger.debug(f"\nFinal result: {final_result.sql} {final_result.result} {final_result.explanation}")
            assert result is not None, "Response should not be None"

        except Exception as e:
            pytest.fail(f"Failed to get response from API: {str(e)}")

    @pytest.mark.asyncio
    @traceable(name="test_with_mcp")
    async def test_with_mcp_sqlite(self):
        try:
            # Create model instance for MCP testing
            model = QwenModel(model_config=self.model.model_config)

            # Define SQLite expert instructions
            instructions = """You are a SQLite expert. Your task is to:
            1. Understand the user's question about data analysis
            2. Generate appropriate SQL queries for SQLite
            3. Execute the queries using the provided tools
            4. Present the results in a clear and concise manner
                *Follow SQLite SQL syntax requirements.* When referencing table names
                in SQLite SQL, use appropriate naming conventions.
                output format: {
                    "sql": "SELECT * FROM table LIMIT 10",
                    "result": "Results here...",
                    "explanation": "Explanation here..."
                }
            """
            # Test query for basic data analysis
            question = "database_type='sqlite' task='Create a simple table for tracking products and show sample data' "

            # Use a test database path
            test_db_path = "./test_sqlite_mcp.db"
            mcp_server = MCPServer.get_sqlite_mcp_server(db_path=test_db_path)

            # Execute MCP generation with specified parameters
            result = await model.generate_with_mcp(
                prompt=question,
                output_type=str,
                mcp_servers={"sqlite": mcp_server},
                instruction=instructions,
            )

            # Log and validate MCP response
            logger.debug(f"\nReceived response: {result.get('content', '')}")
            logger.debug(f"\nSQL contexts: {result.get('sql_contexts', '')}")

            assert result is not None, "Response should not be None"

        except Exception as e:
            pytest.fail(f"Failed to get response from API: {str(e)}")
