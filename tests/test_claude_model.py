import pytest
from dotenv import load_dotenv

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
        config = load_agent_config(config="tests/conf/agent.yml")
        self.model = ClaudeModel(model_config=config["anthropic"])

    def test_generate(self):
        """Test basic text generation functionality."""
        result = self.model.generate("Hello", temperature=0.5, max_tokens=100)
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"Generated response: {result}")

    def test_generate_with_json_output(self):
        """Test JSON output generation."""
        result = self.model.generate_with_json_output(
            "Respond with a JSON object containing a greeting message"
        )
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"JSON response: {result}")

    def test_generate_with_system_prompt(self):
        """Test generation with system and user prompts."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond in JSON format with 'question' and 'answer' fields."
            },
            {
                "role": "user", 
                "content": "What is 2+2?"
            }
        ]
        
        result = self.model.generate_with_json_output(messages)
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"System prompt response: {result}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp(self):
        """Test MCP integration with SSB database."""
        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database. 
        The database contains tables: customer, supplier, part, date, and lineorder.
        Focus on business analytics and data relationships.
        
        Output format: {
            "sql": "SELECT ...",
            "result": "Query results...",
            "explanation": "Business explanation..."
        }"""
        
        question = """database_type='sqlite' task='Find the top 5 customers by total revenue from the SSB database'"""
        ssb_db_path = "tests/SSB.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=ssb_db_path)
        
        result = await self.model.generate_with_mcp(
            prompt=question,
            output_type=str,
            mcp_servers={"sqlite": mcp_server},
            instruction=instructions,
        )
        
        assert result is not None, "MCP response should not be None"
        assert "content" in result, "Response should contain content"
        assert "sql_contexts" in result, "Response should contain sql_contexts"
        
        logger.debug(f"MCP response: {result.get('content', '')}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream(self):
        """Test MCP streaming functionality with SSB database."""
        instructions = """You are a SQLite expert analyzing the Star Schema Benchmark database.
        Provide comprehensive business insights with detailed SQL analysis.
        
        Output format: {
            "sql": "SELECT ...",
            "result": "Analysis results...",
            "explanation": "Business insights..."
        }"""
        
        question = """database_type='sqlite' task='Analyze seasonal sales patterns by month and region in the SSB database'"""
        ssb_db_path = "tests/SSB.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=ssb_db_path)
        
        action_count = 0
        async for action in self.model.generate_with_mcp_stream(
            prompt=question,
            output_type=str,
            mcp_servers={"sqlite": mcp_server},
            instruction=instructions,
        ):
            action_count += 1
            assert action is not None, "Stream action should not be None"
            logger.debug(f"Stream action {action_count}: {type(action)}")
        
        assert action_count > 0, "Should receive at least one streaming action"