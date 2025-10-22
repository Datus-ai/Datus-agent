import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.claude_model import ClaudeModel
from datus.tools.func_tool import db_function_tools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config
from tests.test_tracing import auto_traceable

logger = get_logger(__name__)
set_tracing_disabled(True)


@auto_traceable
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
        result = self.model.generate_with_json_output("Respond with a JSON object containing a greeting message")

        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert len(result) > 0, "Response should not be empty"

        logger.debug(f"JSON response: {result}")

    def test_generate_with_system_prompt(self):
        """Test generation with system and user prompts."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond in JSON format with 'question' and 'answer' fields.",
            },
            {"role": "user", "content": "What is 2+2?"},
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

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        result = await self.model.generate_with_tools(
            prompt=question,
            output_type=str,
            tools=tools,
            instruction=instructions,
        )

        assert result is not None, "MCP response should not be None"
        assert "content" in result, "Response should contain content"
        assert "sql_contexts" in result, "Response should contain sql_contexts"

        logger.debug(f"MCP response: {result.get('content', '')}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream(self):
        """Acceptance test for MCP streaming with complex SSB analytics."""
        instructions = """You are a SQLite expert performing comprehensive analysis on the Star Schema Benchmark
          database. Provide detailed business analytics with multiple queries and insights."""

        complex_scenarios = [
            "Analyze revenue trends by customer region and supplier nation with year-over-year growth",
            "Calculate profitability metrics by part category and manufacturer with discount impact analysis",
            (
                "Perform comprehensive supplier performance analysis including revenue, volume, and "
                "geographic distribution"
            ),
        ]

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        for i, scenario in enumerate(complex_scenarios):
            question = f"database_type='sqlite' task='{scenario}'"

            action_count = 0
            total_content_length = 0

            try:
                async for action in self.model.generate_with_tools_stream(
                    prompt=question,
                    output_type=str,
                    tools=tools,
                    instruction=instructions,
                ):
                    action_count += 1
                    assert action is not None, f"Stream action should not be None for scenario {i+1}"

                    # Track content if available
                    if hasattr(action, "content") and action.content:
                        total_content_length += len(str(action.content))

                    logger.debug(f"Acceptance stream scenario {i+1}, action {action_count}: {type(action)}")

                assert action_count > 0, f"Should receive at least one streaming action for scenario {i+1}"
                logger.debug(
                    f"Acceptance stream scenario {i+1} completed: {action_count} actions, "
                    f"{total_content_length} total content length"
                )
                logger.info(f"Final Action: {action}")
            except DatusException as e:
                if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                    pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
                else:
                    raise
            except Exception:
                raise

            # Only run one scenario to avoid timeout in normal testing
            break

    @pytest.mark.asyncio
    async def test_generate_with_tools_session(self):
        """Test MCP integration with session management."""
        import uuid

        session_id = f"test_mcp_session_{uuid.uuid4().hex[:8]}"

        # Create session
        session = self.model.create_session(session_id)

        instructions = """You are a SQLite expert working with the SSB database.
        Answer questions about the database schema and data."""

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        # First question: explore schema
        question1 = "database_type='sqlite' task='Show me all the tables in the database'"
        result1 = await self.model.generate_with_tools(
            prompt=question1,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        )

        assert result1 is not None
        assert "content" in result1
        assert "sql_contexts" in result1

        # Second question in same session: follow-up query
        question2 = "database_type='sqlite' task='Count the total number of rows in the customer table'"
        result2 = await self.model.generate_with_tools(
            prompt=question2,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        )

        assert result2 is not None
        assert "content" in result2
        assert "sql_contexts" in result2

        # Third question: reference previous answer to test session continuity
        question3 = "database_type='sqlite' task='What's the result of the previous number plus 5?'"
        result3 = await self.model.generate_with_tools(
            prompt=question3,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        )

        assert result3 is not None
        assert "content" in result3
        assert "sql_contexts" in result3

        # Verify session persistence
        session_info = self.model.session_manager.get_session_info(session_id)
        assert session_info["exists"] is True
        assert session_info["item_count"] > 0

        # Cleanup
        self.model.delete_session(session_id)

        logger.debug(f"MCP session Q1: {result1.get('content', '')[:100]}...")
        logger.debug(f"MCP session Q2: {result2.get('content', '')[:100]}...")
