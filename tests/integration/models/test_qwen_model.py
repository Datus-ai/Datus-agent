import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config import AgentConfig
from datus.models.qwen_model import QwenModel
from datus.tools.func_tool import db_function_tools
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)
set_tracing_disabled(True)


@pytest.fixture
def agent_config() -> AgentConfig:
    load_dotenv()
    return load_acceptance_config()


class TestQwenModel:
    """Test suite for the QwenModel class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, agent_config):
        """Set up test environment before each test method."""

        self.model = QwenModel(model_config=agent_config.models["qwen"])

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

    def test_enable_thinking(self, agent_config: AgentConfig):
        """Test Qwen's enable_thinking functionality."""
        qwen_config = agent_config.models.get("qwen")

        if not qwen_config:
            pytest.skip("qwen configuration not found in test config")

        # Test with enable_thinking=True
        prompt = "Think step by step: If I have 15 apples and give away 4, then buy 7 more, how many do I have?"
        result_with_thinking = self.model.generate(prompt, enable_thinking=True, temperature=0.1, max_tokens=300)

        assert result_with_thinking is not None, "Response with thinking should not be None"
        assert isinstance(result_with_thinking, str), "Response should be a string"
        assert len(result_with_thinking) > 0, "Response should not be empty"

        # Test with enable_thinking=False for comparison
        result_without_thinking = self.model.generate(prompt, enable_thinking=False, temperature=0.1, max_tokens=300)

        assert result_without_thinking is not None, "Response without thinking should not be None"
        assert isinstance(result_without_thinking, str), "Response should be a string"
        assert len(result_without_thinking) > 0, "Response should not be empty"

        # Check if the response with thinking shows reasoning process
        result_lower = result_with_thinking.lower()
        thinking_indicators = ["step", "first", "then", "therefore", "so", "because", "think"]
        has_thinking = any(indicator in result_lower for indicator in thinking_indicators)

        # The response should contain thinking indicators when enable_thinking=True
        logger.debug(f"Qwen enable_thinking=True response: {result_with_thinking}")
        logger.debug(f"Qwen enable_thinking=False response: {result_without_thinking}")
        logger.debug(f"Thinking indicators found: {has_thinking}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp(self):
        """Test MCP integration with SSB database."""
        if not hasattr(self.model, "generate_with_mcp"):
            pytest.skip("QwenModel does not support generate_with_mcp")

        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database.
        The database contains tables: customer, supplier, part, date, and lineorder.
        Your task is to:
        1. Understand the user's business question
        2. Generate appropriate SQL queries for the SSB schema
        3. Execute the queries using the provided tools
        4. Present the results clearly

        Output format: {
            "sql": "SELECT ...",
            "result": "Query results...",
            "explanation": "Business explanation..."
        }"""

        question = """database_type='sqlite' task='Find the total number of customers by region in the SSB database'"""

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
        """Test MCP streaming functionality with SSB database."""
        instructions = """You are a SQLite expert analyzing the Star Schema Benchmark database.
        Provide detailed analysis of the SSB data with business insights.

        Output format: {
            "sql": "SELECT ...",
            "result": "Analysis results...",
            "explanation": "Business insights..."
        }"""

        question = "database_type='sqlite' task='Analyze the revenue trends by year from the lineorder table'"

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        action_count = 0
        async for action in self.model.generate_with_tools_stream(
            prompt=question,
            output_type=str,
            tools=tools,
            instruction=instructions,
        ):
            action_count += 1
            assert action is not None, "Stream action should not be None"
            logger.debug(f"Stream action {action_count}: {type(action)}")

        assert action_count > 0, "Should receive at least one streaming action"
        logger.info(f"action count: {action_count}")
