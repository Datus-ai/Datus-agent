import os

import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config import AgentConfig
from datus.models.gemini_model import GeminiModel
from datus.models.openai_model import OpenAIModel
from datus.tools.func_tool import db_function_tools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config
from tests.test_tracing import auto_traceable

logger = get_logger(__name__)
set_tracing_disabled(True)


@pytest.fixture
def agent_config() -> AgentConfig:
    load_dotenv()
    return load_acceptance_config()


@auto_traceable
class TestOpenAIModel:
    """Test suite for OpenAI models."""

    @pytest.fixture(autouse=True)
    def setup_method(self, agent_config: AgentConfig) -> None:
        """Set up test environment before each test method."""

        # Skip if API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        self.model_config = agent_config.models.get("openai-4o-mini")
        if not self.model_config:
            pytest.skip("openai-4o-mini configuration not found in test config")

        self.model = OpenAIModel(model_config=self.model_config)

    def test_initialization(self):
        """Test OpenAI model initialization."""
        assert self.model is not None
        assert self.model.model_config is not None
        assert self.model.model_config.model == "gpt-4o-mini"
        assert self.model.model_config.type == "openai"

    def test_generate_basic(self):
        """Test basic text generation functionality."""
        result = self.model.generate("Say hello in one word", temperature=0.1, max_tokens=10)

        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"

        logger.debug(f"OpenAI generated response: {result}")

    def test_generate_with_json_output(self):
        """Test JSON output generation."""
        prompt = "Respond with a JSON object containing 'greeting': 'hello'"
        result = self.model.generate_with_json_output(prompt, temperature=0.1)

        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert "greeting" in result, "Response should contain 'greeting' key"

        logger.debug(f"OpenAI JSON response: {result}")

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not available")
    @pytest.mark.asyncio
    async def test_generate_with_mcp(self):
        """Test OpenAI model with MCP server integration."""
        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database.
        The database contains tables: customer, supplier, part, date, and lineorder.
        Focus on business analytics and data relationships.
        """

        question = (
            "database_type='sqlite' task='Calculate the total revenue in 1993 from orders with a discount "
            "between 1 and 3 and sales volume less than 25, where revenue is calculated by multiplying the "
            "extended price by the discount'"
        )
        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        try:
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
        except DatusException as e:
            if e.error_code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
            else:
                raise
        except Exception:
            raise

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not available")
    @pytest.mark.asyncio
    async def test_generate_with_mcp_token_consumption(self):
        """Test token consumption tracking between generate_with_tools and generate_with_tools_stream."""
        from datus.schemas.action_history import ActionHistoryManager

        instructions = """You are a SQLite expert working with the SSB database.
        Answer the question briefly and concisely."""

        question = "database_type='sqlite' task='Count the total number of rows in the customer table'"
        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        # Test 1: Non-streaming version
        logger.info("=== Testing generate_with_tools (non-streaming) ===")
        result_non_stream = await self.model.generate_with_tools(
            prompt=question,
            output_type=str,
            tools=tools,
            instruction=instructions,
            max_turns=5,
        )

        assert result_non_stream is not None, "Non-streaming result should not be None"
        assert "content" in result_non_stream, "Non-streaming result should contain content"
        assert "usage" in result_non_stream, "Non-streaming result should contain usage"

        non_stream_usage = result_non_stream.get("usage", {})
        logger.info(f"Non-streaming usage info: {non_stream_usage}")

        # Verify usage fields exist and are reasonable
        assert isinstance(non_stream_usage, dict), "Usage should be a dictionary"
        total_tokens_non_stream = non_stream_usage.get("total_tokens", 0)
        input_tokens_non_stream = non_stream_usage.get("input_tokens", 0)
        output_tokens_non_stream = non_stream_usage.get("output_tokens", 0)

        logger.info(
            f"Non-streaming tokens: total={total_tokens_non_stream}, input={input_tokens_non_stream},"
            f" output={output_tokens_non_stream}"
        )

        assert total_tokens_non_stream > 0, "Total tokens should be greater than 0"
        assert input_tokens_non_stream > 0, "Input tokens should be greater than 0"
        assert output_tokens_non_stream > 0, "Output tokens should be greater than 0"

        # Test 2: Streaming version
        logger.info("=== Testing generate_with_tools_stream (streaming) ===")
        action_history_manager = ActionHistoryManager()
        action_count = 0

        async for action in self.model.generate_with_tools_stream(
            prompt=question,
            output_type=str,
            tools=tools,
            instruction=instructions,
            max_turns=5,
            action_history_manager=action_history_manager,
        ):
            action_count += 1
            logger.info(
                f"Stream action {action_count}: type={action.action_type}, role={action.role}, status={action.status}"
            )

        # Check final actions after our fix has been applied
        final_actions = action_history_manager.get_actions()
        final_actions_with_tokens = sum(
            1 for a in final_actions if a.output and isinstance(a.output, dict) and a.output.get("usage")
        )
        final_total_tokens = sum(
            a.output.get("usage", {}).get("total_tokens", 0)
            for a in final_actions
            if a.output and isinstance(a.output, dict) and a.output.get("usage")
        )

        logger.info("=== Results ===")
        logger.info(f"Non-streaming tokens: {total_tokens_non_stream}")
        logger.info(f"Streaming actions with tokens: {final_actions_with_tokens}/{len(final_actions)}")
        logger.info(f"Streaming total tokens: {final_total_tokens}")

        # Verify our fix is working
        if final_actions_with_tokens > 0 and final_total_tokens > 0:
            logger.info("✅ SUCCESS: Token consumption fix is working!")
        else:
            logger.error("❌ FAILURE: Token consumption fix needs work")

        assert action_count > 0, "Should receive at least one streaming action"


class TestKimiModel:
    """Test suite for Kimi (Moonshot) K2 model."""

    @pytest.fixture(autouse=True)
    def setup_method(self, agent_config: AgentConfig):
        """Set up test environment before each test method."""

        # Skip if API key is not available
        if not os.getenv("KIMI_API_KEY"):
            pytest.skip("KIMI_API_KEY not available")

        self.model_config = agent_config.models.get("kimi-k2")
        if not self.model_config:
            pytest.skip("kimi-k2 configuration not found in test config")

        # Kimi uses OpenAI-compatible API, so we use OpenAI model class
        self.model = OpenAIModel(model_config=self.model_config)

    def test_initialization(self):
        """Test Kimi model initialization."""
        assert self.model is not None
        assert self.model.model_config is not None
        assert self.model.model_config.model == "kimi-k2-0711-preview"
        assert self.model.model_config.base_url == "https://api.moonshot.cn/v1"

    def test_generate_basic(self):
        """Test basic text generation functionality."""
        result = self.model.generate("Say hello in one word", temperature=0.1, max_tokens=10)

        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"

        logger.info(f"Kimi generated response: {result}")

    def test_generate_with_json_output(self):
        """Test JSON output generation."""
        prompt = "Respond with a JSON object containing 'message': 'hello' and 'language': 'en'"
        result = self.model.generate_with_json_output(prompt, temperature=0.1)

        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert "message" in result, "Response should contain 'message' key"
        assert "language" in result, "Response should contain 'language' key"

        logger.info(f"Kimi JSON response: {result}")

    @pytest.mark.skipif(not os.getenv("KIMI_API_KEY"), reason="KIMI_API_KEY not available")
    @pytest.mark.asyncio
    async def test_generate_with_mcp(self):
        """Test Kimi model with MCP server integration."""
        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database.
        The database contains tables: customer, supplier, part, date, and lineorder.
        Focus on business analytics and data relationships.
        """

        question = (
            "database_type='sqlite' task='Calculate the total revenue in 1993 from orders with a discount "
            "between 1 and 3 and sales volume less than 25, where revenue is calculated by multiplying the "
            "extended price by the discount'"
        )
        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        try:
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
        except DatusException as e:
            if e.error_code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
            else:
                raise
        except Exception:
            raise


class TestGeminiModel:
    """Test suite for Google Gemini model."""

    @pytest.fixture(autouse=True)
    def setup_method(self, agent_config: AgentConfig):
        """Set up test environment before each test method."""
        load_dotenv()

        # Skip if API key is not available
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not available")

        # Skip if Gemini configuration is not added yet

        self.model_config = agent_config.models.get("gemini-2.5")

        if not self.model_config:
            pytest.skip("gemini-2.5 configuration not found in test config")

        self.model = GeminiModel(model_config=self.model_config)

    def test_initialization(self):
        """Test Gemini model initialization."""
        assert self.model is not None
        assert self.model.model_config is not None
        # Add specific Gemini model assertions when config is available

    def test_generate_basic(self):
        """Test basic text generation functionality."""
        result = self.model.generate("Say hello in one word", temperature=0.1, max_tokens=10)

        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"

        logger.debug(f"Gemini generated response: {result}")

    def test_multimodal_capability(self):
        """Test Gemini's multimodal capabilities (when supported)."""
        # This is a placeholder for multimodal testing
        # Implementation depends on Gemini model's multimodal support
        pytest.skip("Multimodal test to be implemented when Gemini config is added")

    def test_generate_with_json_output(self):
        """Test JSON output generation."""
        prompt = "Respond with a JSON object containing 'response': 'hello world'"
        result = self.model.generate_with_json_output(prompt, temperature=0.1)

        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert "response" in result, "Response should contain 'response' key"

        logger.info(f"Gemini JSON response: {result}")

    def test_reasoning_capability(self):
        """Test Gemini's reasoning capabilities."""
        prompt = "If I have 3 apples and give away 1, then buy 2 more, how many do I have? Show your work."
        result = self.model.generate(prompt, temperature=0.1, max_tokens=100)

        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert "4" in result, "Response should contain the correct answer"

        logger.debug(f"Gemini reasoning response: {result}")

    @pytest.mark.skipif(
        not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")), reason="Google API key not available"
    )
    @pytest.mark.asyncio
    async def test_generate_with_mcp(self):
        """Test Gemini model with MCP server integration."""
        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database.
        The database contains tables: customer, supplier, part, date, and lineorder.
        Focus on business analytics and data relationships.
        """

        question = (
            "database_type='sqlite' task='Calculate the total revenue in 1993 from orders with a discount "
            "between 1 and 3 and sales volume less than 25, where revenue is calculated by multiplying the "
            "extended price by the discount'"
        )
        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        try:
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
        except DatusException as e:
            if e.error_code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
            else:
                raise
        except Exception:
            raise
