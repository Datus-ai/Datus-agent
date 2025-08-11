import os

import pytest
from dotenv import load_dotenv

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.gemini_model import GeminiModel
from datus.models.openai_model import OpenAIModel
from datus.tools.mcp_server import MCPServer
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from tests.test_tracing import auto_traceable

logger = get_logger(__name__)


@auto_traceable
class TestOpenAIModel:
    """Test suite for OpenAI models."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method."""
        load_dotenv()
        config = load_agent_config(config="tests/conf/agent.yml")

        # Skip if API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        self.model_config = config.models.get("openai-4o-mini")
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
        ssb_db_path = "tests/data/SSB.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=ssb_db_path)

        try:
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
        except DatusException as e:
            if e.error_code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
            else:
                raise
        except Exception:
            raise


class TestKimiModel:
    """Test suite for Kimi (Moonshot) K2 model."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method."""
        load_dotenv()
        config = load_agent_config(config="tests/conf/agent.yml")

        # Skip if API key is not available
        if not os.getenv("KIMI_API_KEY"):
            pytest.skip("KIMI_API_KEY not available")

        self.model_config = config.models.get("kimi-k2")
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
        ssb_db_path = "tests/data/SSB.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=ssb_db_path)

        try:
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
    def setup_method(self):
        """Set up test environment before each test method."""
        load_dotenv()

        # Skip if API key is not available
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not available")

        # Skip if Gemini configuration is not added yet
        config = load_agent_config(config="tests/conf/agent.yml")
        self.model_config = config.models.get("gemini-2.5")

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
        ssb_db_path = "tests/data/SSB.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=ssb_db_path)

        try:
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
        except DatusException as e:
            if e.error_code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
            else:
                raise
        except Exception:
            raise
