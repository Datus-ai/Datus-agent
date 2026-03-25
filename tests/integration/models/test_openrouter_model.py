"""
Integration tests for OpenRouter model (openai_model.py with type=openrouter).

Nightly-level: requires OPENROUTER_API_KEY environment variable.
Tests real API calls via OpenRouter's unified API endpoint.
"""

import os

import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.openrouter_model import OpenRouterModel
from datus.tools.func_tool import db_function_tools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from tests.conftest import TEST_CONF_DIR, load_acceptance_config
from tests.unit_tests.utils.tracing_utils import auto_traceable

logger = get_logger(__name__)
set_tracing_disabled(True)
load_dotenv()

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"),
]


@auto_traceable
class TestOpenRouterModel:
    """Integration tests for OpenRouter model with real API calls."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        load_dotenv()
        config = load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"))
        self.model = OpenRouterModel(model_config=config["openrouter"])

    def test_initialization(self):
        """Verify OpenRouter model is correctly configured."""
        assert self.model is not None
        assert self.model.model_config.type == "openrouter"

    def test_generate(self):
        """Test basic text generation via OpenRouter."""
        result = self.model.generate("Say hello in one word", temperature=0.1, max_tokens=50)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        logger.info(f"OpenRouter generate result: {result}")

    def test_generate_with_json_output(self):
        """Test JSON output generation via OpenRouter."""
        result = self.model.generate_with_json_output("Respond with a JSON object containing 'greeting': 'hello'")

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0
        logger.info(f"OpenRouter JSON result: {result}")

    def test_generate_with_system_prompt(self):
        """Test generation with system + user messages via OpenRouter."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Always respond in JSON with 'answer' field.",
            },
            {"role": "user", "content": "What is 2+2?"},
        ]

        result = self.model.generate_with_json_output(messages)

        assert result is not None
        assert isinstance(result, dict)
        logger.info(f"OpenRouter system prompt result: {result}")

    @pytest.mark.asyncio
    async def test_generate_with_tools(self):
        """Test OpenRouter model with tool execution (SSB database)."""
        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database.
        The database contains tables: customer, supplier, part, date, and lineorder."""

        question = "database_type='sqlite' task='Count the total number of rows in the customer table'"
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        try:
            result = await self.model.generate_with_tools(
                prompt=question,
                output_type=str,
                tools=tools,
                instruction=instructions,
                max_turns=10,
            )

            assert result is not None
            assert "content" in result
            assert "sql_contexts" in result
            logger.info(f"OpenRouter tools result: {result.get('content', '')[:200]}")
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Skipped: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_generate_with_tools_stream(self):
        """Test OpenRouter model streaming with tool support."""
        instructions = """You are a SQLite expert working with the SSB database.
        Answer questions briefly."""

        question = "database_type='sqlite' task='How many suppliers are there?'"
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        action_count = 0
        try:
            async for action in self.model.generate_with_tools_stream(
                prompt=question,
                output_type=str,
                tools=tools,
                instruction=instructions,
                max_turns=10,
            ):
                action_count += 1
                assert action is not None
                logger.debug(f"OpenRouter stream action {action_count}: {action.action_type}")

            assert action_count > 0
            logger.info(f"OpenRouter stream completed: {action_count} actions")
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Skipped: {e}")
            else:
                raise


@pytest.mark.nightly
@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
class TestOpenRouterMultiModel:
    """Test OpenRouter with different backend models."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        load_dotenv()
        self.config = load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"))

    @pytest.mark.parametrize(
        "model_name",
        [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4.1-mini",
            "google/gemini-2.5-flash",
        ],
    )
    def test_generate_across_backends(self, model_name):
        """Test that OpenRouter routes to different backend providers correctly."""
        from dataclasses import replace

        base_config = self.config["openrouter"]
        model_config = replace(base_config, model=model_name)
        model = OpenRouterModel(model_config=model_config)

        result = model.generate("Say hi in 3 words", temperature=0.1, max_tokens=50)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        logger.info(f"[OpenRouter/{model_name}] result: {result}")
