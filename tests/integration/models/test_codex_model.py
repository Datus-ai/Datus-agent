"""
Integration tests for Codex OAuth model (codex_model.py).

Nightly-level: requires CODEX_OAUTH_TOKEN environment variable.
Tests real API calls to the Codex Responses API via OAuth token.
"""

import os

import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.codex_model import CodexModel
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
    pytest.mark.skipif(not os.getenv("CODEX_OAUTH_TOKEN"), reason="CODEX_OAUTH_TOKEN not set"),
]


@auto_traceable
class TestCodexModel:
    """Integration tests for Codex OAuth model with real API calls."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        load_dotenv()
        config = load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"))
        self.model = CodexModel(model_config=config["codex"])

    def test_generate(self):
        """Test basic text generation via Codex Responses API."""
        result = self.model.generate(
            "Say hello in one word", instructions="You are a helpful assistant.", max_tokens=50
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        logger.info(f"Codex generate result: {result}")

    def test_generate_with_json_output(self):
        """Test JSON output generation via Codex Responses API."""
        result = self.model.generate_with_json_output(
            "Respond with a JSON object containing 'greeting': 'hello'",
            instructions="You are a helpful assistant. Always respond in valid JSON.",
        )

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0
        logger.info(f"Codex JSON result: {result}")

    @pytest.mark.asyncio
    async def test_generate_with_tools(self):
        """Test Codex model with tool execution (SSB database)."""
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
            logger.info(f"Codex tools result: {result.get('content', '')[:200]}")
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Skipped: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_generate_with_tools_stream(self):
        """Test Codex model streaming with tool support."""
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
                logger.debug(f"Codex stream action {action_count}: {action.action_type}")

            assert action_count > 0
            logger.info(f"Codex stream completed: {action_count} actions")
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Skipped: {e}")
            else:
                raise

    def test_token_count(self):
        """Test token count estimation."""
        count = self.model.token_count("Hello world, this is a test prompt.")
        assert count > 0
        assert isinstance(count, int)

    def test_context_length(self):
        """Test context length returns a valid value."""
        length = self.model.context_length()
        assert length is not None
        assert length > 0
