"""
Integration tests for Claude subscription token model (claude_model.py with auth_type=subscription).

Nightly-level: requires CLAUDE_CODE_OAUTH_TOKEN environment variable.
Tests real API calls via Bearer auth + Claude Code client headers.
"""

import os

import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.claude_model import ClaudeModel
from datus.tools.func_tool import db_function_tools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from tests.conftest import TEST_CONF_DIR, load_acceptance_config
from tests.unit_tests.utils.tracing_utils import auto_traceable

logger = get_logger(__name__)
set_tracing_disabled(True)

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(not os.getenv("CLAUDE_CODE_OAUTH_TOKEN"), reason="CLAUDE_CODE_OAUTH_TOKEN not set"),
]


@auto_traceable
class TestClaudeSubscriptionModel:
    """Integration tests for Claude subscription token (Bearer auth) with real API calls."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        load_dotenv()
        config = load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"))
        self.model = ClaudeModel(model_config=config["claude-subscription"])

    def test_auth_config(self):
        """Verify subscription auth is correctly configured."""
        assert self.model._is_oauth_token is True
        assert self.model.use_native_api is True

    def test_generate(self):
        """Test basic text generation via subscription token + Bearer auth."""
        result = self.model.generate("Say hello in one word", max_tokens=50)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        logger.info(f"Claude subscription generate result: {result}")

    def test_generate_with_json_output(self):
        """Test JSON output generation via subscription token."""
        result = self.model.generate_with_json_output("Respond with a JSON object containing 'greeting': 'hello'")

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0
        logger.info(f"Claude subscription JSON result: {result}")

    def test_generate_with_system_prompt(self):
        """Test generation with system + user messages via subscription token."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond in JSON format with 'question' and 'answer' fields.",
            },
            {"role": "user", "content": "What is 2+2?"},
        ]

        result = self.model.generate_with_json_output(messages)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0
        logger.info(f"Claude subscription system prompt result: {result}")

    @pytest.mark.asyncio
    async def test_generate_with_tools(self):
        """Test Claude subscription model with tool execution (SSB database)."""
        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database.
        The database contains tables: customer, supplier, part, date, and lineorder."""

        question = "database_type='sqlite' task='Count the total number of rows in the customer table'"
        agent_config = load_acceptance_config(datasource="ssb_sqlite")
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
            logger.info(f"Claude subscription tools result: {result.get('content', '')[:200]}")
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Skipped: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_generate_with_tools_stream(self):
        """Test Claude subscription model streaming with tool support."""
        instructions = """You are a SQLite expert working with the SSB database.
        Answer questions briefly."""

        question = "database_type='sqlite' task='How many suppliers are there?'"
        agent_config = load_acceptance_config(datasource="ssb_sqlite")
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
                logger.debug(f"Claude subscription stream action {action_count}: {action.action_type}")

            assert action_count > 0
            logger.info(f"Claude subscription stream completed: {action_count} actions")
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Skipped: {e}")
            else:
                raise


@pytest.mark.nightly
@pytest.mark.skipif(not os.getenv("CLAUDE_CODE_OAUTH_TOKEN"), reason="CLAUDE_CODE_OAUTH_TOKEN not set")
class TestClaudeSubscriptionMultiModel:
    """Test that subscription token works with different Claude model variants."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        load_dotenv()
        self.config = load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"))

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
        ],
    )
    def test_generate_across_models(self, model_name):
        """Test that subscription token works across Sonnet and Opus models."""
        from dataclasses import replace

        base_config = self.config["claude-subscription"]
        model_config = replace(base_config, model=model_name)
        model = ClaudeModel(model_config=model_config)

        result = model.generate("Say hi in 3 words", max_tokens=50)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        logger.info(f"[{model_name}] result: {result}")
