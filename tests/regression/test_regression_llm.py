"""
Regression Tests: Multi-LLM Provider Compatibility (R-01 ~ R-06)

Tests each supported LLM provider with 2 model versions:
- R-01: OpenAI (gpt-4o-mini, gpt-4.1-mini)
- R-02: Claude (claude-haiku-4-5, claude-sonnet-4-5)
- R-03: Gemini (gemini-2.5-flash, gemini-3-flash-preview)
- R-04: Qwen (qwen3-coder-plus, qwen-plus)
- R-05: Kimi (kimi-k2.5, kimi-k2-turbo-preview)
- R-06: Runtime model switching across providers
"""
import asyncio
import os
from dataclasses import replace

import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config import AgentConfig, ModelConfig
from datus.models.base import LLMBaseModel
from datus.schemas.action_history import ActionHistoryManager
from datus.tools.func_tool import db_function_tools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)
set_tracing_disabled(True)


# ============================================================
# Provider-Model Registry (the ONLY place to add new versions)
# ============================================================
PROVIDER_MODELS = {
    "R01-openai": {
        "type": "openai",
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4.1-mini", "gpt-5.2"],
    },
    "R02-claude": {
        "type": "claude",
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-haiku-4-5", "claude-sonnet-4-5"],
    },
    "R03-gemini": {
        "type": "gemini",
        "env_var": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-3-flash-preview"],
    },
    "R04-qwen": {
        "type": "qwen",
        "env_var": "DASHSCOPE_API_KEY",
        "models": ["qwen3-coder-plus", "qwen-plus"],
    },
    "R05-kimi": {
        "type": "kimi",
        "env_var": "KIMI_API_KEY",
        "models": ["kimi-k2.5", "kimi-k2-turbo-preview"],
    },
}

# Models with restricted sampling parameters (reasoning/thinking models)
# These models only accept specific temperature/top_p values and may require enable_thinking.
RESTRICTED_MODEL_PARAMS = {
    "kimi-k2.5": {"temperature": 1.0, "top_p": 0.95, "enable_thinking": True},
    "qwen3-coder-plus": {"temperature": 1.0, "top_p": 0.95},
}


# ============================================================
# Helpers
# ============================================================
def require_api_key(env_var: str):
    """Fail test if the required API key environment variable is missing."""
    if not os.getenv(env_var):
        pytest.fail(f"Missing env var: {env_var}")


def _find_config_by_type(agent_config: AgentConfig, provider_type: str) -> ModelConfig:
    """Find first model config matching the given provider type."""
    for config in agent_config.models.values():
        if config.type == provider_type:
            return config
    raise KeyError(f"No config found for type: {provider_type}")


def create_model_with_version(agent_config: AgentConfig, provider_type: str, model_name: str) -> LLMBaseModel:
    """Create a model instance with a specific version, using type-based config lookup."""
    base_config = _find_config_by_type(agent_config, provider_type)
    replace_kwargs = {}
    if base_config.model != model_name:
        replace_kwargs["model"] = model_name
    # Apply restricted sampling parameters for reasoning/thinking models
    if model_name in RESTRICTED_MODEL_PARAMS:
        for param, value in RESTRICTED_MODEL_PARAMS[model_name].items():
            if getattr(base_config, param) is None:
                replace_kwargs[param] = value
    target_config = replace(base_config, **replace_kwargs) if replace_kwargs else base_config
    model_class_name = LLMBaseModel.MODEL_TYPE_MAP[provider_type]
    module = __import__(f"datus.models.{provider_type}_model", fromlist=[model_class_name])
    model_class = getattr(module, model_class_name)
    return model_class(model_config=target_config)


def _build_regression_params():
    """Build pytest.param list from PROVIDER_MODELS."""
    params = []
    for case_id, spec in PROVIDER_MODELS.items():
        for model in spec["models"]:
            params.append(pytest.param(spec["type"], spec["env_var"], model, id=f"{case_id}-{model}"))
    return params


REGRESSION_PARAMS = _build_regression_params()


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module")
def agent_config() -> AgentConfig:
    load_dotenv()
    return load_acceptance_config()


@pytest.fixture(scope="module")
def ssb_agent_config() -> AgentConfig:
    load_dotenv()
    return load_acceptance_config(namespace="ssb_sqlite")


@pytest.fixture(scope="module")
def ssb_tools(ssb_agent_config) -> list:
    return db_function_tools(ssb_agent_config)


# ============================================================
# R-01 ~ R-05: LLM Compatibility Tests
# ============================================================
@pytest.mark.regression
class TestRegressionLLMCompatibility:
    """Test each provider x model version for basic LLM capabilities."""

    @pytest.mark.parametrize("provider_type,env_var,model_name", REGRESSION_PARAMS)
    def test_create_model(self, agent_config, provider_type, env_var, model_name):
        """Factory type resolution and instance creation."""
        require_api_key(env_var)
        model = create_model_with_version(agent_config, provider_type, model_name)
        assert model is not None
        assert model.model_config.type == provider_type
        assert model.model_config.model == model_name

    @pytest.mark.parametrize("provider_type,env_var,model_name", REGRESSION_PARAMS)
    def test_generate(self, agent_config, provider_type, env_var, model_name):
        """Basic text generation."""
        require_api_key(env_var)
        model = create_model_with_version(agent_config, provider_type, model_name)
        # Thinking models need more tokens to accommodate reasoning + content
        max_tokens = 1024 if RESTRICTED_MODEL_PARAMS.get(model_name, {}).get("enable_thinking") else 200
        result = model.generate("Say hello in one word", max_tokens=max_tokens)
        logger.info(f"[{model_name}] generate result: {result}")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("provider_type,env_var,model_name", REGRESSION_PARAMS)
    def test_generate_json(self, agent_config, provider_type, env_var, model_name):
        """JSON output generation."""
        require_api_key(env_var)
        model = create_model_with_version(agent_config, provider_type, model_name)
        result = model.generate_with_json_output("Respond with a JSON object containing 'greeting': 'hello'")
        logger.info(f"[{model_name}] generate_json result: {result}")
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.parametrize("provider_type,env_var,model_name", REGRESSION_PARAMS)
    @pytest.mark.asyncio
    async def test_tool_call(self, ssb_agent_config, ssb_tools, provider_type, env_var, model_name):
        """SSB SQLite tool call via generate_with_tools."""
        require_api_key(env_var)
        model = create_model_with_version(ssb_agent_config, provider_type, model_name)
        instructions = (
            "You are a SQLite expert working with the Star Schema Benchmark (SSB) database. "
            "The database contains tables: customer, supplier, part, date, and lineorder."
        )
        question = "database_type='sqlite' task='Count the total number of rows in the customer table'"
        try:
            result = await model.generate_with_tools(
                prompt=question,
                output_type=str,
                tools=ssb_tools,
                instruction=instructions,
                max_turns=5,
            )
            logger.info(f"[{model_name}] tool_call content: {result.get('content', '')}")
            logger.info(f"[{model_name}] tool_call sql_contexts: {result.get('sql_contexts', [])}")
            assert result is not None
            assert "content" in result
            assert "sql_contexts" in result
            assert len(result["sql_contexts"]) > 0, "Should execute at least one SQL query"
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Max turns exceeded: {e}")
            raise

    @pytest.mark.parametrize("provider_type,env_var,model_name", REGRESSION_PARAMS)
    @pytest.mark.asyncio
    async def test_streaming(self, ssb_agent_config, ssb_tools, provider_type, env_var, model_name):
        """Streaming tool call via generate_with_tools_stream."""
        require_api_key(env_var)
        model = create_model_with_version(ssb_agent_config, provider_type, model_name)
        instructions = "You are a SQLite expert working with the Star Schema Benchmark (SSB) database."
        question = "database_type='sqlite' task='Count rows in the customer table'"
        action_history_manager = ActionHistoryManager()
        action_count = 0
        try:
            async for action in model.generate_with_tools_stream(
                prompt=question,
                output_type=str,
                tools=ssb_tools,
                instruction=instructions,
                max_turns=5,
                action_history_manager=action_history_manager,
            ):
                action_count += 1
                logger.info(
                    f"[{model_name}] stream action #{action_count}: "
                    f"type={action.action_type}, role={action.role}, "
                    f"status={action.status}, output={action.output}"
                )
            # Let the event loop process pending cleanup tasks (async generator teardown)
            await asyncio.sleep(0)
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Max turns exceeded: {e}")
            raise
        assert action_count > 0, "Should produce at least 1 streaming action"


# ============================================================
# R-06: Runtime Model Switching
# ============================================================
@pytest.mark.regression
class TestRegressionModelSwitching:
    """Test runtime model switching across different providers."""

    @pytest.mark.asyncio
    async def test_runtime_model_switching(self, ssb_agent_config, ssb_tools):
        """Switch between two different providers, verify both work independently."""
        load_dotenv()

        # Find at least 2 providers with available API keys
        available = []
        for _case_id, spec in PROVIDER_MODELS.items():
            if os.getenv(spec["env_var"]):
                available.append(spec)
            if len(available) >= 2:
                break

        if len(available) < 2:
            pytest.fail("Need at least 2 providers with API keys for model switching test")

        spec_a, spec_b = available[0], available[1]
        model_a = create_model_with_version(ssb_agent_config, spec_a["type"], spec_a["models"][0])
        model_b = create_model_with_version(ssb_agent_config, spec_b["type"], spec_b["models"][0])

        instructions = "You are a SQLite expert working with the Star Schema Benchmark (SSB) database."

        # Model A: first query
        try:
            result_a = await model_a.generate_with_tools(
                prompt="database_type='sqlite' task='Count rows in the customer table'",
                output_type=str,
                tools=ssb_tools,
                instruction=instructions,
                max_turns=5,
            )
            logger.info(f"[Model A: {spec_a['models'][0]}] content: {result_a.get('content', '')}")
            logger.info(f"[Model A: {spec_a['models'][0]}] sql_contexts: {result_a.get('sql_contexts', [])}")
            assert result_a is not None
            assert "content" in result_a
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Model A max turns exceeded: {e}")
            raise

        # Model B: second query (different provider)
        try:
            result_b = await model_b.generate_with_tools(
                prompt="database_type='sqlite' task='Count rows in the supplier table'",
                output_type=str,
                tools=ssb_tools,
                instruction=instructions,
                max_turns=5,
            )
            logger.info(f"[Model B: {spec_b['models'][0]}] content: {result_b.get('content', '')}")
            logger.info(f"[Model B: {spec_b['models'][0]}] sql_contexts: {result_b.get('sql_contexts', [])}")
            assert result_b is not None
            assert "content" in result_b
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"Model B max turns exceeded: {e}")
            raise

        # Both succeeded with different providers
        assert model_a.model_config.type != model_b.model_config.type
        logger.info(f"Model switching OK: {model_a.model_config.type} -> {model_b.model_config.type}")
