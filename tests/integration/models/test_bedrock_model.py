"""Manual Amazon Bedrock Converse certification matrix.

These tests make billable AWS calls. They are deliberately excluded from
nightly/provider-health selection and run only when explicitly requested.
"""

import asyncio
import os

import boto3
import pytest
from agents.tracing import DefaultTraceProvider, get_trace_provider, set_trace_provider

from datus.configuration.agent_config import ModelConfig
from datus.models.bedrock_model import BedrockModel
from datus.schemas.action_history import ActionHistoryManager
from datus.tools.func_tool import db_function_tools
from tests.conftest import load_acceptance_config

_CERTIFICATION_ENABLED = os.getenv("DATUS_BEDROCK_CERTIFY") == "1"
_OPTIONAL_MODELS_ENABLED = os.getenv("DATUS_BEDROCK_CERTIFY_OPTIONAL") == "1"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.bedrock_certification,
    pytest.mark.skipif(
        not _CERTIFICATION_ENABLED,
        reason="set DATUS_BEDROCK_CERTIFY=1 to run billable AWS Bedrock certification",
    ),
]

BEDROCK_MODELS = [
    pytest.param("us.anthropic.claude-sonnet-5", id="claude-sonnet-5"),
    pytest.param("us.amazon.nova-2-lite-v1:0", id="nova-2-lite"),
    pytest.param(
        "openai.gpt-oss-20b-1:0",
        id="gpt-oss-20b",
        marks=pytest.mark.skipif(
            not _OPTIONAL_MODELS_ENABLED,
            reason="set DATUS_BEDROCK_CERTIFY_OPTIONAL=1 to certify optional models",
        ),
    ),
    pytest.param(
        "deepseek.v3.2",
        id="deepseek-v3.2",
        marks=pytest.mark.skipif(
            not _OPTIONAL_MODELS_ENABLED,
            reason="set DATUS_BEDROCK_CERTIFY_OPTIONAL=1 to certify optional models",
        ),
    ),
    pytest.param(
        "google.gemma-3-12b-it",
        id="gemma-3-12b-it",
        marks=pytest.mark.skipif(
            not _OPTIONAL_MODELS_ENABLED,
            reason="set DATUS_BEDROCK_CERTIFY_OPTIONAL=1 to certify optional models",
        ),
    ),
]
BEDROCK_TOOL_MODELS = [
    pytest.param("us.anthropic.claude-sonnet-5", id="claude-sonnet-5"),
    pytest.param("us.amazon.nova-2-lite-v1:0", id="nova-2-lite"),
    pytest.param(
        "openai.gpt-oss-20b-1:0",
        id="gpt-oss-20b",
        marks=pytest.mark.skipif(
            not _OPTIONAL_MODELS_ENABLED,
            reason="set DATUS_BEDROCK_CERTIFY_OPTIONAL=1 to certify optional models",
        ),
    ),
    pytest.param(
        "deepseek.v3.2",
        id="deepseek-v3.2",
        marks=[
            pytest.mark.skipif(
                not _OPTIONAL_MODELS_ENABLED,
                reason="set DATUS_BEDROCK_CERTIFY_OPTIONAL=1 to certify optional models",
            ),
            pytest.mark.xfail(
                strict=True,
                reason="Bedrock DeepSeek v3.2 does not emit Converse tool calls",
            ),
        ],
    ),
    pytest.param(
        "google.gemma-3-12b-it",
        id="gemma-3-12b-it",
        marks=[
            pytest.mark.skipif(
                not _OPTIONAL_MODELS_ENABLED,
                reason="set DATUS_BEDROCK_CERTIFY_OPTIONAL=1 to certify optional models",
            ),
            pytest.mark.xfail(
                strict=True,
                reason="Bedrock Gemma 3 12B IT does not emit Converse tool calls",
            ),
        ],
    ),
]


@pytest.fixture(scope="module", autouse=True)
def isolated_tracing_provider():
    """Disable tracing only while the opt-in certification module executes."""
    previous_provider = get_trace_provider()
    provider = DefaultTraceProvider()
    provider.set_disabled(True)
    set_trace_provider(provider)
    try:
        yield
    finally:
        set_trace_provider(previous_provider)


@pytest.fixture(scope="module")
def aws_provider_options() -> dict[str, str]:
    """Resolve the same local AWS credential chain used by BedrockModel."""
    profile = os.getenv("AWS_PROFILE")
    session = boto3.Session(profile_name=profile or None)
    if session.get_credentials() is None:
        pytest.skip("AWS credentials are not available")

    region = (
        os.getenv("AWS_REGION_NAME")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or session.region_name
    )
    if not region:
        pytest.skip("AWS region is not configured")

    options = {"aws_region_name": region}
    if profile:
        options["aws_profile_name"] = profile
    return options


@pytest.fixture(scope="module")
def ssb_tools():
    config = load_acceptance_config(datasource="ssb_sqlite")
    return db_function_tools(config)


def _create_model(model_id: str, provider_options: dict[str, str]) -> BedrockModel:
    config = ModelConfig(
        type="bedrock",
        auth_type="aws",
        api_key="",
        model=model_id,
        provider_options=provider_options,
    )
    return BedrockModel(model_config=config)


@pytest.mark.parametrize("model_id", BEDROCK_MODELS)
def test_generate(model_id, aws_provider_options):
    model = _create_model(model_id, aws_provider_options)
    result = model.generate("Reply with exactly: bedrock-ok", max_tokens=200)

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.parametrize("model_id", BEDROCK_MODELS)
def test_generate_json(model_id, aws_provider_options):
    model = _create_model(model_id, aws_provider_options)
    result = model.generate_with_json_output(
        'Return only this JSON object exactly: {"status": "ok"}',
        max_tokens=300,
    )

    assert isinstance(result, dict)
    assert result.get("status") == "ok"


@pytest.mark.parametrize("model_id", BEDROCK_TOOL_MODELS)
@pytest.mark.asyncio
async def test_tool_call(model_id, aws_provider_options, ssb_tools):
    model = _create_model(model_id, aws_provider_options)
    result = await model.generate_with_tools(
        prompt=(
            "database_type='sqlite' task='You must call execute_sql with "
            "SELECT c_name FROM customer WHERE c_custkey = 1 and return the exact value. "
            "Do not answer from memory or calculate it yourself.'"
        ),
        output_type=str,
        tools=ssb_tools,
        instruction=(
            "You are a SQLite expert working with the Star Schema Benchmark database. "
            "Use the database tools to answer the question."
        ),
        max_turns=5,
    )

    assert isinstance(result, dict)
    assert result.get("content")
    assert result.get("sql_contexts"), "the model must execute at least one SQL query"


@pytest.mark.parametrize("model_id", BEDROCK_TOOL_MODELS)
@pytest.mark.asyncio
async def test_tool_call_stream(model_id, aws_provider_options, ssb_tools):
    model = _create_model(model_id, aws_provider_options)
    action_history_manager = ActionHistoryManager()
    action_count = 0

    async for action in model.generate_with_tools_stream(
        prompt=(
            "database_type='sqlite' task='You must call execute_sql with "
            "SELECT s_name FROM supplier WHERE s_suppkey = 1 and return the exact value. "
            "Do not answer from memory or calculate it yourself.'"
        ),
        output_type=str,
        tools=ssb_tools,
        instruction=(
            "You are a SQLite expert working with the Star Schema Benchmark database. "
            "Use the database tools to answer the question."
        ),
        max_turns=5,
        action_history_manager=action_history_manager,
    ):
        assert action is not None
        action_count += 1

    await asyncio.sleep(0)
    assert action_count > 0
    assert any(action.action_type == "execute_sql" for action in action_history_manager.get_actions())
