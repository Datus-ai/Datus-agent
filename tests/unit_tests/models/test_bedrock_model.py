# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deterministic tests for the Amazon Bedrock Converse provider."""

from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import ModelConfig
from datus.models.bedrock_model import BedrockModel

pytestmark = pytest.mark.ci


def _config(model: str = "us.anthropic.claude-sonnet-5", **kwargs) -> ModelConfig:
    provider_options = kwargs.pop("provider_options", {"aws_region_name": "us-east-1"})
    return ModelConfig(
        type="bedrock",
        api_key="",
        model=model,
        auth_type="aws",
        provider_options=provider_options,
        **kwargs,
    )


def _response(content: str = "ok") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.reasoning_content = None
    response.choices[0].finish_reason = "stop"
    response.model = "bedrock-model"
    response.usage.prompt_tokens = 2
    response.usage.completion_tokens = 1
    response.usage.total_tokens = 3
    return response


class TestBedrockModel:
    def test_initializes_without_api_key(self):
        model = BedrockModel(_config())
        assert model.api_key == ""
        assert model.base_url is None
        assert model.litellm_adapter.provider == "bedrock"

    def test_uses_converse_route(self):
        model = BedrockModel(_config())
        assert model.litellm_adapter.litellm_model_name == ("bedrock/converse/us.anthropic.claude-sonnet-5")

    def test_generate_forwards_aws_options_without_sampling_defaults(self):
        model = BedrockModel(
            _config(
                provider_options={
                    "aws_region_name": "us-east-1",
                    "aws_profile_name": "dev",
                }
            )
        )
        with patch("datus.models.openai_compatible.litellm.completion", return_value=_response()) as completion:
            assert model.generate("hello", max_tokens=32) == "ok"

        kwargs = completion.call_args.kwargs
        assert kwargs["model"] == "bedrock/converse/us.anthropic.claude-sonnet-5"
        assert kwargs["aws_region_name"] == "us-east-1"
        assert kwargs["aws_profile_name"] == "dev"
        assert kwargs["max_tokens"] == 32
        assert "api_key" not in kwargs
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs

    def test_explicit_sampling_parameter_is_preserved(self):
        model = BedrockModel(_config())
        with patch("datus.models.openai_compatible.litellm.completion", return_value=_response()) as completion:
            model.generate("hello", temperature=0.2)
        assert completion.call_args.kwargs["temperature"] == 0.2

    def test_json_output_uses_prompt_contract_without_openai_response_format(self):
        model = BedrockModel(_config())
        with patch.object(model, "generate", return_value='{"status": "ok"}') as generate:
            assert model.generate_with_json_output("return JSON") == {"status": "ok"}
        assert "response_format" not in generate.call_args.kwargs

    def test_agent_model_settings_receive_aws_options(self):
        model = BedrockModel(_config())
        agent = model._build_agent(
            instruction="Use tools when needed.",
            output_type=str,
            strict_json_schema=True,
            connected_servers={},
            tools=[],
        )
        assert agent.model_settings.extra_args["aws_region_name"] == "us-east-1"

    def test_rejects_unapproved_provider_option(self):
        config = _config()
        config.provider_options["aws_secret_access_key"] = "must-not-be-stored"
        with pytest.raises(ValueError, match="Unsupported Bedrock provider_options"):
            BedrockModel(config)

    def test_resolves_region_from_boto3_profile(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        config = _config()
        config.provider_options = {"aws_profile_name": "dev"}
        session = MagicMock(region_name="us-west-2")
        with patch("boto3.Session", return_value=session) as session_cls:
            model = BedrockModel(config)
        session_cls.assert_called_once_with(profile_name="dev")
        assert model.provider_request_options["aws_region_name"] == "us-west-2"


def test_model_type_map_registers_bedrock():
    from datus.models.base import LLMBaseModel

    assert LLMBaseModel.MODEL_TYPE_MAP["bedrock"] == "BedrockModel"
