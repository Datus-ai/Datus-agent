"""Tests for classify_model_error — the code a failed model call carries to the host."""

import anthropic
import httpx
import litellm
import pytest
from openai import APIConnectionError

from datus.models.model_error import classify_model_error
from datus.utils.exceptions import DatusException, ErrorCode

# Verbatim from a live run: DeepSeek answers an unknown model name with HTTP 400.
DEEPSEEK_UNKNOWN_MODEL = (
    'DeepseekException - {"error":{"message":"The supported API model names are deepseek-flash, '
    'deepseek-v4-pro, but you passed deepseek-v4-pro-0831. (request_id: cd58d435)",'
    '"type":"invalid_request_error","param":null,"code":"invalid_request_error"}}'
)


def _litellm(cls, message):
    return cls(message=message, model="m", llm_provider="deepseek")


def _anthropic(cls, status, message):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message, response=httpx.Response(status, request=request), body=None)


class TestProviderErrors:
    def test_unknown_model_name_on_400_is_model_not_found(self):
        exc = _litellm(litellm.BadRequestError, DEEPSEEK_UNKNOWN_MODEL)
        assert classify_model_error(exc) == ErrorCode.MODEL_NOT_FOUND

    def test_other_400_is_not_mistaken_for_an_unknown_model(self):
        exc = _litellm(litellm.BadRequestError, "Invalid request: messages must not be empty")
        assert classify_model_error(exc) == ErrorCode.MODEL_INVALID_RESPONSE

    def test_rejected_key_is_authentication_error(self):
        exc = _litellm(litellm.AuthenticationError, "Incorrect API key provided: sk-stub")
        assert classify_model_error(exc) == ErrorCode.MODEL_AUTHENTICATION_ERROR

    def test_anthropic_403_is_permission_error(self):
        # Verbatim shape from a live run against a revoked key.
        exc = _anthropic(
            anthropic.PermissionDeniedError,
            403,
            "Error code: 403 - {'error': {'type': 'forbidden', 'message': 'Request not allowed'}}",
        )
        assert classify_model_error(exc) == ErrorCode.MODEL_PERMISSION_ERROR

    def test_404_is_model_not_found(self):
        exc = _litellm(litellm.NotFoundError, "model gpt-9 not found")
        assert classify_model_error(exc) == ErrorCode.MODEL_NOT_FOUND

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Rate limit reached for requests", ErrorCode.MODEL_RATE_LIMIT),
            ("You exceeded your current quota, check your billing", ErrorCode.MODEL_QUOTA_EXCEEDED),
        ],
    )
    def test_429_separates_quota_from_rate_limit(self, message, expected):
        exc = _litellm(litellm.RateLimitError, message)
        assert classify_model_error(exc) == expected

    def test_overloaded_anthropic_is_overloaded(self):
        exc = _anthropic(anthropic.InternalServerError, 529, "Overloaded")
        assert classify_model_error(exc) == ErrorCode.MODEL_OVERLOADED

    def test_connection_failure_without_status_uses_message_rules(self):
        exc = APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com"))
        assert classify_model_error(exc) == ErrorCode.MODEL_CONNECTION_ERROR


class TestNonProviderErrors:
    def test_model_range_datus_exception_keeps_its_code(self):
        exc = DatusException(ErrorCode.MODEL_AUTHENTICATION_ERROR)
        assert classify_model_error(exc) == ErrorCode.MODEL_AUTHENTICATION_ERROR

    def test_other_datus_exception_is_not_labelled(self):
        exc = DatusException(ErrorCode.COMMON_CONFIG_ERROR, message_args={"config_error": "bad template"})
        assert classify_model_error(exc) is None

    def test_plain_exception_is_not_labelled(self):
        assert classify_model_error(RuntimeError("tool crashed: model not found in cache")) is None
