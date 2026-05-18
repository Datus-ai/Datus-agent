# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for ModelBehaviorError retry logic in OpenAICompatibleModel.

Covers:
- _with_retry_async: ModelBehaviorError retry + exhaust
- generate_with_tools: temperature/top_p propagation from model_config

CI level: zero external deps, mock all SDK interactions.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.exceptions import ModelBehaviorError
from openai import APIError

from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.exceptions import DatusException


@pytest.fixture
def mock_model():
    """Create a minimal OpenAICompatibleModel with mocked internals."""
    model_config = MagicMock()
    model_config.model = "test-model"
    model_config.type = "openai"
    model_config.api_key = "test-key"
    model_config.base_url = "https://api.test.com/v1"
    model_config.temperature = None
    model_config.top_p = None
    model_config.max_tokens = 1000
    model_config.extra_headers = None

    with patch.object(OpenAICompatibleModel, "__init__", lambda self, **kwargs: None):
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.model_config = model_config
        model.model_name = "test-model"
    return model


class TestWithRetryAsyncModelBehaviorError:
    """Tests for _with_retry_async handling of ModelBehaviorError."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_model_behavior_error(self, mock_model):
        """ModelBehaviorError on first attempt, success on second."""
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ModelBehaviorError("Tool not found: hallucinated_tool")
            return "success"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mock_model._with_retry_async(
                flaky_operation,
                operation_name="test_op",
                max_retries=3,
                base_delay=0.01,
            )

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self, mock_model):
        """ModelBehaviorError raised after all retries exhausted."""

        async def always_fails():
            raise ModelBehaviorError("Persistent hallucination")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ModelBehaviorError, match="Persistent hallucination"):
                await mock_model._with_retry_async(
                    always_fails,
                    operation_name="test_op",
                    max_retries=2,
                    base_delay=0.01,
                )

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, mock_model):
        """Verify exponential backoff delays are applied."""
        call_count = 0
        sleep_delays = []

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ModelBehaviorError("error")

        async def mock_sleep(delay):
            sleep_delays.append(delay)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(ModelBehaviorError):
                await mock_model._with_retry_async(
                    always_fails,
                    operation_name="test_op",
                    max_retries=3,
                    base_delay=1.0,
                )

        # 3 retries = 3 sleep calls with exponential backoff: 1, 2, 4
        assert len(sleep_delays) == 3
        assert sleep_delays[0] == 1.0
        assert sleep_delays[1] == 2.0
        assert sleep_delays[2] == 4.0


class _FakeAPIError(APIError):
    """A raisable :class:`APIError` whose ``str()`` is the upstream text.

    ``MagicMock(spec=APIError)`` works for ``classify_openai_compatible_error``
    (which only inspects ``str(e)``) but Python's ``raise`` machinery
    refuses it ("exceptions must derive from BaseException"). A real
    subclass keeps both the isinstance check in ``_with_retry`` and
    Python's raise contract happy.
    """

    def __init__(self, message: str, status_code: int = 400):
        # Skip ``APIError.__init__`` — it requires an httpx.Request we don't
        # need for the test. Set the attributes the wrapper reads directly.
        Exception.__init__(self, message)
        self._message = message
        self.status_code = status_code

    def __str__(self) -> str:
        return self._message


def _make_api_error(message: str) -> APIError:
    """Build a fake non-retryable APIError carrying the upstream message."""
    return _FakeAPIError(message)


class TestRetryErrorLoggingIncludesUpstream:
    """Regression guard for the post-mortem on the Anthropic
    ``temperature and top_p cannot both be specified`` 400.

    The wrapper's ``error_code.code`` / ``error_code.desc`` rolls every
    HTTP 400 into the generic ``300002 / Invalid request format,
    content, or model response``, which gives operators nothing to
    grep on. The upstream message (``str(e)``) carries the actual
    cause and must appear in the same log line so a single CloudWatch
    entry resolves the incident.
    """

    def test_sync_retry_final_error_log_includes_upstream_message(self, mock_model, caplog):
        upstream = "`temperature` and `top_p` cannot both be specified for this model."

        def always_fails():
            raise _make_api_error(upstream)

        with patch("time.sleep"), caplog.at_level(logging.ERROR, logger="datus.models.openai_compatible"):
            with pytest.raises(DatusException):
                mock_model._with_retry(always_fails, operation_name="text generation", max_retries=0)

        # The wrapper classification (whatever code it lands on) AND the
        # raw upstream cause must both appear in the final error log.
        # The hardening contract is "operators get the actual upstream
        # text on a single line"; the specific wrapper code is determined
        # by classify_openai_compatible_error and tested elsewhere.
        final_error_records = [r for r in caplog.records if "after" in r.getMessage() and "attempts" in r.getMessage()]
        assert final_error_records, "expected an 'after N attempts' error log line"
        msg = final_error_records[-1].getMessage()
        assert "text generation" in msg
        assert upstream in msg, f"upstream cause must be in the final error log; got: {msg!r}"

    @pytest.mark.asyncio
    async def test_async_retry_final_error_log_includes_upstream_message(self, mock_model, caplog):
        upstream = "`temperature` and `top_p` cannot both be specified for this model."

        async def always_fails():
            raise _make_api_error(upstream)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            caplog.at_level(logging.ERROR, logger="datus.models.openai_compatible"),
        ):
            with pytest.raises(DatusException):
                await mock_model._with_retry_async(always_fails, operation_name="text generation", max_retries=0)

        final_error_records = [r for r in caplog.records if "after" in r.getMessage() and "attempts" in r.getMessage()]
        assert final_error_records, "expected an 'after N attempts' error log line"
        msg = final_error_records[-1].getMessage()
        assert "text generation" in msg
        assert upstream in msg
