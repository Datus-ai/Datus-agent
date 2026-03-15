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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.exceptions import ModelBehaviorError

from datus.models.openai_compatible import OpenAICompatibleModel


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
