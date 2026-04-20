# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/api/routes/success_story_routes.py."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from datus.api.models.success_story_models import SuccessStoryData, SuccessStoryInput
from datus.api.routes.success_story_routes import save_success_story
from datus.api.services.success_story_service import SubagentNotFoundError


def _mock_svc(**service_attrs):
    svc = MagicMock()
    svc.success_story = MagicMock(**service_attrs)
    return svc


@pytest.mark.asyncio
async def test_save_returns_result_success():
    data = SuccessStoryData(
        csv_path="/tmp/benchmark/gen_sql/success_story.csv",
        subagent_name="gen_sql",
        session_id="s1",
        session_link=None,
        timestamp="2026-04-20 00:00:00",
    )
    svc = _mock_svc(save=MagicMock(return_value=data))
    payload = SuccessStoryInput(session_id="s1", sql="SELECT 1", user_message="q", subagent_id="gen_sql")

    result = await save_success_story(payload, svc)

    assert result.success is True
    assert result.data == data
    svc.success_story.save.assert_called_once_with(payload)


@pytest.mark.asyncio
async def test_save_translates_subagent_not_found_to_400():
    svc = _mock_svc(save=MagicMock(side_effect=SubagentNotFoundError("Subagent 'nope' not found")))
    payload = SuccessStoryInput(session_id="s1", sql="SELECT 1", user_message="q", subagent_id="nope")

    with pytest.raises(HTTPException) as exc:
        await save_success_story(payload, svc)

    assert exc.value.status_code == 400
    assert "nope" in exc.value.detail


@pytest.mark.asyncio
async def test_save_returns_failure_result_on_os_error():
    svc = _mock_svc(save=MagicMock(side_effect=OSError("disk full")))
    payload = SuccessStoryInput(session_id="s1", sql="SELECT 1", user_message="q")

    result = await save_success_story(payload, svc)

    assert result.success is False
    assert result.errorCode == "SUCCESS_STORY_WRITE_FAILED"
    assert "disk full" in result.errorMessage
