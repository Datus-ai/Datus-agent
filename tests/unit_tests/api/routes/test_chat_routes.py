# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/api/routes/chat_routes.py — submit_user_interaction endpoint."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.api.models.cli_models import UserInteractionInput
from datus.api.routes.chat_routes import submit_user_interaction


def _mock_svc(task=None):
    """Build a mock DatusService with task_manager."""
    svc = MagicMock()
    svc.task_manager.get_task.return_value = task
    return svc


def _mock_task(broker_submit_return=True):
    """Build a mock task with node and interaction_broker."""
    task = MagicMock()
    task.node.interaction_broker = AsyncMock()
    task.node.interaction_broker.submit = AsyncMock(return_value=broker_submit_return)
    return task


class TestSubmitUserInteractionConversion:
    """Tests for the List[List[str]] → broker format conversion."""

    @pytest.mark.asyncio
    async def test_single_question_single_select(self):
        """input=[['2']] → broker receives '2' (plain string)."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["2"]])

        result = await submit_user_interaction(request, svc)

        task.node.interaction_broker.submit.assert_called_once_with("k1", "2")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_single_question_multi_select(self):
        """input=[['1','3']] → broker receives '["1", "3"]' (JSON array string)."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["1", "3"]])

        result = await submit_user_interaction(request, svc)

        submitted = task.node.interaction_broker.submit.call_args[0][1]
        assert json.loads(submitted) == ["1", "3"]
        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_mixed(self):
        """input=[['2'], ['1','3']] → broker receives '["2", ["1", "3"]]'."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["2"], ["1", "3"]])

        result = await submit_user_interaction(request, svc)

        submitted = task.node.interaction_broker.submit.call_args[0][1]
        assert json.loads(submitted) == ["2", ["1", "3"]]
        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_all_single_select(self):
        """input=[['a'], ['b']] → broker receives '["a", "b"]'."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["a"], ["b"]])

        await submit_user_interaction(request, svc)

        submitted = task.node.interaction_broker.submit.call_args[0][1]
        assert json.loads(submitted) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        """Returns error when task is not found."""
        svc = _mock_svc(task=None)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["1"]])

        result = await submit_user_interaction(request, svc)

        assert result.success is False
        assert result.errorCode == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_broker_not_found(self):
        """Returns error when broker is None."""
        task = MagicMock()
        task.node.interaction_broker = None
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["1"]])

        result = await submit_user_interaction(request, svc)

        assert result.success is False
        assert result.errorCode == "BROKER_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_broker_submit_failure(self):
        """Returns success=False when broker.submit returns False."""
        task = _mock_task(broker_submit_return=False)
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["1"]])

        result = await submit_user_interaction(request, svc)

        assert result.success is False
