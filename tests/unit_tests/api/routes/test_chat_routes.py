# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/api/routes/chat_routes.py — submit_user_interaction endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from datus.api.models.cli_models import StreamChatInput, UserInteractionInput
from datus.api.routes.chat_routes import (
    _is_valid_subagent_id,
    stream_chat,
    submit_user_interaction,
)


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
    """``submit_user_interaction`` forwards ``List[List[str]]`` to the broker unchanged.

    ``InteractionBroker.submit`` takes ``List[List[str]]`` directly — one inner
    list per question — so the route handler just passes ``request.input``
    through without reshaping it.
    """

    @pytest.mark.asyncio
    async def test_single_question_single_select(self):
        """input=[['2']] is forwarded to the broker verbatim."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["2"]])

        result = await submit_user_interaction(request, svc)

        task.node.interaction_broker.submit.assert_called_once_with("k1", [["2"]])
        assert result.success is True

    @pytest.mark.asyncio
    async def test_single_question_multi_select(self):
        """input=[['1','3']] is forwarded to the broker verbatim."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["1", "3"]])

        result = await submit_user_interaction(request, svc)

        task.node.interaction_broker.submit.assert_called_once_with("k1", [["1", "3"]])
        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_mixed(self):
        """input=[['2'], ['1','3']] is forwarded to the broker verbatim."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["2"], ["1", "3"]])

        result = await submit_user_interaction(request, svc)

        task.node.interaction_broker.submit.assert_called_once_with("k1", [["2"], ["1", "3"]])
        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_all_single_select(self):
        """input=[['a'], ['b']] is forwarded to the broker verbatim."""
        task = _mock_task()
        svc = _mock_svc(task=task)
        request = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["a"], ["b"]])

        await submit_user_interaction(request, svc)

        task.node.interaction_broker.submit.assert_called_once_with("k1", [["a"], ["b"]])

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


def _mock_svc_with_nodes(agentic_nodes=None):
    svc = MagicMock()
    svc.agent_config.agentic_nodes = agentic_nodes or {}
    return svc


class TestIsValidSubagentId:
    """Tests for the _is_valid_subagent_id helper used by stream_chat's 404 gate."""

    def test_builtin_subagent(self):
        svc = _mock_svc_with_nodes()
        assert _is_valid_subagent_id(svc, "gen_sql") is True

    def test_extra_builtin_feedback(self):
        """feedback is dispatched by _create_node but not in BUILTIN_SUBAGENTS."""
        svc = _mock_svc_with_nodes()
        assert _is_valid_subagent_id(svc, "feedback") is True

    def test_custom_node_by_name(self):
        svc = _mock_svc_with_nodes({"my_custom_agent": {"id": "uuid-1", "model": "deepseek"}})
        assert _is_valid_subagent_id(svc, "my_custom_agent") is True

    def test_custom_node_by_uuid(self):
        """Custom sub-agents may be looked up by the original UUID stored under 'id'."""
        svc = _mock_svc_with_nodes({"my_custom_agent": {"id": "uuid-abc", "model": "deepseek"}})
        assert _is_valid_subagent_id(svc, "uuid-abc") is True

    def test_unknown_id_returns_false(self):
        svc = _mock_svc_with_nodes({"existing_agent": {"id": "uuid-1"}})
        assert _is_valid_subagent_id(svc, "nonexistent_xyz") is False

    def test_agentic_nodes_missing_attribute(self):
        """Missing ``agent_config.agentic_nodes`` falls through gracefully."""
        svc = MagicMock()
        svc.agent_config = MagicMock(spec=[])  # no agentic_nodes attribute
        assert _is_valid_subagent_id(svc, "nonexistent") is False

    def test_non_dict_node_entry_is_skipped(self):
        """Non-dict entries in ``agentic_nodes`` are ignored during UUID lookup."""
        svc = _mock_svc_with_nodes({"some_agent": "not_a_dict_value"})
        assert _is_valid_subagent_id(svc, "not_a_dict_value") is False


class TestStreamChat404Gate:
    """Tests for the stream_chat 404 gate on invalid subagent_id."""

    @pytest.mark.asyncio
    async def test_invalid_subagent_raises_404(self):
        svc = _mock_svc_with_nodes()
        ctx = MagicMock()
        request = StreamChatInput(message="hi", subagent_id="nonexistent_xyz")

        with pytest.raises(HTTPException) as exc_info:
            await stream_chat(request, svc, ctx)

        assert exc_info.value.status_code == 404
        assert "nonexistent_xyz" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_none_subagent_bypasses_gate(self):
        """Without a subagent_id the 404 gate is skipped — default routing handles it."""
        svc = _mock_svc_with_nodes()
        svc.chat.stream_chat = MagicMock(return_value=AsyncMock().__aiter__())
        ctx = MagicMock(user_id="u1")
        request = StreamChatInput(message="hi", subagent_id=None)

        # Should not raise — returns a StreamingResponse.
        response = await stream_chat(request, svc, ctx)
        assert response is not None

    @pytest.mark.asyncio
    async def test_valid_builtin_passes_gate(self):
        svc = _mock_svc_with_nodes()
        svc.chat.stream_chat = MagicMock(return_value=AsyncMock().__aiter__())
        ctx = MagicMock(user_id="u1")
        request = StreamChatInput(message="hi", subagent_id="gen_sql")

        response = await stream_chat(request, svc, ctx)
        assert response is not None
