# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/api/routes/chat_routes.py — submit_user_interaction endpoint."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from datus.api.models.cli_models import SSEEndData, SSEEvent, SSESessionData, StreamChatInput, UserInteractionInput
from datus.api.routes.chat_routes import (
    _is_valid_subagent_id,
    stream_chat,
    submit_user_interaction,
)
from datus.tools.data_access_policy import DataAccessConfig


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
            await stream_chat(request, svc, ctx, MagicMock())

        assert exc_info.value.status_code == 404
        assert "nonexistent_xyz" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_none_subagent_bypasses_gate(self):
        """Without a subagent_id the 404 gate is skipped — default routing handles it."""
        svc = _mock_svc_with_nodes()
        svc.chat.stream_chat = MagicMock(return_value=AsyncMock().__aiter__())
        ctx = MagicMock(user_id="u1")
        request = StreamChatInput(message="hi", subagent_id=None)

        response = await stream_chat(request, svc, ctx, MagicMock())

        assert isinstance(response, StreamingResponse)
        assert response.status_code == 200
        assert response.media_type == "text/event-stream"


class TestStreamChatDataAccessPreCheck:
    """Data-access enabled chat requests must carry required request principal fields."""

    @pytest.mark.asyncio
    async def test_enabled_data_access_without_principal_returns_sse_error(self):
        svc = _mock_svc_with_nodes()
        svc.agent_config.data_access_config = DataAccessConfig.from_dict(
            {
                "enabled": True,
                "provider": "x:Y",
                "policies": [{"condition": {"value_from": "principal.market_code"}}],
            }
        )
        svc.chat.stream_chat = MagicMock(side_effect=AssertionError("upstream invoked"))
        ctx = MagicMock(user_id=None)
        ctx.principal = {}
        request = StreamChatInput(message="hi")

        response = await stream_chat(request, svc, ctx, MagicMock())

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        assert len(chunks) == 1
        assert "event: error" in chunks[0]
        payload = json.loads(
            next(line for line in chunks[0].splitlines() if line.startswith("data: "))[len("data: ") :]
        )
        assert payload["error_type"] == "DATA_ACCESS_PRINCIPAL_REQUIRED"
        assert "principal.market_code" in payload["error"]
        assert "provider that populates principal fields" in payload["error"]
        assert "agent.data_access policies" in payload["error"]
        svc.chat.stream_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_data_access_with_required_principal_allows_service_call(self):
        async def empty_stream(*_args, **_kwargs):
            if False:
                yield

        svc = _mock_svc_with_nodes()
        svc.agent_config.data_access_config = DataAccessConfig.from_dict(
            {
                "enabled": True,
                "provider": "x:Y",
                "policies": [{"condition": {"value_from": "principal.market_code"}}],
            }
        )
        svc.chat.stream_chat = MagicMock(return_value=empty_stream())
        ctx = MagicMock(user_id=None)
        ctx.principal = {"market_code": "MKT300"}
        request = StreamChatInput(message="hi")

        response = await stream_chat(request, svc, ctx, MagicMock())
        async for _ in response.body_iterator:
            pass

        svc.chat.stream_chat.assert_called_once()
        assert svc.chat.stream_chat.call_args.kwargs["principal"] == {"market_code": "MKT300"}

    @pytest.mark.asyncio
    async def test_enabled_data_access_without_principal_paths_allows_service_call(self):
        async def empty_stream(*_args, **_kwargs):
            if False:
                yield

        svc = _mock_svc_with_nodes()
        svc.agent_config.data_access_config = DataAccessConfig.from_dict(
            {
                "enabled": True,
                "provider": "x:Y",
                "policies": [{"name": "static_policy", "condition": {"value_from": "literal.MKT300"}}],
            }
        )
        svc.chat.stream_chat = MagicMock(return_value=empty_stream())
        ctx = MagicMock(user_id=None)
        ctx.principal = {}
        request = StreamChatInput(message="hi")

        response = await stream_chat(request, svc, ctx, MagicMock())
        async for _ in response.body_iterator:
            pass

        svc.chat.stream_chat.assert_called_once()
        assert svc.chat.stream_chat.call_args.kwargs["principal"] == {}

    @pytest.mark.asyncio
    async def test_user_id_only_does_not_satisfy_required_business_principal(self):
        svc = _mock_svc_with_nodes()
        svc.agent_config.data_access_config = DataAccessConfig.from_dict(
            {
                "enabled": True,
                "provider": "x:Y",
                "policies": [{"condition": {"value_from": "principal.market_code"}}],
            }
        )
        svc.chat.stream_chat = MagicMock(side_effect=AssertionError("upstream invoked"))
        ctx = MagicMock(user_id="alice")
        ctx.principal = {}
        request = StreamChatInput(message="hi")

        response = await stream_chat(request, svc, ctx, MagicMock())

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        payload = json.loads(
            next(line for line in chunks[0].splitlines() if line.startswith("data: "))[len("data: ") :]
        )
        assert payload["error_type"] == "DATA_ACCESS_PRINCIPAL_REQUIRED"
        assert "principal.market_code" in payload["error"]
        svc.chat.stream_chat.assert_not_called()


@pytest.mark.acceptance
class TestChatRouteAcceptance:
    """Deterministic Chat API entrance coverage for built-in subagent routing."""

    @pytest.mark.asyncio
    async def test_stream_chat_routes_gen_sql_to_service_sse(self):
        async def fake_chat_stream(*args, **kwargs):
            yield SSEEvent(id=1, event="session", data=SSESessionData(session_id="gen_sql_acceptance"))
            yield SSEEvent(
                id=2,
                event="end",
                data=SSEEndData(
                    session_id="gen_sql_acceptance",
                    total_events=2,
                    action_count=1,
                    duration=0.01,
                ),
            )

        svc = _mock_svc_with_nodes()
        svc.chat.stream_chat = MagicMock(return_value=fake_chat_stream())
        ctx = MagicMock(user_id="user-1")
        request = StreamChatInput(
            message="Generate SQL for school count",
            session_id="gen_sql_acceptance",
            subagent_id="gen_sql",
        )

        response = await stream_chat(request, svc, ctx, MagicMock())
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        body = "".join(chunks)
        assert response.media_type == "text/event-stream"
        assert "event: session" in body
        assert "event: end" in body
        svc.chat.stream_chat.assert_called_once()
        call_args = svc.chat.stream_chat.call_args
        assert call_args.args[0] is request
        assert call_args.kwargs["sub_agent_id"] == "gen_sql"
        assert call_args.kwargs["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_valid_builtin_passes_gate(self):
        svc = _mock_svc_with_nodes()
        svc.chat.stream_chat = MagicMock(return_value=AsyncMock().__aiter__())
        ctx = MagicMock(user_id="u1")
        request = StreamChatInput(message="hi", subagent_id="gen_sql")

        response = await stream_chat(request, svc, ctx, MagicMock())

        assert isinstance(response, StreamingResponse)
        assert response.status_code == 200
        assert response.media_type == "text/event-stream"


# ===========================================================================
# /api/v1/chat/insert — mid-run user-text injection
# ===========================================================================


class TestInsertMessageEndpoint:
    """Behavioural contract for the ``/insert`` endpoint.

    The endpoint is the API equivalent of typing in the TUI while the
    agent is streaming: it appends a free-text user message to the
    pending-input queue carried by the running node, so the model sees
    it before its next LLM turn.
    """

    @staticmethod
    def _make_request(message: str = "describe customers", session_id: str = "s1"):
        from datus.api.models.chat_models import InsertMessageInput

        return InsertMessageInput(session_id=session_id, message=message)

    @staticmethod
    def _make_task_with_queue(queue=None):
        from datus.cli.execution_state import PendingInputQueue

        task = MagicMock()
        task.node.pending_input_queue = queue if queue is not None else PendingInputQueue()
        return task

    @pytest.mark.asyncio
    async def test_push_success_returns_queued_count(self):
        from datus.api.routes.chat_routes import insert_message

        task = self._make_task_with_queue()
        svc = _mock_svc(task=task)

        result = await insert_message(self._make_request("hello"), svc)

        assert result.success is True
        assert result.data.session_id == "s1"
        assert result.data.queued_count == 1
        assert task.node.pending_input_queue.snapshot() == ["hello"]

    @pytest.mark.asyncio
    async def test_multiple_pushes_accumulate(self):
        from datus.api.routes.chat_routes import insert_message

        task = self._make_task_with_queue()
        svc = _mock_svc(task=task)

        await insert_message(self._make_request("first"), svc)
        result = await insert_message(self._make_request("second"), svc)

        assert result.success is True
        assert result.data.queued_count == 2
        assert task.node.pending_input_queue.snapshot() == ["first", "second"]

    @pytest.mark.asyncio
    async def test_missing_task_returns_session_not_running(self):
        from datus.api.routes.chat_routes import insert_message

        svc = _mock_svc(task=None)

        result = await insert_message(self._make_request("anything"), svc)

        assert result.success is False
        assert result.errorCode == "SESSION_NOT_RUNNING"

    @pytest.mark.asyncio
    async def test_task_without_node_returns_session_not_running(self):
        from datus.api.routes.chat_routes import insert_message

        task = MagicMock()
        task.node = None
        svc = _mock_svc(task=task)

        result = await insert_message(self._make_request("anything"), svc)

        assert result.success is False
        assert result.errorCode == "SESSION_NOT_RUNNING"

    @pytest.mark.asyncio
    async def test_whitespace_only_message_returns_invalid_input(self):
        from datus.api.routes.chat_routes import insert_message

        task = self._make_task_with_queue()
        svc = _mock_svc(task=task)

        result = await insert_message(self._make_request("   \t  "), svc)

        assert result.success is False
        assert result.errorCode == "INVALID_INPUT"
        # Queue untouched.
        assert len(task.node.pending_input_queue) == 0

    @pytest.mark.asyncio
    async def test_node_without_queue_returns_queue_unavailable(self):
        """Node exists but caller never initialised pending_input_queue —
        the route must surface that as a distinct error rather than
        silently pushing nowhere."""
        from datus.api.routes.chat_routes import insert_message

        task = MagicMock()
        task.node.pending_input_queue = None
        svc = _mock_svc(task=task)

        result = await insert_message(self._make_request("hello"), svc)

        assert result.success is False
        assert result.errorCode == "QUEUE_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_message_is_stripped_before_push(self):
        from datus.api.routes.chat_routes import insert_message

        task = self._make_task_with_queue()
        svc = _mock_svc(task=task)

        await insert_message(self._make_request("  padded  "), svc)

        # Stripped form lands in the queue.
        assert task.node.pending_input_queue.snapshot() == ["padded"]
