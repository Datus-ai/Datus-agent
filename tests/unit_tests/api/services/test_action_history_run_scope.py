# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Exercise run-scoped results through the API's continuation loop."""

from unittest.mock import MagicMock

import pytest

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.api.models.cli_models import StreamChatInput
from datus.api.services.chat_task_manager import ChatTask, ChatTaskManager
from datus.configuration.node_type import NodeType
from tests.unit_tests.mock_llm_model import build_simple_response


@pytest.mark.asyncio
async def test_continuation_keeps_both_replies_and_separate_result_traces(real_agent_config, mock_llm_create):
    managers = []
    results = []

    class _ContinuedChat(ChatAgenticNode):
        NODE_NAME = "chat"

        async def execute_stream_with_interactions(self, action_history_manager):
            managers.append(action_history_manager)
            if len(managers) == 1:
                self.pending_input_queue.push("continue please")
            async for action in super().execute_stream_with_interactions(action_history_manager):
                yield action
            results.append(self.result)

    mock_llm_create.reset(responses=[build_simple_response("first answer"), build_simple_response("second answer")])
    node = _ContinuedChat(
        node_id="continued_chat",
        description="API continuation regression",
        node_type=NodeType.TYPE_CHAT,
        agent_config=real_agent_config,
    )
    manager = ChatTaskManager(node_factory=lambda **kwargs: node)
    task = ChatTask(session_id="continued-chat", asyncio_task=MagicMock())
    await manager._run_loop(
        task, real_agent_config, StreamChatInput(message="first question", session_id="continued-chat")
    )

    assert task.status == "completed"
    assert len(results) == 2
    assert managers[0] is managers[1]
    assert [result.response for result in results] == ["first answer", "second answer"]
    first_ids, second_ids = [{action["action_id"] for action in result.action_history} for result in results]
    assert first_ids.isdisjoint(second_ids)
    assert first_ids | second_ids <= {action.action_id for action in managers[0].get_actions()}
    assert len(node.actions) == len({action.action_id for action in node.actions})

    replies = [
        content.payload["content"]
        for event in task.events
        if event.event == "message" and event.data.payload.role == "assistant"
        for content in event.data.payload.content
        if content.type in {"markdown", "thinking"}
    ]
    assert replies == ["first answer", "second answer"]
    user_inserts = [event for event in task.events if event.event == "message" and event.data.payload.role == "user"]
    assert len(user_inserts) == 1
    assert user_inserts[0].data.payload.content[0].payload["content"] == "continue please"
