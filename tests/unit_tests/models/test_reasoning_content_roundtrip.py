# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Exercise reasoning replay through the real SDK, LiteLLM, HTTP, and session."""

import json

import httpx
import litellm
import pytest
import pytest_asyncio
from agents import RunConfig, Runner, function_tool
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.models.reasoning_replay import REASONING_ENDPOINT_KEY, reasoning_endpoint_identity
from datus.models.sqlite_session import DatusSQLiteSession

pytestmark = pytest.mark.component

ENDPOINT = "https://models.example.test/v1"
REASONING = ["Check 场控 count.\nKeep <= 20% unchanged.", "Cross-check ranking scope.", "Evidence checked."]


@pytest_asyncio.fixture(autouse=True)
async def stop_logging_worker():
    yield
    await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "provider,model_name", [("openai", "glm-5.3"), ("glm", "glm-5.3"), ("openai", "z-ai/yumcode-pro")]
)
async def test_reasoning_survives_tools_and_session_restore(tmp_path, monkeypatch, streaming, provider, model_name):
    requests = []

    @function_tool
    def read_section(section: str) -> str:
        """Read a section of a skill for review."""
        return f"Evidence for {section}."

    def respond(request):
        assert str(request.url) == f"{ENDPOINT}/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        turn = len(requests) - 1
        message = {"role": "assistant", "content": None, "reasoning_content": REASONING[min(turn, 2)]}
        if turn < 2:
            message["tool_calls"] = [
                {
                    "id": f"read_{turn}",
                    "type": "function",
                    "function": {"name": "read_section", "arguments": json.dumps({"section": f"section_{turn}"})},
                }
            ]
        else:
            message["content"] = "Review complete."
        response = {
            "id": f"completion_{turn}",
            "object": "chat.completion",
            "created": 1,
            # A deployment alias need not match the server's response model.
            "model": "glm-5.3-flashx",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if turn < 2 else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        if not streaming:
            return httpx.Response(200, json=response)

        def chunk(delta, finish_reason=None):
            return {
                **response,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        reasoning = message.pop("reasoning_content")
        if turn < 2:
            message["tool_calls"][0]["index"] = 0
        chunks = [
            chunk({"role": "assistant", "reasoning_content": reasoning[:8]}),
            chunk({"reasoning_content": reasoning[8:]}),
            chunk(message, "tool_calls" if turn < 2 else "stop"),
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(value)}\n\n" for value in chunks) + "data: [DONE]\n\n",
        )

    config = ModelConfig(type=provider, model=model_name, api_key="test-key", base_url=ENDPOINT)

    def build_agent():
        # No explicit thinking flag: the deployment returns reasoning by default.
        return OpenAIModel(config)._build_agent(
            "Review the skill against source evidence.", str, False, {}, [read_section]
        )

    async def run(agent, session, prompt):
        run_config = RunConfig(tracing_disabled=True)
        if streaming:
            result = Runner.run_streamed(agent, prompt, session=session, run_config=run_config)
            async for _ in result.stream_events():
                pass
            return result
        return await Runner.run(agent, prompt, session=session, run_config=run_config)

    db_path = str(tmp_path / "session.db")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        # Isolate SDK clients so another test's cached transport cannot escape
        # this HTTP mock or retain this client after it closes.
        monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", type(litellm.in_memory_llm_clients_cache)())
        session = DatusSQLiteSession(create_tables=True, session_id="review", db_path=db_path)
        try:
            agent = build_agent()
            result = await run(agent, session, "Review both sections.")
            assert result.final_output == "Review complete."
            assert len(requests) == 3
            for request in requests:
                assert request["model"] == model_name
            for turn, request in enumerate(requests[1:], start=1):
                calls = [message for message in request["messages"] if message.get("tool_calls")]
                assert [message["reasoning_content"] for message in calls] == REASONING[:turn]
                assert [message["tool_calls"][0]["id"] for message in calls] == [f"read_{i}" for i in range(turn)]
                outputs = [message for message in request["messages"] if message["role"] == "tool"]
                assert [message["tool_call_id"] for message in outputs] == [f"read_{i}" for i in range(turn)]
            history = await session.get_items()
            reasoning_items = [item for item in history if item.get("type") == "reasoning"]
            assert len(reasoning_items) == 3
            assert all(
                item["provider_data"][REASONING_ENDPOINT_KEY] == reasoning_endpoint_identity(ENDPOINT)
                for item in reasoning_items
            )
            assert requests[0]["max_tokens"] == 384000
        finally:
            session.close()

        # A fresh model and reopened SQLite session must retain source provenance.
        session = DatusSQLiteSession(create_tables=True, session_id="review", db_path=db_path)
        try:
            result = await run(build_agent(), session, "Recheck the result.")
            assert result.final_output == "Review complete."
            replayed = [
                message["reasoning_content"] for message in requests[3]["messages"] if "reasoning_content" in message
            ]
            assert replayed == REASONING
        finally:
            session.close()
