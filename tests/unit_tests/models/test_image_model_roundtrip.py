# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Real SDK/tool/session round trips with deterministic model responses."""

import base64
import copy
import io
import json
from unittest.mock import patch

import httpx
import pytest
from agents import ModelSettings, RunConfig
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from PIL import Image

from datus.agent.node.compact_archive import ToolArchive, archive_old_tool_outputs
from datus.agent.node.context_rewriter import estimate_items_tokens
from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.models.session_manager import SessionManager
from datus.models.sqlite_session import DatusSQLiteSession
from datus.schemas.action_history import ActionHistoryManager
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.utils.image_content import contains_images


class ImageSequenceModel(Model):
    def __init__(self):
        self.inputs = []

    async def get_response(self, system_instructions, input, *args, **kwargs):
        self.inputs.append(copy.deepcopy(input))
        if len(self.inputs) == 1:
            output = [
                ResponseFunctionToolCall(
                    name="read_image",
                    arguments='{"path":"chart.png"}',
                    call_id="read_chart",
                    type="function_call",
                )
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id="answer",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[ResponseOutputText(type="output_text", text="A red chart.", annotations=[])],
                )
            ]
        return ModelResponse(output=output, usage=Usage(input_tokens=100, output_tokens=10), response_id=None)

    async def stream_response(self, system_instructions, input, *args, **kwargs):
        result = await self.get_response(system_instructions, input)
        response = Response.model_construct(
            id=f"response_{len(self.inputs)}",
            output=result.output,
            status="completed",
            usage=ResponseUsage(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                input_tokens_details={"cached_tokens": 0},
                output_tokens_details={"reasoning_tokens": 0},
            ),
        )
        yield ResponseCompletedEvent(type="response.completed", sequence_number=1, response=response)


@pytest.fixture
def image_tools(tmp_path):
    Image.new("RGB", (80, 40), "red").save(tmp_path / "chart.png")
    return [tool for tool in FilesystemFuncTool(root_path=str(tmp_path)).available_tools() if tool.name == "read_image"]


def make_model(route, fake, monkeypatch):
    if route == "codex":
        from datus.models.codex_model import CodexModel

        with patch("datus.models.codex_model.OAuthManager"):
            model = CodexModel(ModelConfig(type="codex", model="gpt-test", api_key="", auth_type="oauth"))
        monkeypatch.setattr(model, "_refresh_client_token", lambda: None)
        monkeypatch.setattr(model, "_get_responses_model", lambda: fake)
        monkeypatch.setattr(model, "_codex_model_settings", lambda *args: ModelSettings())
    elif route == "deepseek":
        from datus.models.deepseek_model import DeepSeekModel

        model = DeepSeekModel(
            ModelConfig(type="deepseek", model="deepseek-v4-flash", api_key="test-key", retry_interval=0)
        )
        monkeypatch.setattr(model.litellm_adapter, "get_agents_sdk_model", lambda: fake)
    else:
        model = OpenAIModel(ModelConfig(type="openai", model="gpt-test", api_key="test-key", retry_interval=0))
        monkeypatch.setattr(model.litellm_adapter, "get_agents_sdk_model", lambda: fake)
    monkeypatch.setattr(model, "_build_run_config", lambda **kwargs: RunConfig(tracing_disabled=True))
    return model


async def run_model(model, tools, session, streaming):
    if streaming:
        actions = [
            action
            async for action in model.generate_with_tools_stream(
                "Read chart.png",
                tools=tools,
                session=session,
                action_history_manager=ActionHistoryManager(),
            )
        ]
        assert "base64" not in str([action.model_dump() for action in actions])
    else:
        result = await model.generate_with_tools("Read chart.png", tools=tools, session=session)
        assert result["content"] == "A red chart."


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["deepseek", "openai", "codex"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_image_reaches_next_request_and_survives_resume(tmp_path, image_tools, monkeypatch, route, streaming):
    fake = ImageSequenceModel()
    model = make_model(route, fake, monkeypatch)
    db = str(tmp_path / "image_roundtrip.db")
    session = DatusSQLiteSession(create_tables=True, session_id="image_roundtrip", db_path=db)
    await run_model(model, image_tools, session, streaming)
    assert len(fake.inputs) == 2
    image_result = next(item for item in fake.inputs[1] if item.get("type") == "function_call_output")
    url = image_result["output"][1]["image_url"]
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as decoded:
        assert decoded.getpixel((0, 0)) == (255, 0, 0)
    session.close()
    restored = DatusSQLiteSession(create_tables=True, session_id="image_roundtrip", db_path=db)
    assert contains_images(await restored.get_items())
    await run_model(model, image_tools, restored, streaming)
    assert contains_images(fake.inputs[-1])
    restored.close()

    manager = SessionManager(session_dir=str(tmp_path))
    displayed = manager.get_session_messages("image_roundtrip")
    assert displayed
    assert "base64" not in str(displayed)
    assert "chart.png" in str(displayed)
    manager.close_all_sessions()


@pytest.mark.asyncio
async def test_native_claude_receives_images_and_shows_only_metadata(tmp_path, image_tools):
    from tests.unit_tests.models.test_claude_model import (
        _make_claude_model,
        _make_model_config,
        _make_response,
        _make_text_block,
        _make_tool_use_block,
    )

    model = _make_claude_model(_make_model_config(use_native_api=True))
    requests = []

    def respond(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            return _make_response([_make_tool_use_block("read_image", "image_1", {"path": "chart.png"})])
        return _make_response([_make_text_block("A red chart.")])

    model.anthropic_client.messages.create.side_effect = respond
    session = DatusSQLiteSession(create_tables=True, session_id="native_image", db_path=str(tmp_path / "native.db"))
    actions = [
        action
        async for action in model._generate_with_mcp_stream(
            prompt="Read chart.png",
            mcp_servers={},
            instruction="",
            output_type=str,
            func_tools=image_tools,
            session=session,
            action_history_manager=ActionHistoryManager(),
        )
    ]
    assert len(requests) == 2
    tool_result = requests[1][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"][1]["source"]["type"] == "base64"
    assert "base64" not in str([action.model_dump() for action in actions])
    assert contains_images(await session.get_items())
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("result_shape", ["mapping", "rows", "text"])
async def test_native_claude_delivers_query_data_to_model_and_history(tmp_path, result_shape):
    from agents import function_tool

    from tests.unit_tests.models.test_claude_model import (
        _make_claude_model,
        _make_model_config,
        _make_response,
        _make_text_block,
        _make_tool_use_block,
    )

    rows = [
        {"type": "image_url", "count": 42},
        {"type": "input_image", "count": 7},
        {"type": "image", "source": "warehouse", "count": 5},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/chart.png"}, "count": 3},
        {"type": "input_image", "image_url": "https://example.invalid/chart.png"},
    ]
    payload = {"success": 1, "result": {"data": rows, "preview": "data:image/png;base64,c2VjcmV0"}}
    value = rows if result_shape == "rows" else json.dumps(payload) if result_shape == "text" else payload

    @function_tool
    def query_assets() -> dict | list | str:
        """Return asset counts grouped by type."""
        return value

    model = _make_claude_model(_make_model_config(use_native_api=True))
    requests = []

    def respond(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            return _make_response([_make_tool_use_block("query_assets", "query_1", {})])
        return _make_response([_make_text_block("42 image_url assets.")])

    model.anthropic_client.messages.create.side_effect = respond
    session = DatusSQLiteSession(create_tables=True, session_id="query", db_path=str(tmp_path / "query.db"))
    actions = [
        action
        async for action in model._generate_with_mcp_stream(
            prompt="Count assets",
            mcp_servers={},
            instruction="",
            output_type=str,
            func_tools=[query_assets],
            session=session,
            action_history_manager=ActionHistoryManager(),
        )
    ]
    expected = {"mapping": payload, "rows": rows, "text": payload}[result_shape]
    model_result = requests[1][-1]["content"][0]["content"]
    assert json.loads(model_result) == expected
    history = await session.get_items()
    saved = next(
        block
        for item in history
        if isinstance(item.get("content"), list)
        for block in item["content"]
        if block.get("type") == "tool_result"
    )
    assert json.loads(saved["content"]) == expected
    displayed = next(action.output for action in actions if action.action_id == "complete_query_1")
    displayed_payload = json.loads(displayed["raw_output"])
    redacted_payload = copy.deepcopy(payload)
    redacted_payload["result"]["preview"] = "[image data omitted]"
    expected_display = {"mapping": redacted_payload, "rows": rows, "text": redacted_payload}[result_shape]
    assert displayed_payload == expected_display
    session.close()


@pytest.mark.asyncio
async def test_compact_drops_old_images_but_keeps_recent_image(tmp_path, image_tools):
    from types import SimpleNamespace

    from agents.items import ItemHelpers

    output = await image_tools[0].on_invoke_tool(None, '{"path":"chart.png"}')
    item = ItemHelpers.tool_call_output_item(SimpleNamespace(call_id="image_1"), output)
    original = [item, {**item, "call_id": "image_2"}]
    archive = ToolArchive("project", "image", base_dir=tmp_path / "archive")
    rewritten, count = archive_old_tool_outputs(
        original, item_format="responses", archive=archive, threshold=100, keep_recent=1
    )
    assert count == 1
    assert not contains_images(rewritten[0])
    assert contains_images(rewritten[1])
    assert "chart.png" in json.dumps(rewritten[0])
    assert contains_images(original[0])
    assert 4096 < estimate_items_tokens([item]) < 5000

    from datus.agent.node.compact_archive import maybe_truncate_item
    from datus.utils.image_content import anthropic_image_tool_content

    for old in (
        item,
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "image_1",
                    "content": anthropic_image_tool_content(output),
                }
            ],
        },
    ):
        compacted = maybe_truncate_item(old, archive, 100, 0)
        assert not contains_images(compacted)
        assert "chart.png" in json.dumps(compacted)
        assert maybe_truncate_item(compacted, archive, 100, 0) is compacted
        assert contains_images(old)


async def _run_litellm_image_roundtrip(tmp_path, image_tools, monkeypatch, provider):
    import litellm
    from agents import Agent, Runner

    from datus.models.litellm_image import ImageToolLitellmModel

    requests = []

    async def respond(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read_chart",
                        "type": "function",
                        "function": {"name": "read_image", "arguments": '{"path":"chart.png"}'},
                        "provider_specific_fields": {"thought_signature": "test-signature"},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "A red chart."}
        return litellm.ModelResponse(
            model=f"{provider}/test",
            choices=[{"index": 0, "message": message, "finish_reason": "stop"}],
            usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        )

    monkeypatch.setattr(litellm, "acompletion", respond)
    agent = Agent(
        name="Image reader",
        model=ImageToolLitellmModel(model=f"{provider}/test", api_key="test-key"),
        tools=image_tools,
    )
    session = DatusSQLiteSession(create_tables=True, session_id="litellm", db_path=str(tmp_path / "session.db"))
    result = await Runner.run(agent, "Read chart.png", session=session, run_config=RunConfig(tracing_disabled=True))
    history = await session.get_items()
    session.close()
    return result, requests, history


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepseek", "openai", "anthropic", "gemini"])
async def test_litellm_request_carries_tool_image_as_user_content(tmp_path, image_tools, monkeypatch, provider):
    result, requests, history = await _run_litellm_image_roundtrip(tmp_path, image_tools, monkeypatch, provider)
    assert result.final_output == "A red chart."
    assert len(requests) == 2
    assert requests[1][-2]["role"] == "tool"
    assert isinstance(requests[1][-2]["content"], str)
    assert requests[1][-1]["role"] == "user"
    assert requests[1][-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert contains_images(history)


@pytest.mark.asyncio
async def test_gemini_image_roundtrip_preserves_thought_signature(tmp_path, image_tools, monkeypatch):
    _, requests, _ = await _run_litellm_image_roundtrip(tmp_path, image_tools, monkeypatch, "gemini")
    call = next(item for item in requests[1] if item.get("tool_calls"))["tool_calls"][0]
    assert call["provider_specific_fields"]["thought_signature"] == "test-signature"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_deepseek_http_request_delivers_image_and_reasoning(tmp_path, image_tools, monkeypatch, streaming):
    """Exercise the actual provider conversion, mocking only the HTTP transport."""
    from agents import Runner
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    from datus.models.deepseek_model import DeepSeekModel

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url == "https://api.deepseek.com/chat/completions"
        first = len(requests) == 1
        message = {"role": "assistant", "content": None, "reasoning_content": "Inspect the chart."}
        if first:
            message["tool_calls"] = [
                {
                    "id": "read_chart",
                    "type": "function",
                    "function": {"name": "read_image", "arguments": '{"path":"chart.png"}'},
                }
            ]
        else:
            message["content"] = "A red chart."
        response = {
            "id": f"completion_{len(requests)}",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if first else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }
        if not streaming:
            return httpx.Response(200, json=response)
        if first:
            message["tool_calls"][0]["index"] = 0
        response["object"] = "chat.completion.chunk"
        choice = response["choices"][0]
        choice["delta"] = choice.pop("message")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(AsyncHTTPHandler, "create_client", lambda self, **kwargs: client)
        handler = AsyncHTTPHandler()
        monkeypatch.setattr(
            "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client", lambda **kwargs: handler
        )
        model = DeepSeekModel(
            ModelConfig(type="deepseek", model="deepseek-v4-flash", api_key="test-key", enable_thinking=True)
        )
        agent = model._build_agent("Read the image.", str, False, {}, image_tools)
        session = DatusSQLiteSession(create_tables=True, session_id="deepseek", db_path=str(tmp_path / "session.db"))
        try:
            if streaming:
                result = Runner.run_streamed(
                    agent, "Read chart.png", session=session, run_config=RunConfig(tracing_disabled=True)
                )
                async for _ in result.stream_events():
                    pass
            else:
                result = await Runner.run(
                    agent, "Read chart.png", session=session, run_config=RunConfig(tracing_disabled=True)
                )
            assert result.final_output == "A red chart."
            assert len(requests) == 2
            body = requests[1]
            assert body["model"] == "deepseek-v4-flash"
            assert body["thinking"] == {"type": "enabled"}
            tool_call = next(message for message in body["messages"] if message.get("tool_calls"))
            assert tool_call["reasoning_content"] == "Inspect the chart."
            tool_result, image_message = body["messages"][-2:]
            assert tool_result["role"] == "tool"
            assert tool_result["tool_call_id"] == "read_chart"
            assert "chart.png" in tool_result["content"]
            assert image_message["role"] == "user"
            url = image_message["content"][1]["image_url"]["url"]
            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as decoded:
                assert decoded.getpixel((0, 0)) == (255, 0, 0)
        finally:
            session.close()
