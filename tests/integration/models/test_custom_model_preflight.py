"""Send real SDK HTTP requests to a local endpoint without a registered model."""

import base64
import io
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agents import RunConfig
from PIL import Image, ImageDraw, ImageFont

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.models.request_capacity import ContextCapacityError
from datus.schemas.action_history import ActionHistoryManager
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

pytestmark = pytest.mark.integration


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    from datus.cli import provider_model_catalog as catalog
    from datus.models import openai_compatible

    monkeypatch.setattr(catalog, "_cache_file_path", lambda: tmp_path / catalog.CACHE_FILE_NAME)
    monkeypatch.setattr(catalog, "fetch_openrouter_models", lambda **kwargs: None)
    monkeypatch.setattr(catalog, "_MODEL_DETAILS_CACHE", None)
    monkeypatch.setattr(openai_compatible, "_MODEL_SPECS_CACHE", None)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request))
            has_image = any(
                isinstance(message.get("content"), list)
                and any(part.get("type") == "image_url" for part in message["content"])
                for message in request["messages"]
            )
            call = {
                "id": "read_chart",
                "type": "function",
                "function": {"name": "read_image", "arguments": '{"path":"chart.png"}'},
            }
            message = (
                {"role": "assistant", "content": "image reached endpoint"}
                if has_image or not request.get("tools")
                else {"role": "assistant", "content": None, "tool_calls": [call]}
            )
            finish = "tool_calls" if "tool_calls" in message else "stop"
            self.send_response(200)
            if request.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                delta = {**message}
                if "tool_calls" in delta:
                    delta["tool_calls"] = [{"index": 0, **call}]
                chunk = {
                    "id": "local-response",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": request["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "id": "local-response",
                            "object": "chat.completion",
                            "created": 0,
                            "model": request["model"],
                            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                        }
                    ).encode()
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_read_image_reaches_unregistered_model_over_http(endpoint, tmp_path, monkeypatch, streaming):
    url, requests = endpoint
    Image.new("RGB", (32, 32), "red").save(tmp_path / "chart.png")
    model = OpenAIModel(ModelConfig(type="openai", model="Qwen/private-vision", api_key="local-test", base_url=url))
    monkeypatch.setattr(model, "_build_run_config", lambda **kwargs: RunConfig(tracing_disabled=True))
    tools = [
        tool for tool in FilesystemFuncTool(root_path=str(tmp_path)).available_tools() if tool.name == "read_image"
    ]
    if streaming:
        actions = [
            action
            async for action in model.generate_with_tools_stream(
                "Read chart.png", tools=tools, action_history_manager=ActionHistoryManager(), max_turns=3
            )
        ]
        assert "image reached endpoint" in str(actions)
    else:
        result = await model.generate_with_tools("Read chart.png", tools=tools, max_turns=3)
        assert result["content"] == "image reached endpoint"

    assert len(requests) == 2
    for path, body in requests:
        assert path == "/v1/chat/completions"
        assert body["model"] == "Qwen/private-vision"
        assert body["max_tokens"] == 384000
    content = [
        part
        for message in requests[-1][1]["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ]
    image = next(part["image_url"]["url"] for part in content if part.get("type") == "image_url")
    with Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1]))) as decoded:
        assert decoded.getpixel((0, 0)) == (255, 0, 0)


def test_direct_generation_sends_large_estimated_request(endpoint):
    url, requests = endpoint
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="private-text",
            api_key="local-test",
            base_url=url,
            context_length=8192,
            max_tokens=1024,
        )
    )
    prompt = "\N{CJK UNIFIED IDEOGRAPH-8868}" * 5000
    assert model.generate(prompt) == "image reached endpoint"
    assert len(requests) == 1
    assert requests[0][1]["messages"][-1]["content"] == prompt
    assert requests[0][1]["max_tokens"] == 1024


def test_impossible_explicit_output_reservation_stops_before_http(endpoint):
    url, requests = endpoint
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="private-text",
            api_key="local-test",
            base_url=url,
            context_length=8192,
            max_tokens=16384,
        )
    )
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        model.generate("hello")
    assert requests == []


@pytest.mark.parametrize("output", [None, 512])
def test_direct_generation_keeps_surviving_metadata_in_http_request(endpoint, monkeypatch, output):
    import litellm

    from datus.models.openai_compatible import OpenAICompatibleModel

    url, requests = endpoint
    model = OpenAIModel(ModelConfig(type="openai", model="gpt-4o", api_key="local-test", base_url=url))

    def unavailable(_model):
        raise OSError("catalog unavailable")

    monkeypatch.setattr(OpenAICompatibleModel, "model_specs", property(unavailable))
    monkeypatch.setattr(litellm, "model_cost", {"gpt-4o": {"max_input_tokens": 128000, "max_output_tokens": 16384}})
    kwargs = {} if output is None else {"max_tokens": output}
    assert model.generate("hello", **kwargs) == "image reached endpoint"
    assert len(requests) == 1
    assert requests[0][1]["max_tokens"] == (16384 if output is None else output)


@pytest.mark.asyncio
async def test_trusted_oversized_text_with_image_stops_before_http(endpoint, tmp_path):
    url, requests = endpoint
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(output, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="Qwen/private-vision",
            api_key="local-test",
            base_url=url,
            context_length=8192,
            max_tokens=1024,
            tokenizer_ref="tiktoken:cl100k_base",
        )
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "word " * 12000},
                {"type": "input_image", "image_url": image_url},
            ],
        }
    ]
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        await model.generate_with_tools(prompt, max_turns=1)
    assert requests == []


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(
    not os.environ.get("DATUS_SELF_HOSTED_BASE_URL") or not os.environ.get("DATUS_SELF_HOSTED_MODEL"),
    reason="Set DATUS_SELF_HOSTED_BASE_URL and DATUS_SELF_HOSTED_MODEL for real vision acceptance",
)
@pytest.mark.asyncio
async def test_self_hosted_model_reads_code_from_image(tmp_path):
    """Require actual visual recognition, rather than a canned endpoint response."""
    key_name = os.environ.get("DATUS_SELF_HOSTED_API_KEY_ENV", "LLM_API_KEY")
    if not os.environ.get(key_name):
        pytest.skip(f"Set the credential environment variable {key_name}")
    code = str(100000 + secrets.randbelow(900000))
    picture = Image.new("RGB", (500, 160), "white")
    ImageDraw.Draw(picture).text((40, 40), code, fill="black", font=ImageFont.load_default(size=64))
    picture.save(tmp_path / "code.png")
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model=os.environ["DATUS_SELF_HOSTED_MODEL"],
            base_url=os.environ["DATUS_SELF_HOSTED_BASE_URL"],
            api_key=os.environ[key_name],
            max_tokens=512,
            reasoning_effort="off",
        )
    )
    tools = [
        tool for tool in FilesystemFuncTool(root_path=str(tmp_path)).available_tools() if tool.name == "read_image"
    ]
    result = await model.generate_with_tools(
        "Use read_image to inspect code.png. Return only the six digits printed in the image.",
        tools=tools,
        max_turns=4,
    )
    assert code in result["content"]
