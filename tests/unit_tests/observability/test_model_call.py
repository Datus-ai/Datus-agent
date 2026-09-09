# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytest_asyncio
from agents import Agent, RunConfig, Runner, function_tool
from openai import AsyncOpenAI

from datus.models.observed_model import ObservedLitellmModel, ObservedResponsesModel
from datus.observability.model_call import ModelCall, current_model_call, model_phase, observation_run
from datus.observability.tool_calls import observe_tool_hooks


@pytest_asyncio.fixture(autouse=True)
async def close_model_test_clients():
    # These tests make real local HTTP calls in separate pytest event loops.
    # Drain LiteLLM's process-wide logging worker before each loop closes.
    from litellm import close_litellm_async_clients
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    yield
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=2)
    finally:
        await GLOBAL_LOGGING_WORKER.stop()
        await close_litellm_async_clients()


@pytest.fixture
def model_endpoint():
    """Real HTTP JSON/SSE evidence: header IDs deliberately differ from body IDs."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            number = len(requests)
            responses = self.path.endswith("/responses")
            self.send_response(200)
            self.send_header("x-request-id", f"provider-raw-{number}")
            streaming = request.get("stream", False)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.end_headers()
            tool_called = any(message.get("role") == "tool" for message in request.get("messages", []))
            invoke = bool(request.get("tools")) and not tool_called and not responses
            message = {"role": "assistant", "content": "done"}
            if invoke:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tool-1", "type": "function", "function": {"name": "echo", "arguments": '{"text":"ok"}'}}
                    ],
                }
            if responses:
                response = {
                    "id": f"resp-body-{number}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "test-model",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg-1",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "done", "annotations": []}],
                        }
                    ],
                    "tools": request.get("tools", []),
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                }
                payload = (
                    {"type": "response.completed", "sequence_number": 0, "response": response}
                    if streaming
                    else response
                )
                wire = f"data: {json.dumps(payload)}\n\n" if streaming else json.dumps(payload)
            elif streaming:
                if invoke:
                    message["tool_calls"][0]["index"] = 0
                payload = {
                    "id": f"chatcmpl-body-{number}",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": message, "finish_reason": None}],
                }
                final = {
                    **payload,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if invoke else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
                wire = f"data: {json.dumps(payload)}\n\ndata: {json.dumps(final)}\n\ndata: [DONE]\n\n"
            else:
                wire = json.dumps(
                    {
                        "id": f"chatcmpl-body-{number}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "test-model",
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": "tool_calls" if invoke else "stop"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                    }
                )
            self.wfile.write(wire.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol,expected_execution", [("litellm", ["ok"]), ("responses", [])])
async def test_wire_tools_and_raw_ids_match_each_exported_generation(
    exported_calls, model_endpoint, protocol, expected_execution, streaming, monkeypatch
):
    exporter, _, _ = exported_calls
    endpoint, requests = model_endpoint
    executed = []
    tool_logger = Mock()
    monkeypatch.setattr("datus.observability.tool_calls.logger", tool_logger)

    @function_tool
    async def echo(text: str) -> str:
        """Echo a supplied string."""
        executed.append(text)
        return text

    @function_tool(is_enabled=False)
    async def hidden() -> str:
        """This disabled tool must not appear on the wire or in Available tools."""
        raise AssertionError("disabled tool ran")

    client = AsyncOpenAI(api_key="local-test-only", base_url=endpoint, max_retries=0)
    model = (
        ObservedResponsesModel("test-model", client)
        if protocol == "responses"
        else ObservedLitellmModel(model="openai/test-model", base_url=endpoint, api_key="local-test-only")
    )
    agent = Agent(name="test", model=model, tools=[echo, hidden], hooks=observe_tool_hooks())
    config = RunConfig(tracing_disabled=False, workflow_name="model-call-test")
    try:
        with observation_run():
            if streaming:
                result = Runner.run_streamed(agent, "Use echo.", run_config=config)
                async for _ in result.stream_events():
                    pass
            else:
                result = await Runner.run(agent, "Use echo.", run_config=config)
    finally:
        await client.close()
    assert result.final_output == "done"
    assert executed == expected_execution
    spans = [span for span in exporter.get_finished_spans() if span.attributes.get("openinference.span.kind") == "LLM"]
    expected_events = [
        {
            "tool_call_id": "tool-1",
            "request_id": "provider-raw-1",
            "model_call_id": spans[0].attributes["datus.llm.model_call_id"],
            "status": "success",
        }
        for _ in expected_execution
    ]
    events = [
        {key: entry.kwargs[key] for key in ("tool_call_id", "request_id", "model_call_id", "status")}
        for entry in tool_logger.info.call_args_list
    ]
    assert events == expected_events
    assert [entry.args[0] for entry in tool_logger.info.call_args_list] == ["tool.finished" for _ in expected_execution]
    assert len(spans) == len(requests)
    assert len({span.attributes["datus.llm.model_call_id"] for span in spans}) == len(requests)
    for number, (span, request) in enumerate(zip(spans, requests), 1):
        attrs = span.attributes
        assert attrs["datus.llm.request_id"] == f"provider-raw-{number}"
        assert attrs["datus.llm.request_id_issuer"] == "unknown"  # A proxy's issuer must not be guessed.
        assert attrs["datus.llm.tools_count"] == 1
        assert attrs["datus.llm.tools_capture_state"] == "complete"
        assert json.loads(attrs["llm.tools.0.tool.json_schema"]) == request["tools"][0]
        assert attrs["llm.tools.0.tool.name"] == "echo"
        assert attrs["datus.llm.request_id_coverage"] == "adapter_visible_response"


def test_capture_policy_empty_tools_and_redaction(exported_calls):
    exporter, manager, provider = exported_calls
    tools = [{"type": "function", "function": {"name": "lookup", "description": "secret-123", "parameters": {}}}]
    manager._tracing_config.redact.patterns = [r"secret-\d+"]
    tracer = provider.get_tracer(__name__)
    for enabled, definitions in [(True, tools), (False, tools), (True, [])]:
        manager._tracing_config.capture.tool_definitions = enabled
        with (
            tracer.start_as_current_span("generation") as span,
            ModelCall(model="m", model_impl="test", protocol="test") as call,
        ):
            call.bind_span(span)
            call.request({"tools": definitions})
    spans = exporter.get_finished_spans()
    assert spans[0].attributes["datus.llm.tools_capture_state"] == "redacted"
    assert "secret-123" not in json.dumps(dict(spans[0].attributes))
    assert spans[1].attributes["datus.llm.tools_count"] == 1
    assert spans[1].attributes["datus.llm.tools_capture_state"] == "disabled"
    assert not any(key.startswith("llm.tools.") for key in spans[1].attributes)
    assert spans[2].attributes["datus.llm.tools_count"] == 0
    assert spans[2].attributes["datus.llm.tools_capture_state"] == "complete"


@pytest.mark.asyncio
async def test_cancel_after_headers_and_concurrent_calls_do_not_lose_or_mix_ids(exported_calls):
    exporter, _, provider = exported_calls

    async def invoke(request_id):
        with provider.get_tracer(__name__).start_as_current_span("generation") as span:
            with ModelCall(model="m", model_impl="test", protocol="test") as call:
                call.bind_span(span)
                call.request({"tools": []})
                call.response(
                    SimpleNamespace(_response_headers={"X-Request-ID": request_id}, id="wrong"), streaming=True
                )
                await asyncio.sleep(0)
                assert current_model_call() is call
                if request_id == "cancelled":
                    raise asyncio.CancelledError()

    await asyncio.gather(invoke("cancelled"), invoke("succeeded"), return_exceptions=True)
    spans = {span.attributes["datus.llm.request_id"]: span for span in exporter.get_finished_spans()}
    assert spans["cancelled"].attributes["datus.llm.status"] == "cancelled"
    assert spans["succeeded"].attributes["datus.llm.status"] == "success"
    assert current_model_call() is None


def test_no_synthetic_id_and_summary_phase():
    with model_phase("compact_summary"), ModelCall(model="m", model_impl="test", protocol="test") as call:
        call.request({"tools": [], "tool_choice": "none"})
        call.response(SimpleNamespace(id="__fake_id__", _response_headers={}))
    assert "request_id" not in call.fields
    assert call.fields["request_id_status"] == "absent"
    assert call.fields["phase"] == "compact_summary"


@pytest.mark.asyncio
@pytest.mark.parametrize("tracing_disabled", [False, True])
async def test_same_named_concurrent_agents_do_not_compare_each_others_tools(
    exported_calls, model_endpoint, tracing_disabled, monkeypatch
):
    log = Mock()
    monkeypatch.setattr("datus.observability.model_call.logger", log)

    @function_tool
    async def lookup_a() -> str:
        return "a"

    @function_tool
    async def lookup_b() -> str:
        return "b"

    endpoint, _ = model_endpoint
    client = AsyncOpenAI(api_key="local-test-only", base_url=endpoint, max_retries=0)
    try:
        # Shared model and outer operation are common for parallel sub-agent tasks.
        model = ObservedResponsesModel("test-model", client)
        with observation_run():
            await asyncio.gather(
                *(
                    Runner.run(
                        Agent(name="worker", model=model, tools=[tool]),
                        "Test.",
                        run_config=RunConfig(tracing_disabled=tracing_disabled),
                    )
                    for tool in (lookup_a, lookup_b)
                )
            )
    finally:
        await client.close()
    events = [call.args[0] for call in log.method_calls]
    assert "tools.changed" not in events
    assert events.count("tools.available") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("tracing_disabled", [False, True])
async def test_sdk_handled_tool_exception_is_not_logged_as_success(
    exported_calls, model_endpoint, monkeypatch, tracing_disabled
):
    @function_tool(name_override="echo")
    async def fail(text: str) -> str:
        raise ValueError("controlled failure")

    endpoint, _ = model_endpoint
    tool_logger = Mock()
    monkeypatch.setattr("datus.observability.tool_calls.logger", tool_logger)
    model = ObservedLitellmModel(model="openai/test-model", base_url=endpoint, api_key="local-test-only")
    with observation_run():
        result = await Runner.run(
            Agent(name="test", model=model, tools=[fail], hooks=observe_tool_hooks()),
            "Test.",
            run_config=RunConfig(tracing_disabled=tracing_disabled),
        )
    assert result.final_output == "done"
    tool_logger.info.assert_called_once()
    assert tool_logger.info.call_args.kwargs["status"] == ("returned" if tracing_disabled else "error")


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_tool_definition_capture_is_independent_of_prompt_content(exported_calls, model_endpoint, enabled):
    from openinference.instrumentation import TraceConfig

    from datus.observability.adapters.otlp import _build_openinference_trace_config
    from datus.observability.config import TracingConfig
    from datus.observability.openai_agents import instrument_openai_agents

    exporter, manager, provider = exported_calls
    config = TracingConfig.from_dict(
        {"enabled": True, "capture_content": False, "capture": {"tool_definitions": enabled}}
    )
    manager._tracing_config = config
    instrument_openai_agents(tracer_provider=provider, config=_build_openinference_trace_config(TraceConfig, config))

    @function_tool
    async def lookup() -> str:
        return "ok"

    endpoint, _ = model_endpoint
    client = AsyncOpenAI(api_key="local-test-only", base_url=endpoint, max_retries=0)
    try:
        await Runner.run(
            Agent(name="test", model=ObservedResponsesModel("test-model", client), tools=[lookup]),
            "PRIVATE-PROMPT-TEXT",
            run_config=RunConfig(tracing_disabled=False),
        )
    finally:
        await client.close()
    (span,) = [s for s in exporter.get_finished_spans() if s.attributes.get("openinference.span.kind") == "LLM"]
    assert span.attributes["datus.llm.tools_capture_state"] == ("complete" if enabled else "disabled")
    assert span.attributes["datus.llm.tools_count"] == 1
    if enabled:
        assert json.loads(span.attributes["llm.tools.0.tool.json_schema"])["name"] == "lookup"
    else:
        assert not any(key.startswith("llm.tools.") for key in span.attributes)
    assert "PRIVATE-PROMPT-TEXT" not in json.dumps(dict(span.attributes))
