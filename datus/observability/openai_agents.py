# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OpenAI Agents SDK tracing integration used by Datus observability."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from datus.observability.manager import get_observability_manager
from datus.schemas.tool_summary import detect_tool_failure
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_TOOL_ERROR_MESSAGE_MAX_CHARS = 500
_TOOL_ERROR_MESSAGE_FALLBACK = "Tool returned an unsuccessful result"


class DatusOpenAIAgentsInstrumentor:
    """Install OpenInference tracing while merging the first agent span into the trace root."""

    def instrument(self, *, tracer_provider: Any, config: Any) -> None:
        from agents import set_trace_processors
        from openinference.instrumentation import OITracer
        from openinference.instrumentation.openai_agents.version import __version__
        from opentelemetry import trace as trace_api
        from opentelemetry.trace import Tracer

        tracer = OITracer(
            trace_api.get_tracer("openinference.instrumentation.openai_agents", __version__, tracer_provider),
            config=config,
        )
        set_trace_processors([DatusOpenInferenceTracingProcessor(cast(Tracer, tracer))])

    def uninstrument(self) -> None:
        return None


def instrument_openai_agents(*, tracer_provider: Any, config: Any) -> DatusOpenAIAgentsInstrumentor:
    instrumentor = DatusOpenAIAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, config=config)
    return instrumentor


try:
    from openinference.instrumentation.openai_agents import _processor as _oi_processor
except Exception:  # pragma: no cover - import availability is checked during adapter setup.
    _oi_processor = None  # type: ignore[assignment]


_OpenInferenceTracingProcessorBase = (
    object if _oi_processor is None else _oi_processor.OpenInferenceTracingProcessor  # type: ignore[union-attr]
)


class DatusOpenInferenceTracingProcessor(_OpenInferenceTracingProcessorBase):  # type: ignore[misc]
    """OpenInference processor that avoids a duplicate root agent in Langfuse.

    The upstream processor emits one OpenTelemetry span for the Agents SDK trace and
    another span for the first AgentSpanData. Langfuse renders both as agent-like
    observations, which creates a duplicate-looking ``agent/chat -> chat`` tree.
    Datus uses the trace root as the first agent span and nests model/tool spans
    directly underneath it.
    """

    def __init__(self, tracer: Any) -> None:
        if _oi_processor is None:
            raise RuntimeError("openinference.instrumentation.openai_agents is required")
        super().__init__(tracer)
        self._merged_root_agent_span_ids: set[str] = set()
        self._merged_root_trace_ids: set[str] = set()

    def on_trace_end(self, trace: Any) -> None:
        self._merged_root_trace_ids.discard(trace.trace_id)
        super().on_trace_end(trace)

    def on_span_start(self, span: Any) -> None:
        if self._is_mergeable_root_agent_span(span):
            root_span = self._root_spans.get(span.trace_id)
            if root_span is None:
                super().on_span_start(span)
                return
            root_span.set_attribute(_oi_processor.LLM_SYSTEM, _oi_processor.OpenInferenceLLMSystemValues.OPENAI.value)
            self._otel_spans[span.span_id] = root_span
            self._tokens[span.span_id] = _oi_processor.attach(_oi_processor.set_span_in_context(root_span))
            self._merged_root_agent_span_ids.add(span.span_id)
            return

        super().on_span_start(span)
        if isinstance(span.span_data, (_oi_processor.GenerationSpanData, _oi_processor.ResponseSpanData)):
            from datus.observability.model_call import current_model_call

            if call := current_model_call():
                call.bind_sdk_span(span)
                call.bind_span(self._otel_spans.get(span.span_id))

    def on_span_end(self, span: Any) -> None:
        if span.span_id not in self._merged_root_agent_span_ids:
            self._set_datus_span_attributes(span)
            data = span.span_data
            original_output = None
            if isinstance(data, _oi_processor.GenerationSpanData):
                messages = _generation_output_messages(data.output)
                if messages is not None:
                    original_output = data.output
                    data.output = messages
                    if otel_span := self._otel_spans.get(span.span_id):
                        _set_output_message_details(otel_span, messages)
            call = getattr(span, "_datus_model_call", None)
            original_response = None
            if call is not None:
                call.usage(getattr(data, "usage", None))
                if isinstance(data, _oi_processor.ResponseSpanData) and data.response is not None:
                    original_response = data.response
                    # Export definitions once, under the shared content policy.
                    # The upstream processor otherwise re-exports raw tools from
                    # response.tools, including inside the full output envelope.
                    data.response = data.response.model_copy(update={"tools": []})
                    call.usage(getattr(original_response, "usage", None))
            if isinstance(data, (_oi_processor.GenerationSpanData, _oi_processor.ResponseSpanData)):
                if otel_span := self._otel_spans.get(span.span_id):
                    self._otel_spans[span.span_id] = _ModelCallSpan(otel_span, call)
            try:
                super().on_span_end(span)
            finally:
                if original_output is not None:
                    data.output = original_output
                if original_response is not None:
                    span.span_data.response = original_response
            return

        self._merged_root_agent_span_ids.discard(span.span_id)
        if token := self._tokens.pop(span.span_id, None):
            _oi_processor.detach(token)  # type: ignore[arg-type]
        root_span = self._otel_spans.pop(span.span_id, None)
        if root_span is None:
            return
        data = span.span_data
        if isinstance(data, _oi_processor.AgentSpanData):
            root_span.set_attribute(_oi_processor.GRAPH_NODE_ID, data.name)
            key = f"{data.name}:{span.trace_id}"
            if parent_node := self._reverse_handoffs_dict.pop(key, None):
                root_span.set_attribute(_oi_processor.GRAPH_NODE_PARENT_ID, parent_node)

    def _set_datus_span_attributes(self, span: Any) -> None:
        """Add provider-neutral fields missing from the upstream Agents mapper."""
        otel_span = self._otel_spans.get(span.span_id)
        if otel_span is None:
            return
        data = span.span_data
        if isinstance(data, _oi_processor.GenerationSpanData):
            model_config = data.model_config if isinstance(data.model_config, Mapping) else {}
            provider = model_config.get("provider")
            system = model_config.get("system")
            if isinstance(provider, str) and provider:
                otel_span.set_attribute(_oi_processor.LLM_PROVIDER, provider)
            if isinstance(system, str) and system:
                otel_span.set_attribute(_oi_processor.LLM_SYSTEM, system)

            usage = data.usage if isinstance(data.usage, Mapping) else {}
            _set_numeric_attribute(
                otel_span,
                _oi_processor.LLM_TOKEN_COUNT_TOTAL,
                usage.get("total_tokens"),
            )
            _set_numeric_attribute(
                otel_span,
                "llm.token_count.prompt_details.cache_read",
                usage.get("cache_read_input_tokens"),
            )
            _set_numeric_attribute(
                otel_span,
                "llm.token_count.prompt_details.cache_write",
                usage.get("cache_creation_input_tokens"),
            )
        elif isinstance(data, _oi_processor.FunctionSpanData):
            mcp_data = data.mcp_data if isinstance(data.mcp_data, Mapping) else {}
            tool_call_id = mcp_data.get("tool_call_id")
            if tool_call_id:
                otel_span.set_attribute("tool.id", str(tool_call_id))
            if detect_tool_failure(data.output):
                span.set_error(
                    {
                        "message": _tool_failure_message(data.output),
                        "data": {"exception.type": "ToolError"},
                    }
                )

    def shutdown(self) -> None:
        self._merged_root_agent_span_ids.clear()
        self._merged_root_trace_ids.clear()
        super().shutdown()

    def _is_mergeable_root_agent_span(self, span: Any) -> bool:
        if span.parent_id is not None:
            return False
        if span.trace_id not in self._root_spans:
            return False
        if span.trace_id in self._merged_root_trace_ids:
            return False
        if not isinstance(span.span_data, _oi_processor.AgentSpanData):
            return False
        self._merged_root_trace_ids.add(span.trace_id)
        return True


def _set_numeric_attribute(span: Any, key: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    span.set_attribute(key, value)


def _generation_output_messages(output: Any) -> list[dict[str, Any]] | None:
    """Map the SDK's streamed Chat Completions envelope to observable messages.

    LiteLLM streaming stores ``[Response.model_dump()]`` in GenerationSpanData,
    whose OpenInference mapper expects Chat Completions messages. Normalize only
    this envelope, without changing the SDK response or model conversation.
    """
    if not isinstance(output, list) or len(output) != 1:
        return None
    response = output[0]
    if not isinstance(response, Mapping) or response.get("object") != "response":
        return None
    items = response.get("output")
    if not isinstance(items, list):
        return None
    # Preserve unfamiliar provider items rather than silently dropping them.
    if any(
        not isinstance(item, Mapping) or item.get("type") not in {"message", "reasoning", "function_call"}
        for item in items
    ):
        return None

    content: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for item in items:
        if item["type"] == "message":
            parts = item.get("content")
            if not isinstance(parts, list) or any(not isinstance(part, Mapping) for part in parts):
                return None
            content.extend(
                {**part, "type": "text" if part.get("type") == "output_text" else part.get("type")} for part in parts
            )
        elif item["type"] == "reasoning":
            content.append(dict(item))
        else:
            arguments = item.get("arguments", "{}")
            tool_calls.append(
                {
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return [message]


def _set_output_message_details(span: Any, messages: list[dict[str, Any]]) -> None:
    """Fill OpenInference fields not emitted by its Chat Completions mapper."""
    for index, message in enumerate(messages):
        prefix = f"llm.output_messages.{index}.message"
        for call_index, call in enumerate(message.get("tool_calls", [])):
            # Upstream omits {} arguments, which are meaningful for zero-arg tools.
            span.set_attribute(
                f"{prefix}.tool_calls.{call_index}.tool_call.function.arguments",
                call["function"]["arguments"],
            )
        for part_index, part in enumerate(message["content"]):
            if part.get("type") != "reasoning":
                continue
            content_prefix = f"{prefix}.contents.{part_index}.message_content"
            span.set_attribute(f"{content_prefix}.type", "reasoning")
            for key in ("id", "encrypted_content"):
                if isinstance(value := part.get(key), str):
                    span.set_attribute(f"{content_prefix}.{key}", value)
            text = "\n".join(
                entry["text"]
                for entry in part.get("content") or part.get("summary") or []
                if isinstance(entry, Mapping) and isinstance(entry.get("text"), str)
            )
            if text:
                span.set_attribute(f"{content_prefix}.text", text)


class _ModelCallSpan:
    """Finish common LLM attributes after upstream I/O mapping and masking."""

    def __init__(self, span: Any, call: Any):
        self._span = span
        self._call = call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)

    def end(self, *args: Any, **kwargs: Any) -> None:
        try:
            _wrap_message_values(self._span)
        except Exception as exc:
            logger.warning("llm.capture_failed", field="messages", error_type=type(exc).__name__)
        if self._call is not None:
            self._call.end_sdk_span(self._span, *args, **kwargs)
        else:
            self._span.end(*args, **kwargs)


def _wrap_message_values(span: Any) -> None:
    """Keep captured messages in a JSON object understood by OI consumers.

    Read only the upstream processor's masked values. The indexed OpenInference
    attributes remain available for consumers that use individual message fields.
    """
    attributes = getattr(span, "attributes", None) or {}
    for direction in ("input", "output"):
        key = f"{direction}.value"
        try:
            value = json.loads(attributes.get(key, ""))
        except (TypeError, ValueError):
            continue
        if direction == "output" and isinstance(value, dict) and value.get("object") == "response":
            messages = _generation_output_messages([value])
            if messages is not None:
                value = messages
        if isinstance(value, list):
            if direction == "input":
                value = _response_input_messages(value)
            span.set_attribute(key, json.dumps({"messages": value}, ensure_ascii=False))


def _response_input_messages(items: list[Any]) -> list[Any]:
    """Express Responses call/result history as messages, preserving other items."""
    messages = []
    for item in items:
        if isinstance(item, Mapping) and item.get("type") == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output")})
        elif isinstance(item, Mapping) and item.get("type") in {"function_call", "reasoning"}:
            converted = _generation_output_messages([{"object": "response", "output": [item]}])
            messages.extend(converted if converted is not None else [item])
        else:
            messages.append(item)
    return messages


def _tool_failure_message(output: Any) -> str:
    payload = _tool_result_mapping(output)
    error = payload.get("error") if payload is not None else None
    if not isinstance(error, str) or not error.strip():
        nested = _tool_result_mapping(payload.get("raw_output")) if payload is not None else None
        error = nested.get("error") if nested is not None else None
    if not isinstance(error, str) or not error.strip():
        return _TOOL_ERROR_MESSAGE_FALLBACK

    try:
        redacted = get_observability_manager().redact(error)
    except Exception:  # pragma: no cover - redaction is best-effort and must not break tool execution.
        return _TOOL_ERROR_MESSAGE_FALLBACK
    message = " ".join(str(redacted).split()) or _TOOL_ERROR_MESSAGE_FALLBACK
    if len(message) <= _TOOL_ERROR_MESSAGE_MAX_CHARS:
        return message
    return message[: _TOOL_ERROR_MESSAGE_MAX_CHARS - 1].rstrip() + "…"


def _tool_result_mapping(output: Any) -> Mapping[str, Any] | None:
    if isinstance(output, Mapping):
        return output
    if not isinstance(output, str):
        return None
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None
