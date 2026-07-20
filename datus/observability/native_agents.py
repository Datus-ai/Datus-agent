# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tracing helpers for agent loops that do not run through ``agents.Runner``."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import sys
from collections.abc import Mapping
from typing import Any, AsyncGenerator, Callable

from agents import agent_span, function_span, generation_span, trace

from datus.observability.manager import get_observability_manager
from datus.utils.loggings import get_logger
from datus.utils.trace_context import build_agents_run_config_kwargs, build_trace_span_attributes

logger = get_logger(__name__)


def native_agents_tracing_enabled() -> bool:
    """Return whether Datus has an active external tracing adapter."""
    return bool(get_observability_manager().adapters)


def trace_native_agent_stream(func: Callable[..., AsyncGenerator[Any, None]]):
    """Give a custom async agent loop the same root trace shape as ``Runner``.

    The OpenAI Agents SDK context managers intentionally avoid resetting their
    context on ``GeneratorExit``. That is unsafe for a long-lived async generator,
    so this wrapper starts and finishes the trace/span explicitly and always
    restores both context variables when the consumer closes the stream.
    """

    signature = inspect.signature(func)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        observability = get_observability_manager()
        if not observability.adapters:
            stream = func(*args, **kwargs)
            try:
                async for item in stream:
                    yield item
            finally:
                await _close_async_stream(stream)
            return

        bound = signature.bind_partial(*args, **kwargs)
        extra_kwargs = bound.arguments.get("kwargs") or {}
        agent_name = str(extra_kwargs.get("agent_name") or kwargs.get("agent_name") or "Tools_Agent")
        func_tools = bound.arguments.get("func_tools") or kwargs.get("func_tools") or []
        tool_names = [str(tool.name) for tool in func_tools if getattr(tool, "name", None)] or None
        output_type = _output_type_name(bound.arguments.get("output_type") or kwargs.get("output_type"))

        run_config = build_agents_run_config_kwargs(agent_name=agent_name)
        workflow_name = str(run_config.get("workflow_name") or f"agent/{agent_name}")
        attributes = build_trace_span_attributes(operation=agent_name, run_type="llm")
        trace_name = str(attributes.get("datus.trace.name") or workflow_name)
        request_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        output_queue: asyncio.Queue[tuple[str, Any, Any]] = asyncio.Queue(maxsize=1)

        async def produce() -> None:
            stream = func(*args, **kwargs)
            trace_obj = trace(
                workflow_name=workflow_name,
                group_id=run_config.get("group_id"),
                metadata=run_config.get("trace_metadata"),
            )
            agent_obj = None
            try:
                with observability.trace_baggage(trace_name, attributes):
                    trace_obj.start(mark_as_current=True)
                    agent_obj = agent_span(name=agent_name, tools=tool_names, output_type=output_type)
                    agent_obj.start(mark_as_current=True)

                    while True:
                        await request_queue.get()
                        try:
                            item = await anext(stream)
                        except StopAsyncIteration:
                            await output_queue.put(("done", None, None))
                            break
                        await output_queue.put(("item", item, None))
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                if agent_obj is not None and not isinstance(exc, GeneratorExit):
                    agent_obj.set_error(_span_error(exc, observability))
                await output_queue.put(("error", exc, sys.exc_info()[2]))
            finally:
                await _close_async_stream(stream)
                if agent_obj is not None:
                    try:
                        agent_obj.finish(reset_current=True)
                    except Exception as exc:  # pragma: no cover - processor failures are best-effort
                        logger.debug("Failed to finish native agent span: %s", exc)
                try:
                    trace_obj.finish(reset_current=True)
                except Exception as exc:  # pragma: no cover - processor failures are best-effort
                    logger.debug("Failed to finish native agent trace: %s", exc)

        producer = asyncio.create_task(produce(), name=f"datus-native-trace-{agent_name}")
        try:
            while True:
                await request_queue.put(None)
                event, payload, traceback = await output_queue.get()
                if event == "item":
                    yield payload
                    continue
                if event == "error":
                    raise payload.with_traceback(traceback)
                break
        finally:
            if not producer.done():
                producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass

    return wrapper


def start_native_generation_span(
    *,
    input_messages: list[dict[str, Any]] | None,
    model: str,
    model_config: dict[str, Any],
):
    """Start one model-call span under the current native agent trace."""
    span = generation_span(
        input=input_messages,
        model=model,
        model_config=model_config,
        disabled=not native_agents_tracing_enabled(),
    )
    span.start(mark_as_current=True)
    return span


def start_native_tool_span(*, name: str, tool_call_id: str | None, input_value: Any):
    """Start one tool-call span under the current native agent trace."""
    captured_input = capture_native_trace_content("tool_args", input_value)
    input_json = _json_string(captured_input) if captured_input is not None else None
    span = function_span(name=name, input=input_json, disabled=not native_agents_tracing_enabled())
    if tool_call_id:
        span.span_data.mcp_data = {"tool_call_id": str(tool_call_id)}
    span.start(mark_as_current=True)
    return span


def finish_native_span(span: Any, *, output: Any = None, error: BaseException | str | None = None) -> None:
    """Finish a generation/tool span without allowing telemetry to break a run."""
    if span is None:
        return
    observability = get_observability_manager()
    try:
        if output is not None:
            span.span_data.output = output
        if error is not None and not isinstance(error, GeneratorExit):
            span.set_error(_span_error(error, observability))
        span.finish(reset_current=True)
    except Exception as exc:  # pragma: no cover - processor failures are best-effort
        logger.debug("Failed to finish native agent child span: %s", exc)


def capture_native_trace_content(field_name: str, value: Any) -> Any:
    """Apply the shared Datus capture and redaction policy to native span data."""
    observability = get_observability_manager()
    if not observability.content_enabled(field_name):
        return None
    return _redact_embedded_json(observability.redact(value), observability)


def _output_type_name(output_type: Any) -> str | None:
    if output_type is None:
        return None
    if isinstance(output_type, type):
        return output_type.__name__
    if isinstance(output_type, dict):
        return "dict"
    return str(output_type)


def _span_error(error: BaseException | str, observability: Any) -> dict[str, Any]:
    if isinstance(error, BaseException):
        message = str(error) or type(error).__name__
        error_type = type(error).__name__
    else:
        message = str(error)
        error_type = "ToolError"
    return {
        "message": str(observability.redact(message)),
        "data": {"exception.type": error_type},
    }


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _redact_embedded_json(value: Any, observability: Any) -> Any:
    """Redact sensitive fields inside JSON strings produced by tool protocols."""
    if isinstance(value, Mapping):
        return {str(key): _redact_embedded_json(item, observability) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_embedded_json(item, observability) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_embedded_json(item, observability) for item in value)
    if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value
    redacted = _redact_embedded_json(observability.redact(parsed), observability)
    return json.dumps(redacted, ensure_ascii=False, default=str)


async def _close_async_stream(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if not callable(close):
        return
    try:
        await close()
    except Exception as exc:  # pragma: no cover - cleanup is best-effort
        logger.debug("Failed to close native agent stream: %s", exc)
