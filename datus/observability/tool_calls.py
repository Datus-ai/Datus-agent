# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tool lifecycle summaries composed with the existing agent hooks."""

import asyncio
import json
import time

from agents import AgentHooks

from datus.schemas.tool_summary import detect_tool_failure
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ToolObservationHooks(AgentHooks):
    def __init__(self, delegate=None):
        self.delegate = delegate
        self._started = {}

    async def _forward(self, name, *args, **kwargs):
        method = getattr(self.delegate, name, None)
        if callable(method):
            return await method(*args, **kwargs)

    async def on_start(self, *args, **kwargs):
        return await self._forward("on_start", *args, **kwargs)

    async def on_end(self, *args, **kwargs):
        return await self._forward("on_end", *args, **kwargs)

    async def on_handoff(self, *args, **kwargs):
        return await self._forward("on_handoff", *args, **kwargs)

    async def on_llm_start(self, *args, **kwargs):
        return await self._forward("on_llm_start", *args, **kwargs)

    async def on_llm_end(self, *args, **kwargs):
        return await self._forward("on_llm_end", *args, **kwargs)

    async def on_tool_start(self, context, agent, tool):
        call_id = getattr(context, "tool_call_id", None)
        self._started[call_id or id(context)] = time.monotonic()
        logger.debug("tool.started", tool_name=tool.name, tool_call_id=call_id)
        try:
            return await self._forward("on_tool_start", context, agent, tool)
        except BaseException as exc:
            self._finished(context, tool, error=exc)
            raise

    async def on_tool_end(self, context, agent, tool, result):
        try:
            value = await self._forward("on_tool_end", context, agent, tool, result)
        except BaseException as exc:
            self._finished(context, tool, result=result, error=exc)
            raise
        self._finished(context, tool, result=result)
        return value

    def _finished(self, context, tool, *, result=None, error=None):
        call_id = getattr(context, "tool_call_id", None)
        started = self._started.pop(call_id or id(context), None)
        status = _result_status(result)
        if error is not None:
            status = (
                "cancelled"
                if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                else "permission_denied"
                if type(error).__name__ == "PermissionDeniedException"
                else "error"
            )
        status = getattr(context, "_datus_tool_status", None) or status
        from datus.observability.model_call import tool_call_identity

        logger.info(
            "tool.finished",
            tool_name=tool.name,
            tool_call_id=call_id,
            status=status,
            duration_ms=round((time.monotonic() - started) * 1000, 2) if started is not None else None,
            error_type=type(error).__name__ if error is not None else None,
            **tool_call_identity(call_id),
        )


def _result_status(result):
    from agents.tracing import get_current_span

    span = get_current_span()
    function_span = span is not None and span.span_data.type == "function" and span.span_id != "no-op"
    # The SDK can catch an exception and return its error formatter's plain text.
    # A normal end callback alone therefore does not prove the tool succeeded.
    if function_span and span.error:
        return "error"
    if detect_tool_failure(result):
        return "unsuccessful_result"
    data = result
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (TypeError, ValueError):
            data = None
    if isinstance(data, dict) and data.get("success") in (True, 1):
        return "success"
    return "success" if function_span else "returned"


def observe_tool_hooks(hooks=None):
    return hooks if isinstance(hooks, ToolObservationHooks) else ToolObservationHooks(hooks)
