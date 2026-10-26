# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deliver completed function tools without waiting for their parallel siblings."""

import asyncio
from contextlib import suppress
from typing import Any, AsyncIterator

from agents import AgentHooks, RunItemStreamEvent, ToolCallOutputItem
from agents.tracing import get_current_span


class ToolResultStream(AgentHooks):
    """A run-local hook and stream bridge; the SDK remains the replay owner.

    SDK tool-output events arrive after the entire parallel batch. Hook results
    are available earlier, including on runs cancelled before the batch ends.
    Only the presentation stream is advanced here, never the SDK's model input.
    """

    def __init__(self, delegate: AgentHooks | None = None):
        self.delegate = delegate
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def _forward(self, name: str, *args: Any) -> Any:
        callback = getattr(self.delegate, name, None)
        if callable(callback):
            return await callback(*args)

    async def on_start(self, *args: Any) -> Any:
        return await self._forward("on_start", *args)

    async def on_end(self, *args: Any) -> Any:
        return await self._forward("on_end", *args)

    async def on_handoff(self, *args: Any) -> Any:
        return await self._forward("on_handoff", *args)

    async def on_llm_start(self, *args: Any) -> Any:
        return await self._forward("on_llm_start", *args)

    async def on_llm_end(self, *args: Any) -> Any:
        return await self._forward("on_llm_end", *args)

    async def on_tool_start(self, *args: Any) -> Any:
        return await self._forward("on_tool_start", *args)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> Any:
        try:
            value = await self._forward("on_tool_end", context, agent, tool, result)
        except Exception as exc:
            self._completed(context, agent, {"success": 0, "error": str(exc), "result": None})
            raise
        span = get_current_span()
        # The SDK may turn a raised tool exception into plain text. Preserve
        # its error state instead of presenting that text as successful data.
        if span is not None and span.error:
            result = {"success": 0, "error": span.error.get("message") or str(result), "result": None}
        self._completed(context, agent, result)
        return value

    def _completed(self, context: Any, agent: Any, result: Any) -> None:
        call_id = getattr(context, "tool_call_id", None)
        if not call_id:
            return
        self.queue.put_nowait(
            (
                "event",
                RunItemStreamEvent(
                    name="tool_output",
                    item=ToolCallOutputItem(
                        agent=agent,
                        raw_item={"type": "function_call_output", "call_id": call_id, "output": result},
                        output=result,
                    ),
                ),
            )
        )

    async def stream_events(self, sdk_events: AsyncIterator[Any], result: Any) -> AsyncIterator[Any]:
        """Merge early results and SDK events, ordering and deduplicating by call ID."""

        async def pump() -> None:
            try:
                async for event in sdk_events:
                    self.queue.put_nowait(("event", event))
            except Exception as exc:
                self.queue.put_nowait(("error", exc))
            finally:
                self.queue.put_nowait(("done", None))

        producer = asyncio.create_task(pump())
        started = set()
        completed = set()
        pending = {}
        try:
            while True:
                kind, event = await self.queue.get()
                if kind == "error":
                    raise event
                if kind == "done":
                    # Unmatched SDK outputs retain the previous orphan-result
                    # behavior; hooks must not silently discard them.
                    for pending_event in pending.values():
                        yield pending_event
                    return
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", None)
                raw = getattr(item, "raw_item", None)
                call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
                if item_type == "tool_call_item" and call_id:
                    started.add(call_id)
                    yield event
                    if call_id in pending:
                        completed.add(call_id)
                        yield pending.pop(call_id)
                elif item_type == "tool_call_output_item" and call_id:
                    if call_id in completed:
                        continue
                    if call_id not in started:
                        pending.setdefault(call_id, event)
                        continue
                    completed.add(call_id)
                    yield event
                else:
                    yield event
        finally:
            # Closing a consumer at a yield point otherwise leaves the SDK's
            # background runner (and its tools/spans) alive after chat stops.
            if not result.is_complete:
                result.cancel()
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
