"""Web file review that preserves the owning node's server-side tool implementation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from agents import FunctionTool
from agents.tool_context import ToolContext

from datus.schemas.action_history import ActionHistory, ActionRole
from datus.tools.proxy.tool_result_channel import DEFAULT_RESULT_TIMEOUT, ToolResultChannel

FILE_MUTATIONS = frozenset({"write_file", "edit_file", "delete_file"})


@dataclass
class PendingFileReview:
    accepted: asyncio.Future[bool] = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    result: asyncio.Future[dict[str, Any]] = field(default_factory=lambda: asyncio.get_running_loop().create_future())

    async def execute(self) -> dict[str, Any]:
        if not self.accepted.done():
            self.accepted.set_result(True)
        # A disconnected HTTP request must not cancel execution or a later retry.
        return await asyncio.shield(self.result)


@dataclass
class WebFileReviewContext:
    channel: ToolResultChannel
    pending: dict[str, PendingFileReview] = field(default_factory=dict)


def stamp_tool_execution(action: ActionHistory, node: Any) -> None:
    """Carry execution ownership across nested action buses and history conversion."""
    if action.role != ActionRole.TOOL or action.tool_execution is not None:
        return
    name = action.function_name()
    if name in getattr(node, "file_review_tool_names", set()):
        action.tool_execution = "server_review"
    elif name in getattr(node, "proxied_tool_names", set()):
        action.tool_execution = "client"
    else:
        action.tool_execution = "server"


def _review_tool(original: FunctionTool, context: WebFileReviewContext) -> FunctionTool:
    async def invoke(tool_ctx: ToolContext, args_str: str) -> dict[str, Any]:
        call_id = tool_ctx.tool_call_id
        review = PendingFileReview()
        finished = asyncio.create_task(context.channel.wait_for(call_id))
        failure = {"success": 0, "error": "File review expired or was stopped; inspect the file before retrying."}
        try:
            # Register the channel waiter before exposing the review to HTTP/stop.
            await asyncio.sleep(0)
            context.pending[call_id] = review
            done, _ = await asyncio.wait(
                {review.accepted, finished}, timeout=DEFAULT_RESULT_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
            )
            if finished in done:
                # Cancel before execution. A forged success report is not approval.
                finished.result()
                return {"success": 0, "error": "User cancelled the file change; no file was changed."}
            if review.accepted not in done:
                return failure

            # Execute the original node-bound method, including its mutation guards,
            # validation, snapshots and finalizer bookkeeping. Never trust client output.
            raw = await original.on_invoke_tool(tool_ctx, args_str)
            if isinstance(raw, str):
                raw = json.loads(raw)
            result = raw.model_dump() if hasattr(raw, "model_dump") else raw
            if not isinstance(result, dict):
                result = {"success": 1, "result": result}
            review.result.set_result(result)
            if not result.get("success"):
                return result

            # Accept & Edit holds the agent until the user finishes in the editor.
            decision = await finished
            if not isinstance(decision, dict) or decision.get("success") != 1:
                return {"success": 0, "error": "User cancelled after file execution; the file may already be changed."}
            if isinstance(decision.get("result"), str):
                result = {**result, "result": decision["result"]}
            return result
        except (TimeoutError, RuntimeError):
            return failure
        finally:
            context.pending.pop(call_id, None)
            if not review.result.done():
                review.result.set_result(failure)
            if not finished.done():
                finished.cancel()
            await asyncio.gather(finished, return_exceptions=True)

    return FunctionTool(
        name=original.name,
        description=original.description,
        params_json_schema=original.params_json_schema,
        on_invoke_tool=invoke,
        strict_json_schema=original.strict_json_schema,
    )


def apply_web_file_review(node: Any, *, context: WebFileReviewContext | None = None) -> None:
    """Install only for web/normal; descendants share the top-level HTTP review registry."""
    node.web_file_review_context = context or WebFileReviewContext(node.tool_channel)
    names = getattr(node, "file_review_tool_names", None)
    if not isinstance(names, set):
        names = node.file_review_tool_names = set()
    if getattr(node, "execution_mode", None) == "workflow":
        return
    for index, tool in enumerate(getattr(node, "tools", [])):
        if (
            isinstance(tool, FunctionTool)
            and tool.name in FILE_MUTATIONS
            and tool.name not in node.proxied_tool_names
            and tool.name not in names
        ):
            node.tools[index] = _review_tool(tool, node.web_file_review_context)
            names.add(tool.name)
