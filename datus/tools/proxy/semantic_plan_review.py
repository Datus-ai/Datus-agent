"""Web-only plan review; the server owns both the proposal and its mutation."""

import asyncio
from typing import Any

from agents import FunctionTool

from datus.tools.func_tool.base import FuncToolResult, parse_tool_args
from datus.tools.proxy.tool_result_channel import DEFAULT_RESULT_TIMEOUT

PLAN_WRITER = "write_semantic_model_plan_file"


def apply_semantic_plan_review(node: Any) -> None:
    """Use the existing tool-result channel for Accept/Cancel, never for content.

    Called only by the web chat adapter in normal mode. CLI, VS Code and
    auto/dangerous executions retain their ordinary tool/permission path.
    """
    writer = getattr(node, "semantic_plan_file_tool", None)
    if writer is None:
        return

    previews = {}
    node.tool_review_previews = previews

    async def invoke(tool_ctx, args_str):
        args, _, error = parse_tool_args(args_str, tool_name=PLAN_WRITER)
        if error or set(args) != {"model_name", "plan_json"}:
            return FuncToolResult(success=0, error=error or "model_name and plan_json are required").model_dump()

        prepared = await asyncio.to_thread(writer.prepare, **args)
        if isinstance(prepared, FuncToolResult):
            return prepared.model_dump()

        call_id = tool_ctx.tool_call_id
        previews[call_id] = prepared.preview()
        try:
            decision = await node.tool_channel.wait_for(call_id, timeout=DEFAULT_RESULT_TIMEOUT)
            if not isinstance(decision, dict) or decision.get("success") != 1:
                return FuncToolResult(
                    success=0,
                    error="User cancelled the plan proposal. Do not author the model or retry unchanged. Wait for feedback, then submit a revised plan with this tool.",
                ).model_dump()

            result = await asyncio.to_thread(writer.commit, prepared)
            if result.success:
                result.result["review_status"] = "accepted"
            return result.model_dump()
        except (TimeoutError, RuntimeError):
            return FuncToolResult(
                success=0, error="Plan review expired or was stopped; no plan was written."
            ).model_dump()
        finally:
            previews.pop(call_id, None)

    for index, tool in enumerate(node.tools):
        if isinstance(tool, FunctionTool) and tool.name == PLAN_WRITER:
            node.tools[index] = FunctionTool(
                name=tool.name,
                description=tool.description,
                params_json_schema=tool.params_json_schema,
                on_invoke_tool=invoke,
                strict_json_schema=tool.strict_json_schema,
            )
            # PermissionHooks must defer to this web review, as it does for
            # filesystem proxies. Mutate the shared set rather than replace it.
            node.proxied_tool_names.add(PLAN_WRITER)
            break
