"""Web-only plan review; the server owns both the proposal and its mutation."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agents import FunctionTool

from datus.tools.func_tool.base import FuncToolResult, parse_tool_args
from datus.tools.proxy.tool_result_channel import DEFAULT_RESULT_TIMEOUT, ToolResultChannel

PLAN_WRITER = "write_semantic_model_plan_file"


@dataclass
class SemanticPlanReviewContext:
    channel: ToolResultChannel
    proxied_tool_names: set[str]
    previews: dict[str, dict[str, Any]] = field(default_factory=dict)


def apply_semantic_plan_review(node: Any, *, context: SemanticPlanReviewContext | None = None) -> None:
    """Use the existing tool-result channel for Accept/Cancel, never for content.

    Called only by the web chat adapter in normal mode. CLI, VS Code and
    auto/dangerous executions retain their ordinary tool/permission path.
    """
    # Even a parent without a writer must expose its delegated reviews through
    # the parent session's HTTP routes and SSE converter. Descendants inherit
    # this context, including through intermediate nodes without a writer.
    if context is None:
        context = SemanticPlanReviewContext(node.tool_channel, node.proxied_tool_names)
    node.semantic_plan_review_context = context
    previews = context.previews
    node.tool_review_previews = previews

    from datus.tools.proxy.ontology_file_review import apply_ontology_file_review

    apply_ontology_file_review(node, context=context)

    writer = getattr(node, "semantic_plan_file_tool", None)
    if writer is None:
        return

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
            decision = await context.channel.wait_for(call_id, timeout=DEFAULT_RESULT_TIMEOUT)
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
            # Mark the SSE call for web review; PermissionHooks still checks
            # server-side plan writes. Mutate the shared set rather than replace it.
            node.proxied_tool_names.add(PLAN_WRITER)
            context.proxied_tool_names.add(PLAN_WRITER)
            break
