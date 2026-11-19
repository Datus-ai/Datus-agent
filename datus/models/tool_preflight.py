# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0.

"""Return expected pre-execution rejections through the model's tool protocol."""

import json
from copy import copy
from typing import Any

from agents import Agent, FunctionTool
from agents.tool_guardrails import ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData
from agents.tracing import get_current_span

from datus.tools.permission.permission_hooks import PermissionDeniedException
from datus.utils.exceptions import InvalidToolArgument, ToolScopeDenied


async def tool_rejection(hooks: Any, context: Any, agent: Any, tool: Any) -> dict[str, Any] | None:
    start = getattr(hooks, "on_tool_start", None)
    if start is None:
        return None
    try:
        await start(context, agent, tool)
    except (InvalidToolArgument, PermissionDeniedException, ToolScopeDenied) as exc:
        reason = str(exc)
        if isinstance(exc, PermissionDeniedException):
            reason = exc.model_message
            if exc.user_declined:
                reason += " Respect this decision. Do not retry the declined operation using another tool."
        reason += " Continue with other authorized work and report any blocked step."
        return {"success": 0, "error": reason, "result": None}
    return None


class ToolPreflightAgent(Agent):
    """Check the final SDK tool list, including dynamically discovered MCP tools."""

    async def get_all_tools(self, run_context):
        tools = await super().get_all_tools(run_context)
        return [self._with_preflight(tool) if isinstance(tool, FunctionTool) else tool for tool in tools]

    @staticmethod
    def _with_preflight(tool: FunctionTool) -> FunctionTool:
        async def check(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
            rejection = await tool_rejection(data.agent.hooks, data.context, data.agent, tool)
            if rejection is None:
                # The SDK calls on_tool_start after the input guardrail.
                data.context._datus_tool_start_checked = True
                return ToolGuardrailFunctionOutput.allow()
            span = get_current_span()
            if span is not None:
                span.set_error({"message": "Tool call rejected", "data": {"tool_name": tool.name}})
            return ToolGuardrailFunctionOutput.reject_content(json.dumps(rejection))

        guarded = copy(tool)
        guarded.tool_input_guardrails = [*(tool.tool_input_guardrails or []), ToolInputGuardrail(check)]
        return guarded
