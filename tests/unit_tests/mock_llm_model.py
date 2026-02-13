# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
MockLLMModel for AgenticNode unit testing.

This module provides a mock LLM model that simulates only the LLM decision-making
while executing real tools. It replaces the openai-agents SDK Runner loop but
invokes actual tool functions, ensuring all non-LLM components are tested for real.

Usage:
    responses = [
        MockLLMResponse(
            tool_calls=[
                MockToolCall(name="list_tables", arguments="{}"),
                MockToolCall(name="describe_table", arguments='{"table_name": "orders"}'),
            ],
            content='{"sql": "SELECT * FROM orders", "explanation": "Query all orders"}',
        ),
    ]
    model = MockLLMModel(responses=responses)
    # When generate_with_tools_stream() is called with real tools,
    # list_tables and describe_table will be ACTUALLY EXECUTED.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from agents import SQLiteSession, Tool
from agents.mcp import MCPServerStdio

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class MockToolCall:
    """Represents an LLM's decision to call a tool.

    The tool will be ACTUALLY EXECUTED with the given arguments. The output
    comes from real tool execution, not from a pre-defined value.

    Attributes:
        name: Tool function name (e.g., "list_tables", "execute_sql")
        arguments: Tool arguments as JSON string or dict
    """

    name: str
    arguments: Union[str, dict] = ""

    @property
    def arguments_str(self) -> str:
        if isinstance(self.arguments, dict):
            return json.dumps(self.arguments)
        return self.arguments


@dataclass
class MockLLMResponse:
    """Represents a complete LLM response turn.

    Each MockLLMResponse represents one complete interaction cycle with the LLM,
    which may include zero or more tool calls followed by a final text response.
    Tool calls are executed for real against actual tool implementations.

    Attributes:
        tool_calls: List of tool calls the LLM "decides" to make (executed for real)
        content: Final text response from the LLM (after all tool calls)
        thinking: Optional thinking/reasoning content (for models that support it)
    """

    content: str = ""
    tool_calls: List[MockToolCall] = field(default_factory=list)
    thinking: Optional[str] = None


class _MockHookContext:
    """Minimal context object for hooks invocation during mock tool execution.

    Provides the tool_arguments attribute that PermissionHooks uses
    to parse tool arguments for permission checking and display.
    """

    def __init__(self, tool_arguments: str = "{}"):
        self.tool_arguments = tool_arguments


class MockLLMModel(LLMBaseModel):
    """Mock LLM model that simulates LLM decisions but executes real tools.

    Only the LLM's decision-making is mocked (which tools to call, with what params,
    and the final text response). All tool executions are REAL - actual tool functions
    are invoked with the specified arguments.

    The yield pattern matches openai_compatible.py:
    1. For each tool_call:
       a. Yield TOOL/PROCESSING ActionHistory (tool call start)
       b. Actually invoke tool.on_invoke_tool(ctx, arguments)
       c. Yield TOOL/SUCCESS ActionHistory (tool call complete with REAL output)
    2. If thinking content exists:
       - Yield ASSISTANT/SUCCESS ActionHistory with thinking text
    3. Final response:
       - Yield ASSISTANT/SUCCESS ActionHistory with content as raw_output

    Args:
        responses: List of MockLLMResponse objects to yield in sequence.
                   Each call to generate_with_tools_stream() consumes one response.
        model_config: Optional ModelConfig. If None, a default mock config is created.
    """

    def __init__(
        self,
        responses: Optional[List[MockLLMResponse]] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        if model_config is None:
            model_config = ModelConfig(
                type="mock",
                api_key="mock-api-key",
                model="mock-model",
                base_url="http://localhost:0",
            )
        super().__init__(model_config)
        self._responses = list(responses or [])
        self._response_index = 0
        self._call_history: List[Dict[str, Any]] = []
        self._tool_results: List[Dict[str, Any]] = []
        # Override session manager to avoid path_manager dependency
        self._session_manager = MockSessionManager()

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Access recorded call history for test assertions."""
        return self._call_history

    @property
    def tool_results(self) -> List[Dict[str, Any]]:
        """Access recorded tool execution results for test assertions."""
        return self._tool_results

    def reset(self, responses: Optional[List[MockLLMResponse]] = None):
        """Reset the mock model state, optionally with new responses."""
        if responses is not None:
            self._responses = list(responses)
        self._response_index = 0
        self._call_history.clear()
        self._tool_results.clear()

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """Generate a simple text response (non-streaming, non-tool)."""
        self._call_history.append({"method": "generate", "prompt": prompt, "kwargs": kwargs})
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            return response.content
        return ""

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """Generate a JSON response."""
        self._call_history.append({"method": "generate_with_json_output", "prompt": prompt, "kwargs": kwargs})
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            try:
                return json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                return {"content": response.content}
        return {}

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        **kwargs,
    ) -> Dict:
        """Generate response with tools (non-streaming). Tools are executed for real."""
        self._call_history.append(
            {
                "method": "generate_with_tools",
                "prompt": prompt,
                "instruction": instruction,
                "tools": [t.name for t in tools] if tools else [],
                "kwargs": kwargs,
            }
        )
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1

            # Execute real tools
            for tc in response.tool_calls:
                await self._execute_real_tool(tc, tools)

            return {
                "content": response.content,
                "sql_contexts": [],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                "model": self.model_config.model,
            }
        return {"content": "", "sql_contexts": [], "usage": {}, "model": self.model_config.model}

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate response with streaming and tool support.

        This is the core method that AgenticNode.execute_stream() calls.
        Tool calls are ACTUALLY EXECUTED against real tool implementations.
        Only the LLM's decision (which tool + arguments + final response) is mocked.
        """
        self._call_history.append(
            {
                "method": "generate_with_tools_stream",
                "prompt": prompt,
                "instruction": instruction,
                "tools": [t.name for t in tools] if tools else [],
                "kwargs": kwargs,
            }
        )

        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        if self._response_index >= len(self._responses):
            empty_action = ActionHistory(
                action_id=f"assistant_{uuid.uuid4().hex[:8]}",
                role=ActionRole.ASSISTANT,
                messages="No more mock responses available",
                action_type="response",
                input={},
                output={"raw_output": "", "usage": _mock_usage()},
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(empty_action)
            yield empty_action
            return

        response = self._responses[self._response_index]
        self._response_index += 1

        # Build tool lookup map
        tool_map: Dict[str, Tool] = {}
        if tools:
            for t in tools:
                tool_map[t.name] = t

        # Phase 1: Execute tool calls FOR REAL (with hooks support)
        for tool_call in response.tool_calls:
            call_id = f"call_{uuid.uuid4().hex[:12]}"

            # Parse arguments for display
            try:
                args_dict = json.loads(tool_call.arguments_str) if tool_call.arguments_str else {}
                args_str = str(args_dict)[:80]
            except (json.JSONDecodeError, TypeError):
                args_str = str(tool_call.arguments)[:80]

            # Resolve the real tool object for hook invocation
            real_tool = tool_map.get(tool_call.name)

            # 1a. Invoke hooks.on_tool_start (mirrors agents SDK Runner behavior)
            if hooks and real_tool:
                hook_ctx = _MockHookContext(tool_arguments=tool_call.arguments_str)
                await hooks.on_tool_start(hook_ctx, None, real_tool)

            # 1b. TOOL/PROCESSING - tool call start
            start_action = ActionHistory(
                action_id=call_id,
                role=ActionRole.TOOL,
                messages=f"Tool call: {tool_call.name}('{args_str}...')",
                action_type=tool_call.name,
                input={"function_name": tool_call.name, "arguments": tool_call.arguments_str},
                output={},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(start_action)
            yield start_action

            # 1c. ACTUALLY EXECUTE the tool
            tool_output = await self._execute_real_tool(tool_call, tools)

            # 1d. TOOL/SUCCESS - tool call complete with REAL output
            complete_action = ActionHistory(
                action_id=f"complete_{call_id}",
                role=ActionRole.TOOL,
                messages=f"Tool call: {tool_call.name}('{args_str}...')",
                action_type=tool_call.name,
                input={"function_name": tool_call.name, "arguments": tool_call.arguments_str},
                output={
                    "success": True,
                    "raw_output": tool_output,
                    "summary": "Success",
                    "status_message": "Success",
                },
                status=ActionStatus.SUCCESS,
            )
            complete_action.end_time = datetime.now()
            action_history_manager.add_action(complete_action)
            yield complete_action

            # 1e. Invoke hooks.on_tool_end (mirrors agents SDK Runner behavior)
            # Pass raw tool_output (dict for FuncTool) instead of str() to match real Runner
            if hooks and real_tool:
                hook_ctx = _MockHookContext(tool_arguments=tool_call.arguments_str)
                await hooks.on_tool_end(hook_ctx, None, real_tool, tool_output)

        # Phase 2: Yield thinking action if present
        if response.thinking:
            thinking_text = response.thinking.strip()
            thinking_display = thinking_text if len(thinking_text) <= 200 else f"{thinking_text[:200]}..."
            thinking_action = ActionHistory(
                action_id=f"assistant_{uuid.uuid4().hex[:8]}",
                role=ActionRole.ASSISTANT,
                messages=f"Thinking: {thinking_display}",
                action_type="response",
                input={},
                output={"raw_output": thinking_text},
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(thinking_action)
            yield thinking_action

        # Phase 3: Yield final assistant response with usage info
        final_action = ActionHistory(
            action_id=f"assistant_{uuid.uuid4().hex[:8]}",
            role=ActionRole.ASSISTANT,
            messages=response.content[:200] if len(response.content) > 200 else response.content,
            action_type="response",
            input={},
            output={
                "raw_output": response.content,
                "usage": _mock_usage(),
            },
            status=ActionStatus.SUCCESS,
        )
        action_history_manager.add_action(final_action)
        yield final_action

        # Phase 4: Invoke hooks.on_llm_end (mirrors agents SDK Runner behavior)
        # This is needed for PlanModeHooks which defers plan display until LLM finishes
        if hooks:
            await hooks.on_llm_end(None, None, response.content)

    async def _execute_real_tool(self, tool_call: MockToolCall, tools: Optional[List[Tool]]) -> Any:
        """Find the matching tool and execute it for real.

        Args:
            tool_call: The mock tool call specifying name and arguments
            tools: List of real Tool objects passed to generate_with_tools_stream

        Returns:
            The actual output from the tool execution
        """
        if not tools:
            error_msg = f"No tools provided, cannot execute {tool_call.name}"
            logger.warning(error_msg)
            result = {"success": 0, "error": error_msg, "result": None}
            self._tool_results.append({"tool": tool_call.name, "output": result, "executed": False})
            return result

        # Find matching tool
        matching_tool = None
        for t in tools:
            if t.name == tool_call.name:
                matching_tool = t
                break

        if matching_tool is None:
            error_msg = f"Tool '{tool_call.name}' not found in available tools: {[t.name for t in tools]}"
            logger.warning(error_msg)
            result = {"success": 0, "error": error_msg, "result": None}
            self._tool_results.append({"tool": tool_call.name, "output": result, "executed": False})
            return result

        # Execute the real tool
        try:
            # FunctionTool.on_invoke_tool expects (ctx, args_str)
            # Create a minimal context - most tools don't use it
            result = await matching_tool.on_invoke_tool(None, tool_call.arguments_str)
            logger.debug(f"Tool '{tool_call.name}' executed successfully: {str(result)[:200]}")
            self._tool_results.append({"tool": tool_call.name, "output": result, "executed": True})
            return result
        except Exception as e:
            error_msg = f"Tool '{tool_call.name}' execution failed: {str(e)}"
            logger.error(error_msg)
            result = {"success": 0, "error": error_msg, "result": None}
            self._tool_results.append({"tool": tool_call.name, "output": result, "executed": False, "error": str(e)})
            return result

    def token_count(self, prompt: str) -> int:
        """Return a fixed token count estimate (4 chars per token approximation)."""
        return max(1, len(prompt) // 4)

    def context_length(self) -> Optional[int]:
        """Return a reasonable fixed context length."""
        return 128000

    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context (no-op for mock)."""
        self.workflow = workflow
        self.current_node = current_node


class MockSessionManager:
    """Minimal session manager that avoids filesystem/path_manager dependencies.

    Uses in-memory SQLite databases for session storage, suitable for testing.
    """

    def __init__(self):
        self._sessions: Dict[str, SQLiteSession] = {}
        self._token_counts: Dict[str, int] = {}

    def create_session(self, session_id: str) -> SQLiteSession:
        """Create or get a session backed by in-memory SQLite."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SQLiteSession(session_id, db_path=":memory:")
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions.pop(session_id)
            self._sessions[session_id] = SQLiteSession(session_id, db_path=":memory:")

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._token_counts.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def update_session_tokens(self, session_id: str, total_tokens: int) -> None:
        self._token_counts[session_id] = total_tokens

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            return {"exists": False}
        return {
            "exists": True,
            "session_id": session_id,
            "total_tokens": self._token_counts.get(session_id, 0),
        }


def _mock_usage() -> Dict[str, Any]:
    """Generate mock token usage info matching the real implementation format."""
    return {
        "requests": 1,
        "input_tokens": 500,
        "output_tokens": 200,
        "total_tokens": 700,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cache_hit_rate": 0,
        "context_usage_ratio": 0.005,
    }


# -- Convenience builders for common test scenarios --


def build_simple_response(content: str, thinking: Optional[str] = None) -> MockLLMResponse:
    """Build a simple response with no tool calls."""
    return MockLLMResponse(content=content, thinking=thinking)


def build_sql_response(
    sql: str,
    tables: Optional[List[str]] = None,
    explanation: str = "Generated SQL query",
    tool_calls: Optional[List[MockToolCall]] = None,
) -> MockLLMResponse:
    """Build a response that returns SQL in the expected JSON format."""
    content = json.dumps(
        {
            "sql": sql,
            "tables": tables or [],
            "explanation": explanation,
        }
    )
    return MockLLMResponse(content=content, tool_calls=tool_calls or [])


def build_tool_then_response(
    tool_calls: List[MockToolCall],
    content: str,
    thinking: Optional[str] = None,
) -> MockLLMResponse:
    """Build a response with tool calls followed by a text response."""
    return MockLLMResponse(content=content, tool_calls=tool_calls, thinking=thinking)
