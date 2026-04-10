# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Claude Model - Anthropic Claude model implementation.

Inherits from OpenAICompatibleModel and adds Claude-specific features:
- Prompt caching via Anthropic's native API
- Optional native Anthropic API support (use_native_api config)
- Claude-specific model specifications
"""

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import anthropic
import httpx
from agents import Agent, RunContextWrapper, Usage
from agents.mcp import MCPServerStdio

from datus.configuration.agent_config import ModelConfig
from datus.models.mcp_utils import multiple_mcp_servers
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class _ToolResultPart:
    """A single part of a tool result (matches MCP tool result format)."""

    text: str


@dataclass
class _ToolResult:
    """Lightweight stand-in for MCP CallToolResult (`.content[0].text`)."""

    content: List[_ToolResultPart] = field(default_factory=list)


def wrap_prompt_cache(messages):
    """Wrap messages with Anthropic prompt cache control.

    Adds cache_control to the last content block for efficient prompt caching.
    """
    messages_copy = copy.deepcopy(messages)
    msg_size = len(messages_copy)
    content = messages_copy[msg_size - 1]["content"]
    cnt_size = len(content)
    if isinstance(content, list):
        content[cnt_size - 1]["cache_control"] = {"type": "ephemeral"}

    return messages_copy


def convert_tools_for_anthropic(mcp_tools):
    """Convert MCP tools to Anthropic tool format.

    Args:
        mcp_tools: List of MCP tools

    Returns:
        List of tools in Anthropic format with cache control
    """
    anthropic_tools = []

    for tool in mcp_tools:
        anthropic_tool = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
        }

        # Rename inputSchema's 'properties' to match Anthropic's convention if needed
        if "properties" in anthropic_tool["input_schema"]:
            for _, prop_value in anthropic_tool["input_schema"]["properties"].items():
                if "description" not in prop_value and "desc" in prop_value:
                    prop_value["description"] = prop_value.pop("desc")

        if hasattr(tool, "annotations") and tool.annotations:
            anthropic_tool["annotations"] = tool.annotations

        anthropic_tools.append(anthropic_tool)

    # Add tool cache to last tool (if any tools exist)
    if anthropic_tools:
        anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
    return anthropic_tools


class ClaudeModel(OpenAICompatibleModel):
    """
    Claude model implementation inheriting from OpenAICompatibleModel.

    Supports both:
    - LiteLLM-based API (default, via parent class)
    - Native Anthropic API (when use_native_api=True, enables prompt caching)
    """

    # Beta headers aligned with OpenClaw's current Anthropic OAuth path.
    # Keep this in sync with the OpenClaw PI_AI_OAUTH_ANTHROPIC_BETAS set when
    # using Claude Code setup-tokens (sk-ant-oat01-...).
    OAUTH_BETA_HEADERS = [
        "claude-code-20250219",
        "oauth-2025-04-20",
        "interleaved-thinking-2025-05-14",
        "prompt-caching-scope-2026-01-05",
    ]

    # Claude Code client headers — required for subscription tokens to be accepted.
    # These mimic the official Claude CLI client identity. The version string is
    # cosmetic (Anthropic validates via anthropic-beta + x-app, not user-agent version).
    # Update the version periodically to stay current if desired.
    OAUTH_CLIENT_HEADERS = {
        "user-agent": "claude-cli/2.1.75 (external, cli)",
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
    }

    OAUTH_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

    def __init__(self, model_config: ModelConfig, **kwargs):
        # Initialize parent class (handles LiteLLM adapter, OpenAI client, etc.)
        super().__init__(model_config, **kwargs)

        # Claude-specific: check if we should use native Anthropic API
        self.use_native_api = getattr(model_config, "use_native_api", False)

        # Detect OAuth subscription token via auth_type config (canonical source)
        self._is_oauth_token = getattr(model_config, "auth_type", "api_key") == "subscription"

        # OAuth tokens must use native API to avoid LiteLLM's x-api-key interference
        if self._is_oauth_token:
            self.use_native_api = True

        # Initialize native Anthropic client (always available for prompt caching)
        self._init_anthropic_client()

    def _get_api_key(self) -> str:
        """Get Anthropic API key from config or environment."""
        if self.model_config.auth_type == "subscription":
            from datus.auth.claude_credential import get_claude_subscription_token

            token, _source = get_claude_subscription_token(self.model_config.api_key)
            return token
        api_key = self.model_config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise DatusException(ErrorCode.MODEL_AUTHENTICATION_ERROR)
        return api_key

    def _get_base_url(self) -> Optional[str]:
        """Get Anthropic base URL from config."""
        return self.model_config.base_url or "https://api.anthropic.com"

    def _init_anthropic_client(self):
        """Initialize native Anthropic client for prompt caching and native API support."""
        # Optional proxy configuration
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        self.proxy_client = None

        if proxy_url:
            self.proxy_client = httpx.Client(
                transport=httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url)),
                timeout=60.0,
            )

        # Build headers: merge config default_headers with OAuth headers if needed
        extra_headers = dict(self.default_headers) if self.default_headers else {}
        if self._is_oauth_token:
            extra_headers["anthropic-beta"] = ",".join(self.OAUTH_BETA_HEADERS)
            extra_headers.update(self.OAUTH_CLIENT_HEADERS)
            logger.debug("Using OAuth subscription token — injecting beta + client headers")

        if self._is_oauth_token:
            # Use auth_token (Bearer auth) instead of api_key (x-api-key) for OAuth tokens
            self.anthropic_client = anthropic.Anthropic(
                auth_token=self.api_key,
                api_key=None,
                base_url=self.base_url if self.base_url else None,
                http_client=self.proxy_client,
                default_headers=extra_headers or None,
            )
        else:
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                http_client=self.proxy_client,
                default_headers=extra_headers or None,
            )

        # Wrap with LangSmith if available
        try:
            from langsmith.wrappers import wrap_anthropic

            self.anthropic_client = wrap_anthropic(self.anthropic_client)
        except ImportError:
            logger.debug("No langsmith wrapper available")

        logger.debug(f"Initialized Claude model: {self.model_name}, use_native_api={self.use_native_api}")

    def _inject_oauth_headers(self, kwargs: dict) -> dict:
        """Inject OAuth beta + client headers into kwargs for LiteLLM calls if using subscription token."""
        if self._is_oauth_token:
            existing = kwargs.get("extra_headers", {})
            kwargs["extra_headers"] = {
                **existing,
                "anthropic-beta": ",".join(self.OAUTH_BETA_HEADERS),
                "Authorization": f"Bearer {self.api_key}",
                **self.OAUTH_CLIENT_HEADERS,
            }
        return kwargs

    def _build_system_param(self, system_message: str = "") -> Any:
        """Build Anthropic system param, injecting Claude Code identity for OAuth tokens."""
        if self._is_oauth_token:
            system_blocks: list[dict[str, str]] = [
                {
                    "type": "text",
                    "text": self.OAUTH_SYSTEM_IDENTITY,
                }
            ]
            if system_message:
                system_blocks.append(
                    {
                        "type": "text",
                        "text": system_message,
                    }
                )
            return system_blocks

        return system_message if system_message else anthropic.NOT_GIVEN

    def _anthropic_messages_create(self, **kwargs):
        """Call the correct Anthropic Messages endpoint for the current auth mode."""
        if self._is_oauth_token:
            return self.anthropic_client.beta.messages.create(**kwargs)
        return self.anthropic_client.messages.create(**kwargs)

    def _diagnose_oauth_401(self, original_error: Exception) -> None:
        """Diagnose a 401 error for OAuth subscription tokens and raise a specific exception.

        Checks whether the token is expired (actionable: re-run setup-token) or
        rejected for other reasons (revoked, subscription inactive, corrupted).
        Only acts when ``_is_oauth_token`` is True; otherwise returns silently so
        the caller can re-raise the original error unchanged.
        """
        if not self._is_oauth_token:
            return

        # Try to determine expiry from the credentials file
        credentials_path = Path.home() / ".claude" / ".credentials.json"
        if credentials_path.exists():
            try:
                data = json.loads(credentials_path.read_text(encoding="utf-8"))
                expires_at = data.get("claudeAiOauth", {}).get("expiresAt")
                if expires_at and int(expires_at) / 1000 < time.time():
                    logger.warning("Claude subscription token has expired (expiresAt check)")
                    raise DatusException(ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_EXPIRED) from original_error
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        # Token is not expired (or no expiry info) — something else is wrong
        logger.warning("Claude subscription token rejected (401) but token is not expired")
        raise DatusException(ErrorCode.CLAUDE_SUBSCRIPTION_AUTH_FAILED) from original_error

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """Generate response using LiteLLM (default) or native Anthropic API.

        Default uses LiteLLM path for consistent api_key/base_url handling across
        all code paths (generate, generate_with_json_output, generate_with_tools_stream).
        Set use_native_api=True in model config to use native Anthropic client instead.

        Args:
            prompt: The input prompt (str or list of messages)
            enable_thinking: Enable thinking mode (not supported by Claude, ignored)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        if not self.use_native_api:
            # Claude API does not allow both temperature and top_p simultaneously.
            # Explicitly override top_p to None so the parent's default top_p=1.0
            # is not added to the request — LiteLLM omits None-valued parameters.
            kwargs["top_p"] = None
            self._inject_oauth_headers(kwargs)
            try:
                return super().generate(prompt, enable_thinking=enable_thinking, **kwargs)
            except DatusException as e:
                if self._is_oauth_token and e.code == ErrorCode.MODEL_AUTHENTICATION_ERROR:
                    self._diagnose_oauth_401(e)
                raise

        # Native Anthropic client path (only when use_native_api=True)
        # Build messages
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # Extract system message if present
        system_message = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        try:
            response = self._anthropic_messages_create(
                model=self.model_name,
                messages=filtered_messages,
                system=self._build_system_param(system_message),
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
            )

            if response.content:
                return response.content[0].text
            return ""

        except anthropic.AuthenticationError as e:
            self._diagnose_oauth_401(e)  # raises specific DatusException for OAuth tokens
            raise
        except Exception as e:
            logger.error(f"Error generating with Anthropic: {str(e)}")
            raise

    async def _generate_with_mcp_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        func_tools: Optional[List[Any]] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        interrupt_controller=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Async generator: native Anthropic API with real-time tool call ActionHistory.

        Yields ActionHistory objects for each tool call (PROCESSING then SUCCESS/FAILURE),
        and a final ASSISTANT action containing the result dict.
        """
        # Custom JSON encoder for special types
        self._setup_custom_json_encoder()

        logger.debug(f"Using native Anthropic API with prompt caching, model: {self.model_name}")
        try:
            all_tools = []

            # Use context manager to manage multiple MCP servers
            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                # Get all tools and build tool-name-to-server mapping once
                tool_server_map = {}  # tool_name -> connected_server
                for server_name, connected_server in connected_servers.items():
                    try:
                        agent = Agent(name="mcp-tools-agent")
                        run_context = RunContextWrapper(context=None, usage=Usage())
                        mcp_tools = await connected_server.list_tools(run_context, agent)
                        for tool in mcp_tools:
                            if tool.name in tool_server_map:
                                logger.warning(
                                    f"Duplicate MCP tool name '{tool.name}' from server '{server_name}', "
                                    f"overwriting previous mapping"
                                )
                            tool_server_map[tool.name] = connected_server
                        all_tools.extend(mcp_tools)
                        logger.info(f"Retrieved {len(mcp_tools)} tools from {server_name}")

                    except Exception as e:
                        logger.error(f"Error getting tools from {server_name}: {str(e)}")
                        continue

                logger.info(f"Retrieved {len(all_tools)} total tools from MCP servers")

                tools = convert_tools_for_anthropic(all_tools)

                # Convert and merge function tools (Agent SDK FunctionTool objects)
                func_tool_map = {}
                if func_tools:
                    for ft in func_tools:
                        tools.append(
                            {
                                "name": ft.name,
                                "description": ft.description or "",
                                "input_schema": ft.params_json_schema,
                            }
                        )
                        func_tool_map[ft.name] = ft
                    # Re-apply cache control on last tool
                    if tools:
                        for t in tools:
                            t.pop("cache_control", None)
                        tools[-1]["cache_control"] = {"type": "ephemeral"}
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"{instruction}\n\n{prompt}"}],
                    }
                ]
                tool_call_cache = {}
                sql_contexts = []
                final_content = ""
                # Accumulate token usage across all turns
                cumulative_input_tokens = 0
                cumulative_output_tokens = 0
                cache_creation_tokens = 0
                cache_read_tokens = 0
                last_call_input_tokens = 0

                # Execute conversation loop
                turn = -1
                for turn in range(max_turns):
                    if interrupt_controller and interrupt_controller.is_interrupted:
                        from datus.cli.execution_state import ExecutionInterrupted

                        raise ExecutionInterrupted("Interrupted by user")

                    logger.debug(f"Turn {turn + 1}/{max_turns}")

                    response = self._anthropic_messages_create(
                        model=self.model_name,
                        system=self._build_system_param(instruction),
                        messages=wrap_prompt_cache(messages),
                        tools=tools,
                        max_tokens=kwargs.get("max_tokens", 20480),
                        temperature=kwargs.get("temperature", 0.7),
                    )

                    # Track token usage from this turn
                    if hasattr(response, "usage") and response.usage:
                        cumulative_input_tokens += getattr(response.usage, "input_tokens", 0)
                        cumulative_output_tokens += getattr(response.usage, "output_tokens", 0)
                        cache_creation_tokens += getattr(response.usage, "cache_creation_input_tokens", 0)
                        cache_read_tokens += getattr(response.usage, "cache_read_input_tokens", 0)
                        last_call_input_tokens = getattr(response.usage, "input_tokens", 0)

                    message = response.content

                    # If no tool calls, conversation is complete
                    if not any(block.type == "tool_use" for block in message):
                        final_content = "\n".join([block.text for block in message if block.type == "text"])
                        logger.debug("No tool calls, conversation completed")
                        break

                    for block in message:
                        if block.type == "tool_use":
                            if interrupt_controller and interrupt_controller.is_interrupted:
                                from datus.cli.execution_state import ExecutionInterrupted

                                raise ExecutionInterrupted("Interrupted by user")

                            logger.debug(f"Executing tool: {block.name}")
                            args_str = json.dumps(block.input, ensure_ascii=False)[:80]

                            # Yield PROCESSING action for real-time tool call display
                            start_action = ActionHistory(
                                action_id=block.id,
                                role=ActionRole.TOOL,
                                messages=f"Tool call: {block.name}('{args_str}...')",
                                action_type=block.name,
                                input={"function_name": block.name, "arguments": block.input},
                                output={},
                                status=ActionStatus.PROCESSING,
                            )
                            if action_history_manager is not None:
                                action_history_manager.add_action(start_action)
                            yield start_action

                            tool_executed = False

                            # Try function tools first
                            if block.name in func_tool_map:
                                try:
                                    ft = func_tool_map[block.name]
                                    run_context = RunContextWrapper(context=None, usage=Usage())
                                    result_val = await ft.on_invoke_tool(run_context, json.dumps(block.input))
                                    # Ensure result is a string (Anthropic API requires string content)
                                    result_str = result_val if isinstance(result_val, str) else json.dumps(result_val)
                                    # Wrap in object matching MCP tool result format
                                    func_result = _ToolResult(content=[_ToolResultPart(text=result_str)])
                                    tool_call_cache[block.id] = func_result
                                    tool_executed = True
                                except Exception as e:
                                    logger.error(f"Error executing function tool {block.name}: {str(e)}")

                            # Fall back to MCP servers via pre-built mapping
                            if not tool_executed:
                                target_server = tool_server_map.get(block.name)
                                if target_server:
                                    try:
                                        tool_result = await target_server.call_tool(
                                            tool_name=block.name,
                                            arguments=dict(block.input)
                                            if isinstance(block.input, dict)
                                            else block.input,
                                        )
                                        tool_call_cache[block.id] = tool_result
                                        tool_executed = True
                                    except Exception as e:
                                        logger.error(f"Error executing tool {block.name}: {str(e)}")

                            if not tool_executed:
                                logger.error(f"Tool {block.name} could not be executed")

                            # Yield SUCCESS/FAILURE action for real-time tool call display
                            result_text = ""
                            if block.id in tool_call_cache:
                                result_text = tool_call_cache[block.id].content[0].text
                            result_summary = (
                                self._format_tool_result(result_text, block.name) if tool_executed else "Failed"
                            )
                            complete_action = ActionHistory(
                                action_id=f"complete_{block.id}",
                                role=ActionRole.TOOL,
                                messages=f"Tool call: {block.name}('{args_str}...')",
                                action_type=block.name,
                                input={"function_name": block.name, "arguments": block.input},
                                output={
                                    "success": tool_executed,
                                    "raw_output": result_text,
                                    "summary": result_summary,
                                    "status_message": result_summary,
                                },
                                status=ActionStatus.SUCCESS if tool_executed else ActionStatus.FAILED,
                            )
                            complete_action.end_time = datetime.now()
                            if action_history_manager is not None:
                                action_history_manager.add_action(complete_action)
                            yield complete_action

                    # Build assistant message content from all blocks
                    content = []
                    tool_use_blocks = []
                    for block in message:
                        if block.type == "text":
                            content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            content.append(
                                {
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                }
                            )
                            tool_use_blocks.append(block)

                    if content:
                        messages.append({"role": "assistant", "content": content})

                    for block in tool_use_blocks:
                        if block.id in tool_call_cache:
                            sql_result = tool_call_cache[block.id].content[0].text
                            # Use "Error" to determine execution success
                            if "Error" not in sql_result and block.name == "read_query":
                                sql_query = block.input.get("query") or block.input.get("sql", "")
                                sql_context = SQLContext(
                                    sql_query=sql_query,
                                    sql_return=sql_result,
                                    row_count=None,
                                )
                                sql_contexts.append(sql_context)
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": block.id,
                                            "content": sql_result,
                                        }
                                    ],
                                }
                            )
                        else:
                            error_message = f"Tool {block.name} execution failed"
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": block.id,
                                            "content": error_message,
                                        }
                                    ],
                                }
                            )

                logger.debug("Agent execution completed")
                total_tokens = cumulative_input_tokens + cumulative_output_tokens
                cached_tokens = cache_read_tokens
                usage_info = {
                    "requests": turn + 1,
                    "input_tokens": cumulative_input_tokens,
                    "output_tokens": cumulative_output_tokens,
                    "total_tokens": total_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "reasoning_tokens": 0,
                    "cache_hit_rate": (
                        round(cached_tokens / cumulative_input_tokens, 3) if cumulative_input_tokens > 0 else 0
                    ),
                    "context_usage_ratio": (
                        round(total_tokens / self.context_length(), 3)
                        if self.context_length() and total_tokens > 0
                        else 0
                    ),
                    "last_call_input_tokens": last_call_input_tokens,
                }
                logger.debug(f"Native API cumulative token usage: {usage_info}")

                final_action = ActionHistory(
                    action_id=f"final_{uuid.uuid4().hex[:8]}",
                    role=ActionRole.ASSISTANT,
                    messages=str(final_content)[:200],
                    action_type="final_response",
                    input={},
                    output={
                        "raw_output": final_content,
                        "sql_contexts": sql_contexts,
                        "usage": usage_info,
                    },
                    status=ActionStatus.SUCCESS,
                )
                if action_history_manager is not None:
                    action_history_manager.add_action(final_action)
                yield final_action

        except anthropic.AuthenticationError as e:
            self._diagnose_oauth_401(e)
            raise
        except Exception as e:
            logger.error(f"Error in _generate_with_mcp_stream: {str(e)}")
            raise

    async def generate_with_mcp(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        func_tools: Optional[List[Any]] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> Dict:
        """Non-streaming wrapper: consumes _generate_with_mcp_stream and returns result dict."""
        result: Dict = {"content": "", "sql_contexts": []}
        async for action in self._generate_with_mcp_stream(
            prompt=prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            func_tools=func_tools,
            action_history_manager=action_history_manager,
            **kwargs,
        ):
            if action.role == ActionRole.ASSISTANT and action.action_type == "final_response":
                result = {
                    "content": action.output.get("raw_output", ""),
                    "sql_contexts": action.output.get("sql_contexts", []),
                }
        return result

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Any]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        strict_json_schema: bool = True,
        max_turns: int = 10,
        session=None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        **kwargs,
    ) -> Dict:
        """Generate response with tool support.

        Routes to native Anthropic API when use_native_api=True and mcp_servers provided,
        otherwise uses parent class LiteLLM implementation.
        """
        # Use native Anthropic API when configured (required for OAuth subscription tokens
        # since LiteLLM sends x-api-key which is incompatible with Bearer auth)
        if self.use_native_api and (mcp_servers or self._is_oauth_token):
            return await self.generate_with_mcp(
                prompt=prompt,
                mcp_servers=mcp_servers or {},
                instruction=instruction,
                output_type=output_type,
                max_turns=max_turns,
                func_tools=tools,
                action_history_manager=action_history_manager,
                **kwargs,
            )

        # Use parent class LiteLLM implementation
        self._inject_oauth_headers(kwargs)
        try:
            return await super().generate_with_tools(
                prompt=prompt,
                tools=tools,
                mcp_servers=mcp_servers,
                instruction=instruction,
                output_type=output_type,
                strict_json_schema=strict_json_schema,
                max_turns=max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=hooks,
                **kwargs,
            )
        except DatusException as e:
            if self._is_oauth_token and e.code == ErrorCode.MODEL_AUTHENTICATION_ERROR:
                self._diagnose_oauth_401(e)
            raise

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        tools: Optional[List[Any]] = None,
        instruction: str = "",
        output_type: type = str,
        strict_json_schema: bool = True,
        max_turns: int = 10,
        session=None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate response with streaming and tool support.

        Routes to native Anthropic API for OAuth subscription tokens,
        otherwise uses parent class LiteLLM implementation.
        """
        # For OAuth tokens, use native path (LiteLLM sends x-api-key which is incompatible)
        # Directly iterate the async generator for real-time tool call display
        if self.use_native_api and self._is_oauth_token:
            if action_history_manager is None:
                action_history_manager = ActionHistoryManager()
            async for action in self._generate_with_mcp_stream(
                prompt=prompt,
                mcp_servers=mcp_servers or {},
                instruction=instruction,
                output_type=output_type,
                max_turns=max_turns,
                func_tools=tools,
                action_history_manager=action_history_manager,
                interrupt_controller=kwargs.pop("interrupt_controller", None),
                **kwargs,
            ):
                yield action
            return

        self._inject_oauth_headers(kwargs)
        try:
            async for action in super().generate_with_tools_stream(
                prompt=prompt,
                mcp_servers=mcp_servers,
                tools=tools,
                instruction=instruction,
                output_type=output_type,
                strict_json_schema=strict_json_schema,
                max_turns=max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=hooks,
                **kwargs,
            ):
                yield action
        except DatusException as e:
            if self._is_oauth_token and e.code == ErrorCode.MODEL_AUTHENTICATION_ERROR:
                self._diagnose_oauth_401(e)
            raise

    async def aclose(self):
        """Async cleanup of resources."""
        # Close parent class resources
        # Note: Parent class doesn't have aclose, but we keep this for future compatibility

        if hasattr(self, "proxy_client") and self.proxy_client:
            try:
                self.proxy_client.close()
                logger.debug("Proxy client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing proxy client: {e}")

        if hasattr(self, "anthropic_client") and hasattr(self.anthropic_client, "close"):
            try:
                self.anthropic_client.close()
                logger.debug("Anthropic client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing anthropic client: {e}")

    def close(self):
        """Synchronous close for backward compatibility."""
        if hasattr(self, "proxy_client") and self.proxy_client:
            try:
                self.proxy_client.close()
            except Exception as e:
                logger.warning(f"Error closing proxy client: {e}")

        if hasattr(self, "anthropic_client") and hasattr(self.anthropic_client, "close"):
            try:
                self.anthropic_client.close()
            except Exception as e:
                logger.warning(f"Error closing anthropic client: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error in ClaudeModel destructor: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
