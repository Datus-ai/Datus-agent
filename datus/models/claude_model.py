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
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import anthropic
import httpx
from agents import Agent, RunContextWrapper, Usage
from agents.mcp import MCPServerStdio

from datus.configuration.agent_config import ModelConfig
from datus.models.mcp_utils import multiple_mcp_servers
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.node_models import SQLContext
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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

    def __init__(self, model_config: ModelConfig, **kwargs):
        # Initialize parent class (handles LiteLLM adapter, OpenAI client, etc.)
        super().__init__(model_config, **kwargs)

        # Claude-specific: check if we should use native Anthropic API
        self.use_native_api = getattr(model_config, "use_native_api", False)

        # Initialize native Anthropic client (always available for prompt caching)
        self._init_anthropic_client()

    def _get_api_key(self) -> str:
        """Get Anthropic API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
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

        self.anthropic_client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None,
            http_client=self.proxy_client,
        )

        # Wrap with LangSmith if available
        try:
            from langsmith.wrappers import wrap_anthropic

            self.anthropic_client = wrap_anthropic(self.anthropic_client)
        except ImportError:
            logger.debug("No langsmith wrapper available")

        logger.debug(f"Initialized Claude model: {self.model_name}, use_native_api={self.use_native_api}")

    @property
    def model_specs(self) -> Dict[str, Dict[str, int]]:
        """Model specifications for Claude models."""
        return {
            "claude-sonnet-4-5": {"context_length": 1048576, "max_tokens": 65536},
            "claude-opus-4-1": {"context_length": 200000, "max_tokens": 32000},
            "claude-opus-4": {"context_length": 200000, "max_tokens": 32000},
            "claude-sonnet-4": {"context_length": 1048576, "max_tokens": 65536},
            "claude-3-7-sonnet": {"context_length": 200000, "max_tokens": 128000},
        }

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """Generate response using native Anthropic API.

        Override parent class to use native Anthropic client since
        Anthropic API doesn't support OpenAI-compatible format.

        Args:
            prompt: The input prompt (str or list of messages)
            enable_thinking: Enable thinking mode (not supported by Claude, ignored)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
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
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                messages=filtered_messages,
                system=system_message if system_message else anthropic.NOT_GIVEN,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
            )

            if response.content:
                return response.content[0].text
            return ""

        except Exception as e:
            logger.error(f"Error generating with Anthropic: {str(e)}")
            raise

    async def generate_with_mcp(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate response using native Anthropic API with MCP servers.

        This method uses the native Anthropic client directly, which enables
        prompt caching for better performance with repeated prompts.

        Args:
            prompt: The input prompt
            mcp_servers: Dictionary of MCP servers
            instruction: System instruction
            output_type: Expected output type
            max_turns: Maximum conversation turns
            **kwargs: Additional parameters

        Returns:
            Dict with content and sql_contexts
        """
        # Custom JSON encoder for special types
        self._setup_custom_json_encoder()

        logger.debug(f"Using native Anthropic API with prompt caching, model: {self.model_name}")
        try:
            all_tools = []

            # Use context manager to manage multiple MCP servers
            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                # Get all tools
                for server_name, connected_server in connected_servers.items():
                    try:
                        # Create minimal agent and run context for the new interface
                        agent = Agent(name="mcp-tools-agent")
                        run_context = RunContextWrapper(context=None, usage=Usage())
                        mcp_tools = await connected_server.list_tools(run_context, agent)
                        all_tools.extend(mcp_tools)
                        logger.info(f"Retrieved {len(mcp_tools)} tools from {server_name}")

                    except Exception as e:
                        logger.error(f"Error getting tools from {server_name}: {str(e)}")
                        continue

                logger.info(f"Retrieved {len(all_tools)} total tools from MCP servers")

                tools = convert_tools_for_anthropic(all_tools)
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"{instruction}\n\n{prompt}"}],
                    }
                ]
                tool_call_cache = {}
                sql_contexts = []
                final_content = ""

                # Execute conversation loop
                for turn in range(max_turns):
                    logger.debug(f"Turn {turn + 1}/{max_turns}")

                    response = self.anthropic_client.messages.create(
                        model=self.model_name,
                        system=instruction,
                        messages=wrap_prompt_cache(messages),
                        tools=tools,
                        max_tokens=kwargs.get("max_tokens", 20480),
                        temperature=kwargs.get("temperature", 0.7),
                    )

                    message = response.content

                    # If no tool calls, conversation is complete
                    if not any(block.type == "tool_use" for block in message):
                        final_content = "\n".join([block.text for block in message if block.type == "text"])
                        logger.debug("No tool calls, conversation completed")
                        break

                    for block in message:
                        if block.type == "tool_use":
                            logger.debug(f"Executing tool: {block.name}")
                            tool_executed = False

                            for _, connected_server in connected_servers.items():
                                try:
                                    agent = Agent(name="mcp-claude-agent")
                                    run_context = RunContextWrapper(context=None, usage=Usage())
                                    tmp_tools = await connected_server.list_tools(run_context, agent)
                                    if any(tool.name == block.name for tool in tmp_tools):
                                        tool_result = await connected_server.call_tool(
                                            tool_name=block.name,
                                            arguments=json.loads(json.dumps(block.input)),
                                        )
                                        tool_call_cache[block.id] = tool_result
                                        tool_executed = True
                                        break
                                except Exception as e:
                                    logger.error(f"Error executing tool {block.name}: {str(e)}")
                                    continue

                            if not tool_executed:
                                logger.error(f"Tool {block.name} could not be executed")

                    for block in message:
                        content = []
                        if block.type == "text":
                            content.append({"type": "text", "content": block.text})
                        elif block.type == "tool_use":
                            content.append(
                                {
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                }
                            )
                            messages.append({"role": "assistant", "content": content})

                            if block.id in tool_call_cache:
                                sql_result = tool_call_cache[block.id].content[0].text
                                # Use "Error" to determine execution success
                                if "Error" not in sql_result and block.name == "read_query":
                                    sql_context = SQLContext(
                                        sql_query=block.input["query"],
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
                return {"content": final_content, "sql_contexts": sql_contexts}

        except Exception as e:
            logger.error(f"Error in generate_with_mcp: {str(e)}")
            raise

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
        # Use native Anthropic API for MCP if configured
        if self.use_native_api and mcp_servers and not tools:
            return await self.generate_with_mcp(
                prompt=prompt,
                mcp_servers=mcp_servers,
                instruction=instruction,
                output_type=output_type,
                max_turns=max_turns,
                **kwargs,
            )

        # Use parent class LiteLLM implementation
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

        Uses parent class LiteLLM implementation for streaming.
        Note: Native Anthropic streaming API can be added later if needed.
        """
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
