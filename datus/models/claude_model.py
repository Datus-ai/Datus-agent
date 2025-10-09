import asyncio
import copy
import json
import os
import re
import uuid
from datetime import date, datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import anthropic
import httpx
from agents import Agent, OpenAIChatCompletionsModel, RunContextWrapper, Runner, SQLiteSession, Usage
from agents.exceptions import MaxTurnsExceeded
from agents.mcp import MCPServerStdio
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import AnyUrl

from datus.models.base import LLMBaseModel
from datus.models.mcp_utils import multiple_mcp_servers
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import extract_json_str
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import create_openai_client

logger = get_logger(__name__)

# Monkey patch to fix ResponseTextDeltaEvent logprobs validation issue in openai-agents 0.3.2
try:
    from agents.models.chatcmpl_stream_handler import ResponseTextDeltaEvent

    # Modify the model field annotation to accept both list and None
    if hasattr(ResponseTextDeltaEvent, "__annotations__") and "logprobs" in ResponseTextDeltaEvent.__annotations__:
        # Make logprobs accept list or None
        ResponseTextDeltaEvent.__annotations__["logprobs"] = Union[list, None]
        # Rebuild the pydantic model with new annotations
        ResponseTextDeltaEvent.model_rebuild(force=True)
        logger.debug("Successfully patched ResponseTextDeltaEvent to accept logprobs as list or None")
except ImportError:
    logger.warning("Could not import ResponseTextDeltaEvent - patch not applied")
except Exception as e:
    logger.warning(f"Could not patch ResponseTextDeltaEvent: {e}")


def classify_api_error(error: Exception) -> tuple[ErrorCode, bool]:
    """Classify API errors and return error code and whether it's retryable."""
    error_msg = str(error).lower()

    if isinstance(error, APIError):
        # Handle specific HTTP status codes and error types
        if any(indicator in error_msg for indicator in ["overloaded", "529"]):
            return ErrorCode.MODEL_OVERLOADED, True
        elif any(indicator in error_msg for indicator in ["rate limit", "429"]):
            return ErrorCode.MODEL_RATE_LIMIT, True
        elif any(indicator in error_msg for indicator in ["401", "unauthorized", "authentication"]):
            return ErrorCode.MODEL_AUTHENTICATION_ERROR, False
        elif any(indicator in error_msg for indicator in ["403", "forbidden", "permission"]):
            return ErrorCode.MODEL_PERMISSION_ERROR, False
        elif any(indicator in error_msg for indicator in ["404", "not found"]):
            return ErrorCode.MODEL_NOT_FOUND, False
        elif any(indicator in error_msg for indicator in ["413", "too large", "request size"]):
            return ErrorCode.MODEL_REQUEST_TOO_LARGE, False
        elif any(indicator in error_msg for indicator in ["500", "internal", "server error"]):
            return ErrorCode.MODEL_API_ERROR, True
        elif any(indicator in error_msg for indicator in ["400", "bad request", "invalid"]):
            return ErrorCode.MODEL_INVALID_RESPONSE, False

    if isinstance(error, RateLimitError):
        return ErrorCode.MODEL_RATE_LIMIT, True

    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return ErrorCode.MODEL_CONNECTION_ERROR, True

    # Default to general request failure
    return ErrorCode.MODEL_REQUEST_FAILED, False


def wrap_prompt_cache(messages):
    messages_copy = copy.deepcopy(messages)
    msg_size = len(messages_copy)
    content = messages_copy[msg_size - 1]["content"]
    cnt_size = len(content)
    if isinstance(content, list):
        content[cnt_size - 1]["cache_control"] = {"type": "ephemeral"}

    return messages_copy


def convert_tools_for_anthropic(mcp_tools):
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

    # add tool cache
    tool_size = len(anthropic_tools)
    anthropic_tools[tool_size - 1]["cache_control"] = {"type": "ephemeral"}
    return anthropic_tools


class ClaudeModel(LLMBaseModel):
    """
    Implementation of the BaseModel for Claude's API.
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.api_base = model_config.base_url
        self.model_name = model_config.model
        # fix it, remove os.env with model_config
        self.api_key = model_config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        logger.debug(f"Using Claude model: {self.model_name} base Url: {self.api_base}")

        self.client = create_openai_client(OpenAI, api_key=self.api_key, base_url=self.api_base + "/v1")

        # Optional proxy configuration - only use if environment variable is set
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        self.proxy_client = None  # Store reference for cleanup

        if proxy_url:
            self.proxy_client = httpx.Client(
                transport=httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url)),
                timeout=60.0,
            )

        self.anthropic_client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.api_base if self.api_base else None,
            http_client=self.proxy_client,
        )
        try:
            from langsmith.wrappers import wrap_anthropic

            self.anthropic_client = wrap_anthropic(
                self.anthropic_client,
            )
        except ImportError:
            logger.debug("No langsmith wrapper available")
        # Session manager is initialized lazily in the base class via property

    def generate(self, prompt: Any, **kwargs) -> str:
        """Generate a response from the Claude model.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Merge default parameters with any provided kwargs
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 3000),
            "top_p": 1.0,
            **kwargs,
        }

        # Create messages format expected by OpenAI
        if type(prompt) is list:
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # Call the OpenAI API
        response = self.client.chat.completions.create(messages=messages, **params)

        # Log the response
        logger.debug(f"Model response: {response.choices[0].message.content}")

        return response.choices[0].message.content

    def fix_sql_in_json_string(self, raw_json_str: str):
        match = re.search(r'"sql"\s*:\s*"(.+?)"\s*,\s*"tables"', raw_json_str, re.DOTALL)
        if not match:
            raise ValueError("No sql found")

        raw_sql = match.group(1)
        escaped_sql = raw_sql.replace('"', r"\"")
        fixed_json_str = raw_json_str.replace(raw_sql, escaped_sql)

        return fixed_json_str

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a response and ensure it conforms to the provided JSON schema.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            A dictionary representing the JSON response
        """
        # Add instructions to format the response as JSON according to the schema
        # json_prompt = f"{prompt}\n\nRespond with a JSON object that
        # conforms to the following schema:\n{json.dumps(json_schema, indent=2)}"

        # Generate the response
        response_text = self.generate(prompt, response_format={"type": "json_object"}, **kwargs)

        # Parse the JSON response
        try:
            return json.loads(extract_json_str(response_text), strict=False)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            fixed_response_text = self.fix_sql_in_json_string(response_text)
            try:
                return json.loads(fixed_response_text, strict=False)
            except json.JSONDecodeError:
                pass
            return {}

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        tools: Optional[List[Any]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        hooks=None,
        **kwargs,
    ) -> Dict:
        """
        Generate response with unified tool support (supports both MCP servers and native tools).

        Args:
            prompt: Input prompt
            mcp_servers: Optional MCP servers to use
            tools: Optional native tools to use
            instruction: System instruction
            output_type: Expected output type
            max_turns: Maximum conversation turns
            session: Optional session for context
            hooks: Optional hooks for tool interception
            **kwargs: Additional parameters

        Returns:
            Dict with content and sql_contexts
        """
        # Use the internal method
        return await self._generate_with_tools_internal(
            prompt, mcp_servers, tools, instruction, output_type, max_turns, session, hooks, **kwargs
        )

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        tools: Optional[List[Any]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate response with streaming and tool support."""
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        # Use the unified streaming approach
        async for action in self._generate_with_tools_stream_internal(
            prompt,
            mcp_servers,
            tools,
            instruction,
            output_type,
            max_turns,
            action_history_manager,
            session,
            hooks,
            **kwargs,
        ):
            yield action

    async def generate_with_mcp(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate a response using multiple MCP (Machine Conversation Protocol) servers.

        Args:
            prompt: The input prompt to send to the model
            mcp_servers: Dictionary of MCP servers to use for execution
            instruction: The instruction for the agent
            max_turns: Maximum number of conversation turns
            **kwargs: Additional parameters for the agent

        Returns:
            The result from the MCP agent execution with content and sql_contexts
        """

        # Custom JSON encoder to handle special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        # Create async Anthropic client
        logger.debug(f"Creating async Anthropic client with base_url: " f"{self.api_base}, model: {self.model_name}")
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
                        logger.debug(f"Retrieved {len(mcp_tools)} tools from {server_name}")
                    except Exception as e:
                        logger.error(f"Error getting tools from {server_name}: {str(e)}")
                        continue

                logger.debug(f"Retrieved {len(all_tools)} tools from MCP servers")

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
                        # Save final text response
                        final_content = "\n".join([block.text for block in message if block.type == "text"])
                        logger.debug(f"No tool calls, conversation completed: {final_content}")
                        break

                    for block in message:
                        if block.type == "tool_use":
                            logger.debug(f"Executing tool: {block.name} with input: {block.input}")
                            tool_executed = False

                            for server_name, connected_server in connected_servers.items():
                                try:
                                    # Create minimal agent and run context for the new interface
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
                                        logger.debug(f"Tool {block.name} executed successfully on {server_name}")
                                        break
                                except Exception as e:
                                    logger.error(f"Error executing tool {block.name} on {server_name}: {str(e)}")
                                    continue

                            if not tool_executed:
                                logger.error(f"Tool {block.name} could not be executed on any server")

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
                                # Use "Error" to determine whether the execution was successful,
                                # because there's no way to judge it within MCP.
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
                                # If tool execution failed, add error message
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
                        else:
                            raise ValueError("Unknown block")

                logger.debug("Agent execution completed")
                return {"content": final_content, "sql_contexts": sql_contexts}

        except Exception as e:
            logger.error(f"Error in generate_with_mcp: {str(e)}")
            raise

    async def generate_with_mcp_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate a response using multiple MCP servers with streaming support."""
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        # Setup JSON encoder for special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                async with multiple_mcp_servers(mcp_servers) as connected_servers:
                    agent = self._setup_async_agent(instruction, connected_servers, output_type, **kwargs)
                    result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns)
                    function_call_count = 0

                    while not result.is_complete:
                        async for event in result.stream_events():
                            if not hasattr(event, "type") or event.type != "run_item_stream_event":
                                continue

                            if not (hasattr(event, "item") and hasattr(event.item, "type")):
                                continue

                            action = None
                            item_type = event.item.type

                            if item_type == "tool_call_item":
                                function_call_count += 1
                                action = self._process_tool_call_start(event, action_history_manager)
                            elif item_type == "tool_call_output_item":
                                action = self._process_tool_call_complete(event, action_history_manager)
                            elif item_type == "message_output_item":
                                action = self._process_message_output(event, action_history_manager)

                            if action:
                                yield action

                    # If we reach here, streaming completed successfully
                    break

            except MaxTurnsExceeded as e:
                logger.exception("Max turns exceeded")
                raise DatusException(ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns}) from e

            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_api_error(e)

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{max_retries + 1}): {error_code.code} - "
                        f"{error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Max retries reached or non-retryable error
                    logger.error(f"API error after {attempt + 1} attempts: {error_code.code} - {error_code.desc}")
                    raise DatusException(error_code)

            except Exception as e:
                logger.error(f"Error in streaming MCP execution: {str(e)}")
                raise

    @property
    def model_specs(self) -> Dict[str, Dict[str, int]]:
        """
        Model specifications dictionary containing context_length and max_tokens for Claude models.
        """
        return {
            # Claude Models
            "claude-sonnet-4-5": {"context_length": 1048576, "max_tokens": 65536},
            "claude-opus-4-1": {"context_length": 200000, "max_tokens": 32000},
            "claude-opus-4": {"context_length": 200000, "max_tokens": 32000},
            "claude-sonnet-4": {"context_length": 1048576, "max_tokens": 65536},  # 1M for beta 25/08
            "claude-3-7-sonnet": {"context_length": 200000, "max_tokens": 128000},
        }

    def context_length(self) -> Optional[int]:
        """
        Get the context length from model specs with prefix matching.

        Returns:
            Context length from model specs, or None if unavailable
        """
        # First try exact match
        if self.model_name in self.model_specs:
            return self.model_specs[self.model_name]["context_length"]

        # Try prefix matching for models like claude-3-5-sonnet-20241022
        for spec_model in self.model_specs:
            if self.model_name.startswith(spec_model):
                return self.model_specs[spec_model]["context_length"]

        return None

    async def _generate_with_tools_internal(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]],
        tools: Optional[List[Any]],
        instruction: str,
        output_type: type,
        max_turns: int,
        session: Optional[SQLiteSession] = None,
        hooks=None,
        **kwargs,
    ) -> Dict:
        """Internal method for unified tool execution (both MCP servers and native tools)."""
        from agents import Agent, OpenAIChatCompletionsModel, Runner

        # Custom JSON encoder for special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        # If no tools at all, fall back to basic generation
        if not mcp_servers and not tools:
            logger.warning("No tools or MCP servers provided, using basic generation")
            response = self.generate(f"{instruction}\n\n{prompt}", **kwargs)
            return {"content": response, "sql_contexts": []}

        async_client = create_openai_client(AsyncOpenAI, self.api_key, self.api_base + "/v1")
        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

        # Use multiple_mcp_servers context manager with empty dict if no MCP servers
        async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
            agent_kwargs = {
                "name": kwargs.pop("agent_name", "claude_agent"),
                "instructions": instruction,
                "output_type": output_type,
                "model": async_model,
            }

            # Only add mcp_servers if we have connected servers
            if connected_servers:
                agent_kwargs["mcp_servers"] = list(connected_servers.values())

            # Only add tools if we have them
            if tools:
                agent_kwargs["tools"] = tools

            if hooks:
                agent_kwargs["hooks"] = hooks

            agent = Agent(**agent_kwargs)

            try:
                result = await Runner.run(agent, input=prompt, max_turns=max_turns, session=session)
            except MaxTurnsExceeded as e:
                logger.error(f"Max turns exceeded: {str(e)}")
                from datus.utils.exceptions import DatusException, ErrorCode

                raise DatusException(ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns})

            # Extract SQL contexts from result (if any)
            sql_contexts = []
            if hasattr(result, "final_output"):
                from datus.models.mcp_result_extractors import extract_sql_contexts

                sql_contexts = extract_sql_contexts(result)

            # Extract usage information
            usage_info = {}
            if hasattr(result, "context_wrapper") and hasattr(result.context_wrapper, "usage"):
                usage = result.context_wrapper.usage
                usage_info = {
                    "requests": getattr(usage, "requests", 0),
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                    "cached_tokens": (
                        getattr(usage.input_tokens_details, "cached_tokens", 0)
                        if hasattr(usage, "input_tokens_details") and usage.input_tokens_details
                        else 0
                    ),
                }
                logger.debug(f"Claude tool execution usage: {usage_info}")

            return {
                "content": result.final_output if hasattr(result, "final_output") else "",
                "sql_contexts": sql_contexts,
                "usage": usage_info,
                "model": self.model_name,
                "turns_used": getattr(result, "turn_count", 0),
            }

    async def _generate_with_tools_stream_internal(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]],
        tools: Optional[List[Any]],
        instruction: str,
        output_type: type,
        max_turns: int,
        action_history_manager: ActionHistoryManager,
        session: Optional[SQLiteSession] = None,
        hooks=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Internal method for unified streaming tool execution."""
        from agents import Agent, OpenAIChatCompletionsModel, Runner

        # Custom JSON encoder for special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        # If no tools at all, return a simple text response
        if not mcp_servers and not tools:
            logger.warning("No tools or MCP servers provided, returning basic response")
            response = self.generate(f"{instruction}\n\n{prompt}", **kwargs)
            action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="final_response",
                messages=response,
                input_data={"prompt": prompt},
                output_data={"content": response},
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(action)
            yield action
            return

        async_client = create_openai_client(AsyncOpenAI, self.api_key, self.api_base + "/v1")
        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

        # Use multiple_mcp_servers context manager with empty dict if no MCP servers
        async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
            agent_kwargs = {
                "name": kwargs.pop("agent_name", "claude_agent"),
                "instructions": instruction,
                "output_type": output_type,
                "model": async_model,
            }

            # Only add mcp_servers if we have connected servers
            if connected_servers:
                agent_kwargs["mcp_servers"] = list(connected_servers.values())

            # Only add tools if we have them
            if tools:
                agent_kwargs["tools"] = tools

            if hooks:
                agent_kwargs["hooks"] = hooks

            agent = Agent(**agent_kwargs)

            try:
                result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns, session=session)

                # Buffer to reorder events: thinking messages should come before tool calls
                pending_tool_calls = []

                while not result.is_complete:
                    async for event in result.stream_events():
                        if not hasattr(event, "type") or event.type != "run_item_stream_event":
                            continue

                        if not (hasattr(event, "item") and hasattr(event.item, "type")):
                            continue

                        action = None
                        item_type = event.item.type

                        if item_type == "tool_call_item":
                            action = self._process_tool_call_start(event, action_history_manager)
                            if action:
                                # Buffer tool calls instead of yielding immediately
                                pending_tool_calls.append(action)
                                action = None
                        elif item_type == "tool_call_output_item":
                            action = self._process_tool_call_complete(event, action_history_manager)
                        elif item_type == "message_output_item":
                            action = self._process_message_output(event, action_history_manager)
                            if action:
                                # Yield thinking message first
                                yield action
                                # Then yield buffered tool calls
                                for tool_action in pending_tool_calls:
                                    yield tool_action
                                pending_tool_calls.clear()
                                action = None

                        if action:
                            yield action

                # Yield any remaining buffered tool calls
                for tool_action in pending_tool_calls:
                    yield tool_action

                # Yield final success action
                final_output = result.output if hasattr(result, "output") else {"content": ""}
                final_action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="final_response",
                    messages="Generation completed",
                    input_data={"content": ""},
                    output_data=final_output,
                    status=ActionStatus.SUCCESS,
                )
                action_history_manager.add_action(final_action)
                yield final_action

            except MaxTurnsExceeded as e:
                logger.error(f"Max turns exceeded: {str(e)}")
                error_action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="error",
                    messages=f"Max turns exceeded: {str(e)}",
                    input_data={},
                    output_data={"error": str(e)},
                    status=ActionStatus.FAILED,
                )
                action_history_manager.add_action(error_action)
                yield error_action

    def token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in a text using a simple approximation.

        Args:
            prompt (str): The text to count the tokens of.

        Returns:
            int: The estimated number of tokens in the text.
        """
        # Claude uses a similar tokenization scheme to GPT-3
        # We can use a simple approximation of ~4 characters per token
        return int(len(prompt) / 4 + 0.5)

    def _setup_async_agent(self, instruction: str, mcp_servers: Dict, output_type: dict, **kwargs):
        """Setup async client and agent."""
        async_client = create_openai_client(AsyncOpenAI, self.api_key, self.api_base + "/v1")
        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

        # Claude uses OpenAI-compatible interface via /v1 endpoint
        agent = Agent(
            name=kwargs.pop("agent_name", "MCP_Agent"),
            instructions=instruction,
            mcp_servers=list(mcp_servers.values()),
            output_type=str,  # Use str for compatibility
            model=async_model,
        )
        return agent

    async def _generate_with_mcp_session(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type,
        max_turns: int,
        session: SQLiteSession,
        **kwargs,
    ) -> Dict:
        """Generate response with MCP servers and session support."""

        # Custom JSON encoder to handle special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        try:
            # Use context manager to manage multiple MCP servers
            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                # Set up agent with session support
                async_client = create_openai_client(AsyncOpenAI, self.api_key, self.api_base + "/v1")
                model_params = {"model": self.model_name}
                async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

                # Create agent (session is passed to Runner, not Agent)
                agent = Agent(
                    name=kwargs.pop("agent_name", "MCP_Session_Agent"),
                    instructions=instruction,
                    mcp_servers=list(connected_servers.values()),
                    output_type=str,  # Use str for compatibility
                    model=async_model,
                )

                # Run the agent with session
                result = await Runner.run(agent, input=prompt, max_turns=max_turns, session=session)

                # Extract content and sql_contexts from result
                content = ""
                sql_contexts = []

                if hasattr(result, "content") and result.content:
                    content = str(result.content)
                elif hasattr(result, "text") and result.text:
                    content = str(result.text)

                # For now, return the result in the expected format
                return {"content": content, "sql_contexts": sql_contexts}

        except Exception as e:
            logger.error(f"Error in _generate_with_mcp_session: {str(e)}")
            raise

    def _process_tool_call_start(self, event, action_history_manager: ActionHistoryManager) -> ActionHistory:
        """Process tool_call_item events."""
        import uuid

        raw_item = event.item.raw_item
        call_id = getattr(raw_item, "call_id", None)
        function_name = getattr(raw_item, "name", None)
        arguments = getattr(raw_item, "arguments", None)

        # Generate unique action_id if call_id is None or empty to prevent duplicates
        if not call_id:
            logger.warning("No call_id found in tool_call event; generating a unique action_id.")
        action_id = call_id if call_id else f"tool_call_{uuid.uuid4().hex[:8]}"

        # Check if action with this action_id already exists
        if action_history_manager.find_action_by_id(action_id):
            return None

        action = ActionHistory(
            action_id=action_id,
            role=ActionRole.TOOL,
            messages=f"Tool call: {function_name}" if function_name else "Tool call",
            action_type=function_name or "unknown",
            input={"function_name": function_name, "arguments": arguments, "call_id": call_id},
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        return action

    def _process_tool_call_complete(self, event, action_history_manager: ActionHistoryManager) -> ActionHistory:
        """Process tool_call_output_item events."""
        # try to find the action by call_id, but it seems claude doesn't have call_id in the raw_item sometimes
        call_id = getattr(event.item.raw_item, "call_id", None)
        matching_action = action_history_manager.find_action_by_id(call_id) if call_id else None

        if not matching_action:
            # Try to match by the most recent PROCESSING action as fallback
            processing_actions = [a for a in action_history_manager.actions if a.status == ActionStatus.PROCESSING]
            if processing_actions:
                matching_action = processing_actions[-1]  # Get the most recent
            else:
                return None

        output_data = {
            "call_id": call_id,
            "success": True,
            "raw_output": event.item.output,
        }

        action_history_manager.update_action_by_id(
            matching_action.action_id, output=output_data, end_time=datetime.now(), status=ActionStatus.SUCCESS
        )

        # Don't return the action to avoid duplicate yield
        return None

    def _process_message_output(self, event, action_history_manager: ActionHistoryManager) -> ActionHistory:
        """Process message_output_item events."""
        if not (hasattr(event.item, "raw_item") and hasattr(event.item.raw_item, "content")):
            return None

        content = event.item.raw_item.content
        if not content:
            return None

        # Extract text content
        if isinstance(content, list) and content:
            text_content = content[0].text if hasattr(content[0], "text") else str(content[0])
        else:
            text_content = str(content)

        # Create action with raw content
        if len(text_content) > 0:
            action = ActionHistory(
                action_id=str(uuid.uuid4()),
                role=ActionRole.ASSISTANT,
                messages=(f"Thinking: {text_content}"),
                action_type="message",
                input={},
                output={
                    "success": True,
                    "raw_output": text_content,
                },
                status=ActionStatus.SUCCESS,
            )
            action.end_time = datetime.now()
            action_history_manager.add_action(action)
            return action
        else:
            return None

    def close(self):
        """Close HTTP clients and cleanup resources."""
        if hasattr(self, "proxy_client") and self.proxy_client:
            try:
                self.proxy_client.close()
                logger.debug("Proxy client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing proxy client: {e}")

        # Close the anthropic client if it has a close method
        if hasattr(self, "anthropic_client") and hasattr(self.anthropic_client, "close"):
            try:
                self.anthropic_client.close()
                logger.debug("Anthropic client closed successfully")
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
