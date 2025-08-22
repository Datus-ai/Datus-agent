"""OpenAI-compatible base model for models that use OpenAI-compatible APIs."""

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import yaml
from agents import Agent, OpenAIChatCompletionsModel, Runner, SQLiteSession, Tool
from agents.mcp import MCPServerStdio
from langsmith.wrappers import wrap_openai
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Monkey patch to fix ResponseTextDeltaEvent logprobs validation issue
try:
    from agents.models.chatcmpl_stream_handler import ResponseTextDeltaEvent
    from pydantic import Field

    # Get the original fields and make logprobs optional
    original_fields = ResponseTextDeltaEvent.model_fields.copy()
    if "logprobs" in original_fields:
        # Create a new field annotation that allows None
        original_fields["logprobs"] = Field(default=None)

        # Rebuild the model with optional logprobs
        ResponseTextDeltaEvent.__annotations__["logprobs"] = Optional[Any]
        ResponseTextDeltaEvent.model_fields["logprobs"] = Field(default=None)
        ResponseTextDeltaEvent.model_rebuild()

        logger.debug("Successfully patched ResponseTextDeltaEvent to make logprobs optional")
except ImportError:
    logger.warning("Could not import ResponseTextDeltaEvent - patch not applied")
except Exception as e:
    logger.warning(f"Could not patch ResponseTextDeltaEvent: {e}")


def classify_openai_compatible_error(error: Exception) -> tuple[ErrorCode, bool]:
    """Classify OpenAI-compatible API errors and return error code and whether it's retryable."""
    error_msg = str(error).lower()

    if isinstance(error, APIError):
        # Handle specific HTTP status codes and error types
        if any(indicator in error_msg for indicator in ["401", "unauthorized", "authentication"]):
            return ErrorCode.MODEL_AUTHENTICATION_ERROR, False
        elif any(indicator in error_msg for indicator in ["403", "forbidden", "permission"]):
            return ErrorCode.MODEL_PERMISSION_ERROR, False
        elif any(indicator in error_msg for indicator in ["404", "not found"]):
            return ErrorCode.MODEL_NOT_FOUND, False
        elif any(indicator in error_msg for indicator in ["413", "too large", "request size"]):
            return ErrorCode.MODEL_REQUEST_TOO_LARGE, False
        elif any(indicator in error_msg for indicator in ["429", "rate limit", "quota", "billing"]):
            if any(indicator in error_msg for indicator in ["quota", "billing"]):
                return ErrorCode.MODEL_QUOTA_EXCEEDED, False
            else:
                return ErrorCode.MODEL_RATE_LIMIT, True
        elif any(indicator in error_msg for indicator in ["500", "internal", "server error"]):
            return ErrorCode.MODEL_API_ERROR, True
        elif any(indicator in error_msg for indicator in ["502", "503", "overloaded"]):
            return ErrorCode.MODEL_OVERLOADED, True
        elif any(indicator in error_msg for indicator in ["400", "bad request", "invalid"]):
            return ErrorCode.MODEL_INVALID_RESPONSE, False

    if isinstance(error, RateLimitError):
        return ErrorCode.MODEL_RATE_LIMIT, True

    if isinstance(error, APITimeoutError):
        return ErrorCode.MODEL_TIMEOUT_ERROR, True

    if isinstance(error, APIConnectionError):
        return ErrorCode.MODEL_CONNECTION_ERROR, True

    # Default to general request failure
    return ErrorCode.MODEL_REQUEST_FAILED, False


class OpenAICompatibleModel(LLMBaseModel):
    """
    Base class for models that use OpenAI-compatible APIs.

    Provides common functionality for:
    - Session management for multi-turn conversations
    - OpenAI client setup and configuration
    - Unified tool execution (replacing generate_with_mcp)
    - Streaming support
    - Error handling and retry logic
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

        self.model_config = model_config
        self.model_name = model_config.model
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()

        # Initialize clients
        self.client = self._create_sync_client()
        self._async_client = None

        # Context for tracing ToDo: replace it with Context object
        self.workflow = None
        self.current_node = None

        # Cache for model info
        self._model_info = None

    def _get_api_key(self) -> str:
        """Get API key from config or environment. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_api_key")

    def _get_base_url(self) -> Optional[str]:
        """Get base URL from config. Override in subclasses if needed."""
        return self.model_config.base_url

    def _create_sync_client(self) -> OpenAI:
        """Create synchronous OpenAI client."""
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)
        return wrap_openai(client)

    def _create_async_client(self) -> AsyncOpenAI:
        """Create asynchronous OpenAI client."""
        if self._async_client is None:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            client = AsyncOpenAI(**client_kwargs)
            self._async_client = wrap_openai(client)

        return self._async_client

    def _with_retry(
        self, operation_func, operation_name: str = "operation", max_retries: int = 3, base_delay: float = 1.0
    ):
        """
        Generic retry wrapper for synchronous operations.

        Args:
            operation_func: Function to execute (should raise API exceptions on failure)
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff

        Returns:
            Result from operation_func
        """
        for attempt in range(max_retries + 1):
            try:
                return operation_func()
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"API error in {operation_name} (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_code.code} - {error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Max retries reached or non-retryable error
                    logger.error(
                        f"API error in {operation_name} after {attempt + 1} attempts: "
                        f"{error_code.code} - {error_code.desc}"
                    )
                    raise DatusException(error_code)
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {str(e)}")
                raise

    async def _with_retry_async(
        self, operation_func, operation_name: str = "operation", max_retries: int = 3, base_delay: float = 1.0
    ):
        """
        Generic retry wrapper for asynchronous operations.

        Args:
            operation_func: Async function to execute (should raise API exceptions on failure)
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff

        Returns:
            Result from operation_func
        """
        for attempt in range(max_retries + 1):
            try:
                return await operation_func()
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"API error in {operation_name} (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_code.code} - {error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Max retries reached or non-retryable error
                    logger.error(
                        f"API error in {operation_name} after {attempt + 1} attempts: "
                        f"{error_code.code} - {error_code.desc}"
                    )
                    raise DatusException(error_code)
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {str(e)}")
                raise

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """
        Generate a response from the model with error handling and retry logic.

        Args:
            prompt: The input prompt (string or list of messages)
            enable_thinking: Enable thinking mode for hybrid models (default: False)
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """

        def _generate_operation():
            params = {
                "model": self.model_name,
            }

            # Add temperature and top_p only if explicitly provided
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            elif not hasattr(self, "_uses_completion_tokens_parameter") or not self._uses_completion_tokens_parameter():
                # Add default temperature only for non-reasoning models
                params["temperature"] = 0.7

            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            elif not hasattr(self, "_uses_completion_tokens_parameter") or not self._uses_completion_tokens_parameter():
                # Add default top_p only for non-reasoning models
                params["top_p"] = 1.0

            # Handle both max_tokens and max_completion_tokens parameters (only if explicitly provided)
            if "max_tokens" in kwargs:
                params["max_tokens"] = kwargs["max_tokens"]
            if "max_completion_tokens" in kwargs:
                params["max_completion_tokens"] = kwargs["max_completion_tokens"]

            # Filter out handled parameters from remaining kwargs
            excluded_params = ["temperature", "top_p", "max_tokens", "max_completion_tokens"]
            params.update({k: v for k, v in kwargs.items() if k not in excluded_params})

            # Convert prompt to messages format
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            response = self.client.chat.completions.create(messages=messages, **params)

            message = response.choices[0].message
            content = message.content

            # Handle reasoning content for reasoning models (DeepSeek R1, OpenAI O-series)
            reasoning_content = None
            if enable_thinking:
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                    # If main content is empty but reasoning_content exists, use reasoning_content
                    if not content or content.strip() == "":
                        content = reasoning_content + "\n" + content
                    logger.debug(f"Found reasoning_content: {reasoning_content[:100]}...")

            final_content = content or ""

            # Save LLM trace if method exists (for models that support it like DeepSeekModel)
            if hasattr(self, "_save_llm_trace"):
                self._save_llm_trace(messages, final_content, reasoning_content)

            return final_content

        return self._with_retry(_generate_operation, "text generation")

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a JSON response with error handling.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Parsed JSON dictionary
        """
        # Set JSON mode
        json_kwargs = kwargs.copy()
        json_kwargs["response_format"] = {"type": "json_object"}

        # Pass through enable_thinking if provided
        enable_thinking_param = json_kwargs.pop("enable_thinking", False)
        response_text = self.generate(prompt, enable_thinking=enable_thinking_param, **json_kwargs)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            return {"error": "Failed to parse JSON response", "raw_response": response_text}

    async def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> Dict:
        """
        Generate response with unified tool support (replaces generate_with_mcp).

        Args:
            prompt: Input prompt
            mcp_servers: Optional MCP servers to use
            tools: Optional regular tools to use
            instruction: System instruction
            output_type: Expected output type
            max_turns: Maximum conversation turns
            session: Optional session for context
            action_history_manager: Action history manager for tracking
            **kwargs: Additional parameters

        Returns:
            Dict with content and sql_contexts
        """
        # Use the internal method that returns a Dict
        return await self._generate_with_tools_internal(
            prompt, mcp_servers, tools, instruction, output_type, max_turns, session, **kwargs
        )

    async def generate_with_tools_stream(
        self,
        prompt: str,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        tools: Optional[List[Any]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Generate response with streaming and tool support (replaces generate_with_mcp_stream).

        Args:
            prompt: Input prompt
            mcp_servers: Optional MCP servers
            tools: Optional regular tools
            instruction: System instruction
            output_type: Expected output type
            max_turns: Maximum turns
            session: Optional session
            action_history_manager: Action history manager
            **kwargs: Additional parameters

        Yields:
            ActionHistory objects for streaming updates
        """
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        async for action in self._generate_with_tools_stream_internal(
            prompt, mcp_servers, tools, instruction, output_type, max_turns, session, action_history_manager, **kwargs
        ):
            yield action

    async def _generate_with_tools_internal(
        self,
        prompt: str,
        mcp_servers: Optional[Dict[str, MCPServerStdio]],
        tools: Optional[List[Tool]],
        instruction: str,
        output_type: type,
        max_turns: int,
        session: Optional[SQLiteSession] = None,
        **kwargs,
    ) -> Dict:
        """Internal method for tool execution with error handling."""

        # Custom JSON encoder for special types
        # (for snowflake mcp server, we can remove it after using native db tools)
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        async def _tools_operation():
            async_client = self._create_async_client()
            model_params = {"model": self.model_name}
            async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

            # Use multiple_mcp_servers context manager with empty dict if no MCP servers
            async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
                agent_kwargs = {
                    "name": kwargs.pop("agent_name", "default_agent"),
                    "instructions": instruction,
                    "output_type": output_type,
                    "model": async_model,
                }

                # Only add mcp_servers if we have connected servers
                if connected_servers:
                    agent_kwargs["mcp_servers"] = list(connected_servers.values())

                # Only add tools if we have them
                if tools:
                    agent_kwargs["tools"] = list(tools.values())

                agent = Agent(**agent_kwargs)
                result = await Runner.run(agent, input=prompt, max_turns=max_turns, session=session)

                # Save LLM trace if method exists (for models that support it like DeepSeekModel)
                if hasattr(self, "_save_llm_trace"):
                    # For tools calls, we need to extract messages from the result
                    messages = [{"role": "user", "content": prompt}]
                    if instruction:
                        messages.insert(0, {"role": "system", "content": instruction})

                    # Get complete conversation history including tool calls
                    conversation_history = None
                    if hasattr(result, "to_input_list"):
                        try:
                            conversation_history = result.to_input_list()
                        except Exception as e:
                            logger.debug(f"Failed to get conversation history: {e}")

                    self._save_llm_trace(messages, result.final_output, conversation_history)

                return {"content": result.final_output, "sql_contexts": extract_sql_contexts(result)}

        return await self._with_retry_async(_tools_operation, "tool execution")

    async def _generate_with_tools_stream_internal(
        self,
        prompt: str,
        mcp_servers: Optional[Dict[str, MCPServerStdio]],
        tools: Optional[List[Tool]],
        instruction: str,
        output_type: type,
        max_turns: int,
        session: Optional[SQLiteSession],
        action_history_manager: ActionHistoryManager,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Internal method for tool streaming execution with error handling."""

        # Custom JSON encoder
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        async def _stream_operation():
            async_client = self._create_async_client()
            model_params = {"model": self.model_name}
            async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

            # Use multiple_mcp_servers context manager with empty dict if no MCP servers
            async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
                agent_kwargs = {
                    "name": kwargs.pop("agent_name", "Tools_Agent"),
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

                agent = Agent(**agent_kwargs)

                result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns, session=session)

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
                        elif item_type == "tool_call_output_item":
                            action = self._process_tool_call_complete(event, action_history_manager)
                        elif item_type == "message_output_item":
                            action = self._process_message_output(event, action_history_manager)

                        if action:
                            yield action

                # Save LLM trace if method exists (for models that support it like DeepSeekModel)
                if hasattr(self, "_save_llm_trace"):
                    # For tools calls, we need to extract messages from the result
                    messages = [{"role": "user", "content": prompt}]
                    if instruction:
                        messages.insert(0, {"role": "system", "content": instruction})

                    # Get complete conversation history including tool calls
                    conversation_history = None
                    if hasattr(result, "to_input_list"):
                        try:
                            conversation_history = result.to_input_list()
                        except Exception as e:
                            logger.debug(f"Failed to get conversation history: {e}")

                    final_output = result.final_output if hasattr(result, "final_output") else ""
                    self._save_llm_trace(messages, final_output, conversation_history)

        # Execute the streaming operation directly without retry logic
        async for action in _stream_operation():
            yield action

    def _process_stream_event(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
        """Process streaming events and route to appropriate handlers."""
        if not hasattr(event, "type") or event.type != "run_item_stream_event":
            return None

        if not (hasattr(event, "item") and hasattr(event.item, "type")):
            return None

        action = None
        item_type = event.item.type

        if item_type == "tool_call_item":
            action = self._process_tool_call_start(event, action_history_manager)
        elif item_type == "tool_call_output_item":
            action = self._process_tool_call_complete(event, action_history_manager)
        elif item_type == "message_output_item":
            action = self._process_message_output(event, action_history_manager)

        return action

    def _process_tool_call_start(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
        """Process tool_call_item events."""
        import uuid

        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

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
            messages="Tool call",
            action_type=function_name or "unknown",
            input={"function_name": function_name, "arguments": arguments, "call_id": call_id},
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        return action

    def _process_tool_call_complete(
        self, event, action_history_manager: ActionHistoryManager
    ) -> Optional[ActionHistory]:
        """Process tool_call_output_item events."""
        from datus.schemas.action_history import ActionStatus

        # Try to find the action by call_id, but it seems some models don't have call_id in the raw_item sometimes
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

    def _process_message_output(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
        """Process message_output_item events."""
        import uuid

        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        if not (hasattr(event.item, "raw_item") and hasattr(event.item.raw_item, "content")):
            return None

        content = event.item.raw_item.content
        if not content:
            return None

        logger.debug(f"Processing message output: {content}")

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
                messages=f"Thinking: {text_content}",
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
            logger.debug(f"No text content found in message output: {content}")
            return None

    @property
    def model_specs(self) -> Dict[str, Dict[str, int]]:
        """
        Model specifications dictionary containing context_length and max_tokens for various models.
        """
        return {
            # OpenAI Models
            "gpt-5": {"context_length": 400000, "max_tokens": 128000},
            "gpt-4o": {"context_length": 128000, "max_tokens": 16384},
            "gpt-03": {"context_length": 200000, "max_tokens": 200000},
            
            # DeepSeek Models
            "deepseek-chat": {"context_length": 65535, "max_tokens": 8192},
            "deepseek-v3": {"context_length": 65535, "max_tokens": 8192},
            "deepseek-reasoner": {"context_length": 65535, "max_tokens": 65535},
            "deepseek-r1": {"context_length": 65535, "max_tokens": 65535},
            
            # Moonshot (Kimi) Models
            "kimi-k2": {"context_length": 128000, "max_tokens": 8192},
            
            # Qwen Models
            "qwen3-coder": {"context_length": 128000, "max_tokens": 8192},
            
            # Gemini Models
            "gemini-2.5-pro": {"context_length": 1048576, "max_tokens": 65535},
            "gemini-2.5-flash": {"context_length": 1048576, "max_tokens": 8192},
            "gemini-2.5-flash-lite": {"context_length": 1048576, "max_tokens": 8192},
        }

    def max_tokens(self) -> Optional[int]:
        """
        Get the max tokens from model specs with prefix matching.

        Returns:
            Max tokens from model specs, or None if unavailable
        """
        # First try exact match
        if self.model_name in self.model_specs:
            return self.model_specs[self.model_name]["max_tokens"]
        
        # Try prefix matching for models like gpt-4o-mini, kimi-k2-0711-preview
        for spec_model in self.model_specs:
            if self.model_name.startswith(spec_model):
                return self.model_specs[spec_model]["max_tokens"]
        
        return None

    def context_length(self) -> Optional[int]:
        """
        Get the context length from model specs with prefix matching.

        Returns:
            Context length from model specs, or None if unavailable
        """
        # First try exact match
        if self.model_name in self.model_specs:
            return self.model_specs[self.model_name]["context_length"]
        
        # Try prefix matching for models like gpt-4o-mini, kimi-k2-0711-preview
        for spec_model in self.model_specs:
            if self.model_name.startswith(spec_model):
                return self.model_specs[spec_model]["context_length"]
        
        return None

    def token_count(self, prompt: str) -> int:
        """
        Count tokens in prompt. Default implementation uses character approximation.
        Override in subclasses for model-specific tokenization.
        """
        return len(prompt) // 4

    def _save_llm_trace(self, prompt: Any, response_content: str, reasoning_content: Any = None):
        """Save LLM input/output trace to YAML file if tracing is enabled.

        Args:
            prompt: The input prompt (str or list of messages)
            response_content: The response content from the model
            reasoning_content: Optional reasoning content for reasoning models
        """
        if not self.model_config.save_llm_trace:
            return

        try:
            # Get workflow and node context from current execution
            if (
                not hasattr(self, "workflow")
                or not self.workflow
                or not hasattr(self, "current_node")
                or not self.current_node
            ):
                logger.debug("No workflow or node context available for trace saving")
                return

            # Create trace directory
            trajectory_dir = Path(self.workflow.global_config.trajectory_dir)
            task_id = self.workflow.task.id
            trace_dir = trajectory_dir / task_id
            trace_dir.mkdir(parents=True, exist_ok=True)

            # Parse prompt to separate system and user content
            system_prompt = ""
            user_prompt = ""

            if isinstance(prompt, list):
                # Handle message format like [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                for message in prompt:
                    if isinstance(message, dict):
                        role = message.get("role", "")
                        content = message.get("content", "")
                        if role == "system":
                            # Concatenate multiple system messages with newlines
                            if system_prompt:
                                system_prompt += "\n" + content
                            else:
                                system_prompt = content
                        elif role == "user":
                            # Concatenate multiple user messages with newlines
                            if user_prompt:
                                user_prompt += "\n" + content
                            else:
                                user_prompt = content
                        elif role == "assistant":
                            # Skip assistant messages in prompt parsing
                            continue
            elif isinstance(prompt, str):
                # Handle string prompt - put it all in user_prompt
                user_prompt = prompt
            else:
                # Handle other types by converting to string
                user_prompt = str(prompt)

            # Ensure we have valid strings
            system_prompt = system_prompt or ""
            user_prompt = user_prompt or ""
            response_content = response_content or ""

            # Create trace data with improved structure
            trace_data = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "reason_content": reasoning_content or "",
                "output_content": response_content,
            }

            # Save to YAML file named after node ID
            trace_file = trace_dir / f"{self.current_node.id}.yml"
            with open(trace_file, "w", encoding="utf-8") as f:
                yaml.dump(trace_data, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)

            logger.debug(f"LLM trace saved to {trace_file}")

        except Exception as e:
            logger.error(f"Failed to save LLM trace: {str(e)}")
            # Don't re-raise to avoid breaking the main execution flow

    # ToDo: delete it later
    # Backward compatibility methods (with deprecation warnings)
    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type = str,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        logger.warn(
            "generate_with_mcp is deprecated. Use generate_with_tools instead.", DeprecationWarning, stacklevel=2
        )
        return await self._generate_with_tools_internal(
            prompt,
            mcp_servers,
            None,  # no regular tools for backward compatibility
            instruction,
            output_type,
            max_turns,
            **kwargs,  # session will be passed here if provided, otherwise defaults to None
        )

    async def generate_with_mcp_stream(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type = str,
        max_turns: int = 10,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Deprecated: Use generate_with_tools_stream instead.
        """
        logger.warn(
            "generate_with_mcp_stream is deprecated. Use generate_with_tools_stream instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        async for action in self.generate_with_tools_stream(
            prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            action_history_manager=action_history_manager,
            **kwargs,
        ):
            yield action
