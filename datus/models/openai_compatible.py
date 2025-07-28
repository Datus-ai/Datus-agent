"""OpenAI-compatible base model for models that use OpenAI-compatible APIs."""

import asyncio
import json
import time
import warnings
from datetime import date, datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio
from agents import SQLiteSession
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI, APIConnectionError, APIError, APITimeoutError, RateLimitError
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.models.session_manager import SessionManager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Monkey patch to fix ResponseTextDeltaEvent logprobs validation issue
try:
    from agents.models.chatcmpl_stream_handler import ResponseTextDeltaEvent
    from pydantic import Field
    from typing import Optional, Any
    
    # Get the original fields and make logprobs optional
    original_fields = ResponseTextDeltaEvent.model_fields.copy()
    if 'logprobs' in original_fields:
        # Create a new field annotation that allows None
        original_fields['logprobs'] = Field(default=None)
        
        # Rebuild the model with optional logprobs
        ResponseTextDeltaEvent.__annotations__['logprobs'] = Optional[Any]
        ResponseTextDeltaEvent.model_fields['logprobs'] = Field(default=None)
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
        
        # Session management is handled by the base class
        
        # Context for tracing
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
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                params = {
                    "model": self.model_name,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "top_p": kwargs.get("top_p", 1.0),
                    **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "top_p"]}
                }
                
                # Convert prompt to messages format
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": str(prompt)}]
                
                response = self.client.chat.completions.create(messages=messages, **params)
                return response.choices[0].message.content
                
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)
                
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{max_retries + 1}): {error_code.code} - "
                        f"{error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Max retries reached or non-retryable error
                    logger.error(f"API error after {attempt + 1} attempts: {error_code.code} - {error_code.desc}")
                    raise DatusException(error_code)
                    
            except Exception as e:
                logger.error(f"Unexpected error in generate: {str(e)}")
                raise
    
    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a JSON response with error handling.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON dictionary
        """
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                # Set JSON mode
                kwargs["response_format"] = {"type": "json_object"}
                
                response_text = self.generate(prompt, **kwargs)
                
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
                    
                    return {
                        "error": "Failed to parse JSON response",
                        "raw_response": response_text
                    }
                    
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)
                
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"API error in JSON generation (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_code.code} - {error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"API error in JSON generation after {attempt + 1} attempts: {error_code.code} - {error_code.desc}")
                    raise DatusException(error_code)
                    
            except Exception as e:
                logger.error(f"Unexpected error in JSON generation: {str(e)}")
                raise
    
    # Session management methods are inherited from LLMBaseModel
    
    # New unified tool methods (replacing generate_with_mcp)
    async def generate_with_tools(
        self,
        prompt: str,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        tools: Optional[List[Any]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        **kwargs
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
            **kwargs: Additional parameters
            
        Returns:
            Result with content and sql_contexts
        """
        # For now, focus on MCP server support since that's what existing code uses
        if not mcp_servers:
            # Fallback to basic generation if no tools
            response = self.generate(f"{instruction}\n\n{prompt}", **kwargs)
            return {"content": response, "sql_contexts": []}
        
        return await self._generate_with_mcp_servers(
            prompt, mcp_servers, instruction, output_type, max_turns, session, **kwargs
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
        **kwargs
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
        
        # For now, focus on MCP server support
        if not mcp_servers:
            # Basic streaming not implemented yet
            return
        
        async for action in self._generate_with_mcp_servers_stream(
            prompt, mcp_servers, instruction, output_type, max_turns, 
            session, action_history_manager, **kwargs
        ):
            yield action
    
    async def _generate_with_mcp_servers(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type,
        max_turns: int,
        session: Optional[SQLiteSession],
        **kwargs
    ) -> Dict:
        """Internal method for MCP server execution with error handling."""
        # Custom JSON encoder for special types
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
                async_client = self._create_async_client()
                model_params = {"model": self.model_name}
                async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)
                
                async with multiple_mcp_servers(mcp_servers) as connected_servers:
                    agent = Agent(
                        name=kwargs.pop("agent_name", "MCP_Agent"),
                        instructions=instruction,
                        mcp_servers=list(connected_servers.values()),
                        output_type=output_type,
                        model=async_model,
                    )
                    
                    result = await Runner.run(agent, input=prompt, max_turns=max_turns, session=session)
                    
                    return {
                        "content": result.final_output,
                        "sql_contexts": extract_sql_contexts(result)
                    }
                    
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)
                
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"API error in MCP execution (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_code.code} - {error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"API error in MCP execution after {attempt + 1} attempts: {error_code.code} - {error_code.desc}")
                    raise DatusException(error_code)
                    
            except Exception as e:
                logger.error(f"Error in MCP execution: {str(e)}")
                raise
    
    async def _generate_with_mcp_servers_stream(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type,
        max_turns: int,
        session: Optional[SQLiteSession],
        action_history_manager: ActionHistoryManager,
        **kwargs
    ) -> AsyncGenerator[ActionHistory, None]:
        """Internal method for MCP server streaming execution with error handling."""
        # Custom JSON encoder
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
                async_client = self._create_async_client()
                model_params = {"model": self.model_name}
                async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)
                
                async with multiple_mcp_servers(mcp_servers) as connected_servers:
                    agent = Agent(
                        name=kwargs.pop("agent_name", "MCP_Agent"),
                        instructions=instruction,
                        mcp_servers=list(connected_servers.values()),
                        output_type=output_type,
                        model=async_model,
                    )
                    
                    result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns, session=session)
                    
                    while not result.is_complete:
                        async for event in result.stream_events():
                            action = self._process_stream_event(event, action_history_manager)
                            if action:
                                yield action
                    
                    # If we reach here, streaming completed successfully
                    break
                    
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)
                
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"API error in MCP streaming (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_code.code} - {error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"API error in MCP streaming after {attempt + 1} attempts: {error_code.code} - {error_code.desc}")
                    raise DatusException(error_code)
                    
            except Exception as e:
                logger.error(f"Error in MCP streaming: {str(e)}")
                raise
    
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
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        raw_item = event.item.raw_item
        call_id = getattr(raw_item, "call_id", None)
        function_name = getattr(raw_item, "name", None)
        arguments = getattr(raw_item, "arguments", None)

        # Check if action with this call_id already exists
        if call_id and action_history_manager.find_action_by_id(call_id):
            return None

        action = ActionHistory(
            action_id=call_id,
            role=ActionRole.TOOL,
            messages="MCP call",
            action_type=function_name or "unknown",
            input={"function_name": function_name, "arguments": arguments, "call_id": call_id},
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        return action

    def _process_tool_call_complete(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
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
            matching_action.action_id, 
            output=output_data, 
            end_time=datetime.now(), 
            status=ActionStatus.SUCCESS
        )

        # Don't return the action to avoid duplicate yield
        return None

    def _process_message_output(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
        """Process message_output_item events."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
        import uuid

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
    
    # Backward compatibility methods (with deprecation warnings)
    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type = str,
        max_turns: int = 10,
        **kwargs
    ) -> Dict:
        """
        Deprecated: Use generate_with_tools instead.
        """
        warnings.warn(
            "generate_with_mcp is deprecated. Use generate_with_tools instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return await self.generate_with_tools(
            prompt, mcp_servers=mcp_servers, instruction=instruction,
            output_type=output_type, max_turns=max_turns, **kwargs
        )
    
    async def generate_with_mcp_stream(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type = str,
        max_turns: int = 10,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Deprecated: Use generate_with_tools_stream instead.
        """
        warnings.warn(
            "generate_with_mcp_stream is deprecated. Use generate_with_tools_stream instead.",
            DeprecationWarning,
            stacklevel=2
        )
        async for action in self.generate_with_tools_stream(
            prompt, mcp_servers=mcp_servers, instruction=instruction,
            output_type=output_type, max_turns=max_turns,
            action_history_manager=action_history_manager, **kwargs
        ):
            yield action
    
    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context for tracing."""
        self.workflow = workflow
        self.current_node = current_node
    
    def get_model_info(self) -> Optional[Dict]:
        """
        Get model information from the /v1/models API endpoint.
        
        Returns:
            Dictionary with model info, or None if unavailable
        """
        if self._model_info is not None:
            return self._model_info
        
        try:
            # Use the OpenAI client to get model info
            model_info = self.client.models.retrieve(self.model_name)
            
            # Convert to dict for easier access
            self._model_info = {
                "id": getattr(model_info, 'id', None),
                "context_length": getattr(model_info, 'context_length', None),
                "max_tokens": getattr(model_info, 'max_tokens', None),
                "owned_by": getattr(model_info, 'owned_by', None),
                "created": getattr(model_info, 'created', None),
            }
            
            logger.debug(f"Retrieved model info for {self.model_name}: {self._model_info}")
            return self._model_info
            
        except Exception as e:
            logger.warning(f"Failed to retrieve model info for {self.model_name}: {str(e)}")
            self._model_info = {}  # Cache empty result to avoid repeated failures
            return None
    
    def max_tokens(self) -> Optional[int]:
        """
        Get the max tokens from model info.
        
        Returns:
            Max tokens from model info, or None if unavailable
        """
        model_info = self.get_model_info()
        if model_info:
            return model_info.get('max_tokens')
        return None

    def token_count(self, prompt: str) -> int:
        """
        Count tokens in prompt. Default implementation uses character approximation.
        Override in subclasses for model-specific tokenization.
        """
        return len(prompt) // 4