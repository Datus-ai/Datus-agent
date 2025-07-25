"""OpenAI-compatible base model for models that use OpenAI-compatible APIs."""

import asyncio
import json
import warnings
from datetime import date, datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio
from agents.sessions import SQLiteSession
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.models.session_manager import SessionManager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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
        
        # Session management
        self.session_manager = SessionManager()
        
        # Context for tracing
        self.workflow = None
        self.current_node = None
    
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
        Generate a response from the model.
        
        Args:
            prompt: The input prompt (string or list of messages)
            enable_thinking: Enable thinking mode for hybrid models (default: False)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
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
    
    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a JSON response.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON dictionary
        """
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
    
    # Session management methods
    def create_session(self, session_id: str) -> SQLiteSession:
        """Create or get a session for multi-turn conversations."""
        return self.session_manager.create_session(session_id)
    
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.session_manager.clear_session(session_id)
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session completely."""
        self.session_manager.delete_session(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        return self.session_manager.list_sessions()
    
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
        """Internal method for MCP server execution."""
        # Custom JSON encoder for special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)
        
        json._default_encoder = CustomJSONEncoder()
        
        async_client = self._create_async_client()
        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)
        
        try:
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
        """Internal method for MCP server streaming execution."""
        # Custom JSON encoder
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)
        
        json._default_encoder = CustomJSONEncoder()
        
        async_client = self._create_async_client()
        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)
        
        try:
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
        except Exception as e:
            logger.error(f"Error in MCP streaming: {str(e)}")
            raise
    
    def _process_stream_event(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
        """Process streaming events. Override in subclasses for custom handling."""
        # Basic implementation - subclasses can override for more sophisticated processing
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
    
    def token_count(self, prompt: str) -> int:
        """
        Count tokens in prompt. Default implementation uses character approximation.
        Override in subclasses for model-specific tokenization.
        """
        return len(prompt) // 4