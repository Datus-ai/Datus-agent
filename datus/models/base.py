# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import hashlib
import multiprocessing
import os
import platform
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, AsyncGenerator, ClassVar, Dict, List, Optional, Tuple, Union

from agents import SQLiteSession, Tool
from agents.mcp import MCPServerStdio

from datus.configuration.agent_config import AgentConfig, ModelConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.constants import LLMProvider
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def configure_multiprocessing_start_method() -> None:
    """Set a safe multiprocessing start method for the current platform."""
    try:
        if platform.system() == "Windows":
            multiprocessing.set_start_method("spawn", force=True)
        else:
            multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        # set_start_method can only be called once
        pass


# Fix multiprocessing issues with PyTorch/sentence-transformers in Python 3.12
configure_multiprocessing_start_method()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LLMBaseModel(ABC):  # Changed from BaseModel to LLMBaseModel
    """
    Abstract base class for all language model implementations.
    Provides a common interface for different LLM providers.
    """

    MODEL_TYPE_MAP: ClassVar[Dict[str, str]] = {
        LLMProvider.DEEPSEEK: "DeepSeekModel",
        LLMProvider.QWEN: "QwenModel",
        LLMProvider.OPENAI: "OpenAIModel",
        LLMProvider.CLAUDE: "ClaudeModel",
        LLMProvider.GEMINI: "GeminiModel",
        LLMProvider.KIMI: "KimiModel",
        LLMProvider.CODEX: "CodexModel",
        LLMProvider.OPENROUTER: "OpenRouterModel",
        LLMProvider.MINIMAX: "MiniMaxModel",
        LLMProvider.GLM: "GLMModel",
    }

    # Module-level LRU cache for instantiated models. Keyed on the
    # configuration fingerprint so ``/model`` can flip ``active_model()``
    # between ``(openai, gpt-4.1)`` and ``(kimi, kimi-k2.5)`` without
    # repeatedly paying the client-construction cost (LiteLLM + tokenizer
    # download). Size is intentionally tiny — users seldom juggle more
    # than a handful of models — so stale entries are evicted quickly.
    _MODEL_CACHE_MAXSIZE: ClassVar[int] = 4
    _MODEL_CACHE: ClassVar["OrderedDict[Tuple, LLMBaseModel]"] = OrderedDict()
    _MODEL_CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_config: ModelConfig, **kwargs):
        """Initialize model with configuration and parameters"""
        self.model_config = model_config  # Model configuration
        # Initialize session manager for all models
        self._session_manager = None
        self.session_dir = kwargs.get("session_dir")
        self.session_scope = kwargs.get("session_scope")

    @classmethod
    def create_model(cls, agent_config: AgentConfig, model_name: str = None, **kwargs) -> "LLMBaseModel":
        """Resolve the active LLM and return an instance (cached per config).

        Caching is keyed on the fingerprint of the resolved
        :class:`ModelConfig` (type, model, base_url, api_key digest,
        auth_type, session scope). That way a ``/model`` switch produces
        a different key and a fresh instance; subsequent calls under the
        same selection reuse the cached client. The cache is process-wide
        and bounded to ``_MODEL_CACHE_MAXSIZE`` entries so it is safe to
        hold across long-running sessions.
        """
        if not model_name or model_name == "default":
            target_config = agent_config.active_model()
        elif model_name in agent_config.models:
            target_config = agent_config.model_config(model_name)
        else:
            raise KeyError(f"Model {model_name} not found in agent_config")

        model_type = target_config.type

        if (model_class_name := cls.MODEL_TYPE_MAP.get(model_type)) is None:
            raise KeyError(f"Unsupported model type: {model_type}")

        scope = kwargs.get("scope")
        api_key_digest = (
            hashlib.sha1((target_config.api_key or "").encode("utf-8")).hexdigest()[:12]
            if target_config.api_key
            else ""
        )
        cache_key: Tuple = (
            model_type,
            target_config.model,
            target_config.base_url or "",
            api_key_digest,
            target_config.auth_type,
            scope or "",
            # Include reasoning-related fields so ``/effort`` and a toggled
            # ``enable_thinking`` produce a fresh adapter instead of reusing
            # one bound to the previous effort level.
            bool(target_config.enable_thinking),
            target_config.reasoning_effort or "",
        )

        with cls._MODEL_CACHE_LOCK:
            cached = cls._MODEL_CACHE.get(cache_key)
            if cached is not None:
                cls._MODEL_CACHE.move_to_end(cache_key)
                return cached

        module = __import__(f"datus.models.{model_type}_model", fromlist=[model_class_name])
        model_class = getattr(module, model_class_name)
        instance = model_class(
            model_config=target_config,
            session_dir=agent_config.session_dir,
            session_scope=scope,
        )

        with cls._MODEL_CACHE_LOCK:
            cls._MODEL_CACHE[cache_key] = instance
            cls._MODEL_CACHE.move_to_end(cache_key)
            while len(cls._MODEL_CACHE) > cls._MODEL_CACHE_MAXSIZE:
                cls._MODEL_CACHE.popitem(last=False)
        return instance

    @abstractmethod
    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """
        Generate a response from the language model.

        Args:
            prompt: The input prompt to send to the model
            enable_thinking: Enable thinking mode for hybrid models (default: False)
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """

    @abstractmethod
    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a response and ensure it conforms to the provided JSON schema.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            A dictionary representing the JSON response
        """

    @abstractmethod
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
        """Generate response with unified tool support.

        Args:
            prompt: Input prompt(user prompt)
            tools: Optional regular tools to use
            mcp_servers: Optional MCP servers to use
            instruction: System instruction(system prompt)
            output_type: Expected output type
            max_turns: Maximum conversation turns
            session: Optional session for multi-turn context
            **kwargs: Additional parameters

        Returns:
            Result with content and sql_contexts
        """

    @abstractmethod
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
        interrupt_controller=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate response with streaming and tool support.

        This replaces generate_with_mcp_stream and supports both MCP servers and regular tools.

        Args:
            prompt: Input prompt
            mcp_servers: Optional MCP servers
            tools: Optional regular tools
            instruction: System instruction
            output_type: Expected output type
            max_turns: Maximum turns
            session: Optional session for multi-turn context
            action_history_manager: Action history manager for streaming
            hooks: Optional hooks for tool interception
            **kwargs: Additional parameters

        Yields:
            ActionHistory objects for streaming updates
        """

    def set_context(self, workflow=None, current_node=None):
        """
        Set workflow and node context for potential trace saving.
        """
        self.workflow = workflow
        self.current_node = current_node

    @abstractmethod
    def token_count(self, prompt: str) -> int:
        pass

    @abstractmethod
    def context_length(self) -> Optional[int]:
        """
        Get the context length for the model.

        Returns:
            Context length, or None if unavailable
        """

    def to_dict(self) -> Dict[str, str]:
        return {"model_name": self.model_config.model}

    async def test_connection(self, timeout: float = 10.0) -> Tuple[bool, str]:
        """Probe the model with a 1-token request; return (ok, error_message).

        Used by ``/model`` and the ``datus init`` wizard to verify provider
        credentials before persisting them. The default implementation runs
        the synchronous :meth:`generate` inside a worker thread wrapped in
        ``asyncio.wait_for`` so both blocking HTTP clients and genuinely
        async providers converge on the same coroutine signature. Subclasses
        (e.g. Claude subscription) override when their endpoint rejects
        ``max_tokens=1``.
        """
        try:
            probe = asyncio.to_thread(self.generate, "hi", max_tokens=1)
            response = await asyncio.wait_for(probe, timeout=timeout)
            if response is None or not str(response).strip():
                return False, "Empty response from model"
            return True, ""
        except asyncio.TimeoutError:
            return False, f"Timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    @property
    def session_manager(self):
        """Lazy initialization of session manager."""
        if self._session_manager is None:
            from datus.models.session_manager import SessionManager

            self._session_manager = SessionManager(session_dir=self.session_dir, scope=self.session_scope)
        return self._session_manager

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
