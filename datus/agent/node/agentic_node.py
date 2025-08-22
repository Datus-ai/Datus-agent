"""
Agentic Node Architecture for Datus-agent.

This module provides a new agentic node system that supports session-based,
streaming interactions with tool integration and action history management.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import SQLiteSession, Tool
from agents.mcp import MCPServerStdio

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class Plan:
    """Placeholder for future plan implementation."""

    def __init__(self):
        self.steps = []
        self.current_step = 0


class AgenticNode(ABC):
    """
    Base agentic node that provides session-based, streaming interactions
    with tool integration and automatic context management.

    This is a new architecture that doesn't inherit from the existing Node class
    and provides more flexible, agentic capabilities.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the agentic node.

        Args:
            tools: List of function tools available to this node
            mcp_servers: Dictionary of MCP servers available to this node
            agent_config: Agent configuration
        """
        self.tools = tools or []
        self.mcp_servers = mcp_servers or {}
        self.agent_config = agent_config
        self.plan = Plan()
        self.actions: List[ActionHistory] = []
        self.session_id: Optional[str] = None
        self._session: Optional[SQLiteSession] = None
        self._session_tokens: int = 0

        # Initialize the model using agent config
        if agent_config:
            model_name = None
            nodes_config = agent_config.nodes if hasattr(agent_config, "nodes") else {}

            # Get node name for config lookup
            node_name = self.get_node_name()

            # Check for node-specific model configuration
            if node_name in nodes_config:
                node_config = nodes_config[node_name]
                if hasattr(node_config, "model"):
                    model_name = node_config.model

            # Create model with node-specific or default model
            self.model = LLMBaseModel.create_model(model_name=model_name, agent_config=agent_config)
            # Store context length for efficient token validation
            self.context_length = self.model.context_length() if self.model else None
        else:
            self.model = None
            self.context_length = None

        # Generate system prompt using prompt manager
        self.system_prompt = self._get_system_prompt()

    def get_node_name(self) -> str:
        """
        Get the template name for this agentic node. Overwrite this method if you need a special name

        Default implementation extracts from class name:
        - ChatAgenticNode -> "chat"
        - GenerateAgenticNode -> "generate"

        Returns:
            Node name that will be used to construct the full template filename and use in agent.yml
        """
        class_name = self.__class__.__name__
        # Remove "AgenticNode" suffix and convert to lowercase
        if class_name.endswith("AgenticNode"):
            template_name = class_name[:-11]  # Remove "AgenticNode" (11 characters)
        else:
            template_name = class_name

        return template_name.lower()

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this agentic node using PromptManager.

        The template name follows the pattern: {get_node_name()}_system_{version}

        Returns:
            System prompt string loaded from the template

        Raises:
            DatusException: If template is not found
        """
        # Get prompt version from agent config or use default
        version = None
        if self.agent_config and hasattr(self.agent_config, "prompt_version"):
            version = self.agent_config.prompt_version

        root_path = None
        if self.agent_config and hasattr(self.agent_config, "nodes") and "chat" in self.agent_config.nodes:
            chat_node_config = self.agent_config.nodes["chat"]
            if chat_node_config.input and hasattr(chat_node_config.input, "workspace_root"):
                root_path = chat_node_config.input.workspace_root

        # Construct template name: {template_name}_system_{version}
        template_name = f"{self.get_node_name()}_system"

        try:
            # Use prompt manager to render the template
            return prompt_manager.render_template(
                template_name=template_name,
                version=version,
                # Add common template variables
                agent_config=self.agent_config,
                namespace=getattr(self.agent_config, "current_namespace", None) if self.agent_config else None,
                workspace_root=root_path,
            )

        except FileNotFoundError as e:
            # Template not found - throw DatusException
            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": version or "latest"},
            ) from e
        except Exception as e:
            # Other template errors - wrap in DatusException
            logger.error(f"Template loading error for '{template_name}': {e}")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"{self.get_node_name()}_session_{str(uuid.uuid4())[:8]}"

    def _get_or_create_session(self) -> SQLiteSession:
        """Get or create the session for this node."""
        if self._session is None:
            if self.session_id is None:
                self.session_id = self._generate_session_id()
                logger.info(f"Generated new session ID: {self.session_id}")

            if self.model:
                self._session = self.model.create_session(self.session_id)
                logger.debug(f"Created session: {self.session_id}")

        return self._session

    def _count_session_tokens(self) -> int:
        """
        Count the total tokens in the current session.
        Returns the cumulative token count stored in self._session_tokens.

        Returns:
            Total token count in the session
        """
        return self._session_tokens

    def _add_session_tokens(self, tokens_used: int) -> None:
        """
        Add tokens to the current session count.
        Validates that the total doesn't exceed the model's context length.

        Args:
            tokens_used: Number of tokens to add to the session count
        """
        if tokens_used <= 0:
            return

        # Validate against context length if available
        if self.context_length and (self._session_tokens + tokens_used) > self.context_length:
            logger.warning(
                f"Cannot add {tokens_used} tokens: would exceed context length "
                f"({self._session_tokens + tokens_used} > {self.context_length})"
            )
            return

        self._session_tokens += tokens_used
        logger.debug(f"Added {tokens_used} tokens to session. Total: {self._session_tokens}")

    async def _manual_compact(self) -> bool:
        """
        Manually compact the session by summarizing conversation history.
        This creates a new session and resets token count to 0.

        Returns:
            True if compacting was successful, False otherwise
        """
        if not self.model or not self._session:
            logger.warning("Cannot compact: no model or session available")
            return False

        try:
            logger.info(f"Starting manual compacting for session {self.session_id}")

            # Store old session info for logging
            old_session_id = self.session_id
            old_tokens = self._session_tokens

            # Get current session content
            # This would involve:
            # 1. Retrieving all messages from the current session
            # 2. Generating a summary using the LLM
            # 3. Creating a new session with the summary as context
            # 4. Clearing the old session

            # Create new session (force recreation)
            self.session_id = self._generate_session_id()
            self._session = None  # Force recreation on next access

            # Reset token count for new session
            self._session_tokens = 0

            logger.info(
                f"Manual compacting completed. Session: {old_session_id} -> {self.session_id}, Token count reset: {old_tokens} -> 0"
            )
            return True

        except Exception as e:
            logger.error(f"Manual compacting failed: {e}")
            return False

    async def _auto_compact(self) -> bool:
        """
        Automatically compact when session approaches token limit (~90%).

        Returns:
            True if compacting was triggered and successful, False otherwise
        """
        if not self.model or not self.context_length:
            return False

        try:
            current_tokens = self._count_session_tokens()

            if current_tokens > (self.context_length * 0.9):
                logger.info(f"Auto-compacting triggered: {current_tokens}/{self.context_length} tokens")
                return await self._manual_compact()  # Will reset tokens to 0

            return False

        except Exception as e:
            logger.error(f"Auto-compact check failed: {e}")
            return False

    @abstractmethod
    async def execute_stream(
        self, user_prompt: str, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the agentic node with streaming support.

        This method should be implemented by subclasses to provide specific
        functionality while using the common session and tool management.

        Args:
            user_prompt: User input prompt
            action_history_manager: Optional action history manager for tracking

        Yields:
            ActionHistory: Progress updates during execution
        """

    def clear_session(self) -> None:
        """Clear the current session and reset token count."""
        if self.model and self.session_id:
            self.model.clear_session(self.session_id)
            self._session = None
            self._session_tokens = 0  # Reset token count
            logger.info(f"Cleared session: {self.session_id}, tokens reset to 0")

    def delete_session(self) -> None:
        """Delete the current session completely and reset token count."""
        if self.model and self.session_id:
            self.model.delete_session(self.session_id)
            self._session = None
            self.session_id = None
            self._session_tokens = 0  # Reset token count
            logger.info("Deleted session, tokens reset to 0")

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.

        Returns:
            Dictionary with session information
        """
        if not self.session_id:
            return {"session_id": None, "active": False}

        current_tokens = self._count_session_tokens()

        return {
            "session_id": self.session_id,
            "active": self._session is not None,
            "token_count": current_tokens,
            "action_count": len(self.actions),
            "context_usage_ratio": current_tokens / self.context_length if self.context_length else 0,
            "context_remaining": self.context_length - current_tokens if self.context_length else 0,
            "context_length": self.context_length,
        }
