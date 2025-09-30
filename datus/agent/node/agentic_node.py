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
        self.plan_hooks = None
        self.actions: List[ActionHistory] = []
        self.session_id: Optional[str] = None
        self._session: Optional[SQLiteSession] = None
        self._session_tokens: int = 0
        self.last_summary: Optional[str] = None

        # Parse node configuration from agent.yml (available to all agentic nodes)
        self.node_config = self._parse_node_config(agent_config, self.get_node_name())

        # Initialize the model using agent config
        if agent_config:
            model_name = self.node_config.get("model")
            # Create model with agentic-node-specific or default model
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

    def _get_system_prompt(
        self, conversation_summary: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> str:
        """
        Get the system prompt for this agentic node using PromptManager.

        The template name follows the pattern: {get_node_name()}_system_{version}

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use, overrides agent config version

        Returns:
            System prompt string loaded from the template

        Raises:
            DatusException: If template is not found
        """
        # Get prompt version from parameter, fallback to agent config, then use default
        version = prompt_version
        if version is None and self.agent_config and hasattr(self.agent_config, "prompt_version"):
            version = self.agent_config.prompt_version

        root_path = "."
        if self.agent_config and hasattr(self.agent_config, "workspace_root"):
            root_path = self.agent_config.workspace_root

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
                # Add conversation summary if available
                conversation_summary=conversation_summary,
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

    def _build_plan_prompt(self, original_prompt: str) -> str:
        """Build enhanced prompt for plan mode based on current phase."""
        # Check current phase and replan feedback
        current_phase = getattr(self.plan_hooks, "plan_phase", "generating") if self.plan_hooks else "generating"
        replan_feedback = getattr(self.plan_hooks, "replan_feedback", "") if self.plan_hooks else ""

        execution_prompt = (
            "After the plan has been confirmed, execute the pending steps.\n\n"
            + "Execution steps for each pending step:\n"
            + "1. FIRST: call todo_update_pending(todo_id) to mark step as pending (triggers user confirmation)\n"
            + "2. then execute the actual task (SQL queries, data processing, etc.)\n"
            + "3. then call todo_update_completed(todo_id) to mark step as completed\n\n"
            + "Start with the first pending step in the plan."
        )

        # Only enter replan mode if we have feedback AND we're still in generating phase
        if replan_feedback and current_phase == "generating":
            # REPLAN MODE: Generate revised plan
            plan_prompt_addition = (
                "\n\nREPLAN MODE\n"
                + f"Revise the current plan based on USER FEEDBACK: {replan_feedback}\n\n"
                + "STEPS:\n"
                + "1. FIRST: call todo_read to review the current plan, the completed and pending steps\n"
                + "2. then call todo_write to generate revised plan following these rules:\n"
                + "   - COMPLETED steps(if there are any): keep items as 'completed'\n"
                + "   - PENDING steps that are no longer needed: DISCARD (don't include in new plan)\n"
                + "   - PENDING steps that are still needed: keep as 'pending' or revise content\n"
                + "   - NEW steps(if there are any): add as 'pending'\n"
                + "3. Only include steps that are actually needed in the revised plan\n"
                + execution_prompt
            )
        elif current_phase == "generating":
            # INITIAL PLANNING PHASE
            plan_prompt_addition = (
                "\n\nPLAN MODE - PLANNING PHASE\n"
                + "Task: Break down user request into 3-8 steps.\n\n"
                + "call todo_write to generate complete todo list (3-8 steps)\n"
                + 'Example: todo_write(\'[{"content": "Connect to database", "status": "pending"}, '
                + '{"content": "Query data", "status": "pending"}]\')'
                + execution_prompt
            )
        else:
            # Default fallback
            plan_prompt_addition = (
                "\n\nPLAN MODE\n" + "Check todo_read to see current plan status and proceed accordingly."
            )

        return original_prompt + plan_prompt_addition

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"{self.get_node_name()}_session_{str(uuid.uuid4())[:8]}"

    def _get_or_create_session(self) -> tuple[SQLiteSession, Optional[str]]:
        """
        Get or create the session for this node.

        Returns:
            Tuple of (session, summary) where summary is the conversation summary
            from previous compact (if any), None otherwise
        """
        summary = None

        if self._session is None:
            if self.session_id is None:
                self.session_id = self._generate_session_id()
                logger.info(f"Generated new session ID: {self.session_id}")

            if self.model:
                self._session = self.model.create_session(self.session_id)
                logger.debug(f"Created session: {self.session_id}")

                # If we have a summary from previous compact, return it
                if self.last_summary:
                    summary = self.last_summary
                    logger.debug(f"Returning summary from previous compact: {len(summary)} chars")

                    # Clear the summary after using it once
                    self.last_summary = None

        return self._session, summary

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

        # Update SQLite session with current token count via model's session manager
        if self.model and hasattr(self.model, "session_manager") and self.session_id:
            self.model.session_manager.update_session_tokens(self.session_id, self._session_tokens)

    async def _manual_compact(self) -> dict:
        """
        Manually compact the session by summarizing conversation history.
        This clears the session and stores summary for next session creation.

        Returns:
            Dict with success, summary, and summary_token count
        """
        if not self.model or not self._session:
            logger.warning("Cannot compact: no model or session available")
            return {"success": False, "summary": "", "summary_token": 0}

        try:
            logger.info(f"Starting manual compacting for session {self.session_id}")

            # Store old session info for logging
            old_session_id = self.session_id
            old_tokens = self._session_tokens

            # 1. Generate summary using LLM with existing session
            summarization_prompt = (
                "Summarize our conversation up to this point. The summary should be a concise yet comprehensive "
                "overview of all key topics, questions, answers, and important details discussed. This summary "
                "will replace the current chat history to conserve tokens, so it must capture everything "
                "essential to understand the context and continue our conversation effectively as if no "
                "information was lost."
            )

            try:
                result = await self.model.generate_with_tools(
                    prompt=summarization_prompt, session=self._session, max_turns=1, temperature=0.3, max_tokens=2000
                )
                summary = result.get("content", "")
                summary_token = result.get("usage", {}).get("output_tokens", 0)
                logger.debug(f"Generated summary: {len(summary)} characters, {summary_token} tokens")
            except Exception as e:
                logger.error(f"Failed to generate summary with LLM: {e}")
                return {"success": False, "summary": "", "summary_token": 0}

            # 2. Store summary for next session creation
            self.last_summary = summary
            logger.info(f"Stored summary for next session: {len(summary)} characters")

            # 3. Clear current session
            if old_session_id:
                try:
                    self.model.delete_session(old_session_id)
                    logger.debug(f"Deleted old session: {old_session_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete old session {old_session_id}: {e}")

            # Clear session references
            self.session_id = None
            self._session = None

            # Reset token count for new session
            self._session_tokens = 0

            logger.info(
                f"Manual compacting completed. Cleared session: {old_session_id}, "
                f"Token reset: {old_tokens} -> 0, Summary stored: {len(summary)} chars"
            )
            return {"success": True, "summary": summary, "summary_token": summary_token}

        except Exception as e:
            logger.error(f"Manual compacting failed: {e}")
            return {"success": False, "summary": "", "summary_token": 0}

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

    def _parse_node_config(self, agent_config: Optional[AgentConfig], node_name: str) -> dict:
        """
        Parse node configuration from agent.yml.

        Args:
            agent_config: Agent configuration
            node_name: Name of the node configuration

        Returns:
            Dictionary containing node configuration
        """
        if not agent_config or not hasattr(agent_config, "agentic_nodes"):
            return {}

        nodes_config = agent_config.agentic_nodes
        if node_name not in nodes_config:
            logger.warning(f"Node configuration '{node_name}' not found in agent.yml")
            return {}

        node_config = nodes_config[node_name]

        # Extract configuration attributes
        config = {}

        # Basic node config attributes
        if isinstance(node_config, dict):
            config["model"] = node_config.get("model")
        elif hasattr(node_config, "model"):
            config["model"] = node_config.model

        # Check direct attributes on node_config
        direct_attributes = [
            "system_prompt",
            "prompt_version",
            "prompt_language",
            "tools",
            "mcp",
            "rules",
            "max_turns",
            "workspace_root",
        ]
        for attr in direct_attributes:
            # Handle both dict and object access patterns
            if attr not in config:
                value = None
                if isinstance(node_config, dict):
                    value = node_config.get(attr)
                elif hasattr(node_config, attr):
                    value = getattr(node_config, attr)

                if value is not None:
                    config[attr] = value

        logger.info(f"Parsed node configuration for '{node_name}': {config}")
        return config

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
