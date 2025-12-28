# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Agentic Node Architecture for Datus-agent.

This module provides a new agentic node system that supports session-based,
streaming interactions with tool integration and action history management.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from agents import SQLiteSession, Tool
from agents.mcp import MCPServerStdio

from datus.agent.node.node import Node
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.workflow import Workflow

logger = get_logger(__name__)


class AgenticNode(Node):
    """
    Base agentic node that provides session-based, streaming interactions
    with tool integration and automatic context management.
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: BaseInput = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
    ):
        """
        Initialize the agentic node.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (e.g., 'chat', 'gensql')
            input_data: Input data for the node
            agent_config: Agent configuration
            tools: List of function tools available to this node
            mcp_servers: Dictionary of MCP servers available to this node
        """
        # Initialize Node base class
        super().__init__(node_id, description, node_type, input_data, agent_config, tools)

        # AgenticNode-specific attributes
        self.mcp_servers = mcp_servers or {}
        self.actions: List[ActionHistory] = []
        self.session_id: Optional[str] = None
        self._session: Optional[SQLiteSession] = None
        self._session_tokens: int = 0
        self.last_summary: Optional[str] = None
        self.context_length: Optional[int] = None

        # Parse node configuration from agent.yml (available to all agentic nodes)
        self.node_config = self._parse_node_config(agent_config, self.get_node_name())

        # Initialize model: use node-specific model if configured, otherwise use default from agent_config
        if agent_config:
            model_name = self.node_config.get("model")  # Can be None, which will use active_model()
            self.model = LLMBaseModel.create_model(model_name=model_name, agent_config=agent_config)
            self.context_length = self.model.context_length() if self.model else None

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
                workspace_root=root_path,  # DEPRECATED: Use semantic_model_dir or sql_summary_dir instead
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
            logger.debug(f"Node configuration '{node_name}' not found in agent.yml, using default configuration")
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
            "agent_description",
            "prompt_version",
            "prompt_language",
            "tools",
            "mcp",
            "hooks",
            "rules",
            "max_turns",
            "workspace_root",
            "scoped_context",
            "scoped_kb_path",
            "adapter_type",
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

        # Normalize rules: convert dict items to strings (YAML parsing issue workaround)
        if "rules" in config and isinstance(config["rules"], list):
            normalized_rules = []
            for rule in config["rules"]:
                if isinstance(rule, dict):
                    # Convert dict to string format "key: value"
                    rule_str = ", ".join(f"{k}: {v}" for k, v in rule.items())
                    normalized_rules.append(rule_str)
                else:
                    normalized_rules.append(str(rule))
            config["rules"] = normalized_rules

        logger.info(f"Parsed node configuration for '{node_name}': {config}")
        return config

    def setup_input(self, workflow: "Workflow") -> Dict:
        """
        Setup input for agentic node from workflow context.

        Default implementation extracts common fields from workflow context
        and populates the input object. Subclasses can override for custom behavior.

        Args:
            workflow: Workflow instance containing context and task

        Returns:
            Dictionary with success status and message
        """
        if self.input is None:
            self.input = BaseInput()

        # Populate common fields from workflow context if input has these attributes
        if hasattr(self.input, "catalog"):
            self.input.catalog = workflow.task.catalog_name
        if hasattr(self.input, "database"):
            self.input.database = workflow.task.database_name
        if hasattr(self.input, "db_schema"):
            self.input.db_schema = workflow.task.schema_name
        if hasattr(self.input, "schemas"):
            self.input.schemas = workflow.context.table_schemas
        if hasattr(self.input, "metrics"):
            self.input.metrics = workflow.context.metrics

        return {"success": True, "message": f"Agentic node {self.type} input prepared"}

    def update_context(self, workflow: "Workflow") -> Dict:
        """
        Update workflow context with agentic node results.

        Default implementation stores SQL results if present.
        Subclasses can override for custom context updates.

        Args:
            workflow: Workflow instance to update

        Returns:
            Dictionary with success status and message
        """
        if not self.result:
            return {"success": False, "message": "No result to update context"}

        result = self.result

        # Store SQL generation results if present
        if hasattr(result, "sql") and result.sql:
            from datus.schemas.node_models import SQLContext

            new_record = SQLContext(
                sql_query=result.sql,
                explanation=getattr(result, "response", "") or getattr(result, "explanation", ""),
            )
            workflow.context.sql_contexts.append(new_record)

        return {"success": True, "message": "Agentic node context updated"}

    def execute(self) -> BaseResult:
        """
        Synchronous execution wrapper for agentic nodes.

        Agentic nodes are async by nature, so this wraps the async method
        to provide synchronous execution interface required by Node base class.

        Returns:
            BaseResult object with execution results
        """
        action_history_manager = ActionHistoryManager()

        async def _run_async():
            final_action = None
            async for action in self.execute_stream(action_history_manager):
                if action.status == ActionStatus.SUCCESS:
                    final_action = action
            return final_action

        try:
            # Get the final action from streaming execution
            final_action = asyncio.run(_run_async())

            # Extract result from final action output
            if final_action and final_action.output:
                output_data = final_action.output
                if isinstance(output_data, dict):
                    # Try to determine the result class from the subclass
                    result_class = self._get_result_class()
                    if result_class:
                        try:
                            self.result = result_class.model_validate(output_data)
                        except Exception as e:
                            logger.warning(f"Failed to validate result as {result_class.__name__}: {e}")
                            # Fallback: create a generic BaseResult
                            self.result = BaseResult(
                                success=output_data.get("success", True),
                                error=output_data.get("error"),
                            )
                    else:
                        # No specific result class, create generic BaseResult
                        self.result = BaseResult(
                            success=output_data.get("success", True),
                            error=output_data.get("error"),
                        )
                else:
                    # Output is already a BaseResult instance
                    self.result = output_data

            if not self.result:
                self.result = BaseResult(success=False, error="No result from execution")

            return self.result

        except Exception as e:
            logger.error(f"Agentic node execution error: {e}")
            self.result = BaseResult(success=False, error=str(e))
            return self.result

    def _get_result_class(self):
        """
        Get the result class for this node type.

        Subclasses can override this to return their specific result class.
        Default implementation tries to infer from common naming patterns.

        Returns:
            Result class or None if cannot determine
        """
        # Try to import and return the appropriate result class
        class_name = self.__class__.__name__

        # Map node class names to result class names
        result_class_map = {
            "ChatAgenticNode": "ChatNodeResult",
            "GenSQLAgenticNode": "GenSQLNodeResult",
            "CompareAgenticNode": "CompareResult",
        }

        result_class_name = result_class_map.get(class_name)
        if not result_class_name:
            return None

        try:
            # Try to import the result class from corresponding schema module
            if class_name == "ChatAgenticNode":
                from datus.schemas.chat_agentic_node_models import ChatNodeResult

                return ChatNodeResult
            elif class_name == "GenSQLAgenticNode":
                from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeResult

                return GenSQLNodeResult
            elif class_name == "CompareAgenticNode":
                from datus.schemas.compare_node_models import CompareResult

                return CompareResult
        except ImportError as e:
            logger.debug(f"Could not import result class {result_class_name}: {e}")
            return None

        return None

    @abstractmethod
    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the agentic node with streaming support.

        This method should be implemented by subclasses to provide specific
        functionality while using the common session and tool management.

        Input should be accessed from self.input instead of parameters.

        Args:
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

    def _resolve_workspace_root(self) -> str:
        """
        Resolve workspace_root with priority: node-specific > global storage > legacy > default.
        Expands ~ to user home directory if present.

        Returns:
            Resolved workspace_root path with ~ expanded
        """
        import os

        workspace_root = None

        # Priority: node-specific workspace_root > global storage.workspace_root > legacy > default "."
        node_workspace_root = self.node_config.get("workspace_root")
        if node_workspace_root:
            workspace_root = node_workspace_root
            logger.debug(f"Using node-specific workspace_root: {workspace_root}")
        elif (
            self.agent_config
            and hasattr(self.agent_config, "storage")
            and hasattr(self.agent_config.storage, "workspace_root")
        ):
            global_workspace_root = self.agent_config.storage.workspace_root
            if global_workspace_root:
                workspace_root = global_workspace_root
                logger.debug(f"Using global workspace_root: {workspace_root}")
        elif self.agent_config and hasattr(self.agent_config, "workspace_root"):
            # Fallback to old workspace_root location
            legacy_workspace_root = self.agent_config.workspace_root
            if legacy_workspace_root is not None:
                workspace_root = legacy_workspace_root
                logger.debug(f"Using legacy workspace_root: {workspace_root}")

        # Default to current directory if not configured
        if workspace_root is None:
            workspace_root = "."
            logger.debug("Using default workspace_root: .")

        # Expand ~ to user home directory
        expanded_path = os.path.expanduser(workspace_root)

        if expanded_path != workspace_root:
            logger.debug(f"Expanded workspace_root from '{workspace_root}' to '{expanded_path}'")

        return expanded_path
