# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenExtKnowledgeAgenticNode implementation for external knowledge generation workflow.

This module provides a specialized implementation of AgenticNode focused on
business search_text and concept management with support for filesystem tools,
generation tools, and hooks.
"""

from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import pandas as pd

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.compare_agentic_node import CompareAgenticNode
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput, ExtKnowledgeNodeResult
from datus.schemas.node_models import SQLContext, SqlTask
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.utils.benchmark_utils import ComparisonOutcome, TableComparator
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)


@dataclass
class VerifyResult:
    """Result of SQL verification without suggestions."""

    success: bool
    match_rate: float
    error: Optional[str] = None
    user_df: Optional[pd.DataFrame] = None
    gold_df: Optional[pd.DataFrame] = None
    outcome: Optional[ComparisonOutcome] = None


class GenExtKnowledgeAgenticNode(AgenticNode):
    """
    External knowledge generation agentic node with enhanced configuration.

    This node provides specialized business search_text and concept management with:
    - Enhanced system prompt with template variables
    - Filesystem tools for file operations
    - Generation tools for knowledge ID generation
    - Hooks support for custom behavior
    - Subject tree management with 3-level priority
    - Session-based conversation management
    """

    def __init__(
        self,
        node_name: str,
        agent_config: Optional[AgentConfig] = None,
        execution_mode: str = "interactive",
        build_mode: str = "incremental",
        subject_tree: Optional[list] = None,
    ):
        """
        Initialize the GenExtKnowledgeAgenticNode.

        Args:
            node_name: Name of the node configuration in agent.yml (should be "gen_ext_knowledge")
            agent_config: Agent configuration
            execution_mode: Execution mode - "interactive" (default) or "workflow"
            build_mode: "overwrite" or "incremental" (default: "incremental")
            subject_tree: Optional predefined subject tree categories
        """
        self.configured_node_name = node_name
        self.execution_mode = execution_mode
        self.build_mode = build_mode
        self.subject_tree = subject_tree

        # Get max_turns from agentic_nodes configuration
        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        # Verification retry configuration and state tracking
        self.max_verification_retries = 3
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_verification_retries = agentic_node_config.get("max_verification_retries", 3)

        # Verification state tracking
        self._verification_passed: bool = False
        self._last_verification_result: Optional[VerifyResult] = None
        self._verification_attempt_count: int = 0

        path_manager = get_path_manager()
        self.ext_knowledge_dir = str(path_manager.ext_knowledge_path(agent_config.current_namespace))

        from datus.configuration.node_type import NodeType

        node_type = NodeType.TYPE_EXT_KNOWLEDGE

        # Call parent constructor first to set up node_config
        super().__init__(
            node_id="ext_knowledge_node",
            description="External knowledge generation node",
            node_type=node_type,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
        )

        # Initialize external knowledge storage for context queries
        from datus.storage.ext_knowledge.store import ExtKnowledgeRAG

        self.ext_knowledge_store = ExtKnowledgeRAG(agent_config)

        # Setup tools based on hardcoded configuration
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.hooks = None
        self.setup_tools()

        logger.info(f"Hooks after setup: {self.hooks} (type: {type(self.hooks)})")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this external knowledge agentic node.

        Returns:
            The configured node name from agent.yml
        """
        return self.configured_node_name

    def setup_tools(self):
        """Setup tools based on hardcoded configuration."""
        if not self.agent_config:
            return

        self.tools = []

        # Hardcoded tool configuration: specific methods from generation_tools and filesystem_tools
        # filesystem_tools.read_file, filesystem_tools.read_multiple_files, filesystem_tools.write_file,
        # filesystem_tools.edit_file, filesystem_tools.list_directory
        # Chat node uses all available tools by default
        db_manager = db_manager_instance(self.agent_config.namespaces)
        self.conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
        self.db_func_tool = DBFuncTool(self.conn, agent_config=self.agent_config)
        self.context_search_tools = ContextSearchTools(self.agent_config)
        if self.db_func_tool:
            self.tools.extend(self.db_func_tool.available_tools())
        if self.context_search_tools:
            self.tools.extend(self.context_search_tools.available_tools())
        self._setup_specific_generation_tools()
        self._setup_specific_filesystem_tool()

        logger.info(
            f"Setup {len(self.tools)} tools for {self.configured_node_name}: {[tool.name for tool in self.tools]}"
        )

        # Setup hooks (only in interactive mode)
        if self.execution_mode == "interactive":
            self._setup_hooks()

    def _setup_specific_generation_tools(self):
        """Setup specific generation tools: verify_sql, end_knowledge_generation."""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.generation_tools = GenerationTools(self.agent_config)

            # Add verify_sql tool for SQL result verification (replaces get_gold_sql)
            self.tools.append(trans_to_function_tool(self.verify_sql))

            # Add end_knowledge_generation tool to finalize and verify completion
            # self.tools.append(trans_to_function_tool(self.end_knowledge_generation))
        except Exception as e:
            logger.error(f"Failed to setup specific generation tools: {e}")

    def _reset_verification_state(self):
        """Reset verification state for a new agentic loop attempt."""
        self._verification_passed = False
        self._last_verification_result = None
        logger.debug("Verification state reset for new attempt")

    def _get_retry_prompt(self, attempt: int) -> str:
        """
        Generate a retry prompt to inject when verification failed.

        Args:
            attempt: Current retry attempt number (1-based)

        Returns:
            Prompt string to inject for retry
        """
        last_result = self._last_verification_result
        match_rate = last_result.match_rate if last_result else 0.0

        return f"""[VERIFICATION RETRY - Attempt {attempt}/{self.max_verification_retries}]

Your previous SQL verification FAILED with match_rate={match_rate * 100:.1f}%.

IMPORTANT: You MUST create or correct the external knowledge to fix the SQL.

Actions required:
1. Review the suggestions and column differences from the last verify_sql call
2. Identify what knowledge is missing or incorrect
3. Use edit_file to MODIFY existing knowledge entries in the external knowledge file,
   or use write_file to CREATE NEW ones if no file exists yet
4. Based on the new/corrected knowledge, modify your SQL to match expected results
5. Call verify_sql again with your corrected SQL

Focus on adding or fixing knowledge entries that help generate the correct SQL.
Do NOT give up. Continue iterating until verify_sql returns success=1.
"""

    def _verify_result(self, sql: str) -> VerifyResult:
        """
        Core SQL verification logic - compare results without generating suggestions.

        Used by both verify_sql tool and AccomplishHook.

        Args:
            sql: The SQL to validate.

        Returns:
            VerifyResult with success, match_rate, and optional error/dataframes.
        """
        # Check if reference SQL is available
        if not hasattr(self, "_gold_sql") or not self._gold_sql:
            return VerifyResult(success=True, match_rate=1.0)

        connector = self.conn

        # Execute user SQL
        try:
            user_result = connector.execute_query(sql, result_format="pandas")
            if not user_result.success:
                return VerifyResult(success=False, match_rate=0.0, error=f"SQL execution failed: {user_result.error}")
            user_df = user_result.sql_return
            if not isinstance(user_df, pd.DataFrame):
                user_df = pd.DataFrame(user_df) if user_result.sql_return else pd.DataFrame()
        except Exception as e:
            return VerifyResult(success=False, match_rate=0.0, error=str(e))

        # Execute gold SQL
        try:
            gold_result = connector.execute_query(self._gold_sql, result_format="pandas")
            if not gold_result.success:
                return VerifyResult(success=False, match_rate=0.0, error=f"Gold SQL error: {gold_result.error}")
            gold_df = gold_result.sql_return
            if not isinstance(gold_df, pd.DataFrame):
                gold_df = pd.DataFrame(gold_df) if gold_result.sql_return else pd.DataFrame()
        except Exception as e:
            return VerifyResult(success=False, match_rate=0.0, error=f"Gold SQL error: {e}")

        # Compare using TableComparator
        comparator = TableComparator()
        outcome = comparator.compare(user_df, gold_df)

        return VerifyResult(
            success=(outcome.match_rate == 1.0),
            match_rate=outcome.match_rate,
            user_df=user_df,
            gold_df=gold_df,
            outcome=outcome,
        )

    def verify_sql(self, sql: str) -> FuncToolResult:
        """
        Validate SQL against a hidden reference. The reference SQL is not exposed.

        This tool compares execution results of the provided SQL with a hidden reference.
        The model cannot see the reference SQL - it can only learn from comparison feedback
        (match rate, column differences, data preview, and improvement suggestions).

        Args:
            sql: The SQL to validate.

        Returns:
            FuncToolResult:
                - success=1: SQL matches the reference, or no reference available
                - success=0: Mismatch detected, includes suggestions for improvement
        """
        # Use _verify_result for core verification logic
        result = self._verify_result(sql)

        # Update verification state for retry logic
        self._last_verification_result = result
        self._verification_passed = result.success
        logger.info(f"Verification status updated: passed={self._verification_passed}, match_rate={result.match_rate}")

        # No reference available
        if not hasattr(self, "_gold_sql") or not self._gold_sql:
            self._verification_passed = True  # Mark as passed when no gold_sql
            return FuncToolResult(
                success=1,
                result="No reference available. Your SQL will be accepted.",
            )

        # Success - SQL matches
        if result.success:
            return FuncToolResult(
                success=1,
                result={
                    "message": "SQL verification PASSED!",
                    "match_rate": 1.0,
                    "your_result_shape": f"{result.user_df.shape[0]} rows x {result.user_df.shape[1]} columns"
                    if result.user_df is not None
                    else "N/A",
                },
            )

        # Failure - generate suggestions
        logger.warning(f"SQL verification failed: match_rate={result.match_rate}")

        # Prepare user/gold result strings for suggestions
        user_result_str = (
            result.user_df.to_csv(index=False) if result.user_df is not None and not result.user_df.empty else ""
        )
        gold_result_str = (
            result.gold_df.to_csv(index=False) if result.gold_df is not None and not result.gold_df.empty else ""
        )

        suggestions = self._generate_compare_suggestions(
            user_sql=sql,
            gold_sql=self._gold_sql,
            user_result=user_result_str,
            gold_result=gold_result_str,
            user_error=result.error,
        )

        # Return error result with suggestions
        if result.error:
            return FuncToolResult(
                success=0,
                error=f"SQL execution error: {result.error}",
                result={
                    "match_rate": 0,
                    "suggestions": suggestions,
                },
            )

        outcome = result.outcome
        return FuncToolResult(
            success=0,
            error=f"SQL verification FAILED! Match rate: {result.match_rate * 100:.1f}%",
            result={
                "match_rate": result.match_rate,
                "your_result_shape": f"{outcome.actual_shape[0] if outcome and outcome.actual_shape else 0} rows x "
                f"{outcome.actual_shape[1] if outcome and outcome.actual_shape else 0} columns",
                "expected_result_shape": (
                    f"{outcome.expected_shape[0] if outcome and outcome.expected_shape else 0} rows x "
                    f"{outcome.expected_shape[1] if outcome and outcome.expected_shape else 0} columns"
                ),
                "column_differences": {
                    "matched": outcome.matched_columns if outcome else [],
                    "missing": outcome.missing_columns if outcome else [],
                    "extra": outcome.extra_columns if outcome else [],
                },
                "data_preview_yours": outcome.actual_preview if outcome else None,
                "data_preview_expected": outcome.expected_preview if outcome else None,
                "suggestions": suggestions,
            },
        )

    def _generate_compare_suggestions(
        self,
        user_sql: str,
        gold_sql: str,
        user_result: str,
        gold_result: str,
        user_error: str = None,
        generated_knowledge: list = None,
    ) -> dict:
        """
        Generate suggestions for SQL improvement using CompareAgenticNode.

        Args:
            user_sql: The user's SQL query.
            gold_sql: The expected gold SQL query.
            user_result: The execution result of user's SQL (as string).
            gold_result: The execution result of gold SQL (as string).
            user_error: Error message if user SQL failed to execute.
            generated_knowledge: List of already generated knowledge items.

        Returns:
            dict: Contains 'explanation' and 'suggest' from CompareAgenticNode.
        """
        try:
            # Build SqlTask from agent_config
            sql_task = SqlTask(
                database_type=self.agent_config.database_type if hasattr(self.agent_config, "database_type") else "",
                database_name=self.agent_config.current_database
                if hasattr(self.agent_config, "current_database")
                else "",
                task=getattr(self, "_current_question", "SQL verification task"),
                external_knowledge=str(generated_knowledge) if generated_knowledge else "",
            )

            # Build SQLContext from user's SQL
            sql_context = SQLContext(
                sql_query=user_sql,
                explanation="User generated SQL for verification",
                sql_return=user_result,
                sql_error=user_error or "",
            )

            # Build CompareInput
            compare_input = CompareInput(
                sql_task=sql_task,
                sql_context=sql_context,
                expectation=f"Expected SQL:\n{gold_sql}\n\nExpected Result:\n{gold_result}",
            )

            # Use CompareAgenticNode to generate suggestions
            _, _, messages = CompareAgenticNode._prepare_prompt_components(compare_input)
            raw_result = self.model.generate_with_json_output(messages)
            result_dict = CompareAgenticNode._parse_comparison_output(raw_result)

            return {
                "explanation": result_dict.get("explanation", "No explanation provided"),
                "suggest": result_dict.get("suggest", "No suggestions provided"),
            }

        except Exception as e:
            logger.error(f"Failed to generate compare suggestions: {e}")
            return {
                "explanation": f"Failed to generate suggestions: {str(e)}",
                "suggest": "Please manually compare your SQL with the gold SQL and identify the differences.",
            }

    def _setup_specific_filesystem_tool(self):
        """Setup specific filesystem tools"""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.filesystem_func_tool = FilesystemFuncTool(root_path=self.ext_knowledge_dir)
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.read_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.edit_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.write_file))
        except Exception as e:
            logger.error(f"Failed to setup specific filesystem tool: {e}")

    def _setup_hooks(self):
        """Setup hooks (hardcoded to generation_hooks)."""
        try:
            broker = self._get_or_create_broker()
            self.hooks = GenerationHooks(broker=broker, agent_config=self.agent_config)
            logger.info("Setup hooks: generation_hooks")
        except Exception as e:
            logger.error(f"Failed to setup generation_hooks: {e}")

    def _get_existing_subject_trees(self) -> list:
        """
        Query existing subject_path values from external knowledge storage.

        Returns:
            List of unique subject_path values as List[str]
        """
        try:
            # Get all subject paths from the subject tree
            subject_paths = sorted(self.ext_knowledge_store.store.get_subject_tree_flat())
            logger.debug(f"Found {len(subject_paths)} unique external knowledge subject_paths")
            return subject_paths
        except Exception as e:
            logger.error(f"Error getting existing subject_paths: {e}")
            return []

    def _prepare_template_context(self, user_input: ExtKnowledgeNodeInput) -> dict:
        """
        Prepare template context variables for the external knowledge generation template.

        Args:
            user_input: User input

        Returns:
            Dictionary of template variables
        """
        context = {}

        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["ext_knowledge_dir"] = self.ext_knowledge_dir

        # Priority 1: User-specified subject_path (highest priority)
        if user_input.subject_path:
            context["has_user_specified_subject"] = True
            context["user_subject_path"] = user_input.subject_path
            logger.info(f"Using user-specified subject_path: {user_input.subject_path}")
        else:
            context["has_user_specified_subject"] = False

        # Priority 2 & 3: Handle subject_tree context based on whether predefined or query from storage
        if self.subject_tree:
            # Predefined mode: use provided subject_tree (Priority 2)
            context["has_subject_tree"] = True
            context["subject_tree"] = self.subject_tree
            context["classification_mode"] = "predefined"
        else:
            # Learning mode: query existing subject_trees from LanceDB (Priority 3)
            context["has_subject_tree"] = False
            existing_trees = self._get_existing_subject_trees()
            context["existing_subject_trees"] = existing_trees
            context["classification_mode"] = "learning"
            if existing_trees:
                logger.info(f"Found {len(existing_trees)} existing external knowledge subject_trees for context")

        logger.debug(f"Prepared template context: {context}")
        return context

    async def _parse_user_message(self, user_message: str) -> tuple[str, Optional[str]]:
        """
        Use lightweight LLM to find SQL boundaries in user_message.

        This is used in agentic mode when question/gold_sql fields are not provided directly.
        SQL may appear anywhere in the message (beginning, middle, or end).

        Args:
            user_message: Raw user input message

        Returns:
            tuple[str, Optional[str]]: (question, gold_sql)
            - If SQL found: question = text before + after SQL, gold_sql = extracted SQL
            - If no SQL: question = user_message, gold_sql = None
        """
        parse_prompt = """Identify the SQL statement in the following input and return its start and end substrings.

Input:
```
{user_message}
```

Output in JSON format:
```json
{{
  "sql_start_string": "<first 30-50 characters of the SQL statement, must be UNIQUE in the input, or null if no SQL>",
  "sql_end_string": "<last 30-50 characters of the SQL statement, must be UNIQUE in the input, or null if no SQL>"
}}
```

Rules:
- SQL typically starts with SELECT, WITH, INSERT, UPDATE, DELETE, CREATE, etc.
- SQL may be in code blocks (```sql), after labels like "SQL:", "Answer:", "Reference:", or standalone
- SQL may appear at the beginning, middle, or end of the input
- sql_start_string: first 30-50 characters of the SQL, enough to be UNIQUE in the input
- sql_end_string: last 30-50 characters of the SQL (including the final semicolon if present), enough to be UNIQUE
- If the same substring appears multiple times, extend it to make it unique
- Return null for both if no SQL found
- Do NOT include code block markers (```) in the returned strings"""

        try:
            # Use lightweight model for fast parsing with JSON output
            result = self.model.generate_with_json_output(
                prompt=parse_prompt.format(user_message=user_message),
            )

            if result:
                sql_start_string = result.get("sql_start_string")
                sql_end_string = result.get("sql_end_string")
                if sql_start_string:
                    # Find the start index
                    sql_start_index = user_message.find(sql_start_string)
                    if sql_start_index < 0:
                        logger.warning(f"SQL start string '{sql_start_string[:30]}...' not found in message")
                        return user_message, None

                    # Verify start uniqueness
                    if user_message.count(sql_start_string) > 1:
                        logger.warning(
                            f"SQL start string '{sql_start_string[:30]}...' " "is not unique, appears multiple times"
                        )
                        return user_message, None

                    # Find the end index
                    if sql_end_string:
                        sql_end_pos = user_message.find(sql_end_string)
                        if sql_end_pos >= sql_start_index:
                            if user_message.count(sql_end_string) > 1:
                                logger.warning(
                                    f"SQL end string '{sql_end_string[:30]}...' "
                                    "is not unique, appears multiple times"
                                )
                                return user_message, None
                            sql_end_index = sql_end_pos + len(sql_end_string)
                        else:
                            # End string not found after start, fall back to end of message
                            logger.warning("SQL end string not found after start, using end of message")
                            sql_end_index = len(user_message)
                    else:
                        # No end string provided, assume SQL goes to end
                        sql_end_index = len(user_message)

                    # Extract question (text before + after SQL) and gold_sql
                    text_before = user_message[:sql_start_index].strip()
                    text_after = user_message[sql_end_index:].strip()
                    gold_sql = user_message[sql_start_index:sql_end_index].strip()

                    if text_before and text_after:
                        question = f"{text_before}\n{text_after}"
                    else:
                        question = text_before or text_after

                    logger.info(
                        f"Parsed user message: sql_range=[{sql_start_index}:{sql_end_index}], "
                        f"question_len={len(question)}, sql_len={len(gold_sql)}"
                    )
                    return question, gold_sql
                else:
                    logger.info("No SQL found in user message (sql_start_string is null)")
                    return user_message, None
        except Exception as e:
            logger.warning(f"Failed to parse user message: {e}. Using original message.")

        # Parse failed, return original input
        return user_message, None

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """
        Get the system prompt for this external knowledge node using enhanced template context.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use (ignored, hardcoded to "1.0")
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """

        # Hardcoded system_prompt based on node name
        template_name = f"{self.configured_node_name}_system"

        try:
            # Prepare template variables
            template_vars = {
                "agent_config": self.agent_config,
                "conversation_summary": conversation_summary,
            }

            # Add template context if provided
            if template_context:
                template_vars.update(template_context)

            # Use prompt manager to render the template
            from datus.prompts.prompt_manager import prompt_manager

            return prompt_manager.render_template(template_name=template_name, version=prompt_version, **template_vars)

        except FileNotFoundError as e:
            # Template not found - throw DatusException
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": prompt_version or "latest"},
            ) from e
        except Exception as e:
            # Other template errors - wrap in DatusException
            logger.error(f"Template loading error for '{template_name}': {e}")
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the external knowledge node interaction with streaming support.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Get input from self.input
        if self.input is None:
            raise ValueError("External knowledge input not set. Set self.input before calling execute_stream.")
        user_input = self.input
        prompt_version = getattr(user_input, "prompt_version", None) or self.node_config.get("prompt_version")

        # Create initial action
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=self.get_node_name(),
            messages=f"User: {user_input.user_message}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        yield action

        try:
            # Session management (only in interactive mode)
            session = None
            conversation_summary = None
            if self.execution_mode == "interactive":
                # Check for auto-compact before session creation to ensure fresh context
                await self._auto_compact()

                # Get or create session and any available summary
                session, conversation_summary = self._get_or_create_session()

            # Prepare enhanced template context
            template_context = self._prepare_template_context(user_input)

            # Get system instruction from template with enhanced context
            system_instruction = self._get_system_prompt(conversation_summary, prompt_version, template_context)

            # Determine question and gold_sql based on mode
            # workflow mode: use fields directly; agentic mode: parse user_message
            if user_input.question is not None:
                # workflow mode: use provided question and gold_sql directly
                question = user_input.question
                gold_sql = user_input.gold_sql
                logger.info("Using directly provided question and gold_sql (workflow mode)")
            else:
                # agentic mode: parse user_message to extract question and gold_sql
                question, gold_sql = await self._parse_user_message(user_input.user_message)
                logger.info(f"Parsed from user_message (agentic mode): has_gold_sql={gold_sql is not None}")

            # Store gold_sql for get_gold_sql tool access (not exposed in prompt)
            self._current_question = question
            if gold_sql:
                self._gold_sql = gold_sql

            # Build enhanced message using question only (gold_sql accessed via tool)
            enhanced_message = question
            enhanced_parts = []

            # Add search_text context if provided
            if user_input.search_text:
                enhanced_parts.append(f"search_text: {user_input.search_text}")

            if user_input.explanation:
                enhanced_parts.append(f"Existing Explanation: {user_input.explanation}")

            if user_input.subject_path:
                enhanced_parts.append(f"Subject Path: {user_input.subject_path}")

            if enhanced_parts:
                enhanced_message = f"{'\n\n'.join(enhanced_parts)}\n\nUser question: {question}"

            logger.debug(f"Tools available: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")
            logger.info(f"Passing hooks to model: {self.hooks} (type: {type(self.hooks)})")

            # Initialize verification retry loop
            self._verification_attempt_count = 0
            current_prompt = enhanced_message

            # Initialize response collection variables (outside retry loop to preserve final values)
            response_content = ""
            ext_knowledge_file = None
            tokens_used = 0
            last_successful_output = None

            # Verification retry loop - continues if verification fails
            while self._verification_attempt_count <= self.max_verification_retries:
                # Reset verification state for this attempt
                self._reset_verification_state()

                # Stream response using the model's generate_with_tools_stream
                async for stream_action in self.model.generate_with_tools_stream(
                    prompt=current_prompt,
                    tools=self.tools,
                    mcp_servers=self.mcp_servers,
                    instruction=system_instruction,
                    max_turns=self.max_turns,
                    session=session,
                    action_history_manager=action_history_manager,
                    hooks=self.hooks if self.execution_mode == "interactive" else None,
                ):
                    yield stream_action

                    # Collect response content from successful actions
                    if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                        if isinstance(stream_action.output, dict):
                            last_successful_output = stream_action.output
                            # Look for content in various possible fields
                            raw_output = stream_action.output.get("raw_output", "")
                            # Handle case where raw_output is already a dict
                            if isinstance(raw_output, dict):
                                response_content = raw_output
                            elif raw_output:
                                response_content = raw_output

                # Agentic loop ended, check verification status
                logger.info(
                    f"Agentic loop ended. Verification passed: {self._verification_passed}, "
                    f"attempt: {self._verification_attempt_count + 1}/{self.max_verification_retries + 1}"
                )

                # Exit retry loop if verification passed or no gold_sql to verify against
                if self._verification_passed or not hasattr(self, "_gold_sql") or not self._gold_sql:
                    logger.info("Verification passed or no gold_sql available, exiting retry loop")
                    break

                # Verification failed, check if we have retries left
                self._verification_attempt_count += 1
                if self._verification_attempt_count > self.max_verification_retries:
                    logger.warning(
                        f"Max verification retries ({self.max_verification_retries}) exceeded, "
                        f"giving up on verification"
                    )
                    break

                # Inject retry prompt and continue agentic loop
                current_prompt = self._get_retry_prompt(self._verification_attempt_count)

                # Create retry notification action
                retry_action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="verification_retry",
                    messages=f"Verification failed, retrying "
                    f"({self._verification_attempt_count}/{self.max_verification_retries})...",
                    input_data={"retry_prompt": current_prompt},
                    status=ActionStatus.PROCESSING,
                )
                action_history_manager.add_action(retry_action)
                yield retry_action

                logger.info(f"Starting verification retry attempt {self._verification_attempt_count}")

            # If we still don't have response_content, check the last successful output
            if not response_content and last_successful_output:
                logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
                # Try different fields that might contain the response
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = raw_output
                elif raw_output:
                    response_content = raw_output
                else:
                    response_content = str(last_successful_output)  # Fallback to string representation

            # Extract ext_knowledge_file and output from the final response_content
            ext_knowledge_file, extracted_output = self._extract_ext_knowledge_and_output_from_response(
                {"content": response_content}
            )
            if extracted_output:
                response_content = extracted_output

            if not isinstance(response_content, str):
                response_content = str(response_content) if response_content else ""
            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")

            # Extract token usage (only in interactive mode with session)
            tokens_used = 0
            if self.execution_mode == "interactive":
                final_actions = action_history_manager.get_actions()

                # Find the final assistant action with token usage
                for action in reversed(final_actions):
                    if action.role == "assistant":
                        if action.output and isinstance(action.output, dict):
                            usage_info = action.output.get("usage", {})
                            if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                                conversation_tokens = usage_info.get("total_tokens", 0)
                                if conversation_tokens > 0:
                                    # Add this conversation's tokens to the session
                                    self._add_session_tokens(conversation_tokens)
                                    tokens_used = conversation_tokens
                                    logger.info(f"Added {conversation_tokens} tokens to session")
                                    break
                            else:
                                logger.warning(f"no usage token found in this action {action.messages}")

            # Auto-save to database in workflow mode
            if self.execution_mode == "workflow" and ext_knowledge_file:
                try:
                    self._save_to_db(ext_knowledge_file)
                    logger.info(f"Auto-saved to database: {ext_knowledge_file}")
                except Exception as e:
                    logger.error(f"Failed to auto-save to database: {e}")

            # Create final result
            result = ExtKnowledgeNodeResult(
                success=True,
                response=response_content,
                ext_knowledge_file=ext_knowledge_file,
                tokens_used=int(tokens_used),
            )

            # Add to internal actions list
            self.actions.extend(action_history_manager.get_actions())

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="ext_knowledge_response",
                messages=f"{self.get_node_name()} interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except Exception as e:
            logger.error(f"{self.get_node_name()} execution error: {e}")

            # Create error result
            error_result = ExtKnowledgeNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while processing your request.",
                tokens_used=0,
            )

            # Update action with error
            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {str(e)}",
            )

            # Create error action
            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="error",
                messages=f"{self.get_node_name()} interaction failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

    def _extract_ext_knowledge_and_output_from_response(self, output: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extract ext_knowledge_file and formatted output from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"ext_knowledge_file": "path", "output": "markdown text"}

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (ext_knowledge_file, output_string) - both can be None if not found
        """
        try:
            from datus.utils.json_utils import strip_json_str

            content = output.get("content", "")

            # Case 1: content is already a dict (most common)
            if isinstance(content, dict):
                ext_knowledge_file = content.get("ext_knowledge_file")
                output_text = content.get("output")
                if ext_knowledge_file or output_text:
                    logger.debug(f"Extracted from dict: ext_knowledge_file={ext_knowledge_file}")
                    return ext_knowledge_file, output_text
                else:
                    logger.warning(f"Dict format but missing expected keys: {content.keys()}")

            # Case 2: content is a JSON string (possibly wrapped in markdown code blocks)
            elif isinstance(content, str) and content.strip():
                # Use strip_json_str to handle markdown code blocks and extract JSON
                cleaned_json = strip_json_str(content)
                if cleaned_json:
                    try:
                        import json_repair

                        parsed = json_repair.loads(cleaned_json)
                        if isinstance(parsed, dict):
                            ext_knowledge_file = parsed.get("ext_knowledge_file")
                            output_text = parsed.get("output")
                            if ext_knowledge_file or output_text:
                                logger.debug(f"Extracted from JSON string: ext_knowledge_file={ext_knowledge_file}")
                                return ext_knowledge_file, output_text
                            else:
                                logger.warning(f"Parsed JSON but missing expected keys: {parsed.keys()}")
                    except Exception as e:
                        logger.warning(f"Failed to parse cleaned JSON: {e}. Cleaned content: {cleaned_json[:200]}")

            logger.warning(f"Could not extract ext_knowledge_file from response. Content type: {type(content)}")
            return None, None

        except Exception as e:
            logger.error(f"Unexpected error extracting ext_knowledge_file: {e}", exc_info=True)
            return None, None

    def _save_to_db(self, ext_knowledge_file: str):
        """
        Save generated external knowledge to database (synchronous).

        Args:
            ext_knowledge_file: Name of the external knowledge file (e.g., "gmv_001.yaml")
        """
        try:
            import os

            # Construct full path
            full_path = os.path.join(self.ext_knowledge_dir, ext_knowledge_file)

            if not os.path.exists(full_path):
                logger.warning(f"External knowledge file not found: {full_path}")
                return

            # Call static method to save to database with build_mode
            result = GenerationHooks._sync_ext_knowledge_to_db(full_path, self.agent_config, self.build_mode)

            if result.get("success"):
                logger.info(f"Successfully saved to database: {result.get('message')}")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to save to database: {error}")

        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            raise
