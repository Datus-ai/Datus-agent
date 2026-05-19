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
from typing import Any, Dict, Iterable, List, Literal, Optional

import pandas as pd

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.compare_agentic_node import CompareAgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput, ExtKnowledgeNodeResult
from datus.schemas.node_models import SQLContext, SqlTask
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.utils.benchmark_utils import ComparisonOutcome, TableComparator
from datus.utils.loggings import get_logger

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


class VerifySqlRetryPolicy:
    """:class:`~datus.agent.node.retry_policy.RetryPolicy` driven by ``verify_sql``.

    Used exclusively by :class:`GenExtKnowledgeAgenticNode`. The node owns
    the ``_verification_passed`` flag and the ``_get_retry_prompt`` helper;
    this policy is a thin adapter that lets the template's generic retry
    loop consume them. When gold_sql is absent (``node._gold_sql`` falsy)
    verification is considered passed and the loop exits after the first
    attempt.

    Lives in this module — not in a shared ``policies/`` package — because
    it closes over a GenExtKnowledgeAgenticNode instance and would not be
    reused by any other node.
    """

    def __init__(self, node: "GenExtKnowledgeAgenticNode"):
        self.node = node
        # ``max_verification_retries`` is the number of *retries*; the
        # total attempt count is one larger (initial + retries).
        self.max_attempts = max(1, node.max_verification_retries + 1)

    def reset(self, ctx: StreamRunContext) -> None:
        # Verification state is owned by the node so the ``verify_sql`` tool's
        # ``on_end`` hook can keep updating it during the stream.
        self.node._reset_verification_state()

    def should_retry(self, ctx: StreamRunContext) -> bool:
        if self.node._verification_passed:
            return False
        # No gold_sql means there is nothing to verify against — treat as passed.
        if not getattr(self.node, "_gold_sql", None):
            return False
        logger.info(
            "Verification failed for %s (attempt %d/%d), scheduling retry",
            self.node.get_node_name(),
            ctx.attempt,
            self.max_attempts,
        )
        return True

    def next_prompt(self, ctx: StreamRunContext) -> Optional[str]:
        # ``ctx.attempt`` is the iteration we just finished; the next attempt
        # uses it as the retry index (1-based for the user-facing
        # "(N/max)" suffix the node's prompt builder embeds).
        return self.node._get_retry_prompt(ctx.attempt)

    def on_retry_actions(self, ctx: StreamRunContext) -> Iterable[ActionHistory]:
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="verification_retry",
            messages=(f"Verification failed, retrying ({ctx.attempt}/{self.node.max_verification_retries})..."),
            input_data={"attempt": ctx.attempt},
            status=ActionStatus.PROCESSING,
        )
        return (action,)

    def finalise(self, ctx: StreamRunContext) -> None:
        return None


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

    result_class = ExtKnowledgeNodeResult

    def __init__(
        self,
        node_name: str,
        agent_config: Optional[AgentConfig] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        build_mode: str = "incremental",
        subject_tree: Optional[list] = None,
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
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

        self.ext_knowledge_dir = str(agent_config.path_manager.ext_knowledge_path())
        self.knowledge_base_dir = str(agent_config.path_manager.subject_dir)

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
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Initialize external knowledge storage for context queries
        from datus.storage.ext_knowledge.store import ExtKnowledgeRAG

        self.ext_knowledge_store = ExtKnowledgeRAG(agent_config)

        # Setup tools based on hardcoded configuration
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.ask_user_tool = None
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
        # filesystem_tools: read_file, write_file, edit_file
        # Chat node uses all available tools by default
        node_name = self.get_node_name()
        self.db_func_tool = DBFuncTool(agent_config=self.agent_config, sub_agent_name=node_name)
        self.context_search_tools = ContextSearchTools(self.agent_config, sub_agent_name=node_name)
        if self.db_func_tool:
            self.tools.extend(self.db_func_tool.available_tools())
        if self.context_search_tools:
            self.tools.extend(self.context_search_tools.available_tools())
        self._setup_specific_generation_tools()
        self._setup_specific_filesystem_tool()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.info(
            f"Setup {len(self.tools)} tools for {self.configured_node_name}: {[tool.name for tool in self.tools]}"
        )

        # Setup hooks (only in interactive mode)
        if self.execution_mode == "interactive":
            self._setup_hooks()

    def _setup_specific_generation_tools(self):
        """Setup specific generation tools.

        Note: ``verify_sql`` is intentionally NOT registered here. It is bound
        lazily in :meth:`execute_stream` only when a non-empty gold_sql is
        supplied and passes pre-validation. See
        :meth:`_enable_verify_sql_tool` / :meth:`_disable_verify_sql_tool`.
        """
        try:
            self.generation_tools = GenerationTools(self.agent_config)
        except Exception as e:
            logger.error(f"Failed to setup specific generation tools: {e}")

    def _enable_verify_sql_tool(self) -> None:
        """Register ``verify_sql`` on self.tools if not already present (idempotent)."""
        from datus.tools.func_tool import trans_to_function_tool

        if not any(getattr(t, "name", None) == "verify_sql" for t in self.tools):
            self.tools.append(trans_to_function_tool(self.verify_sql))

    def _disable_verify_sql_tool(self) -> None:
        """Remove ``verify_sql`` from self.tools if present (idempotent)."""
        self.tools = [t for t in self.tools if getattr(t, "name", None) != "verify_sql"]

    def _validate_gold_sql(self, gold_sql: str) -> None:
        """Execute gold SQL once to ensure it is runnable before entering the agent loop.

        Raises:
            DatusException: with code NODE_EXT_KNOWLEDGE_GOLD_SQL_INVALID when
                the gold SQL fails to execute. Callers let the exception
                propagate to :meth:`execute_stream`'s top-level handler, which
                turns it into a FAILED action.
        """
        from datus.utils.exceptions import DatusException, ErrorCode

        try:
            result = self.db_func_tool.connector.execute_query(gold_sql, result_format="pandas")
        except Exception as e:
            raise DatusException(
                code=ErrorCode.NODE_EXT_KNOWLEDGE_GOLD_SQL_INVALID,
                message_args={"error_message": str(e)},
            ) from e
        if not result.success:
            raise DatusException(
                code=ErrorCode.NODE_EXT_KNOWLEDGE_GOLD_SQL_INVALID,
                message_args={"error_message": result.error or "unknown error"},
            )

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

        connector = self.db_func_tool.connector

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
                    "your_result_shape": (
                        f"{result.user_df.shape[0]} rows x {result.user_df.shape[1]} columns"
                        if result.user_df is not None
                        else "N/A"
                    ),
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
                database_name=(
                    self.agent_config.current_datasource if hasattr(self.agent_config, "current_datasource") else ""
                ),
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
            _, _, messages = CompareAgenticNode._prepare_prompt_components(
                compare_input, agent_config=self.agent_config
            )
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

            self.filesystem_func_tool = self._make_filesystem_tool()
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

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Route tools to permission categories so profile rules apply.

        ``verify_sql`` is lazily bound just before generation. Route it
        into ``db_tools`` when present so profile rules for ``db_tools.*``
        govern the SQL it executes; otherwise a DENY on ``db_tools.*``
        would silently not cover the one tool that runs model-supplied SQL.
        """
        mapping = super()._tool_category_map()
        if self.db_func_tool:
            mapping["db_tools"] = list(self.db_func_tool.available_tools())
        verify_sql_tools = [tool for tool in self.tools if getattr(tool, "name", None) == "verify_sql"]
        if verify_sql_tools:
            mapping.setdefault("db_tools", []).extend(verify_sql_tools)
        if getattr(self, "context_search_tools", None):
            mapping["context_search_tools"] = list(self.context_search_tools.available_tools())
        if self.filesystem_func_tool:
            mapping["filesystem_tools"] = list(self.filesystem_func_tool.available_tools())
        if self.ask_user_tool:
            mapping.setdefault("tools", []).extend(self.ask_user_tool.available_tools())
        return mapping

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

    def _prepare_template_context(self, user_input: ExtKnowledgeNodeInput, gold_sql: Optional[str] = None) -> dict:
        """
        Prepare template context variables for the external knowledge generation template.

        Args:
            user_input: User input
            gold_sql: Reference SQL resolved for this run. When non-empty the
                template renders the verify_sql tool and the BLOCKING
                PHASE 2; when empty/None the template skips PHASE 2 entirely.

        Returns:
            Dictionary of template variables
        """
        from datus.utils.node_utils import build_datasource_prompt_context

        context = {}

        tool_names = [tool.name for tool in self.tools] if self.tools else []
        context["native_tools"] = ", ".join(tool_names) if tool_names else "None"
        context["tool_names"] = tool_names
        context["has_search_knowledge_tool"] = "search_knowledge" in tool_names
        context["has_get_knowledge_tool"] = "get_knowledge" in tool_names
        context["ext_knowledge_dir"] = self.ext_knowledge_dir
        context["knowledge_base_dir"] = self.knowledge_base_dir
        # Filesystem tool is rooted at project_root; full path required.
        context["kind_subdir"] = "subject/ext_knowledge"
        context["current_datasource"] = self.agent_config.current_datasource
        context.update(build_datasource_prompt_context(self.agent_config))
        context["has_filesystem_tools"] = bool(self.filesystem_func_tool)
        context["has_ask_user_tool"] = self.ask_user_tool is not None
        context["has_gold_sql"] = bool(gold_sql)

        # ``subject_path`` was removed from ExtKnowledgeNodeInput; no caller
        # populated it in production. Always fall back to subject-tree.
        context["has_user_specified_subject"] = False

        # Priority 2 & 3: Handle subject_tree context based on whether predefined or query from storage
        if self.subject_tree:
            # Predefined mode: use provided subject_tree (Priority 2)
            context["has_subject_tree"] = True
            context["subject_tree"] = self.subject_tree
            context["classification_mode"] = "predefined"
        else:
            # Learning mode: query existing subject_trees from vector store (Priority 3)
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
                            f"SQL start string '{sql_start_string[:30]}...' is not unique, appears multiple times"
                        )
                        return user_message, None

                    # Find the end index
                    if sql_end_string:
                        sql_end_pos = user_message.find(sql_end_string)
                        if sql_end_pos >= sql_start_index:
                            if user_message.count(sql_end_string) > 1:
                                logger.warning(
                                    f"SQL end string '{sql_end_string[:30]}...' is not unique, appears multiple times"
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
            from datus.prompts.prompt_manager import get_prompt_manager

            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, version=prompt_version, **template_vars
            )
            return self._finalize_system_prompt(base_prompt)

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

    # ── template hooks ────────────────────────────────────────────────

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        # Parse question / gold_sql out of either workflow fields or the raw
        # ``user_message`` before tools and prompts are wired up: the presence
        # of gold_sql decides whether ``verify_sql`` is registered, and the
        # template context branches on it too.
        user_input = ctx.user_input
        if user_input.question is not None:
            question = user_input.question
            gold_sql = user_input.gold_sql
            logger.info("Using directly provided question and gold_sql (workflow mode)")
        else:
            question, gold_sql = await self._parse_user_message(user_input.user_message)
            logger.info(f"Parsed from user_message (agentic mode): has_gold_sql={gold_sql is not None}")

        self._current_question = question

        if gold_sql:
            self._validate_gold_sql(gold_sql)
            self._gold_sql = gold_sql
            self._enable_verify_sql_tool()
        else:
            self._disable_verify_sql_tool()
            if hasattr(self, "_gold_sql"):
                delattr(self, "_gold_sql")

        # The parsed ``question`` replaces ``user_input.user_message`` when
        # the template calls ``_build_enhanced_message``.
        ctx.user_message_override = question
        ctx.extras["ext_knowledge_gold_sql"] = gold_sql

    def _build_template_context(self, ctx: StreamRunContext) -> Optional[dict]:
        gold_sql = ctx.extras.get("ext_knowledge_gold_sql")
        return self._prepare_template_context(ctx.user_input, gold_sql=gold_sql)

    def _get_retry_policy(self):
        return VerifySqlRetryPolicy(node=self)

    def _build_success_result(self, ctx: StreamRunContext) -> ExtKnowledgeNodeResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            if isinstance(raw_output, dict) or raw_output:
                response_content = raw_output
            else:
                response_content = str(ctx.last_successful_output)

        ext_knowledge_file, extracted_output = self._extract_ext_knowledge_and_output_from_response(
            {"content": response_content}
        )
        if extracted_output:
            response_content = extracted_output
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())

        # Workflow-mode auto-save side effect preserved from the legacy
        # execute_stream — fires only when an ext_knowledge file was produced.
        if self.execution_mode == "workflow" and ext_knowledge_file:
            try:
                self._save_to_db(ext_knowledge_file)
                logger.info(f"Auto-saved to database: {ext_knowledge_file}")
            except Exception as e:
                logger.error(f"Failed to auto-save to database: {e}")

        return ExtKnowledgeNodeResult(
            success=True,
            response=response_content,
            ext_knowledge_file=ext_knowledge_file,
            tokens_used=int(tokens_used),
        )

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
            ext_knowledge_file: Path of the ext-knowledge file as reported by the LLM.
                Absolute, KB-root-relative (e.g. ``ext_knowledge/<db>/gmv.yaml``)
                and bare-filename forms are all accepted — the same normalizer
                used on the write side resolves them to the actual on-disk path.
        """
        try:
            import os

            from datus.cli.generation_hooks import resolve_kb_sandbox_path

            full_path = resolve_kb_sandbox_path(ext_knowledge_file, "ext_knowledge", self.knowledge_base_dir)
            if not full_path:
                logger.warning(f"External knowledge file rejected by sandbox check: {ext_knowledge_file!r}")
                return

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
