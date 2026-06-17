# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SqlSummaryAgenticNode implementation for SQL summary generation workflow.

This module provides a specialized implementation of AgenticNode focused on
SQL query summarization and classification with support for filesystem tools,
generation tools, and hooks.
"""

import re
from typing import List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput, SqlSummaryNodeResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SqlSummaryAgenticNode(AgenticNode):
    """
    SQL summary generation agentic node with enhanced configuration.

    This node provides specialized SQL query summarization and classification with:
    - Enhanced system prompt with template variables
    - Filesystem tools for file operations
    - Generation tools for SQL summary context preparation
    - Hooks support for custom behavior
    - Configurable tool sets
    - Session-based conversation management
    """

    result_class = SqlSummaryNodeResult

    def __init__(
        self,
        node_name: str,
        agent_config: Optional[AgentConfig] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        build_mode: str = "incremental",
        subject_tree: Optional[list] = None,
        storage_type: str = "reference_sql",
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the SqlSummaryAgenticNode.

        Args:
            node_name: Name of the node configuration in agent.yml (should be "gen_sql_summary")
            agent_config: Agent configuration
            execution_mode: Execution mode - "interactive" (default) or "workflow"
            build_mode: "overwrite" or "incremental" (default: "incremental")
            subject_tree: Optional predefined subject tree categories
            storage_type: Storage target - "reference_sql" (default) or "reference_template"
        """
        self.configured_node_name = node_name
        self.execution_mode = execution_mode
        self.build_mode = build_mode
        self.subject_tree = subject_tree
        self.storage_type = storage_type

        # Get max_turns from agentic_nodes configuration
        self.max_turns = 50
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 50)

        self.sql_summary_dir = str(agent_config.path_manager.sql_summary_path())
        self.knowledge_base_dir = str(agent_config.path_manager.subject_dir)

        from datus.configuration.node_type import NodeType

        node_type = NodeType.TYPE_SQL_SUMMARY

        # Call parent constructor first to set up node_config
        super().__init__(
            node_id="sql_summary_node",
            description="SQL summary generation node",
            node_type=node_type,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Initialize reference SQL storage for context queries
        from datus.storage.reference_sql.store import ReferenceSqlRAG

        self.reference_sql_rag = ReferenceSqlRAG(agent_config)

        # Setup tools based on hardcoded configuration
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.ask_user_tool = None
        self.hooks = None
        self.setup_tools()

        logger.debug(f"Hooks after setup: {self.hooks} (type: {type(self.hooks)})")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this SQL summary agentic node.

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
        # tools: generation_tools.generate_sql_summary_id,
        # filesystem_tools: read_file, write_file, edit_file, glob, grep
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
        """Setup specific generation tools: generate_sql_summary_id."""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.generation_tools = GenerationTools(self.agent_config)
            self.tools.append(trans_to_function_tool(self.generation_tools.generate_sql_summary_id))
        except Exception as e:
            logger.error(f"Failed to setup specific generation tools: {e}")

    def _setup_specific_filesystem_tool(self):
        """Setup specific filesystem tools"""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()

            self.tools.extend(self.filesystem_func_tool.available_tools())
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
        Query existing subject_path values from reference SQL storage.

        Returns:
            List of unique subject_path values as List[str]
        """
        try:
            # Get all metrics with subject_path field
            subject_paths = sorted(self.reference_sql_rag.reference_sql_storage.get_subject_tree_flat())
            logger.debug(f"Found {len(subject_paths)} unique reference SQL subject_paths")
            return subject_paths
        except Exception as e:
            logger.error(f"Error getting existing subject_paths: {e}")
            return []

    def _get_similar_sqls(self, query_text: str, top_n: int = 5) -> list:
        """
        Find similar reference SQLs based on query text.

        Args:
            query_text: Text to use for similarity search (comment or SQL)
            top_n: Number of similar results to return

        Returns:
            List of similar reference SQLs with fields: name, subject_tree, tags, comment, summary
        """
        try:
            if not query_text:
                return []

            # Search using vector similarity on summary field
            similar_items = self.reference_sql_rag.search_reference_sql(query_text=query_text, top_n=top_n)

            # Extract relevant fields and format results
            results = []
            for item in similar_items:
                # Get subject_path from item
                subject_path = item.get("subject_path", [])
                # Format as string for display
                subject_tree = "/".join(subject_path) if subject_path else ""

                results.append(
                    {
                        "name": item.get("name", ""),
                        "subject_tree": subject_tree,
                        "tags": item.get("tags", ""),
                        "comment": item.get("comment", ""),
                        "summary": item.get("summary", ""),
                    }
                )

            logger.debug(f"Found {len(results)} similar reference SQLs")
            return results

        except Exception as e:
            logger.error(f"Error getting similar reference SQLs: {e}")
            return []

    def _prepare_template_context(self, user_input: SqlSummaryNodeInput) -> dict:
        """
        Prepare template context variables for the SQL summary generation template.

        Args:
            user_input: User input

        Returns:
            Dictionary of template variables
        """
        from datus.utils.node_utils import build_datasource_prompt_context

        context = {}

        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["sql_summary_dir"] = self.sql_summary_dir
        context["knowledge_base_dir"] = self.knowledge_base_dir
        # Filesystem tool is rooted at project_root; full path required.
        context["kind_subdir"] = "subject/sql_summaries"
        context["current_datasource"] = self.agent_config.current_datasource
        context["has_ask_user_tool"] = self.ask_user_tool is not None
        context.update(build_datasource_prompt_context(self.agent_config))

        # Handle subject_tree context based on whether predefined or query from storage
        if self.subject_tree:
            # Predefined mode: use provided subject_tree
            context["has_subject_tree"] = True
            context["subject_tree"] = self.subject_tree
        else:
            # Learning mode: query existing subject_trees from vector store
            context["has_subject_tree"] = False
            existing_trees = self._get_existing_subject_trees()
            context["existing_subject_trees"] = existing_trees
            if existing_trees:
                logger.info(f"Found {len(existing_trees)} existing reference SQL subject_trees for context")

        # Query similar reference SQLs for classification reference
        # Use first 200 chars of SQL as query text
        query_text = user_input.sql_query[:200] if user_input.sql_query else ""

        similar_items = self._get_similar_sqls(query_text, top_n=5)
        context["similar_items"] = similar_items
        if similar_items:
            logger.info(f"Found {len(similar_items)} similar reference SQLs for context")

        logger.debug(f"Prepared template context: {context}")
        return context

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """
        Get the system prompt for this SQL summary node using enhanced template context.

        Args:
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
            }

            # Add template context if provided
            if template_context:
                template_vars.update(template_context)

            # Use prompt manager to render the template
            from datus.prompts.prompt_manager import get_prompt_manager

            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, **template_vars
            )
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError as e:
            # Template not found - throw DatusException
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": prompt_version},
            ) from e
        except Exception as e:
            # Other template errors - wrap in DatusException
            logger.error(f"Template loading error for '{template_name}': {e}")
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    def _build_template_context(self, ctx: StreamRunContext) -> Optional[dict]:
        """Delegate to the existing ``_prepare_template_context`` helper."""
        return self._prepare_template_context(ctx.user_input)

    def _build_enhanced_message(self, user_input, extra_enhanced_parts=None) -> str:
        """Splice sql_query / comment into the enhanced section."""
        extra_parts: List[str] = list(extra_enhanced_parts or [])
        sql_query = getattr(user_input, "sql_query", "")
        comment = getattr(user_input, "comment", "")
        if sql_query:
            extra_parts.append(f"SQL Query:\n```sql\n{sql_query}\n```")
        if comment:
            extra_parts.append(f"Comment: {comment}")
        return super()._build_enhanced_message(user_input, extra_enhanced_parts=extra_parts)

    def _build_success_result(self, ctx: StreamRunContext) -> SqlSummaryNodeResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            if isinstance(raw_output, dict) or raw_output:
                response_content = raw_output
            else:
                response_content = str(ctx.last_successful_output)

        sql_summary_file, extracted_output = self._extract_sql_summary_and_output_from_response(
            {"content": response_content}
        )
        if extracted_output:
            response_content = extracted_output

        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())

        # Workflow mode auto-saves discovered sql_summary files to the DB.
        # Persistence is the only durability path in workflow mode, so let
        # the exception propagate to the template's error handler instead
        # of silently returning success when the DB write failed.
        if self.execution_mode == "workflow" and sql_summary_file:
            self._save_to_db(sql_summary_file)
            logger.info(f"Auto-saved to database: {sql_summary_file}")

        return SqlSummaryNodeResult(
            success=True,
            response=response_content,
            sql_summary_file=sql_summary_file,
            tokens_used=int(tokens_used),
        )

    def _extract_sql_summary_and_output_from_response(self, output: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extract sql_summary_file and formatted output from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"sql_summary_file": "path", "output": "markdown text"}

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (sql_summary_file, output_string) - both can be None if not found
        """
        try:
            from datus.utils.json_utils import strip_json_str

            content = output.get("content", "")
            logger.info(f"extract_sql_summary_and_output_from_final_resp: {content} (type: {type(content)})")

            def _extract_from_parsed(parsed: object) -> tuple[Optional[str], Optional[str]]:
                if isinstance(parsed, dict):
                    sql_summary_file = parsed.get("sql_summary_file")
                    output_text = parsed.get("output")
                    if sql_summary_file or output_text:
                        return sql_summary_file, output_text
                    logger.warning(f"Parsed JSON but missing expected keys: {parsed.keys()}")
                return None, None

            def _parse_json_payload(payload: str) -> tuple[Optional[str], Optional[str]]:
                if not payload:
                    return None, None
                try:
                    import json_repair

                    parsed = json_repair.loads(payload)
                    return _extract_from_parsed(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse JSON payload: {e}. Content: {payload[:200]}")
                    return None, None

            def _looks_relevant_json(payload: str) -> bool:
                return "sql_summary_file" in payload or '"output"' in payload or "'output'" in payload

            def _iter_json_object_candidates(text: str):
                for start in [i for i, char in enumerate(text) if char == "{"]:
                    in_string = False
                    escape = False
                    quote_char = ""
                    depth = 0
                    for pos in range(start, len(text)):
                        char = text[pos]
                        if in_string:
                            if escape:
                                escape = False
                            elif char == "\\":
                                escape = True
                            elif char == quote_char:
                                in_string = False
                            continue
                        if char in ('"', "'"):
                            in_string = True
                            quote_char = char
                        elif char == "{":
                            depth += 1
                        elif char == "}":
                            depth -= 1
                            if depth == 0:
                                candidate = text[start : pos + 1]
                                if _looks_relevant_json(candidate):
                                    yield candidate
                                break

            def _iter_json_payload_candidates(text: str):
                # Prefer fenced blocks when present, but accept any fence label.
                for match in re.finditer(r"```[^\n`]*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
                    block = match.group(1).strip()
                    if _looks_relevant_json(block):
                        yield block

                stripped = text.strip()
                if stripped.startswith("{"):
                    yield stripped

                yield from _iter_json_object_candidates(text)

                cleaned_json = strip_json_str(text)
                if cleaned_json and cleaned_json != text:
                    yield cleaned_json

            # Case 1: content is already a dict (most common)
            if isinstance(content, dict):
                sql_summary_file, output_text = _extract_from_parsed(content)
                if sql_summary_file or output_text:
                    logger.debug(f"Extracted from dict: sql_summary_file={sql_summary_file}")
                    return sql_summary_file, output_text

            # Case 2: content is a JSON string (possibly wrapped in markdown code blocks)
            elif isinstance(content, str) and content.strip():
                # Free-form explanations can contain Jinja/SQL braces before the
                # final JSON. Try all plausible JSON payloads instead of assuming
                # the first "{" starts the response object.
                for candidate in _iter_json_payload_candidates(content):
                    sql_summary_file, output_text = _parse_json_payload(candidate)
                    if sql_summary_file or output_text:
                        logger.debug(f"Extracted from response JSON candidate: sql_summary_file={sql_summary_file}")
                        return sql_summary_file, output_text

                match = re.search(r'"sql_summary_file"\s*:\s*"([^"]+)"', content)
                if match:
                    logger.debug(f"Extracted sql_summary_file via regex: {match.group(1)}")
                    return match.group(1), None

            logger.warning(f"Could not extract sql_summary_file from response. Content type: {type(content)}")
            return None, None

        except Exception as e:
            logger.error(f"Unexpected error extracting sql_summary_file: {e}", exc_info=True)
            return None, None

    def _save_to_db(self, sql_summary_file: str):
        """
        Save generated SQL summary to database (synchronous).

        Args:
            sql_summary_file: Path of the SQL summary file as reported by the LLM.
                Absolute, KB-root-relative (e.g. ``sql_summaries/<db>/q_001.yaml``)
                and bare-filename forms are all accepted — the same normalizer
                used on the write side resolves them to the actual on-disk path.
        """
        try:
            import os

            from datus.cli.generation_hooks import resolve_kb_sandbox_path

            full_path = resolve_kb_sandbox_path(sql_summary_file, "sql_summary", self.knowledge_base_dir)
            if not full_path:
                logger.warning(f"SQL summary file rejected by sandbox check: {sql_summary_file!r}")
                return

            if not os.path.exists(full_path):
                logger.warning(f"SQL summary file not found: {full_path}")
                return

            # Call static method to save to database with build_mode
            if self.storage_type == "reference_template":
                result = GenerationHooks._sync_reference_template_to_db(full_path, self.agent_config, self.build_mode)
            else:
                result = GenerationHooks._sync_reference_sql_to_db(full_path, self.agent_config, self.build_mode)

            if result.get("success"):
                logger.info(f"Successfully saved to database: {result.get('message')}")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to save to database: {error}")

        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            raise
