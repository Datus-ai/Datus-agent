# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenReportAgenticNode implementation for generic report generation.

This module provides a base implementation of AgenticNode focused on
report generation with semantic and database tools. It can be used directly
or extended by specialized report nodes like AttributionAgenticNode.
"""

from typing import Any, Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput, GenReportNodeResult
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, FilesystemFuncTool
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenReportAgenticNode(AgenticNode):
    """
    Generic report generation agentic node.

    This node provides a flexible base for report generation with:
    - Configuration-based tool setup (semantic_tools.*, db_tools.*)
    - Common streaming execution logic
    - Template context building
    - Result extraction framework

    Can be instantiated directly or extended by specialized nodes.
    """

    NODE_NAME = "gen_report"
    result_class = GenReportNodeResult

    # Default tools when not configured in agent.yml
    DEFAULT_TOOLS = "semantic_tools.*, context_search_tools.list_subject_tree"

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[GenReportNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the GenReportAgenticNode.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node
            input_data: Report generation input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
            node_name: Name of the node configuration in agent.yml
            execution_mode: Execution mode - "interactive" (default) or "workflow"
            is_subagent: When True, skip SubAgentTaskTool setup (2-level depth enforcement)
        """
        self.execution_mode = execution_mode
        # Determine node name from node_type if not provided
        self.configured_node_name = node_name

        self.max_turns = 50
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 50)

        # Initialize tool attributes BEFORE calling parent constructor
        # This is required because parent's __init__ calls _get_system_prompt()
        # which may reference these attributes
        self.db_func_tool: Optional[DBFuncTool] = None
        self.semantic_tools: Optional[SemanticTools] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None

        # Call parent constructor with all required Node parameters
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},  # No MCP servers for report nodes by default
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Setup tools based on configuration (includes subagent task tool wiring)
        self.setup_tools()

        # Setup ask_user tool for clarification questions (interactive mode only)
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(f"GenReportAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this report agentic node.

        Returns:
            The configured node name from agent.yml or NODE_NAME default
        """
        return self.configured_node_name or self.NODE_NAME

    def setup_tools(self):
        """
        Setup tools based on configuration.

        Reads 'tools' from node_config and sets up each tool pattern.
        If no tools configured, self.tools remains empty.
        """
        if not self.agent_config:
            return

        self.tools = []

        # Setup tools from configuration, falling back to DEFAULT_TOOLS
        config_value = self.node_config.get("tools")
        if config_value is None:
            config_value = self.DEFAULT_TOOLS

        tool_patterns = [p.strip() for p in config_value.split(",") if p.strip()]
        for pattern in tool_patterns:
            self._setup_tool_pattern(pattern)

        # Ensure filesystem tools are always available (required for memory and file operations)
        if not self.filesystem_func_tool:
            self._setup_filesystem_tools()

        # Rebuild subagent task tool so repeated setup_tools() calls (e.g. via
        # ChatCommands.update_chat_node_tools after a datasource switch) keep the
        # "task" tool available for delegation.
        self._setup_sub_agent_task_tool()
        if self.sub_agent_task_tool:
            self.tools.extend(self.sub_agent_task_tool.available_tools())

        logger.info(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

    def _setup_tool_pattern(self, pattern: str):
        """
        Setup tools based on pattern.

        Supports patterns like:
        - "semantic_tools.*" -> all semantic tools
        - "db_tools.*" -> all db tools
        - "context_search_tools.*" -> all context search tools
        - "semantic_tools.search_metrics" -> specific method
        - "db_tools.list_tables" -> specific method
        - "context_search_tools.list_subject_tree" -> specific method
        """
        try:
            # Handle wildcard patterns (e.g., "semantic_tools.*")
            if pattern.endswith(".*"):
                base_type = pattern[:-2]
                if base_type == "semantic_tools":
                    self._setup_semantic_tools()
                elif base_type == "db_tools":
                    self._setup_db_tools()
                elif base_type == "context_search_tools":
                    self._setup_context_search_tools()
                elif base_type == "filesystem_tools":
                    self._setup_filesystem_tools()
                else:
                    logger.warning(f"Unknown tool type: {base_type}")

            # Handle exact type patterns
            elif pattern == "semantic_tools":
                self._setup_semantic_tools()
            elif pattern == "db_tools":
                self._setup_db_tools()
            elif pattern == "context_search_tools":
                self._setup_context_search_tools()
            elif pattern == "filesystem_tools":
                self._setup_filesystem_tools()

            # Handle specific method patterns (e.g., "db_tools.describe_table")
            elif "." in pattern:
                tool_type, method_name = pattern.split(".", 1)
                self._setup_specific_tool_method(tool_type, method_name)

            else:
                logger.warning(f"Unknown tool pattern: {pattern}")

        except Exception as e:
            logger.error(f"Failed to setup tool pattern '{pattern}': {e}")

    def _setup_db_tools(self):
        """Setup database tools."""
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
            )
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug("Added database tools from DBFuncTool")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_semantic_tools(self):
        """Setup semantic tools for report analysis."""
        try:
            from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type

            adapter_type = resolve_semantic_adapter_type(self.agent_config)
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
                adapter_type=adapter_type,
                runtime_db_context_provider=self._semantic_runtime_db_context,
            )
            self.tools.extend(self.semantic_tools.available_tools())
            logger.debug("Added semantic tools from SemanticTools")
        except Exception as e:
            logger.error(f"Failed to setup semantic tools: {e}")

    def _setup_context_search_tools(self):
        """Setup context search tools."""
        try:
            self.context_search_tools = ContextSearchTools(
                self.agent_config, sub_agent_name=self.node_config.get("system_prompt")
            )
            self.tools.extend(self.context_search_tools.available_tools())
            logger.debug("Added context search tools from ContextSearchTools")
        except Exception as e:
            logger.error(f"Failed to setup context search tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug(f"Setup filesystem tools with root path: {self.filesystem_func_tool.root_path}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_specific_tool_method(self, tool_type: str, method_name: str):
        """Setup a specific tool method."""
        try:
            if tool_type == "semantic_tools":
                if not self.semantic_tools:
                    from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type

                    adapter_type = resolve_semantic_adapter_type(self.agent_config)
                    self.semantic_tools = SemanticTools(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                        adapter_type=adapter_type,
                        runtime_db_context_provider=self._semantic_runtime_db_context,
                    )
                tool_instance = self.semantic_tools
            elif tool_type == "db_tools":
                if not self.db_func_tool:
                    self.db_func_tool = DBFuncTool(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                    )
                tool_instance = self.db_func_tool
            elif tool_type == "context_search_tools":
                if not self.context_search_tools:
                    self.context_search_tools = ContextSearchTools(
                        self.agent_config, sub_agent_name=self.node_config.get("system_prompt")
                    )
                tool_instance = self.context_search_tools
            elif tool_type == "filesystem_tools":
                if not self.filesystem_func_tool:
                    self.filesystem_func_tool = self._make_filesystem_tool()
                tool_instance = self.filesystem_func_tool
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                return

            if hasattr(tool_instance, method_name):
                method = getattr(tool_instance, method_name)
                from datus.tools.func_tool import trans_to_function_tool

                self.tools.append(trans_to_function_tool(method))
                logger.debug(f"Added specific tool method: {tool_type}.{method_name}")
            else:
                logger.warning(f"Method '{method_name}' not found in {tool_type}")
        except Exception as e:
            logger.error(f"Failed to setup {tool_type}.{method_name}: {e}")

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Get the system prompt for this report generation node.

        Args:
            prompt_version: Optional prompt version to use

        Returns:
            System prompt string loaded from the template
        """
        context = {
            "has_semantic_tools": bool(self.semantic_tools),
            "has_db_tools": bool(self.db_func_tool),
            "has_ask_user_tool": self.ask_user_tool is not None,
            "has_task_tool": bool(self.sub_agent_task_tool),
            "agent_config": self.agent_config,
        }

        # Add rules from configuration
        context["rules"] = self.node_config.get("rules", [])

        # Add agent description from configuration
        context["agent_description"] = self.node_config.get("agent_description", "")

        # Add datasource info
        if self.agent_config:
            from datus.utils.node_utils import build_datasource_prompt_context

            context.update(build_datasource_prompt_context(self.agent_config))
            context["db_name"] = context.get("datasource")

        version = None if prompt_version in (None, "") else str(prompt_version)

        # Construct template name: {system_prompt}_system or fallback to {node_name}_system
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        # Use prompt manager to render the template
        from datus.prompts.prompt_manager import get_prompt_manager

        pm = get_prompt_manager(agent_config=self.agent_config)
        try:
            base_prompt = pm.render_template(template_name=template_name, version=version, **context)

        except FileNotFoundError:
            # Template not found - use default gen_report template
            logger.warning(
                f"Failed to render system prompt '{system_prompt_name}', using the default gen_report template"
            )
            base_prompt = pm.render_template(template_name="gen_report_system", version=version, **context)

        return self._finalize_system_prompt(base_prompt)

    def _extract_report_result(self, actions: List[ActionHistory]) -> Optional[Dict[str, Any]]:
        """
        Extract report result from tool call actions.

        Subclasses can override this to extract specific tool results.

        Args:
            actions: List of action history entries

        Returns:
            Report result dict if found, None otherwise
        """
        # Base implementation returns None - subclasses should override
        return None

    def _extract_report_from_response(self, output: dict) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Extract report content and metadata from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"report": "markdown content", "data_sources": [...], "key_findings": [...]}

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (report_markdown: str, metadata: Optional[Dict])
            - report_markdown: The markdown report to display to user
            - metadata: Additional metadata (data_sources, key_findings) or None
        """
        try:
            from datus.utils.json_utils import strip_json_str

            # Check both 'content' and 'raw_output' fields (claude_model uses 'raw_output')
            content = output.get("content", "") or output.get("raw_output", "") or output.get("response", "")
            logger.debug(f"_extract_report_from_response input: {str(content)[:200]} (type: {type(content)})")

            # Case 1: content is already a dict
            if isinstance(content, dict):
                report = content.get("report", "")
                if report:
                    metadata = {
                        "data_sources": content.get("data_sources", []),
                        "key_findings": content.get("key_findings", []),
                    }
                    logger.debug(f"Extracted from dict: report length={len(report)}")
                    return report, metadata
                else:
                    # No report field, return content as-is
                    logger.debug("Dict format but no 'report' field, returning raw content")
                    return str(content), None

            # Case 2: content is a JSON string (possibly wrapped in markdown code blocks)
            elif isinstance(content, str) and content.strip():
                # Use strip_json_str to handle markdown code blocks and extract JSON
                cleaned_json = strip_json_str(content)
                if cleaned_json:
                    try:
                        import json_repair

                        parsed = json_repair.loads(cleaned_json)
                        if isinstance(parsed, dict):
                            report = parsed.get("report", "")
                            if report:
                                metadata = {
                                    "data_sources": parsed.get("data_sources", []),
                                    "key_findings": parsed.get("key_findings", []),
                                }
                                logger.debug(f"Extracted from JSON string: report length={len(report)}")
                                return report, metadata
                    except Exception as e:
                        logger.warning(f"Failed to parse cleaned JSON: {e}. Returning raw content.")

                # If JSON parsing failed or no report field, return original content
                return content, None

            # Fallback: return empty string if content is empty
            logger.warning(f"Could not extract report from response. Content type: {type(content)}")
            return str(content) if content else "", None

        except Exception as e:
            logger.error(f"Unexpected error extracting report: {e}", exc_info=True)
            return "", None

    def _maybe_rewrite_stream_action(self, action: ActionHistory, ctx: StreamRunContext) -> Optional[ActionHistory]:
        """Swap raw JSON payloads for rendered markdown while streaming.

        GenReport's prompt instructs the LLM to emit ``{"report": "<markdown>"}``.
        Without this rewrite the UI would briefly show raw JSON before the
        final action carries the extracted report.
        """
        if (
            action.role != ActionRole.ASSISTANT
            or action.status != ActionStatus.SUCCESS
            or not isinstance(action.output, dict)
        ):
            return None

        candidate = (
            action.output.get("content", "") or action.output.get("response", "") or action.output.get("raw_output", "")
        )
        if not isinstance(candidate, str) or not candidate:
            return None
        if '{"report"' not in candidate and '"report":' not in candidate:
            return None

        extracted_report, _ = self._extract_report_from_response(action.output)
        if not extracted_report:
            return None

        action.output["content"] = extracted_report
        action.output["response"] = extracted_report
        action.output["raw_output"] = extracted_report
        preview = extracted_report[:200] + "..." if len(extracted_report) > 200 else extracted_report
        action.messages = f"Report generated: {preview}"
        return action  # in-place mutation; return same object to short-circuit

    def _build_success_result(self, ctx: StreamRunContext) -> GenReportNodeResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            response_content = (
                ctx.last_successful_output.get("content", "")
                or ctx.last_successful_output.get("text", "")
                or ctx.last_successful_output.get("response", "")
                or str(ctx.last_successful_output)
            )

        # Final-pass JSON extraction in case the streaming rewrite missed
        # (e.g. report split across multiple assistant turns).
        report_metadata = None
        if ctx.last_successful_output:
            extracted_report, report_metadata = self._extract_report_from_response(ctx.last_successful_output)
            if extracted_report:
                response_content = extracted_report

        all_actions = ctx.action_history_manager.get_actions()
        report_result = self._extract_report_result(all_actions)
        if report_metadata and not report_result:
            report_result = report_metadata

        tokens_used = self._extract_total_tokens(all_actions)
        tool_calls = [a for a in all_actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        return GenReportNodeResult(
            success=True,
            response=response_content,
            report_result=report_result,
            tokens_used=int(tokens_used),
            action_history=[a.model_dump() for a in all_actions],
            execution_stats={
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": list({a.action_type for a in tool_calls}),
                "total_tokens": int(tokens_used),
            },
        )
