# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
UniversalAgenticNode — a general-purpose agentic node that loads all available tools.

Workflow logic is defined by SKILL.md files; this node simply provides the full
tool palette (DB, semantic, filesystem, generation, skill, ask-user, etc.) so
that any skill can orchestrate them.
"""

import os
from typing import AsyncGenerator, Optional

import pandas as pd

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.compare_agentic_node import CompareAgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.node_models import SQLContext, SqlTask
from datus.schemas.universal_agentic_node_models import UniversalNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.utils.benchmark_utils import ComparisonOutcome, TableComparator
from datus.utils.loggings import get_logger
from datus.utils.message_utils import MessagePart, build_structured_content
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)


class UniversalAgenticNode(AgenticNode):
    """
    General-purpose agentic node that registers all available tools.

    Designed to work with any skill — the SKILL.md tells the LLM which
    tools to use and how to orchestrate the workflow.  The node itself
    is tool-agnostic: it simply provides the full set.
    """

    def __init__(
        self,
        node_name: str,
        agent_config: AgentConfig,
        execution_mode: str = "interactive",
        subject_tree: Optional[list] = None,
        build_mode: str = "incremental",
    ):
        self.configured_node_name = node_name
        self.execution_mode = execution_mode
        self.subject_tree = subject_tree
        self.build_mode = build_mode

        # max_turns from agentic_nodes configuration
        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        # Verification retry (ext_knowledge scenario)
        self.max_verification_retries = 3
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_verification_retries = agentic_node_config.get("max_verification_retries", 3)

        self._verification_passed: bool = False
        self._last_verification_result = None
        self._verification_attempt_count: int = 0

        # Resolve filesystem root path based on node_name
        path_manager = get_path_manager()
        if node_name in ("gen_semantic_model", "gen_metrics"):
            self.workspace_dir = str(path_manager.semantic_model_path(agent_config.current_namespace))
        elif node_name == "gen_ext_knowledge":
            self.workspace_dir = str(path_manager.ext_knowledge_path(agent_config.current_namespace))
        else:
            home_str = str(agent_config.home) if agent_config.home else "."
            self.workspace_dir = os.path.expanduser(home_str)

        from datus.configuration.node_type import NodeType

        super().__init__(
            node_id=f"{node_name}_node",
            description=f"Universal agentic node: {node_name}",
            node_type=NodeType.TYPE_CHAT,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
        )

        # Tool instances (populated by setup_tools)
        self.db_func_tool: Optional[DBFuncTool] = None
        self.conn = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.date_parsing_tools = None
        self.skill_func_tool: Optional[SkillFuncTool] = None
        self.hooks = None

        self.setup_tools()

    def get_node_name(self) -> str:
        return self.configured_node_name

    # ── Tool Setup ──────────────────────────────────────────────────────

    def setup_tools(self):
        """Register all available tools. Each _setup_* swallows its own errors."""
        if not self.agent_config:
            return

        self.tools = []

        self._setup_db_tools()
        self._setup_gen_semantic_model_tools()
        self._setup_semantic_tools()
        self._setup_context_search_tools()
        self._setup_generation_tools_all()
        self._setup_filesystem_tools()
        self._setup_date_parsing_tools()
        self._setup_skill_tools()
        self._setup_platform_doc_tools()
        self._setup_verify_sql_tool()
        self._setup_ask_user_tool()

        if self.execution_mode == "interactive":
            self._setup_hooks()

        logger.debug(f"UniversalAgenticNode setup {len(self.tools)} tools: {[t.name for t in self.tools]}")

    def _setup_db_tools(self):
        try:
            db_manager = db_manager_instance(self.agent_config.namespaces)
            self.conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
            self.db_func_tool = DBFuncTool(self.conn, agent_config=self.agent_config)
            self.tools.extend(self.db_func_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_gen_semantic_model_tools(self):
        try:
            if not self.db_func_tool:
                return
            from datus.tools.func_tool.gen_semantic_model_tools import GenSemanticModelTools

            gen_sm_tools = GenSemanticModelTools(self.db_func_tool)
            self.tools.extend(gen_sm_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup gen_semantic_model tools: {e}")

    def _setup_semantic_tools(self):
        try:
            from datus.tools.func_tool.semantic_tools import SemanticTools

            adapter_type = "metricflow"
            if (
                hasattr(self.agent_config, "agentic_nodes")
                and self.configured_node_name in self.agent_config.agentic_nodes
            ):
                node_config = self.agent_config.agentic_nodes[self.configured_node_name]
                if isinstance(node_config, dict) and node_config.get("semantic_adapter"):
                    adapter_type = node_config.get("semantic_adapter")

            semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.configured_node_name,
                adapter_type=adapter_type,
            )
            self.tools.extend(semantic_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup semantic tools: {e}")

    def _setup_context_search_tools(self):
        try:
            self.context_search_tools = ContextSearchTools(self.agent_config)
            self.tools.extend(self.context_search_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup context search tools: {e}")

    def _setup_generation_tools_all(self):
        try:
            self.generation_tools = GenerationTools(self.agent_config)
            self.tools.append(trans_to_function_tool(self.generation_tools.check_semantic_object_exists))
            self.tools.append(trans_to_function_tool(self.generation_tools.end_semantic_model_generation))
            self.tools.append(trans_to_function_tool(self.generation_tools.end_metric_generation))
        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_filesystem_tools(self):
        try:
            self.filesystem_func_tool = FilesystemFuncTool(root_path=self.workspace_dir)
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.read_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.read_multiple_files))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.write_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.edit_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.list_directory))
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_date_parsing_tools(self):
        try:
            from datus.tools.func_tool.date_parsing_tools import DateParsingTools

            self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            self.tools.extend(self.date_parsing_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup date parsing tools: {e}")

    def _setup_skill_tools(self):
        try:
            from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule

            base_config = self.agent_config.permissions_config
            if base_config is not None:
                base_config = base_config.model_copy(deep=True)
            else:
                base_config = PermissionConfig()

            has_bash_rule = any(r.tool == "skills" and r.pattern == "skill_execute_command" for r in base_config.rules)
            if not has_bash_rule:
                base_config.rules.insert(
                    0,
                    PermissionRule(
                        tool="skills",
                        pattern="skill_execute_command",
                        permission=PermissionLevel.ASK,
                    ),
                )

            from datus.tools.permission.permission_manager import PermissionManager

            self.permission_manager = PermissionManager(global_config=base_config)

            self.skill_manager = SkillManager(permission_manager=self.permission_manager)

            skill_extra_env = {}
            if self.agent_config:
                skill_extra_env["DATUS_HOME"] = str(self.agent_config.home) if self.agent_config.home else ""
                skill_extra_env["DATUS_NAMESPACE"] = self.agent_config.current_namespace or ""
                # config_path lives on ConfigurationManager, not AgentConfig
                try:
                    from datus.configuration.agent_config_loader import configuration_manager

                    cm = configuration_manager()
                    skill_extra_env["DATUS_CONFIG_PATH"] = str(cm.config_path) if cm and cm.config_path else ""
                except Exception:
                    skill_extra_env["DATUS_CONFIG_PATH"] = ""

            self.skill_func_tool = SkillFuncTool(
                manager=self.skill_manager,
                node_name=self.configured_node_name,
                extra_env=skill_extra_env,
            )
            self.tools.extend(self.skill_func_tool.available_tools())
            logger.debug(f"Setup skill tools: {self.skill_manager.get_skill_count()} skills discovered")
        except Exception as e:
            logger.error(f"Failed to setup skill tools: {e}")

    def _setup_platform_doc_tools(self):
        try:
            from datus.tools.func_tool import PlatformDocSearchTool

            platform_doc_tool = PlatformDocSearchTool(self.agent_config)
            self.tools.extend(platform_doc_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup platform doc tools: {e}")

    def _setup_verify_sql_tool(self):
        try:
            self.tools.append(trans_to_function_tool(self.verify_sql))
        except Exception as e:
            logger.error(f"Failed to setup verify_sql tool: {e}")

    def _setup_ask_user_tool(self):
        try:
            from datus.tools.func_tool.ask_user_tool import AskUserTool

            broker = self._get_or_create_broker()
            ask_user_tool = AskUserTool(broker=broker)
            self.tools.extend(ask_user_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup ask_user tool: {e}")

    def _setup_hooks(self):
        try:
            broker = self._get_or_create_broker()
            self.hooks = GenerationHooks(broker=broker, agent_config=self.agent_config)
        except Exception as e:
            logger.error(f"Failed to setup hooks: {e}")

    # ── verify_sql (from GenExtKnowledgeAgenticNode) ────────────────────

    def verify_sql(self, sql: str) -> FuncToolResult:
        """
        Validate SQL against a hidden reference. The reference SQL is not exposed.

        Compares execution results of the provided SQL with a hidden reference.
        The model cannot see the reference SQL — it can only learn from comparison
        feedback (match rate, column differences, data preview, and improvement suggestions).

        Args:
            sql: The SQL to validate.

        Returns:
            FuncToolResult with success=1 if matched, success=0 with suggestions otherwise.
        """
        from dataclasses import dataclass

        @dataclass
        class VerifyResult:
            success: bool
            match_rate: float
            error: Optional[str] = None
            user_df: Optional[pd.DataFrame] = None
            gold_df: Optional[pd.DataFrame] = None
            outcome: Optional[ComparisonOutcome] = None

        # Core verification
        if not hasattr(self, "_gold_sql") or not self._gold_sql:
            self._verification_passed = True
            return FuncToolResult(success=1, result="No reference available. Your SQL will be accepted.")

        connector = self.conn
        if not connector:
            return FuncToolResult(success=0, error="No database connection available for verification.")

        # Execute user SQL
        try:
            user_result = connector.execute_query(sql, result_format="pandas")
            if not user_result.success:
                result = VerifyResult(success=False, match_rate=0.0, error=f"SQL execution failed: {user_result.error}")
            else:
                user_df = user_result.sql_return
                if not isinstance(user_df, pd.DataFrame):
                    user_df = pd.DataFrame(user_df) if user_result.sql_return else pd.DataFrame()

                # Execute gold SQL
                gold_result = connector.execute_query(self._gold_sql, result_format="pandas")
                if not gold_result.success:
                    result = VerifyResult(success=False, match_rate=0.0, error=f"Gold SQL error: {gold_result.error}")
                else:
                    gold_df = gold_result.sql_return
                    if not isinstance(gold_df, pd.DataFrame):
                        gold_df = pd.DataFrame(gold_df) if gold_result.sql_return else pd.DataFrame()

                    comparator = TableComparator()
                    outcome = comparator.compare(user_df, gold_df)
                    result = VerifyResult(
                        success=(outcome.match_rate == 1.0),
                        match_rate=outcome.match_rate,
                        user_df=user_df,
                        gold_df=gold_df,
                        outcome=outcome,
                    )
        except Exception as e:
            result = VerifyResult(success=False, match_rate=0.0, error=str(e))

        # Update state
        self._last_verification_result = result
        self._verification_passed = result.success

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

        if result.error:
            return FuncToolResult(success=0, error=f"SQL execution error: {result.error}", result={"match_rate": 0})

        outcome = result.outcome
        # Generate suggestions via CompareAgenticNode
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

        return FuncToolResult(
            success=0,
            error=f"SQL verification FAILED! Match rate: {result.match_rate * 100:.1f}%",
            result={
                "match_rate": result.match_rate,
                "your_result_shape": (
                    f"{outcome.actual_shape[0]} rows x {outcome.actual_shape[1]} columns"
                    if outcome and outcome.actual_shape
                    else "N/A"
                ),
                "expected_result_shape": (
                    f"{outcome.expected_shape[0]} rows x {outcome.expected_shape[1]} columns"
                    if outcome and outcome.expected_shape
                    else "N/A"
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
        try:
            sql_task = SqlTask(
                database_type=self.agent_config.database_type if hasattr(self.agent_config, "database_type") else "",
                database_name=(
                    self.agent_config.current_database if hasattr(self.agent_config, "current_database") else ""
                ),
                task=getattr(self, "_current_question", "SQL verification task"),
                external_knowledge=str(generated_knowledge) if generated_knowledge else "",
            )
            sql_context = SQLContext(
                sql_query=user_sql,
                explanation="User generated SQL for verification",
                sql_return=user_result,
                sql_error=user_error or "",
            )
            compare_input = CompareInput(
                sql_task=sql_task,
                sql_context=sql_context,
                expectation=f"Expected SQL:\n{gold_sql}\n\nExpected Result:\n{gold_result}",
            )
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

    # ── System Prompt ───────────────────────────────────────────────────

    def _prepare_template_context(self, user_input) -> dict:
        """Build template context shared by all generation workflows."""
        context = {}
        context["native_tools"] = ", ".join([t.name for t in self.tools]) if self.tools else "None"
        context["mcp_tools"] = ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None"
        context["semantic_model_dir"] = self.workspace_dir

        # Subject tree handling (metrics / ext_knowledge)
        if self.subject_tree:
            context["has_subject_tree"] = True
            context["subject_tree"] = self.subject_tree
        else:
            context["has_subject_tree"] = False
            context["existing_subject_trees"] = self._get_existing_subject_trees()

        # ext_knowledge: user-specified subject path
        if hasattr(user_input, "subject_path") and user_input.subject_path:
            context["has_user_specified_subject"] = True
            context["user_subject_path"] = user_input.subject_path
        else:
            context["has_user_specified_subject"] = False

        # ext_knowledge dir alias
        if self.configured_node_name == "gen_ext_knowledge":
            context["ext_knowledge_dir"] = self.workspace_dir

        return context

    def _get_existing_subject_trees(self) -> list:
        """Query existing subject_tree values from the relevant storage."""
        try:
            if self.configured_node_name == "gen_metrics":
                from datus.storage.metric.store import MetricRAG

                rag = MetricRAG(self.agent_config)
                return sorted(rag.storage.get_subject_tree_flat())
            elif self.configured_node_name == "gen_ext_knowledge":
                from datus.storage.ext_knowledge.store import ExtKnowledgeRAG

                rag = ExtKnowledgeRAG(self.agent_config)
                return sorted(rag.store.get_subject_tree_flat())
            elif self.configured_node_name == "gen_semantic_model":
                from datus.storage.metric.store import MetricRAG

                rag = MetricRAG(self.agent_config)
                return sorted(rag.storage.get_subject_tree_flat())
        except Exception as e:
            logger.error(f"Error getting existing subject_trees: {e}")
        return []

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """Get system prompt: skill content > Jinja2 template fallback."""
        version = self.node_config.get("prompt_version")
        template_name = "universal_system"

        try:
            template_vars = {
                "agent_config": self.agent_config,
                "conversation_summary": conversation_summary,
            }
            if template_context:
                template_vars.update(template_context)

            from datus.prompts.prompt_manager import prompt_manager

            base_prompt = prompt_manager.render_template(template_name=template_name, version=version, **template_vars)
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError:
            # No template found — use a minimal prompt; skill content will be injected by _finalize_system_prompt
            base_prompt = (
                f"You are an AI assistant for the '{self.configured_node_name}' workflow. "
                "Follow the loaded skill instructions carefully."
            )
            if conversation_summary:
                base_prompt += f"\n\nPrevious conversation summary:\n{conversation_summary}"
            return self._finalize_system_prompt(base_prompt)

        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    # ── Execution ───────────────────────────────────────────────────────

    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        if self.input is None:
            raise ValueError("Input not set. Set self.input before calling execute_stream.")

        user_input = self.input

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
            session = None
            conversation_summary = None
            if self.execution_mode == "interactive":
                await self._auto_compact()
                session, conversation_summary = self._get_or_create_session()

            template_context = self._prepare_template_context(user_input)
            system_instruction = self._get_system_prompt(conversation_summary, template_context)

            # Enhanced message
            enhanced_message = user_input.user_message
            enhanced_parts = []
            if hasattr(user_input, "catalog") and (
                getattr(user_input, "catalog", None)
                or getattr(user_input, "database", None)
                or getattr(user_input, "db_schema", None)
            ):
                context_parts = []
                if getattr(user_input, "catalog", None):
                    context_parts.append(f"catalog: {user_input.catalog}")
                if getattr(user_input, "database", None):
                    context_parts.append(f"database: {user_input.database}")
                if getattr(user_input, "db_schema", None):
                    context_parts.append(f"schema: {user_input.db_schema}")
                enhanced_parts.append(f'Context: {", ".join(context_parts)}')

            if enhanced_parts:
                enhanced_message = build_structured_content(
                    [
                        MessagePart(type="enhanced", content="\n\n".join(enhanced_parts)),
                        MessagePart(type="user", content=user_input.user_message),
                    ]
                )

            # Stream
            response_content = ""
            tokens_used = 0
            last_successful_output = None

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=self.hooks if self.execution_mode == "interactive" else None,
                interrupt_controller=self.interrupt_controller,
            ):
                yield stream_action

                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        raw_output = stream_action.output.get("raw_output", "")
                        if isinstance(raw_output, dict):
                            response_content = raw_output
                        elif raw_output:
                            response_content = raw_output

            if not response_content and last_successful_output:
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = raw_output
                elif raw_output:
                    response_content = raw_output
                else:
                    response_content = str(last_successful_output)

            # Token usage
            if self.execution_mode == "interactive":
                for act in reversed(action_history_manager.get_actions()):
                    if act.role == "assistant" and act.output and isinstance(act.output, dict):
                        usage_info = act.output.get("usage", {})
                        if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                            tokens_used = usage_info.get("total_tokens", 0)
                            if tokens_used > 0:
                                break

            # Build result
            result = UniversalNodeResult(
                success=True,
                output=response_content if isinstance(response_content, str) else str(response_content),
                tokens_used=int(tokens_used),
            )

            self.actions.extend(action_history_manager.get_actions())

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=f"{self.configured_node_name}_response",
                messages=f"{self.get_node_name()} interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise

        except Exception as e:
            logger.error(f"{self.get_node_name()} execution error: {e}")

            error_result = UniversalNodeResult(
                success=False,
                error=str(e),
                output="Sorry, I encountered an error while processing your request.",
                tokens_used=0,
            )

            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {str(e)}",
            )

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
