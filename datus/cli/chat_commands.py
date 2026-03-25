# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Chat-related commands for the Datus CLI.
This module provides a class to handle all chat-related commands including
chat execution, session management, and display utilities.
"""

import asyncio
import json
import platform
import re
import subprocess
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.cli._cli_utils import select_choice
from datus.cli.action_display.display import ActionHistoryDisplay
from datus.cli.execution_state import ExecutionInterrupted, auto_submit_interaction
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import interrupt_on_escape

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class ChatCommands:
    """Handles all chat-related commands and functionality."""

    def __init__(self, cli_instance: "DatusCLI"):
        """Initialize with reference to the CLI instance for shared resources."""
        self.cli = cli_instance
        self.console = cli_instance.console

        # Chat state management - unified node management
        self.current_node: ChatAgenticNode | None = None  # Can be ChatAgenticNode or GenSQLAgenticNode
        self.chat_node: ChatAgenticNode | None = None  # Kept for backward compatibility
        self.current_subagent_name: str | None = None  # Track current subagent name
        self.chat_history = []
        self.last_actions = []
        self.all_turn_actions: List[Tuple[str, List[ActionHistory]]] = []
        self._trace_verbose = False  # toggle state for post-run Ctrl+O

    def update_chat_node_tools(self):
        """Update current node tools when namespace changes."""
        if self.current_node and hasattr(self.current_node, "setup_tools"):
            self.current_node.setup_tools()
        # Keep backward compatibility
        if self.chat_node:
            self.chat_node.setup_tools()

    def _should_create_new_node(self, subagent_name: str = None) -> bool:
        """Determine if a new node should be created."""
        if self.current_node is None:
            return True

        if subagent_name:
            # Create new node if switching from regular to subagent, or subagent changed
            return self.current_subagent_name != subagent_name
        else:
            # Create new node only if switching from subagent to regular
            return bool(self.current_subagent_name)

    def _trigger_compact_for_current_node(self):
        """Trigger compact on current node before switching."""
        if self.current_node and hasattr(self.current_node, "_manual_compact"):
            try:

                async def _get_info():
                    return await self.current_node.get_session_info()

                session_info = asyncio.run(_get_info())
                if session_info.get("session_id"):
                    self.console.print("[yellow]Switching node, compacting current session...[/]")

                    async def run_compact():
                        return await self.current_node._manual_compact()

                    result = asyncio.run(run_compact())

                    if result.get("success"):
                        self.console.print("[green]✓ Session compacted successfully![/]")
                        logger.info(
                            f"Session compact details - New Token Count: {result.get('new_token_count', 'N/A')}"
                            f"Tokens Saved: {result.get('tokens_saved', 'N/A')}"
                            f"Compression Ratio: {result.get('compression_ratio', 'N/A')}"
                        )
                    else:
                        error_msg = result.get("error", "Unknown error occurred")
                        self.console.print(f"[bold red]✗ Failed to compact session:[/] {error_msg}")

            except Exception as e:
                logger.error(f"Compact error during node switch: {e}")
                self.console.print(f"[bold red]Compact error:[/] {str(e)}")

    def _create_new_node(self, subagent_name: str = None):
        """Create new node based on subagent_name and configuration.

        Node class selection priority:
        1. Hardcoded special cases (gen_semantic_model, gen_metrics, gen_sql_summary)
        2. node_class field from configuration
        3. Default to gensql
        """
        if subagent_name:
            # Get node configuration
            node_config = {}
            if hasattr(self.cli.agent_config, "agentic_nodes") and self.cli.agent_config.agentic_nodes:
                node_config = self.cli.agent_config.agentic_nodes.get(subagent_name, {})
                if hasattr(node_config, "model_dump"):
                    node_config = node_config.model_dump()

            # Get node_class from config, default to None
            node_class_type = node_config.get("node_class") if isinstance(node_config, dict) else None

            # Hardcoded special cases (existing nodes with special constructors)
            if subagent_name == "gen_semantic_model":
                from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session...[/]")
                return GenSemanticModelAgenticNode(
                    agent_config=self.cli.agent_config,
                    execution_mode="interactive",
                )
            elif subagent_name == "gen_metrics":
                from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session...[/]")
                return GenMetricsAgenticNode(
                    agent_config=self.cli.agent_config,
                    execution_mode="interactive",
                )
            elif subagent_name == "gen_sql_summary":
                from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session...[/]")
                return SqlSummaryAgenticNode(
                    node_name=subagent_name,
                    agent_config=self.cli.agent_config,
                    execution_mode="interactive",
                )
            # gen_report: either direct /gen_report command or custom subagent with node_class="gen_report"
            elif subagent_name == "gen_report" or node_class_type == "gen_report":
                from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session (gen_report)...[/]")
                return GenReportAgenticNode(
                    node_id=f"{subagent_name}_cli",
                    description=f"Report generation node for {subagent_name}",
                    node_type="gen_report",
                    input_data=None,
                    agent_config=self.cli.agent_config,
                    tools=None,
                    node_name=subagent_name,
                )
            # Use GenExtKnowledgeAgenticNode for gen_ext_knowledge
            elif subagent_name == "gen_ext_knowledge":
                from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session...[/]")
                return GenExtKnowledgeAgenticNode(
                    node_name=subagent_name,
                    agent_config=self.cli.agent_config,
                    execution_mode="interactive",
                )
            else:
                # Default: Create GenSQLAgenticNode
                from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session...[/]")
                return GenSQLAgenticNode(
                    node_id=f"{subagent_name}_cli",
                    description=f"SQL generation node for {subagent_name}",
                    node_type="gensql",
                    input_data=None,
                    agent_config=self.cli.agent_config,
                    tools=None,
                    node_name=subagent_name,
                )
        else:
            # Create ChatAgenticNode for default chat
            self.console.print("[dim]Creating new chat session...[/]")
            return ChatAgenticNode(
                node_id="chat_cli",
                description="Chat node for CLI interactions",
                node_type="chat",
                input_data=None,
                agent_config=self.cli.agent_config,
                tools=None,
            )

    def create_node_input(
        self,
        user_message: str,
        current_node,
        at_tables,
        at_metrics,
        at_sqls,
        plan_mode: bool = False,
    ):
        """Create node input based on node type - shared logic for CLI and web"""
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        if isinstance(current_node, (GenSemanticModelAgenticNode, GenMetricsAgenticNode)):
            from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

            return (
                SemanticNodeInput(
                    user_message=user_message,
                    catalog=self.cli.cli_context.current_catalog if self.cli.cli_context.current_catalog else None,
                    database=self.cli.cli_context.current_db_name if self.cli.cli_context.current_db_name else None,
                    db_schema=self.cli.cli_context.current_schema if self.cli.cli_context.current_schema else None,
                    prompt_version=None,
                    prompt_language="en",
                ),
                "semantic",
            )
        elif isinstance(current_node, SqlSummaryAgenticNode):
            from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

            return (
                SqlSummaryNodeInput(
                    user_message=user_message,
                    catalog=self.cli.cli_context.current_catalog if self.cli.cli_context.current_catalog else None,
                    database=self.cli.cli_context.current_db_name if self.cli.cli_context.current_db_name else None,
                    db_schema=self.cli.cli_context.current_schema if self.cli.cli_context.current_schema else None,
                    prompt_version=None,
                    prompt_language="en",
                ),
                "sql_summary",
            )
        elif isinstance(current_node, GenExtKnowledgeAgenticNode):
            from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput

            return (
                ExtKnowledgeNodeInput(
                    user_message=user_message,
                    prompt_version=None,
                    prompt_language="en",
                ),
                "ext_knowledge",
            )
        elif isinstance(current_node, GenSQLAgenticNode):
            from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

            return (
                GenSQLNodeInput(
                    user_message=user_message,
                    catalog=self.cli.cli_context.current_catalog if self.cli.cli_context.current_catalog else None,
                    database=self.cli.cli_context.current_db_name if self.cli.cli_context.current_db_name else None,
                    db_schema=self.cli.cli_context.current_schema if self.cli.cli_context.current_schema else None,
                    schemas=at_tables,
                    metrics=at_metrics,
                    reference_sql=at_sqls,
                    prompt_version=None,
                    prompt_language="en",
                    plan_mode=plan_mode,
                ),
                "gensql",
            )
        elif isinstance(current_node, GenReportAgenticNode):
            from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput

            return (
                GenReportNodeInput(
                    user_message=user_message,
                    catalog=self.cli.cli_context.current_catalog if self.cli.cli_context.current_catalog else None,
                    database=self.cli.cli_context.current_db_name if self.cli.cli_context.current_db_name else None,
                    db_schema=self.cli.cli_context.current_schema if self.cli.cli_context.current_schema else None,
                    prompt_version=None,
                ),
                "gen_report",
            )
        else:
            from datus.schemas.chat_agentic_node_models import ChatNodeInput

            return (
                ChatNodeInput(
                    user_message=user_message,
                    catalog=self.cli.cli_context.current_catalog if self.cli.cli_context.current_catalog else None,
                    database=self.cli.cli_context.current_db_name if self.cli.cli_context.current_db_name else None,
                    db_schema=self.cli.cli_context.current_schema if self.cli.cli_context.current_schema else None,
                    schemas=at_tables,
                    metrics=at_metrics,
                    reference_sql=at_sqls,
                    plan_mode=plan_mode,
                ),
                "chat",
            )

    def execute_chat_command(
        self,
        message: str,
        plan_mode: bool = False,
        subagent_name: Optional[str] = None,
        compact_when_new_subagent: bool = True,
    ):
        """Execute a chat command in interactive REPL mode."""
        self._execute_chat(
            message,
            plan_mode=plan_mode,
            subagent_name=subagent_name,
            compact_when_new_subagent=compact_when_new_subagent,
            interactive=True,
        )

    def _resolve_clean_output(
        self,
        sql: Optional[str],
        response: Optional[str],
        extracted_output: Optional[str],
    ) -> Optional[str]:
        """Resolve clean output text from response and extraction results.

        Used by both interactive (execute_chat_command) and non-interactive (execute_prompt_command) paths.
        """
        if sql:
            return extracted_output or response
        elif isinstance(extracted_output, dict):
            return extracted_output.get("raw_output", str(extracted_output))
        else:
            clean_output = self._extract_report_from_json(response)
            if not clean_output:
                if response is None:
                    clean_output = ""
                else:
                    try:
                        import ast

                        response_dict = ast.literal_eval(response)
                        clean_output = (
                            response_dict.get("raw_output", response) if isinstance(response_dict, dict) else response
                        )
                    except (ValueError, SyntaxError, TypeError):
                        clean_output = response
            return clean_output

    def execute_prompt_command(self, message: str):
        """Execute a single prompt non-interactively (reuses interactive logic)."""
        self._execute_chat(message, plan_mode=False, interactive=False)

    def _execute_chat(
        self,
        message: str,
        plan_mode: bool = False,
        subagent_name: Optional[str] = None,
        compact_when_new_subagent: bool = True,
        interactive: bool = True,
    ):
        """Core chat execution logic shared by interactive and non-interactive modes."""
        if not message.strip():
            self.console.print("[yellow]Please provide a message to chat with the AI.[/]")
            return

        try:
            at_tables, at_metrics, at_sqls = self.cli.at_completer.parse_at_context(message)

            if interactive:
                # Decision logic: determine if we need to create a new node
                need_new_node = self._should_create_new_node(subagent_name)

                # If creating new node and have existing node, trigger compact
                if need_new_node and self.current_node is not None and compact_when_new_subagent:
                    self._trigger_compact_for_current_node()

                # Get or create node
                if need_new_node:
                    self.current_node = self._create_new_node(subagent_name)
                    self.current_subagent_name = subagent_name if subagent_name else None
                    self.all_turn_actions = []
                    if not subagent_name:
                        self.chat_node = self.current_node

                current_node = self.current_node

                # Show session info for existing session
                if not need_new_node:
                    session_info = asyncio.run(current_node.get_session_info())
                    if session_info.get("session_id"):
                        session_display = (
                            f"[dim]Using existing session: {session_info['session_id']} "
                            f"(tokens: {session_info['token_count']}, actions: {session_info['action_count']})[/]"
                        )
                        self.console.print(session_display)
            else:
                # Non-interactive: always create a new node
                self.current_node = self._create_new_node(None)
                current_node = self.current_node

            # Create input using shared method
            node_input, node_type = self.create_node_input(
                message, current_node, at_tables, at_metrics, at_sqls, plan_mode
            )
            current_node.input = node_input

            # Initialize action history display
            action_display = ActionHistoryDisplay(self.console)
            incremental_actions = []
            node_final_action = None  # Node's final ASSISTANT action (e.g. chat_response)

            if interactive:
                self.console.print("[dim]Press ESC or Ctrl+C to interrupt[/dim]")

                async def run_chat_stream():
                    """Run chat stream — INTERACTION actions flow into incremental_actions."""
                    streaming_ctx.set_event_loop(asyncio.get_running_loop())
                    async for action in current_node.execute_stream_with_interactions(
                        action_history_manager=self.cli.actions
                    ):
                        # Skip USER actions (depth=0) — already printed by _echo_user_input
                        if action.role == ActionRole.USER and action.depth == 0:
                            continue
                        # Skip TOOL PROCESSING entries — SUCCESS version follows
                        if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                            continue
                        # Skip non-thinking ASSISTANT actions (final output) —
                        # rendered by the final response display below instead.
                        if (
                            action.role == ActionRole.ASSISTANT
                            and isinstance(action.output, dict)
                            and not action.output.get("is_thinking", True)
                        ):
                            continue
                        # Node final actions (e.g. chat_response) — keep for
                        # final response rendering but skip streaming trace.
                        # Only capture depth-0 (main node) responses; sub-agent
                        # responses (depth=1) should flow into incremental_actions
                        # so they are not mistakenly rendered as the final answer
                        # when the user interrupts execution via ESC.
                        if (
                            action.role == ActionRole.ASSISTANT
                            and action.action_type
                            and action.action_type.endswith("_response")
                            and action.depth == 0
                        ):
                            nonlocal node_final_action
                            node_final_action = action
                            continue
                        incremental_actions.append(action)

                streaming_ctx = action_display.display_streaming_actions(
                    incremental_actions,
                    history_turns=self.all_turn_actions,
                    current_user_message=message,
                    interaction_broker=current_node.interaction_broker,
                )
                with (
                    interrupt_on_escape(
                        current_node.interrupt_controller,
                        key_callbacks={b"\x0f": streaming_ctx.toggle_verbose},
                    ) as esc_guard,
                    streaming_ctx,
                ):
                    streaming_ctx.set_input_collector(self._make_input_collector(esc_guard))
                    try:
                        asyncio.run(run_chat_stream())
                    except KeyboardInterrupt:
                        current_node.interrupt_controller.interrupt()
                        logger.info("KeyboardInterrupt caught, execution interrupted gracefully")
                    except ExecutionInterrupted:
                        logger.info("ExecutionInterrupted caught, execution stopped gracefully")
            else:

                async def run_stream():
                    nonlocal node_final_action
                    async for action in current_node.execute_stream_with_interactions(
                        action_history_manager=self.cli.actions
                    ):
                        if action.role == ActionRole.INTERACTION:
                            # In non-interactive mode, auto-submit default choice for
                            # PROCESSING interactions so the node is not left hanging.
                            if action.status == ActionStatus.PROCESSING:
                                broker = current_node.interaction_broker
                                if broker:
                                    await auto_submit_interaction(broker, action)
                            continue
                        if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                            continue
                        # Skip non-thinking ASSISTANT actions (final output) —
                        # rendered by the final response display below instead.
                        if (
                            action.role == ActionRole.ASSISTANT
                            and isinstance(action.output, dict)
                            and not action.output.get("is_thinking", True)
                        ):
                            continue
                        # Node final actions (e.g. chat_response) — keep for
                        # final response rendering but skip streaming trace.
                        # Only capture depth-0 (main node) responses; sub-agent
                        # responses (depth=1) should flow into incremental_actions
                        # so they are not mistakenly rendered as the final answer
                        # when the user interrupts execution via ESC.
                        if (
                            action.role == ActionRole.ASSISTANT
                            and action.action_type
                            and action.action_type.endswith("_response")
                            and action.depth == 0
                        ):
                            node_final_action = action
                            continue
                        incremental_actions.append(action)

                with action_display.display_streaming_actions(incremental_actions):
                    try:
                        asyncio.run(run_stream())
                    except KeyboardInterrupt:
                        current_node.interrupt_controller.interrupt()
                        logger.info("KeyboardInterrupt caught, execution interrupted gracefully")
                    except ExecutionInterrupted:
                        logger.info("ExecutionInterrupted caught, execution stopped gracefully")

            # Display final response from the node's final action
            # (separated from incremental_actions to avoid streaming trace rendering)
            if node_final_action:
                final_action = node_final_action
            elif incremental_actions:
                final_action = incremental_actions[-1]
            else:
                final_action = None

            if final_action:
                if (
                    final_action.output
                    and isinstance(final_action.output, dict)
                    and final_action.status == ActionStatus.SUCCESS
                ):
                    sql = final_action.output.get("sql")
                    response = final_action.output.get("response")

                    extracted_sql, extracted_output = self._extract_sql_and_output_from_content(response)
                    sql = sql or extracted_sql

                    clean_output = self._resolve_clean_output(sql, response, extracted_output)

                    if sql:
                        self.add_in_sql_context(sql, clean_output, incremental_actions)

                    self._render_final_response(final_action)

                    # Merge node_final_action back for history tracking
                    all_actions = incremental_actions + ([node_final_action] if node_final_action else [])
                    self.last_actions = all_actions
                    self.all_turn_actions.append((message, all_actions))
                    self._trace_verbose = False  # reset toggle for new chat round

                if interactive:
                    self.cli.console.print("[bold bright_black]Press Ctrl+O to toggle trace details.[/]")

            if interactive:
                self.chat_history.append(
                    {
                        "user": message,
                        "response": (
                            final_action.output.get("response", "")
                            if final_action and final_action.output and isinstance(final_action.output, dict)
                            else ""
                        ),
                        "actions": len(incremental_actions),
                    }
                )

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _render_final_response(self, final_action: "ActionHistory") -> None:
        """Render the final response output (SQL, markdown, etc.) from a node action.

        This is used both after streaming completes and when Ctrl+O re-renders.
        Side-effect free — does not modify history or state.
        """
        if (
            not final_action
            or not final_action.output
            or not isinstance(final_action.output, dict)
            or final_action.status != ActionStatus.SUCCESS
        ):
            return

        sql = final_action.output.get("sql")
        response = final_action.output.get("response")

        extracted_sql, extracted_output = self._extract_sql_and_output_from_content(response)
        sql = sql or extracted_sql

        clean_output = self._resolve_clean_output(sql, response, extracted_output)

        if sql:
            self._display_sql_with_copy(sql)

        semantic_models = final_action.output.get("semantic_models")
        if semantic_models:
            self._display_semantic_model(semantic_models)

        sql_summary_file = final_action.output.get("sql_summary_file")
        if sql_summary_file:
            self._display_sql_summary_file(sql_summary_file)

        ext_knowledge_file = final_action.output.get("ext_knowledge_file")
        if ext_knowledge_file:
            self._display_ext_knowledge_file(ext_knowledge_file)

        if clean_output:
            self._display_markdown_response(clean_output)

    def _find_node_final_action(self, actions: List["ActionHistory"]) -> Optional["ActionHistory"]:
        """Find the node final action (e.g. chat_response) from an action list."""
        for a in reversed(actions):
            if a.role == ActionRole.ASSISTANT and a.action_type and a.action_type.endswith("_response"):
                return a
        return None

    def _display_sql_with_copy(self, sql: str):
        """
        Display SQL in a formatted panel with automatic clipboard copy functionality.

        Args:
            sql: SQL query string to display and copy
        """
        try:
            # Store SQL for reference
            self.cli.last_sql = sql

            # Try to copy to clipboard
            copied_indicator = ""
            try:
                # Try pyperclip first
                try:
                    import pyperclip

                    pyperclip.copy(sql)
                    copied_indicator = " (copied)"
                except ImportError:
                    # Fallback to system clipboard commands
                    system = platform.system()
                    if system == "Darwin":  # macOS
                        subprocess.run("pbcopy", input=sql.encode(), check=True)
                        copied_indicator = " (copied)"
                    elif system == "Linux":
                        subprocess.run("xclip", input=sql.encode(), check=True)
                        copied_indicator = " (copied)"
                    elif system == "Windows":
                        subprocess.run("clip", input=sql.encode(), shell=True, check=True)
                        copied_indicator = " (copied)"
            except Exception:
                # Clipboard copy failed, continue without it
                pass

            # Display the SQL in a formatted panel
            self.console.print()
            sql_panel = Panel(
                Syntax(sql, "sql", theme="monokai", word_wrap=True),
                title=f"[bold cyan]Generated SQL{copied_indicator}[/]",
                border_style="cyan",
                expand=False,
            )
            self.console.print(sql_panel)

        except Exception as e:
            logger.error(f"Error displaying SQL: {e}")
            # Fallback to simple display
            self.console.print(f"\n[bold cyan]Generated SQL:[/]\n```sql\n{sql}\n```")

    def _display_markdown_response(self, response: str):
        """
        Display clean response content as formatted markdown.

        Skip JSON responses since they are for backend processing only.

        Args:
            response: Clean response text to display as markdown
        """
        try:
            # Handle JSON responses - try to extract user-facing content
            stripped = response.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                # Try to extract report field from JSON
                extracted = self._extract_report_from_json(response)
                if extracted:
                    response = extracted
                # If extraction fails, fall through to display raw content

            # Display as markdown with proper formatting
            markdown_content = Markdown(response)
            self.console.print()  # Add spacing
            self.console.print(markdown_content)

        except Exception as e:
            logger.error(f"Error displaying markdown: {e}")
            # Fallback to plain text display
            self.console.print(f"\n[bold blue]Assistant:[/] {response}")

    def _display_semantic_model(self, semantic_models: Optional[List[str]]):
        """
        Display semantic model file paths.

        Args:
            semantic_models: List of semantic model file paths, or None
        """
        try:
            self.console.print()
            if not semantic_models:
                self.console.print("[bold magenta]Semantic Model Files:[/] None")
            elif len(semantic_models) == 1:
                self.console.print(f"[bold magenta]Semantic Model File:[/] [cyan]{semantic_models[0]}[/]")
            else:
                self.console.print("[bold magenta]Semantic Model Files:[/]")
                for model_file in semantic_models:
                    self.console.print(f"  [cyan]{model_file}[/]")

        except Exception as e:
            logger.error(f"Error displaying semantic models: {e}")
            # Fallback to simple display
            if semantic_models:
                models_str = ", ".join(semantic_models)
                self.console.print(f"\n[bold magenta]Semantic Model Files:[/] {models_str}")
            else:
                self.console.print("\n[bold magenta]Semantic Model Files:[/] None")

    def _display_sql_summary_file(self, sql_summary_file: str):
        """
        Display SQL summary file path.

        Args:
            sql_summary_file: SQL summary file path
        """
        try:
            self.console.print()
            self.console.print(f"[bold yellow]SQL Summary File:[/] [cyan]{sql_summary_file}[/]")

        except Exception as e:
            logger.error(f"Error displaying SQL summary file: {e}")
            # Fallback to simple display
            self.console.print(f"\n[bold yellow]SQL Summary File:[/] {sql_summary_file}")

    def _display_ext_knowledge_file(self, ext_knowledge_file: str):
        """
        Display external knowledge file path.

        Args:
            ext_knowledge_file: External knowledge file path
        """
        try:
            self.console.print()
            self.console.print(f"[bold green]External Knowledge File:[/] [cyan]{ext_knowledge_file}[/]")

        except Exception as e:
            logger.error(f"Error displaying external knowledge file: {e}")
            # Fallback to simple display
            self.console.print(f"\n[bold green]External Knowledge File:[/] {ext_knowledge_file}")

    def _make_input_collector(self, esc_guard):
        """Create a synchronous input collector callback for INTERACTION actions.

        The returned callback is invoked from the daemon thread in InlineStreamingContext
        when an INTERACTION PROCESSING action arrives.

        Reads ``contents`` / ``choices`` / ``default_choices`` from ``action.input``.
        Single question (len==1) uses select_choice; batch (len>1) iterates.
        """

        def _collect_single_choice(console, choices, default_choice, allow_free_text):
            """Collect a single choice or free-text answer."""
            if not choices:
                console.print()
                console.print("[dim](Paste supported. Enter to submit)[/]")
                return self.cli.prompt_input(message="Your input", multiline=True) or ""

            keys = list(choices.keys())
            default_key = default_choice if default_choice in keys else keys[0]
            result = select_choice(
                console,
                choices=choices,
                default=default_key,
                allow_free_text=allow_free_text,
            )
            if result in choices:
                console.print(f"[dim]Selected: {choices[result]}[/]")
            if allow_free_text and result == "":
                console.print("[yellow]No input provided.[/]")
                return ""
            return result or default_key

        def collect(action: ActionHistory, console) -> Optional[str]:

            try:
                input_data = action.input or {}
                contents = input_data.get("contents", [])
                choices_list = input_data.get("choices", [])
                default_choices = input_data.get("default_choices", [])
                allow_free_text = input_data.get("allow_free_text", False)

                with esc_guard.paused():
                    if len(contents) > 1:
                        return self._collect_batch(console, contents, choices_list)

                    # --- single question ---
                    ch = choices_list[0] if choices_list else {}
                    default = default_choices[0] if default_choices else ""
                    return _collect_single_choice(console, ch, default, allow_free_text)
            except Exception as e:
                logger.error(f"Error collecting interaction input: {e}")
                return None

        return collect

    def _collect_batch(self, console, contents: list, choices_list: list) -> Optional[str]:
        """Collect answers for a batch of questions.

        Steps through each question, showing progress (e.g. [1/3]),
        and returns a JSON-encoded list of answer strings.

        Caller is responsible for holding ``esc_guard.paused()`` context.
        """
        if not contents:
            return json.dumps([])

        answers = []
        total = len(contents)

        for idx, q_text in enumerate(contents):
            ch = choices_list[idx] if idx < len(choices_list) else {}

            # Show progress header
            if total > 1:
                if answers:
                    prev_q = contents[idx - 1]
                    short_q = prev_q[:50] + "..." if len(prev_q) > 50 else prev_q
                    console.print(f"  [green]\u2705[/green] [dim]{short_q} \u2192 {answers[-1]}[/dim]")
                console.print(f"\n  [bold bright_cyan][{idx + 1}/{total}][/bold bright_cyan] {q_text}")
            if not ch:
                console.print()
                console.print("[dim](Paste supported. Enter to submit)[/]")
                answer = self.cli.prompt_input(message="Your input", multiline=True) or ""
            else:
                default_key = next(iter(ch.keys()))
                result = select_choice(
                    console,
                    choices=ch,
                    default=default_key,
                    allow_free_text=True,
                )
                if result in ch:
                    answer = ch[result]
                    console.print(f"[dim]Selected: {answer}[/]")
                else:
                    answer = result

            answers.append(answer)

        # Show summary for multi-question batch
        if total > 1:
            console.print()
            console.print(f"  [green]\u2705 Answers submitted ({total}/{total})[/green]")
            for idx, answer in enumerate(answers):
                short_q = contents[idx][:40] + "..." if len(contents[idx]) > 40 else contents[idx]
                console.print(f"     [dim]{short_q} \u2192 {answer}[/dim]")

        return json.dumps(answers, ensure_ascii=False)

    def _extract_report_from_json(self, response: str) -> Optional[str]:
        """
        Extract 'report' field from gen_report JSON format response.

        Args:
            response: Response string that may contain JSON with 'report' field

        Returns:
            Extracted report content or None if not found
        """
        if not response:
            return None

        try:
            import json_repair

            from datus.utils.json_utils import strip_json_str

            # First try to extract JSON from code blocks or other wrappers
            stripped = response.strip()
            cleaned_json = strip_json_str(stripped)
            if not cleaned_json:
                return None
            # Check if cleaned content looks like JSON
            if not (cleaned_json.startswith("{") and cleaned_json.endswith("}")):
                return None

            parsed = json_repair.loads(cleaned_json)
            if isinstance(parsed, dict) and "report" in parsed:
                return parsed.get("report", "")
        except ValueError as e:
            logger.debug(f"Failed to extract report from JSON: {e}")
        except TypeError as e:
            logger.debug(f"Invalid input type for JSON extraction: {e}")

        return None

    def _extract_sql_and_output_from_content(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract SQL and output from content string that might contain JSON or debug format.

        Args:
            content: Content string to parse

        Returns:
            Tuple of (sql_string, output_string) - both can be None if not found
        """
        try:
            # Try to extract JSON from various patterns
            # Pattern 1: json\n{...} format
            json_match = re.search(r"json\s*\n\s*({.*?})\s*$", content, re.DOTALL)
            if json_match:
                try:
                    json_content = json.loads(json_match.group(1))
                    sql = json_content.get("sql")
                    output = json_content.get("output") or json_content.get("raw_output")
                    if output:
                        output = output.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
                    return sql, output
                except json.JSONDecodeError:
                    pass

            # Pattern 2: Direct JSON in content
            try:
                # Handle escaped quotes in the JSON string
                unescaped_content = content.replace("\\'", "'").replace('\\"', '"')
                json_content = json.loads(unescaped_content)
                sql = json_content.get("sql")
                output = json_content.get("output") or json_content.get("raw_output")
                if output and isinstance(output, str):
                    output = output.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
                return sql, output
            except json.JSONDecodeError as e:
                logger.debug(f"DEBUG: JSON decode failed for content: {content[:100]}... Error: {e}")

            # Pattern 3: Look for SQL code blocks
            sql_pattern = r"```sql\s*(.*?)\s*```"
            sql_matches = re.findall(sql_pattern, content, re.DOTALL | re.IGNORECASE)
            sql = sql_matches[0].strip() if sql_matches else None

            return sql, None

        except Exception as e:
            logger.warning(f"Failed to extract SQL and output from content: {e}")
            return None, None

    # Chat management commands

    def cmd_clear_chat(self, args: str):
        """Clear the console screen and current session."""
        # Clear the console screen using Rich
        self.console.clear()

        # Clear current session
        if self.current_node:
            try:
                self.current_node.delete_session()
                self.console.print("[green]Console and current session cleared.[/]")
            except Exception as e:
                logger.error(f"Error deleting session: {e}")
                self.console.print("[green]Console cleared. Next chat will create a new session.[/]")
        else:
            self.console.print("[green]Console cleared. Next chat will create a new session.[/]")

        # Reset all node references
        self.current_node = None
        self.chat_node = None  # Keep backward compatibility
        self.all_turn_actions = []

    def cmd_chat_info(self, args: str):
        """Display information about the current session."""
        if self.current_node:
            session_info = asyncio.run(self.current_node.get_session_info())
            if session_info.get("session_id"):
                # Determine node type for display
                node_type = "Chat" if isinstance(self.current_node, ChatAgenticNode) else "Subagent"

                self.console.print(f"[bold green]{node_type} Session Info:[/]")
                self.console.print(f"  Session ID: {session_info['session_id']}")
                self.console.print(f"  Token Count: {session_info['token_count']}")
                self.console.print(f"  Action Count: {session_info['action_count']}")
                self.console.print(f"  Total Conversations: {len(self.chat_history)}")

                if self.chat_history:
                    self.console.print("\n[bold blue]Recent Conversations:[/]")
                    for i, chat in enumerate(self.chat_history[-3:]):  # Show last 3
                        user_msg = chat["user"][:50] + "..." if len(chat["user"]) > 50 else chat["user"]
                        self.console.print(f"  {i + 1}. User: {user_msg}")
                        self.console.print(f"     Actions: {chat['actions']}")
            else:
                self.console.print("[yellow]No active session.[/]")
        else:
            self.console.print("[yellow]No active session.[/]")

    def display_inline_trace_details(self, actions: List[ActionHistory]) -> None:
        """Toggle action history between compact and verbose modes (post-run Ctrl+O)."""
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return
        self._trace_verbose = not self._trace_verbose
        mode_label = "verbose" if self._trace_verbose else "compact"
        self.console.clear()
        sys.stdout.write("\033[3J")
        sys.stdout.flush()
        self.console.print(f"[bold bright_black]  ⎯ switched to {mode_label} mode ⎯[/]")
        action_display = ActionHistoryDisplay(self.console)

        def _render_turn_response(turn_actions: List[ActionHistory]) -> None:
            """Callback to render the final response for each turn."""
            final_action = self._find_node_final_action(turn_actions)
            if final_action and final_action.depth == 0 and final_action.status == ActionStatus.SUCCESS:
                self._render_final_response(final_action)

        if self.all_turn_actions:
            action_display.render_multi_turn_history(
                self.all_turn_actions, verbose=self._trace_verbose, per_turn_callback=_render_turn_response
            )
        else:
            action_display.render_action_history(actions, verbose=self._trace_verbose)
            _render_turn_response(actions)

        self.cli.console.print("[bold bright_black]Press Ctrl+O to toggle trace details.[/]")

    def cmd_compact(self, args: str):
        """Manually compact the current session by summarizing conversation history."""
        if not self.current_node:
            self.console.print("[yellow]No active session to compact.[/]")
            return

        session_info = asyncio.run(self.current_node.get_session_info())
        if not session_info.get("session_id"):
            self.console.print("[yellow]No active session to compact.[/]")
            return

        try:
            # Determine node type for display
            node_type = "Chat" if isinstance(self.current_node, ChatAgenticNode) else "Subagent"

            # Display session info before compacting
            self.console.print(f"[bold blue]Compacting {node_type} Session...[/]")
            self.console.print(f"  Current Session ID: {session_info['session_id']}")
            self.console.print(f"  Current Token Count: {session_info['token_count']}")
            self.console.print(f"  Current Action Count: {session_info['action_count']}")

            # Call the manual compact method asynchronously
            async def run_compact():
                return await self.current_node._manual_compact()

            # Run the compact operation
            result = asyncio.run(run_compact())

            if result.get("success"):
                self.console.print("[green]✓ Session compacted successfully![/]")
                self.console.print(f"  New Token Count: {result.get('new_token_count', 'N/A')}")
                self.console.print(f"  Tokens Saved: {result.get('tokens_saved', 'N/A')}")
                self.console.print(f"  Compression Ratio: {result.get('compression_ratio', 'N/A')}")
            else:
                error_msg = result.get("error", "Unknown error occurred")
                self.console.print(f"[bold red]✗ Failed to compact session:[/] {error_msg}")

        except Exception as e:
            logger.error(f"Error during manual compact: {e}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_list_sessions(self, args: str):
        """List all available chat sessions."""
        try:
            # Create a session manager directly (don't rely on chat_node)
            from datus.models.session_manager import SessionManager

            session_manager = SessionManager(self.cli.agent_config.session_dir)
            sessions = session_manager.list_sessions()

            if not sessions:
                self.console.print("[yellow]No chat sessions found.[/]")
                return

            # Get current session ID for highlighting (if current_node exists)
            current_session_id = None
            if self.current_node and hasattr(self.current_node, "session_id"):
                current_session_id = self.current_node.session_id

            # Get session info for all sessions first to enable sorting
            sessions_with_info = []
            for session_data in sessions:
                session_id = session_data["session_id"]
                try:
                    # Get detailed session info if available
                    if self.current_node and hasattr(self.current_node, "_get_session_details"):
                        detailed_info = self.current_node._get_session_details(session_id)
                        session_data.update(detailed_info)
                    sessions_with_info.append(session_data)
                except Exception as e:
                    logger.debug(f"Could not get detailed info for session {session_id}: {e}")
                    sessions_with_info.append(session_data)

            # Sort by last_updated (most recent first)
            sessions_with_info.sort(
                key=lambda x: x.get("last_updated", x.get("created_at", "")),
                reverse=True,
            )

            # Create a table to display sessions
            table = Table(title="Chat Sessions", show_header=True, header_style="bold blue")
            table.add_column("Session ID", style="cyan", no_wrap=True)
            table.add_column("Created", style="green")
            table.add_column("Last Updated", style="yellow")
            table.add_column("Conversations", justify="right", style="magenta")
            table.add_column("SQL Queries", justify="right", style="blue")

            for session in sessions_with_info:
                session_id = session["session_id"]
                created = session.get("created_at", "Unknown")[:19]  # Trim to datetime
                updated = session.get("last_updated", "Unknown")[:19]
                conversations = session.get("total_turns", 0)
                sql_count = len(session.get("last_sql_queries", []))

                # Highlight current session
                if session_id == current_session_id:
                    session_id = f"→ {session_id}"

                table.add_row(session_id, created, updated, str(conversations), str(sql_count))

            self.console.print(table)

            if current_session_id:
                self.console.print("\n[dim]→ indicates current active session[/]")

        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    @staticmethod
    def _extract_node_type_from_session_id(session_id: str) -> str:
        """Extract node type from session_id format {node_name}_session_{uuid}."""
        if "_session_" in session_id:
            return session_id.rsplit("_session_", 1)[0]
        return "chat"

    def cmd_resume(self, args: str):
        """Resume a previous chat session."""
        from datus.cli._cli_utils import select_list
        from datus.models.session_manager import SessionManager

        try:
            session_manager = SessionManager(self.cli.agent_config.session_dir)

            # If session_id provided directly, use it
            target_session_id = args.strip() if args else None

            if not target_session_id:
                # List all sessions for user to choose
                sessions = session_manager.list_sessions(sort_by_modified=True)
                if not sessions:
                    self.console.print("[yellow]No sessions found.[/]")
                    return

                # Get session info and filter empty sessions
                session_infos = []
                for sid in sessions:
                    info = session_manager.get_session_info(sid)
                    if info.get("exists") and info.get("message_count", 0) > 0:
                        session_infos.append(info)

                if not session_infos:
                    self.console.print("[yellow]No sessions with messages found.[/]")
                    return

                # Sort by updated_at descending (newest first)
                session_infos.sort(
                    key=lambda x: x.get("updated_at") or x.get("latest_message_at") or "",
                    reverse=True,
                )

                # Build items for interactive list selector (two-line per item)
                # Line 1: first user message (no newlines, clip to screen width)
                # Line 2: session_id, updated time, message count
                list_items = []
                for info in session_infos:
                    sid = info["session_id"]
                    first_msg = (info.get("first_user_message", "") or "").replace("\n", " ").replace("\r", " ")
                    if not first_msg:
                        first_msg = "(empty)"
                    updated = (info.get("updated_at") or info.get("latest_message_at") or "N/A")[:19]
                    msg_count = str(info.get("message_count", 0))
                    list_items.append([first_msg, sid, f"Updated: {updated}", f"Msgs: {msg_count}"])

                idx = select_list(self.console, list_items)
                if idx is None:
                    self.console.print("[dim]Cancelled.[/]")
                    return

                target_session_id = session_infos[idx]["session_id"]

            # Validate the session exists
            if not session_manager.session_exists(target_session_id):
                self.console.print(f"[bold red]Session not found:[/] {target_session_id}")
                return

            # Extract node type and create the appropriate node
            node_name = self._extract_node_type_from_session_id(target_session_id)
            subagent_name = node_name if node_name != "chat" else None

            self.console.print(f"[dim]Resuming session: {target_session_id} (type: {node_name})...[/]")

            new_node = self._create_new_node(subagent_name)
            new_node.session_id = target_session_id

            # Update state
            self.current_node = new_node
            self.current_subagent_name = subagent_name
            self.chat_node = new_node if not subagent_name else self.chat_node

            # Show conversation history with full formatting
            from rich.rule import Rule

            from datus.cli.web.session_loader import SessionLoader

            loader = SessionLoader(session_dir=self.cli.agent_config.session_dir)
            messages = loader.get_session_messages(target_session_id)
            if messages:
                self.console.print(f"\n[bold green]Session resumed![/] Showing {len(messages)} message(s):\n")
                action_display = ActionHistoryDisplay(self.console)
                last_assistant_actions = []
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if role == "user":
                        self.console.print(f"[bold blue]You:[/] {content}")
                    else:
                        actions = msg.get("actions")
                        if actions:
                            action_display.render_action_history(actions)
                            last_assistant_actions = actions
                        sql = msg.get("sql")
                        if sql:
                            self._display_sql_with_copy(sql)
                        if content:
                            stripped = content.strip()
                            is_json = stripped.startswith("{") and stripped.endswith("}")
                            if not (is_json and (sql or actions)):
                                self._display_markdown_response(content)
                    self.console.print(Rule(style="dim"))
                self.console.print()
                if last_assistant_actions:
                    self.last_actions = last_assistant_actions
                    self._trace_verbose = False

                # Rebuild all_turn_actions from session messages
                self.all_turn_actions = []
                current_user_msg = ""
                for msg in messages:
                    role = msg.get("role", "unknown")
                    if role == "user":
                        current_user_msg = msg.get("content", "")
                    else:
                        actions = msg.get("actions", [])
                        if actions and current_user_msg:
                            self.all_turn_actions.append((current_user_msg, actions))
                        current_user_msg = ""

            # Get session info to check token usage
            info = session_manager.get_session_info(target_session_id)
            total_tokens = info.get("total_tokens", 0)
            if total_tokens > 50000:
                self.console.print(
                    "[yellow]Note: This session has high token usage. "
                    "Consider using .compact to reduce context size.[/]"
                )

            self.console.print("[green]You can now continue the conversation.[/]")

        except Exception as e:
            logger.error(f"Error resuming session: {e}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_rewind(self, args: str) -> Optional[str]:
        """Rewind the current session to before a specific user turn.

        Creates a new branched session containing all messages before the selected
        user turn, and returns the selected user message so the caller can prefill
        the input buffer.

        Returns:
            The selected user message text, or None if cancelled/error.
        """
        from datus.cli._cli_utils import select_list
        from datus.cli.web.session_loader import SessionLoader
        from datus.models.session_manager import SessionManager

        try:
            # Check for an active session
            if not self.current_node or not self.current_node.session_id:
                self.console.print("[yellow]No active session. Start a conversation first or use .resume.[/]")
                return

            source_session_id = self.current_node.session_id

            # Load conversation history
            loader = SessionLoader(session_dir=self.cli.agent_config.session_dir)
            messages = loader.get_session_messages(source_session_id)
            if not messages:
                self.console.print("[yellow]Current session has no messages.[/]")
                return

            # Build a table of user turns
            user_turns = []
            for msg in messages:
                if msg.get("role") == "user":
                    user_turns.append(msg)

            if not user_turns:
                self.console.print("[yellow]No user turns found in current session.[/]")
                return

            # Get user choice (from args or interactive list)
            turn_str = args.strip() if args else None
            if turn_str:
                # Direct turn number from args
                if turn_str.lower() == "q":
                    self.console.print("[dim]Cancelled.[/]")
                    return
                try:
                    turn_num = int(turn_str)
                except ValueError:
                    self.console.print("[bold red]Invalid input. Please enter a number.[/]")
                    return
                if turn_num < 1 or turn_num > len(user_turns):
                    self.console.print(f"[bold red]Invalid turn number. Must be between 1 and {len(user_turns)}.[/]")
                    return
            else:
                # Interactive list selector (two-line per item)
                # Line 1: user message (no newlines, clip to screen width)
                # Line 2: turn number, timestamp
                list_items = []
                for idx, turn_msg in enumerate(user_turns, 1):
                    content = (turn_msg.get("content", "") or "").replace("\n", " ").replace("\r", " ")
                    if not content:
                        content = "(empty)"
                    timestamp = (turn_msg.get("created_at") or "")[:19]
                    list_items.append([content, f"Turn: {idx}", timestamp])

                selected = select_list(self.console, list_items)
                if selected is None:
                    self.console.print("[dim]Cancelled.[/]")
                    return
                turn_num = selected + 1

            # Get the selected user message to return for input prefill
            rewind_user_message = user_turns[turn_num - 1].get("content", "")

            # Create the rewound session (keep everything BEFORE the selected turn)
            session_manager = SessionManager(self.cli.agent_config.session_dir)
            if turn_num == 1:
                # First turn selected — no prior messages, create a fresh session
                node_name = self._extract_node_type_from_session_id(source_session_id)
                new_node = self._create_new_node(node_name if node_name != "chat" else None)
                self.current_node = new_node
                self.current_subagent_name = node_name if node_name != "chat" else None
                self.chat_node = new_node if not self.current_subagent_name else self.chat_node
                self.chat_history = []
                self.all_turn_actions = []
                self.last_actions = []
                self.console.print(
                    f"\n[bold green]Rewound to before turn 1.[/] New session: [cyan]{new_node.session_id}[/]\n"
                )
                self.console.print("[green]Selected message placed in input buffer.[/]")
                return rewind_user_message

            new_session_id = session_manager.rewind_session(
                source_session_id, turn_num - 1, include_assistant_response=True
            )

            # Switch to the new session (same pattern as cmd_resume)
            node_name = self._extract_node_type_from_session_id(new_session_id)
            subagent_name = node_name if node_name != "chat" else None

            new_node = self._create_new_node(subagent_name)
            new_node.session_id = new_session_id

            self.current_node = new_node
            self.current_subagent_name = subagent_name
            self.chat_node = new_node if not subagent_name else self.chat_node

            # Show the rewound conversation
            from rich.rule import Rule

            new_messages = loader.get_session_messages(new_session_id)
            if new_messages:
                self.console.print(
                    f"\n[bold green]Rewound to before turn {turn_num}.[/] "
                    f"New session: [cyan]{new_session_id}[/] ({len(new_messages)} messages)\n"
                )
                action_display = ActionHistoryDisplay(self.console)
                for msg in new_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if role == "user":
                        self.console.print(f"[bold blue]You:[/] {content}")
                    else:
                        actions = msg.get("actions")
                        if actions:
                            action_display.render_action_history(actions)
                        sql = msg.get("sql")
                        if sql:
                            self._display_sql_with_copy(sql)
                        if content:
                            stripped = content.strip()
                            is_json = stripped.startswith("{") and stripped.endswith("}")
                            if not (is_json and (sql or actions)):
                                self._display_markdown_response(content)
                    self.console.print(Rule(style="dim"))
                self.console.print()

                # Rebuild all_turn_actions from rewound messages
                self.all_turn_actions = []
                current_user_msg = ""
                for msg in new_messages:
                    role = msg.get("role", "unknown")
                    if role == "user":
                        current_user_msg = msg.get("content", "")
                    else:
                        actions = msg.get("actions", [])
                        if actions and current_user_msg:
                            self.all_turn_actions.append((current_user_msg, actions))
                        current_user_msg = ""

            self.console.print("[green]Selected message placed in input buffer.[/]")
            return rewind_user_message

        except Exception as e:
            logger.error(f"Error rewinding session: {e}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")
        return None

    def add_in_sql_context(self, sql: str, explanation: str, incremental_actions: List[ActionHistory]):
        last_sql_action = None
        for i in range(len(incremental_actions) - 1, -1, -1):
            action = incremental_actions[i]
            if (
                action
                and action.is_done()
                and action.role == ActionRole.TOOL
                and action.function_name() == "read_query"
            ):
                last_sql_action = action
                break

        if last_sql_action is None:
            # No SQL action found, skip adding to context
            action_types = [
                (a.action_type, a.role.value if hasattr(a.role, "value") else a.role) for a in incremental_actions
            ]
            logger.warning(f"No SQL action found in incremental_actions. Actions: {action_types}")
            return

        action_output = last_sql_action.output
        if not action_output.get("success", "True"):
            error = action_output.get("error", "") or action_output.get("raw_output", "")
            sql_return = None
            row_count = 0
        else:
            tool_result = action_output.get("raw_output", {})
            if tool_result.get("success", 0) == 1:
                data_result = tool_result.get("result")
                error = None
                row_count = data_result.get("original_rows", 0)
                sql_return = data_result.get("compressed_data", "")
            else:
                error = tool_result.get("error", "")
                sql_return = ""
                row_count = 0

        sql_context = SQLContext(
            sql_query=sql,
            sql_error=error,
            sql_return=sql_return,
            row_count=row_count,
            explanation=explanation,
        )
        self.cli.cli_context.add_sql_context(sql_context)
