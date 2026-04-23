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
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Tuple

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.cli._cli_utils import select_choice
from datus.cli.action_display.display import ActionHistoryDisplay
from datus.cli.execution_state import ExecutionInterrupted, auto_submit_interaction
from datus.cli.list_selector_app import ListItem, ListSelectorApp
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import EscapeGuard, interrupt_on_escape


@contextmanager
def _noop_escape_guard():
    """Context manager that yields an inert :class:`EscapeGuard`.

    Used in TUI mode where prompt_toolkit owns stdin and installing the
    termios-based ESC listener would conflict. The inert guard's
    ``paused()`` is a no-op so callers written against the termios path
    (e.g. ``_make_input_collector``) continue to work unchanged.
    """
    yield EscapeGuard()


if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)

_MODEL_CONFIG_ERROR_PATTERNS = re.compile(
    r"no active model configured|not found in agent_config|unsupported model type"
    r"|api.?key|invalid.{0,10}key",
    re.IGNORECASE,
)
_AUTH_ERROR_PATTERNS = re.compile(r"unauthorized|authentication", re.IGNORECASE)
_MODEL_CONTEXT_PATTERNS = re.compile(r"model|llm|provider|openai|anthropic|gemini|codex", re.IGNORECASE)


def _is_model_config_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a model configuration or auth error."""
    msg = str(exc)
    if isinstance(exc, KeyError):
        msg = exc.args[0] if exc.args else ""
    msg = str(msg)
    return bool(
        _MODEL_CONFIG_ERROR_PATTERNS.search(msg)
        or (_AUTH_ERROR_PATTERNS.search(msg) and _MODEL_CONTEXT_PATTERNS.search(msg))
    )


def _drop_if_matches_final(
    pending: Optional[ActionHistory],
    final_action: ActionHistory,
    incremental_actions: list,
) -> Optional[ActionHistory]:
    """Reconcile a pending ASSISTANT action with an incoming *_response action.

    The model layer tags the tail LLM text with is_thinking=False and the node
    wraps the same text into a *_response action. When both are present we drop
    the pending entry so the final response is not rendered twice. If the texts
    differ (e.g. LLM emitted thinking before any tool call, then the node built
    the final response from a later turn), flush the pending entry into the
    incremental stream so the thinking is preserved.
    """
    if pending is None:
        return None
    pending_text = ""
    if isinstance(pending.output, dict):
        pending_text = (pending.output.get("raw_output") or "").strip()
    final_text = ""
    if isinstance(final_action.output, dict):
        final_text = (final_action.output.get("response") or "").strip()
    # Drop the pending entry when it has nothing to contribute (empty text)
    # or when its body duplicates the final response exactly.
    if not pending_text or pending_text == final_text:
        return None
    incremental_actions.append(pending)
    return None


class ChatCommands:
    """Handles all chat-related commands and functionality."""

    def __init__(self, cli_instance: "DatusCLI"):
        """Initialize with reference to the CLI instance for shared resources."""
        self.cli = cli_instance
        self.console = cli_instance.console

        # Chat state management - unified node management
        self.current_node: ChatAgenticNode | None = None  # Can be ChatAgenticNode or GenSQLAgenticNode
        self.current_subagent_name: str | None = None  # Track current subagent name
        self.chat_history = []
        self.last_actions = []
        self.all_turn_actions: List[Tuple[str, List[ActionHistory]]] = []
        self._trace_verbose = False  # toggle state for post-run Ctrl+O
        # Live handle to the active streaming context, consumed by the TUI
        # to route ESC (interrupt) and Ctrl+O (verbose toggle) key bindings
        # to the currently running agent loop. ``None`` when idle.
        self.current_streaming_ctx = None

    def update_chat_node_tools(self):
        """Update current node tools when datasource changes."""
        if self.current_node and hasattr(self.current_node, "setup_tools"):
            self.current_node.setup_tools()

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

    def _is_agent_switch(self, subagent_name: str = None) -> bool:
        """Check if this is a node type switch (not a fresh start)."""
        if self.current_node is None:
            return False
        effective_current = self.current_subagent_name or ""
        effective_new = subagent_name or ""
        return effective_current != effective_new

    def _copy_session_for_switch(self, prev_session_id: str, new_node) -> str:
        """Copy session data from the previous node to a new session matching the new node's name prefix.

        Uses :meth:`SessionManager.copy_session` so that the new session_id prefix
        matches ``new_node.get_node_name()`` and :meth:`_extract_node_type_from_session_id`
        resolves the correct type on ``.resume``.

        Returns:
            New session_id with the correct node-name prefix.
        """
        from datus.models.session_manager import SessionManager

        try:
            session_manager = SessionManager(self.cli.agent_config.session_dir, scope=self.cli.scope)
            return session_manager.copy_session(prev_session_id, new_node.get_node_name())
        except Exception as e:
            logger.warning(f"Failed to copy session on agent switch, starting fresh: {e}")
            return new_node.session_id  # fall back to whatever the node already has (None → auto-generate)

    def _create_new_node(self, subagent_name: str = None):
        """Create new node based on subagent_name and configuration.

        Delegates to the shared node factory for actual node creation.
        """
        from datus.agent.node.node_factory import create_interactive_node

        label = subagent_name or "chat"
        self.console.print(f"[dim]Creating new {label} session...[/]")
        return create_interactive_node(
            subagent_name, self.cli.agent_config, node_id_suffix="_cli", scope=self.cli.scope
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
        """Create node input based on node type - shared logic for CLI and web.

        Returns:
            Tuple of (node_input, node_type_string) for backward compatibility.
        """
        from datus.agent.node.node_factory import create_node_input as _create_node_input

        node_input = _create_node_input(
            user_message=user_message,
            node=current_node,
            catalog=self.cli.cli_context.current_catalog or None,
            database=self.cli.cli_context.current_db_name or None,
            db_schema=self.cli.cli_context.current_schema or None,
            at_tables=at_tables,
            at_metrics=at_metrics,
            at_sqls=at_sqls,
            plan_mode=plan_mode,
        )
        return node_input, current_node.type

    def execute_chat_command(
        self,
        message: str,
        plan_mode: bool = False,
        subagent_name: Optional[str] = None,
    ):
        """Execute a chat command in interactive REPL mode."""
        self._execute_chat(
            message,
            plan_mode=plan_mode,
            subagent_name=subagent_name,
            interactive=True,
        )

    def _resolve_clean_output(
        self,
        sql: Optional[str],
        response: Optional[str],
        extracted_output: Optional[str],
    ) -> Optional[str]:
        """Resolve clean output text from response and extraction results.

        Used by execute_chat_command to resolve clean output text.
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

    def _execute_chat(
        self,
        message: str,
        plan_mode: bool = False,
        subagent_name: Optional[str] = None,
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
                is_switch = self._is_agent_switch(subagent_name)

                # Get or create node
                if need_new_node:
                    # Copy session when switching agents to preserve conversation
                    # while keeping the session_id prefix consistent with the new node type.
                    prev_session_id = None
                    prev_node_name = None
                    if is_switch and self.current_node:
                        prev_session_id = getattr(self.current_node, "session_id", None)
                        prev_node_name = self.current_node.get_node_name()
                    self.current_node = self._create_new_node(subagent_name)
                    if prev_session_id:
                        self.current_node.session_id = self._copy_session_for_switch(prev_session_id, self.current_node)
                    if prev_node_name:
                        # Pass the previous node's name explicitly so downstream
                        # nodes (e.g. feedback) can route memory to the caller
                        # without having to parse the session id prefix.
                        self.current_node.caller_node_name = prev_node_name
                    self.current_subagent_name = subagent_name if subagent_name else None
                    if not is_switch:
                        self.all_turn_actions = []

                current_node = self.current_node

                # Show session info for existing session
                if not need_new_node or is_switch:
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
            # Buffer for ASSISTANT text tagged as non-thinking by the model layer.
            # Tail text often duplicates the node's *_response; defer rendering
            # so we can drop it when the *_response arrives, but flush it on any
            # other action so mid-turn thinking before a tool call is preserved.
            pending_non_thinking = None

            if interactive:
                self.console.print("[dim]Press ESC or Ctrl+C to interrupt[/dim]")

                async def run_chat_stream():
                    """Run chat stream — INTERACTION actions flow into incremental_actions."""
                    nonlocal node_final_action, pending_non_thinking
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
                            pending_non_thinking = _drop_if_matches_final(
                                pending_non_thinking, action, incremental_actions
                            )
                            continue
                        # Defer ASSISTANT text flagged as non-thinking — it may
                        # be the tail text that duplicates the upcoming *_response.
                        # If a previous pending is still buffered, flush it first
                        # so back-to-back non-thinking chunks are not dropped.
                        if (
                            action.role == ActionRole.ASSISTANT
                            and isinstance(action.output, dict)
                            and not action.output.get("is_thinking", True)
                        ):
                            if pending_non_thinking is not None:
                                incremental_actions.append(pending_non_thinking)
                            pending_non_thinking = action
                            continue
                        # Any other action: flush pending first to preserve order.
                        if pending_non_thinking is not None:
                            incremental_actions.append(pending_non_thinking)
                            pending_non_thinking = None
                        incremental_actions.append(action)
                    # Stream ended: flush remaining pending only when no node
                    # final action captured it (otherwise it was already handled).
                    if pending_non_thinking is not None and node_final_action is None:
                        incremental_actions.append(pending_non_thinking)
                        pending_non_thinking = None

                streaming_ctx = action_display.display_streaming_actions(
                    incremental_actions,
                    history_turns=self.all_turn_actions,
                    current_user_message=message,
                    interaction_broker=current_node.interaction_broker,
                )
                # Reprint the CLI banner at the top after Ctrl+O clears the screen.
                banner_callback = getattr(self.cli, "_print_welcome", None)
                if banner_callback is not None:
                    streaming_ctx.set_clear_header_callback(banner_callback)

                # In TUI mode the persistent prompt_toolkit Application owns
                # stdin, so the termios-based ``interrupt_on_escape`` listener
                # would fight the main input loop. Skip it and rely on
                # dedicated ESC / Ctrl+O key bindings registered on the TUI
                # (see ``DatusCLI._init_tui_app``), which consult this
                # streaming_ctx and the node's interrupt_controller directly.
                if getattr(self.cli, "_use_tui", False):
                    esc_cm = _noop_escape_guard()
                else:
                    esc_cm = interrupt_on_escape(
                        current_node.interrupt_controller,
                        key_callbacks={b"\x0f": streaming_ctx.toggle_verbose},
                    )

                # Publish the streaming context so the TUI Ctrl+O / ESC
                # bindings can locate it while the agent runs.
                self.current_streaming_ctx = streaming_ctx
                try:
                    with esc_cm as esc_guard, streaming_ctx:
                        streaming_ctx.set_input_collector(self._make_input_collector(esc_guard))
                        try:
                            self.cli.run_on_bg_loop(run_chat_stream())
                        except KeyboardInterrupt:
                            current_node.interrupt_controller.interrupt()
                            logger.info("KeyboardInterrupt caught, execution interrupted gracefully")
                        except ExecutionInterrupted:
                            logger.info("ExecutionInterrupted caught, execution stopped gracefully")
                finally:
                    self.current_streaming_ctx = None
            else:

                async def run_stream():
                    nonlocal node_final_action, pending_non_thinking
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
                            pending_non_thinking = _drop_if_matches_final(
                                pending_non_thinking, action, incremental_actions
                            )
                            continue
                        # Defer ASSISTANT text flagged as non-thinking — it may
                        # be the tail text that duplicates the upcoming *_response.
                        # If a previous pending is still buffered, flush it first
                        # so back-to-back non-thinking chunks are not dropped.
                        if (
                            action.role == ActionRole.ASSISTANT
                            and isinstance(action.output, dict)
                            and not action.output.get("is_thinking", True)
                        ):
                            if pending_non_thinking is not None:
                                incremental_actions.append(pending_non_thinking)
                            pending_non_thinking = action
                            continue
                        if pending_non_thinking is not None:
                            incremental_actions.append(pending_non_thinking)
                            pending_non_thinking = None
                        incremental_actions.append(action)
                    if pending_non_thinking is not None and node_final_action is None:
                        incremental_actions.append(pending_non_thinking)
                        pending_non_thinking = None

                with action_display.display_streaming_actions(incremental_actions):
                    try:
                        self.cli.run_on_bg_loop(run_stream())
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
            self.console.print(f"[red]Error:[/] {str(e)}")
            if _is_model_config_error(e):
                self.console.print("[yellow]Hint: Use /model to configure or switch your model.[/]")

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

    def _get_turn_token_usage_from_node(self, node) -> Optional[dict]:
        """Get detailed token usage from the node's session manager."""
        try:
            if (
                hasattr(node, "session_manager")
                and node.session_manager
                and hasattr(node, "session_id")
                and node.session_id
            ):
                return node.session_manager.get_detailed_usage(node.session_id)
        except Exception:
            pass
        return None

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
            self.console.print(f"[green]External Knowledge File:[/] [cyan]{ext_knowledge_file}[/]")

        except Exception as e:
            logger.error(f"Error displaying external knowledge file: {e}")
            # Fallback to simple display
            self.console.print(f"\n[green]External Knowledge File:[/] {ext_knowledge_file}")

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

        def _collect_multi_choice(console, choices, allow_free_text):
            """Collect multiple choices via checkbox-style selector."""
            from datus.cli._cli_utils import select_multi_choice

            if not choices:
                console.print()
                console.print("[dim](Paste supported. Enter to submit)[/]")
                return self.cli.prompt_input(message="Your input", multiline=True) or ""

            selected_keys = select_multi_choice(
                console,
                choices=choices,
                allow_free_text=allow_free_text,
            )
            if selected_keys:
                selected_display = [choices.get(k, k) for k in selected_keys]
                console.print(f"[dim]Selected: {', '.join(selected_display)}[/]")
            else:
                console.print("[yellow]No items selected.[/]")
            return json.dumps(selected_keys, ensure_ascii=False)

        def collect(action: ActionHistory, console) -> Optional[str]:

            try:
                input_data = action.input or {}
                contents = input_data.get("contents", [])
                choices_list = input_data.get("choices", [])
                default_choices = input_data.get("default_choices", [])
                allow_free_text = input_data.get("allow_free_text", False)
                multi_selects = input_data.get("multi_selects", [])

                with esc_guard.paused():
                    if len(contents) > 1:
                        return self._collect_batch(console, contents, choices_list, multi_selects)

                    # --- single question ---
                    ch = choices_list[0] if choices_list else {}
                    default = default_choices[0] if default_choices else ""
                    is_multi = multi_selects[0] if multi_selects else False

                    if is_multi and ch:
                        return _collect_multi_choice(console, ch, allow_free_text)
                    return _collect_single_choice(console, ch, default, allow_free_text)
            except Exception as e:
                logger.error(f"Error collecting interaction input: {e}")
                return None

        return collect

    def _collect_batch(
        self, console, contents: list, choices_list: list, multi_selects: Optional[list] = None
    ) -> Optional[str]:
        """Collect answers for a batch of questions.

        Steps through each question, showing progress (e.g. [1/3]),
        and returns a JSON-encoded list of answer strings.

        Caller is responsible for holding ``esc_guard.paused()`` context.
        """
        if not contents:
            return json.dumps([])

        answers = []
        total = len(contents)
        if multi_selects is None:
            multi_selects = []

        for idx, q_text in enumerate(contents):
            ch = choices_list[idx] if idx < len(choices_list) else {}
            is_multi = multi_selects[idx] if idx < len(multi_selects) else False

            # Show progress header
            if total > 1:
                if answers:
                    prev_q = contents[idx - 1]
                    short_q = prev_q[:50] + "..." if len(prev_q) > 50 else prev_q
                    prev_answer = answers[-1]
                    if isinstance(prev_answer, list):
                        prev_ch = choices_list[idx - 1] if (idx - 1) < len(choices_list) else {}
                        prev_display = ", ".join(str(prev_ch.get(k, k)) for k in prev_answer)
                    else:
                        prev_display = str(prev_answer)
                    if len(prev_display) > 60:
                        prev_display = prev_display[:60] + "..."
                    console.print(f"  [green]\u2705[/green] [dim]{short_q} \u2192 {prev_display}[/dim]")
                console.print(f"\n  [bold bright_cyan][{idx + 1}/{total}][/bold bright_cyan] {q_text}")
            if not ch:
                console.print()
                console.print("[dim](Paste supported. Enter to submit)[/]")
                answer = self.cli.prompt_input(message="Your input", multiline=True) or ""
            elif is_multi:
                from datus.cli._cli_utils import select_multi_choice

                selected_keys = select_multi_choice(
                    console,
                    choices=ch,
                    allow_free_text=True,
                )
                if selected_keys:
                    selected_display = [ch.get(k, k) for k in selected_keys]
                    console.print(f"[dim]Selected: {', '.join(selected_display)}[/]")
                    answer = selected_keys
                else:
                    console.print("[yellow]No items selected.[/]")
                    answer = []
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
        self.all_turn_actions = []

    def cmd_chat_info(self, args: str):
        """Display information about the current session."""
        if self.current_node:
            session_info = asyncio.run(self.current_node.get_session_info())
            if session_info.get("session_id"):
                # Determine node type for display
                node_type = "Chat" if isinstance(self.current_node, ChatAgenticNode) else "Subagent"

                self.console.print(f"[green]{node_type} Session Info:[/]")
                self.console.print(f"  Session ID: {session_info['session_id']}")
                self.console.print(f"  Action Count: {session_info['action_count']}")
                self.console.print(f"  Total Conversations: {len(self.chat_history)}")

                # Detailed token usage
                turn_usage = self._get_turn_token_usage_from_node(self.current_node)
                if turn_usage:
                    total = turn_usage.get("total", {})
                    total_tokens = total.get("total_tokens", 0)
                    inp = total.get("input_tokens", 0)
                    out = total.get("output_tokens", 0)
                    cached = total.get("cached_tokens", 0)
                    self.console.print(f"  Token Usage: {total_tokens:,} total ({inp:,} in / {out:,} out)")
                    if cached > 0:
                        rate = cached / inp * 100 if inp > 0 else 0
                        self.console.print(f"  Cached Tokens: {cached:,} ({rate:.1f}% hit rate)")
                else:
                    token_count = session_info.get("token_count", 0)
                    self.console.print(f"  Token Count: {token_count}")

                ctx_length = session_info.get("context_length", 0)
                last_turn_usage = asyncio.run(self.current_node.get_last_turn_usage())
                ctx_tokens = last_turn_usage.session_total_tokens if last_turn_usage else 0
                if ctx_length and ctx_tokens:
                    ratio = ctx_tokens / ctx_length * 100
                    self.console.print(f"  Context: {ctx_tokens:,}/{ctx_length:,} ({ratio:.1f}%)")

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
        banner_callback = getattr(self.cli, "_print_welcome", None)
        if banner_callback is not None:
            banner_callback()
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

                # Reload in-memory state from the compacted session
                self._reload_state_from_session()
            else:
                error_msg = result.get("error", "Unknown error occurred")
                self.console.print(f"[red]✗ Failed to compact session:[/] {error_msg}")

        except Exception as e:
            logger.error(f"Error during manual compact: {e}")
            self.console.print(f"[red]Error:[/] {str(e)}")

    def _reload_state_from_session(self):
        """Reload in-memory state from the current session after compaction.

        Clears accumulated action/chat history and rebuilds it from the
        persisted session messages, then re-renders the conversation on
        screen so the CLI display matches the compacted DB state.
        """
        from datus.models.session_manager import SessionManager

        session_id = self.current_node.session_id
        session_manager = SessionManager(self.cli.agent_config.session_dir, scope=self.cli.scope)
        messages = session_manager.get_session_messages(session_id)

        # Reset in-memory state
        self.all_turn_actions = []
        self.last_actions = []
        self.chat_history = []
        self._trace_verbose = False
        if self.current_node:
            self.current_node.actions = []

        # Rebuild state from session messages
        if messages:
            from rich.rule import Rule

            self.console.print()
            action_display = ActionHistoryDisplay(self.console)
            last_assistant_actions = []
            current_user_msg = ""

            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    current_user_msg = content
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
                    # Rebuild all_turn_actions
                    if actions and current_user_msg:
                        self.all_turn_actions.append((current_user_msg, actions))
                    current_user_msg = ""
                self.console.print(Rule(style="dim"))

            if last_assistant_actions:
                self.last_actions = last_assistant_actions

    @staticmethod
    def _extract_node_type_from_session_id(session_id: str) -> str:
        """Extract node type from session_id format {node_name}_session_{uuid}."""
        from datus.models.session_manager import extract_agent_from_session_id

        return extract_agent_from_session_id(session_id)

    def cmd_resume(self, args: str):
        """Resume a previous chat session for the active agent."""
        from datus.models.session_manager import SessionManager, session_matches_agent

        try:
            session_manager = SessionManager(self.cli.agent_config.session_dir, scope=self.cli.scope)

            # If session_id provided directly, use it
            target_session_id = args.strip() if args else None
            agent_label = self.current_subagent_name or "chat"

            if not target_session_id:
                # List sessions for the current agent only
                sessions = session_manager.list_sessions(sort_by_modified=True)
                sessions = [sid for sid in sessions if session_matches_agent(sid, self.current_subagent_name)]
                if not sessions:
                    self.console.print(f"[yellow]No sessions found for agent '{agent_label}'.[/]")
                    return

                # Get session info and filter empty sessions
                session_infos = []
                for sid in sessions:
                    info = session_manager.get_session_info(sid)
                    if info.get("exists") and info.get("message_count", 0) > 0:
                        session_infos.append(info)

                if not session_infos:
                    self.console.print(f"[yellow]No sessions with messages found for agent '{agent_label}'.[/]")
                    return

                # Sort by updated_at descending (newest first)
                session_infos.sort(
                    key=lambda x: x.get("updated_at") or x.get("latest_message_at") or "",
                    reverse=True,
                )

                items = []
                for info in session_infos:
                    sid = info["session_id"]
                    raw_first_msg = info.get("first_user_message", "") or ""
                    if not isinstance(raw_first_msg, str):
                        raw_first_msg = str(raw_first_msg)
                    first_msg = raw_first_msg.replace("\n", " ").replace("\r", " ")
                    if not first_msg:
                        first_msg = "(empty)"
                    updated = (info.get("updated_at") or info.get("latest_message_at") or "N/A")[:19]
                    msg_count = str(info.get("message_count", 0))
                    items.append(
                        ListItem(key=sid, primary=first_msg, secondary=f"{sid}  Updated: {updated}  Msgs: {msg_count}")
                    )

                app = ListSelectorApp(title=f"Resume session ({agent_label})", items=items)
                tui_app = getattr(self.cli, "tui_app", None)
                if tui_app is not None:
                    with tui_app.suspend_input():
                        selection = app.run()
                else:
                    selection = app.run()
                if selection is None:
                    self.console.print("[dim]Cancelled.[/]")
                    return

                target_session_id = selection.key

            # Validate the session exists
            if not session_manager.session_exists(target_session_id):
                self.console.print(f"[red]Session not found:[/] {target_session_id}")
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

            # Show conversation history with full formatting
            from rich.rule import Rule

            messages = session_manager.get_session_messages(target_session_id)
            if messages:
                self.console.print(f"\n[green]Session resumed![/] Showing {len(messages)} message(s):\n")
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

            self.console.print("[green]You can now continue the conversation.[/]")

        except Exception as e:
            logger.error(f"Error resuming session: {e}")
            self.console.print(f"[red]Error:[/] {str(e)}")

    def cmd_rewind(self, args: str) -> Optional[str]:
        """Rewind the current session to before a specific user turn.

        Creates a new branched session containing all messages before the selected
        user turn, and returns the selected user message so the caller can prefill
        the input buffer.

        Returns:
            The selected user message text, or None if cancelled/error.
        """
        from datus.models.session_manager import SessionManager

        try:
            # Check for an active session
            if not self.current_node or not self.current_node.session_id:
                self.console.print("[yellow]No active session. Start a conversation first or use .resume.[/]")
                return

            source_session_id = self.current_node.session_id
            session_manager = SessionManager(self.cli.agent_config.session_dir, scope=self.cli.scope)

            # Load conversation history
            messages = session_manager.get_session_messages(source_session_id)
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
                    self.console.print("[red]Invalid input. Please enter a number.[/]")
                    return
                if turn_num < 1 or turn_num > len(user_turns):
                    self.console.print(f"[red]Invalid turn number. Must be between 1 and {len(user_turns)}.[/]")
                    return
            else:
                items = []
                for idx, turn_msg in enumerate(user_turns, 1):
                    content = (turn_msg.get("content", "") or "").replace("\n", " ").replace("\r", " ")
                    if not content:
                        content = "(empty)"
                    timestamp = (turn_msg.get("created_at") or "")[:19]
                    items.append(ListItem(key=str(idx), primary=content, secondary=f"Turn: {idx}  {timestamp}"))

                app = ListSelectorApp(title="Rewind to before turn", items=items)
                tui_app = getattr(self.cli, "tui_app", None)
                if tui_app is not None:
                    with tui_app.suspend_input():
                        selection = app.run()
                else:
                    selection = app.run()
                if selection is None:
                    self.console.print("[dim]Cancelled.[/]")
                    return
                turn_num = int(selection.key)

            # Get the selected user message to return for input prefill
            rewind_user_message = user_turns[turn_num - 1].get("content", "")

            # Create the rewound session (keep everything BEFORE the selected turn)
            if turn_num == 1:
                # First turn selected — no prior messages, create a fresh session
                node_name = self._extract_node_type_from_session_id(source_session_id)
                new_node = self._create_new_node(node_name if node_name != "chat" else None)
                self.current_node = new_node
                self.current_subagent_name = node_name if node_name != "chat" else None
                self.chat_history = []
                self.all_turn_actions = []
                self.last_actions = []
                self.console.print(
                    f"\n[green]Rewound to before turn 1.[/] New session: [cyan]{new_node.session_id}[/]\n"
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

            # Show the rewound conversation
            from rich.rule import Rule

            new_messages = session_manager.get_session_messages(new_session_id)
            if new_messages:
                self.console.print(
                    f"\n[green]Rewound to before turn {turn_num}.[/] "
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
            self.console.print(f"[red]Error:[/] {str(e)}")
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
