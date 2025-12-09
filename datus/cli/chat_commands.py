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
from typing import TYPE_CHECKING, List, Optional, Tuple

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.cli.action_history_display import ActionHistoryDisplay
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.utils.loggings import get_logger

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
                session_info = self.current_node.get_session_info()
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
        """Create new node based on subagent_name."""
        if subagent_name:
            # Use SemanticAgenticNode for gen_semantic_model and gen_metrics
            if subagent_name in ["gen_semantic_model", "gen_metrics"]:
                from datus.agent.node.semantic_agentic_node import SemanticAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session...[/]")
                return SemanticAgenticNode(
                    node_name=subagent_name,
                    agent_config=self.cli.agent_config,
                    execution_mode="interactive",
                )
            # Use SqlSummaryAgenticNode for gen_sql_summary
            elif subagent_name == "gen_sql_summary":
                from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

                self.console.print(f"[dim]Creating new {subagent_name} session...[/]")
                return SqlSummaryAgenticNode(
                    node_name=subagent_name,
                    agent_config=self.cli.agent_config,
                    execution_mode="interactive",
                )
            else:
                # Create GenSQLAgenticNode for other subagents
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
        self, user_message: str, current_node, at_tables, at_metrics, at_sqls, plan_mode: bool = False
    ):
        """Create node input based on node type - shared logic for CLI and web"""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.agent.node.semantic_agentic_node import SemanticAgenticNode
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        if isinstance(current_node, SemanticAgenticNode):
            from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

            return (
                SemanticNodeInput(
                    user_message=user_message,
                    catalog=self.cli.cli_context.current_catalog if self.cli.cli_context.current_catalog else None,
                    database=self.cli.cli_context.current_db_name if self.cli.cli_context.current_db_name else None,
                    db_schema=self.cli.cli_context.current_schema if self.cli.cli_context.current_schema else None,
                    prompt_version="1.0",
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
                    prompt_version="1.0",
                    prompt_language="en",
                ),
                "sql_summary",
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
                    prompt_version="1.0",
                    prompt_language="en",
                ),
                "gensql",
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
        self, message: str, plan_mode: bool = False, subagent_name: str = None, compact_when_new_subagent: bool = True
    ):
        """Execute a chat command with simplified node management."""
        if not message.strip():
            self.console.print("[yellow]Please provide a message to chat with the AI.[/]")
            return

        try:
            at_tables, at_metrics, at_sqls = self.cli.at_completer.parse_at_context(message)

            # Decision logic: determine if we need to create a new node
            need_new_node = self._should_create_new_node(subagent_name)

            # If creating new node and have existing node, trigger compact
            if need_new_node and self.current_node is not None and compact_when_new_subagent:
                self._trigger_compact_for_current_node()

            # Get or create node
            if need_new_node:
                self.current_node = self._create_new_node(subagent_name)
                self.current_subagent_name = subagent_name if subagent_name else None
                if not subagent_name:
                    self.chat_node = self.current_node

            # Use current node
            current_node = self.current_node

            # Show session info for existing session
            if not need_new_node:
                session_info = current_node.get_session_info()
                if session_info.get("session_id"):
                    session_display = (
                        f"[dim]Using existing session: {session_info['session_id']} "
                        f"(tokens: {session_info['token_count']}, actions: {session_info['action_count']})[/]"
                    )
                    self.console.print(session_display)

            # Create input using shared method
            node_input, node_type = self.create_node_input(
                message, current_node, at_tables, at_metrics, at_sqls, plan_mode
            )

            # Set input on the node (new interface: input is accessed from self.input)
            current_node.input = node_input

            # Display streaming execution
            self.console.print(f"[bold green]Processing {node_type} request...[/]")

            # Initialize action history display for incremental actions only
            action_display = ActionHistoryDisplay(self.console)
            incremental_actions = []

            # Run streaming execution with real-time display
            # Create a live display like the !reason command (shows only new actions)
            # Skip live display in plan mode to avoid conflicts
            if not plan_mode:
                with action_display.display_streaming_actions(incremental_actions):
                    # Run the async streaming method
                    async def run_chat_stream():
                        async for action in current_node.execute_stream(action_history_manager=self.cli.actions):
                            incremental_actions.append(action)

                    # Execute the streaming chat
                    asyncio.run(run_chat_stream())
            else:
                # In plan mode, run without live display to avoid conflicts with plan hooks
                async def run_chat_stream():
                    async for action in current_node.execute_stream(action_history_manager=self.cli.actions):
                        incremental_actions.append(action)
                        # No delay needed in plan mode

                # Execute the streaming chat
                asyncio.run(run_chat_stream())

            # Display final response from the last successful action
            if incremental_actions:
                final_action = incremental_actions[-1]

                if (
                    final_action.output
                    and isinstance(final_action.output, dict)
                    and final_action.status == ActionStatus.SUCCESS
                ):
                    # Parse response to extract clean SQL and output
                    sql = None
                    clean_output = None

                    # First check if SQL and response are directly available
                    sql = final_action.output.get("sql")
                    response = final_action.output.get("response")

                    # Try to extract SQL and output from the string response
                    extracted_sql, extracted_output = self._extract_sql_and_output_from_content(response)
                    sql = sql or extracted_sql

                    # Determine clean_output based on sql and extracted_output
                    clean_output = None

                    if sql:
                        # Has SQL: use extracted_output or fallback to response
                        clean_output = extracted_output or response
                        self.add_in_sql_context(sql, clean_output, incremental_actions)
                    elif isinstance(extracted_output, dict):
                        # No SQL, extracted_output is dict: get raw_output from dict
                        clean_output = extracted_output.get("raw_output", str(extracted_output))
                    else:
                        # No SQL, no extracted_output: try to parse raw_output from response string
                        try:
                            import ast

                            response_dict = ast.literal_eval(response)
                            clean_output = (
                                response_dict.get("raw_output", response)
                                if isinstance(response_dict, dict)
                                else response
                            )
                        except (ValueError, SyntaxError):
                            clean_output = response

                    # Display using simple, focused methods
                    if sql:
                        self._display_sql_with_copy(sql)

                    # Check for semantic_model field (from SemanticAgenticNode)
                    semantic_model = final_action.output.get("semantic_model")
                    if semantic_model:
                        self._display_semantic_model(semantic_model)

                    # Check for sql_summary_file field (from SqlSummaryAgenticNode)
                    sql_summary_file = final_action.output.get("sql_summary_file")
                    if sql_summary_file:
                        self._display_sql_summary_file(sql_summary_file)

                    if clean_output:
                        self._display_markdown_response(clean_output)
                    self.last_actions = incremental_actions
                    self.cli.console.print("[bold bright_black]Use `Ctrl+O` to display trace details.[/]")

            # Update chat history for potential context in future interactions
            self.chat_history.append(
                {
                    "user": message,
                    "response": (
                        incremental_actions[-1].output.get("response", "")
                        if incremental_actions and incremental_actions[-1].output
                        else ""
                    ),
                    "actions": len(incremental_actions),
                }
            )

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

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

        Args:
            response: Clean response text to display as markdown
        """
        try:
            # Display as markdown with proper formatting
            markdown_content = Markdown(response)
            self.console.print()  # Add spacing
            self.console.print(markdown_content)

        except Exception as e:
            logger.error(f"Error displaying markdown: {e}")
            # Fallback to plain text display
            self.console.print(f"\n[bold blue]Assistant:[/] {response}")

    def _display_semantic_model(self, semantic_model: str):
        """
        Display semantic model file path.

        Args:
            semantic_model: Semantic model file path
        """
        try:
            self.console.print()
            self.console.print(f"[bold magenta]Semantic Model File:[/] [cyan]{semantic_model}[/]")

        except Exception as e:
            logger.error(f"Error displaying semantic model: {e}")
            # Fallback to simple display
            self.console.print(f"\n[bold magenta]Semantic Model File:[/] {semantic_model}")

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

    def cmd_chat_info(self, args: str):
        """Display information about the current session."""
        if self.current_node:
            session_info = self.current_node.get_session_info()
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
                        self.console.print(f"  {i+1}. User: {chat['user'][:50]}...")
                        self.console.print(f"     Actions: {chat['actions']}")
            else:
                self.console.print("[yellow]No active session.[/]")
        else:
            self.console.print("[yellow]No active session.[/]")

    def cmd_compact(self, args: str):
        """Manually compact the current session by summarizing conversation history."""
        if not self.current_node:
            self.console.print("[yellow]No active session to compact.[/]")
            return

        session_info = self.current_node.get_session_info()
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

            session_manager = SessionManager()
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
            sessions_with_info.sort(key=lambda x: x.get("last_updated", x.get("created_at", "")), reverse=True)

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
            sql_query=sql, sql_error=error, sql_return=sql_return, row_count=row_count, explanation=explanation
        )
        self.cli.cli_context.add_sql_context(sql_context)
