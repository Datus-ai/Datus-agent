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
from typing import List, Optional, Tuple

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.cli.action_history_display import ActionHistoryDisplay
from datus.schemas.action_history import ActionHistory, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ChatCommands:
    """Handles all chat-related commands and functionality."""

    def __init__(self, cli_instance):
        """Initialize with reference to the CLI instance for shared resources."""
        self.cli = cli_instance
        self.console = cli_instance.console

        # Chat state management
        self.chat_node: ChatAgenticNode | None = None
        self.chat_history = []

    def update_chat_node_tools(self):
        """Update chat node tools when namespace changes."""
        if self.chat_node:
            self.chat_node.setup_tools()

    def execute_chat_command(self, message: str):
        """Execute a chat command (/ prefix) using ChatAgenticNode."""
        if not message.strip():
            self.console.print("[yellow]Please provide a message to chat with the AI.[/]")
            return

        try:
            # Create chat input with current database context
            chat_input = ChatNodeInput(
                user_message=message,
                catalog=self.cli.cli_context.current_catalog if self.cli.cli_context.current_catalog else None,
                database=self.cli.cli_context.current_db_name if self.cli.cli_context.current_db_name else None,
                db_schema=self.cli.cli_context.current_schema if self.cli.cli_context.current_schema else None,
            )

            # Get or create persistent ChatAgenticNode
            if self.chat_node is None:
                self.console.print("[dim]Creating new chat session...[/]")
                self.chat_node = ChatAgenticNode(
                    namespace=self.cli.agent_config.current_namespace,
                    agent_config=self.cli.agent_config,
                )
            else:
                # Show session info for existing session
                session_info = self.chat_node.get_session_info()
                if session_info["session_id"]:
                    session_display = (
                        f"[dim]Using existing session: {session_info['session_id']} "
                        f"(tokens: {session_info['token_count']}, actions: {session_info['action_count']})[/]"
                    )
                    self.console.print(session_display)

            # Display streaming execution
            self.console.print("[bold green]Processing chat request...[/]")

            # Initialize action history display for incremental actions only
            action_display = ActionHistoryDisplay(self.console)
            incremental_actions = []

            # Run streaming execution with real-time display
            # Create a live display like the !reason command (shows only new actions)
            with action_display.display_streaming_actions(incremental_actions):
                # Run the async streaming method
                async def run_chat_stream():
                    async for action in self.chat_node.execute_stream(chat_input, self.cli.actions):
                        incremental_actions.append(action)
                        # Add delay to make the streaming visible
                        await asyncio.sleep(0.5)

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

                    logger.debug(f"DEBUG: final_action.output: {final_action.output}")

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

                    if clean_output:
                        self._display_markdown_response(clean_output)

                    self._show_detail(incremental_actions)

            # Add all actions from chat to our main action history
            self.cli.actions.actions.extend(incremental_actions)

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

    def _show_detail(self, actions: List[ActionHistory]):
        """Show detailed action information with user confirmation."""
        while True:
            choice = self.cli._prompt_input(
                "Would you like to check the details?",
                choices=["y", "n"],
                default="y",
            )
            # modify the node input
            if choice == "y":
                from datus.cli.screen.action_display_app import ChatApp

                app = ChatApp(actions)
                app.run()
                break
            else:
                return

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
                logger.debug(f"DEBUG: Successfully parsed JSON: {json_content}")
                sql = json_content.get("sql")
                output = json_content.get("output") or json_content.get("raw_output")
                logger.debug(f"DEBUG: Extracted sql={sql}, output={output} (type: {type(output)})")
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
        """Clear the console screen and chat session."""
        # Clear the console screen using Rich
        self.console.clear()

        # Clear chat session
        if self.chat_node:
            self.chat_node.delete_session()
            self.console.print("[green]Console and chat session cleared.[/]")
        else:
            self.console.print("[green]Console cleared. Next chat will create a new session.[/]")
        self.chat_node = None

    def cmd_chat_info(self, args: str):
        """Display information about the current chat session."""
        if self.chat_node:
            session_info = self.chat_node.get_session_info()
            if session_info["session_id"]:
                self.console.print("[bold green]Chat Session Info:[/]")
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
                self.console.print("[yellow]No active chat session.[/]")
        else:
            self.console.print("[yellow]No active chat session.[/]")

    def cmd_compact(self, args: str):
        """Manually compact the chat session by summarizing conversation history."""
        if not self.chat_node:
            self.console.print("[yellow]No active chat session to compact.[/]")
            return

        session_info = self.chat_node.get_session_info()
        if not session_info["session_id"]:
            self.console.print("[yellow]No active chat session to compact.[/]")
            return

        try:
            # Display session info before compacting
            self.console.print("[bold blue]Compacting Chat Session...[/]")
            self.console.print(f"  Current Session ID: {session_info['session_id']}")
            self.console.print(f"  Current Token Count: {session_info['token_count']}")
            self.console.print(f"  Current Action Count: {session_info['action_count']}")

            # Call the manual compact method asynchronously
            async def run_compact():
                return await self.chat_node._manual_compact()

            # Run the compact operation
            result = asyncio.run(run_compact())

            if result["success"]:
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

            # Get current session ID for highlighting (if chat_node exists)
            current_session_id = None
            if self.chat_node and hasattr(self.chat_node, "session_id"):
                current_session_id = self.chat_node.session_id

            # Get session info for all sessions first to enable sorting
            sessions_with_info = []
            for session_data in sessions:
                session_id = session_data["session_id"]
                try:
                    # Get detailed session info if available
                    if self.chat_node and hasattr(self.chat_node, "_get_session_details"):
                        detailed_info = self.chat_node._get_session_details(session_id)
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
