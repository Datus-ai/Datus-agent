"""
Independent chat executor for handling real-time streaming chat sessions with mode switching support.

This module provides the ChatExecutor class that manages:
- Real-time streaming interaction with ChatAgenticNode
- Dynamic mode switching between Rich and Screen display modes
- Concurrent execution of business logic and display logic
- Thread-safe state management and resource cleanup
"""

import asyncio
import logging
import threading
from typing import List, Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.cli.action_history_display import ActionHistoryDisplay
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.utils.async_utils import force_cleanup_event_loop, is_event_loop_running, run_async

logger = logging.getLogger(__name__)


class ChatExecutor:
    """Independent chat executor that manages streaming chat sessions and mode switching."""

    def __init__(
        self, history_manager: ActionHistoryManager, chat_node: ChatAgenticNode, console: Optional[Console] = None
    ):
        self.history_manager = history_manager
        self.chat_node: ChatAgenticNode = chat_node
        self.console = console or Console()

        # Chat session state
        self.incremental_actions: List[ActionHistory] = []  # Actions for both modes

        # Execution state
        self.streaming_completed = False
        self.final_results_processed = False
        self.mode_switch_enabled = False  # Controls whether Ctrl+D is enabled
        self.is_agent_running = False

        # Thread safety
        self.execution_lock = threading.Lock()

    def reset_session(self):
        """Reset the chat session state for a new conversation."""
        with self.execution_lock:
            self.incremental_actions.clear()
            self.streaming_completed = False
            self.final_results_processed = False
            self.is_agent_running = False
        self._disable_mode_switch()

    def _enable_mode_switch(self):
        """Enable Ctrl+D mode switching after completion."""
        with self.execution_lock:
            self.mode_switch_enabled = True

    def _disable_mode_switch(self):
        """Disable Ctrl+D mode switching during execution."""
        with self.execution_lock:
            self.mode_switch_enabled = False

    def is_mode_switch_available(self) -> bool:
        """Check if mode switching is currently available."""
        return self.mode_switch_enabled and self.streaming_completed and self.final_results_processed

    def _add_action(self, action: ActionHistory):
        """Add action to the incremental actions list."""
        with self.execution_lock:
            self.incremental_actions.append(action)

    def _display_headless_actions(self, action_display: ActionHistoryDisplay):
        """Display actions in headless mode (when event loop is already running)."""
        try:
            self.console.print("\n[bold blue]=== Screen Mode (Headless) ===[/bold blue]")

            # Display all actions using the action display
            action_display.display_action_list(self.incremental_actions)

            self.console.print("\n[dim]Press any key to continue...[/dim]")

        except Exception as e:
            logger.error(f"Headless display error: {e}")
            self.console.print(f"[red]Error displaying actions: {e}[/red]")

    async def _execute_chat_stream(self, chat_input: ChatNodeInput) -> bool:
        """Execute chat stream interaction with ChatAgenticNode (pure business logic)."""
        try:
            if self.chat_node is None:
                raise RuntimeError("Chat node is not available")

            self.is_agent_running = True
            logger.info(f"Starting chat stream for message: {chat_input.user_message[:50]}...")

            async for action in self.chat_node.execute_stream(chat_input, self.history_manager):
                # Use safe method to add action and update queue if in screen mode
                self._add_action(action)

                # Small delay for streaming effect
                await asyncio.sleep(0.2)

            # Stream completed normally
            self.streaming_completed = True
            logger.info(f"Chat stream completed successfully with {len(self.incremental_actions)} actions")
            return True

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            self.streaming_completed = True

            # Add error action using safe method
            error_action = ActionHistory.create_action(
                role=ActionRole.SYSTEM,
                action_type="stream_error",
                messages=f"Stream error: {str(e)}",
                status=ActionStatus.FAILED,
                input_data={"error": str(e)},
            )
            self._add_action(error_action)
            return False

        finally:
            self.is_agent_running = False

    def execute_chat(self, chat_input: ChatNodeInput) -> Tuple[List[ActionHistory], str]:
        """
        Execute a chat command with real-time streaming Rich display.
        After streaming completion, supports Ctrl+D to switch to App display.

        Returns:
            bool: True if execution was successful, False otherwise
        """
        try:
            # Reset session for new chat and disable Ctrl+D during execution
            self.reset_session()

            # Use non-truncated display for better viewing
            self.action_display = ActionHistoryDisplay(console=self.console, enable_truncation=True)

            # Execute with real-time streaming Rich display
            self._execute_with_realtime_streaming_display(chat_input)
            sql = self._display_final_result()
            return (self.incremental_actions, sql)

        except Exception as e:
            logger.error(f"Chat error: {e}")
            self.console.print(f"[bold red]Error:[/] Failed to process chat request: {str(e)}")

            # Add error action to history
            error_action = ActionHistory.create_action(
                role=ActionRole.USER,
                action_type="chat_error",
                messages=f"Chat command failed: {str(e)}",
                input_data={"message": chat_input.user_message},
                status=ActionStatus.FAILED,
            )
            return ([error_action], "")

    def _execute_with_realtime_streaming_display(self, chat_input: ChatNodeInput) -> bool:
        """
        Execute chat with real-time streaming Rich display.
        Background streaming task runs concurrently with Rich Live display.
        After completion, enable Ctrl+D for App display switching.
        """
        success = False
        # Create streaming display context with Live display
        with self.action_display.display_streaming_actions(actions=self.incremental_actions):
            # Execute background streaming task concurrently with Live display
            success = run_async(self._execute_chat_stream(chat_input))

        # After streaming is complete, show completion message
        if success:
            # Mark final results as processed
            self.final_results_processed = True

            # Enable mode switching after completion
            self._enable_mode_switch()

            # Show mode switch instruction
            # self.console.print("\n[yellow]💡 Press Ctrl+D to switch to App display mode[/yellow]")
        else:
            self.console.print("\n[red]❌ Chat execution failed[/red]")

        return success

    def switch_to_app_display(self) -> bool:
        """
        Switch to App display mode if available.
        This method can be called by external components (like CLI) when Ctrl+D is pressed.
        """
        # Debug state information
        logger.debug(
            f"Mode switch check - streaming_completed: {self.streaming_completed}, "
            f"final_results_processed: {self.final_results_processed}, "
            f"mode_switch_enabled: {self.mode_switch_enabled}"
        )

        if not self.is_mode_switch_available():
            # Provide detailed feedback about why mode switching is not available
            status_details = []
            if not self.streaming_completed:
                status_details.append("streaming not completed")
            if not self.final_results_processed:
                status_details.append("results not processed")
            if not self.mode_switch_enabled:
                status_details.append("mode switch not enabled")

            details = ", ".join(status_details) if status_details else "unknown reason"
            self.console.print(f"[yellow]Mode switching is not available at this time ({details})[/yellow]")
            return False

        try:
            # Execute screen display with existing data
            return self._execute_screen_with_existing_stream()

        except Exception as e:
            logger.error(f"Error switching to app display: {e}")
            self.console.print(f"[red]Failed to switch to app display: {e}[/red]")
            return False

    def _execute_screen_with_existing_stream(self):
        """
        Execute Screen display with existing stream data.
        Shows current data from the single background streaming task.
        """
        try:
            from datus.cli.screen.action_display_app import ChatApp

            # Create ChatApp instance with incremental_actions
            app = ChatApp(self.incremental_actions)

            force_cleanup_event_loop()

            # Run the app with proper async handling
            if is_event_loop_running():
                # We're already in an async context, run in headless mode
                self.console.print(
                    (f"[dim]Running in headless mode due to existing event loop[/dim], {asyncio.get_running_loop()}")
                )
                # For headless mode, just display the actions and return immediately
                self._display_headless_actions(self.action_display)
            else:
                # No running event loop, can run normally
                app.run()

            return True

        except Exception as e:
            logger.error(f"Screen display error: {e}")
            self.console.print(f"[yellow]Screen mode failed: {str(e)}.[/]")
            return False

    def _display_final_result(self) -> str:
        # Display final response from the last successful action
        final_sql = ""
        if self.incremental_actions:
            final_action = self.incremental_actions[-1]

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

                # If response contains debug format, extract from it
                if isinstance(response, dict) and "raw_output" in response:
                    extracted_sql, extracted_output = _extract_sql_and_output_from_content(response["raw_output"])
                    sql = sql or extracted_sql  # Use extracted if not already available
                    clean_output = extracted_output
                elif isinstance(response, str):
                    clean_output = response

                # If we still don't have clean output, check other actions for content
                if not clean_output:
                    for action in reversed(self.incremental_actions):
                        if action.status == ActionStatus.SUCCESS and action.output and isinstance(action.output, dict):
                            content = action.output.get("content")
                            if content:
                                extracted_sql, extracted_output = _extract_sql_and_output_from_content(content)
                                sql = sql or extracted_sql
                                clean_output = extracted_output or content
                                break

                # Display using simple, focused methods
                if sql:
                    self._display_sql_with_copy(sql)
                    final_sql = sql

                if clean_output:
                    self._display_markdown_response(clean_output)
        return final_sql

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

    def _display_sql_with_copy(self, sql: str):
        """
        Display SQL in a formatted panel with automatic clipboard copy functionality.

        Args:
            sql: SQL query string to display and copy
        """
        try:
            # Try to copy to clipboard
            copied_indicator = ""
            try:
                # Try pyperclip first
                import pyperclip

                pyperclip.copy(sql)
                copied_indicator = " (copied)"
            except Exception as e:
                logger.debug(f"Failed to copy SQL to clipboard: {e}")
                # If clipboard fails, don't show the indicator

            # Display SQL in a beautiful syntax-highlighted panel
            sql_syntax = Syntax(sql, "sql", theme="default", line_numbers=False)
            sql_panel = Panel(
                sql_syntax,
                title=f"Generated SQL{copied_indicator}",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            )

            self.console.print()  # Add spacing
            self.console.print(sql_panel)

        except Exception as e:
            logger.error(f"Error displaying SQL: {e}")
            # Fallback to simple display
            self.console.print(f"\n[bold cyan]Generated SQL:[/]\n```sql\n{sql}\n```")


def _extract_sql_and_output_from_content(content: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract SQL and output from content string that might contain JSON or debug format.

    Args:
        content: Content string to parse

    Returns:
        Tuple of (sql_string, output_string) - both can be None if not found
    """
    try:
        import json
        import re

        # Try to extract JSON from various patterns
        # Pattern 1: json\n{...} format
        json_match = re.search(r"json\s*\n\s*({.*?})\s*$", content, re.DOTALL)
        if json_match:
            try:
                json_content = json.loads(json_match.group(1))
                sql = json_content.get("sql")
                output = json_content.get("output")
                if output:
                    output = output.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
                return sql, output
            except json.JSONDecodeError:
                pass

        # Pattern 2: Direct JSON in content
        try:
            json_content = json.loads(content)
            sql = json_content.get("sql")
            output = json_content.get("output")
            if output:
                output = output.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
            return sql, output
        except json.JSONDecodeError:
            pass

        # Pattern 3: Look for SQL code blocks
        sql_pattern = r"```sql\s*(.*?)\s*```"
        sql_matches = re.findall(sql_pattern, content, re.DOTALL | re.IGNORECASE)
        sql = sql_matches[0].strip() if sql_matches else None

        return sql, None

    except Exception as e:
        logger.warning(f"Failed to extract SQL and output from content: {e}")
        return None, None
