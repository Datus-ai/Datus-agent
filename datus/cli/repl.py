"""
Datus-CLI REPL (Read-Eval-Print Loop) implementation.
This module provides the main interactive shell for the CLI.
"""

import sys
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers.sql import SqlLexer
from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.cli.action_history_display import ActionHistoryDisplay
from datus.cli.agent_commands import AgentCommands
from datus.cli.autocomplete import SQLCompleter
from datus.cli.context_commands import ContextCommands
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.utils.constants import DBType
from datus.utils.exceptions import setup_exception_handler
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CommandType(Enum):
    """Type of command entered by the user."""

    SQL = "sql"  # Regular SQL
    TOOL = "tool"  # !command (tool/workflow)
    CONTEXT = "context"  # @command (context explorer)
    CHAT = "chat"  # /command (chat)
    INTERNAL = "internal"  # .command (CLI control)
    EXIT = "exit"  # exit/quit command


class DatusCLI:
    """Main REPL for the Datus CLI application."""

    def __init__(self, args):
        """Initialize the CLI with the given arguments."""
        self.args = args
        self.console = Console()
        self.console_column_width = 16
        self.selected_catalog_path = ""
        self.selected_catalog_data = {}

        setup_exception_handler(
            console_logger=self.console.print, prefix_wrap_func=lambda x: f"[bold red]{x}[/bold red]"
        )
        self.db_connector = None

        self.agent = None
        self.agent_initializing = False
        self.agent_ready = False

        # Setup history
        history_file = Path(args.history_file)
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(history_file))

        # Setup prompt session
        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            lexer=PygmentsLexer(SqlLexer),
            completer=SQLCompleter(),
            multiline=False,
            key_bindings=KeyBindings(),
            enable_history_search=True,
            search_ignore_case=True,
            style=Style.from_dict(
                {
                    "prompt": "ansigreen bold",
                }
            ),
        )

        self.agent_config = load_agent_config(**vars(self.args))

        # Initialize agent commands handler
        self.agent_commands = AgentCommands(self)

        # Initialize context commands handler
        self.context_commands = ContextCommands(self)

        # Dictionary of available commands
        self.commands = {
            "!darun": self.agent_commands.cmd_darun,
            "!darun_screen": self.agent_commands.cmd_darun_screen,
            "!dastart": self.agent_commands.cmd_dastart,
            "!sl": self.agent_commands.cmd_sl,
            "!gen": self.agent_commands.cmd_gen,
            "!run": self.agent_commands.cmd_run,
            "!fix": self.agent_commands.cmd_fix,
            "!daend": self.agent_commands.cmd_daend,
            # "!rf": self.agent_commands.cmd_reflect,
            "!compare": self.agent_commands.cmd_compare_stream,
            # "!compare_stream": self.agent_commands.cmd_compare_stream,
            "!reason": self.agent_commands.cmd_reason_stream,
            # "!reason_stream": self.agent_commands.cmd_reason_stream,
            "!gen_metrics": self.agent_commands.cmd_gen_metrics_stream,
            # "!gen_metrics_stream": self.agent_commands.cmd_gen_metrics_stream,
            "!gen_semantic_model": self.agent_commands.cmd_gen_semantic_model_stream,
            # "!gen_semantic_model_stream": self.agent_commands.cmd_gen_semantic_model_stream,
            "!set": self.agent_commands.cmd_set_context,
            "!save": self.agent_commands.cmd_save,
            "!bash": self._cmd_bash,
            "@catalogs": self.context_commands.cmd_catalogs,
            "@tables": self.context_commands.cmd_tables,
            "@metrics": self.context_commands.cmd_metrics,
            "@context": self.context_commands.cmd_context,
            "@screen": self.context_commands.cmd_context_screen,
            ".help": self._cmd_help,
            ".exit": self._cmd_exit,
            ".quit": self._cmd_exit,
            ".clear": self._cmd_clear_chat,
            ".chat_info": self._cmd_chat_info,
            # temporary commands for sqlite, remove after mcp server is ready
            ".databases": self._cmd_list_databases,
            ".database": self._cmd_switch_database,
            ".tables": self._cmd_tables,
            ".schemas": self._cmd_schemas,
            ".schema": self._cmd_switch_schema,
            ".table_schema": self._cmd_table_schema,
            ".show": self._cmd_show,
            ".namespace": self._cmd_switch_namespace,
            ".mcp": self._cmd_mcp,
        }

        # Last executed SQL and result
        self.last_sql = None
        self.last_result = None
        self.chat_history = []

        # Action history manager for tracking all CLI operations
        self.actions = ActionHistoryManager()

        # Persistent chat node for session continuity
        self.chat_node: ChatAgenticNode | None = None

        self.current_db_name = getattr(args, "database", "")
        self.current_catalog = getattr(args, "catalog", "")
        self.current_schema = getattr(args, "schema", "")
        self.db_manager = db_manager_instance(self.agent_config.namespaces)

        # Start agent initialization in background
        self._async_init_agent()
        self._init_connection()

    def run(self):
        """Run the REPL loop."""
        self._print_welcome()

        while True:
            try:
                # Check if we have a selected catalog path to inject
                prompt_text = "Datus-sql> "
                # TODO use selected_catalog_path
                if self.selected_catalog_path:
                    # prompt_text = f"Datus-sql> {self.selected_catalog_path}"
                    selected_path = self.selected_catalog_path
                    self.console.print(f"Selected catalog: {selected_path}")
                    self.selected_catalog_path = None

                # Get user input
                user_input = self.session.prompt(
                    message=prompt_text,
                ).strip()
                if not user_input:
                    continue

                # Parse and execute the command
                cmd_type, cmd, args = self._parse_command(user_input)
                if cmd_type == CommandType.EXIT:
                    return True

                # Execute the command based on type
                if cmd_type == CommandType.SQL:
                    self._execute_sql(user_input)
                elif cmd_type == CommandType.TOOL:
                    self._execute_tool_command(cmd, args)
                elif cmd_type == CommandType.CONTEXT:
                    self._execute_context_command(cmd, args)
                elif cmd_type == CommandType.CHAT:
                    self._execute_chat_command(args)
                elif cmd_type == CommandType.INTERNAL:
                    self._execute_internal_command(cmd, args)

            except KeyboardInterrupt:
                continue
            except EOFError:
                return 0
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _process_command(self, user_input: str) -> bool:
        # Parse and execute the command
        cmd_type, cmd, args = self._parse_command(user_input)
        if cmd_type == CommandType.EXIT:
            return True

        # Execute the command based on type
        if cmd_type == CommandType.SQL:
            self._execute_sql(user_input)
        elif cmd_type == CommandType.TOOL:
            self._execute_tool_command(cmd, args)
        elif cmd_type == CommandType.CONTEXT:
            self._execute_context_command(cmd, args)
        elif cmd_type == CommandType.CHAT:
            self._execute_chat_command(args)
        elif cmd_type == CommandType.INTERNAL:
            self._execute_internal_command(cmd, args)

        return False

    def _async_init_agent(self):
        """Initialize the agent asynchronously in a background thread."""
        if self.agent_initializing or self.agent_ready:
            return

        self.agent_initializing = True
        self.console.print("[dim]Initializing AI capabilities in background...[/]")

        # Start initialization in a separate thread
        thread = threading.Thread(target=self._background_init_agent)
        thread.daemon = True  # Daemon thread will exit when main thread exits
        thread.start()

    def _background_init_agent(self):
        """Background thread function to initialize the agent."""
        try:
            # Create a mock args object based on CLI args
            from argparse import Namespace

            agent_args = Namespace(
                temperature=0.7,
                top_p=0.9,
                max_tokens=8000,
                plan="reflection",
                max_steps=20,
                debug=self.args.debug,
                load_cp=False,
                components=["metrics", "metadata", "table_lineage", "document"],
            )

            from datus.agent.agent import Agent

            self.agent = Agent(agent_args, self.agent_config)

            self.agent_ready = True
            self.agent_initializing = False

            self.agent_commands.update_agent_reference()
            # self.console.print("[dim]Agent initialized successfully in background[/]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/]Failed to initialize agent in background: {str(e)}")
            self.agent_initializing = False
            self.agent = None

    def _check_agent_available(self):
        """Check if agent is available, and inform the user if it's still initializing."""
        if self.agent_ready and self.agent:
            return True
        elif self.agent_initializing:
            self.console.print(
                "[yellow]AI features are still initializing in the background. Please try again shortly.[/]"
            )
            return False
        else:
            self.console.print("[bold red]Error:[/] AI features are not available. Agent initialization failed.")
            return False

    def _cmd_list_namespaces(self):
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Namespace")
        for namespace in self.agent_config.namespaces.keys():
            if self.agent_config.current_namespace == namespace:
                table.add_row(f"[bold green]{namespace}[/]")
            else:
                table.add_row(namespace)
        self.console.print(table)
        return

    def _cmd_mcp(self, args):
        from datus.cli.mcp_commands import MCPCommands

        MCPCommands(self).cmd_mcp(args)

    def _smart_display_table(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Smart table display that handles wide tables by limiting columns and truncating content.

        Args:
            data: List of dictionaries representing table rows
            columns: The columns to display, if not provided, all columns will be displayed
        """
        if not data:
            self.console.print("[yellow]No data to display[/]")
            return

        if columns:
            all_columns_list = columns
        else:
            # Get all unique column names
            all_columns_list = []
            for row in data:
                all_columns_list.extend(list(row.keys()))
        # Calculate the maximum number of columns based on the terminal width.
        max_columns = max(4, self.console.width // self.console_column_width)

        # Smart column selection: show front + back + ellipsis based on terminal width
        if len(all_columns_list) > max_columns:
            show_back = max_columns // 2
            show_front = max_columns - show_back  # -1 for ellipsis

            # Select columns to display
            front_columns = all_columns_list[:show_front]
            back_columns = all_columns_list[-show_back:] if show_back > 0 else []
            display_columns = front_columns + ["..."] + back_columns
        else:
            display_columns = all_columns_list

        table = Table(show_header=True, header_style="bold green")

        # Add columns with width constraints
        for col in display_columns:
            if col == "...":
                table.add_column(col, width=5, justify="center")
            else:
                # Calculate optimal column width based on terminal width
                table.add_column(col, width=self.console_column_width)

        # Add rows with truncated content
        for row in data:
            row_values: List[Any] = []
            for col in display_columns:
                if col == "...":
                    row_values.append("...")
                else:
                    row_values.append(str(row.get(col)))
            table.add_row(*row_values)

        self.console.print(table)

    def _cmd_switch_namespace(self, args: str):
        if args.strip() == "":
            self._cmd_list_namespaces()
        elif self.agent_config.current_namespace == args.strip():
            self.console.print(
                (
                    f"[yellow]It's now under the namespace [bold]{self.agent_config.current_namespace}[/]"
                    " and doesn't need to be switched[/]"
                )
            )
            self._cmd_list_namespaces()
            return
        else:
            self.agent_config.current_namespace = args.strip()
            name, self.db_connector = self.db_manager.first_conn_with_name(self.agent_config.current_namespace)
            self.current_catalog = self.db_connector.catalog_name
            self.current_db_name = self.db_connector.database_name if not name else name
            self.current_schema = self.db_connector.schema_name
            if self.chat_node:
                self.chat_node.setup_tools()
            self.console.print(f"[bold green]Namespace changed to: {self.agent_config.current_namespace}[/]")

    def _cmd_switch_database(self, args: str = ""):
        new_db = args.strip()
        if not new_db:
            self.console.print("[bold red]Error:[/] Database name is required")
            self._cmd_list_databases()
            return
        if new_db == self.current_db_name:
            self.console.print(
                f"[yellow]It's now under the database [bold]{new_db}[/] and doesn't need to be switched[/]"
            )
            return

        self.db_connector.switch_context(database_name=new_db)
        self.current_db_name = new_db
        if self.agent_config.db_type == DBType.SQLITE or self.agent_config.db_type == DBType.DUCKDB:
            self.db_connector = self.db_manager.get_conn(self.agent_config.current_namespace, self.current_db_name)
        self.agent_config._current_database = new_db
        if self.chat_node and (
            self.agent_config.db_type == DBType.SQLITE or self.agent_config.db_type == DBType.DUCKDB
        ):
            self.chat_node.setup_tools()
        self.console.print(f"[bold green]Database switched to: {self.current_db_name}[/]")

    def _parse_command(self, text: str) -> Tuple[CommandType, str, str]:
        """
        Parse the command and determine its type.

        Returns:
            Tuple containing (command_type, command, arguments)
        """
        text = text.strip()

        # Remove trailing semicolons (common in SQL)
        if text.endswith(";"):
            text = text[:-1].strip()

        # Exit commands
        if text.lower() in [".exit", ".quit", "exit", "quit"]:
            return CommandType.EXIT, "", ""

        # Tool commands (!prefix)
        if text.startswith("!"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.TOOL, cmd, args

        # Context commands (@prefix)
        if text.startswith("@"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.CONTEXT, cmd, args

        # Chat commands (/prefix)
        if text.startswith("/"):
            return CommandType.CHAT, "", text[1:].strip()

        # Internal commands (.prefix)
        if text.startswith("."):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.INTERNAL, cmd, args

        # Default to SQL
        return CommandType.SQL, "", text

    def _execute_sql(self, sql: str, system: bool = False):
        """Execute a SQL query and display results."""
        logger.debug(f"Executing SQL query: '{sql}'")

        # Create action for SQL execution
        sql_action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="sql_execution",
            messages=f"Executing SQL: {sql[:100]}..." if len(sql) > 100 else f"Executing SQL: {sql}",
            input_data={"sql": sql, "system": system},
            status=ActionStatus.PROCESSING,
        )
        self.actions.add_action(sql_action)

        try:
            if not self.db_connector:
                error_msg = "No database connection. Please initialize a connection first."
                self.console.print(f"[bold red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return

            # Execute the query
            import time

            start_time = time.time()
            result = self.db_connector.execute_arrow(sql)
            end_time = time.time()
            exec_time = end_time - start_time

            if not result:
                error_msg = "No result from the query."
                self.console.print(f"[bold red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return

            # Save for later reference
            self.last_sql = sql
            self.last_result = result

            # Display results and update action
            if result and result.success and hasattr(result.sql_return, "column_names"):
                # Convert Arrow data to list of dictionaries for smart display
                rows = result.sql_return.to_pylist()
                self._smart_display_table(data=rows, columns=result.sql_return.column_names)

                row_count = result.sql_return.num_rows
                self.console.print(f"[dim]Returned {row_count} rows in {exec_time:.2f} seconds[/]")

                # Update action with success
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.SUCCESS,
                    output={
                        "row_count": row_count,
                        "execution_time": exec_time,
                        "columns": result.sql_return.column_names,
                        "success": True,
                    },
                    messages=f"SQL executed successfully: {row_count} rows in {exec_time:.2f}s",
                )

                if not system and self.agent and self.agent.workflow:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.sql_return),
                        row_count=row_count,
                        explanation=f"Manual sql: Returned {row_count} rows in {exec_time:.2f} seconds",
                    )
                    self.agent.workflow.context.sql_contexts.append(new_record)

            elif result and not result.success:
                error_msg = result.error or "Unknown SQL error"
                self.console.print(f"[bold red]SQL Error:[/] {error_msg}")

                # Update action with SQL error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "sql_error": True},
                    messages=f"SQL error: {error_msg}",
                )

                if not system and self.agent and self.agent.workflow:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.error) if result.error else "Unknown error",
                        row_count=0,
                        explanation="Manual sql",
                    )
                    self.agent.workflow.context.sql_contexts.append(new_record)

            elif result and isinstance(result.sql_return, str):
                error_msg = (
                    f"Query execution failed - received string instead of Arrow data: {result.error or 'Unknown error'}"
                )
                self.console.print(f"[bold red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "result_type_error": True},
                    messages=f"Result format error: {error_msg}",
                )
            else:
                error_msg = "No valid result from the query."
                self.console.print(f"[bold red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"No valid result: {error_msg}",
                )

        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

            # Update action with exception
            self.actions.update_action_by_id(
                sql_action.action_id,
                status=ActionStatus.FAILED,
                output={"error": str(e), "exception": True},
                messages=f"SQL execution exception: {str(e)}",
            )

    def _execute_tool_command(self, cmd: str, args: str):
        """Execute a tool command (! prefix)."""
        if cmd in self.commands:
            self.commands[cmd](args)
            # if cmd == "!darun_screen":
            #    asyncio.run(self.commands[cmd](args))
            # else:
            #    self.commands[cmd](args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _execute_context_command(self, cmd: str, args: str):
        """Execute a context command (@ prefix)."""
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _execute_chat_command(self, message: str):
        """Execute a chat command (/ prefix) using ChatAgenticNode."""
        if not message.strip():
            self.console.print("[yellow]Please provide a message to chat with the AI.[/]")
            return

        try:
            # Import here to avoid circular imports
            from datus.schemas.chat_agentic_node_models import ChatNodeInput

            # Create chat input with current database context
            chat_input = ChatNodeInput(
                user_message=message,
                catalog=self.current_catalog if self.current_catalog else None,
                database=self.current_db_name if self.current_db_name else None,
                db_schema=self.current_schema if self.current_schema else None,
            )

            # Get or create persistent ChatAgenticNode
            if self.chat_node is None:
                self.console.print("[dim]Creating new chat session...[/]")
                self.chat_node = ChatAgenticNode(
                    namespace=self.agent_config.current_namespace,
                    agent_config=self.agent_config,
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
            import asyncio

            # Create a live display like the !reason command (shows only new actions)
            with action_display.display_streaming_actions(incremental_actions):
                # Run the async streaming method
                async def run_chat_stream():
                    async for action in self.chat_node.execute_stream(chat_input, self.actions):
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

                    # First check if SQL and response are directly available
                    sql = final_action.output.get("sql")
                    response = final_action.output.get("response")

                    # If response contains debug format, extract from it
                    if isinstance(response, dict) and "raw_output" in response:
                        extracted_sql, extracted_output = self._extract_sql_and_output_from_content(
                            response["raw_output"]
                        )
                        sql = sql or extracted_sql  # Use extracted if not already available
                        clean_output = extracted_output
                    elif isinstance(response, str):
                        clean_output = response

                    # If we still don't have clean output, check other actions for content
                    if not clean_output:
                        for action in reversed(incremental_actions):
                            if (
                                action.status == ActionStatus.SUCCESS
                                and action.output
                                and isinstance(action.output, dict)
                            ):
                                content = action.output.get("content")
                                if content:
                                    extracted_sql, extracted_output = self._extract_sql_and_output_from_content(content)
                                    sql = sql or extracted_sql
                                    clean_output = extracted_output or content
                                    break

                    # Display using simple, focused methods
                    if sql:
                        self._display_sql_with_copy(sql)

                    if clean_output:
                        self._display_markdown_response(clean_output)

            # Add all actions from chat to our main action history
            self.actions.actions.extend(incremental_actions)

            # Update chat history for potential context in future interactions
            self.chat_history.append(
                {
                    "user": message,
                    "response": incremental_actions[-1].output.get("response", "")
                    if incremental_actions and incremental_actions[-1].output
                    else "",
                    "actions": len(incremental_actions),
                }
            )

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to process chat request: {str(e)}")

            # Add error action to history
            error_action = ActionHistory.create_action(
                role=ActionRole.USER,
                action_type="chat_error",
                messages=f"Chat command failed: {str(e)}",
                input_data={"message": message},
                status=ActionStatus.FAILED,
            )
            self.actions.add_action(error_action)

    def _execute_internal_command(self, cmd: str, args: str):
        """Execute an internal command (. prefix)."""
        logger.debug(f"Executing internal command: '{cmd}' with args: '{args}'")
        if cmd in self.commands:
            self.commands[cmd](args)
        elif self.db_connector.get_type() == DBType.SQLITE:
            self._execute_sqlite_internal_command(cmd, args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _execute_sqlite_internal_command(self, cmd: str, args: str):
        """Execute an internal command for SQLite."""
        base_cmd = cmd

        try:
            if base_cmd == ".schema":
                table_name = args
                if table_name:
                    # Execute SQL and check if there was an error
                    sql = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                    result = self.db_connector.execute_arrow(sql)
                else:
                    sql = "SELECT sql FROM sqlite_master WHERE type='table'"
                    result = self.db_connector.execute_arrow(sql)

                # Check if result is None or failed
                if result is None or not result.success:
                    error_msg = result.error if result and hasattr(result, "error") else "Query failed"
                    self.console.print(f"[bold red]Error:[/] {error_msg}")
                    return True

                # Check if sql_return is an Arrow table
                if hasattr(result.sql_return, "to_pylist"):
                    schemas = result.sql_return.to_pylist()
                    if schemas:
                        for schema in schemas:
                            # Handle both tuple/list and dict formats
                            sql_text = None
                            if isinstance(schema, (list, tuple)) and len(schema) > 0:
                                sql_text = schema[0]
                            elif isinstance(schema, dict) and "sql" in schema:
                                sql_text = schema["sql"]
                            elif hasattr(schema, "sql"):
                                sql_text = schema.sql

                            if sql_text:
                                self.console.print(Syntax(sql_text, "sql", theme="default"))
                    else:
                        if table_name:
                            self.console.print(f"[yellow]Table '{table_name}' not found[/]")
                        else:
                            self.console.print("[yellow]No table schemas found[/]")
                else:
                    self.console.print(f"[bold red]Error:[/] Unexpected result format: {type(result.sql_return)}")
                return True
            elif base_cmd == ".show":
                settings = {"database": self.args.db_path, "Python": sys.version.split()[0]}
                settings_table = Table(title="Current Settings")
                settings_table.add_column("Setting")
                settings_table.add_column("Value")
                for k, v in settings.items():
                    settings_table.add_row(k, str(v))
                self.console.print(settings_table)
                return True
            elif base_cmd == ".indexes":
                table_name = args
                if not table_name:
                    self.console.print("[bold red]Error:[/] Table name required")
                    return True

                # Execute SQL directly and check result
                sql = f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'"
                result = self.db_connector.execute_arrow(sql)

                # Check if result is None or failed
                if result is None or not result.success:
                    self.console.print("[bold red]Error:[/] Query failed")
                    return True

                indexes = result.sql_return.to_pylist()
                if indexes:
                    index_table = Table(title=f"Indexes for {table_name}")
                    index_table.add_column("Index Name")
                    for idx in indexes:
                        index_table.add_row(idx[0])
                    self.console.print(index_table)
                else:
                    self.console.print(f"[yellow]Table {table_name} has no indexes[/]")
                return True
            else:
                self.console.print(f"[bold red]未知命令:[/] {cmd}")
                return True
        except Exception as e:
            logger.error(f"Internal command error: {e}", exc_info=True)
            self.console.print(f"[bold red]Command execution error:[/] {e}")
            return True

    def _wait_for_agent_available(self, max_attempts=5, delay=1):
        """Wait for the agent to become available, with timeout."""
        if self._check_agent_available():
            return True

        self.console.print("[yellow]Waiting for the agent to initialize...[/]")

        import time

        for _ in range(max_attempts):
            time.sleep(delay)
            if self._check_agent_available():
                return True

        self.console.print("[bold red]Agent initialization timed out. Try again later.[/]")
        return False

    def _cmd_bash(self, args: str):
        """Execute a bash command."""
        # Define a whitelist of allowed commands
        whitelist = ["pwd", "ls", "cat", "head", "tail", "echo"]

        if not args.strip():
            self.console.print("[yellow]Please provide a bash command.[/]")
            return

        # Parse the command to check against whitelist
        cmd_parts = args.split()
        base_cmd = cmd_parts[0]

        if base_cmd not in whitelist:
            self.console.print(
                f"[bold red]Security:[/] Command '{base_cmd}' not in whitelist. Allowed: {', '.join(whitelist)}"
            )
            return

        try:
            # Execute the command
            import subprocess

            result = subprocess.run(args, shell=True, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                if result.stdout:
                    self.console.print(result.stdout)
            else:
                self.console.print(f"[bold red]Command failed with code {result.returncode}:[/]\n{result.stderr}")

        except subprocess.TimeoutExpired:
            self.console.print("[bold red]Error:[/] Command timed out after 10 seconds.")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _cmd_tables(self, args: str):
        """List all tables in the current database (internal command)."""
        # Reuse functionality from context commands, but with internal command styling
        if not self.db_connector:
            self.console.print("[bold red]Error:[/] No database connection.")
            return

        try:
            # For SQLite, query the sqlite_master table
            result = self.db_connector.get_tables(
                catalog_name=self.current_catalog, database_name=self.current_db_name, schema_name=self.current_schema
            )
            self.last_result = result
            if result:
                # Display results
                table = Table(
                    show_header=True,
                    header_style="bold green",
                )
                # Add columns
                table.add_column("Table Name")
                for row in result:
                    table.add_row(row)
                if self.current_schema:
                    if self.current_db_name:
                        show_name = f"{self.current_db_name}.{self.current_schema}"
                    else:
                        show_name = self.current_schema
                else:
                    show_name = self.current_db_name
                panel = Panel(table, title=f"Tables in Database {show_name}", title_align="left", box=SIMPLE_HEAD)
                self.console.print(panel)
            else:
                # For other database types, execute the appropriate query
                self.console.print("[yellow]Empty set.[/]")

        except Exception as e:
            logger.error(f"Table listing error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _cmd_help(self, args: str):
        """Display help information with aligned command explanations."""
        CMD_WIDTH = 30
        lines = []
        lines.append("[bold green]Datus-CLI Help[/]\n")
        lines.append("[bold]SQL Commands:[/]")
        lines.append(f"    {'<sql>':<{CMD_WIDTH}}Execute SQL query directly\n")

        lines.append("[bold]Tool Commands (! prefix):[/]")
        tool_cmds = [
            ("!darun <query>", "Run a natural language query through the agentic way"),
            ("!darun_screen <query>", "Run a query with live workflow status display"),
            ("!dastart <query>", "Start a new workflow session with manual input"),
            ("!sl", "Schema linking: show list of recommended tables and values"),
            ("!gen", "Generate SQL, optionally with table constraints"),
            ("!run", "Run the last generated SQL"),
            ("!fix <description>", "Fix the last SQL query"),
            ("!reason", "Run the full reasoning node to exploring"),
            ("!reason_stream", "Run SQL reasoning with streaming output"),
            ("!gen_metrics", "Generate metrics from SQL queries and tables"),
            ("!gen_metrics_stream", "Generate metrics with streaming output"),
            ("!gen_semantic_model", "Generate semantic model for data modeling"),
            ("!gen_semantic_model_stream", "Generate semantic model with streaming output"),
            ("!save", "Save the last result to a file"),
            ("!set <context_type>", "Set the context type for the current workflow"),
            ("    context_type: sql, lastsql, schema, schema_values, metrics, task", ""),
            ("!bash <command>", "Execute a bash command (limited to safe commands)"),
            ("!daend", "End the current agent session and save trajectory to file"),
        ]
        for cmd, desc in tool_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        lines.append("")

        lines.append("[bold]Context Commands (@ prefix):[/]")
        context_cmds = [
            ("@catalogs", "Display database catalogs"),
            ("@tables table_name", "Display table details"),
            ("@metrics", "Display metrics"),
            ("@context [type]", "Display the current context in the terminal"),
            ("@screen [type]", "Display the current context in an interactive screen"),
        ]
        for cmd, desc in context_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        lines.append("")

        lines.append("[bold]Chat Commands (/ prefix):[/]")
        chat_cmds = [
            ("/<message>", "Chat with the AI assistant"),
        ]
        for cmd, desc in chat_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        lines.append("")

        lines.append("[bold]Internal Commands (. prefix):[/]")
        internal_cmds = [
            (".help", "Display this help message"),
            (".exit, .quit", "Exit the CLI"),
            (".clear", "Clear console and chat session"),
            (".chat_info", "Show current chat session information"),
            (".databases", "List all databases"),
            (".database database_name", "Switch current database"),
            (".tables", "List all tables"),
            (".schemas table_name", "Show schema information"),
            (".namespace namespace", "Switch current namespace"),
            (".mcp", "Manage MCP (Model Configuration Protocol) servers"),
            ("     .mcp list", "List all MCP servers"),
            (
                "     .mcp add --transport \\[stdio/sse/http] <name> <command> \\[args1 args2 ...]",
                "Add a new MCP server configuration",
            ),
            ("     .mcp remove <name>", "Remove an MCP server configuration"),
            ("     .mcp check <name>", "Check connectivity to an MCP server"),
            ("     .mcp call <server.tool> \\[params]", "Call a tool on an MCP server"),
            ("     .mcp filter", "Manage tool filters for MCP servers"),
            (
                "       .mcp filter set <server> \\[--allowed tool1,tool2] "
                + "\\[--blocked tool3,tool4] \\[--enabled true/false]",
                "Set tool filter",
            ),
            ("       .mcp filter get <server>", "Get current tool filter configuration"),
            ("       .mcp filter remove <server>", "Remove tool filter configuration"),
        ]
        for cmd, desc in internal_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        help_text = "\n".join(lines)
        self.console.print(help_text)

    def _cmd_exit(self, args: str):
        """Exit the CLI."""
        if self.db_connector:
            try:
                # Close the connection
                self.db_connector.close()
            except Exception as e:
                logger.warning(f"Database connection closed failed, reason:{e}")
        sys.exit(0)

    def _cmd_clear_chat(self, args: str):
        """Clear the console screen and chat session."""
        # Clear the console screen using Rich
        self.console.clear()

        # Clear the chat session
        if self.chat_node:
            self.chat_node.delete_session()
            self.console.print("[green]Console and chat session cleared.[/]")
        else:
            self.console.print("[green]Console cleared. Next chat will create a new session.[/]")
        self.chat_node = None

    def _cmd_chat_info(self, args: str):
        """Display information about the current chat session."""
        if self.chat_node:
            session_info = self.chat_node.get_session_info()
            if session_info["session_id"]:
                self.console.print("[bold green]Chat Session Info:[/]")
                self.console.print(f"  Session ID: {session_info['session_id']}")
                self.console.print(f"  Active: {session_info['active']}")
                self.console.print(f"  Token Count: {session_info['token_count']}")
                self.console.print(f"  Action Count: {session_info['action_count']}")
                self.console.print(f"  Context Usage Ratio: {session_info['context_usage_ratio']:.2%}")
                self.console.print(f"  Context Remaining: {session_info['context_remaining']}")
                self.console.print(f"  Context Length: {session_info['context_length']}")
            else:
                self.console.print("[yellow]Chat node exists but no active session.[/]")
        else:
            self.console.print("[yellow]No active chat session.[/]")

    def catalogs_callback(self, selected_path: str = "", selected_data: Optional[Dict[str, Any]] = None):
        if not selected_path:
            return
        self.selected_catalog_path = selected_path
        self.selected_catalog_data = selected_data

    def _cmd_list_databases(self, args: str = ""):
        """List all databases in the current connection."""
        try:
            # For SQLite, this is simply the current database file
            namespace = self.agent_config.current_namespace
            connections = self.db_manager.get_connections(namespace)
            result = []
            show_uri = False
            if isinstance(connections, dict):
                show_uri = True
                for name, conn in connections.items():
                    result.append(
                        {
                            "name": name if name != self.current_db_name else f"[bold green]{name}[/]",
                            "uri": conn.connection_string,
                        }
                    )
            else:
                db_type = connections.dialect
                self.db_connector = connections
                if db_type == DBType.SQLITE:
                    show_uri = True
                    # FIXME use database_name
                    result.append({"name": namespace, "uri": connections.connection_string})
                elif db_type == DBType.DUCKDB:
                    show_uri = True
                    result.append({"name": connections.database_name, "uri": connections.connection_string})
                else:
                    for db_name in connections.get_databases(catalog_name=self.current_catalog):
                        result.append({"name": db_name})

            self.last_result = result

            # Display results
            table = Table(title="Databases", show_header=True, header_style="bold green")
            table.add_column("Database Name")
            if show_uri:
                table.add_column("URI")
                for db_config in result:
                    name = db_config["name"]
                    table.add_row(name if name != self.current_db_name else f"[bold green]{name}[/]", db_config["uri"])
            else:
                for db_config in result:
                    table.add_row(db_config["name"])
            self.console.print(table)

        except Exception as e:
            logger.error(f"Database listing error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _cmd_schemas(self, args: str):
        dialect = self.db_connector.dialect
        if not DBType.support_schema(dialect):
            self.console.print(f"[bold red]The {dialect} database does not support schema[/]")
            return
        result = self.db_connector.get_schemas(catalog_name=self.current_catalog, database_name=self.current_db_name)
        self.last_result = result
        if result:
            # Display results
            table = Table(
                show_header=True,
                header_style="bold green",
            )
            # Add columns
            table.add_column("Schema Name")
            for row in result:
                table.add_row(row)
            if self.current_catalog:
                if self.current_db_name:
                    show_name = f"{self.current_catalog}.{self.current_db_name}"
                else:
                    show_name = self.current_catalog
            else:
                show_name = self.current_db_name
            panel = Panel(table, title=f"Schema in Database {show_name}", title_align="left", box=SIMPLE_HEAD)
            self.console.print(panel)
        else:
            # For other database types, execute the appropriate query
            self.console.print("[yellow]Empty set.[/]")

    def _cmd_switch_schema(self, args: str):
        dialect = self.db_connector.dialect
        if not DBType.support_schema(dialect):
            self.console.print(f"[bold red]The {dialect} database does not support schema[/]")
            return
        schema_name = args.strip()
        if not schema_name:
            self.console.print("[yellow]You need to give the name of the schema you want to switch to[/]")
            return
        self.db_connector.switch_context(
            catalog_name=self.current_catalog, database_name=self.current_db_name, schema_name=schema_name
        )
        self.console.print(f"[bold green]Schema switched to: {self.current_db_name}[/]")
        self.current_schema = schema_name

    def _cmd_table_schema(self, args: str):
        """Show schema information for tables."""
        if not self.db_connector:
            self.console.print("[bold red]Error:[/] No database connection.")
            return

        try:
            if args.strip():
                table_name = args.strip()
                result = self.db_connector.get_schema(
                    catalog_name=self.current_db_name,
                    database_name=self.current_db_name,
                    schema_name=self.current_schema,
                    table_name=table_name,
                )
                self.last_result = result

                # Display schema for the specific table
                schema_table = Table(
                    title=f"Schema for {table_name}",
                    show_header=True,
                    header_style="bold green",
                )
                schema_table.add_column("Column Position")
                schema_table.add_column("Name")
                schema_table.add_column("Type")
                schema_table.add_column("Nullable")
                schema_table.add_column("Default")
                schema_table.add_column("PK")

                for row in result:
                    schema_table.add_row(
                        str(row.get("cid", "")),
                        str(row.get("name", "")),
                        str(row.get("type", "")),
                        str(row.get("nullable", "")),
                        str(row.get("default_value", "")) if row.get("default_value") is not None else "",
                        str(row.get("pk", "")),
                    )

                self.console.print(schema_table)
            else:
                # List all tables with basic schema info
                table_names = self.db_connector.get_tables(
                    catalog_name=self.current_catalog,
                    database_name=self.current_db_name,
                    schema_name=self.current_schema,
                )
                self.last_result = table_names

                # Display list of tables
                self.console.print("[bold green]Available tables:[/]")
                # Display table list
                for idx, table_name in enumerate(table_names):
                    self.console.print(f"{idx + 1}. {table_name}")

                self.console.print("\n[dim]Use .schemas [table_name] to view detailed schema.[/]")

        except Exception as e:
            logger.error(f"Schema listing error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")
            if "result" in locals():
                logger.debug(f"Result object structure: {dir(result)}")
                for key in dir(result):
                    if not key.startswith("_"):
                        try:
                            value = getattr(result, key)
                            logger.debug(f"  {key}: {value}")
                        except Exception as e:
                            logger.debug(f"  {key}: Error accessing - {e}")
                if hasattr(result, "__dict__"):
                    logger.debug(f"Result __dict__: {result.__dict__}")
                logger.debug(f"Result type: {type(result)}")

    def _cmd_show(self, args: str):
        """Show help about available dot-commands."""
        help_text = """
        [bold green]Available Commands:[/]

        [bold].namespace[/]           Switch the current namespace
        [bold].catalog[/]             Switch the current catalog
        [bold].databases[/]           List all databases
        [bold].database[/]            Switch the current database
        [bold].tables[/]              List all tables in the current database
        [bold].schemas[/]             Show schemas in current database
        [bold].schema [schema][/]     Switch schema
        [bold].table_schema [table][/]Show schema information for all tables or a specific table
        [bold].help[/]                Display help information
        [bold].exit, .quit[/]         Exit the CLI
        """
        self.console.print(help_text)
        self.last_result = {"success": True, "message": "Showed available commands"}

    def _print_welcome(self):
        """Print the welcome message."""
        welcome_text = """
[bold green]Datus-CLI[/] - [bold]AI-powered SQL command-line interface[/]
Type '.help' for a list of commands or '.exit' to quit.
"""
        self.console.print(welcome_text)

        namespace = getattr(self.args, "namespace", "")
        # TODO use default namespace if not set
        if namespace:
            self.console.print(f"Namespace [bold green]{namespace}[/] selected")
        else:
            self.console.print("[yellow]Warning: No namespace selected, please use .namespace to select a namespace[/]")
        # Display connection info
        if self.db_connector:
            db_info = f"Connected to [bold green]{self.agent_config.db_type}[/]"
            if self.current_db_name:
                db_info += f" using database [bold]{self.current_db_name}[/]"

            self.console.print(db_info)
            self.console.print("Type SQL statements or use ! @ . commands to interact.")
        else:
            self.console.print("[yellow]Warning: No database connection initialized.[/]")

    def _prompt_input(self, message: str, default: str = "", choices: list = None, multiline: bool = False):
        """
        Unified input method using prompt_toolkit to avoid conflicts with rich.Prompt.ask().

        Args:
            message: The prompt message to display
            default: Default value if user presses Enter without input
            choices: List of valid choices (validates input)
            multiline: Whether to allow multiline input

        Returns:
            User input string or default value
        """
        try:
            from prompt_toolkit import prompt
            from prompt_toolkit.formatted_text import HTML
            from prompt_toolkit.validation import ValidationError, Validator

            # Format the prompt message
            if default:
                prompt_text = f"{message} ({default}): "
            else:
                prompt_text = f"{message}: "

            # Create validator for choices if provided
            validator = None
            if choices:

                class ChoiceValidator(Validator):
                    def validate(self, document):
                        text = document.text.strip()
                        if text and text not in choices:
                            raise ValidationError(message=f"Please choose from: {', '.join(choices)}")

                validator = ChoiceValidator()

                # Add choices to prompt text
                prompt_text = f"{message} ({'/'.join(choices)}): "
                if default:
                    prompt_text = f"{message} ({'/'.join(choices)}) ({default}): "

            # Use the existing session for consistency but create a temporary one for this input
            from prompt_toolkit.history import InMemoryHistory

            result = prompt(
                HTML(f"<ansigreen><b>{prompt_text}</b></ansigreen>"),
                default=default,
                validator=validator,
                multiline=multiline,
                history=InMemoryHistory(),  # Separate history for sub-prompts
                style=self.session.style,  # Use same style as main session
            )

            return result.strip()

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D gracefully
            self.console.print("\n[yellow]Input cancelled[/]")
            return default
        except Exception as e:
            logger.error(f"Input prompt error: {e}")
            self.console.print(f"[bold red]Input error:[/] {str(e)}")
            return default

    def _init_connection(self):
        """Initialize database connection."""
        current_namespace = self.agent_config.current_namespace
        if not self.current_db_name:
            self.current_db_name, self.db_connector = self.db_manager.first_conn_with_name(current_namespace)
        else:
            self.db_connector = self.db_manager.get_conn(current_namespace, self.current_db_name)
        if not self.db_connector:
            self.console.print("[bold red]Error:[/] No database connection.")
            return

        try:
            # Test the connection
            connection_result = self.db_connector.test_connection()
            logger.debug(f"Connection test result: {connection_result}")

        except Exception as e:
            self.console.print(f"[bold red]Connection Error:[/] {str(e)}")
            raise

    def _display_sql_with_copy(self, sql: str):
        """
        Display SQL in a formatted panel with automatic clipboard copy functionality.

        Args:
            sql: SQL query string to display and copy
        """
        try:
            # Store SQL for reference
            self.last_sql = sql

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
                    import platform
                    import subprocess

                    system = platform.system()
                    if system == "Darwin":  # macOS
                        subprocess.run("pbcopy", input=sql.encode(), check=True)
                        copied_indicator = " (copied)"
                    elif system == "Linux":
                        # Try xclip or xsel
                        try:
                            subprocess.run(["xclip", "-selection", "clipboard"], input=sql.encode(), check=True)
                            copied_indicator = " (copied)"
                        except FileNotFoundError:
                            try:
                                subprocess.run(["xsel", "--clipboard", "--input"], input=sql.encode(), check=True)
                                copied_indicator = " (copied)"
                            except FileNotFoundError:
                                pass  # No clipboard tool available
                    elif system == "Windows":
                        subprocess.run("clip", input=sql.encode(), shell=True, check=True)
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

    def _extract_sql_and_output_from_content(self, content: str) -> tuple[Optional[str], Optional[str]]:
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
