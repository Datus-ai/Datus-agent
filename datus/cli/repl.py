# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Datus-CLI REPL (Read-Eval-Print Loop) implementation.
This module provides the main interactive shell for the CLI.
"""

import sys
import threading
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style, merge_styles, style_from_pygments_cls
from rich.console import Console
from rich.table import Table

from datus.agent.workflow_runner import WorkflowRunner
from datus.cli.agent_commands import AgentCommands
from datus.cli.autocomplete import AtReferenceCompleter, CustomPygmentsStyle, CustomSqlLexer, SubagentCompleter
from datus.cli.chat_commands import ChatCommands
from datus.cli.context_commands import ContextCommands
from datus.cli.metadata_commands import MetadataCommands
from datus.cli.sub_agent_commands import SubAgentCommands
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.utils.constants import SYS_SUB_AGENTS, DBType, SQLType
from datus.utils.exceptions import setup_exception_handler
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_sql_type

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
        self.streamlit_mode = False
        self.selected_catalog_data = {}

        setup_exception_handler(
            console_logger=self.console.print, prefix_wrap_func=lambda x: f"[bold red]{x}[/bold red]"
        )
        self.db_connector: BaseSqlConnector

        self.agent = None
        self.agent_initializing = False
        self.agent_ready = False
        self._workflow_runner: WorkflowRunner | None = None

        # Plan mode support
        self.plan_mode_active = False

        # Load agent config first to initialize path_manager with correct home directory
        self.agent_config = load_agent_config(**vars(self.args))
        self.configuration_manager = configuration_manager()

        if args.history_file:
            history_file = Path(args.history_file).expanduser().resolve()
        else:
            from datus.utils.path_manager import get_path_manager

            history_file = get_path_manager().history_file_path()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(history_file))
        self.at_completer: AtReferenceCompleter
        self._init_prompt_session()

        # Last executed SQL and result
        self.last_sql = None
        self.last_result = None

        # Action history manager for tracking all CLI operations
        self.actions = ActionHistoryManager()

        # Initialize CLI context for state management
        from datus.cli.cli_context import CliContext

        self.cli_context = CliContext(
            current_db_name=getattr(args, "database", ""),
            current_catalog=getattr(args, "catalog", ""),
            current_schema=getattr(args, "schema", ""),
        )
        self.db_manager = db_manager_instance(self.agent_config.namespaces)

        # Initialize available subagents from agentic_nodes (excluding 'chat') and include built-in subagents
        self.available_subagents = set(SYS_SUB_AGENTS)
        if hasattr(self.agent_config, "agentic_nodes") and self.agent_config.agentic_nodes:
            self.available_subagents.update(name for name in self.agent_config.agentic_nodes.keys() if name != "chat")

        # Initialize command handlers after cli_context is created
        self.agent_commands = AgentCommands(self, self.cli_context)
        self.chat_commands = ChatCommands(self)
        self.context_commands = ContextCommands(self)
        self.metadata_commands = MetadataCommands(self)
        self.sub_agent_commands = SubAgentCommands(self)

        # Dictionary of available commands - created after handlers are initialized
        self.commands = {
            # "!run": self.agent_commands.cmd_darun_screen,
            "!sl": self.agent_commands.cmd_schema_linking,
            "!schema_linking": self.agent_commands.cmd_schema_linking,
            "!sm": self.agent_commands.cmd_search_metrics,
            "!search_metrics": self.agent_commands.cmd_search_metrics,
            "!sq": self.agent_commands.cmd_search_reference_sql,
            "!search_sql": self.agent_commands.cmd_search_reference_sql,
            # "!doc_search": self.agent_commands.cmd_doc_search,
            # "!gen": self.agent_commands.cmd_gen,
            # "!fix": self.agent_commands.cmd_fix,
            "!save": self.agent_commands.cmd_save,
            "!bash": self._cmd_bash,
            # to be deprecated when sub agent is read
            # "!reason": self.agent_commands.cmd_reason_stream,
            # "!compare": self.agent_commands.cmd_compare_stream,
            # catalog commands
            "@catalog": self.context_commands.cmd_catalog,
            "@subject": self.context_commands.cmd_subject,
            # interal commands
            ".clear": self.chat_commands.cmd_clear_chat,
            ".chat_info": self.chat_commands.cmd_chat_info,
            ".compact": self.chat_commands.cmd_compact,
            ".sessions": self.chat_commands.cmd_list_sessions,
            ".databases": self.metadata_commands.cmd_list_databases,
            ".database": self.metadata_commands.cmd_switch_database,
            ".tables": self.metadata_commands.cmd_tables,
            ".schemas": self.metadata_commands.cmd_schemas,
            ".schema": self.metadata_commands.cmd_switch_schema,
            ".table_schema": self.metadata_commands.cmd_table_schema,
            ".indexes": self.metadata_commands.cmd_indexes,
            ".namespace": self._cmd_switch_namespace,
            ".subagent": self.sub_agent_commands.cmd,
            ".mcp": self._cmd_mcp,
            ".help": self._cmd_help,
            ".exit": self._cmd_exit,
            ".quit": self._cmd_exit,
        }

        # Start agent initialization in background
        self._async_init_agent()
        self._init_connection()

    @property
    def workflow_runner(self) -> WorkflowRunner:
        if not self.check_agent_available():
            raise RuntimeError("Agent not initialized. Cannot create workflow runner.")
        if not self._workflow_runner:
            self._workflow_runner = self.agent.create_workflow_runner()
        return self._workflow_runner

    def _create_custom_key_bindings(self):
        """Create custom key bindings for the REPL."""
        kb = KeyBindings()

        @kb.add("tab")
        def _(event):
            """The Tab key triggers completion only, not navigation."""
            buffer = event.app.current_buffer

            if buffer.complete_state:
                # If the menu is already open, close it.
                buffer.complete_next()
            else:
                # If the menu is incomplete, trigger completion.
                buffer.start_completion(select_first=False)

        @kb.add("s-tab")
        def _(event):
            """Shift+Tab: Toggle Plan Mode on/off"""
            self.plan_mode_active = not self.plan_mode_active

            # Clear current input buffer and force exit current prompt
            buffer = event.app.current_buffer
            buffer.reset()

            # Force the prompt to exit and restart with new prefix
            # This will cause the main loop to regenerate the prompt
            buffer.validation_state = None
            event.app.exit()

            # Show mode change message
            if self.plan_mode_active:
                self.console.print("[bold green]Plan Mode Activated![/]")
                self.console.print("[dim]Enter your planning task and press Enter to generate plan[/]")
            else:
                self.console.print("[yellow]Plan Mode Deactivated[/]")

        @kb.add("enter")
        def _(event):
            """
            Enter key:
                if completion menu is open, apply the highlighted item (if any) or close the menu; otherwise execute.
            """
            buffer = event.app.current_buffer

            if buffer.complete_state:
                # If there is an actively highlighted completion, apply it.
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    # No item highlighted (e.g., select_first=False). Close the menu and proceed as normal Enter.
                    buffer.cancel_completion()
                    buffer.validate_and_handle()
                return

            # Performs normal Enter behavior when there is no completion menu.
            buffer.validate_and_handle()

        @kb.add("c-o")
        def _(event):
            """Show details for display_actions"""
            event.app.exit(result="_open_chat_sql_details")

        return kb

    def _get_prompt_text(self):
        """Get the current prompt text based on mode"""
        if self.plan_mode_active:
            return "[PLAN MODE] Datus> "
        else:
            return "Datus> "

    def _update_prompt(self):
        """Update the prompt display (called when mode changes)"""
        # The prompt will be updated on the next iteration of the main loop
        # This is a limitation of prompt_toolkit's PromptSession
        # For immediate feedback, we could force a redraw, but it's complex

    def _init_prompt_session(self):
        # Setup prompt session with custom key bindings
        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            lexer=PygmentsLexer(CustomSqlLexer),
            completer=self.create_combined_completer(),
            multiline=False,
            key_bindings=self._create_custom_key_bindings(),
            enable_history_search=True,
            search_ignore_case=True,
            style=merge_styles(
                [
                    style_from_pygments_cls(CustomPygmentsStyle),
                    Style.from_dict(
                        {
                            "prompt": "ansigreen bold",
                        }
                    ),
                ]
            ),
            complete_while_typing=True,
        )

    # Create combined completer
    def create_combined_completer(self):
        """Create combined completer: SubagentCompleter + AtReferenceCompleter + SqlCompleter"""
        from datus.cli.autocomplete import SQLCompleter

        sql_completer = SQLCompleter()
        self.at_completer = AtReferenceCompleter(self.agent_config)  # Router completer
        subagent_completer = SubagentCompleter(self.agent_config)  # Subagent completer

        # Use merge_completers to combine completers
        from prompt_toolkit.completion import merge_completers

        return merge_completers(
            [
                subagent_completer,  # Subagent completer (highest priority)
                self.at_completer,  # @ reference completer
                sql_completer,  # SQL keyword completer (lowest priority)
            ]
        )

    def run(self):
        """Run the REPL loop."""
        self._print_welcome()

        while True:
            try:
                # Get dynamic prompt text
                prompt_text = self._get_prompt_text()

                # Get user input
                user_input_raw = self.session.prompt(
                    message=prompt_text,
                )
                if user_input_raw is None:
                    continue
                if user_input_raw == "_open_chat_sql_details":
                    if not self.streamlit_mode and self.chat_commands and self.chat_commands.last_actions:
                        from datus.cli.screen.action_display_app import ChatApp

                        app = ChatApp(self.chat_commands.last_actions)
                        app.run()
                    continue
                user_input = user_input_raw.strip()

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
                    self._execute_chat_command(args, subagent_name=cmd)
                elif cmd_type == CommandType.INTERNAL:
                    self._execute_internal_command(cmd, args)

            except KeyboardInterrupt:
                continue
            except EOFError:
                return 0
            except Exception as e:
                # Check if this is an exit event (for plan mode toggle)
                if "exit" in str(e).lower() and "app" in str(e).lower():
                    # This is expected from shift+tab toggle, continue loop
                    continue
                logger.error(f"Error: {str(e)}")
                self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _async_init_agent(self):
        """Initialize the agent asynchronously in a background thread."""
        if self.agent_initializing or self.agent_ready:
            return

        # Skip background initialization in Streamlit mode to avoid vector DB conflicts
        if hasattr(self, "streamlit_mode") and self.streamlit_mode:
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
                workflow="reflection",
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
            self._pre_load_storage()
            self._workflow_runner = self.agent.create_workflow_runner()
            # self.console.print("[dim]Agent initialized successfully in background[/]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/]Failed to initialize agent in background: {str(e)}")
            logger.error(f"[bold red]Failed to initialize agent in background: {e}")
            self.agent_initializing = False
            self.agent = None

    def _pre_load_storage(self):
        """Preload rag to avoid unnecessary printing"""
        if self.at_completer:
            self.at_completer.reload_data()

    def check_agent_available(self):
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

        # Calculate dynamic column width based on number of columns
        # With folding enabled, we can use narrower columns and fit more on screen
        num_display_columns = len([col for col in display_columns if col != "..."])
        if num_display_columns <= 2:
            # For 1-2 columns, use moderate width (content will fold if needed)
            dynamic_column_width = max(25, self.console.width // max(2, num_display_columns) - 4)
        elif num_display_columns <= 4:
            # For 3-4 columns, use compact width
            dynamic_column_width = max(20, self.console.width // num_display_columns - 3)
        elif num_display_columns <= 8:
            # For 5-8 columns, use narrow width (content will fold if needed)
            dynamic_column_width = max(18, self.console.width // num_display_columns - 2)
        else:
            # For many columns, use the default compact width
            dynamic_column_width = self.console_column_width

        table = Table(show_header=True, header_style="bold green")

        # Add columns with width constraints and folding for overflow
        for col in display_columns:
            if col == "...":
                table.add_column(col, width=5, justify="center")
            else:
                # Use dynamic column width with folding enabled for long content
                table.add_column(col, width=dynamic_column_width, overflow="fold", no_wrap=False)

        # Add rows with truncated content
        for row in data:
            row_values: List[Any] = []
            for col in display_columns:
                if col == "...":
                    row_values.append("...")
                else:
                    row_value = row.get(col)
                    if isinstance(row_value, datetime):
                        row_value = row_value.strftime("%Y-%m-%d %H:%M:%S")
                    elif isinstance(row_value, date):
                        row_value = row_value.strftime("%Y-%m-%d")
                    else:
                        row_value = str(row_value)
                    row_values.append(row_value)
            table.add_row(*row_values)

        self.console.print(table)

    def reset_session(self):
        self.chat_commands.update_chat_node_tools()
        if self.at_completer:
            # Perhaps we should reload the data here.
            self.at_completer.reload_data()

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
            db_name = self.db_connector.database_name
            db_logic_name = name or self.agent_config.current_namespace
            self.cli_context.update_database_context(
                catalog=self.db_connector.catalog_name,
                db_name=db_name,
                schema=self.db_connector.schema_name,
                db_logic_name=db_logic_name,
            )
            self.reset_session()
            self.chat_commands.update_chat_node_tools()
            self.console.print(f"[bold green]Namespace changed to: {self.agent_config.current_namespace}[/]")

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
            message = text[1:].strip()
            parts = message.split(maxsplit=1)
            if len(parts) > 1:
                # Check if first part is a valid subagent
                potential_subagent = parts[0]
                if potential_subagent in self.available_subagents:
                    # Sub-agent syntax: /subagent_name message
                    subagent_name = potential_subagent
                    actual_message = parts[1]
                    return CommandType.CHAT, subagent_name, actual_message
                else:
                    # Regular chat: /message (first part is not a valid subagent)
                    return CommandType.CHAT, "", message
            else:
                # Regular chat: /message
                return CommandType.CHAT, "", message

        # Internal commands (.prefix)
        if text.startswith("."):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.INTERNAL, cmd, args

        # Determine if text is SQL or chat using parse_sql_type
        try:
            # Get current database dialect from agent_config.db_type (set from current namespace)
            dialect = self.agent_config.db_type if self.agent_config.db_type else "snowflake"
            sql_type = parse_sql_type(text, dialect)

            # If parse_sql_type returns a valid SQL type (not UNKNOWN), treat as SQL
            if sql_type != SQLType.UNKNOWN:
                return CommandType.SQL, "", text
            else:
                return CommandType.CHAT, "", text.strip()
        except Exception:
            # If any exception occurs, treat as chat
            return CommandType.CHAT, "", text.strip()

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
            result = self.db_connector.execute(input_params={"sql_query": sql}, result_format="arrow")
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
            if result.success:
                if not hasattr(result.sql_return, "column_names"):
                    if result.row_count is not None and result.row_count > 0:
                        # Update action with success
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.SUCCESS,
                            output={
                                "row_count": result.row_count,
                                "execution_time": exec_time,
                                "success": True,
                            },
                            messages=f"SQL executed successfully: {result.row_count} rows in {exec_time:.2f}s",
                        )
                        self.console.print(f"[dim]Update {result.sql_return} rows in {exec_time:.2f} seconds[/]")
                    elif result.sql_return:
                        self.console.print(f"[dim]SQL execution successful in {exec_time:.2f} seconds[/]")
                        # Update action with success
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.SUCCESS,
                            output={
                                "row_count": 0,
                                "execution_time": exec_time,
                                "success": True,
                            },
                            messages=f"SQL executed successfully in {exec_time:.2f}s",
                        )
                    else:
                        error_msg = (
                            f"Query execution failed - received string instead of Arrow data:"
                            f" {result.error or 'Unknown error'}"
                        )
                        self.console.print(f"[bold red]Error:[/] {error_msg}")

                        # Update action with error
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.FAILED,
                            output={"error": error_msg, "result_type_error": True},
                            messages=f"Result format error: {error_msg}",
                        )
                    return
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
                workflow_ready = self._workflow_runner and self._workflow_runner.workflow_ready
                if not system and workflow_ready:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.sql_return),
                        row_count=row_count,
                        explanation=f"Manual sql: Returned {row_count} rows in {exec_time:.2f} seconds",
                    )
                    self.workflow_runner.workflow.context.sql_contexts.append(new_record)

            else:
                error_msg = result.error or "Unknown SQL error"
                self.console.print(f"[bold red]SQL Error:[/] {error_msg}")

                # Update action with SQL error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "sql_error": True},
                    messages=f"SQL error: {error_msg}",
                )
                workflow_ready = self._workflow_runner and self._workflow_runner.workflow_ready
                if not system and workflow_ready:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.error) if result.error else "Unknown error",
                        row_count=0,
                        explanation="Manual sql",
                    )
                    self._workflow_runner.workflow.context.sql_contexts.append(new_record)
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
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _execute_context_command(self, cmd: str, args: str):
        """Execute a context command (@ prefix)."""
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _execute_chat_command(self, message: str, subagent_name: str = None):
        """Execute a chat command (/ prefix) using ChatAgenticNode."""
        self.chat_commands.execute_chat_command(message, plan_mode=self.plan_mode_active, subagent_name=subagent_name)

    def _execute_internal_command(self, cmd: str, args: str):
        """Execute an internal command (. prefix)."""
        logger.debug(f"Executing internal command: '{cmd}' with args: '{args}'")
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _wait_for_agent_available(self, max_attempts=5, delay=1):
        """Wait for the agent to become available, with timeout."""
        if self.check_agent_available():
            return True

        self.console.print("[yellow]Waiting for the agent to initialize...[/]")

        import time

        for _ in range(max_attempts):
            time.sleep(delay)
            if self.check_agent_available():
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

    def _cmd_help(self, args: str):
        """Display help information with aligned command explanations."""
        CMD_WIDTH = 30
        lines = []
        lines.append("[bold green]Datus-CLI Help[/]\n")
        lines.append("[bold]SQL Commands:[/]")
        lines.append(f"    {'<sql>':<{CMD_WIDTH}}Execute SQL query directly\n")

        lines.append("[bold]Tool Commands (! prefix):[/]")
        tool_cmds = [
            # ("!run <query>", "Run a natural language query with live workflow status display"),
            ("!sl/!schema_linking", "Schema linking: show list of recommended tables and values"),
            ("!sm/!search_metrics", "Use natural language to search for corresponding metrics"),
            ("!sq/!search_sql", "Use natural language to search for reference SQL"),
            # ("!gen", "Generate SQL, optionally with table constraints"),
            # ("!fix <description>", "Fix the last SQL query"),
            ("!save", "Save the last result to a file"),
            ("!bash <command>", "Execute a bash command (limited to safe commands)"),
            # remove this when sub agent is ready
            # ("!reason", "Run SQL reasoning with streaming output"),
            # ("!compare", "Compare SQL results with streaming output"),
        ]
        for cmd, desc in tool_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        lines.append("")

        lines.append("[bold]Context Commands (@ prefix):[/]")
        context_cmds = [
            ("@catalog", "Display database catalog"),
            ("@subject", "Display Semantic Model, Metrics etc."),
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
            (".compact", "Compact chat session by summarizing conversation history"),
            (".sessions", "List all stored SQLite sessions with detailed information"),
            (".databases", "List all databases"),
            (".database database_name", "Switch current database"),
            (".tables", "List all tables"),
            (".schemas", "List all schemas or show detailed schema information"),
            (".schema schema_name", "Switch current schema"),
            (".table_schema table_name", "Show table field details"),
            (".indexes table_name", "Show indexes for a table"),
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

    def catalogs_callback(self, selected_path: str = "", selected_data: Optional[Dict[str, Any]] = None):
        if not selected_path:
            return
        self.selected_catalog_path = selected_path
        self.selected_catalog_data = selected_data

    def _print_welcome(self):
        """Print the welcome message."""
        welcome_text = """
[bold green]Datus[/] - [bold]AI-powered SQL command-line interface[/]
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
            if self.cli_context.current_db_name:
                db_info += f" using database [bold]{self.cli_context.current_db_name}[/]"

            self.console.print(db_info)

            # Show CLI context summary
            context_summary = self.cli_context.get_context_summary()
            if context_summary != "No context available":
                self.console.print(f"[dim]Context: {context_summary}[/]")

            self.console.print("Type SQL statements or use ! @ . commands to interact.")
        else:
            self.console.print("[yellow]Warning: No database connection initialized.[/]")

    def prompt_input(self, message: str, default: str = "", choices: list = None, multiline: bool = False):
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
                # if default:
                #     prompt_text = f"{message} ({'/'.join(choices)}) ({default}): "

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
        if not self.cli_context.current_db_name:
            db_name, self.db_connector = self.db_manager.first_conn_with_name(current_namespace)
            if self.db_connector.dialect in (DBType.SQLITE, DBType.DUCKDB):
                self.cli_context.update_database_context(db_name=self.db_connector.database_name, db_logic_name=db_name)
            else:
                self.cli_context.update_database_context(
                    catalog=self.db_connector.catalog_name,
                    db_name=self.db_connector.database_name,
                    db_logic_name=db_name or self.db_connector.database_name or self.agent_config.current_namespace,
                )
        else:
            self.db_connector = self.db_manager.get_conn(current_namespace, self.cli_context.current_db_name)
            if self.db_connector.dialect in (DBType.SQLITE, DBType.DUCKDB):
                self.cli_context.update_database_context(
                    db_name=self.db_connector.database_name, db_logic_name=self.cli_context.current_db_name
                )
        if not self.db_connector:
            self.console.print("[bold red]Error:[/] No database connection.")
            return

        # Test the connection, ff there is an exception, it will be handled by unified exception handling.
        connection_result = self.db_connector.test_connection()
        logger.debug(f"Connection test result: {connection_result}")
