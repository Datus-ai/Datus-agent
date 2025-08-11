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
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from datus.cli.agent_commands import AgentCommands
from datus.cli.autocomplete import SQLCompleter
from datus.cli.context_commands import ContextCommands
from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
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
            # temporary commands for sqlite, remove after mcp server is ready
            ".databases": self._cmd_databases,
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
        else:
            self.agent_config.current_namespace = args.strip()
            name, self.db_connector = self.db_manager.first_conn_with_name(self.agent_config.current_namespace)
            self.current_catalog = self.db_connector.catalog_name
            self.current_db_name = self.db_connector.database_name if not name else name
            self.current_schema = self.db_connector.schema_name
            self.console.print(f"[bold green]Namespace changed to: {self.agent_config.current_namespace}[/]")

    def _cmd_switch_database(self, args: str):
        new_db = args.strip()
        if not new_db:
            self.console.print("[bold red]Error:[/] Database name is required")
            return
        if self.agent_config.db_type == DBType.SQLITE or self.agent_config.db_type == DBType.DUCKDB:
            self.db_connector = self.db_manager.get_conn(self.agent_config.current_namespace, self.current_db_name)
        self.db_connector.switch_context(database_name=new_db)
        self.console.print(f"[bold green]Database switched to: {self.current_db_name}[/]")
        self.current_db_name = new_db

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
        try:
            if not self.db_connector:
                self.console.print("[bold red]Error:[/] No database connection. Please initialize a connection first.")
                return

            # Execute the query
            import time

            start_time = time.time()
            result = self.db_connector.execute_arrow(sql)
            end_time = time.time()
            if not result:
                self.console.print("[bold red]Error:[/] No result from the query.")

            # Save for later reference
            self.last_sql = sql
            self.last_result = result

            # Display results
            if result and result.success and hasattr(result.sql_return, "column_names"):
                # Convert Arrow data to list of dictionaries for smart display
                rows = result.sql_return.to_pylist()
                self._smart_display_table(data=rows, columns=result.sql_return.column_names)

                row_count = result.sql_return.num_rows
                exec_time = end_time - start_time
                self.console.print(f"[dim]Returned {row_count} rows in {exec_time:.2f} seconds[/]")

                if not system and self.agent.workflow:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.sql_return),
                        row_count=row_count,
                        explanation=f"Manual sql: Returned {row_count} rows in {exec_time:.2f} seconds",
                    )
                    self.agent.workflow.context.sql_contexts.append(new_record)
            elif result and not result.success:
                self.console.print(f"[bold red]SQL Error:[/] {result.error}")
                if not system and self.agent.workflow:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.error) if result.error else "Unknown error",
                        row_count=0,
                        explanation="Manual sql",
                    )
                    self.agent.workflow.context.sql_contexts.append(new_record)
            elif result and isinstance(result.sql_return, str):
                self.console.print(
                    f"[bold red]Error:[/] Query execution failed - received string instead of Arrow data: "
                    f"{result.error or 'Unknown error'}"
                )
            else:
                self.console.print("[bold red]Error:[/] No valid result from the query.")

        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

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
        """Execute a chat command (/ prefix)."""
        if not message.strip():
            self.console.print("[yellow]Please provide a message to chat with the AI.[/]")
            return

        if not self._check_agent_available():
            return

        try:
            # Add context to message
            context = f"sql_task: {self.agent.workflow.task.to_dict()}"
            if self.last_sql:
                context += f"\nLast SQL: {self.last_sql} \nLast SQL Result: {self.last_result}"

            prompt = f"{context}\n\nUser: {message}"

            # Create model using the same approach as the agent
            llm_model = LLMBaseModel.create_model(model_name="default", agent_config=self.agent.global_config)
            result = llm_model.generate(prompt)
            self.console.print(result)

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to generate response: {str(e)}")

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
            (".databases", "List all databases"),
            (".database database_name", "Switch current database"),
            (".tables", "List all tables"),
            (".schemas table_name", "Show schema information"),
            (".namespace namespace", "Switch current namespace"),
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

    def _cmd_databases(self, args: str):
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
                    result.append({"name": name, "uri": conn.connection_string})
            else:
                db_type = connections.dialect
                self.db_connector = connections
                if db_type == DBType.SQLITE:
                    show_uri = True
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
                    table.add_row(db_config["name"], db_config["uri"])
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
