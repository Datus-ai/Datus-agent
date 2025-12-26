# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Agent, workflow, and node-related commands for the Datus CLI.
This module provides a class to handle all agent-related commands.
"""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from datus.agent.evaluate import setup_node_input, update_context_from_node
from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.cli.subject_rich_utils import build_historical_sql_tags
from datus.configuration.node_type import NodeType
from datus.schemas.base import BaseInput
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.node_models import ExecuteSQLInput, GenerateSQLInput, OutputInput, SqlTask
from datus.schemas.reason_sql_node_models import ReasoningInput
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.output_tools import OutputTool
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.rich_util import dict_to_tree

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI


class AgentCommands:
    """Handles all agent, workflow, and node-related commands."""

    def __init__(self, cli_instance: "DatusCLI", cli_context):
        """Initialize with reference to the CLI instance and CLI context."""
        self.cli = cli_instance
        self.cli_context = cli_context
        self.console = cli_instance.console
        self.agent = cli_instance.agent
        self.darun_is_running = False
        self.agent_thread = None
        self._context_search_tools: ContextSearchTools | None = None
        self.output_tool: OutputTool | None = None

    @property
    def context_search_tools(self):
        if not self._context_search_tools:
            self._context_search_tools = ContextSearchTools(self.cli.agent_config)
        return self._context_search_tools

    def update_agent_reference(self):
        """Update the agent reference if it has changed in the CLI."""
        self.agent = self.cli.agent

    def create_node_input(self, node_type: str, task_text: str = None) -> BaseInput:
        """Create input for a specific node type with console prompts."""
        # Get or create SQL task
        try:
            sql_task = self._gen_sql_task(task_text or "", use_existing=True)
            if not sql_task:
                return None
        except ValueError as e:
            self.console.print(f"[bold red]Error:[/] {str(e)}")
            return None

        if node_type == NodeType.TYPE_SCHEMA_LINKING:
            if not task_text:
                task_text = sql_task.task or self.cli.prompt_input(
                    "Enter task description for schema linking", default=""
                )
            top_n = self.cli.prompt_input("Enter number of tables to link", default="5")
            matching_rate = self.cli.prompt_input(
                "Enter matching method",
                choices=["fast", "medium", "slow", "from_llm"],
                default="fast",
            )
            return SchemaLinkingInput(
                input_text=task_text,
                catalog_name=sql_task.catalog_name,
                database_name=sql_task.database,
                schema_name=sql_task.schema,
                database_type=sql_task.database_type,
                top_n=int(top_n.strip()),
                matching_rate=matching_rate.strip(),
            )

        elif node_type == NodeType.TYPE_GENERATE_SQL:
            sql_task.task = (
                task_text
                or sql_task.task
                or self.cli.prompt_input("Enter task description for SQL generation", default="")
            )
            return GenerateSQLInput(
                sql_task=sql_task,
                database_type=sql_task.database_type,
                table_schemas=self.cli_context.get_recent_tables(),
                metrics=self.cli_context.get_recent_metrics(),
            )

        elif node_type == NodeType.TYPE_FIX:
            last_sql = self.cli_context.get_last_sql()
            if not last_sql:
                self.console.print("[bold red]Error:[/] No recent SQL to fix")
                return None
            fix_description = self.cli.prompt_input("Describe the issue to fix", default="")
            return ExecuteSQLInput(
                sql_query=last_sql, sql_task=sql_task, database_type=sql_task.database_type, expectation=fix_description
            )

        elif node_type == NodeType.TYPE_REASONING:
            sql_query = self.cli.prompt_input(
                "Enter SQL query to reason about", default=self.cli_context.get_last_sql() or ""
            )
            return ReasoningInput(
                sql_query=sql_query,
                sql_task=sql_task,
                database_type=sql_task.database_type,
                table_schemas=self.cli_context.get_recent_tables(),
            )

        elif node_type == NodeType.TYPE_COMPARE:
            expectation = self.cli.prompt_input("Enter expectation (SQL query or expected data format)", default="")
            if not expectation.strip():
                self.console.print("[bold red]Error:[/] Expectation cannot be empty")
                return None
            return CompareInput(
                sql_context=self.cli_context.get_last_sql_context(), expectation=expectation, sql_task=sql_task
            )

        else:
            raise ValueError(f"Unsupported node type: {node_type}")

    def run_standalone_node(self, node_type: str, input_data: BaseInput, need_confirm: bool = True) -> any:
        """Run a node standalone without workflow dependency."""
        try:
            # Show input confirmation if needed
            if need_confirm:
                self.console.print(f"\n[bold blue]About to run {node_type} node with input:[/]")
                self.console.print(dict_to_tree(input_data.to_dict()))

                if not Confirm.ask("Continue with this configuration?", default=True):
                    self.console.print("[yellow]Operation cancelled[/]")
                    return None

            # Create node instance
            node = Node.new_instance(
                node_id=f"standalone_{node_type}",
                description=f"Standalone {node_type} node",
                node_type=node_type,
                agent_config=self.cli.agent_config,
                input_data=input_data,
            )

            # Run the node
            self.console.print(f"[dim]Running {node_type} node...[/]")
            result = asyncio.run(node.run_async())

            return result

        except Exception as e:
            self.console.print(f"[bold red]Error running {node_type} node:[/] {str(e)}")
            logger.error(f"Error in standalone node execution: {e}")
            return None

    def cmd_darun_screen(self, args: str, task: SqlTask = None):
        """Run a natural language query through the agent."""
        try:
            import threading
            import time

            from datus.cli.screen import show_workflow_screen

            # create a new sql task
            sql_task = task or self._gen_sql_task(args)
            if not sql_task:
                return

            # Reset previous runner/thread state
            if self.agent_thread:
                self.agent_thread.join(timeout=1)
                if self.agent_thread.is_alive():
                    self.console.print(
                        "[bold red]Warning: preious Agent thread is still running, attempting to terminate...[/]"
                    )
                    self.agent_thread._stop()
                self.agent_thread = None

            if not self.cli.check_agent_available():
                self.console.print("[bold red]Error:[/] Agent not available")
                return
            runner = self.cli.workflow_runner

            agent_done = threading.Event()
            self.darun_is_running = True
            result_holder: Dict[str, Any] = {"result": None}

            def run_agent(sql_task):
                # nonlocal result, error
                try:
                    result_holder["result"] = runner.run(sql_task)
                except Exception as e:
                    logger.error(f"Agent query error: {str(e)}")
                finally:
                    agent_done.set()

            thread = threading.Thread(target=run_agent, kwargs={"sql_task": sql_task})
            thread.daemon = True
            thread.start()
            self.agent_thread = thread

            while not runner.workflow_ready:
                if agent_done.is_set():
                    break
                self.console.print("[bold yellow]Waiting for workflow to be initialized...[/]")
                time.sleep(1)

            if not runner.workflow_ready or not runner.workflow:
                self.console.print("[bold red]Failed to initialize workflow[/]")
                self.darun_is_running = False
                return

            # Store the new workflow

            show_workflow_screen(runner.workflow, run_new_loop=False)

            # If agent is still running, leave it and

            if agent_done.is_set():
                self.darun_is_running = False
                if runner.is_complete():
                    self.console.print("[bold green]Query Result:[/]")
                    final_result = result_holder["result"] or runner.workflow.get_final_result()
                    self.console.print(final_result)
                else:
                    self.console.print(f"[bold red]Query is not complete: {runner.workflow.status}[/]")
            else:
                self.console.print("[bold yellow]Agent is still running...[/]")
                # thread.join()
                self.darun_is_running = False

        except Exception as e:
            logger.error(f"Agent query error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _gen_sql_task(self, args: str, use_existing: bool = True):
        """Generate a SQL task from the user input, optionally reusing existing task."""
        try:
            # Check if we can reuse existing task
            if use_existing and self.cli_context.current_sql_task and not args.strip():
                return self.cli_context.current_sql_task

            # 1. Create SQL Task with user input
            self.console.print("[bold blue]Creating a new SQL task[/]")

            # Generate a unique ID for the task
            task_id = str(uuid.uuid1())[:8]

            # Get database type from connector
            database_type = self.cli.db_connector.get_type() if self.cli.db_connector else DBType.SQLITE

            # Task description - required input from user
            if args.strip():
                task_description = args
                # Use current_db_name from CLI context
                database_name = self.cli_context.current_db_name or self.cli.args.db_path
                output_dir = self.cli.agent_config.output_dir
                external_knowledge = ""
                current_date = ""
            else:  # If no input, use a prompt to get the task info
                task_id = self.cli.prompt_input("Enter task ID", default=task_id)

                # Use existing task description as default if available
                default_task = self.cli_context.current_sql_task.task if self.cli_context.current_sql_task else ""
                task_description = self.cli.prompt_input("Enter task description", default=default_task)
                if not task_description.strip():
                    self.console.print("[bold red]Error:[/] Task description is required")
                    return

                # Database name - use CLI context as default
                default_db = (
                    self.cli_context.current_db_name
                    or (self.cli.args.db_path if hasattr(self.cli.args, "db_path") else "")
                    or ""
                )
                database_name = self.cli.prompt_input("Enter database name", default=default_db)
                if not database_name.strip():
                    self.console.print("[bold red]Error:[/] Database name is required")
                    return

                # Output directory - use agent config
                output_dir = self.cli.agent_config.output_dir

                # External knowledge - optional input
                external_knowledge = self.cli.prompt_input("Enter external knowledge (optional)", default="")

                # Current date - optional input for relative time expressions
                current_date = self.cli.prompt_input("Enter current date (optional, e.g., '2025-07-01')", default="")

            # Create the SQL task
            sql_task = SqlTask(
                id=task_id,
                database_type=database_type,
                task=task_description,
                database_name=database_name,
                output_dir=output_dir,
                external_knowledge=external_knowledge,
                current_date=current_date if current_date.strip() else None,
            )

            # Store in CLI context
            self.cli_context.set_current_sql_task(sql_task)

            self.console.print(f"[green]SQL Task created: {task_id}[/]")
            self.console.print(f"[dim]Database: {database_type} - {database_name}[/]")
            return sql_task
        except Exception as e:
            logger.error(f"Failed to create SQL task: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")
            return None

    def cmd_dastart(self, args: str = ""):
        """Start a new agent session with interactive SQL task creation."""
        try:
            # 1. Generate SQL task
            sql_task = self._gen_sql_task(args)
            if not sql_task:
                return

            # 2. Initialize workflow (ToDO implement load_cp)
            workflow = Workflow(name=sql_task.id, task=sql_task, agent_config=self.cli.agent_config)
            workflow.add_node(
                Node.new_instance(
                    node_id="START",
                    description="Start node",
                    node_type="start",
                    input_data=sql_task.id,
                    agent_config=self.cli.agent_config,
                    tools=workflow.tools,
                )
            )

            if not workflow:
                self.console.print("[bold red]Error:[/] Failed to create workflow")
                return

            # 3. Setup basic context
            workflow.task = sql_task
            workflow.status = "running"

            self.console.print(f"[bold green]Started new agent session (ID: {sql_task.id})[/]")
            # self.console.print(f"[dim]Next node: {workflow.get_current_node().type}[/]")

        except Exception as e:
            logger.error(f"Failed to start agent session: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_schema_linking(self, args: str):
        """
        Command to perform schema linking. Corresponds to !sl
        """
        self.console.print("[bold blue]Schema Linking[/]")
        input_text = args.strip() or self.cli.prompt_input("Enter search text for tables")
        if not input_text:
            self.console.print("[bold red]Error:[/] Input text cannot be empty.")
            return

        catalog_name, database_name, schema_name = self._prompt_db_layers()
        top_n = self.cli.prompt_input("Enter top_n to match", default="5")

        # The tool's search_similar seems to handle table_type internally.
        # The PDF mentions table_type, but the tool implementation has it fixed to "full".
        # I will omit prompting for it as it won't be used.

        with self.console.status("[bold green]Searching for relevant tables...[/]"):
            from datus.storage.schema_metadata import SchemaWithValueRAG

            schema_rag = SchemaWithValueRAG(self.cli.agent_config)
            metadata, sample_data = schema_rag.search_similar(
                query_text=input_text,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                top_n=int(top_n.strip()),
            )

        if metadata.num_rows > 0 or sample_data.num_rows > 0:
            self.console.print(
                f"Found [bold green]{len(metadata)}[/] relevant tables and [bold blue]{len(sample_data)}[/] sample rows"
            )

            if metadata.num_rows > 0:
                self._print_metadata_table(
                    metadata.to_pylist(), data_column="definition", data_column_dsc="Definition (DDL)", lexer="sql"
                )

            if sample_data.num_rows > 0:
                self._print_metadata_table(
                    sample_data.to_pylist(), data_column="sample_rows", data_column_dsc="Sample Rows", lexer="markdown"
                )

        else:
            self.console.print("[yellow]No relevant tables found.[/]")

    def _prompt_db_layers(self) -> Tuple[str, str, str]:
        dialect = self.cli.db_connector.dialect
        catalog_name, database_name, schema_name = "", "", ""

        if DBType.support_catalog(dialect):
            catalog_name = self.cli.prompt_input("Enter catalog name", default=self.cli_context.current_catalog or "")
        if DBType.SQLITE == dialect or DBType.support_database(dialect):
            database_name = self.cli.prompt_input("Enter database name", default=self.cli_context.current_db_name or "")
        if DBType.support_schema(dialect):
            schema_name = self.cli.prompt_input("Enter schema name", default=self.cli_context.current_schema or "")

        return catalog_name, database_name, schema_name

    def _print_metadata_table(
        self, data_list: List[Dict[str, Any]], data_column: str, data_column_dsc: str = "", lexer: str = "sql"
    ):
        table = Table(
            title="Schema Linking Results",
            show_header=True,
            border_style="blue",
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Catalog", style="green", max_width=20)
        table.add_column("Database", style="green", max_width=20)
        table.add_column("Schema", style="green", max_width=20)
        table.add_column("Table Name", style="bold green")
        table.add_column("Type", style="yellow")
        table.add_column(data_column_dsc if data_column_dsc else data_column, style="default")
        table.add_column("Distance", style="dim")
        for item in data_list:
            table.add_row(
                item.get("catalog_name"),
                item.get("database_name"),
                item.get("schema_name"),
                item.get("table_name"),
                item.get("table_type"),
                Syntax(item.get(data_column, ""), lexer=lexer, line_numbers=True, word_wrap=True),
                str(item.get("_distance", "")),
            )
        self.console.print(table)

    def cmd_search_metrics(self, args: str):
        """
        Command to search for metrics. Corresponds to !sm and !search_metrics
        """
        self.console.print("[bold blue]Search Metrics[/]")
        input_text = args.strip() or self.cli.prompt_input("Enter search text for metrics")
        if not input_text:
            self.console.print("[bold red]Error:[/] Input text cannot be empty.")
            return
        subject_path = self._prompt_subject_path()
        top_n = self.cli.prompt_input("Enter top_n to match", default="5")

        with self.console.status("[bold green]Searching for metrics...[/]"):
            result = self.context_search_tools.search_metrics(
                query_text=input_text,
                subject_path=subject_path,
                top_n=int(top_n.strip()),
            )
        if result.success and result.result:
            metrics = result.result
            self.console.print(f"[bold green]Found {len(metrics)} metrics.[/]")
            table = Table(
                title="Metrics Search Results",
                show_header=True,
                header_style="bold cyan",
                border_style="blue",
                expand=True,
            )
            table.add_column("Name", style="bold green")
            table.add_column("LLM Text", style="default")

            for metric in metrics:
                table.add_row(
                    metric.get("name"),
                    metric.get("description"),
                )
            self.console.print(table)
        elif not result.success:
            self.console.print(f"[bold red]Error searching metrics:[/] {result.error}")
        else:
            self.console.print("[yellow]No metrics found.[/]")

    def _prompt_subject_path(self) -> Optional[List[str]]:
        """Prompt user for subject path input.

        Returns:
            List of subject path components or None if empty
        """
        subject_path_str = self.cli.prompt_input(
            "Enter subject path (e.g., 'Finance/Revenue/Q1', or leave empty for no filter)"
        )
        if not subject_path_str or not subject_path_str.strip():
            return None
        return [component.strip() for component in subject_path_str.split("/") if component.strip()]

    def cmd_search_reference_sql(self, args: str):
        """
        Command to search reference SQL queries. Corresponds to !sq and !search_sql
        """
        self.console.print("[bold blue]Search Reference SQL[/]")
        input_text = args.strip() or self.cli.prompt_input("Enter search text for reference SQL")
        if not input_text:
            self.console.print("[bold red]Error:[/] Input text cannot be empty.")
            return

        subject_path = self._prompt_subject_path()
        top_n = self.cli.prompt_input("Enter top_n to match", default="5")
        with self.console.status("[bold green]Searching reference SQL...[/]"):
            result = self.context_search_tools.search_reference_sql(
                query_text=input_text, subject_path=subject_path, top_n=int(top_n.strip())
            )

        if result.success and result.result:
            history = result.result
            self.console.print(f"[bold]Found [green]{len(history)}[/] reference SQL queries.[/]")
            table = Table(
                title="Reference SQL Search Results",
                show_header=True,
                border_style="blue",
                header_style="bold cyan",
                expand=True,
            )
            table.add_column("Name", style="bold green")
            table.add_column("SQL", style="default")
            table.add_column("Summary", style="default")
            table.add_column("Comment", style="default")
            table.add_column("Tags", style="blue")
            table.add_column("Subject Path", style="yellow")
            table.add_column("File Path", style="dim", overflow="fold")
            table.add_column("Distance", style="dim")
            #
            for item in history:
                # Format subject_path as string
                subject_path_display = "/".join(item.get("subject_path", []))
                table.add_row(
                    item.get("name"),
                    Syntax(item.get("sql"), lexer="sql", line_numbers=True, word_wrap=True),
                    item.get("summary"),
                    item.get("comment"),
                    build_historical_sql_tags(item.get("tags", ""), "\n"),
                    subject_path_display,
                    item.get("filepath"),
                    str(item.get("_distance", "")),
                )
            self.console.print(table)
        elif not result.success:
            self.console.print(f"[bold red]Error searching reference SQL:[/] {result.error}")
        else:
            self.console.print("[yellow]No reference SQL queries found.[/]")

    def cmd_save(self, args: str):
        """
        Command to save the last result to a file. Corresponds to !output/save
        """
        self.console.print("[bold blue]Save Output[/]")
        last_sql = self.cli.cli_context.get_last_sql_context()
        if not last_sql:
            self.console.print("[bold red]Error:[/] No previous result to save.")
            return

        file_type = self.cli.prompt_input(
            "Enter file type (json/csv/sql/all)", default="all", choices=["json", "csv", "sql", "all"]
        )

        from datus.utils.path_manager import get_path_manager

        default_save_dir = str(get_path_manager().save_dir)
        target_dir = self.cli.prompt_input("Enter save directory (optional)", default=default_save_dir)
        from datetime import datetime

        file_name = self.cli.prompt_input("Enter file name(optional)", default=datetime.now().strftime("%Y%m%d%H%M%S"))
        try:
            with self.console.status("[bold green]Saving SQL...[/]"):
                if not self.output_tool:
                    self.output_tool = OutputTool()
                result = self.output_tool.execute(
                    OutputInput(
                        task="",
                        database_name=self.cli.cli_context.current_db_name,
                        task_id=file_name,
                        gen_sql=last_sql.sql_query,
                        sql_result=last_sql.sql_return or "",
                        row_count=last_sql.row_count or 0,
                        file_type=file_type,
                        check_result=False,
                        error=last_sql.sql_error,
                        finished=not last_sql.sql_error,
                        output_dir=target_dir,
                    ),
                    self.cli.db_connector,
                )
            self.console.print(f"[green]SQL query saved to {result.output}[/]")

        except Exception as e:
            self.console.print(f"[bold red]Error saving file:[/] {e}")

    def _modify_input(self, input: BaseInput):
        if isinstance(input, SchemaLinkingInput):
            top_n = self.cli.prompt_input("Enter number of tables to link", default="5")
            input.top_n = int(top_n.strip())
            matching_rate = self.cli.prompt_input(
                "Enter matching method",
                choices=["fast", "medium", "slow", "from_llm"],
                default="fast",
            )
            input.matching_rate = matching_rate.strip()
            database_name = self.cli.prompt_input("Enter database name", default=input.database_name)
            input.database_name = database_name.strip()
        elif isinstance(input, GenerateSQLInput):
            pass
        elif isinstance(input, ExecuteSQLInput):
            pass
        elif isinstance(input, ReasoningInput):
            pass
        elif isinstance(input, OutputInput):
            workflow = self.cli.workflow_runner.workflow
            if workflow and workflow.context and workflow.context.sql_contexts:
                self.console.print("[bold blue]SQL Contexts:[/]")
                for i, sql_context in enumerate(workflow.context.sql_contexts):
                    self.console.print(f"\n[bold]Context {i + 1}:[/]")
                    self.console.print(sql_context.to_dict())

                sql_context_id = self.cli.prompt_input(
                    "Enter SQL context ID", default=str(len(workflow.context.sql_contexts))
                )
                try:
                    context_id = int(sql_context_id.strip())
                    if context_id < 1 or context_id > len(workflow.context.sql_contexts):
                        self.console.print("[bold red]Error:[/] Invalid SQL context ID")
                        return
                    input.sql_result = workflow.context.sql_contexts[context_id - 1].sql_return
                    input.row_count = workflow.context.sql_contexts[context_id - 1].row_count
                except ValueError:
                    self.console.print("[bold red]Error:[/] Invalid SQL context ID")
                    return
            else:
                self.console.print("[bold red]Error:[/] No SQL context available")

            # Interactive prompts for metadata (now using sql_task fields)
            self.console.print("[bold blue]Semantic Model Metadata:[/]")
            catalog_name = self.cli.prompt_input("Enter catalog name", default=input.sql_task.catalog_name)
            input.sql_task.catalog_name = catalog_name.strip()

            database_name = self.cli.prompt_input("Enter database name", default=input.sql_task.database_name)
            input.sql_task.database_name = database_name.strip()

            schema_name = self.cli.prompt_input("Enter schema name", default=input.sql_task.schema_name)
            input.sql_task.schema_name = schema_name.strip()

            # Subject path input
            default_subject_path = "/".join(input.sql_task.subject_path) if input.sql_task.subject_path else ""
            subject_path_str = self.cli.prompt_input(
                message="Enter subject path (e.g., 'Finance/Revenue/Q1')", default=default_subject_path
            )
            if subject_path_str and subject_path_str.strip():
                input.sql_task.subject_path = [c.strip() for c in subject_path_str.split("/") if c.strip()]
            else:
                input.sql_task.subject_path = None

            prompt_version = self.cli.prompt_input("Enter prompt version", default=input.prompt_version)
            input.prompt_version = prompt_version.strip()
        elif isinstance(input, CompareInput):
            # Allow user to modify expectation
            if not input.expectation:
                expectation = self.cli.prompt_input("Enter expectation (SQL query or expected data)", default="")
                input.expectation = expectation.strip()

    def cmd_gen(self, args: str):
        """Generate SQL for a task."""
        # Create input for SQL generation node
        input_data = self.create_node_input(NodeType.TYPE_GENERATE_SQL, args)
        if not input_data:
            return

        # Run standalone node
        result = self.run_standalone_node(NodeType.TYPE_GENERATE_SQL, input_data)

        if result and result.success:
            # Store generated SQL in CLI context
            if hasattr(result, "sql_contexts") and result.sql_contexts:
                for sql_context in result.sql_contexts:
                    self.cli_context.add_sql_context(sql_context)
                    self.console.print(f"[green]Generated SQL:[/] {sql_context.sql_query}")
            elif hasattr(result, "sql_query") and result.sql_query:
                self.console.print(f"[green]SQL generation completed:[/] {result.sql_query}")
            else:
                self.console.print("[green]SQL generation completed[/]")
        else:
            self.console.print("[bold red]SQL generation failed[/]")

    def cmd_fix(self, args: str):
        """Fix the last SQL query."""
        # Create input for fix node
        input_data = self.create_node_input(NodeType.TYPE_FIX, args)
        if not input_data:
            return

        # Run standalone node
        result = self.run_standalone_node(NodeType.TYPE_FIX, input_data)

        if result and result.success:
            # Store fixed SQL in CLI context
            if hasattr(result, "sql_contexts") and result.sql_contexts:
                for sql_context in result.sql_contexts:
                    self.cli_context.add_sql_context(sql_context)
                    self.console.print(f"[green]Fixed SQL:[/] {sql_context.sql_query}")
            else:
                self.console.print("[green]SQL fix completed[/]")
        else:
            self.console.print("[bold red]SQL fix failed[/]")

    def cmd_reason(self, args: str):
        """Run the full reasoning node."""
        # Create input for reasoning node
        input_data = self.create_node_input(NodeType.TYPE_REASONING, args)
        if not input_data:
            return

        # Run standalone node
        result = self.run_standalone_node(NodeType.TYPE_REASONING, input_data)

        if result and result.success:
            self.console.print("[green]SQL reasoning completed[/]")
            # Display reasoning if available
            if hasattr(result, "explanation") and result.explanation:
                self.console.print(f"[blue]Explanation:[/] {result.explanation}")
        else:
            self.console.print("[bold red]SQL reasoning failed[/]")

    def cmd_reason_stream(self, args: str):
        """Run SQL reasoning with streaming output and action history."""
        # For now, redirect to normal reason - streaming can be added later
        self.cmd_reason(args)

    def cmd_daend(self, args: str):
        """End the current agent session."""
        if self.cli.workflow_runner:
            runner = self.cli.workflow_runner
            output_file = f"{runner.workflow.task.output_dir}/{runner.workflow.name}.yaml"
            runner.workflow.save(output_file)
            self.console.print(f"[green]Ending workflow session, save to {output_file}[/]")
        else:
            self.console.print("[yellow]No active workflow session to end.[/]")

    def run_node(self, node_type: str, node_args=None, need_confirm: bool = True):
        """
        Run a specific node type in the current workflow.

        Args:
            node_type: The type of node to run (e.g., "SCHEMA_LINKING", "GENERATE_SQL")
            node_args: Optional arguments for the node

        Returns:
            Dict containing the result of the node execution
        """
        if not self.agent:
            self.console.print("[bold red]Error:[/] Agent not available")
            return {"success": False, "error": "Agent not available"}

        if not self.cli.workflow_runner:
            self.console.print("[bold red]Error:[/] No active workflow")
            return {"success": False, "error": "No active workflow"}

        try:
            if not self.cli.workflow_runner.workflow_ready:
                self.console.print("[bold red]Error:[/] Workflow not initialized")
                return {"success": False, "error": "Workflow not initialized"}
            workflow = self.cli.workflow_runner.workflow

            # 1. Create a new node
            node_id = f"{node_type.lower()}_{str(uuid.uuid1())[:8]}"
            description = f"Execute {node_type} operation"
            next_node = Node.new_instance(
                node_id=node_id,
                description=description,
                node_type=node_type.lower(),
                input_data=node_args,
                agent_config=self.cli.agent_config,
                tools=workflow.tools,
            )

            # 2. Setup input for the node
            setup_result = setup_node_input(node=next_node, workflow=workflow)
            workflow.add_node(next_node)

            if not setup_result.get("success", False):
                self.console.print(
                    "[bold red]Error:[/] Failed to setup node input: " f"{setup_result.get('message', 'Unknown error')}"
                )
                return {
                    "success": False,
                    "error": "Failed to setup node input: " f"{setup_result.get('message', 'Unknown error')}",
                }

            # Display node input for confirmation
            self.console.print(f"[bold blue]Node Type:[/] {node_type}")
            self.console.print("[bold blue]Node Input:[/]")
            self.console.print(next_node.input)

            # 3. Human confirmation
            if need_confirm:
                while True:
                    choice = self.cli.prompt_input(
                        "Do you want to execute this node? yes/no/edit",
                        choices=["y", "n", "e"],
                        default="y",
                    )
                    # modify the node input
                    if choice == "e":
                        self._modify_input(next_node.input)
                        self.console.print("[bold blue]Modified Node Input:[/]")
                        self.console.print(next_node.input)
                    # execute the node
                    elif choice == "y":
                        break
                    # cancel the node
                    elif choice == "n":
                        return

            # 4. Run the node
            self.console.print(f"[bold green]Executing {node_type} node...[/]")
            next_node.run()
            # Check if the node execution was successful
            if next_node.status == "failed":
                error_msg = "Unknown error"
                if next_node.result and hasattr(next_node.result, "error") and next_node.result.error:
                    error_msg = next_node.result.error
                elif hasattr(next_node.result, "get") and next_node.result.get("error"):
                    error_msg = next_node.result.get("error")
                elif hasattr(next_node, "error") and next_node.error:
                    error_msg = next_node.error
                self.console.print(f"[bold red]Node execution failed:[/] {error_msg}")
                return {"success": False, "error": f"Node execution failed: {error_msg}"}

            # 5. Display the result
            self.console.print("[bold green]Node Result:[/]")

            # Check if result is from a generate SQL task for SQL syntax highlighting
            if next_node.type in [NodeType.TYPE_GENERATE_SQL, NodeType.TYPE_FIX]:
                # Get result dict and extract SQL query
                result_dict = next_node.result.__dict__ if hasattr(next_node.result, "__dict__") else next_node.result
                sql_query = result_dict.get("sql_query")
                # Display SQL separately without tree structure for easy copying
                if sql_query:
                    # Display title separately
                    self.console.print("[bold green]📋 SQL Query[/]")

                    # Display SQL without panel border
                    sql_syntax = Syntax(sql_query, "sql", theme="light", line_numbers=False, word_wrap=True)
                    self.console.print(sql_syntax)

                    # Create a copy of result_dict without sql_query for tree display
                    other_info = {k: v for k, v in result_dict.items() if k != "sql_query"}

                    # Display other information in tree structure
                    if other_info:
                        result_tree = dict_to_tree(other_info, console=self.console)
                        self.console.print(result_tree)
                else:
                    # Fallback to normal tree display
                    logger.warning(f"normal result_dict: {result_dict}")
                    result_tree = dict_to_tree(result_dict, console=self.console)
                    self.console.print(result_tree)
            else:
                # Use normal print for results without sql_query
                self.console.print(dict_to_tree(next_node.result.to_dict(), console=self.console))
            logger.info(f"Node result: {next_node.result}")
            workflow.advance_to_next_node()  # ToDo: this should be modified for status

            # 6. Update workflow context
            update_result = update_context_from_node(node=next_node, workflow=workflow)

            if not update_result.get("success", False):
                self.console.print(
                    "[bold red]Warning:[/] Failed to update workflow context: "
                    f"{update_result.get('message', 'Unknown error')}"
                )
            else:
                self.console.print("[bold green]Context updated successfully[/]")

            # Save relevant results to CLI history
            if hasattr(next_node.result, "sql_query"):
                self.cli.last_sql = next_node.result.sql_query

            if hasattr(next_node.result, "sql_return"):
                self.cli.last_result = next_node.result.sql_return

            return {"success": True, "node": next_node, "result": next_node.result}

        except Exception as e:
            logger.error(f"Node execution error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")
            return {"success": False, "error": str(e)}

    def cmd_compare(self, args: str):
        """Compare SQL with expectations - interactive analysis."""
        # Create input for compare node
        input_data = self.create_node_input(NodeType.TYPE_COMPARE, args)
        if not input_data:
            return

        # Run standalone node
        result = self.run_standalone_node(NodeType.TYPE_COMPARE, input_data)

        if result and result.success:
            self.console.print("[green]SQL comparison completed[/]")
            # Display comparison result if available
            if hasattr(result, "comparison_result") and result.comparison_result:
                self.console.print(f"[blue]Comparison result:[/] {result.comparison_result}")
        else:
            self.console.print("[bold red]SQL comparison failed[/]")

    def cmd_compare_stream(self, args: str):
        """Compare SQL with streaming output and action history."""
        # For now, redirect to normal compare - streaming can be added later
        self.cmd_compare(args)

    def _extract_sql_from_streaming_actions(self, actions, workflow, node):
        """
        Extract SQL from streaming actions and add to workflow context.
        This method handles the _reason_sql_stream case where we need to update
        the workflow context with the SQL from the final action.
        """
        try:
            from datus.schemas.node_models import SQLContext
            from datus.utils.json_utils import llm_result2json

            logger.debug(f"Starting SQL extraction from streaming actions. Total actions: {len(actions)}")
            logger.debug(f"Workflow context before extraction: {len(workflow.context.sql_contexts)} SQL contexts")

            # Look for actions that contain SQL execution results or final message
            sql_contexts = []

            # First, check if the node has an action_history_manager with sql_contexts
            if hasattr(node, "action_history_manager") and node.action_history_manager:
                if hasattr(node.action_history_manager, "sql_contexts"):
                    sql_contexts.extend(node.action_history_manager.sql_contexts)
                    logger.info(f"Found {len(sql_contexts)} SQL contexts from action history manager")

            # If no SQL contexts found, try to extract from actions
            if not sql_contexts:
                # Look for SQL execution results in actions
                for action in actions:
                    # Handle both string and enum status
                    status_value = action.status.value if hasattr(action.status, "value") else action.status

                    if action.action_type == "read_query" and status_value == "success":
                        # This is a SQL execution result, create SQLContext from it
                        sql_input = action.input or {}
                        sql_output = action.output or {}

                        sql_query = sql_input.get("sql", "")
                        sql_result = sql_output.get("result", "")
                        sql_error = sql_output.get("error", "")

                        sql_context = SQLContext(
                            sql_query=sql_query,
                            explanation="",
                            sql_return=sql_result,
                            sql_error=sql_error,
                            row_count=0,
                        )
                        sql_contexts.append(sql_context)
                        logger.info(f"Added SQL context from read_query action: {sql_query[:100]}...")

                # Look for final message with SQL result
                for action in reversed(actions):  # Start from the last action
                    # Handle both string and enum role
                    role_value = action.role.value if hasattr(action.role, "value") else action.role
                    if action.action_type == "message" and role_value == "assistant":
                        # This could be the final reasoning result
                        if action.output and action.output.get("raw_output"):
                            raw_output = action.output.get("raw_output", "")

                            try:
                                # Parse the final result to extract SQL
                                content_dict = llm_result2json(raw_output)
                                sql_query = content_dict.get("sql", "")
                                explanation = content_dict.get("explanation", "")

                                if sql_query:
                                    # Create SQLContext with the final result SQL
                                    final_sql_context = SQLContext(
                                        sql_query=sql_query,
                                        explanation=explanation,
                                        sql_return="",  # Will be filled by execution
                                        sql_error="",
                                        row_count=0,
                                    )
                                    sql_contexts.append(final_sql_context)
                                    logger.info(f"Added final result SQL to SQLContext: {sql_query[:100]}...")
                                    break  # Only take the first (last) valid final result

                            except Exception as e:
                                logger.debug(f"Could not parse final message as JSON: {e}")

            # Add successful SQL contexts to workflow context
            added_count = 0
            for sql_ctx in sql_contexts:
                if sql_ctx.sql_error == "":  # only add the successful sql context
                    workflow.context.sql_contexts.append(sql_ctx)
                    added_count += 1
                    logger.info(f"✓ Added SQL context to workflow: {sql_ctx.sql_query[:100]}...")
                else:
                    logger.warning(
                        f"✗ Skipping failed SQL context: {sql_ctx.sql_query[:100]}..., error: {sql_ctx.sql_error}"
                    )

            if added_count == 0:
                logger.warning("No successful SQL contexts found in streaming execution")

        except Exception as e:
            logger.error(f"Failed to extract SQL from streaming actions: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't fail the entire process, just log the error
