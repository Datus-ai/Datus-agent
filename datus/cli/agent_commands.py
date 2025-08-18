"""
Agent, workflow, and node-related commands for the Datus CLI.
This module provides a class to handle all agent-related commands.
"""

import asyncio
import uuid

from rich.prompt import Confirm

from datus.agent.evaluate import setup_node_input, update_context_from_node
from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.cli.action_history_display import ActionHistoryDisplay
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.base import BaseInput
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput
from datus.schemas.node_models import ExecuteSQLInput, GenerateSQLInput, OutputInput, SqlTask
from datus.schemas.reason_sql_node_models import ReasoningInput
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.tools.lineage_graph_tools.schema_lineage import SchemaLineageTool
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.rich_util import dict_to_tree

logger = get_logger(__name__)


class AgentCommands:
    """Handles all agent, workflow, and node-related commands."""

    def __init__(self, cli_instance):
        """Initialize with reference to the CLI instance for shared resources."""
        self.cli = cli_instance
        self.console = cli_instance.console
        self.agent = cli_instance.agent
        self.darun_is_running = False
        # Only one interactive workflow is supported, the other one is the background workflow
        self.workflow = None
        self.agent_thread = None
        # Get current database from CLI args
        self.current_database = getattr(cli_instance.args, "database", "") or ""

    def update_agent_reference(self):
        """Update the agent reference if it has changed in the CLI."""
        self.agent = self.cli.agent
        # self.workflow = self.cli.workflow

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

            # Reset agent's workflow to ensure we create a new one
            if self.agent_thread:
                self.agent_thread.join(timeout=1)
                if self.agent_thread.is_alive():
                    self.console.print(
                        "[bold red]Warning: preious Agent thread is still running, attempting to terminate...[/]"
                    )
                    self.agent_thread._stop()
                self.agent_thread = None
            self.agent.workflow = None
            self.agent.workflow_ready = False

            agent_done = threading.Event()
            self.darun_is_running = True

            def run_agent(sql_task):
                # nonlocal result, error
                try:
                    self.agent.run(sql_task)
                except Exception as e:
                    logger.error(f"Agent query error: {str(e)}")
                finally:
                    agent_done.set()

            thread = threading.Thread(target=run_agent, kwargs={"sql_task": sql_task})
            thread.daemon = True
            thread.start()
            self.agent_thread = thread

            while not hasattr(self.agent, "workflow") or not self.agent.workflow_ready:
                self.console.print("[bold yellow]Waiting for workflow to be initialized...[/]")
                time.sleep(1)

            # Store the new workflow
            self.workflow = self.agent.workflow
            show_workflow_screen(self.workflow, run_new_loop=False)

            # If agent is still running, leave it and

            if agent_done.is_set():
                self.darun_is_running = False
                workflow = self.agent.workflow
                if workflow and workflow.is_complete():
                    self.console.print("[bold green]Query Result:[/]")
                    self.console.print(workflow.get_final_result())
                else:
                    self.console.print(f"[bold red]Query is not complete: {workflow.status}[/]")
            else:
                self.console.print("[bold yellow]Agent is still running...[/]")
                # thread.join()
                self.darun_is_running = False

        except Exception as e:
            logger.error(f"Agent query error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_darun(self, args: str):
        """Run a natural language query through the agent."""
        # ignore args for now
        try:
            sql_task = self._gen_sql_task(args)
            if not sql_task:
                return
            # Run the query through the agent
            result = self.agent.run(sql_task)

            # Display the result
            if self.agent.workflow.is_complete():
                self.console.print("[bold green]Query Result:[/]")
                self.console.print(result)
            else:
                self.console.print(f"[bold red]Query is failed. {self.agent.workflow.status}[/]")
                self.console.print(result)
        except Exception as e:
            logger.error(f"Agent query error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _gen_sql_task(self, args: str):
        """Generate a SQL task from the user input."""
        try:
            # 1. Create SQL Task with user input
            self.console.print("[bold blue]Creating a new SQL task[/]")

            # Generate a unique ID for the task
            task_id = str(uuid.uuid1())[:8]

            # Get database type from connector
            database_type = self.cli.db_connector.get_type() if self.cli.db_connector else DBType.SQLITE

            # Task description - required input from user
            if args.strip():
                task_description = args
                # Use current_db_name from CLI (updated by .database command), fallback to current_database from args
                database_name = self.cli.current_db_name or self.current_database or self.cli.args.db_path
                output_dir = "output"
                external_knowledge = ""
            else:  # If no input, use a prompt to get the task info
                task_id = self.cli._prompt_input("Enter task ID", default=task_id)
                task_description = self.cli._prompt_input("Enter task description", default="")
                if not task_description.strip():
                    self.console.print("[bold red]Error:[/] Task description is required")
                    return
                # Database name - optional input, use current_db_name from CLI as default
                default_db = (
                    self.cli.current_db_name
                    or self.current_database
                    or (self.cli.args.db_path if hasattr(self.cli.args, "db_path") else "")
                )
                database_name = self.cli._prompt_input("Enter database name", default=default_db)
                # Output directory - optional input
                output_dir = "output"
                # output_dir = self.cli._prompt_input("Enter output directory", default="output")
                # External knowledge - optional input
                external_knowledge = self.cli._prompt_input("Enter external knowledge (optional)", default="")

            # Create the SQL task
            sql_task = SqlTask(
                id=task_id,
                database_type=database_type,
                task=task_description,
                database_name=database_name,
                output_dir=output_dir,
                external_knowledge=external_knowledge,
            )

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
                )
            )

            if not workflow:
                self.console.print("[bold red]Error:[/] Failed to create workflow")
                return

            # 3. Setup basic context
            workflow.task = sql_task
            workflow.status = "running"

            self.agent.workflow = workflow
            self.workflow = workflow
            self.console.print(f"[bold green]Started new agent session (ID: {sql_task.id})[/]")
            # self.console.print(f"[dim]Next node: {workflow.get_current_node().type}[/]")

        except Exception as e:
            logger.error(f"Failed to start agent session: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_sl(self, args: str):
        """Show list of recommended tables."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the schema linking node
        self.run_node(NodeType.TYPE_SCHEMA_LINKING)

    def _modify_input(self, input: BaseInput):
        if isinstance(input, SchemaLinkingInput):
            top_n = self.cli._prompt_input("Enter number of tables to link", default="5")
            input.top_n = int(top_n.strip())
            matching_rate = self.cli._prompt_input(
                "Enter matching method",
                choices=["fast", "medium", "slow", "from_llm"],
                default="fast",
            )
            input.matching_rate = matching_rate.strip()
            database_name = self.cli._prompt_input("Enter database name", default=input.database_name)
            input.database_name = database_name.strip()
        elif isinstance(input, GenerateSQLInput):
            pass
        elif isinstance(input, ExecuteSQLInput):
            pass
        elif isinstance(input, ReasoningInput):
            pass
        elif isinstance(input, OutputInput):
            if self.workflow.context.sql_contexts:
                self.console.print("[bold blue]SQL Contexts:[/]")
                for i, sql_context in enumerate(self.workflow.context.sql_contexts):
                    self.console.print(f"\n[bold]Context {i + 1}:[/]")
                    self.console.print(sql_context.to_dict())

                sql_context_id = self.cli._prompt_input(
                    "Enter SQL context ID", default=str(len(self.workflow.context.sql_contexts))
                )
                try:
                    context_id = int(sql_context_id.strip())
                    if context_id < 1 or context_id > len(self.workflow.context.sql_contexts):
                        self.console.print("[bold red]Error:[/] Invalid SQL context ID")
                        return
                    input.sql_result = self.workflow.context.sql_contexts[context_id - 1].sql_return
                    input.row_count = self.workflow.context.sql_contexts[context_id - 1].row_count
                except ValueError:
                    self.console.print("[bold red]Error:[/] Invalid SQL context ID")
                    return
            else:
                self.console.print("[bold red]Error:[/] No SQL context available")
        elif isinstance(input, GenerateMetricsInput):
            # Allow user to modify SQL query and prompt version
            sql_query = self.cli._prompt_input("Enter SQL query to generate metrics from", default=input.sql_query)
            input.sql_query = sql_query.strip()
            prompt_version = self.cli._prompt_input("Enter prompt version", default=input.prompt_version)
            input.prompt_version = prompt_version.strip()
        elif isinstance(input, GenerateSemanticModelInput):
            # Allow user to modify table name
            table_name = self.cli._prompt_input(
                "Enter table name to generate semantic model from", default=input.table_name
            )
            input.table_name = table_name.strip()

            # Interactive prompts for metadata (now using sql_task fields)
            self.console.print("[bold blue]Semantic Model Metadata:[/]")
            catalog_name = self.cli._prompt_input("Enter catalog name", default=input.sql_task.catalog_name)
            input.sql_task.catalog_name = catalog_name.strip()

            database_name = self.cli._prompt_input("Enter database name", default=input.sql_task.database_name)
            input.sql_task.database_name = database_name.strip()

            schema_name = self.cli._prompt_input("Enter schema name", default=input.sql_task.schema_name)
            input.sql_task.schema_name = schema_name.strip()

            layer1 = self.cli._prompt_input("Enter layer1 (business layer)", default=input.sql_task.layer1)
            input.sql_task.layer1 = layer1.strip()

            layer2 = self.cli._prompt_input("Enter layer2 (sub-layer)", default=input.sql_task.layer2)
            input.sql_task.layer2 = layer2.strip()

            domain = self.cli._prompt_input("Enter domain", default=input.sql_task.domain)
            input.sql_task.domain = domain.strip()

            prompt_version = self.cli._prompt_input("Enter prompt version", default=input.prompt_version)
            input.prompt_version = prompt_version.strip()
        elif isinstance(input, CompareInput):
            # Allow user to modify expectation
            if not input.expectation:
                expectation = self.cli._prompt_input("Enter expectation (SQL query or expected data)", default="")
                input.expectation = expectation.strip()

    def cmd_gen(self, args: str):
        """Generate SQL for a task."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the SQL generation node
        self.run_node(NodeType.TYPE_GENERATE_SQL, args)

    def cmd_run(self, args: str):
        """Run the last generated SQL."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the SQL generation node
        self.run_node(NodeType.TYPE_EXECUTE_SQL, args)

        # if not self.cli.last_sql:
        #    self.console.print("[bold red]Error:[/] No SQL to run. Generate SQL first.")
        #    return
        #
        # try:
        #    # Execute the SQL
        #    self.console.print(f"[dim]Running SQL: {self.cli.last_sql}[/]")
        #    self.cli._execute_sql(self.cli.last_sql)
        # except Exception as e:
        #    logger.error(f"SQL execution error: {str(e)}")
        #    self.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_fix(self, args: str):
        """Fix the last SQL query."""
        if not self.cli.last_sql:
            self.console.print("[bold red]Error:[/] No SQL to fix. Generate SQL first.")
            return

        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        self.run_node(NodeType.TYPE_FIX, args)

    def cmd_reflect(self, args: str):
        """Reflect on the last result and improve the query."""
        if not self.cli.last_result:
            self.console.print("[bold red]Error:[/] No result to reflect on. Run a query first.")
            return

        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the reflection node
        node_args = {
            "sql": self.cli.last_sql,
            "result": self.cli.last_result,
            "prompt": args if args else "Reflect on this result",
        }
        self.run_node(NodeType.TYPE_REFLECT, node_args)

    def cmd_reason(self, args: str):
        """Run the full reasoning node."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the reasoning node
        self.run_node(NodeType.TYPE_REASONING, args)

    def cmd_reason_stream(self, args: str):
        """Run SQL reasoning with streaming output and action history."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()
        self._run_node_stream(NodeType.TYPE_REASONING, args)

    def cmd_gen_metrics(self, args: str):
        """Generate metrics from SQL queries and tables."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the generate metrics node
        self.run_node(NodeType.TYPE_GENERATE_METRICS, args)

    def cmd_gen_metrics_stream(self, args: str):
        """Generate metrics with streaming output and action history."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        self._run_node_stream(NodeType.TYPE_GENERATE_METRICS, args)

    def cmd_gen_semantic_model(self, args: str):
        """Generate semantic model for data modeling."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the generate semantic model node
        self.run_node(NodeType.TYPE_GENERATE_SEMANTIC_MODEL, args)

    def cmd_gen_semantic_model_stream(self, args: str):
        """Generate semantic model with streaming output and action history."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()
        self._run_node_stream(NodeType.TYPE_GENERATE_SEMANTIC_MODEL, args)

    def cmd_daend(self, args: str):
        """End the current agent session."""
        if self.workflow:
            output_file = f"{self.workflow.task.output_dir}/{self.workflow.name}.yaml"
            self.workflow.save(output_file)
            self.console.print(f"[green]Ending workflow session, save to {output_file}[/]")
            self.workflow = None
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

        if not self.workflow:
            self.console.print("[bold red]Error:[/] No active workflow")
            return {"success": False, "error": "No active workflow"}

        try:
            workflow = self.workflow

            # 1. Create a new node
            node_id = f"{node_type.lower()}_{str(uuid.uuid1())[:8]}"
            description = f"Execute {node_type} operation"
            next_node = Node.new_instance(
                node_id=node_id,
                description=description,
                node_type=node_type.lower(),
                input_data=node_args,
                agent_config=self.cli.agent_config,
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
                    choice = self.cli._prompt_input(
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
                    from rich.syntax import Syntax

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

    def cmd_save(self, args: str, file_name: str = None):
        """Save the last SQL and result files"""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Run the reasoning node
        self.run_node(NodeType.TYPE_OUTPUT, args)

    def cmd_set_context(self, context_type: str = "sql"):
        """Set the context for the agent."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # get lastest sql context
        if context_type == "sql":
            if not self.workflow.context.sql_contexts:
                self.console.print("[yellow]No SQL contexts available[/]")
                return

            # Display all SQL contexts
            self.console.print("[bold blue]SQL Contexts:[/]")
            for i, sql_context in enumerate(self.workflow.context.sql_contexts):
                self.console.print(f"\n[bold]Context {i + 1}:[/]")
                self.console.print(sql_context.to_dict())

            confirmed = Confirm.ask("[bold yellow]Do you want to update the context?[/]", default=True)
            if not confirmed:
                self.console.print("[yellow]Operation cancelled by user[/]")
                return

            # update context
        elif context_type == "schema":
            self.console.print("[bold blue]Table Context:[/]")
            for i, table_schema in enumerate(self.workflow.context.table_schemas):
                self.console.print(f"{i + 1}. {table_schema.to_dict()}")

            tables = self.cli._prompt_input("Enter table names you want to update the context for", default="")
            if not tables.strip():
                self.console.print("[bold red]Error:[/] Table names are required")
                return

            schema_tool = SchemaLineageTool(agent_config=self.cli.agent_config)
            table_schemas, table_values = schema_tool.get_table_and_values(
                self.workflow.task.database_name, tables.strip().split(",")
            )
            self.workflow.context.update_schema_and_values(table_schemas, table_values)
            self.console.print(f"[bold green]Table context updated to {tables}[/]")

        elif context_type == "metrics":
            pass
        else:
            self.console.print("[bold red]Error:[/] Invalid context type")
            return

        self.console.print("[yellow]Feature not implemented in MVP.[/]")

    def cmd_compare(self, args: str):
        """Compare SQL with expectations - interactive analysis."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Get expectation from user
        expectation = self.cli._prompt_input("Enter expectation (SQL query or expected data format)", default="")

        if not expectation.strip():
            self.console.print("[bold red]Error:[/] Expectation cannot be empty")
            return

        # Run compare node with expectation
        self.run_node(NodeType.TYPE_COMPARE, expectation)

    def cmd_compare_stream(self, args: str):
        """Compare SQL with streaming output and action history."""
        if not self.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # Get expectation from user
        expectation = self.cli._prompt_input("Enter expectation (SQL query or expected data format)", default="")

        if not expectation.strip():
            self.console.print("[bold red]Error:[/] Expectation cannot be empty")
            return

        # Run compare node with streaming, passing expectation as args
        self._run_node_stream(NodeType.TYPE_COMPARE, expectation)

    def _run_node_stream(self, node_type: str, node_args: str):
        """Run a node with streaming output and action history display."""
        try:
            workflow = self.workflow

            # Create a new node
            node_id = f"{node_type.lower()}_{str(uuid.uuid1())[:8]}"
            description = f"Execute {node_type} operation with streaming"
            next_node = Node.new_instance(
                node_id=node_id,
                description=description,
                node_type=node_type.lower(),
                input_data=node_args,
                agent_config=self.cli.agent_config,
            )

            # Setup input for the node
            setup_result = setup_node_input(node=next_node, workflow=workflow)
            workflow.add_node(next_node)

            if not setup_result.get("success", False):
                self.console.print(
                    "[bold red]Error:[/] Failed to setup node input: " f"{setup_result.get('message', 'Unknown error')}"
                )
                return {"success": False, "error": "Failed to setup node input"}

            # Interactive input modification
            if isinstance(next_node.input, BaseInput):
                edit_mode = self._modify_input(next_node.input)
                if edit_mode == "cancel":
                    return {"success": False, "error": "Operation cancelled by user"}

            # Initialize action history
            action_history_manager = ActionHistoryManager()
            action_display = ActionHistoryDisplay(self.console)

            # Start streaming execution
            self.console.print(f"[bold green]Executing {node_type} node with streaming...[/]")

            # Initialize the node first to set up the model
            next_node._initialize()

            # Run the streaming method
            streaming_method = None
            if hasattr(next_node, "_generate_semantic_model_stream"):
                streaming_method = next_node._generate_semantic_model_stream
            elif hasattr(next_node, "_generate_metrics_stream"):
                streaming_method = next_node._generate_metrics_stream
            elif hasattr(next_node, "_reason_sql_stream"):
                streaming_method = next_node._reason_sql_stream
            elif hasattr(next_node, "_compare_sql_stream"):
                streaming_method = next_node._compare_sql_stream

            if streaming_method:
                actions = []

                # Create a live display
                with action_display.display_streaming_actions(actions):
                    # Run the async streaming method
                    async def run_stream():
                        async for action in streaming_method(action_history_manager):
                            actions.append(action)
                            # Longer delay to make the streaming visible and avoid caching
                            await asyncio.sleep(0.5)

                    # Execute the streaming
                    asyncio.run(run_stream())

                show_details = self.cli._prompt_input("Show Full Action History? (y/N)", default="n").strip().lower()
                if show_details == "y":
                    self.console.print("\n[bold blue]Full Action History:[/]")
                    action_display.display_final_action_history(actions)

                # Extract result from final action
                if actions:
                    final_action = actions[-1]
                    if final_action.output and isinstance(final_action.output, dict):
                        success = final_action.output.get("success", False)
                        if success:
                            self.console.print("[bold green]Streaming execution completed successfully![/]")

                            # For _reason_sql_stream, extract SQL from the final action and add to workflow context
                            if node_type == NodeType.TYPE_REASONING and hasattr(next_node, "_reason_sql_stream"):
                                logger.debug("Detected _reason_sql_stream node, calling SQL extraction...")
                                self._extract_sql_from_streaming_actions(actions, workflow, next_node)
                            else:
                                has_method = hasattr(next_node, "_reason_sql_stream") if next_node else False
                                logger.debug(f"Not a _reason_sql_stream node (type: {node_type}, method: {has_method})")

                            return {"success": True, "actions": actions}
                        else:
                            error_msg = final_action.output.get("error", "Unknown error")
                            self.console.print(f"[bold red]Streaming execution failed:[/] {error_msg}")
                            return {"success": False, "error": error_msg, "actions": actions}

                return {"success": True, "actions": actions}
            else:
                self.console.print("[bold red]Error:[/] Node does not support streaming")
                return {"success": False, "error": "Node does not support streaming"}

        except Exception as e:
            logger.error(f"Streaming node execution error: {str(e)}")

            # Import DatusException for proper error handling
            from datus.utils.exceptions import DatusException, ErrorCode

            # Handle DatusException with structured error codes
            if isinstance(e, DatusException):
                error_code = e.code

                if error_code in [ErrorCode.MODEL_OVERLOADED, ErrorCode.MODEL_RATE_LIMIT]:
                    self.console.print(f"[bold red]API Error:[/] {error_code.desc}")
                    self.console.print(
                        "[yellow]Suggestion:[/] Please wait a moment and try again with the same command."
                    )
                    self.console.print(f"[dim]Error code: {error_code.code}[/]")
                elif error_code == ErrorCode.MODEL_CONNECTION_ERROR:
                    self.console.print(f"[bold red]Connection Error:[/] {error_code.desc}")
                    self.console.print("[yellow]Suggestion:[/] Check your internet connection and try again.")
                    self.console.print(f"[dim]Error code: {error_code.code}[/]")
                elif error_code == ErrorCode.MODEL_AUTHENTICATION_ERROR:
                    self.console.print(f"[bold red]Authentication Error:[/] {error_code.desc}")
                    self.console.print("[yellow]Suggestion:[/] Check your API key configuration.")
                    self.console.print(f"[dim]Error code: {error_code.code}[/]")
                else:
                    self.console.print(f"[bold red]Error:[/] {error_code.desc}")
                    self.console.print(f"[dim]Error code: {error_code.code}[/]")
            else:
                # Fallback for non-DatusException errors
                error_msg = str(e).lower()
                if any(indicator in error_msg for indicator in ["overloaded", "rate limit", "timeout"]):
                    self.console.print("[bold red]API Error:[/] The API is temporarily overloaded or rate limited.")
                    self.console.print(
                        "[yellow]Suggestion:[/] Please wait a moment and try again with the same command."
                    )
                    self.console.print(f"[dim]Original error: {str(e)}[/]")
                elif any(indicator in error_msg for indicator in ["connection", "network"]):
                    self.console.print("[bold red]Connection Error:[/] Unable to connect to the API.")
                    self.console.print("[yellow]Suggestion:[/] Check your internet connection and try again.")
                    self.console.print(f"[dim]Original error: {str(e)}[/]")
                else:
                    self.console.print(f"[bold red]Error:[/] {str(e)}")

            return {"success": False, "error": str(e)}

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
