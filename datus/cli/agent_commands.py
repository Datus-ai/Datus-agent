"""
Agent, workflow, and node-related commands for the Datus CLI.
This module provides a class to handle all agent-related commands.
"""

import uuid

from rich.prompt import Confirm, Prompt

from datus.agent.evaluate import setup_node_input, update_context_from_node
from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.base import BaseInput
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

            from datus.cli.workflow_screen import show_workflow_screen

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
                time.sleep(7)

            # Store the new workflow
            self.workflow = self.agent.workflow
            show_workflow_screen(self.workflow, run_new_loop=True)

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
                # Use current_database from CLI args, fallback to db_path
                database_name = self.current_database or self.cli.args.db_path
                output_dir = "output"
                external_knowledge = ""
            else:  # If no input, use a prompt to get the task info
                task_id = Prompt.ask("[bold]Enter task ID[/]", default=task_id)
                task_description = Prompt.ask("[bold]Enter task description[/]", default="")
                if not task_description.strip():
                    self.console.print("[bold red]Error:[/] Task description is required")
                    return
                # Database name - optional input, use current_database as default
                default_db = self.current_database or (
                    self.cli.args.db_path if hasattr(self.cli.args, "db_path") else ""
                )
                database_name = Prompt.ask("[bold]Enter database name[/]", default=default_db)
                # Output directory - optional input
                output_dir = Prompt.ask("[bold]Enter output directory[/]", default="output")
                # External knowledge - optional input
                external_knowledge = Prompt.ask("[bold]Enter external knowledge (optional)[/]", default="")

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
            self.console.print(f"[dim]Next node: {workflow.get_current_node().type}[/]")

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
            top_n = Prompt.ask("[bold]Enter number of tables to link[/]", default="5")
            input.top_n = int(top_n.strip())
            matching_rate = Prompt.ask(
                "[bold]Enter matching method[/]",
                choices=["fast", "medium", "slow", "from_llm"],
                default="fast",
            )
            input.matching_rate = matching_rate.strip()
            database_name = Prompt.ask("[bold]Enter database name[/]", default=input.database_name)
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

                sql_context_id = Prompt.ask(
                    "[bold]Enter SQL context ID[/]", default=len(self.workflow.context.sql_contexts)
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
                    choice = Prompt.ask(
                        "[bold yellow]Do you want to execute this node? yes/no/edit[/]",
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
        if not self.agent.workflow:
            self.console.print("[bold yellow]Warning:[/] No active workflow. Starting a new one.")
            self.cmd_dastart()

        # get lastest sql context
        if context_type == "sql":
            if not self.agent.workflow.context.sql_contexts:
                self.console.print("[yellow]No SQL contexts available[/]")
                return

            # Display all SQL contexts
            self.console.print("[bold blue]SQL Contexts:[/]")
            for i, sql_context in enumerate(self.agent.workflow.context.sql_contexts):
                self.console.print(f"\n[bold]Context {i + 1}:[/]")
                self.console.print(sql_context.to_dict())

            confirmed = Confirm.ask("[bold yellow]Do you want to update the context?[/]", default=True)
            if not confirmed:
                self.console.print("[yellow]Operation cancelled by user[/]")
                return

            # update context
        elif context_type == "schema":
            self.console.print("[bold blue]Table Context:[/]")
            for i, table_schema in enumerate(self.agent.workflow.context.table_schemas):
                self.console.print(f"{i + 1}. {table_schema.to_dict()}")

            tables = Prompt.ask("[bold]Enter table names you want to update the context for[/]", default="")
            if not tables.strip():
                self.console.print("[bold red]Error:[/] Table names are required")
                return

            schema_tool = SchemaLineageTool(agent_config=self.agent.global_config)
            table_schemas, table_values = schema_tool.get_table_and_values(
                self.agent.workflow.task.database_name, tables.strip().split(",")
            )
            self.agent.workflow.context.update_schema_and_values(table_schemas, table_values)
            self.console.print(f"[bold green]Table context updated to {tables}[/]")

        elif context_type == "metrics":
            pass
        else:
            self.console.print("[bold red]Error:[/] Invalid context type")
            return

        self.console.print("[yellow]Feature not implemented in MVP.[/]")

    def cmd_compare(self, args: str):
        pass
