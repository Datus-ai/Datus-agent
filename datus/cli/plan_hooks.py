"""
Plan mode hooks implementation for intercepting agent execution flow.
"""

import asyncio
from enum import Enum

from agents import SQLiteSession
from agents.lifecycle import AgentHooks
from rich.console import Console

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class UserChoice(Enum):
    """User choice options during manual task confirmation"""

    CONTINUE = "continue"
    CANCEL = "cancel"
    REPLAN = "replan"


# Plan mode specific exceptions
class PlanningPhaseException(Exception):
    """Exception raised when trying to execute tools during planning phase."""

    pass


class SkipStepException(Exception):
    """Exception raised when user wants to skip a step."""

    pass


class AskAgainException(Exception):
    """Exception raised when user needs to be asked again."""

    pass


class PlanningCompletedException(Exception):
    """Exception raised when planning phase is completed and execution should begin"""

    pass


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution"""

    pass


class ReplanRequestedException(Exception):
    """Exception raised when user requests replanning with feedback"""

    def __init__(self, feedback: str = ""):
        self.feedback = feedback
        super().__init__(f"User requested replanning with feedback: {feedback}")


class PlanModeHooks(AgentHooks):
    """Enhanced Plan Mode hooks for complete workflow management"""

    def __init__(self, console: Console, session: SQLiteSession, plan_message: str):
        self.console = console
        self.session = session
        self.plan_message = plan_message
        from datus.tools.plan_tools import SessionTodoStorage

        self.todo_storage = SessionTodoStorage(session)
        self.plan_phase = "generating"  # generating, confirming, executing, completed
        self.execution_mode = "manual"  # manual, auto
        self.current_step = 0
        self.replan_requested = False
        self.replan_feedback = ""
        self.planning_completed = False
        self.should_continue_execution = True
        self._user_choice_in_progress = False

    async def on_tool_start(self, context, agent, tool) -> None:
        """Called before tool execution - control plan mode workflow"""
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        # Check if replan was requested
        if self.replan_requested:
            self.console.print("[yellow]🔄 Replanning requested, stopping current execution...[/]")
            raise ReplanRequestedException(self.replan_feedback)

        if self.plan_phase == "generating":
            # Generation phase: only allow plan-related tools
            if tool_name in ["todo_write", "todo_read"]:
                return  # Allow plan tools
            else:
                # Block execution tools, wait for plan completion
                self.console.print(f"[yellow]⏸️  Waiting for plan completion... ({tool_name} blocked)[/]")
                raise PlanningPhaseException("Please complete the plan generation first")

        elif self.plan_phase == "executing":
            # Execution phase: control execution flow
            if tool_name == "todo_write":
                return  # Allow replanning
            elif tool_name.startswith("todo_update"):
                await self._handle_execution_step(tool_name)
            elif tool_name == "todo_read":
                return  # Allow reading plan status

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Called after tool execution - handle plan progression"""
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if tool_name == "todo_write" and self.plan_phase == "generating":
            # Plan generation completed
            await self._on_plan_generated()
        elif tool_name.startswith("todo_update") and self.plan_phase == "executing":
            # Execution step completed
            await self._on_execution_step_completed(tool_name, result)

    async def on_error(self, context, agent, error) -> None:
        """Called when there's an error during execution"""
        _ = context, agent  # Mark parameters as used to avoid warnings
        if self.execution_mode == "manual":
            self.console.print(f"[red]Error occurred: {str(error)[:100]}[/]")

            # In manual mode, ask user if they want to continue despite the error
            try:
                response = input("Continue despite error? (y/n) [y]: ").strip().lower() or "y"
                if response != "y":
                    self.should_continue_execution = False
                    self.console.print("[yellow]Cancelled due to error[/]")
            except (KeyboardInterrupt, EOFError):
                self.should_continue_execution = False
                self.console.print("[yellow]Cancelled[/]")

    async def _handle_manual_confirmation(self, task_description: str) -> UserChoice:
        """Handle manual task confirmation with replan option"""
        try:
            import asyncio
            import sys

            # Pause Action Stream display during manual confirmation
            streaming_context = None
            if hasattr(self, "repl_executor") and self.repl_executor:
                if hasattr(self.repl_executor, "chat_commands") and self.repl_executor.chat_commands:
                    streaming_context = self.repl_executor.chat_commands.current_streaming_context
                    if streaming_context:
                        streaming_context.pause_display()

            # Brief pause to let streams settle
            await asyncio.sleep(0.2)

            # Force flush all output streams before prompting
            sys.stdout.flush()
            sys.stderr.flush()

            # Show current todo list status
            self.console.print("\n[bold cyan]Current Todo List Status:[/]")
            todo_list = None

            if self._get_todo_storage():
                todo_list = self._get_todo_storage().get_todo_list()
            if todo_list:
                for i, item in enumerate(todo_list.items, 1):
                    status_icon = "✓" if item.status == "completed" else "○" if item.status == "pending" else "✗"
                    status_color = (
                        "green" if item.status == "completed" else "yellow" if item.status == "pending" else "red"
                    )
                    self.console.print(f"  {i}. [{status_color}]{status_icon}[/{status_color}] {item.content}")
            else:
                self.console.print("  [yellow]No todo list found[/]")

            # Show task confirmation options
            self.console.print(f"\n[bold]Ready to start task:[/] {task_description}")
            self.console.print("")
            self.console.print("  [bold]1. Continue[/] - Execute this task")
            self.console.print("  [bold]2. Cancel[/] - Cancel execution")
            self.console.print("  [bold]3. Replan[/] - Provide feedback and revise plan")
            self.console.print("  [bold]4. Auto[/] - Switch to auto mode")

            loop = asyncio.get_event_loop()

            def get_user_choice():
                while True:
                    try:
                        user_input = input("\nYour choice (1-4) [1]: ").strip() or "1"
                        if user_input in ["1", "2", "3", "4"]:
                            return user_input
                        print("Please enter a valid choice (1-4)")
                    except (KeyboardInterrupt, EOFError):
                        return "interrupt"

            response = await loop.run_in_executor(None, get_user_choice)

            # Handle user choice
            if response == "1":
                choice = UserChoice.CONTINUE
            elif response == "2":
                self.console.print("[yellow]Cancelling execution[/]")
                choice = UserChoice.CANCEL
            elif response == "3":
                # Get feedback for replanning
                def get_feedback():
                    try:
                        return input("Feedback for replanning: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        return ""

                feedback = await loop.run_in_executor(None, get_feedback)
                if feedback:
                    self._replan_feedback = feedback
                    self.console.print(f"[blue]Will replan with your feedback: '{feedback}'[/]")
                    # Directly raise the exception with feedback
                    raise ReplanRequestedException(feedback)
                else:
                    self.console.print("[yellow]No feedback provided, stopping instead[/]")
                    choice = UserChoice.CANCEL
            elif response == "4":
                self.console.print("[yellow]Switching to auto mode[/]")
                self.execution_mode = "auto"
                choice = UserChoice.CONTINUE
            else:  # interrupt
                self.console.print("[red]Interrupted[/]")
                choice = UserChoice.CANCEL

            # Resume Action Stream display after manual confirmation
            if streaming_context:
                streaming_context.resume_display()

            return choice

        except ReplanRequestedException:
            # Let ReplanRequestedException propagate up
            raise
        except Exception as e:
            logger.error(f"Manual task confirmation error: {str(e)}")
            return UserChoice.CONTINUE

    async def clear_todo_storage(self):
        """Clear todo storage - useful for session cleanup"""
        if self.todo_storage:
            await self.todo_storage.clear_all()

    async def _on_plan_generated(self):
        """Handle plan generation completion."""
        self.plan_phase = "confirming"

        # Display generated plan
        todo_list = await self.todo_storage.get_todo_list()
        if not todo_list:
            self.console.print("[red]❌ No plan generated[/]")
            return

        self.console.print("\n[bold green]✅ Plan Generated Successfully![/]")
        self.console.print("[bold cyan]📋 Execution Plan:[/]")

        for i, item in enumerate(todo_list.items, 1):
            status_icon = "📋" if item.status == "pending" else "✅" if item.status == "completed" else "❌"
            self.console.print(f"  {i}. {status_icon} {item.content}")

        # Get user confirmation
        await self._get_user_confirmation()

    async def _get_user_confirmation(self):
        """Get user confirmation for plan execution."""
        self.console.print("\n[bold cyan]🤔 Choose execution mode:[/]")
        self.console.print("  1. 🔄 Auto Execute - Run all steps automatically")
        self.console.print("  2. 👣 Step-by-Step - Confirm each step")
        self.console.print("  3. ✏️  Revise - Provide feedback and regenerate plan")
        self.console.print("  4. ❌ Cancel")

        try:
            loop = asyncio.get_event_loop()

            def get_choice():
                choice = input("\nYour choice (1-4): ").strip()
                return choice

            choice = await loop.run_in_executor(None, get_choice)

            if choice == "1":
                self.execution_mode = "auto"
                self.plan_phase = "executing"
                self.console.print("[green]🚀 Auto execution mode selected[/]")
            elif choice == "2":
                self.execution_mode = "manual"
                self.plan_phase = "executing"
                self.console.print("[blue]👣 Step-by-step mode selected[/]")
            elif choice == "3":
                await self._handle_replan_request()
            elif choice == "4":
                self.plan_phase = "cancelled"
                self.console.print("[yellow]❌ Plan cancelled[/]")
            else:
                self.console.print("[red]Invalid choice, please try again[/]")
                await self._get_user_confirmation()

        except (KeyboardInterrupt, EOFError):
            self.plan_phase = "cancelled"
            self.console.print("\n[yellow]❌ Plan cancelled[/]")

    async def _handle_replan_request(self):
        """Handle replan request."""
        try:
            feedback = input("\n💭 Feedback for replanning: ").strip()
            if feedback:
                self.replan_feedback = feedback
                self.replan_requested = True
                self.console.print(f"[blue]🔄 Will replan with feedback: {feedback}[/]")
                # Trigger replanning
                raise ReplanRequestedException(feedback)
            else:
                self.console.print("[yellow]No feedback provided, showing options again[/]")
                await self._get_user_confirmation()
        except (KeyboardInterrupt, EOFError):
            self.plan_phase = "cancelled"

    async def _handle_execution_step(self, _tool_name: str):
        """Handle individual execution step in manual mode."""
        _ = _tool_name  # Mark as used
        if self.execution_mode == "manual":
            # Get current pending items
            todo_list = await self.todo_storage.get_todo_list()
            pending_items = [item for item in todo_list.items if item.status == "pending"]

            if pending_items:
                current_item = pending_items[0]
                self.console.print(f"\n[bold cyan]🎯 Next step:[/] {current_item.content}")
                self.console.print("Options:")
                self.console.print("  1. ✅ Execute this step")
                self.console.print("  2. 🚀 Execute this step and continue automatically")
                self.console.print("  3. ⏭️  Skip this step")
                self.console.print("  4. ✏️  Revise remaining plan")
                self.console.print("  5. ❌ Cancel")

                try:
                    loop = asyncio.get_event_loop()

                    def get_step_choice():
                        choice = input("\nYour choice (1-5): ").strip()
                        return choice

                    choice = await loop.run_in_executor(None, get_step_choice)

                    if choice == "1":
                        self.console.print("[green]✅ Executing step...[/]")
                        return  # Allow execution
                    elif choice == "2":
                        self.execution_mode = "auto"
                        self.console.print("[green]🚀 Switching to auto mode...[/]")
                        return  # Allow execution and continue
                    elif choice == "3":
                        # Skip this step
                        self.console.print("[yellow]⏭️  Skipping step...[/]")
                        raise SkipStepException("Step skipped by user")
                    elif choice == "4":
                        # Replan remaining parts
                        await self._handle_partial_replan()
                    elif choice == "5":
                        self.plan_phase = "cancelled"
                        self.console.print("[yellow]❌ Execution cancelled[/]")
                    else:
                        self.console.print("[red]Invalid choice, please try again[/]")
                        # Ask again
                        raise AskAgainException("Invalid choice")

                except (KeyboardInterrupt, EOFError):
                    self.plan_phase = "cancelled"
                    self.console.print("\n[yellow]❌ Execution cancelled[/]")

    async def _handle_partial_replan(self):
        """Handle partial replan from current step."""
        try:
            feedback = input("\n💭 Feedback for revising remaining plan: ").strip()
            if feedback:
                # Mark current step as completed, regenerate remaining steps
                todo_list = await self.todo_storage.get_todo_list()
                completed_items = [item for item in todo_list.items if item.status == "completed"]

                if completed_items:
                    self.console.print(f"[blue]📝 Keeping {len(completed_items)} completed steps[/]")

                self.replan_feedback = feedback
                self.replan_requested = True
                raise ReplanRequestedException(feedback)
            else:
                self.console.print("[yellow]No feedback provided[/]")

        except (KeyboardInterrupt, EOFError):
            self.plan_phase = "cancelled"

    async def _on_execution_step_completed(self, _tool_name: str, result):
        """Handle completion of execution step."""
        _ = _tool_name  # Mark as used
        if result.success and result.result:
            # Display execution result
            message = result.result.get("message", "Step completed")
            self.console.print(f"[green]✅ {message}[/]")

            # Check if all steps are completed
            todo_list = await self.todo_storage.get_todo_list()
            pending_items = [item for item in todo_list.items if item.status == "pending"]

            if not pending_items:
                self.plan_phase = "completed"
                self.console.print("\n[bold green]🎉 All tasks completed successfully![/]")

    def get_plan_tools(self):
        """Get plan-specific tools for the agent."""
        from datus.tools.plan_tools import PlanTool

        plan_tool = PlanTool(self.session)
        return plan_tool.available_tools()
