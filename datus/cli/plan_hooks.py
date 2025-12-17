# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Plan mode hooks implementation for intercepting agent execution flow."""

import asyncio
import time

from agents import SQLiteSession
from agents.lifecycle import AgentHooks
from rich.console import Console

from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PlanningPhaseException(Exception):
    """Exception raised when trying to execute tools during planning phase."""


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution"""


class PlanModeHooks(AgentHooks):
    """Plan Mode hooks for workflow management"""

    def __init__(self, console: Console, session: SQLiteSession, auto_mode: bool = False):
        self.console = console
        self.session = session
        self.auto_mode = auto_mode
        from datus.tools.func_tool.plan_tools import SessionTodoStorage

        self.todo_storage = SessionTodoStorage(session)
        self.plan_phase = "generating"
        self.execution_mode = "auto" if auto_mode else "manual"
        self.replan_feedback = ""
        self._state_transitions = []
        self._plan_generated_pending = False  # Flag to defer plan display until LLM ends

    async def on_start(self, context, agent) -> None:
        logger.debug(f"Plan mode start: phase={self.plan_phase}")

    async def on_tool_start(self, context, agent, tool) -> None:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        logger.debug(f"Plan mode tool start: {tool_name}, phase: {self.plan_phase}, mode: {self.execution_mode}")

        if tool_name == "todo_update" and self.execution_mode == "manual" and self.plan_phase == "executing":
            # Check if this is updating to pending status
            if self._is_pending_update(context):
                await self._handle_execution_step(tool_name)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if tool_name == "todo_write":
            logger.info("Plan generation completed, will show plan after LLM finishes current turn")
            # Set flag instead of immediately showing plan
            # This allows any remaining "Thinking" messages to be generated first
            self._plan_generated_pending = True

    async def on_llm_end(self, context, agent, response) -> None:
        """Called when LLM finishes a turn - perfect time to show plan after all thinking is done"""
        if self._plan_generated_pending and self.plan_phase == "generating":
            self._plan_generated_pending = False
            await self._on_plan_generated()

    async def on_end(self, context, agent, output) -> None:
        logger.info(f"Plan mode end: phase={self.plan_phase}")

    def _transition_state(self, new_state: str, context: dict = None):
        old_state = self.plan_phase
        self.plan_phase = new_state

        transition_data = {
            "from_state": old_state,
            "to_state": new_state,
            "context": context or {},
            "timestamp": time.time(),
        }

        self._state_transitions.append(transition_data)
        logger.info(f"Plan mode state transition: {old_state} -> {new_state}")
        return transition_data

    async def _on_plan_generated(self):
        todo_list = self.todo_storage.get_todo_list()
        logger.info(f"Plan generation - todo_list: {todo_list.model_dump() if todo_list else None}")

        # Clear replan feedback BEFORE transitioning state to ensure prompt updates correctly
        self.replan_feedback = ""
        self._transition_state("confirming", {"todo_count": len(todo_list.items) if todo_list else 0})

        if not todo_list:
            self.console.print("[red]No plan generated[/]")
            return

        # Stop live display BEFORE showing the plan (keep registered for restart)
        # At this point, LLM has finished its turn, so all thinking/tool messages are already displayed
        execution_controller.stop_live_display()
        await asyncio.sleep(0.3)

        self.console.print("[bold green]Plan Generated Successfully![/]")
        self.console.print("[bold cyan]Execution Plan:[/]")

        for i, item in enumerate(todo_list.items, 1):
            self.console.print(f"  {i}. {item.content}")

        # Auto mode: skip user confirmation
        if self.auto_mode:
            self.execution_mode = "auto"
            self._transition_state("executing", {"mode": "auto"})
            self.console.print("[green]Auto execution mode (workflow/benchmark context)[/]")
            return

        # Interactive mode: ask for user confirmation
        try:
            await self._get_user_confirmation()
        except PlanningPhaseException:
            # Re-raise to be handled by chat_agentic_node.py
            raise

    async def _get_user_confirmation(self):
        import asyncio
        import sys

        try:
            sys.stdout.flush()
            sys.stderr.flush()

            self.console.print("\n" + "=" * 50)
            self.console.print("\n[bold cyan]CHOOSE EXECUTION MODE:[/]")
            self.console.print("")
            self.console.print("  1. Manual Confirm - Confirm each step")
            self.console.print("  2. Auto Execute - Run all steps automatically")
            self.console.print("  3. Revise - Provide feedback and regenerate plan")
            self.console.print("  4. Cancel")
            self.console.print("")

            # Pause execution while getting user input (live display already stopped by caller)
            async with execution_controller.pause_execution():
                # Small delay for console stability after flushing
                await asyncio.sleep(0.2)

                # Get input using blocking_input_manager
                def get_user_input():
                    return blocking_input_manager.get_blocking_input(
                        lambda: input("Your choice (1-4) [1]: ").strip() or "1"
                    )

                choice = await execution_controller.request_user_input(get_user_input)

            if choice == "1":
                self.execution_mode = "manual"
                self._transition_state("executing", {"mode": "manual"})
                self.console.print("[green]Manual confirmation mode selected[/]")
                # Recreate live display from current cursor position (brand new display)
                execution_controller.recreate_live_display()
                return
            elif choice == "2":
                self.execution_mode = "auto"
                self._transition_state("executing", {"mode": "auto"})
                self.console.print("[green]Auto execution mode selected[/]")
                # Recreate live display from current cursor position (brand new display)
                execution_controller.recreate_live_display()
                return
            elif choice == "3":
                await self._handle_replan()
                # Recreate live display for regeneration phase
                execution_controller.recreate_live_display()
                raise PlanningPhaseException(f"REPLAN_REQUIRED: Revise the plan with feedback: {self.replan_feedback}")
            elif choice == "4":
                self._transition_state("cancelled", {})
                self.console.print("[yellow]Plan cancelled[/]")
                raise UserCancelledException("User cancelled plan execution")
            else:
                self.console.print("[red]Invalid choice, please try again[/]")
                await self._get_user_confirmation()

        except (KeyboardInterrupt, EOFError):
            self._transition_state("cancelled", {"reason": "keyboard_interrupt"})
            self.console.print("\n[yellow]Plan cancelled[/]")

    async def _handle_replan(self):
        try:
            # Stop live display before prompting (keep registered for restart)
            execution_controller.stop_live_display()

            async with execution_controller.pause_execution():
                await asyncio.sleep(0.1)

                self.console.print("\n[bold yellow]Provide feedback for replanning:[/]")

                def get_user_input():
                    return blocking_input_manager.get_blocking_input(lambda: input("> ").strip())

                feedback = await execution_controller.request_user_input(get_user_input)
            if feedback:
                todo_list = self.todo_storage.get_todo_list()
                completed_items = [item for item in todo_list.items if item.status == "completed"] if todo_list else []

                if completed_items:
                    self.console.print(f"[blue]Found {len(completed_items)} completed steps[/]")

                self.console.print(f"[green]Replanning with feedback: {feedback}[/]")
                self.replan_feedback = feedback
                # Transition back to generating phase for replan
                self._transition_state("generating", {"replan_triggered": True, "feedback": feedback})
            else:
                self.console.print("[yellow]No feedback provided[/]")
                if self.plan_phase == "confirming":
                    await self._get_user_confirmation()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Replan cancelled[/]")

    async def _handle_execution_step(self, _tool_name: str):
        import asyncio
        import sys

        logger.info(f"PlanHooks: _handle_execution_step called with tool: {_tool_name}")

        # Auto mode: skip all step confirmations
        if self.auto_mode:
            logger.info("Auto mode enabled, executing step without confirmation")
            return

        todo_list = self.todo_storage.get_todo_list()
        logger.info(f"PlanHooks: Retrieved todo list with {len(todo_list.items) if todo_list else 0} items")

        if not todo_list:
            logger.warning("PlanHooks: No todo list found!")
            return

        pending_items = [item for item in todo_list.items if item.status == "pending"]
        logger.info(f"PlanHooks: Found {len(pending_items)} pending items")

        if not pending_items:
            return

        current_item = pending_items[0]

        # Stop live display BEFORE showing step progress (keep registered for restart)
        execution_controller.stop_live_display()

        await asyncio.sleep(0.2)
        sys.stdout.flush()
        sys.stderr.flush()

        # Print newlines to push content down and avoid overlap when resuming
        self.console.print("\n" * 2)
        self.console.print("-" * 40)

        try:
            if self.execution_mode == "auto":
                # Display full todo list with progress indicators in auto mode too
                self.console.print("\n[bold cyan]Plan Progress:[/]")
                for i, item in enumerate(todo_list.items, 1):
                    if item.status == "completed":
                        status_icon = "[green]✓[/]"
                        text_style = "[dim]"
                        close_tag = "[/]"
                    elif item.id == current_item.id:
                        status_icon = "[yellow]▶[/]"  # Current step
                        text_style = "[bold cyan]"
                        close_tag = "[/]"
                    else:
                        status_icon = "[white]○[/]"  # Pending
                        text_style = ""
                        close_tag = ""

                    self.console.print(f"  {status_icon} {text_style}{i}. {item.content}{close_tag}")

                self.console.print(f"\n[bold cyan]Auto Mode:[/] {current_item.content}")

                # Pause execution while getting user input (live display already stopped)
                async with execution_controller.pause_execution():
                    await asyncio.sleep(0.1)

                    def get_user_input():
                        return blocking_input_manager.get_blocking_input(
                            lambda: input("Execute? (y/n) [y]: ").strip().lower() or "y"
                        )

                    choice = await execution_controller.request_user_input(get_user_input)

                if choice in ["y", "yes"]:
                    self.console.print("[green]Executing...[/]")
                    # Recreate live display from current cursor position
                    execution_controller.recreate_live_display()
                    return
                elif choice in ["cancel", "c", "n", "no"]:
                    self.console.print("[yellow]Execution cancelled[/]")
                    self.plan_phase = "cancelled"
                    raise UserCancelledException("Execution cancelled by user")
            else:
                # Display full todo list with progress indicators
                self.console.print("\n[bold cyan]Plan Progress:[/]")
                for i, item in enumerate(todo_list.items, 1):
                    if item.status == "completed":
                        status_icon = "[green]✓[/]"
                        text_style = "[dim]"
                        close_tag = "[/]"
                    elif item.id == current_item.id:
                        status_icon = "[yellow]▶[/]"  # Current step
                        text_style = "[bold cyan]"
                        close_tag = "[/]"
                    else:
                        status_icon = "[white]○[/]"  # Pending
                        text_style = ""
                        close_tag = ""

                    self.console.print(f"  {status_icon} {text_style}{i}. {item.content}{close_tag}")

                self.console.print(f"\n[bold cyan]Next step:[/] {current_item.content}")
                self.console.print("Options:")
                self.console.print("  1. Execute this step")
                self.console.print("  2. Execute this step and continue automatically")
                self.console.print("  3. Revise remaining plan")
                self.console.print("  4. Cancel")

                while True:
                    # Pause execution while getting user input (live display already stopped)
                    async with execution_controller.pause_execution():
                        await asyncio.sleep(0.1)

                        def get_user_input():
                            return blocking_input_manager.get_blocking_input(
                                lambda: input("\nYour choice (1-4) [1]: ").strip() or "1"
                            )

                        choice = await execution_controller.request_user_input(get_user_input)

                    if choice == "1":
                        self.console.print("[green]Executing step...[/]")
                        # Recreate live display from current cursor position
                        execution_controller.recreate_live_display()
                        return
                    elif choice == "2":
                        self.execution_mode = "auto"
                        self.console.print("[green]Switching to auto mode...[/]")
                        # Recreate live display from current cursor position
                        execution_controller.recreate_live_display()
                        return
                    elif choice == "3":
                        await self._handle_replan()
                        # Recreate live display for regeneration phase
                        execution_controller.recreate_live_display()
                        raise PlanningPhaseException(
                            f"REPLAN_REQUIRED: Revise the plan with feedback: {self.replan_feedback}"
                        )
                    elif choice == "4":
                        self._transition_state("cancelled", {"step": current_item.content, "user_choice": choice})
                        self.console.print("[yellow]Execution cancelled[/]")
                        raise UserCancelledException("User cancelled execution")
                    else:
                        self.console.print(f"[red]Invalid choice '{choice}'. Please enter 1, 2, 3, or 4.[/]")

        except (KeyboardInterrupt, EOFError):
            self._transition_state("cancelled", {"reason": "execution_interrupted"})
            self.console.print("\n[yellow]Execution cancelled[/]")

    def _is_pending_update(self, context) -> bool:
        """
        Check if todo_update is being called with status='pending'.

        Args:
            context: ToolContext with tool_arguments field (JSON string)

        Returns:
            bool: True if this is a pending status update
        """
        try:
            import json

            if hasattr(context, "tool_arguments"):
                if context.tool_arguments:
                    tool_args = json.loads(context.tool_arguments)

                    # Check if status is 'pending'
                    if isinstance(tool_args, dict):
                        if tool_args.get("status") == "pending":
                            logger.debug(f"Detected pending status update with args: {tool_args}")
                            return True

            logger.debug("Not a pending status update")
            return False

        except Exception as e:
            logger.debug(f"Error checking tool arguments: {e}")
            return False

    def get_plan_tools(self):
        from datus.tools.func_tool.plan_tools import PlanTool

        plan_tool = PlanTool(self.session)
        plan_tool.storage = self.todo_storage
        return plan_tool.available_tools()
