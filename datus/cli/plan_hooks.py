"""Plan mode hooks implementation for intercepting agent execution flow."""

import asyncio
import time

from agents import SQLiteSession
from agents.lifecycle import AgentHooks
from langsmith import traceable
from rich.console import Console

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PlanningPhaseException(Exception):
    """Exception raised when trying to execute tools during planning phase."""

    pass


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution"""

    pass


@traceable(name="PlanModeHooks", run_type="chain")
class PlanModeHooks(AgentHooks):
    """Plan Mode hooks for workflow management"""

    def __init__(self, console: Console, session: SQLiteSession, plan_message: str):
        self.console = console
        self.session = session
        self.plan_message = plan_message
        from datus.tools.plan_tools import SessionTodoStorage

        self.todo_storage = SessionTodoStorage(session)
        self.plan_phase = "generating"
        self.execution_mode = "manual"
        self.current_step = 0
        self.replan_feedback = ""
        self.planning_completed = False
        self.should_continue_execution = True
        self._user_choice_in_progress = False
        self._state_transitions = []

    async def on_agent_start(self, context, agent) -> None:
        logger.info(f"Plan mode agent start: phase={self.plan_phase}")

    async def on_start(self, context, agent) -> None:
        logger.info(f"Plan mode start: phase={self.plan_phase}")

    @traceable(name="on_tool_start", run_type="chain")
    async def on_tool_start(self, context, agent, tool) -> None:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        logger.info(f"Plan mode tool start: {tool_name}, phase: {self.plan_phase}, mode: {self.execution_mode}")

        # if self.plan_phase == "generating":
        #     if tool_name in ["todo_write", "todo_read"]:
        #         logger.info(f"Allowing plan tool: {tool_name}")
        #         return
        #     else:
        #         logger.warning(f"Blocking execution tool during planning: {tool_name}")
        #         raise PlanningPhaseException("Please complete the plan generation first")

        # elif self.plan_phase == "executing":
        #     if tool_name == "todo_write":
        #         return
        #     elif tool_name == "todo_update_pending":
        #         await self._handle_execution_step(tool_name)
        #     elif tool_name in ["todo_update_completed", "todo_update_failed", "todo_read"]:
        #         return
        if tool_name == "todo_update_pending":
            logger.info(f"Plan mode tool start: {tool_name}, phase: {self.plan_phase}, mode: {self.execution_mode}")
            await self._handle_execution_step(tool_name)

    @traceable(name="on_tool_end", run_type="chain")
    async def on_tool_end(self, context, agent, tool, result) -> None:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        logger.info(f"Plan mode tool end: {tool_name}, phase: {self.plan_phase}, result_type: {type(result)}")

        if tool_name == "todo_write":
            logger.info("Plan generation completed, transitioning to confirmation")
            await self._on_plan_generated()
        # elif tool_name.startswith("todo_update") and self.plan_phase == "executing":
        #     logger.info(f"Execution step completed: {tool_name}")
        #     await self._on_execution_step_completed(tool_name, result)

    async def on_handoff(self, context, agent, source) -> None:
        pass

    async def on_agent_end(self, context, agent, output) -> None:
        pass
        # logger.info(f"Plan mode agent end: phase={self.plan_phase}")
        # if self.plan_phase == "generating":
        #     todo_list = await self.todo_storage.get_todo_list()
        #     if not todo_list or len(todo_list.items) == 0:
        #         self.console.print("[red]❌ No plan generated[/]")
        #         self.console.print("[yellow]The agent completed without creating a todo list.[/]")

    async def on_end(self, context, agent, output) -> None:
        logger.info(f"Plan mode end: phase={self.plan_phase}")

    @traceable(name="on_error", run_type="chain")
    async def on_error(self, context, agent, error) -> None:
        pass
        # error_info = {
        #     "error_type": type(error).__name__,
        #     "error_message": str(error)[:200],
        #     "plan_phase": self.plan_phase,
        #     "execution_mode": self.execution_mode,
        #     "current_step": self.current_step,
        # }
        # logger.error(f"Plan mode error: {error_info}")

        # if self.execution_mode == "manual":
        #     self.console.print(f"[red]Error occurred: {str(error)[:100]}[/]")
        #     try:
        #         loop = asyncio.get_event_loop()
        #         response = await loop.run_in_executor(
        #             None, lambda: input("Continue despite error? (y/n) [y]: ").strip().lower() or "y"
        #         )
        #         if response != "y":
        #             self.should_continue_execution = False
        #             self._transition_state("cancelled", {"reason": "error_cancelled", "error": str(error)[:100]})
        #             self.console.print("[yellow]Cancelled due to error[/]")
        #     except (KeyboardInterrupt, EOFError):
        #         self.should_continue_execution = False
        #         self._transition_state("cancelled", {"reason": "error_interrupted"})
        #         self.console.print("[yellow]Cancelled[/]")

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

    def get_trace_summary(self) -> dict:
        return {
            "current_phase": self.plan_phase,
            "execution_mode": self.execution_mode,
            "current_step": self.current_step,
            "state_transitions": self._state_transitions,
            "replan_requested": self.replan_requested,
            "planning_completed": self.planning_completed,
            "should_continue_execution": self.should_continue_execution,
            "total_transitions": len(self._state_transitions),
        }

    @traceable(name="_on_plan_generated", run_type="chain")
    async def _on_plan_generated(self):
        todo_list = await self.todo_storage.get_todo_list()
        logger.info(f"Plan generation - todo_list: {todo_list.model_dump() if todo_list else None}")
        self._transition_state("confirming", {"todo_count": len(todo_list.items) if todo_list else 0})
        if not todo_list:
            self.console.print("[red]❌ No plan generated[/]")
            return

        self.console.print("\n[bold green]✅ Plan Generated Successfully![/]")
        self.console.print("[bold cyan]Execution Plan:[/]")

        for i, item in enumerate(todo_list.items, 1):
            self.console.print(f"  {i}. {item.content}")

        try:
            await self._get_user_confirmation()
        except PlanningPhaseException:
            # Re-raise to be handled by chat_agentic_node.py
            raise

    @traceable(name="_get_user_confirmation", run_type="chain")
    async def _get_user_confirmation(self):
        import asyncio
        import sys

        try:
            await asyncio.sleep(0.2)
            sys.stdout.flush()
            sys.stderr.flush()

            self.console.print("\n" + "=" * 50)
            self.console.print("\n[bold cyan]CHOOSE EXECUTION MODE:[/]")
            self.console.print("")
            self.console.print("  1. Step-by-Step - Confirm each step")
            self.console.print("  2. Auto Execute - Run all steps automatically")
            self.console.print("  3. Revise - Provide feedback and regenerate plan")
            self.console.print("  4. Cancel")
            self.console.print("")

            loop = asyncio.get_event_loop()
            choice = await loop.run_in_executor(None, lambda: input("Your choice (1-4) [1]: ").strip() or "1")

            if choice == "1":
                self.execution_mode = "manual"
                self._transition_state("executing", {"mode": "manual"})
                self.console.print("[green]Step-by-step mode selected[/]")
            elif choice == "2":
                self.execution_mode = "auto"
                self._transition_state("executing", {"mode": "auto"})
                self.console.print("[green]Auto execution mode selected[/]")
            elif choice == "3":
                await self._handle_replan()
                raise PlanningPhaseException(f"REPLAN_REQUIRED: Use todo_write with feedback: {self.replan_feedback}")
            elif choice == "4":
                self._transition_state("cancelled", {})
                self.console.print("[yellow]Plan cancelled[/]")
            else:
                self.console.print("[red]Invalid choice, please try again[/]")
                await self._get_user_confirmation()

        except (KeyboardInterrupt, EOFError):
            self._transition_state("cancelled", {"reason": "keyboard_interrupt"})
            self.console.print("\n[yellow]Plan cancelled[/]")

    @traceable(name="_handle_replan", run_type="chain")
    async def _handle_replan(self):
        try:
            loop = asyncio.get_event_loop()
            feedback = await loop.run_in_executor(None, lambda: input("\nFeedback for replanning: ").strip())
            if feedback:
                todo_list = await self.todo_storage.get_todo_list()
                completed_items = [item for item in todo_list.items if item.status == "completed"] if todo_list else []

                if completed_items:
                    self.console.print(f"[blue]Found {len(completed_items)} completed steps[/]")

                self.console.print(f"[green]Replanning with feedback: {feedback}[/]")
                self.replan_feedback = feedback
            else:
                self.console.print("[yellow]No feedback provided[/]")
                if self.plan_phase == "confirming":
                    await self._get_user_confirmation()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Replan cancelled[/]")

    @traceable(name="_handle_execution_step", run_type="chain")
    async def _handle_execution_step(self, _tool_name: str):
        import asyncio
        import sys

        logger.info(f"PlanHooks: _handle_execution_step called with tool: {_tool_name}")

        todo_list = await self.todo_storage.get_todo_list()
        logger.info(f"PlanHooks: Retrieved todo list with {len(todo_list.items) if todo_list else 0} items")

        if not todo_list:
            logger.warning("PlanHooks: No todo list found!")
            return

        pending_items = [item for item in todo_list.items if item.status == "pending"]
        logger.info(f"PlanHooks: Found {len(pending_items)} pending items")

        if not pending_items:
            return

        current_item = pending_items[0]

        await asyncio.sleep(0.2)
        sys.stdout.flush()
        sys.stderr.flush()
        self.console.print("\n" + "-" * 40)

        try:
            if self.execution_mode == "auto":
                self.console.print(f"\n[bold cyan]Auto Mode:[/] {current_item.content}")
                loop = asyncio.get_event_loop()
                choice = await loop.run_in_executor(None, lambda: input("Execute? (y/n) [y]: ").strip().lower() or "y")

                if choice in ["y", "yes"]:
                    self.console.print("[green]Executing...[/]")
                    return
                elif choice in ["cancel", "c", "n", "no"]:
                    self.console.print("[yellow]Execution cancelled[/]")
                    self.plan_phase = "cancelled"
                    raise UserCancelledException("Execution cancelled by user")
            else:
                self.console.print(f"\n[bold cyan]Next step:[/] {current_item.content}")
                self.console.print("Options:")
                self.console.print("  1. Execute this step")
                self.console.print("  2. Execute this step and continue automatically")
                self.console.print("  3. Revise remaining plan")
                self.console.print("  4. Cancel")

                while True:
                    loop = asyncio.get_event_loop()
                    choice = await loop.run_in_executor(None, lambda: input("\nYour choice (1-4) [1]: ").strip() or "1")

                    if choice == "1":
                        self.console.print("[green]Executing step...[/]")
                        return
                    elif choice == "2":
                        self.execution_mode = "auto"
                        self.console.print("[green]Switching to auto mode...[/]")
                        return
                    elif choice == "3":
                        await self._handle_replan()
                        raise PlanningPhaseException(
                            f"REPLAN_REQUIRED: Use todo_write with feedback: {self.replan_feedback}"
                        )
                    elif choice == "4":
                        self._transition_state("cancelled", {"step": current_item.content, "user_choice": choice})
                        self.console.print("[yellow]Execution cancelled[/]")
                        return
                    else:
                        self.console.print(f"[red]Invalid choice '{choice}'. Please enter 1, 2, 3, or 4.[/]")

        except (KeyboardInterrupt, EOFError):
            self._transition_state("cancelled", {"reason": "execution_interrupted"})
            self.console.print("\n[yellow]Execution cancelled[/]")

    # @traceable(name="_on_execution_step_completed", run_type="chain")
    # async def _on_execution_step_completed(self, _tool_name: str, result):
    #     if isinstance(result, dict):
    #         success = result.get("success", 0) == 1
    #         result_data = result.get("result", {})
    #     else:
    #         success = result.success == 1
    #         result_data = result.result

    #     if success and result_data:
    #         message = result_data.get("message", "Step completed")
    #         self.console.print(f"[green]✅ {message}[/]")

    #         todo_list = await self.todo_storage.get_todo_list()
    #         if todo_list:
    #             pending_items = [item for item in todo_list.items if item.status == "pending"]
    #             if not pending_items:
    #                 self._transition_state("completed", {"total_steps": len(todo_list.items)})
    #                 self.console.print("\n[bold green]🎉 All tasks completed successfully![/]")

    def get_plan_tools(self):
        from datus.tools.plan_tools import PlanTool

        plan_tool = PlanTool(self.session)
        plan_tool.storage = self.todo_storage
        return plan_tool.available_tools()
