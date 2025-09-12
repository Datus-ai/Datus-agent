"""
Plan Mode Manager for Datus CLI
Handles plan mode state, todo list management, and execution coordination
"""

from enum import Enum

from agents import AgentHooks
from rich.console import Console

from datus.tools.plan_tools import TodoStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PlanningCompletedException(Exception):
    """Exception raised when planning phase is completed and execution should begin"""

    pass


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution"""

    pass


class PlanModeState(Enum):
    """Simplified states for plan mode management"""

    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"


class UserChoice(Enum):
    """User choice options during manual task confirmation"""

    CONTINUE = "continue"
    CANCEL = "cancel"
    REPLAN = "replan"


class PlanModeHooksWithInterception(AgentHooks):
    """Hooks that intercept after planning phase to get user choice"""

    def __init__(self, console: Console, plan_manager, user_choice_func, repl_executor=None):
        self.console = console
        self.plan_manager = plan_manager
        self.user_choice_func = user_choice_func
        self.repl_executor = repl_executor
        self.planning_completed = False
        self.execution_mode = "auto"
        self.should_continue_execution = True
        self.execution_ready = False
        self._user_choice_in_progress = False
        self._replan_requested = False
        self._replan_feedback = ""

    async def on_tool_start(self, context, agent, tool) -> None:
        """Called before executing any tool"""
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if (
            self.planning_completed
            and self.execution_mode == "manual"
            and tool_name == "todo_update_pending"
            and self.should_continue_execution
        ):
            try:
                user_choice = await self._handle_manual_task_confirmation_async("next planned task")

                if user_choice == UserChoice.CANCEL:
                    self.should_continue_execution = False
                    raise UserCancelledException("Execution cancelled by user")
                elif user_choice == UserChoice.REPLAN:
                    self.should_continue_execution = False
                    self._replan_requested = True
                    raise PlanningCompletedException("User requested replanning during task execution")

            except (PlanningCompletedException, UserCancelledException):
                raise
            except Exception as e:
                logger.error(f"Manual confirmation failed for {tool_name}: {str(e)}")

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Called after tool execution - check if planning is complete"""
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if tool_name == "todo_write" and not self.planning_completed and not self._user_choice_in_progress:
            self.console.print("[green]Plan created[/]")
            self.planning_completed = True
            self.execution_ready = True

        elif self.planning_completed and self.execution_mode == "manual" and tool_name == "todo_update_pending":
            try:
                if hasattr(result, "result") and result.result:
                    if hasattr(result, "success") and result.success:
                        task_info = result.result.get("message", "Task completed")
                        self.console.print(f"[green]{task_info}[/]")
                    else:
                        task_info = getattr(result, "error", "Task failed")
                        self.console.print(f"[red]{task_info}[/]")

                        # In manual mode, let user decide whether to continue after failure
                        try:
                            continue_response = input("Task failed. Continue? (y/n) [y]: ").strip().lower() or "y"
                            if continue_response != "y":
                                self.should_continue_execution = False
                                self.console.print("[yellow]Cancelled after failure[/]")
                        except (KeyboardInterrupt, EOFError):
                            self.should_continue_execution = False
            except Exception as e:
                logger.error(f"Error handling todo_update completion: {str(e)}")

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

    async def _handle_manual_task_confirmation_async(self, task_description: str) -> UserChoice:
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
            await asyncio.sleep(0.1)

            # Force flush all output streams before prompting
            sys.stdout.flush()
            sys.stderr.flush()

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
                        user_input = input("\nYour choice (1-4): ").strip()
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
                    self.console.print("[blue]Will replan with your feedback[/]")
                    choice = UserChoice.REPLAN
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

        except Exception as e:
            logger.error(f"Manual task confirmation error: {str(e)}")
            return UserChoice.CONTINUE


class PlanModeManager:
    """Manages Plan Mode state and execution flow"""

    def __init__(self, console: Console):
        self.console = console
        self.state = PlanModeState.INACTIVE
        self.repl_executor = None  # Will be set by REPL instance
        self.current_plan = None
        self._plan_tool = None  # Lazy initialization

    @property
    def plan_tool(self):
        """Lazy initialization of plan tool"""
        if self._plan_tool is None:
            from datus.tools.plan_tools import PlanTool

            self._plan_tool = PlanTool()
        return self._plan_tool

    def set_repl_executor(self, repl_executor):
        """Set the REPL executor for actual task execution"""
        self.repl_executor = repl_executor

    def _build_execution_instruction(self, database_name: str, message: str, mode: str = "auto") -> str:
        """Build execution instruction based on mode to avoid duplication"""
        base_instruction = f"""Database: {database_name}
Task: {message}

"""

        if mode == "manual":
            return (
                base_instruction
                + """**CRITICAL - MANUAL MODE ACTIVE**: You are in MANUAL CONFIRMATION MODE. """
                + """You MUST get approval before each step.

MANDATORY WORKFLOW - NO EXCEPTIONS:
1. Call todo_read() to get current todo list and their todo_id values
2. For EVERY single step in the todo list:
   a) Call todo_update_pending(todo_id) - THIS TRIGGERS MANUAL CONFIRMATION
   b) STOP and wait (you will be asked to confirm)
   c) Only after confirmation: execute the actual task
   d) Call todo_update_completed(todo_id)

IMPORTANT: Use the exact todo_id from the todo_read() result. Each todo item has an ID field that you must use.

CRITICAL: You MUST call todo_update_pending BEFORE doing ANY work on each step. This is not optional."""
            )
        else:
            return base_instruction + "Execute the planned steps to complete this task."

    def execute_unified_plan(self, message: str, database_name: str, chat_executor, user_choice_func) -> bool:
        """Execute unified plan with hook interception for user choice"""
        try:
            self.state = PlanModeState.ACTIVE
            self.plan_tool.storage.clear_all()

            # Phase 1: Planning - Generate plan with interception
            unified_instruction = self._get_unified_instruction_with_interception(database_name, message)
            hooks = PlanModeHooksWithInterception(
                console=self.console,
                plan_manager=self,
                user_choice_func=user_choice_func,
                repl_executor=self.repl_executor,
            )

            chat_executor(
                unified_instruction, hooks=hooks, show_details=False, plan_mode=True, shared_plan_tool=self.plan_tool
            )

            # Check planning result
            if not hooks.execution_ready:
                self.state = PlanModeState.INACTIVE if not hooks.should_continue_execution else PlanModeState.COMPLETED
                return hooks.should_continue_execution

            # Phase 2: User Choice - Load plan and get user decision
            if not self._load_and_display_current_plan():
                self.console.print("[red]❌ Failed to load plan[/]")
                self.state = PlanModeState.INACTIVE
                return False

            user_choice = user_choice_func()
            if user_choice["action"] == "replan":
                return self._handle_replanning(
                    user_choice.get("feedback", ""), database_name, message, chat_executor, user_choice_func
                )
            elif user_choice["action"] != "execute":
                self.state = PlanModeState.INACTIVE
                return False

            # Phase 3: Execution - Execute the plan
            execution_mode = user_choice.get("mode", "auto")
            return self._execute_plan(database_name, message, execution_mode, chat_executor, user_choice_func)

        except Exception as e:
            logger.error(f"Unified plan execution error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to execute unified plan: {str(e)}")
            self.state = PlanModeState.INACTIVE
            return False

    def _execute_plan(
        self, database_name: str, message: str, execution_mode: str, chat_executor, user_choice_func
    ) -> bool:
        """Execute the plan with given mode"""
        try:
            actual_instruction = self._build_execution_instruction(database_name, message, execution_mode)

            execution_hooks = PlanModeHooksWithInterception(
                console=self.console,
                plan_manager=self,
                user_choice_func=user_choice_func,
                repl_executor=self.repl_executor,
            )
            execution_hooks.planning_completed = True
            execution_hooks.execution_mode = execution_mode
            execution_hooks.should_continue_execution = True

            chat_executor(
                actual_instruction,
                hooks=execution_hooks,
                show_details=False,
                plan_mode=True,
                shared_plan_tool=self.plan_tool,
            )

            # Check if replanning was requested during execution
            if execution_hooks._replan_requested:
                feedback = getattr(execution_hooks, "_replan_feedback", "")
                return self._handle_replanning(feedback, database_name, message, chat_executor, user_choice_func)

            self.state = PlanModeState.COMPLETED
            return True

        except UserCancelledException:
            self.console.print("[yellow]Execution cancelled by user[/]")
            self.state = PlanModeState.INACTIVE
            return False

        except PlanningCompletedException as e:
            if "User requested replanning" in str(e):
                feedback = getattr(execution_hooks, "_replan_feedback", "")
                return self._handle_replanning(feedback, database_name, message, chat_executor, user_choice_func)
            raise

    def _get_unified_instruction_with_interception(self, database_name: str, user_message: str) -> str:
        """Get unified instruction that will be intercepted after planning"""
        return f"""You are an expert SQL analyst working with a database system.
        Your task is to create a detailed execution plan for the user's request.

**IMPORTANT: Follow these steps exactly:**

1. **Analyze the task**: {user_message}
2. **Database context**: {database_name}
3. **Create a focused plan** by calling the todo_write tool with:
   - 3-8 specific, actionable steps needed to complete this task
   - Each step should be clear and focused on a single action
   - Prioritize the most important steps to achieve the goal efficiently
   - Use descriptive step names that explain what will be accomplished

4. **After calling todo_write, respond with "Planning completed" and STOP.**

Example steps might include:
- "Identify relevant tables and columns for the query"
- "Generate SQL query to answer the user's question"
- "Execute the query and verify the results"
- "Format the output in a clear, readable way"

Now create your plan using todo_write, then respond with "Planning completed"."""

    def _load_and_display_current_plan(self) -> bool:
        """Load the most recent plan and display it"""
        try:
            # Get the todo list from storage
            todo_list = self.plan_tool.storage.get_todo_list()

            if not todo_list:
                self.console.print("[yellow]No todo list found after planning[/]")
                return False
            self.current_plan = todo_list

            self.console.print("\n[bold cyan]Execution Plan:[/]")

            for i, item in enumerate(todo_list.items, 1):
                self.console.print(f"  {i}. {item.content}")

            return True

        except Exception as e:
            logger.error(f"Error loading current plan: {str(e)}")
            self.console.print(f"[red]Error loading plan: {str(e)}[/]")
            return False

    def toggle_plan_mode(self):
        """Toggle Plan Mode on/off"""
        if self.state == PlanModeState.INACTIVE:
            # Turn on plan mode
            self.state = PlanModeState.ACTIVE
        else:
            # Turn off plan mode
            self.state = PlanModeState.INACTIVE

    def reset(self):
        """Reset plan mode to inactive state and clear all todo data"""
        self.state = PlanModeState.INACTIVE
        self.current_plan = None
        # Clear the todo list from storage to ensure clean state
        self.plan_tool.storage.clear_all()

    def get_prompt_prefix(self) -> str:
        """Get the prompt prefix to show current mode"""
        if self.state == PlanModeState.ACTIVE:
            if self.current_plan:
                try:
                    completed = sum(1 for item in self.current_plan.items if item.status == TodoStatus.COMPLETED)
                    total = len(self.current_plan.items)
                    return f"[EXEC {completed}/{total}] "
                except Exception:
                    return "[PLAN] "
            else:
                return "[PLAN] "
        elif self.state == PlanModeState.COMPLETED:
            return "[DONE] "
        else:
            return ""

    def _handle_replanning(
        self, feedback: str, database_name: str, message: str, chat_executor, user_choice_func
    ) -> bool:
        """Handle the replanning process when user provides feedback"""
        try:
            # Generate replanning instruction with feedback
            replanning_instruction = f"""You need to create a REVISED execution plan based on user feedback.

**Original Task**: {message}
**Database**: {database_name}
**User Feedback**: {feedback}

**IMPORTANT: Follow these steps exactly:**
1. **Consider the feedback**: {feedback}
2. **Create an improved plan** by calling the todo_write tool with revised steps
3. **After calling todo_write, respond with "Replanning completed" and STOP.**

Now create your revised plan using todo_write, then respond with "Replanning completed"."""

            # Execute replanning
            replanning_hooks = PlanModeHooksWithInterception(
                console=self.console,
                plan_manager=self,
                user_choice_func=user_choice_func,
                repl_executor=self.repl_executor,
            )

            chat_executor(
                replanning_instruction,
                hooks=replanning_hooks,
                show_details=False,
                plan_mode=True,
                shared_plan_tool=self.plan_tool,
            )

            # Check if replanning succeeded and handle user choice
            if not replanning_hooks.execution_ready:
                self.console.print("[red]Replanning failed[/]")
                self.state = PlanModeState.INACTIVE
                return False

            if not self._load_and_display_current_plan():
                self.console.print("[red]Failed to load revised plan[/]")
                self.state = PlanModeState.INACTIVE
                return False

            # Get user choice for the revised plan
            user_choice = user_choice_func()
            if user_choice["action"] == "execute":
                execution_mode = user_choice.get("mode", "auto")
                return self._execute_plan(database_name, message, execution_mode, chat_executor, user_choice_func)
            elif user_choice["action"] == "replan":
                # Prevent deep recursion by limiting replanning cycles
                return self._handle_replanning(
                    user_choice.get("feedback", ""), database_name, message, chat_executor, user_choice_func
                )
            else:
                self.state = PlanModeState.INACTIVE
                return False

        except Exception as e:
            logger.error(f"Replanning error: {str(e)}")
            self.console.print(f"[bold red]Error during replanning:[/] {str(e)}")
            self.state = PlanModeState.INACTIVE
            return False

    # Backward compatibility properties
    @property
    def is_plan_mode(self) -> bool:
        """Backward compatibility property"""
        return self.state != PlanModeState.INACTIVE
