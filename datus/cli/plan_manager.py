"""
Plan Mode Manager for Datus CLI
Handles plan mode state, todo list management, and execution coordination
"""

from enum import Enum
from typing import Any, Dict, Optional

from agents import AgentHooks
from rich.console import Console

from datus.tools.plan_tools.todo_models import TodoStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PlanModeState(Enum):
    """States for plan mode management"""

    INACTIVE = "inactive"  # Plan mode is off
    PLANNING = "planning"  # AI is generating plan
    READY = "ready"  # Plan created, waiting for execution mode choice
    EXECUTING_AUTO = "executing_auto"  # Auto execution in progress
    EXECUTING_MANUAL = "executing_manual"  # Manual execution in progress
    COMPLETED = "completed"  # Execution finished


class PlanExecutionHook:
    """Hook for plan execution events"""

    def on_step_start(self, step_num: int, total_steps: int, task_content: str) -> Dict[str, Any]:
        """Called before executing each step. Return {'action': 'continue'|'skip'|'stop'}"""
        _ = step_num, total_steps, task_content  # Suppress unused variable warnings
        return {"action": "continue"}

    def on_step_end(self, step_num: int, total_steps: int, task_content: str, success: bool) -> None:
        """Called after executing each step"""
        _ = step_num, total_steps, task_content, success  # Suppress unused variable warnings


class ManualConfirmationHook(PlanExecutionHook):
    """Hook that asks for user confirmation before each step"""

    def __init__(self, console: Console, prompt_func, plan_manager=None):
        self.console = console
        self._prompt_input = prompt_func
        self.plan_manager = plan_manager

    def on_step_start(self, step_num: int, total_steps: int, task_content: str) -> Dict[str, Any]:
        """Ask user for confirmation before executing each step"""
        self.console.print(f"\n[bold cyan]Step {step_num} of {total_steps}:[/]")
        self.console.print(f"[white]{task_content}[/]")

        # Show options for this step
        self.console.print("\n[bold]Choose action:[/]")
        self.console.print("  [green]1[/] - Execute this step")
        self.console.print("  [yellow]2[/] - Execute this step and auto-execute remaining steps")
        self.console.print("  [red]3[/] - Skip this step and provide feedback for adjustment")

        # Define options for step selection
        step_options = [
            ("Execute this step", "continue", "green"),
            ("Execute this step and auto-execute remaining steps", "switch_to_auto", "yellow"),
            ("Skip this step and provide feedback for adjustment", "adjust", "red"),
        ]

        if self.plan_manager:
            selected = self.plan_manager._prompt_with_esc_support(
                "\nEnter your choice (1-3, ESC for feedback): ", step_options, default="1", esc_option="3"
            )
        else:
            # Fallback to simple input if no plan_manager
            try:
                choice = input("\nEnter your choice (1-3): ").strip() or "1"
                choice_num = int(choice)
                if 1 <= choice_num <= len(step_options):
                    selected = step_options[choice_num - 1][1]
                else:
                    selected = None
            except (ValueError, KeyboardInterrupt, EOFError):
                selected = None

        if selected == "continue":
            return {"action": "continue"}
        elif selected == "switch_to_auto":
            return {"action": "switch_to_auto"}
        elif selected == "adjust":
            # Ask for feedback and request plan adjustment
            feedback = input("\nWhat would you like to adjust about this step or the remaining plan? ").strip()
            if feedback:
                return {"action": "adjust", "feedback": feedback, "step_num": step_num}
            else:
                return {"action": "stop"}
        else:
            # Cancelled or invalid
            return {"action": "stop"}

    def on_step_end(self, step_num: int, total_steps: int, task_content: str, success: bool) -> None:
        """Step completion - status is handled by main execution loop"""
        pass  # Main execution loop handles status display


class AutoExecutionHook(PlanExecutionHook):
    """Hook for automatic execution without confirmation"""

    def __init__(self, console: Console):
        self.console = console

    def on_step_start(self, step_num: int, total_steps: int, task_content: str) -> Dict[str, Any]:
        """Auto-execute without asking"""
        return {"action": "continue"}

    def on_step_end(self, step_num: int, total_steps: int, task_content: str, success: bool) -> None:
        """Step completion - status is handled by main execution loop"""
        pass  # Main execution loop handles status display


class PlanModeToolHooks(AgentHooks):
    """Hook for intercepting tool calls in Plan Mode - implements agents library AgentHooks interface"""

    def __init__(self, console: Console, prompt_func, execution_mode: str = "manual", plan_manager=None):
        self.console = console
        self._prompt_input = prompt_func
        self.execution_mode = execution_mode
        self.todo_tools = ["todo_write", "todo_update", "todo_read"]
        self.plan_manager = plan_manager  # Reference to plan manager for direct todo list setting

    async def on_tool_start(self, context, agent, tool) -> None:
        """Called before executing any tool - agents library signature"""
        _ = context, agent  # Suppress unused variable warnings
        try:
            tool_name = getattr(tool, "name", str(tool))

            # Only intercept todo tools in manual mode for detailed info
            if self.execution_mode != "manual" or tool_name not in self.todo_tools:
                return

            # Show tool information
            self.console.print(f"\n[bold cyan]🔧 About to call tool: [yellow]{tool_name}[/][/]")

            # Try to get tool arguments
            if hasattr(tool, "arguments") or hasattr(tool, "args"):
                tool_args = getattr(tool, "arguments", None) or getattr(tool, "args", None)
                if tool_args:
                    self.console.print("[dim]Arguments:[/]")
                    for key, value in tool_args.items():
                        # Truncate long values
                        display_value = str(value)
                        if len(display_value) > 100:
                            display_value = display_value[:100] + "..."
                        self.console.print(f"  [dim]{key}:[/] {display_value}")

            # For now, just show info without blocking (agents library doesn't support cancelling)
            self.console.print("[dim]Proceeding with tool execution...[/]")

        except Exception as e:
            self.console.print(f"[red]Error in tool hook: {e}[/]")

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Called after tool execution - agents library signature"""
        _ = context, agent  # Suppress unused variable warnings
        try:
            tool_name = getattr(tool, "name", str(tool))
            if tool_name in self.todo_tools:
                self.console.print(f"[dim]Tool [yellow]{tool_name}[/] completed[/]")

        except Exception as e:
            self.console.print(f"[red]Error in tool end hook: {e}[/]")

    async def on_start(self, context, agent) -> None:
        """Called when agent starts"""
        _ = context, agent  # Suppress unused variable warnings

    async def on_end(self, context, agent, output) -> None:
        """Called when agent ends"""
        _ = context, agent, output  # Suppress unused variable warnings

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        """Called on agent handoff"""
        _ = context, from_agent, to_agent  # Suppress unused variable warnings


class PlanModeManager:
    """Manages Plan Mode state and execution flow"""

    def __init__(self, console: Console):
        self.console = console
        self.state = PlanModeState.INACTIVE
        self.repl_executor = None  # Will be set by REPL instance
        self.tool_hook: Optional[PlanModeToolHooks] = None
        self.current_plan = None
        self.execution_hook = None

        # Initialize plan tool
        from datus.tools.plan_tools.plan_tool import PlanTool

        self.plan_tool = PlanTool()

    def set_repl_executor(self, repl_executor):
        """Set the REPL executor for actual task execution"""
        self.repl_executor = repl_executor

    def _create_execution_hook(self):
        """Create the appropriate execution hook based on execution mode"""
        if (
            self.state == PlanModeState.EXECUTING_MANUAL
            and self.repl_executor
            and hasattr(self.repl_executor, "_prompt_input")
        ):
            self.execution_hook = ManualConfirmationHook(self.console, self.repl_executor._prompt_input, self)
        else:
            # Auto mode or fallback when manual mode not available
            if self.state == PlanModeState.EXECUTING_MANUAL:
                self.console.print("[yellow]⚠️  Manual confirmation not available, using auto mode[/]")
            self.execution_hook = AutoExecutionHook(self.console)

    def toggle_plan_mode(self):
        """Toggle Plan Mode on/off"""
        if self.state == PlanModeState.INACTIVE:
            # Turn on plan mode
            self.state = PlanModeState.PLANNING
            self.tool_hook = None
            self.console.print("[bold green]📋 Plan Mode ON[/]")
            self.console.print("[dim]Send your task to start planning[/]")
        else:
            # Turn off plan mode
            self.state = PlanModeState.INACTIVE
            self.tool_hook = None
            self.console.print("[bold blue]💬 Plan Mode OFF - Direct execution[/]")

    def show_execution_mode_prompt(self):
        """Show execution mode selection after plan is created - blocking prompt with arrow key navigation"""
        options = [
            ("Execute automatically", "auto", "green"),
            ("Execute with manual confirmation", "manual", "yellow"),
            ("Reject plan and provide feedback for replanning", "reject", "red"),
        ]

        selected = self._number_input_selector(options)

        if selected == "auto":
            self.state = PlanModeState.EXECUTING_AUTO
            self._create_tool_hook()
        elif selected == "manual":
            self.state = PlanModeState.EXECUTING_MANUAL
            self._create_tool_hook()
        elif selected == "reject":
            self.console.print("\n[yellow]Plan rejected. Please provide feedback for replanning:[/]")
            feedback = input("What changes would you like? ").strip()
            if feedback:
                self.console.print(f"[blue]Feedback received:[/] {feedback}")
                self.state = PlanModeState.PLANNING
                return feedback
            else:
                self.console.print("[red]No feedback provided. Exiting plan mode.[/]")
                self.state = PlanModeState.INACTIVE
        else:
            # Cancelled
            self.console.print("\n[red]Plan mode cancelled.[/]")
            self.state = PlanModeState.INACTIVE

        return None

    def _prompt_with_esc_support(self, message: str, options: list, default: str = "1", esc_option: str = "3"):
        """Unified prompt with ESC key support that maps to a specific option

        Args:
            message: Prompt message
            options: List of (text, value, color) tuples
            default: Default option number
            esc_option: Option number that ESC key should map to

        Returns:
            The selected option value or None if cancelled
        """
        while True:
            try:
                from prompt_toolkit import prompt
                from prompt_toolkit.key_binding import KeyBindings
                from prompt_toolkit.keys import Keys

                # Create key bindings for ESC key
                bindings = KeyBindings()

                @bindings.add(Keys.Escape)
                def _(event):
                    """Handle ESC key - maps to specified option"""
                    event.app.current_buffer.text = esc_option
                    event.app.current_buffer.validate_and_handle()

                choice = prompt(message, default=default, key_bindings=bindings).strip()

                if choice == "0":
                    return None

                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    _, value, _ = options[choice_num - 1]
                    return value
                else:
                    self.console.print(f"[red]Invalid choice. Please enter 1-{len(options)}[/]")

            except (ValueError, KeyboardInterrupt, EOFError):
                return None

    def _number_input_selector(self, options):
        """Number input for execution mode selection"""
        self.console.print("[bold green]✅ Plan created successfully![/]")
        self.console.print("\n[bold]Please choose how to proceed:[/]")

        for i, (text, _, color) in enumerate(options):
            self.console.print(f"  [{color}]{i+1}[/] - {text}")

        return self._prompt_with_esc_support(
            f"\nEnter your choice (1-{len(options)}, ESC for feedback): ", options, default="1", esc_option="3"
        )

    def _create_tool_hook(self):
        """Create the appropriate tool hook based on execution mode"""
        execution_mode = "manual" if self.state == PlanModeState.EXECUTING_MANUAL else "auto"

        # Simple prompt function for manual mode
        def simple_prompt(message: str, choices: list = None, default: str = "y"):
            try:
                if choices:
                    choice_str = "/".join(choices)
                    prompt_text = f"{message} ({choice_str}) [{default}]: "
                else:
                    prompt_text = f"{message} [{default}]: "
                result = input(prompt_text).strip()
                return result if result else default
            except (KeyboardInterrupt, EOFError):
                return default

        prompt_func = simple_prompt if execution_mode == "manual" else None
        self.tool_hook = PlanModeToolHooks(self.console, prompt_func, execution_mode, self)

    def get_prompt_prefix(self) -> str:
        """Get the prompt prefix to show current mode"""
        if self.state in [PlanModeState.EXECUTING_AUTO, PlanModeState.EXECUTING_MANUAL] and self.current_plan:
            completed = sum(1 for item in self.current_plan.items if item.status == TodoStatus.COMPLETED)
            total = len(self.current_plan.items)
            return f"[EXEC {completed}/{total}] "
        elif self.state != PlanModeState.INACTIVE:
            return "[PLAN] "
        else:
            return ""

    # Backward compatibility properties
    @property
    def is_plan_mode(self) -> bool:
        """Backward compatibility property"""
        return self.state != PlanModeState.INACTIVE

    @property
    def execution_mode(self) -> Optional[str]:
        """Backward compatibility property"""
        if self.state == PlanModeState.EXECUTING_AUTO:
            return "auto"
        elif self.state == PlanModeState.EXECUTING_MANUAL:
            return "manual"
        elif self.state == PlanModeState.PLANNING:
            return "planning"
        else:
            return None

    @property
    def is_executing(self) -> bool:
        """Backward compatibility property"""
        return self.state in [PlanModeState.EXECUTING_AUTO, PlanModeState.EXECUTING_MANUAL]

    def execute_current_plan(self):
        """Execute the current plan step by step"""
        if not self.current_plan:
            self.console.print(
                "[red]❌ No execution plan available. The plan should have been created during the planning phase.[/]"
            )
            self.console.print("[dim]This is likely a bug - please try creating a new plan.[/]")
            self.console.print("[yellow]Plan mode reset. Ready for new tasks.[/]")
            self.state = PlanModeState.INACTIVE
            return

        # Ensure we're in execution mode
        if self.state not in [PlanModeState.EXECUTING_MANUAL, PlanModeState.EXECUTING_AUTO]:
            self.console.print("[red]❌ Invalid state for execution. Please restart plan mode.[/]")
            self.state = PlanModeState.INACTIVE
            return
        total_steps = len(self.current_plan.items)

        try:
            self.console.print(f"\n[bold green]🚀 Starting execution of {total_steps} steps...[/]\n")

            for i, todo_item in enumerate(self.current_plan.items):
                if todo_item.status == TodoStatus.COMPLETED:
                    continue

                step_num = i + 1

                # Call hook before executing step
                if self.execution_hook:
                    hook_result = self.execution_hook.on_step_start(step_num, total_steps, todo_item.content)
                    action = hook_result.get("action", "continue")

                    if action == "stop":
                        self.console.print(f"[yellow]⏹️  Execution stopped at step {step_num}[/]")
                        break
                    elif action == "switch_to_auto":
                        # Switch to automatic execution for remaining steps
                        self.console.print("[yellow]🔄 Switching to automatic execution for remaining steps...[/]")
                        self.state = PlanModeState.EXECUTING_AUTO
                        self._create_execution_hook()  # Switch to auto execution hook
                        # Continue with current step (will be executed automatically from now on)
                    elif action == "adjust":
                        # Handle plan adjustment request
                        feedback = hook_result.get("feedback", "")
                        current_step_num = hook_result.get("step_num", step_num)
                        self.console.print("[blue]🔄 Adjusting plan based on your feedback...[/]")

                        # Call plan adjustment method
                        adjusted = self._adjust_plan_with_feedback(feedback, current_step_num)
                        if adjusted:
                            # Restart execution with the new plan, but mark completed steps appropriately
                            self.console.print("[green]✓ Restarting with adjusted plan...[/]")
                            return self.execute_current_plan()
                        else:
                            self.console.print("[red]❌ Failed to adjust plan, stopping execution[/]")
                            break

                # Execute the task
                execution_success = True
                try:
                    if not self.repl_executor:
                        raise RuntimeError("No executor available - plan mode setup error")
                    self.repl_executor._execute_normal_chat_command(todo_item.content)
                except Exception as e:
                    logger.error(f"Failed to execute task: {str(e)}")
                    execution_success = False

                # Update status based on execution result
                status = "completed" if execution_success else "failed"
                result = self.plan_tool.todo_update(self.current_plan.list_id, todo_item.id, status)

                if result.success:
                    status_icon = "✓" if execution_success else "❌"
                    status_color = "green" if execution_success else "red"
                    self.console.print(
                        f"[EXEC {step_num}/{total_steps}] {status_icon} {todo_item.content} - "
                        f"[{status_color}]{status}[/]"
                    )

                    # Update local state
                    todo_item.status = TodoStatus.COMPLETED if execution_success else TodoStatus.FAILED

                    # Call hook after executing step (but don't let it print duplicate messages)
                    if self.execution_hook:
                        self.execution_hook.on_step_end(step_num, total_steps, todo_item.content, execution_success)
                else:
                    self.console.print(f"[EXEC {step_num}/{total_steps}] ❌ {todo_item.content} - [red]update failed[/]")
                    logger.error(f"Failed to update todo item {todo_item.id}: {result.error}")

                # If execution failed and in manual mode, ask user
                if not execution_success and self.execution_mode == "manual":
                    # Ask user if they want to continue
                    try:
                        continue_execution = input("\nContinue with remaining steps? [y/N]: ").lower().startswith("y")
                        if not continue_execution:
                            break
                    except KeyboardInterrupt:
                        self.console.print("\n[yellow]Execution cancelled by user[/]")
                        break

            # Show completion summary
            completed = sum(1 for item in self.current_plan.items if item.status == TodoStatus.COMPLETED)
            self.console.print(f"\n🎉 Plan execution completed! {completed}/{total_steps} steps finished.")

            if completed == total_steps:
                self.console.print("[bold green]All tasks completed successfully![/]")
            else:
                self.console.print(f"[yellow]{total_steps - completed} tasks were not completed[/]")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Execution interrupted by user[/]")
        except Exception as e:
            logger.error(f"Error during plan execution: {str(e)}")
            self.console.print(f"[red]❌ Error during execution: {str(e)}[/]")
        finally:
            # Reset plan mode state after execution completes
            self.current_plan = None
            self.state = PlanModeState.COMPLETED

    def _adjust_plan_with_feedback(self, feedback: str, current_step_num: int) -> bool:
        """Adjust the current plan based on user feedback, keeping completed steps intact"""
        try:
            if not self.current_plan or not feedback.strip():
                return False

            # Get completed steps summary for context
            completed_steps = []
            for i, item in enumerate(self.current_plan.items):
                if i + 1 < current_step_num and item.status.value == "completed":
                    completed_steps.append(f"{i+1}. {item.content} ✓")

            # Get current and remaining steps
            remaining_steps = []
            for i, item in enumerate(self.current_plan.items):
                if i + 1 >= current_step_num:
                    remaining_steps.append(f"{i+1}. {item.content}")

            adjustment_instruction = f"""The user has provided SPECIFIC technical instructions for the task. \
You must implement their exact approach.

USER'S TECHNICAL FEEDBACK: {feedback}

CRITICAL: The user has given you the EXACT solution approach. DO NOT explore, investigate, or identify anything.

Your task: Create a focused todo list using todo_write that implements the user's specific instructions.

RULES:
1. First step must DIRECTLY query the specific table/data the user mentioned
2. Use the EXACT column names and conditions the user provided
3. NO exploratory steps like "explore database", "identify tables", "analyze schema"
4. Start with concrete SQL/database operations using user's specifications
5. Each step must be actionable and specific, not exploratory

TOOL REQUIREMENT: Use ONLY todo_write tool.

Example based on typical user feedback:
If user says "use frpm table, Educational Option Type = 'Continuation School'", your first todo should be:
"Query frpm table WHERE Educational Option Type = 'Continuation School'"

NOT: "Explore database to identify continuation schools"

Create the todo list now using todo_write with steps that directly implement what the user specified."""

            # Get AI to create adjusted plan
            if self.repl_executor:
                self.console.print("[blue]🔄 Creating adjusted plan based on your feedback...[/]")
                self.repl_executor.chat_commands.execute_chat_command(adjustment_instruction, show_details=False)

                # Reload the adjusted plan
                from datus.tools.plan_tools.plan_tool import PlanTool

                plan_tool = PlanTool()
                all_lists = plan_tool.storage.list_all_lists()

                if all_lists:
                    # Get the most recent list (should be the adjusted one)
                    most_recent_list = max(
                        all_lists.values(),
                        key=lambda x: x.created_at if hasattr(x, "created_at") and x.created_at else 0,
                    )

                    self.current_plan = most_recent_list
                    self.console.print(
                        f"[green]✓ Plan adjusted! New plan has {len(most_recent_list.items)} total steps[/]"
                    )

                    # Display the new plan
                    self.console.print("\n[bold cyan]📋 Updated Execution Plan:[/]")
                    for i, item in enumerate(most_recent_list.items, 1):
                        status_icon = "⏳"  # All steps start as pending in new plan
                        self.console.print(f"  {i}. {status_icon} {item.content}")

                    return True

                return False

            return False

        except Exception as e:
            logger.error(f"Error adjusting plan: {str(e)}")
            return False
