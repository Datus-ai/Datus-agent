"""
Plan Mode Manager for Datus CLI
Handles plan mode state, todo list management, and execution coordination
"""

from enum import Enum

from rich.console import Console

from datus.cli.plan_hooks import PlanModeHooks, ReplanRequestedException
from datus.tools.plan_tools import TodoStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PlanModeState(Enum):
    """Simplified states for plan mode management"""

    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"


class PlanModeManager:
    """Manages Plan Mode state and execution flow"""

    def __init__(self, console: Console):
        self.console = console
        self.state = PlanModeState.INACTIVE
        self.repl_executor = None  # Will be set by REPL instance
        self.current_plan = None
        self._cached_plan_tool = None

    def get_plan_tool_for_session(self, chat_node):
        """Get or create PlanTool connected to current session's TodoStorage"""
        # Create tool once and reuse
        if self._cached_plan_tool is None:
            from datus.tools.plan_tools import PlanTool

            self._cached_plan_tool = PlanTool()

        # Connect to session's TodoStorage if available
        if chat_node and hasattr(chat_node, "session") and chat_node.session:
            if hasattr(chat_node.session, "todo_storage"):
                self._cached_plan_tool.storage = chat_node.session.todo_storage

        return self._cached_plan_tool

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
                + """**MANUAL MODE ACTIVE**: You are in step-by-step manual confirmation mode.

CRITICAL WORKFLOW:
1. First call todo_read() to see the current todo list
2. Find the FIRST item with status 'pending' (not completed)
3. Call todo_update_pending(todo_id) for that item - this will trigger manual confirmation
4. The user may: approve, cancel, request replanning, or switch to auto mode
5. If user approves: execute the actual task for that todo item
6. If user requests replanning: STOP immediately and let the system handle replanning
7. When a task is complete, call todo_update_completed(todo_id)
8. Repeat for the next pending item

IMPORTANT:
- You must call todo_update_pending() before executing each task step to get user confirmation
- Be prepared to stop execution if user requests replanning with feedback
- The system will automatically restart with a revised plan if replanning is requested"""
            )
        else:
            return base_instruction + "Execute the planned steps to complete this task."

    def execute_unified_plan(
        self, message: str, database_name: str, chat_executor, user_choice_func, chat_node=None, is_replan: bool = False
    ) -> bool:
        """Execute unified plan - simplified with helper methods"""
        try:
            self.state = PlanModeState.ACTIVE

            # Only clear storage on initial plan, not on replan
            if not is_replan and chat_node:
                chat_node.clear_todo_storage()

            # Planning phase
            instruction = self._get_instruction(database_name, message)
            hooks = PlanModeHooks(console=self.console, todo_storage=chat_node.get_todo_storage())
            chat_executor(instruction, hooks=hooks, show_details=False, plan_mode=True)

            if not hooks.planning_completed:
                self.state = PlanModeState.INACTIVE if not hooks.should_continue_execution else PlanModeState.COMPLETED
                return hooks.should_continue_execution

            # User choice phase
            if not self._load_and_display_current_plan(chat_node):
                self.console.print("[red]❌ Failed to load plan[/]")
                self.state = PlanModeState.INACTIVE
                return False

            user_choice = user_choice_func()
            if user_choice["action"] == "replan":
                raise ReplanRequestedException(user_choice.get("feedback", ""))
            elif user_choice["action"] != "execute":
                self.state = PlanModeState.INACTIVE
                return False

            # Execution phase - reuse planning hooks with updated config
            execution_mode = user_choice.get("mode", "auto")
            hooks.planning_completed = True
            hooks.execution_mode = execution_mode
            return self._execute_plan(database_name, message, execution_mode, chat_executor, hooks)

        except ReplanRequestedException:
            raise
        except Exception as e:
            logger.error(f"Unified plan execution error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to execute unified plan: {str(e)}")
            self.state = PlanModeState.INACTIVE
            return False

    def _execute_plan(self, database_name: str, message: str, execution_mode: str, chat_executor, hooks) -> bool:
        """Execute the plan in the specified mode"""
        try:
            execution_instruction = self._build_execution_instruction(database_name, message, execution_mode)

            chat_executor(
                execution_instruction,
                hooks=hooks,
                show_details=False,
                plan_mode=True,
            )

            if hooks.should_continue_execution:
                self.state = PlanModeState.COMPLETED
                return True
            else:
                self.state = PlanModeState.INACTIVE
                return False

        except Exception as e:
            logger.error(f"Plan execution error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to execute plan: {str(e)}")
            self.state = PlanModeState.INACTIVE
            return False

    def _get_instruction(self, database_name: str, message: str) -> str:
        """Get unified instruction for planning phase with interception"""
        return f"""Database: {database_name}
Task: {message}

You are an expert data analyst. Create a detailed execution plan for this task.

**IMPORTANT: Follow these steps exactly:**
1. **Analyze the task** and understand what needs to be done
2. **Create a detailed plan** by calling todo_write with a list of steps
3. **After calling todo_write, respond with "Planning completed" and STOP.**

**Format**: Call todo_write with a JSON string of dicts, each with 'content' and 'status':
```
todo_write('[{{"content": "Identify relevant tables", "status": "pending"}},
{{"content": "Generate SQL query", "status": "pending"}},
{{"content": "Execute and verify results", "status": "pending"}}]')
```

Do NOT execute any other tools. Your job is ONLY to create the plan.

Now create your plan using todo_write, then respond with "Planning completed"."""

    def _load_and_display_current_plan(self, chat_node) -> bool:
        """Load the most recent plan and display it through chat_node interface"""
        try:
            if not chat_node:
                self.console.print("[yellow]No chat_node provided[/]")
                return False

            # Use chat_node interface to get todo storage
            todo_storage = chat_node.get_todo_storage()
            if not todo_storage:
                self.console.print("[yellow]No todo storage available[/]")
                return False

            todo_list = todo_storage.get_todo_list()

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
            self.state = PlanModeState.ACTIVE
        else:
            self.state = PlanModeState.INACTIVE

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

    # Backward compatibility properties
    @property
    def is_plan_mode(self) -> bool:
        """Backward compatibility property"""
        return self.state != PlanModeState.INACTIVE
