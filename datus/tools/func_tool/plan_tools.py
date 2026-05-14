# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Simplified plan tools - merged from multiple files into single module
"""

import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from uuid import uuid4

from agents import SQLiteSession, Tool
from pydantic import BaseModel, Field, ValidationError

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)


class TodoStatus(str, Enum):
    """Status of a todo item"""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class TodoItem(BaseModel):
    """Individual todo item"""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the todo item")
    content: str = Field(..., description="Content/description of the todo item")
    status: TodoStatus = Field(default=TodoStatus.PENDING, description="Status of the todo item")


class TodoList(BaseModel):
    """Collection of todo items"""

    items: List[TodoItem] = Field(default_factory=list, description="List of todo items")

    def add_item(self, content: str) -> TodoItem:
        """Add a new todo item to the list"""
        item = TodoItem(content=content)
        self.items.append(item)
        return item

    def get_item(self, item_id: str) -> Optional[TodoItem]:
        """Get a todo item by ID"""
        return next((item for item in self.items if item.id == item_id), None)

    def update_item_status(self, item_id: str, status: TodoStatus) -> bool:
        """Update the status of a todo item and optionally save execution result"""
        item = self.get_item(item_id)
        if item:
            item.status = status
            return True
        return False

    def get_completed_items(self) -> List[TodoItem]:
        """Get all completed items"""
        return [item for item in self.items if item.status == TodoStatus.COMPLETED]


class SessionTodoStorage:
    """Per-session todo list storage with on-disk persistence.

    The ``session_id`` argument accepts either a ``str`` (immediate value)
    or a zero-arg callable that returns the current session id when
    invoked. The callable form is what main-agent setup uses: ``PlanTool``
    is constructed during ``setup_tools`` *before* ``_get_or_create_session``
    has allocated a session_id, so we must defer resolution until each
    actual ``todo_*`` call — otherwise the storage would snapshot
    ``None`` forever and never persist to disk.

    When the resolver returns a non-empty session_id, the todo list is
    written through to ``{project_data_dir}/todos/{session_id}.json`` on
    every ``save_list`` / ``clear_all`` call, and lazily reloaded from disk
    the first time ``get_todo_list`` / ``has_todo_list`` is invoked. This
    is what lets ``datus chat --resume`` recover the todolist after the
    process exits.

    When the resolver returns ``None`` (e.g. tests that bypass the agentic
    flow), the storage falls back to a pure in-memory dict.
    """

    def __init__(
        self,
        session: SQLiteSession,
        session_id: Union[Optional[str], Callable[[], Optional[str]]] = None,
    ):
        self.session = session
        # Keep the original value (str or callable) so ``_resolve_session_id``
        # can re-evaluate on every disk access.
        self._session_id_source = session_id
        self._current_todo_list: Optional[TodoList] = None
        # Lazy-load flag — disk is read at most once per instance unless
        # ``save_list`` / ``clear_all`` already wrote authoritative state.
        # Note: this latch keys off the *first* non-empty session_id; if
        # session_id transitions from None to a real value we still want
        # the first real ``get_todo_list`` to attempt a disk load, so
        # ``_ensure_loaded`` resets the flag when the resolved id changes.
        self._loaded_from_disk = False
        self._loaded_for_session_id: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        """Resolve the current session_id (re-invokes the callable each access)."""
        src = self._session_id_source
        if callable(src):
            try:
                return src()
            except Exception as exc:  # noqa: BLE001
                logger.debug("session_id resolver raised: %s", exc)
                return None
        return src

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id_source = value

    def _disk_path(self) -> Optional[Path]:
        sid = self.session_id
        if not sid:
            return None
        try:
            from datus.utils.path_manager import get_path_manager

            return get_path_manager().todo_list_path(sid)
        except Exception as exc:  # noqa: BLE001 — never crash todo IO over a path-manager hiccup
            logger.debug("todo_list_path unavailable: %s", exc)
            return None

    def _ensure_loaded(self) -> None:
        # Re-attempt disk load when the resolved session_id changes (e.g.
        # PlanTool was constructed before ``_get_or_create_session``
        # generated the id). Without this, the latch from the initial
        # session_id=None call would prevent the first real lookup from
        # ever hitting disk.
        sid = self.session_id
        if self._loaded_from_disk and self._loaded_for_session_id == sid:
            return
        self._loaded_from_disk = True
        self._loaded_for_session_id = sid
        path = self._disk_path()
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._current_todo_list = TodoList(**data)
            logger.debug("Loaded todo list from %s with %d items", path, len(self._current_todo_list.items))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Failed to load todolist from %s: %s", path, exc)

    def save_list(self, todo_list: TodoList) -> bool:
        """Persist the todo list to memory and (if session-bound) to disk."""
        try:
            self._current_todo_list = todo_list
            self._loaded_from_disk = True
            self._loaded_for_session_id = self.session_id
            path = self._disk_path()
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # ensure_ascii=False keeps CJK / accents readable on disk.
                    path.write_text(
                        json.dumps(todo_list.model_dump(), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except OSError as exc:
                    logger.warning("Failed to persist todolist to %s: %s", path, exc)
            logger.debug("Saved todo list with %d items", len(todo_list.items))
            return True
        except Exception as e:
            logger.error(f"Failed to save todo list: {e}")
            return False

    def get_todo_list(self) -> Optional[TodoList]:
        self._ensure_loaded()
        return self._current_todo_list

    def clear_all(self) -> None:
        try:
            self._current_todo_list = None
            self._loaded_from_disk = True
            self._loaded_for_session_id = self.session_id
            path = self._disk_path()
            if path is not None and path.exists():
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("Failed to delete todolist %s: %s", path, exc)
            logger.debug("Cleared todo list")
        except Exception as e:
            logger.error(f"Failed to clear todo list: {e}")

    def has_todo_list(self) -> bool:
        self._ensure_loaded()
        return self._current_todo_list is not None


class PlanTool:
    """Main tool for todo list management with read, write, and update capabilities"""

    def __init__(
        self,
        session: SQLiteSession,
        session_id: Union[Optional[str], Callable[[], Optional[str]]] = None,
    ):
        """Initialize the plan tool with session.

        ``session_id`` is forwarded to :class:`SessionTodoStorage` so the
        todolist persists across process restarts (resume support). It may
        be a string or a zero-arg callable — see ``SessionTodoStorage`` for
        why the callable form matters during setup_tools().
        """
        self.storage = SessionTodoStorage(session, session_id=session_id)

    def available_tools(self) -> List[Tool]:
        """Get list of available plan tools"""
        methods_to_convert = [
            self.todo_read,
            self.todo_write,
            self.todo_update,
        ]

        bound_tools = []
        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    def todo_read(self) -> FuncToolResult:
        """Read the todo list from storage"""
        todo_list = self.storage.get_todo_list()

        if todo_list:
            return FuncToolResult(
                result={
                    "message": "Successfully retrieved todo list",
                    "lists": [todo_list.model_dump()],
                    "total_lists": 1,
                }
            )
        else:
            return FuncToolResult(
                result={
                    "message": "No todo list found",
                    "lists": [],
                    "total_lists": 0,
                }
            )

    def todo_write(self, todos_json: str) -> FuncToolResult:
        """Append new todo items to the current list (does NOT overwrite).

        ``todo_write`` is incremental: pass only the items you want to add.
        Existing items stay untouched — use ``todo_update`` to change a
        single item's status, or call this again later to add more.

        Args:
            todos_json: JSON string of list of dicts with 'content' and
                'status' keys. Status can be 'pending' or 'completed'
                (default 'pending' if omitted).

                Example: '[{"content": "Query database", "status": "pending"}]'

        Behaviour notes:
            * Empty / whitespace-only ``content`` entries are skipped.
            * Each new item is assigned a fresh uuid; no content dedupe is
              performed, so callers are responsible for not re-submitting
              the same task twice.
        """
        try:
            import json

            todos = json.loads(todos_json)
        except (json.JSONDecodeError, TypeError):
            return FuncToolResult(success=0, error="Invalid JSON format for todos")

        if not todos:
            return FuncToolResult(success=0, error="Cannot create todo list: no todo items provided")

        # Start from the existing list (read-modify-write append semantics).
        # ``get_todo_list`` lazily loads from disk if needed, so a fresh
        # PlanTool instance after resume still sees prior items.
        todo_list = self.storage.get_todo_list() or TodoList()

        added = 0
        added_completed = 0
        for todo_item in todos:
            content = todo_item.get("content", "").strip()
            status = todo_item.get("status", "pending").lower()

            if not content:
                continue

            if status == "completed":
                new_item = TodoItem(content=content, status=TodoStatus.COMPLETED)
                todo_list.items.append(new_item)
                added_completed += 1
                logger.info(f"Appended completed step: {content}")
            else:
                todo_list.add_item(content)
                added += 1
                logger.info(f"Appended pending step: {content}")

        if added == 0 and added_completed == 0:
            return FuncToolResult(success=0, error="Cannot append todo list: no valid items after filtering")

        if self.storage.save_list(todo_list):
            return FuncToolResult(
                result={
                    "message": (
                        f"Appended {added + added_completed} item(s) "
                        f"({added_completed} completed, {added} pending); "
                        f"list now has {len(todo_list.items)} item(s)."
                    ),
                    "todo_list": todo_list.model_dump(),
                }
            )
        return FuncToolResult(success=0, error="Failed to save todo list to storage")

    def todo_update(self, todo_id: str, status: str) -> FuncToolResult:
        """Update a todo item's status.

        Execution flow:
        1. todo_update(todo_id, "pending") - Mark as about to be executed
        2. [execute task]
        3. todo_update(todo_id, "completed") - Mark as successfully executed
           OR todo_update(todo_id, "failed") - Mark as failed

        Args:
            todo_id: The ID of the todo item to update
            status: New status - must be 'pending', 'completed', or 'failed'

        Returns:
            FuncToolResult: Success/error status
        """
        return self._update_todo_status(todo_id, status)

    def _update_todo_status(
        self, todo_id: str, status: str, execution_output: Optional[str] = None, error_message: Optional[str] = None
    ) -> FuncToolResult:
        """Internal method to update todo item status and optionally save execution result"""
        _ = execution_output, error_message  # Mark as used for future extensibility
        try:
            status_enum = TodoStatus(status.lower())
        except ValueError:
            return FuncToolResult(
                success=0, error=f"Invalid status '{status}'. Must be 'completed', 'pending', or 'failed'"
            )

        todo_list = self.storage.get_todo_list()
        if not todo_list:
            return FuncToolResult(success=0, error="No todo list found")

        todo_item = todo_list.get_item(todo_id)
        if not todo_item:
            return FuncToolResult(success=0, error=f"Todo item with ID '{todo_id}' not found")

        if todo_list.update_item_status(todo_id, status_enum):
            if self.storage.save_list(todo_list):
                updated_item = todo_list.get_item(todo_id)
                return FuncToolResult(
                    result={
                        "message": f"Successfully updated todo item to '{status}' status",
                        "updated_item": updated_item.model_dump(),
                    }
                )
            else:
                return FuncToolResult(success=0, error="Failed to save updated todo list to storage")
        else:
            return FuncToolResult(success=0, error="Failed to update todo item status")


class ConfirmPlanTool:
    """Tool wrapping the user-facing ``confirm_plan`` call.

    The tool reads the plan file the LLM has been editing, pushes its
    contents to the user via :meth:`InteractionBroker.send`, and then
    prompts the user with a *confirm-or-revise* interaction. Confirming
    exits plan mode; any free-text response is returned to the LLM as
    feedback so it can iterate.
    """

    def __init__(self, node: "AgenticNode"):
        self.node = node

    def available_tools(self) -> List[Tool]:
        return [trans_to_function_tool(self.confirm_plan)]

    async def confirm_plan(self) -> FuncToolResult:
        """Confirm the current plan with the user.

        Preconditions:
        - Plan mode MUST be active (user activated via Shift+Tab / --plan-mode).
          Otherwise this tool returns an error so the LLM does not surface a
          confirmation prompt outside the formal plan-mode workflow.

        Workflow:
        - Read ``node.plan_file_path``; if missing, return an error so the
          LLM knows to write the plan first.
        - Push the plan content to the user as an assistant message.
        - Ask the user to either ``confirm`` or type free-text feedback.
        - On ``confirm``: deactivate plan mode and return success.
        - On feedback: return the text so the LLM can revise the plan.
        """
        # Local imports to avoid cycles with execution_state / schemas.
        from datus.cli.execution_state import InteractionCancelled
        from datus.schemas.interaction_event import InteractionEvent

        if not self.node.is_in_plan_mode():
            return FuncToolResult(
                success=0,
                error=(
                    "plan mode is not active; the user must enable plan mode "
                    "(Shift+Tab in REPL or --plan-mode flag) before calling confirm_plan"
                ),
            )

        path = self.node.plan_file_path
        if not path or not Path(path).exists():
            return FuncToolResult(
                success=0,
                error=(
                    "plan file not found at "
                    f"{path or '<unset>'}; write the plan to this path before calling confirm_plan"
                ),
            )

        broker = getattr(self.node, "interaction_broker", None)
        if broker is None:
            return FuncToolResult(success=0, error="interaction broker unavailable on node")

        try:
            plan_md = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            return FuncToolResult(success=0, error=f"failed to read plan file: {exc}")

        preview = f"\n---\n\n{plan_md}"
        await broker.send(content=preview, content_type="markdown", action_type="plan_preview")

        event = InteractionEvent(
            title="Plan",
            content="Confirm this plan, or type feedback to revise:",
            content_type="markdown",
            choices={"confirm": "Confirm"},
            default_choice="confirm",
            allow_free_text=True,
        )
        try:
            answers = await broker.request([event])
        except InteractionCancelled:
            return FuncToolResult(success=0, error="user cancelled plan confirmation")

        # ``answers`` is List[List[str]]; for a single-question prompt the
        # user's response is answers[0][0] (or "" when nothing was provided).
        user_choice = ""
        if answers and isinstance(answers, list) and answers[0]:
            user_choice = answers[0][0] or ""

        if user_choice == "confirm":
            plan_path = self.node.plan_file_path
            self.node.deactivate_plan_mode()
            # Set a one-shot flag so the next user prompt carries an "execute
            # the confirmed plan" reminder in the enhanced section.
            self.node._plan_just_confirmed = True
            return FuncToolResult(
                result={
                    "status": "confirmed",
                    "plan_file": plan_path,
                    "next_action": (
                        f"The plan at {plan_path} has been approved. Plan mode is now exited.\n"
                        "**Do NOT end this turn with a natural-language message yet.** "
                        "Your immediate next steps MUST be:\n"
                        f"  1. Read {plan_path} via read_file to recall the plan content.\n"
                        "  2. Call todo_write to convert the plan's concrete actionable steps "
                        "into a todo list (one todo per step).\n"
                        "  3. Execute the first pending todo by calling the relevant tools "
                        "(grep / read_file / list_tables / read_query / write_file — whatever "
                        "the step requires).\n"
                        "  4. After completing each step, call todo_update to mark it completed, "
                        "then move to the next step.\n"
                        "  5. Continue executing steps without asking the user for permission, "
                        "until either all todos are done or you hit a blocker that genuinely "
                        "requires user input (in which case use ask_user)."
                    ),
                }
            )

        return FuncToolResult(
            result={
                "status": "feedback",
                "feedback": user_choice,
                "next_action": (
                    "The user requested revisions to the plan (see ``feedback`` above). "
                    "Apply the changes via edit_file on the plan file. Do NOT call "
                    "confirm_plan again until the feedback is addressed."
                ),
            }
        )
