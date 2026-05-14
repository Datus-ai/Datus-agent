# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Simplified plan tools - merged from multiple files into single module
"""

import json
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from agents import SQLiteSession, Tool
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)


TITLE_WORD_LIMIT = 8


class TodoStatus(str, Enum):
    """Status of a todo item.

    Recommended flow: ``pending`` → ``in_progress`` → ``completed``.
    ``failed`` is reachable from any state via ``todo_update``; transitions
    are not enforced because an LLM may need to revert an in_progress item
    back to pending after a partial rollback.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TodoItem(BaseModel):
    """Individual todo item.

    ``id`` is a per-list incrementing integer assigned by
    :meth:`TodoList.add_item`. ``title`` is a short headline (≤ 8 words)
    used by the sidebar and tool summaries; ``content`` carries the full
    task description and is only fetched by ``todo_read(id)``.
    """

    id: int = Field(..., description="Per-list auto-incrementing identifier")
    title: str = Field(..., description="Short headline, at most 8 whitespace-separated words")
    content: str = Field(..., description="Full task description")
    status: TodoStatus = Field(default=TodoStatus.PENDING, description="Status of the todo item")

    @field_validator("title")
    @classmethod
    def _title_word_count(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("title must not be empty")
        words = [w for w in re.split(r"\s+", stripped) if w]
        if len(words) > TITLE_WORD_LIMIT:
            raise ValueError(f"title must be {TITLE_WORD_LIMIT} words or fewer (got {len(words)})")
        return stripped

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("content must not be empty")
        return value.strip()


class TodoList(BaseModel):
    """Collection of todo items with a per-list id counter.

    ``next_id`` is the value that :meth:`add_item` will assign to the
    *next* new item. It is persisted alongside ``items`` so that ids stay
    monotonic across process restarts (resume). The
    :func:`_reconcile_next_id` model validator repairs the counter on load
    when callers persisted ``items`` without ``next_id`` (or with a stale
    one) — without it, a hand-edited JSON file could collide an existing
    id, and ``todo_update`` would silently target the wrong row.
    """

    items: List[TodoItem] = Field(default_factory=list, description="List of todo items")
    next_id: int = Field(default=1, description="Next id to assign on add_item; reconciled on load")

    @model_validator(mode="after")
    def _reconcile_next_id(self) -> "TodoList":
        max_id = max((it.id for it in self.items), default=0)
        if self.next_id <= max_id:
            self.next_id = max_id + 1
        return self

    def add_item(self, title: str, content: str) -> TodoItem:
        """Append a new pending todo item and return it."""
        item = TodoItem(id=self.next_id, title=title, content=content)
        self.next_id += 1
        self.items.append(item)
        return item

    def get_item(self, item_id: int) -> Optional[TodoItem]:
        """Get a todo item by integer id."""
        return next((item for item in self.items if item.id == item_id), None)

    def update_item_status(self, item_id: int, status: TodoStatus) -> bool:
        """Update the status of a todo item."""
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
            # Pre-refactor todo files used uuid string ids and lacked a
            # ``title`` field; they will fail validation under the new
            # schema. Todos are session-local and cheap to regenerate, so
            # we drop the legacy payload instead of attempting a field map.
            logger.warning("Discarding incompatible todolist at %s: %s", path, exc)
            self._current_todo_list = None

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
        """Get list of available plan tools."""
        methods_to_convert = [
            self.todo_list,
            self.todo_read,
            self.todo_write,
            self.todo_update,
        ]

        bound_tools = []
        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    @staticmethod
    def _overview_items(todo_list: TodoList) -> List[dict]:
        return [{"id": it.id, "title": it.title, "status": it.status.value} for it in todo_list.items]

    def todo_list(self) -> FuncToolResult:
        """Return an overview of all todos in the current session.

        Each entry is ``{id, title, status}`` — ``content`` is NOT included.
        Use ``todo_read(id)`` to fetch the full description of a single item
        before you start executing it.

        Returns ``items: []`` (and total/completed = 0) when no list exists,
        rather than an error: callers can treat "no list yet" as a normal
        state and decide whether to call ``todo_write``.
        """
        todo_list = self.storage.get_todo_list()
        if not todo_list:
            return FuncToolResult(
                result={
                    "message": "No todo list found",
                    "items": [],
                    "total": 0,
                    "completed": 0,
                }
            )
        items = self._overview_items(todo_list)
        completed = sum(1 for it in todo_list.items if it.status == TodoStatus.COMPLETED)
        return FuncToolResult(
            result={
                "message": "Successfully retrieved todo list",
                "items": items,
                "total": len(items),
                "completed": completed,
            }
        )

    def todo_read(self, todo_id: int) -> FuncToolResult:
        """Read the full detail of a single todo item by id.

        Args:
            todo_id: Integer id from ``todo_list`` / ``todo_write`` output.

        Returns ``{id, title, status, content}``. Errors if no list exists
        or the id is unknown.
        """
        todo_list = self.storage.get_todo_list()
        if not todo_list:
            return FuncToolResult(success=0, error="No todo list found")
        item = todo_list.get_item(todo_id)
        if not item:
            return FuncToolResult(success=0, error=f"Todo item with id {todo_id} not found")
        return FuncToolResult(
            result={
                "id": item.id,
                "title": item.title,
                "status": item.status.value,
                "content": item.content,
            }
        )

    def todo_write(self, todos_json: str) -> FuncToolResult:
        """Append new todo items to the current list (does NOT overwrite).

        ``todo_write`` is incremental: pass only the items you want to add.
        Existing items stay untouched — use ``todo_update`` to change a
        single item's status, or call this again later to add more.

        Args:
            todos_json: JSON string of list of dicts with ``title`` (≤ 8
                words, used in the sidebar) and ``content`` (full task
                description). ``status`` is NOT accepted — new items are
                always created in the ``pending`` state.

                Example: '[{"title": "Query orders table", "content":
                "Run SELECT COUNT(*) FROM orders WHERE created_at >= ..."}]'

        Behaviour:
            * The batch is validated as a whole: if any item fails (title
              empty / > 8 words / content empty / malformed JSON), the
              **entire call** is rejected and the existing list is not
              mutated. This avoids partial-write states the LLM has to
              reason about.
            * Each new item gets the next incrementing integer id; ids stay
              monotonic across appends and across resume.
            * Returns the same overview shape as ``todo_list``: a list of
              ``{id, title, status}`` for every item now in the list.
        """
        try:
            todos = json.loads(todos_json)
        except (json.JSONDecodeError, TypeError):
            return FuncToolResult(success=0, error="Invalid JSON format for todos")

        if not isinstance(todos, list) or not todos:
            return FuncToolResult(success=0, error="Cannot append todo list: expected a non-empty JSON array")

        # Validate up-front so nothing is appended on partial failure.
        prepared: List[tuple[str, str]] = []
        for idx, todo_item in enumerate(todos):
            if not isinstance(todo_item, dict):
                return FuncToolResult(success=0, error=f"Item {idx} is not an object")
            # Type-check before ``.strip()``: an LLM emitting ``{"title": 123}``
            # would otherwise hit ``AttributeError`` on ``(123 or "").strip()``
            # and bubble out of the tool instead of returning a clean error.
            raw_title = todo_item.get("title")
            raw_content = todo_item.get("content")
            if raw_title is not None and not isinstance(raw_title, str):
                return FuncToolResult(success=0, error=f"Item {idx}: 'title' must be a string")
            if raw_content is not None and not isinstance(raw_content, str):
                return FuncToolResult(success=0, error=f"Item {idx}: 'content' must be a string")
            title = (raw_title or "").strip()
            content = (raw_content or "").strip()
            if not title:
                return FuncToolResult(success=0, error=f"Item {idx}: 'title' is required")
            if not content:
                return FuncToolResult(success=0, error=f"Item {idx}: 'content' is required")
            words = [w for w in re.split(r"\s+", title) if w]
            if len(words) > TITLE_WORD_LIMIT:
                return FuncToolResult(
                    success=0,
                    error=f"Item {idx}: title must be {TITLE_WORD_LIMIT} words or fewer (got {len(words)})",
                )
            prepared.append((title, content))

        # Start from the existing list (read-modify-write append semantics).
        # ``get_todo_list`` lazily loads from disk if needed, so a fresh
        # PlanTool instance after resume still sees prior items.
        todo_list = self.storage.get_todo_list() or TodoList()

        for title, content in prepared:
            try:
                todo_list.add_item(title=title, content=content)
            except ValidationError as exc:
                # The pre-flight check above already mirrors the model
                # validators, but if a constraint diverges we fail loudly
                # rather than persisting half a batch.
                return FuncToolResult(success=0, error=f"Validation error while appending: {exc}")
            logger.info("Appended pending todo: %s", title)

        if not self.storage.save_list(todo_list):
            return FuncToolResult(success=0, error="Failed to save todo list to storage")

        return FuncToolResult(
            result={
                "message": f"Appended {len(prepared)} item(s); list now has {len(todo_list.items)} item(s).",
                "items": self._overview_items(todo_list),
            }
        )

    def todo_update(self, todo_id: int, status: str) -> FuncToolResult:
        """Update a todo item's status.

        Recommended flow: ``pending`` → ``in_progress`` → ``completed``.
        Call this with ``in_progress`` immediately before starting work on
        a todo, and again with ``completed`` (or ``failed``) when done, so
        the sidebar reflects what you are currently doing.

        Args:
            todo_id: Integer id of the todo item to update.
            status: One of ``pending``, ``in_progress``, ``completed``,
                ``failed``.
        """
        try:
            status_enum = TodoStatus(status.lower())
        except ValueError:
            valid = ", ".join(s.value for s in TodoStatus)
            return FuncToolResult(success=0, error=f"Invalid status '{status}'. Must be one of: {valid}")

        todo_list = self.storage.get_todo_list()
        if not todo_list:
            return FuncToolResult(success=0, error="No todo list found")

        todo_item = todo_list.get_item(todo_id)
        if not todo_item:
            return FuncToolResult(success=0, error=f"Todo item with id {todo_id} not found")

        todo_list.update_item_status(todo_id, status_enum)
        if not self.storage.save_list(todo_list):
            return FuncToolResult(success=0, error="Failed to save updated todo list to storage")

        updated_item = todo_list.get_item(todo_id)
        return FuncToolResult(
            result={
                "message": f"Successfully updated todo item to '{status_enum.value}' status",
                "updated_item": {
                    "id": updated_item.id,
                    "title": updated_item.title,
                    "status": updated_item.status.value,
                    "content": updated_item.content,
                },
            }
        )


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
                        "  2. Call todo_write with [{title, content}] for each concrete actionable "
                        "step (title ≤ 8 words; content is the detailed instruction).\n"
                        "  3. Before starting work on a step, call todo_update(id, 'in_progress') "
                        "so the sidebar reflects what you are doing.\n"
                        "  4. Execute the step by calling the relevant tools "
                        "(grep / read_file / list_tables / read_query / write_file — whatever "
                        "the step requires).\n"
                        "  5. After completing the step, call todo_update(id, 'completed'); "
                        "use 'failed' instead if it could not be finished. Then move to the "
                        "next step.\n"
                        "  6. Continue executing steps without asking the user for permission, "
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
