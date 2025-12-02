# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Simplified plan tools - merged from multiple files into single module
"""

from enum import Enum
from typing import List, Optional
from uuid import uuid4

from agents import SQLiteSession, Tool
from pydantic import BaseModel, Field

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

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
    """In-memory storage for todo lists to avoid conflicts with agents library session"""

    def __init__(self, session: SQLiteSession):
        """Initialize storage with session"""
        self.session = session
        self._current_todo_list: Optional[TodoList] = None

    def save_list(self, todo_list: TodoList) -> bool:
        """Save the todo list to in-memory storage"""
        try:
            self._current_todo_list = todo_list
            logger.debug(f"Saved todo list to memory with {len(todo_list.items)} items")
            return True
        except Exception as e:
            logger.error(f"Failed to save todo list to memory: {e}")
            return False

    def get_todo_list(self) -> Optional[TodoList]:
        """Get the todo list from in-memory storage"""
        return self._current_todo_list

    def clear_all(self) -> None:
        """Clear the todo list from in-memory storage"""
        try:
            self._current_todo_list = None
            logger.debug("Cleared todo list from memory")
        except Exception as e:
            logger.error(f"Failed to clear todo list from memory: {e}")

    def has_todo_list(self) -> bool:
        """Check if storage has a todo list"""
        return self._current_todo_list is not None


class PlanTool:
    """Main tool for todo list management with read, write, and update capabilities"""

    def __init__(self, session: SQLiteSession):
        """Initialize the plan tool with session"""
        self.storage = SessionTodoStorage(session)

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
        """Create or update the todo list from todo items with explicit status

        Args:
            todos_json: JSON string of list of dicts with 'content' and 'status' keys.
                       Status can be 'pending' or 'completed'.

                       IMPORTANT: In replan mode, only include steps that are actually needed:
                       - 'completed': Steps that were actually executed and finished
                       - 'pending': Steps that still need to be executed (existing or new)
                       - DISCARD: Don't include steps that are no longer needed

                       Example: '[{"content": "Query database", "status": "completed"},
                                {"content": "Generate report", "status": "pending"}]'
        """
        try:
            import json

            todos = json.loads(todos_json)
        except (json.JSONDecodeError, TypeError):
            return FuncToolResult(success=0, error="Invalid JSON format for todos")

        if not todos:
            return FuncToolResult(success=0, error="Cannot create todo list: no todo items provided")

        todo_list = TodoList()

        # Create todo list with LLM-specified status
        for todo_item in todos:
            content = todo_item.get("content", "").strip()
            status = todo_item.get("status", "pending").lower()

            if not content:
                continue

            if status == "completed":
                # Create completed item - should only be for actually executed steps
                new_item = TodoItem(content=content, status=TodoStatus.COMPLETED)
                todo_list.items.append(new_item)
                logger.info(f"Keeping completed step: {content}")
            else:
                # Create pending step - for steps that still need execution
                todo_list.add_item(content)
                logger.info(f"Added pending step: {content}")

        if self.storage.save_list(todo_list):
            completed_count = sum(1 for item in todo_list.items if item.status == TodoStatus.COMPLETED)
            return FuncToolResult(
                result={
                    "message": (
                        f"Successfully saved todo list with {len(todo_list.items)} items "
                        f"({completed_count} already completed)"
                    ),
                    "todo_list": todo_list.model_dump(),
                }
            )
        else:
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
