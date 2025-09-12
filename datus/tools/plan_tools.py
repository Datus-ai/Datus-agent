"""
Simplified plan tools - merged from multiple files into single module
"""
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from agents import Tool
from pydantic import BaseModel, Field

from datus.tools.tools import FuncToolResult, trans_to_function_tool
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
        """Update the status of a todo item"""
        item = self.get_item(item_id)
        if item:
            item.status = status
            return True
        return False


class TodoStorage:
    """Simple in-memory storage for todo lists"""

    def __init__(self):
        """Initialize storage"""
        self._todo_list: Optional[TodoList] = None

    def save_list(self, todo_list: TodoList) -> bool:
        """Save the todo list to memory"""
        self._todo_list = todo_list
        return True

    def get_todo_list(self) -> Optional[TodoList]:
        """Get the todo list from memory"""
        return self._todo_list

    def clear_all(self) -> None:
        """Clear the todo list from memory"""
        self._todo_list = None


class PlanTool:
    """Main tool for todo list management with read, write, and update capabilities"""

    def __init__(self):
        """Initialize the plan tool"""
        self.storage = TodoStorage()

    def available_tools(self) -> List[Tool]:
        """Get list of available plan tools"""
        methods_to_convert = [
            self.todo_read,
            self.todo_write,
            self.todo_update_pending,
            self.todo_update_completed,
            self.todo_update_failed,
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

    def todo_write(self, todos: List[str]) -> FuncToolResult:
        """Create or update the todo list from todo items"""
        todos = [item.strip() for item in todos if item and item.strip()]

        if not todos:
            return FuncToolResult(success=0, error="Cannot create todo list: no todo items provided")

        todo_list = TodoList()
        for todo_content in todos:
            if todo_content.strip():
                todo_list.add_item(todo_content.strip())

        if self.storage.save_list(todo_list):
            return FuncToolResult(
                result={
                    "message": f"Successfully saved todo list with {len(todo_list.items)} items",
                    "todo_list": todo_list.model_dump(),
                }
            )
        else:
            return FuncToolResult(success=0, error="Failed to save todo list to storage")

    def todo_update_pending(self, todo_id: str) -> FuncToolResult:
        """Update the status of a specific todo item to 'pending' - this triggers manual confirmation"""
        return self._update_todo_status(todo_id, "pending")

    def todo_update_completed(self, todo_id: str) -> FuncToolResult:
        """Update the status of a specific todo item to 'completed'"""
        return self._update_todo_status(todo_id, "completed")

    def todo_update_failed(self, todo_id: str) -> FuncToolResult:
        """Update the status of a specific todo item to 'failed'"""
        return self._update_todo_status(todo_id, "failed")

    def _update_todo_status(self, todo_id: str, status: str) -> FuncToolResult:
        """Internal method to update todo item status"""
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
