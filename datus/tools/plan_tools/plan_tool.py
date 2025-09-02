from typing import List, Optional

from agents import Tool

from datus.tools.tools import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

from .todo_models import TodoStatus
from .todo_storage import TodoStorage

logger = get_logger(__name__)


class PlanTool:
    """Main tool for todo list management with read, write, and update capabilities"""

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the plan tool

        Args:
            storage_dir: Directory where todo files are stored
        """
        self.storage = TodoStorage(storage_dir)

    def available_tools(self) -> List[Tool]:
        """Get list of available plan tools, similar to DBFuncTool"""
        methods_to_convert = [
            self.todo_read,
            self.todo_write,
            self.todo_update,
        ]

        bound_tools = []
        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    def todo_read(self, list_id: Optional[str] = None) -> FuncToolResult:
        """Read todo lists from storage

        Args:
            list_id: Optional specific list ID to read. If None, reads all lists.

        Returns:
            TodoReadResult with the retrieved todo lists
        """
        try:
            logger.info(f"Reading todos with list_id: {list_id}")

            if list_id:
                # Read specific list
                todo_list = self.storage.load_list(list_id)
                if todo_list:
                    return FuncToolResult(
                        result={
                            "message": f"Successfully retrieved todo list '{todo_list.name or list_id}'",
                            "lists": [todo_list.model_dump()],
                            "total_lists": 1,
                        }
                    )
                else:
                    return FuncToolResult(success=0, error=f"Todo list with ID '{list_id}' not found")
            else:
                # Read all lists
                all_lists_dict = self.storage.list_all_lists()
                all_lists = list(all_lists_dict.values())

                return FuncToolResult(
                    result={
                        "message": f"Successfully retrieved {len(all_lists)} todo list(s)",
                        "lists": [todo_list.model_dump() for todo_list in all_lists],
                        "total_lists": len(all_lists),
                    }
                )

        except Exception as e:
            error_message = f"Failed to read todos: {str(e)}"
            logger.error(error_message)
            return FuncToolResult(success=0, error=error_message)

    def todo_write(self, todos: List[str], list_name: Optional[str] = None) -> FuncToolResult:
        """Create a new todo list from todo items

        Args:
            todos: List of todo item contents
            list_name: Optional name for the todo list

        Returns:
            FuncToolResult with the created todo list
        """
        try:
            # Clean up todo items
            todos = [item.strip() for item in todos if item and str(item).strip()]

            logger.info(f"Creating todo list with {len(todos)} items, name: {list_name}")

            if not todos:
                return FuncToolResult(success=0, error="Cannot create todo list: no todo items provided")
            # Import here to avoid circular imports
            from .todo_models import TodoList

            # Create new todo list (this will be the only one)
            todo_list = TodoList(name=list_name or "")

            # Add all todo items
            for todo_content in todos:
                if todo_content.strip():  # Skip empty items
                    todo_list.add_item(todo_content.strip())

            # Save to storage
            if self.storage.save_list(todo_list):
                logger.info(f"Successfully created todo list {todo_list.list_id} with {len(todo_list.items)} items")
                return FuncToolResult(
                    result={
                        "message": (
                            f"Successfully created todo list '{todo_list.name or todo_list.list_id}' "
                            f"with {len(todo_list.items)} items"
                        ),
                        "list_id": todo_list.list_id,
                        "todo_list": todo_list.model_dump(),
                    }
                )
            else:
                return FuncToolResult(success=0, error="Failed to save todo list to storage")

        except Exception as e:
            error_message = f"Failed to create todo list: {str(e)}"
            logger.error(error_message)
            return FuncToolResult(success=0, error=error_message)

    def todo_update(self, list_id: str, todo_id: str, status: str) -> FuncToolResult:
        """Update the status of a specific todo item

        Args:
            list_id: ID of the todo list containing the item
            todo_id: ID of the todo item to update
            status: New status ('completed', 'pending', or 'failed')

        Returns:
            TodoUpdateResult with the update result
        """
        try:
            logger.info(f"Updating todo item {todo_id} in list {list_id} to status {status}")

            # Validate status
            try:
                status_enum = TodoStatus(status.lower())
            except ValueError:
                return FuncToolResult(
                    success=0, error=f"Invalid status '{status}'. Must be 'completed', 'pending', or 'failed'"
                )

            # Load the todo list
            todo_list = self.storage.load_list(list_id)
            if not todo_list:
                return FuncToolResult(success=0, error=f"Todo list with ID '{list_id}' not found")

            # Find and update the item
            todo_item = todo_list.get_item(todo_id)
            if not todo_item:
                return FuncToolResult(success=0, error=f"Todo item with ID '{todo_id}' not found in list '{list_id}'")

            # Update the item status
            if todo_list.update_item_status(todo_id, status_enum):
                # Save the updated list
                if self.storage.save_list(todo_list):
                    updated_item = todo_list.get_item(todo_id)
                    logger.info(f"Successfully updated todo item {todo_id} to status {status}")
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

        except Exception as e:
            error_message = f"Failed to update todo status: {str(e)}"
            logger.error(error_message)
            return FuncToolResult(success=0, error=error_message)
