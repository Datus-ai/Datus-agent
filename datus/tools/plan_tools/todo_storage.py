import json
import os
from pathlib import Path
from typing import Dict, Optional

from datus.utils.loggings import get_logger

from .todo_models import TodoList

logger = get_logger(__name__)


class TodoStorage:
    """Simple JSON file-based storage for todo lists"""

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize storage with directory path

        Args:
            storage_dir: Directory to store todo files. Defaults to ~/.datus/todos/
        """
        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.datus/todos")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"TodoStorage initialized with directory: {self.storage_dir}")

    def _get_file_path(self, list_id: str) -> Path:
        """Get the file path for a given list ID"""
        return self.storage_dir / f"{list_id}.json"

    def save_list(self, todo_list: TodoList) -> bool:
        """Save a todo list to file

        Args:
            todo_list: The todo list to save

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_file_path(todo_list.list_id)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(todo_list.model_dump(), f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"Saved todo list {todo_list.list_id} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save todo list {todo_list.list_id}: {str(e)}")
            return False

    def load_list(self, list_id: str) -> Optional[TodoList]:
        """Load a todo list from file

        Args:
            list_id: ID of the list to load

        Returns:
            TodoList if found and valid, None otherwise
        """
        try:
            file_path = self._get_file_path(list_id)
            if not file_path.exists():
                logger.debug(f"Todo list file not found: {file_path}")
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            todo_list = TodoList(**data)
            logger.debug(f"Loaded todo list {list_id} with {len(todo_list.items)} items")
            return todo_list
        except Exception as e:
            logger.error(f"Failed to load todo list {list_id}: {str(e)}")
            return None

    def list_all_lists(self) -> Dict[str, TodoList]:
        """Load all todo lists from storage

        Returns:
            Dictionary mapping list_id to TodoList objects
        """
        all_lists = {}

        try:
            # Find all .json files in storage directory
            for file_path in self.storage_dir.glob("*.json"):
                list_id = file_path.stem  # filename without extension
                todo_list = self.load_list(list_id)
                if todo_list:
                    all_lists[list_id] = todo_list

            logger.debug(f"Loaded {len(all_lists)} todo lists from storage")
            return all_lists
        except Exception as e:
            logger.error(f"Failed to list all todo lists: {str(e)}")
            return {}

    def delete_list(self, list_id: str) -> bool:
        """Delete a todo list file

        Args:
            list_id: ID of the list to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            file_path = self._get_file_path(list_id)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted todo list {list_id} from {file_path}")
                return True
            else:
                logger.debug(f"Todo list {list_id} file not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Failed to delete todo list {list_id}: {str(e)}")
            return False

    def update_item(self, list_id: str, item_id: str, status: str) -> bool:
        """Update a specific todo item status

        Args:
            list_id: ID of the todo list
            item_id: ID of the todo item
            status: New status for the item

        Returns:
            True if successfully updated, False otherwise
        """
        try:
            todo_list = self.load_list(list_id)
            if not todo_list:
                logger.error(f"Todo list {list_id} not found for item update")
                return False

            from .todo_models import TodoStatus

            status_enum = TodoStatus(status)

            if todo_list.update_item_status(item_id, status_enum):
                success = self.save_list(todo_list)
                if success:
                    logger.debug(f"Updated item {item_id} in list {list_id} to status {status}")
                return success
            else:
                logger.error(f"Todo item {item_id} not found in list {list_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to update item {item_id} in list {list_id}: {str(e)}")
            return False

    def get_storage_info(self) -> Dict[str, any]:
        """Get information about the storage directory

        Returns:
            Dictionary with storage statistics
        """
        try:
            json_files = list(self.storage_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in json_files)

            return {
                "storage_directory": str(self.storage_dir),
                "total_lists": len(json_files),
                "total_size_bytes": total_size,
                "exists": self.storage_dir.exists(),
                "is_writable": os.access(self.storage_dir, os.W_OK),
            }
        except Exception as e:
            logger.error(f"Failed to get storage info: {str(e)}")
            return {"storage_directory": str(self.storage_dir), "error": str(e)}
