from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


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
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class TodoList(BaseModel):
    """Collection of todo items with metadata"""

    list_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the todo list")
    name: str = Field(default="", description="Name/title of the todo list")
    items: List[TodoItem] = Field(default_factory=list, description="List of todo items")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    def add_item(self, content: str) -> TodoItem:
        """Add a new todo item to the list"""
        item = TodoItem(content=content)
        self.items.append(item)
        self.updated_at = datetime.now()
        return item

    def get_item(self, item_id: str) -> Optional[TodoItem]:
        """Get a todo item by ID"""
        return next((item for item in self.items if item.id == item_id), None)

    def update_item_status(self, item_id: str, status: TodoStatus) -> bool:
        """Update the status of a todo item"""
        item = self.get_item(item_id)
        if item:
            item.status = status
            item.updated_at = datetime.now()
            self.updated_at = datetime.now()
            return True
        return False
