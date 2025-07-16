from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ActionRole(str, Enum):
    """Role of the action creator
    - SYSTEM: The system prompt use this role
    - ASSISTANT: The AI assistant role is used to create the response for the user.
    - USER: The user role is used to call the node or send the message
    - TOOL: The tool role is used to create the input for the assistant, it's a MCP tool.
    - WORKFLOW: The workflow role is
    """

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    WORKFLOW = "workflow"


class ActionStatus(str, Enum):
    """Status of the action"""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class ActionHistory(BaseModel):
    """History of actions during LLM execution with streaming support"""

    action_id: str = Field(..., description="Unique identifier for the action")
    role: ActionRole = Field(..., description="Role of the action creator")
    messages: str = Field(default="", description="Thought or reasoning behind the action (AI or human message)")
    action_type: str = Field(..., description="Type of action performed (NodeType / MCP tool name / message)")
    input: Any = Field(default=None, description="Input data for the action")
    output: Any = Field(default=None, description="Output data from the action")
    status: ActionStatus = Field(..., description="Status of the action")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time of the action")
    end_time: Optional[datetime] = Field(default=None, description="End time of the action")

    class Config:
        use_enum_values = True


class ActionHistoryManager:
    """Manager for action history during streaming execution"""

    def __init__(self):
        self.actions: List[ActionHistory] = []
        self.current_action_id: Optional[str] = None

    def add_action(self, action: ActionHistory) -> None:
        """Add an action to the history"""
        self.actions.append(action)
        self.current_action_id = action.action_id

    def update_current_action(self, **kwargs) -> None:
        """Update the current action with new data"""
        if self.current_action_id and self.actions:
            current_action = self.actions[-1]
            if current_action.action_id == self.current_action_id:
                for key, value in kwargs.items():
                    if hasattr(current_action, key):
                        setattr(current_action, key, value)

    def get_actions(self) -> List[ActionHistory]:
        """Get all actions in the history"""
        return self.actions

    def clear(self) -> None:
        """Clear all actions"""
        self.actions.clear()
        self.current_action_id = None

    def find_action_by_id(self, action_id: str) -> Optional[ActionHistory]:
        """Find an action by its action_id"""
        for action in self.actions:
            if action.action_id == action_id:
                return action
        return None

    def update_action_by_id(self, action_id: str, **kwargs) -> bool:
        """Update an action by its action_id"""
        action = self.find_action_by_id(action_id)
        if action:
            for key, value in kwargs.items():
                if hasattr(action, key):
                    setattr(action, key, value)
            return True
        return False
