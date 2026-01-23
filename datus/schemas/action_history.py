# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ActionRole(str, Enum):
    """Role of the action creator
    - SYSTEM: The system prompt use this role
    - ASSISTANT: The AI assistant role is used to create the response for the user.
    - USER: The user role is used to call the node or send the message
    - TOOL: The tool role is used to create the input for the assistant, it's a MCP tool.
    - WORKFLOW: The workflow role is used for workflow execution
    - INTERACTION: The interaction role is used for user input requests from hooks
    """

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    WORKFLOW = "workflow"
    INTERACTION = "interaction"


class ActionStatus(str, Enum):
    """Status of the action"""

    PROCESSING = "processing"
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

    @classmethod
    def create_action(
        cls,
        role: ActionRole,
        action_type: str,
        messages: str,
        input_data: dict,
        output_data: dict = None,
        status: ActionStatus = ActionStatus.PROCESSING,
    ) -> "ActionHistory":
        import uuid

        return cls(
            action_id=str(uuid.uuid4()),
            role=role,
            messages=messages,
            action_type=action_type,
            input=input_data,
            output=output_data,
            status=status,
        )

    def is_done(self) -> bool:
        return self.status == ActionStatus.SUCCESS or self.status == ActionStatus.FAILED

    def function_name(self) -> str:
        return "" if not self.input else str(self.input.get("function_name", "unknown"))

    class Config:
        use_enum_values = True


class ActionHistoryManager:
    """Manager for action history during streaming execution"""

    def __init__(self):
        self.actions: List[ActionHistory] = []
        self.current_action_id: Optional[str] = None

    def add_action(self, action: ActionHistory) -> None:
        """Add an action to the history, preventing duplicates by action_id"""
        # Check if action with same ID already exists (prevents duplicates from retries)
        if self.find_action_by_id(action.action_id):
            logger.debug(f"Action with id {action.action_id} already exists, skipping duplicate")
            return

        logger.debug(f"Adding action: {action}")
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
        logger.debug(f"Updating action: {action_id} with kwargs: {kwargs}")
        if action:
            for key, value in kwargs.items():
                if hasattr(action, key):
                    setattr(action, key, value)
            return True
        return False
