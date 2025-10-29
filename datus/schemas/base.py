# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TABLE_TYPE = Literal["table", "view", "mv", "full"]


def parse_table_type_by_db(db_table_type: str) -> TABLE_TYPE:
    db_table_type = db_table_type.upper()
    if db_table_type in ("TABLE", "BASE TABLE"):
        return "table"
    if db_table_type == "VIEW":
        return "view"
    return "mv"


class BaseInput(BaseModel):
    """
    Base class for all node input data validation.
    Provides common validation functionality for node inputs.
    """

    # context: Optional[List[str]] = Field(default=[], description="Optional context information for the input")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with an optional default value."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)

    def to_str(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_str(cls, json_str: str) -> "BaseInput":
        return cls.model_validate_json(json_str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseInput":
        """Create SqlTask instance from dictionary."""
        return cls.model_validate(data)

    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model


class BaseResult(BaseModel):
    """
    Base class for all node result data validation.
    Provides common validation functionality for node results.
    """

    success: bool = Field(..., description="Indicates whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if operation failed")

    # Action history and execution stats for agentic nodes
    action_history: Optional[List[dict]] = Field(
        default=None, description="Complete history of tool calls and actions during execution"
    )
    execution_stats: Optional[dict] = Field(
        default=None, description="Execution statistics (tokens, tools called, duration, etc.)"
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with an optional default value."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)

    def to_str(self) -> str:
        """Convert the result to a string representation, including all nested objects."""
        return self.model_dump_json()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_str(cls, json_str: str) -> "BaseResult":
        return cls.model_validate_json(json_str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseInput":
        """Create SqlTask instance from dictionary."""
        return cls.model_validate(data)

    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model


class CommonData(BaseModel):
    """
    Base class for all common data validation.
    Provides common validation functionality for common data.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with an optional default value."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)

    def to_str(self) -> str:
        """Convert the result to a string representation, including all nested objects."""
        return self.model_dump_json()

    @classmethod
    def from_str(cls, json_str: str) -> "BaseResult":
        return cls.model_validate_json(json_str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseInput":
        """Create SqlTask instance from dictionary."""
        return cls.model_validate(data)

    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model
