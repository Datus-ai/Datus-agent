"""Data models for chat and tool call functionality."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Structured result from tool execution."""

    success: Literal[0, 1] = Field(..., description="1 for success, 0 for failure")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    result: Optional[Any] = Field(None, description="Tool execution result data if successful")


class ToolResultInput(BaseModel):
    """Input model for receiving tool execution results from frontend."""

    session_id: Optional[str] = Field(None, description="Session ID for the active chat task")
    call_tool_id: str = Field(..., description="Unique identifier for the tool call", examples=["tc_abc123"])
    tool_result: ToolResult = Field(
        ...,
        description="Tool execution result containing success status and data/error",
        examples=[
            {
                "success": 1,
                "data": {"content": "def main():\n    print('Hello')", "size": 28},
            },
            {"success": 0, "error": "File not found: src/missing.py"},
        ],
    )


class ToolResultData(BaseModel):
    """Data for tool result submission response."""

    call_tool_id: str = Field(..., description="Unique identifier for the tool call")
    status: str = Field(..., description="Status of the tool result submission", examples=["received"])
