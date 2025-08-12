from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from datus.schemas.node_models import BaseInput, BaseResult


class AgenticInput(BaseInput):
    """Input for AgenticNode - supports multi-turn conversation"""

    message: str = Field(..., description="User message/prompt")
    session_id: str = Field(..., description="Session identifier for multi-turn conversation")
    database_name: Optional[str] = Field(default=None, description="Target database name")
    context_compression: bool = Field(default=True, description="Enable context compression")
    max_context_length: int = Field(default=8000, description="Maximum context length before compression")

    # Optional workflow integration
    workflow_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional workflow context")


class AgenticResult(BaseResult):
    """Result from AgenticNode execution"""

    response: str = Field(..., description="AI assistant response")
    sql: Optional[str] = Field(default=None, description="Generated SQL query (if any)")
    context_compressed: bool = Field(default=False, description="Whether context was compressed during execution")
    session_id: str = Field(..., description="Session identifier")

    # Additional metadata
    actions_taken: List[str] = Field(default_factory=list, description="List of actions/tools used")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")


class CliContextState(BaseModel):
    """State object for CLI context management"""

    session_id: str = Field(..., description="Session identifier")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation turns")
    context_summary: Optional[str] = Field(default=None, description="Compressed context summary")
    database_context: Optional[str] = Field(default=None, description="Current database context")
    last_sql_queries: List[str] = Field(default_factory=list, description="Recent SQL queries")
    created_at: str = Field(..., description="Session creation timestamp")
    last_updated: str = Field(..., description="Last update timestamp")
    total_turns: int = Field(default=0, description="Total conversation turns")

    class Config:
        arbitrary_types_allowed = True
