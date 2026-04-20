"""Data models for success-story persistence endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class SuccessStoryInput(BaseModel):
    """Input model for saving a success story.

    The CSV row is keyed by ``subagent_id``; the server resolves the id to
    a canonical, safe directory name before writing.
    """

    session_id: str = Field(..., description="Session ID that produced the SQL", examples=["sess_01HX..."])
    sql: str = Field(..., description="The generated SQL query to archive")
    user_message: str = Field(..., description="The user's original natural-language question")
    subagent_id: Optional[str] = Field(
        None,
        description=(
            "Subagent ID (builtin name, agentic_nodes key, or custom DB UUID). Defaults to 'default' when omitted."
        ),
        examples=["gen_sql"],
    )
    session_link: Optional[str] = Field(
        None,
        description=(
            "Fully qualified URL that reopens the session in the UI. "
            "Clients should pass the URL they would use to reopen the session; "
            "leave empty if not applicable."
        ),
    )


class SuccessStoryData(BaseModel):
    """Data returned after a success story is persisted."""

    csv_path: str = Field(..., description="Absolute path to the CSV file that was appended")
    subagent_name: str = Field(..., description="Canonical subagent directory name used for storage")
    session_id: str = Field(..., description="Echoed session ID")
    session_link: Optional[str] = Field(None, description="Echoed session link, if one was provided")
    timestamp: str = Field(..., description="UTC timestamp of the write (YYYY-MM-DD HH:MM:SS)")
