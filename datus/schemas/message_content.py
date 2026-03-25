# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageContent(BaseModel):
    """Message content with type and payload."""

    type: str = Field(..., description="Content type (markdown, code, csv, thinking, call-tool, etc.)")
    payload: Dict[str, Any] = Field(..., description="Content payload")


class MessagePayload(BaseModel):
    """Message payload for stdin/stdout JSON communication."""

    message_id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user, assistant)")
    content: List[MessageContent] = Field(default_factory=list, description="Message content list")
    depth: int = Field(default=0, description="Nesting depth (0=main, 1=sub-agent)")
    parent_action_id: Optional[str] = Field(default=None, description="Parent action ID for sub-agent grouping")
