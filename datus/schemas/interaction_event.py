# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Formal data model for user interaction events."""

from typing import Dict, List

from pydantic import BaseModel, Field


class InteractionEvent(BaseModel):
    """A single interaction question to present to the user.

    Used as the structured element type for ``ActionHistory.input`` when
    ``role == ActionRole.INTERACTION``.  The broker stores events as
    ``{"events": [event.model_dump(), ...]}`` in ``ActionHistory.input``.
    """

    title: str = Field(default="", description="Tab header label (1-2 words, e.g. 'Permission', 'Plan')")
    content: str = Field(default="", description="Question or body text")
    content_type: str = Field(default="markdown", description="Rendering hint: markdown | sql | yaml | text")
    choices: Dict[str, str] = Field(
        default_factory=dict, description="Ordered {key: display_text}, empty = free-text only"
    )
    default_choice: str = Field(default="", description="Pre-selected choice key")
    allow_free_text: bool = Field(default=False, description="Append a free-text input row below choices")
    multi_select: bool = Field(default=False, description="Checkbox-style multi-select mode")

    @classmethod
    def from_broker_input(cls, input_data: dict) -> List["InteractionEvent"]:
        """Deserialize ``ActionHistory.input`` into a list of InteractionEvent."""
        if not input_data:
            return []
        return [cls(**ev) for ev in input_data.get("events", [])]
