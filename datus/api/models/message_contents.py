# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""User-authored chat content, independent of model transport formats."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class MarkdownPayload(BaseModel):
    content: str = Field(max_length=200_000)


class AttachmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attachmentId: str = Field(pattern=r"^[a-f0-9]{32}$")
    mimeType: str
    createdAt: str = Field(max_length=40)


class ImagePayload(AttachmentPayload):
    mimeType: Literal["image/png", "image/jpeg", "image/webp"]
    width: int = Field(gt=0, le=2048)
    height: int = Field(gt=0, le=2048)


class MarkdownMessageContent(BaseModel):
    type: Literal["markdown"]
    payload: MarkdownPayload


class ImageMessageContent(BaseModel):
    type: Literal["image"]
    payload: ImagePayload


ChatMessageContent = Annotated[MarkdownMessageContent | ImageMessageContent, Field(discriminator="type")]


class PreparedChatMessage(str):
    """Carry an API submission identity through the existing text-only queue."""

    content_id: str

    def __new__(cls, text: str, content_id: str):
        value = super().__new__(cls, text)
        value.content_id = content_id
        return value

    def __getnewargs__(self):
        return str(self), self.content_id
