# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Host-owned attachment resolution for structured user messages."""

from typing import Awaitable, Callable

from fastapi import HTTPException, Request

from datus.api.models.chat_models import InsertMessageInput
from datus.api.models.message_contents import ChatMessageContent

MessageResolver = Callable[[Request, list[ChatMessageContent]], Awaitable[tuple[str, list[ChatMessageContent]]]]
_resolver: MessageResolver | None = None


def set_chat_message_resolver(resolver: MessageResolver | None) -> None:
    global _resolver
    _resolver = resolver


async def prepare_chat_messages(http_request: Request, request) -> None:
    if request.messages is None:
        return

    if _resolver is not None:
        request.message, request.messages = await _resolver(http_request, request.messages)
    elif any(part.type == "image" for part in request.messages):
        raise HTTPException(status_code=400, detail="This host does not support chat attachments")
    else:
        request.message = "\n\n".join(part.payload.content for part in request.messages)

    if isinstance(request, InsertMessageInput) and len(request.message) > 4000:
        raise HTTPException(status_code=422, detail="Prepared insert message must be at most 4000 characters")
