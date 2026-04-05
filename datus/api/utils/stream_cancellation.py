"""Lightweight SSE stream cancellation token management."""

import asyncio

_tokens: dict[str, asyncio.Event] = {}


def create_cancel_token(stream_id: str) -> asyncio.Event:
    """Create a cancellation token for a stream."""
    event = asyncio.Event()
    _tokens[stream_id] = event
    return event


def cancel_stream(stream_id: str) -> bool:
    """Signal cancellation for a stream. Returns True if the token existed."""
    event = _tokens.get(stream_id)
    if event:
        event.set()
        return True
    return False


def cleanup_cancel_token(stream_id: str) -> None:
    """Remove a cancellation token after stream ends."""
    _tokens.pop(stream_id, None)
