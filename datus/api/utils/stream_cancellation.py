"""Lightweight SSE stream cancellation token management."""

import asyncio
import time

# Maximum number of concurrent streams tracked before oldest entries are evicted.
_MAX_TOKENS = 1024
# Time-to-live (seconds) for idle tokens; prevents unbounded growth from leaked entries.
_TOKEN_TTL = 3600


class _BoundedTokenMap:
    """Bounded TTL+LRU map to prevent unbounded memory growth."""

    def __init__(self, maxsize: int, ttl: float) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._data: dict[str, asyncio.Event] = {}
        self._expiry: dict[str, float] = {}

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, exp in self._expiry.items() if exp <= now]
        for sid in expired:
            self._data.pop(sid, None)
            self._expiry.pop(sid, None)

    def __setitem__(self, stream_id: str, event: asyncio.Event) -> None:
        self._evict_expired()
        if stream_id in self._data:
            self._data.move_to_end(stream_id)
        self._data[stream_id] = event
        self._expiry[stream_id] = time.monotonic() + self._ttl
        while len(self._data) > self._maxsize:
            sid, _ = self._data.popitem(last=False)
            self._expiry.pop(sid, None)

    def get(self, stream_id: str) -> asyncio.Event | None:
        self._evict_expired()
        return self._data.get(stream_id)

    def pop(self, stream_id: str, default=None):
        self._expiry.pop(stream_id, None)
        return self._data.pop(stream_id, default)


_tokens = _BoundedTokenMap(maxsize=_MAX_TOKENS, ttl=_TOKEN_TTL)


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
