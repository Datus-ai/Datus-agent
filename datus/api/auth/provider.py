"""Auth provider protocol — DatusService layer depends only on this interface."""

from typing import Awaitable, Callable, Protocol, runtime_checkable

from fastapi import Request

from datus.api.auth.context import AppContext

EvictCallback = Callable[[str], Awaitable[None]]  # project_id -> evict


@runtime_checkable
class AuthProvider(Protocol):
    """Authentication plugin interface."""

    async def authenticate(self, request: Request) -> AppContext:
        """Authenticate a request and return AppContext (with loaded AgentConfig)."""
        ...

    def on_evict(self, callback: EvictCallback) -> None:
        """Register a cache eviction callback. Called when config changes are detected."""
        ...
