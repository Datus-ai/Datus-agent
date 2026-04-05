"""Open-source default auth provider — bypass authentication."""

from fastapi import Request

from datus.api.auth.context import AppContext
from datus.api.auth.provider import EvictCallback
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class NoAuthProvider:
    """Open-source bypass provider — no authentication required.

    Auth provider only handles identification, not config loading.
    Config is loaded on-demand by get_datus_service if needed.
    """

    def __init__(self, namespace: str = "default"):
        self._evict_callbacks: list[EvictCallback] = []
        self._namespace = namespace

    async def authenticate(self, request: Request) -> AppContext:
        """Bypass auth, return AppContext without config.

        Config is loaded on-demand by get_datus_service if needed.
        """
        return AppContext(
            user_id="anonymous",
            project_id=self._namespace,
            config=None,
        )

    def on_evict(self, callback: EvictCallback) -> None:
        """Register eviction callback (no-op)."""
        # We don't have a config change listener.
        # Callbacks are registered for compatibility with other providers.
        self._evict_callbacks.append(callback)
