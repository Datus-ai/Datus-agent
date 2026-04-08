"""Open-source default auth provider — header-based identification, no secret."""

from fastapi import Request

from datus.api.auth.context import AppContext
from datus.api.auth.provider import EvictCallback
from datus.api.constants import HEADER_USER_ID, USER_ID_PATTERN
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class NoAuthProvider:
    """Open-source default provider — identifies the caller via HTTP header.

    Reads ``X-Datus-User-Id`` from the request. When present and valid it is
    used as ``AppContext.user_id`` (and downstream as ``SessionManager.scope``
    for per-user session isolation). When absent, ``user_id`` is ``None`` and
    sessions fall back to the default (unscoped) directory.

    Auth provider only handles identification, not config loading.
    Config is loaded on-demand by ``get_datus_service``.
    """

    def __init__(self) -> None:
        self._evict_callbacks: list[EvictCallback] = []

    async def authenticate(self, request: Request) -> AppContext:
        raw = request.headers.get(HEADER_USER_ID)
        user_id: str | None = None
        if raw is not None:
            candidate = raw.strip()
            if candidate:
                if not USER_ID_PATTERN.match(candidate):
                    raise DatusException(
                        ErrorCode.COMMON_VALIDATION_FAILED,
                        message=(
                            f"Invalid {HEADER_USER_ID} header value: {candidate!r}. "
                            "Only letters, digits, underscore and hyphen are allowed."
                        ),
                    )
                user_id = candidate
        return AppContext(user_id=user_id, project_id=None, config=None)

    def on_evict(self, callback: EvictCallback) -> None:
        """Register eviction callback (no-op for no-auth provider)."""
        self._evict_callbacks.append(callback)
