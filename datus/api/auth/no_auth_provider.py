"""Open-source default auth provider — header-based identification, no secret."""

import json
from typing import Any

from fastapi import Request

from datus.api.auth.context import AppContext
from datus.api.auth.provider import EvictCallback
from datus.api.constants import HEADER_PRINCIPAL, HEADER_USER_ID, USER_ID_PATTERN
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class NoAuthProvider:
    """Open-source default provider — reads optional request context headers.

    ``X-Datus-User-Id`` is optional caller identity for per-user session
    isolation. ``X-Datus-Principal`` is an optional JSON object whose fields are
    exposed to data-access policies as ``AppContext.principal``.

    Auth provider only handles identification, not config loading.
    Config is loaded on-demand by ``get_datus_service``.
    """

    def __init__(self) -> None:
        self._evict_callbacks: list[EvictCallback] = []

    async def authenticate(self, request: Request) -> AppContext:
        user_id = self._read_user_id(request)
        principal = self._read_principal(request)
        return AppContext(user_id=user_id, project_id=None, config=None, principal=principal)

    def on_evict(self, callback: EvictCallback) -> None:
        """Register eviction callback (no-op for no-auth provider)."""
        self._evict_callbacks.append(callback)

    @staticmethod
    def _read_user_id(request: Request) -> str | None:
        raw = request.headers.get(HEADER_USER_ID)
        if raw is None:
            return None
        candidate = raw.strip()
        if not candidate:
            return None
        if not USER_ID_PATTERN.match(candidate):
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=(
                    f"Invalid {HEADER_USER_ID} header value: {candidate!r}. "
                    "Only letters, digits, underscore and hyphen are allowed."
                ),
            )
        return candidate

    @staticmethod
    def _read_principal(request: Request) -> dict[str, Any]:
        raw = request.headers.get(HEADER_PRINCIPAL)
        if raw is None or not raw.strip():
            return {}

        try:
            principal = json.loads(raw)
        except json.JSONDecodeError as e:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=(
                    f"Invalid {HEADER_PRINCIPAL} header value: expected a JSON object with "
                    f"data-access principal fields ({e.msg})."
                ),
            ) from e

        if not isinstance(principal, dict):
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=(
                    f"Invalid {HEADER_PRINCIPAL} header value: expected a JSON object with "
                    "data-access principal fields."
                ),
            )
        if "user_id" in principal:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=(
                    f"Invalid {HEADER_PRINCIPAL} header value: field 'user_id' is reserved for "
                    f"{HEADER_USER_ID}; use a business principal field for data-access policy."
                ),
            )
        return principal
